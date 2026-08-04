"""Deterministic end-to-end execution for the frozen TEAR-CPP v0.1 study."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any, Mapping, Sequence
import weakref

import torch

from .calibration import (
    FrozenScopeEstimator,
    ScopeEstimator,
    canonical_state_digest,
    fit_temperature,
    train_scope_estimator,
)
from .config import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CONFIRMATORY_SEEDS,
    DIFFICULTIES,
    ESTIMATOR_EPOCHS,
    HIDDEN_WIDTH,
    INTEGRITY_SEED,
    LEARNING_RATE,
    POLICY_EPOCHS,
    REGISTERED_BASE_WORLD_COUNTS,
    TEMPERATURE_FIT_MAX_ITERATIONS,
    WATCHDOG_ABORT_SECONDS,
    safe_task_value,
)
from .gate import execute_expected_cost_gate, threshold_allows_commit
from .freeze import (
    FreezeVerificationError,
    sha256_bytes,
    sha256_file,
    verify_freeze_chain,
)
from .metrics import (
    CalibrationEvaluationRow,
    DecisionEvaluationRow,
    PosteriorEvaluationRow,
    evaluate_registered_metrics,
)
from .policy import (
    CoupledPolicy,
    SeparateActor,
    build_matched_policies,
    clone_policy,
    train_policy_cell,
)
from .schema import (
    Arm,
    BaseWorld,
    DeclarationCondition,
    Difficulty,
    Split,
    TerminalAction,
    WorldInstance,
)
from .splits import RegisteredSplits, generate_registered_splits
from .world import execute_terminal


ESTIMATOR_MODEL_SEED_XOR = 0x43505021
ACTOR_MODEL_SEED_XOR = 0x43505022
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ACTION_BY_INDEX = tuple(TerminalAction)
_RESERVED_SEEDS = frozenset((INTEGRITY_SEED, *CONFIRMATORY_SEEDS))


def configure_deterministic_execution() -> None:
    """Bind the registered deterministic single-thread Torch profile."""

    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError as exc:
        if torch.get_num_interop_threads() != 1:
            raise RuntimeError(
                "TEAR-CPP requires one Torch inter-op thread"
            ) from exc
    if torch.get_num_threads() != 1:
        raise RuntimeError("TEAR-CPP requires one Torch intra-op thread")
    if torch.get_num_interop_threads() != 1:
        raise RuntimeError("TEAR-CPP requires one Torch inter-op thread")


# Bind inter-op threads during module import, before any experiment work can
# initialize Torch's inter-op pool in a wider host-default configuration.
configure_deterministic_execution()

@dataclass(slots=True)
class Watchdog:
    started: float
    limit_seconds: float = float(WATCHDOG_ABORT_SECONDS)

    @classmethod
    def start(cls) -> "Watchdog":
        return cls(time.monotonic())

    def check(self, stage: str) -> None:
        elapsed = time.monotonic() - self.started
        if elapsed >= self.limit_seconds:
            raise TimeoutError(
                f"TEAR-CPP watchdog reached before {stage}: "
                f"{elapsed:.3f}s >= {self.limit_seconds:.3f}s"
            )


@dataclass(frozen=True, slots=True)
class SeedExecution:
    seed: int
    posterior_rows: tuple[PosteriorEvaluationRow, ...]
    calibration_rows: tuple[CalibrationEvaluationRow, ...]
    decision_rows: tuple[DecisionEvaluationRow, ...]
    training_audit: Mapping[str, Any]
    b_frozen_diagnostic: Mapping[str, Any]
    integrity: Mapping[str, Any]

class StudyAuthorizationError(RuntimeError):
    """Raised before any registered seed when execution is not authorized."""


@dataclass(frozen=True, slots=True)
class _VerifiedStudyState:
    root: Path
    preflight_path: Path
    reservation_path: Path
    preregistration_sha256: str
    implementation_manifest_sha256: str
    preflight_sha256: str
    reservation_sha256: str


@dataclass(frozen=True, slots=True)
class _IssuedStudyState:
    verified: _VerifiedStudyState
    consumption_sha256: str


class _RegisteredStudyAuthorization:
    """Opaque process-local handle backed by durable one-use consumption."""

    __slots__ = ("__weakref__",)

    def __new__(cls) -> "_RegisteredStudyAuthorization":
        raise TypeError(
            "registered-study authorization is issued only after verification"
        )


_AUTHORIZATION_STATES: weakref.WeakKeyDictionary[
    _RegisteredStudyAuthorization, _IssuedStudyState
] = weakref.WeakKeyDictionary()
_ISSUED_RESERVATIONS: set[tuple[Path, str]] = set()
_AUTHORIZATION_LOCK = threading.Lock()


def _feature_tensor(worlds: Sequence[BaseWorld]) -> torch.Tensor:
    return torch.tensor(
        [world.estimator_input.as_tuple() for world in worlds],
        dtype=torch.float32,
        device="cpu",
    )


def _target_tensor(worlds: Sequence[BaseWorld]) -> torch.Tensor:
    return torch.tensor(
        [world.scope_truth for world in worlds],
        dtype=torch.float32,
        device="cpu",
    )


def _host_probabilities(
    estimator: ScopeEstimator,
    features: torch.Tensor,
) -> tuple[float, ...]:
    estimator.eval()
    with torch.inference_mode():
        values = estimator(features).detach().cpu().reshape(-1)
    return tuple(float(value) for value in values)


def _coupled_probabilities(
    policy: CoupledPolicy,
    features: torch.Tensor,
) -> tuple[float, ...]:
    policy.eval()
    with torch.inference_mode():
        values = policy.scope_probabilities(features).detach().cpu().reshape(-1)
    return tuple(float(value) for value in values)


def _proposed_actions(
    policy: CoupledPolicy | SeparateActor,
    features: torch.Tensor,
) -> tuple[TerminalAction, ...]:
    policy.eval()
    with torch.inference_mode():
        indices = (
            policy.proposal_probabilities(features)
            .argmax(dim=1)
            .detach()
            .cpu()
            .tolist()
        )
    return tuple(_ACTION_BY_INDEX[int(index)] for index in indices)


def _execute_registered_action(
    *,
    arm: Arm,
    proposed: TerminalAction,
    q_post: float,
    difficulty: Difficulty,
) -> TerminalAction:
    # REPORT_EXIT has matched actor width in every arm, but it is recognized
    # only as D's registered fallback after a blocked irreversible proposal.
    if proposed is TerminalAction.REPORT_EXIT:
        return TerminalAction.SAFE_COMPLETE
    if proposed is TerminalAction.SAFE_COMPLETE:
        return proposed
    if arm is Arm.A:
        return TerminalAction.IRREVERSIBLE_COMMIT
    if arm is Arm.B:
        return (
            TerminalAction.IRREVERSIBLE_COMMIT
            if threshold_allows_commit(q_post)
            else TerminalAction.SAFE_COMPLETE
        )
    return execute_expected_cost_gate(arm, q_post, difficulty).executed_action


def _train_estimator(
    root_seed: int,
    splits: RegisteredSplits,
    watchdog: Watchdog,
) -> tuple[ScopeEstimator, dict[str, Any]]:
    training_features = _feature_tensor(splits.estimator_train)
    training_targets = _target_tensor(splits.estimator_train)
    estimator, training_report = train_scope_estimator(
        training_features,
        training_targets,
        seed=root_seed ^ ESTIMATOR_MODEL_SEED_XOR,
        checkpoint=watchdog.check,
    )
    calibration_features = _feature_tensor(splits.temperature_calibration)
    calibration_targets = _target_tensor(splits.temperature_calibration)
    temperature_report = fit_temperature(
        estimator,
        calibration_features,
        calibration_targets,
    )
    estimator.eval()
    return estimator, {
        "proper_score": "binary_log_loss",
        "training": asdict(training_report),
        "temperature_fit": asdict(temperature_report),
        "state_digest": canonical_state_digest(estimator),
    }


def _train_policy_cells(
    *,
    estimator: ScopeEstimator,
    root_seed: int,
    policy_features: torch.Tensor,
    watchdog: Watchdog,
) -> tuple[
    dict[Arm, dict[Difficulty, CoupledPolicy | SeparateActor]],
    FrozenScopeEstimator,
    FrozenScopeEstimator,
    dict[str, Any],
]:
    matched = build_matched_policies(
        estimator,
        actor_seed=root_seed ^ ACTOR_MODEL_SEED_XOR,
    )
    matched.assert_matched_actor_initialization()
    frozen_fixture = policy_features[: min(32, len(policy_features))]
    c_before = matched.c_estimator.probabilities(frozen_fixture)
    d_before = matched.d_estimator.probabilities(frozen_fixture)
    if c_before != d_before:
        raise AssertionError("C/D frozen fixture probabilities differ before training")

    base_policies: dict[Arm, CoupledPolicy | SeparateActor] = {
        Arm.A: matched.a,
        Arm.B: matched.b,
        Arm.C: matched.c,
        Arm.D: matched.d,
    }
    trained: dict[
        Arm, dict[Difficulty, CoupledPolicy | SeparateActor]
    ] = {arm: {} for arm in Arm}
    reports: dict[str, dict[str, Any]] = {arm.value: {} for arm in Arm}
    for difficulty in DIFFICULTIES:
        for arm in Arm:
            watchdog.check(f"policy cell {root_seed}/{arm.value}/{int(difficulty)}")
            policy = clone_policy(base_policies[arm])
            frozen = (
                matched.c_estimator
                if arm is Arm.C
                else matched.d_estimator
                if arm is Arm.D
                else None
            )
            report = train_policy_cell(
                arm.value,
                policy,
                policy_features,
                q_safe=safe_task_value(difficulty),
                frozen_estimator=frozen,
                checkpoint=watchdog.check,
            )
            trained[arm][difficulty] = policy
            reports[arm.value][str(int(difficulty))] = asdict(report)

    matched.c_estimator.assert_integrity()
    matched.d_estimator.assert_integrity()
    c_after = matched.c_estimator.probabilities(frozen_fixture)
    d_after = matched.d_estimator.probabilities(frozen_fixture)
    if c_after != c_before or d_after != d_before or c_after != d_after:
        raise AssertionError("C/D frozen fixture probabilities changed")
    return (
        trained,
        matched.c_estimator,
        matched.d_estimator,
        {
            "actor_seed": root_seed ^ ACTOR_MODEL_SEED_XOR,
            "cells": reports,
            "c_frozen_estimator_digest": matched.c_estimator.digest,
            "d_frozen_estimator_digest": matched.d_estimator.digest,
        },
    )


def _condition_name(worlds: Sequence[BaseWorld]) -> str:
    declarations = {world.declaration for world in worlds}
    if declarations == {DeclarationCondition.PRESENT}:
        return "declared"
    if declarations == {DeclarationCondition.ABSENT}:
        return "no_declaration"
    raise AssertionError("evaluation condition mixes declaration states")


def _empty_b_frozen_cell() -> dict[str, int]:
    return {
        "rows": 0,
        "commit_proposals": 0,
        "coupled_executed_commits": 0,
        "frozen_executed_commits": 0,
        "execution_disagreements": 0,
    }


def _evaluate_condition(
    *,
    root_seed: int,
    worlds: Sequence[BaseWorld],
    estimator: ScopeEstimator,
    trained: Mapping[Arm, Mapping[Difficulty, CoupledPolicy | SeparateActor]],
    c_estimator: FrozenScopeEstimator,
    d_estimator: FrozenScopeEstimator,
) -> tuple[
    tuple[PosteriorEvaluationRow, ...],
    tuple[CalibrationEvaluationRow, ...],
    tuple[DecisionEvaluationRow, ...],
    dict[str, Any],
]:
    condition = _condition_name(worlds)
    features = _feature_tensor(worlds)
    q_pre = _host_probabilities(estimator, features)
    c_q = c_estimator.probabilities(features)
    d_q = d_estimator.probabilities(features)
    if c_q != d_q or c_q != q_pre:
        raise AssertionError("common pre-policy and C/D frozen probabilities differ")

    calibration_rows = tuple(
        CalibrationEvaluationRow(
            seed=root_seed,
            base_world_id=world.world_id,
            condition=condition,
            y=world.scope_truth,
            q=q,
        )
        for world, q in zip(worlds, q_pre, strict=True)
    )
    posterior_rows: list[PosteriorEvaluationRow] = []
    decision_rows: list[DecisionEvaluationRow] = []
    b_frozen: dict[str, Any] = {
        str(int(difficulty)): _empty_b_frozen_cell()
        for difficulty in DIFFICULTIES
    }

    for difficulty in DIFFICULTIES:
        for arm in Arm:
            policy = trained[arm][difficulty]
            proposals = _proposed_actions(policy, features)
            if arm in {Arm.A, Arm.B}:
                if not isinstance(policy, CoupledPolicy):
                    raise AssertionError(f"Arm {arm.value} lost coupled policy")
                q_post = _coupled_probabilities(policy, features)
            elif arm is Arm.C:
                q_post = c_q
            else:
                q_post = d_q

            for world, before, after, proposed in zip(
                worlds, q_pre, q_post, proposals, strict=True
            ):
                executed = _execute_registered_action(
                    arm=arm,
                    proposed=proposed,
                    q_post=after,
                    difficulty=difficulty,
                )
                instance = WorldInstance(world, difficulty)
                terminal = execute_terminal(instance, executed)
                posterior_rows.append(
                    PosteriorEvaluationRow(
                        seed=root_seed,
                        base_world_id=world.world_id,
                        condition=condition,
                        arm=arm.value,
                        difficulty=int(difficulty),
                        y=world.scope_truth,
                        q_post=after,
                        q_pre=before,
                    )
                )
                decision_rows.append(
                    DecisionEvaluationRow(
                        seed=root_seed,
                        base_world_id=world.world_id,
                        condition=condition,
                        arm=arm.value,
                        difficulty=int(difficulty),
                        y=world.scope_truth,
                        q_post=after,
                        proposed_action=proposed.value,
                        executed_action=executed.value,
                        task_return=terminal.task_value,
                    )
                )
                if arm is Arm.B:
                    frozen_executed = _execute_registered_action(
                        arm=Arm.B,
                        proposed=proposed,
                        q_post=before,
                        difficulty=difficulty,
                    )
                    cell = b_frozen[str(int(difficulty))]
                    cell["rows"] += 1
                    cell["commit_proposals"] += int(
                        proposed is TerminalAction.IRREVERSIBLE_COMMIT
                    )
                    cell["coupled_executed_commits"] += int(
                        executed is TerminalAction.IRREVERSIBLE_COMMIT
                    )
                    cell["frozen_executed_commits"] += int(
                        frozen_executed is TerminalAction.IRREVERSIBLE_COMMIT
                    )
                    cell["execution_disagreements"] += int(
                        frozen_executed is not executed
                    )
    return (
        tuple(posterior_rows),
        calibration_rows,
        tuple(decision_rows),
        b_frozen,
    )


def run_registered_seed(
    root_seed: int,
    *,
    watchdog: Watchdog | None = None,
) -> SeedExecution:
    """Run one nonreserved synthetic seed at the exact registered budget."""

    if root_seed in _RESERVED_SEEDS:
        raise PermissionError(
            "timing and confirmatory seeds require their private launch path"
        )
    return _run_registered_seed_unchecked(root_seed, watchdog=watchdog)


def _run_registered_seed_unchecked(
    root_seed: int,
    *,
    watchdog: Watchdog | None = None,
) -> SeedExecution:
    """Run one exact-budget seed after the caller authorizes its purpose."""
    configure_deterministic_execution()
    active_watchdog = watchdog or Watchdog.start()
    splits = generate_registered_splits(root_seed)
    active_watchdog.check(f"split generation for seed {root_seed}")
    estimator, estimator_audit = _train_estimator(
        root_seed,
        splits,
        active_watchdog,
    )
    active_watchdog.check(f"scope estimator for seed {root_seed}")
    policy_features = _feature_tensor(splits.policy_train)
    (
        trained,
        c_estimator,
        d_estimator,
        policy_audit,
    ) = _train_policy_cells(
        estimator=estimator,
        root_seed=root_seed,
        policy_features=policy_features,
        watchdog=active_watchdog,
    )
    active_watchdog.check(f"policy training for seed {root_seed}")

    posterior_rows: list[PosteriorEvaluationRow] = []
    calibration_rows: list[CalibrationEvaluationRow] = []
    decision_rows: list[DecisionEvaluationRow] = []
    b_frozen: dict[str, Any] = {}
    for worlds in (
        splits.declared_evaluation,
        splits.no_declaration_evaluation,
    ):
        posterior, calibration, decisions, diagnostic = _evaluate_condition(
            root_seed=root_seed,
            worlds=worlds,
            estimator=estimator,
            trained=trained,
            c_estimator=c_estimator,
            d_estimator=d_estimator,
        )
        condition = _condition_name(worlds)
        posterior_rows.extend(posterior)
        calibration_rows.extend(calibration)
        decision_rows.extend(decisions)
        b_frozen[condition] = diagnostic
        active_watchdog.check(f"{condition} evaluation for seed {root_seed}")

    c_estimator.assert_integrity()
    d_estimator.assert_integrity()
    estimator_digest_stable = (
        canonical_state_digest(estimator) == estimator_audit["state_digest"]
    )
    if not estimator_digest_stable:
        raise AssertionError("common estimator changed after policy training")
    expected_eval_worlds = (
        REGISTERED_BASE_WORLD_COUNTS[Split.DECLARED_EVALUATION]
        + REGISTERED_BASE_WORLD_COUNTS[Split.NO_DECLARATION_EVALUATION]
    )
    expected_arm_cells = expected_eval_worlds * len(tuple(Arm)) * len(DIFFICULTIES)
    if len(posterior_rows) != expected_arm_cells:
        raise AssertionError("posterior row count does not match registered budget")
    if len(decision_rows) != expected_arm_cells:
        raise AssertionError("decision row count does not match registered budget")
    if len(calibration_rows) != expected_eval_worlds:
        raise AssertionError("calibration row count does not match registered budget")

    return SeedExecution(
        seed=root_seed,
        posterior_rows=tuple(posterior_rows),
        calibration_rows=tuple(calibration_rows),
        decision_rows=tuple(decision_rows),
        training_audit={
            "estimator": estimator_audit,
            "policy": policy_audit,
        },
        b_frozen_diagnostic=b_frozen,
        integrity={
            "c_d_frozen_digests_equal": (
                c_estimator.digest == d_estimator.digest
            ),
            "frozen_digest_stable_after_all_policy_cells": True,
            "common_estimator_digest_stable": estimator_digest_stable,
            "posterior_rows": len(posterior_rows),
            "calibration_rows": len(calibration_rows),
            "decision_rows": len(decision_rows),
        },
    )


def _aggregate_b_frozen(
    executions: Sequence[SeedExecution],
) -> dict[str, Any]:
    pooled: dict[str, dict[str, dict[str, Any]]] = {
        condition: {
            str(int(difficulty)): _empty_b_frozen_cell()
            for difficulty in DIFFICULTIES
        }
        for condition in ("declared", "no_declaration")
    }
    per_seed: dict[str, Any] = {}
    for execution in executions:
        per_seed[str(execution.seed)] = execution.b_frozen_diagnostic
        for condition, cells in execution.b_frozen_diagnostic.items():
            for difficulty, cell in cells.items():
                target = pooled[condition][difficulty]
                for key in target:
                    target[key] += int(cell[key])
    for cells in pooled.values():
        for cell in cells.values():
            rows = cell["rows"]
            cell["coupled_commit_rate"] = (
                cell["coupled_executed_commits"] / rows if rows else None
            )
            cell["frozen_commit_rate"] = (
                cell["frozen_executed_commits"] / rows if rows else None
            )
            cell["disagreement_rate"] = (
                cell["execution_disagreements"] / rows if rows else None
            )
    return {
        "status": "diagnostic_not_fifth_arm",
        "inference_limit": (
            "May localize an inference-time frozen-estimator contribution; "
            "cannot identify expected cost as superior to thresholding."
        ),
        "fixed_context_replay": pooled,
        "matched_full_episode": pooled,
        "full_episode_equals_replay_reason": (
            "The registered traversal is forced and has one terminal decision."
        ),
        "per_seed": per_seed,
    }


def _require_sha256(name: str, value: str) -> str:
    if not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _reject_duplicate_authorization_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StudyAuthorizationError(
                f"duplicate authorization JSON key: {key}"
            )
        result[key] = value
    return result


def _load_authorization_json(
    path: Path,
    *,
    label: str,
) -> tuple[dict[str, Any], bytes]:
    try:
        payload = path.read_bytes()
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_authorization_keys,
        )
    except StudyAuthorizationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StudyAuthorizationError(f"cannot read {label}") from exc
    if type(value) is not dict:
        raise StudyAuthorizationError(f"{label} must be a JSON object")
    return value, payload


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _verify_preflight_authorization(
    record: Mapping[str, Any],
    payload: bytes,
    *,
    preregistration_sha256: str,
    implementation_manifest_sha256: str,
) -> str:
    required = {
        "schema_version": "tear-cpp-preflight-v0.1",
        "seed": INTEGRITY_SEED,
        "scientific_metrics_suppressed": True,
        "prediction_rule": "single_seed_seconds_times_5_plus_120",
        "watchdog_seconds": WATCHDOG_ABORT_SECONDS,
        "pass": True,
        "preregistration_sha256": preregistration_sha256,
        "implementation_manifest_sha256": implementation_manifest_sha256,
    }
    allowed_keys = set(required) | {
        "elapsed_seconds",
        "predicted_confirmatory_seconds",
        "integrity",
    }
    if set(record) != allowed_keys:
        raise StudyAuthorizationError(
            "preflight contains unregistered fields or omits required fields"
        )
    for key, expected in required.items():
        actual = record[key]
        if type(actual) is not type(expected) or actual != expected:
            raise StudyAuthorizationError(f"preflight binding failed: {key}")

    elapsed = record["elapsed_seconds"]
    predicted = record["predicted_confirmatory_seconds"]
    if (
        type(elapsed) not in (int, float)
        or not math.isfinite(float(elapsed))
        or float(elapsed) < 0.0
        or type(predicted) not in (int, float)
        or not math.isfinite(float(predicted))
        or not 0.0 <= float(predicted) < WATCHDOG_ABORT_SECONDS
    ):
        raise StudyAuthorizationError(
            "preflight prediction does not clear the watchdog"
        )
    expected_prediction = (
        float(elapsed) * len(CONFIRMATORY_SEEDS) + 120.0
    )
    if float(predicted) != expected_prediction:
        raise StudyAuthorizationError(
            "preflight prediction does not match the frozen formula"
        )

    integrity = record["integrity"]
    expected_integrity_keys = {
        "c_d_frozen_digests_equal",
        "frozen_digest_stable_after_all_policy_cells",
        "common_estimator_digest_stable",
        "posterior_row_count_matches_registered_budget",
    }
    if (
        type(integrity) is not dict
        or set(integrity) != expected_integrity_keys
        or not all(value is True for value in integrity.values())
    ):
        raise StudyAuthorizationError(
            "preflight integrity assertions did not all pass"
        )
    if payload != _canonical_json_bytes(record):
        raise StudyAuthorizationError("timing preflight bytes are not canonical")
    return sha256_bytes(payload)


def _verify_registered_study_state(
    root: Path,
    preflight_path: Path,
) -> _VerifiedStudyState:
    try:
        resolved_root = Path(root).resolve(strict=True)
    except OSError as exc:
        raise StudyAuthorizationError("study root does not exist") from exc
    if not resolved_root.is_dir():
        raise StudyAuthorizationError("study root is not a directory")

    canonical_preflight = (
        resolved_root / "artifacts" / "cpp-preflight" / "timing.json"
    )
    try:
        resolved_preflight = Path(preflight_path).resolve(strict=True)
        expected_preflight = canonical_preflight.resolve(strict=True)
    except OSError as exc:
        raise StudyAuthorizationError(
            "canonical timing preflight does not exist"
        ) from exc
    if resolved_preflight != expected_preflight:
        raise StudyAuthorizationError(
            "preflight path is not the canonical timing record"
        )

    reservation_path = (
        resolved_root / "artifacts" / "cpp-confirmatory.lock"
    )
    try:
        freeze_receipt = verify_freeze_chain(resolved_root)
        implementation_digest = sha256_file(
            resolved_root / "tearrl" / "cpp" / "frozen_implementation.json"
        )
    except (FreezeVerificationError, OSError, UnicodeError) as exc:
        raise StudyAuthorizationError(
            "frozen implementation state did not verify"
        ) from exc
    if not isinstance(freeze_receipt, Mapping):
        raise StudyAuthorizationError("freeze receipt is invalid")
    preregistration_value = freeze_receipt.get("preregistration_sha256")
    if type(preregistration_value) is not str:
        raise StudyAuthorizationError("freeze receipt is invalid")
    try:
        preregistration_digest = _require_sha256(
            "preregistration_sha256", preregistration_value
        )
        implementation_digest = _require_sha256(
            "implementation_manifest_sha256", implementation_digest
        )
    except ValueError as exc:
        raise StudyAuthorizationError("freeze receipt is invalid") from exc

    preflight_record, preflight_payload = _load_authorization_json(
        resolved_preflight,
        label="timing preflight",
    )
    preflight_digest = _verify_preflight_authorization(
        preflight_record,
        preflight_payload,
        preregistration_sha256=preregistration_digest,
        implementation_manifest_sha256=implementation_digest,
    )

    reservation_record, reservation_payload = _load_authorization_json(
        reservation_path,
        label="confirmatory execution reservation",
    )
    expected_reservation = {
        "schema_version": "tear-cpp-execution-reservation-v0.1",
        "status": "reserved_before_any_confirmatory_seed",
        "confirmatory_seeds": list(CONFIRMATORY_SEEDS),
        "preregistration_sha256": preregistration_digest,
        "implementation_manifest_sha256": implementation_digest,
        "preflight_sha256": preflight_digest,
    }
    if reservation_record != expected_reservation:
        raise StudyAuthorizationError(
            "confirmatory execution reservation does not match frozen receipts"
        )
    if reservation_payload != _canonical_json_bytes(expected_reservation):
        raise StudyAuthorizationError(
            "confirmatory execution reservation bytes are not canonical"
        )

    return _VerifiedStudyState(
        root=resolved_root,
        preflight_path=resolved_preflight,
        reservation_path=reservation_path,
        preregistration_sha256=preregistration_digest,
        implementation_manifest_sha256=implementation_digest,
        preflight_sha256=preflight_digest,
        reservation_sha256=sha256_bytes(reservation_payload),
    )


def _authorization_consumption_path(state: _VerifiedStudyState) -> Path:
    return state.reservation_path.with_name(
        f"{state.reservation_path.stem}.authorization-consumed.json"
    )


def _authorization_consumption_record(
    state: _VerifiedStudyState,
) -> dict[str, Any]:
    return {
        "schema_version": "tear-cpp-confirmatory-authorization-v0.1",
        "status": "authorization_consumed_before_token_issue",
        "confirmatory_seeds": list(CONFIRMATORY_SEEDS),
        "preregistration_sha256": state.preregistration_sha256,
        "implementation_manifest_sha256": (
            state.implementation_manifest_sha256
        ),
        "preflight_sha256": state.preflight_sha256,
        "reservation_sha256": state.reservation_sha256,
    }


def _durably_consume_execution_reservation(
    state: _VerifiedStudyState,
) -> str:
    path = _authorization_consumption_path(state)
    payload = _canonical_json_bytes(_authorization_consumption_record(state))
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise StudyAuthorizationError(
            "authorization was already issued for this execution reservation; "
            "durable consumption receipt exists"
        ) from exc
    except OSError as exc:
        raise StudyAuthorizationError(
            "cannot durably consume the execution reservation"
        ) from exc

    try:
        persisted = path.read_bytes()
    except OSError as exc:
        raise StudyAuthorizationError(
            "cannot verify the durable authorization receipt"
        ) from exc
    if persisted != payload:
        raise StudyAuthorizationError(
            "durable authorization receipt failed byte verification"
        )
    return sha256_bytes(payload)


def _verify_authorization_consumption(
    state: _VerifiedStudyState,
    expected_sha256: str,
) -> None:
    expected = _authorization_consumption_record(state)
    record, payload = _load_authorization_json(
        _authorization_consumption_path(state),
        label="durable authorization receipt",
    )
    if record != expected or payload != _canonical_json_bytes(expected):
        raise StudyAuthorizationError(
            "durable authorization receipt does not match frozen launch state"
        )
    if sha256_bytes(payload) != expected_sha256:
        raise StudyAuthorizationError(
            "durable authorization receipt digest changed"
        )


def _require_trusted_runtime(root: Path) -> None:
    required_flags = {
        "isolated": sys.flags.isolated,
        "ignore_environment": sys.flags.ignore_environment,
        "no_site": sys.flags.no_site,
        "no_user_site": sys.flags.no_user_site,
        "safe_path": sys.flags.safe_path,
        "dont_write_bytecode": sys.flags.dont_write_bytecode,
    }
    missing_flags = sorted(
        name for name, enabled in required_flags.items() if not enabled
    )
    if missing_flags:
        raise StudyAuthorizationError(
            "trusted runtime attestation failed: missing interpreter flags: "
            + ", ".join(missing_flags)
        )
    if sys.flags.optimize != 0 or sys.warnoptions or sys._xoptions:
        raise StudyAuthorizationError(
            "trusted runtime attestation failed: altered interpreter options"
        )
    loaded_startup_hooks = sorted(
        name
        for name in ("site", "sitecustomize", "usercustomize")
        if name in sys.modules
    )
    if loaded_startup_hooks:
        raise StudyAuthorizationError(
            "trusted runtime attestation failed: startup hooks are loaded: "
            + ", ".join(loaded_startup_hooks)
        )

    expected_environment = {
        "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    invalid_environment = sorted(
        name
        for name, expected in expected_environment.items()
        if os.environ.get(name) != expected
    )
    if invalid_environment:
        raise StudyAuthorizationError(
            "trusted runtime attestation failed: invalid environment: "
            + ", ".join(invalid_environment)
        )

    try:
        resolved_root = Path(root).resolve(strict=True)
        base_prefix = Path(sys.base_prefix).resolve(strict=True)
        base_paths = (
            (base_prefix / "DLLs").resolve(strict=True),
            (base_prefix / "Lib").resolve(strict=True),
        )
        expected_interpreter = (
            resolved_root / ".venv" / "Scripts" / "python.exe"
        ).resolve(strict=True)
        site_packages = (
            resolved_root / ".venv" / "Lib" / "site-packages"
        ).resolve(strict=True)
        actual_interpreter = Path(sys.executable).resolve(strict=True)
        if type(sys.pycache_prefix) is not str or not sys.pycache_prefix:
            raise OSError("bytecode cache prefix is absent")
        pycache_prefix = Path(sys.pycache_prefix).resolve(strict=True)
        if not all(path.is_dir() for path in base_paths):
            raise OSError("base runtime path is not a directory")
        if not resolved_root.is_dir() or not site_packages.is_dir():
            raise OSError("trusted project runtime path is not a directory")
        if not expected_interpreter.is_file():
            raise OSError("trusted interpreter is not a file")
        if not pycache_prefix.is_dir() or any(pycache_prefix.iterdir()):
            raise OSError("bytecode cache prefix is not an empty directory")
        if any(type(entry) is not str for entry in sys.path):
            raise OSError("import path contains a non-string entry")
        actual_paths = tuple(
            Path(entry).resolve(strict=True) for entry in sys.path
        )
    except (OSError, RuntimeError) as exc:
        raise StudyAuthorizationError(
            "trusted runtime attestation failed: runtime path state is invalid"
        ) from exc

    if actual_interpreter != expected_interpreter:
        raise StudyAuthorizationError(
            "trusted runtime attestation failed: interpreter path mismatch"
        )
    expected_paths = (*base_paths, site_packages, resolved_root)
    if actual_paths != expected_paths:
        raise StudyAuthorizationError(
            "trusted runtime attestation failed: import path order mismatch"
        )


def authorize_registered_study(
    root: Path,
    preflight_path: Path,
) -> _RegisteredStudyAuthorization:
    """Mint one opaque capability after all frozen launch state verifies."""

    _require_trusted_runtime(root)
    state = _verify_registered_study_state(root, preflight_path)
    reservation_key = (state.root, state.reservation_sha256)
    with _AUTHORIZATION_LOCK:
        if reservation_key in _ISSUED_RESERVATIONS:
            raise StudyAuthorizationError(
                "authorization was already issued for this execution reservation"
            )
        consumption_sha256 = _durably_consume_execution_reservation(state)
        _ISSUED_RESERVATIONS.add(reservation_key)

    current_state = _verify_registered_study_state(root, preflight_path)
    if current_state != state:
        raise StudyAuthorizationError(
            "frozen launch state changed while authorization was consumed"
        )
    _verify_authorization_consumption(state, consumption_sha256)

    with _AUTHORIZATION_LOCK:
        token = object.__new__(_RegisteredStudyAuthorization)
        _AUTHORIZATION_STATES[token] = _IssuedStudyState(
            verified=state,
            consumption_sha256=consumption_sha256,
        )
        return token


def _consume_registered_study_authorization(
    authorization: _RegisteredStudyAuthorization,
) -> _VerifiedStudyState:
    if type(authorization) is not _RegisteredStudyAuthorization:
        raise StudyAuthorizationError(
            "registered study requires a verified private authorization"
        )
    with _AUTHORIZATION_LOCK:
        try:
            issued = _AUTHORIZATION_STATES[authorization]
        except KeyError as exc:
            raise StudyAuthorizationError(
                "registered study authorization is forged, spent, or unknown"
            ) from exc
    _require_trusted_runtime(issued.verified.root)
    with _AUTHORIZATION_LOCK:
        try:
            consumed = _AUTHORIZATION_STATES.pop(authorization)
        except KeyError as exc:
            raise StudyAuthorizationError(
                "registered study authorization is forged, spent, or unknown"
            ) from exc
    if consumed != issued:
        raise StudyAuthorizationError(
            "registered study authorization state changed before consumption"
        )
    current_state = _verify_registered_study_state(
        issued.verified.root,
        issued.verified.preflight_path,
    )
    if current_state != issued.verified:
        raise StudyAuthorizationError(
            "frozen launch state changed after authorization"
        )
    _verify_authorization_consumption(
        current_state,
        issued.consumption_sha256,
    )
    return current_state

def run_confirmatory_study(
    *,
    authorization: _RegisteredStudyAuthorization,
    watchdog: Watchdog | None = None,
) -> dict[str, Any]:
    """Consume a verified capability and execute the five frozen seeds once."""

    verified_state = _consume_registered_study_authorization(authorization)
    configure_deterministic_execution()
    active_watchdog = watchdog or Watchdog.start()
    prereg_digest = verified_state.preregistration_sha256
    implementation_digest = verified_state.implementation_manifest_sha256
    executions: list[SeedExecution] = []
    for root_seed in CONFIRMATORY_SEEDS:
        active_watchdog.check(f"confirmatory seed {root_seed}")
        executions.append(
            _run_registered_seed_unchecked(
                root_seed, watchdog=active_watchdog
            )
        )
    active_watchdog.check("registered row aggregation")

    posterior_rows = tuple(
        row for execution in executions for row in execution.posterior_rows
    )
    calibration_rows = tuple(
        row for execution in executions for row in execution.calibration_rows
    )
    decision_rows = tuple(
        row for execution in executions for row in execution.decision_rows
    )
    result = evaluate_registered_metrics(
        posterior_rows=posterior_rows,
        calibration_rows=calibration_rows,
        decision_rows=decision_rows,
        checkpoint=active_watchdog.check,
    )
    active_watchdog.check("registered metric aggregation complete")
    result.update(
        {
            "mode": "confirmatory",
            "seeds": list(CONFIRMATORY_SEEDS),
            "registered_outcome_order": [
                "calibration_optimism_drift_primary",
                "irreversible_out_of_scope_execution_secondary",
                "task_return_diagnostic_only",
            ],
            "provenance": {
                "preregistration_sha256": prereg_digest,
                "implementation_manifest_sha256": implementation_digest,
            },
            "configuration": {
                "base_world_counts": {
                    split.value: count
                    for split, count in REGISTERED_BASE_WORLD_COUNTS.items()
                },
                "estimator_epochs": ESTIMATOR_EPOCHS,
                "temperature_fit_max_iterations": (
                    TEMPERATURE_FIT_MAX_ITERATIONS
                ),
                "policy_epochs_per_arm_difficulty": POLICY_EPOCHS,
                "learning_rate": LEARNING_RATE,
                "hidden_width": HIDDEN_WIDTH,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "estimator_model_seed_xor": ESTIMATOR_MODEL_SEED_XOR,
                "actor_model_seed_xor": ACTOR_MODEL_SEED_XOR,
                "torch_intra_op_threads": 1,
                "torch_inter_op_threads": 1,
                "device": "cpu",
            },
            "training_audit": {
                str(execution.seed): execution.training_audit
                for execution in executions
            },
            "integrity": {
                "all_seed_checks_pass": all(
                    execution.integrity[
                        "c_d_frozen_digests_equal"
                    ]
                    and execution.integrity[
                        "frozen_digest_stable_after_all_policy_cells"
                    ]
                    and execution.integrity[
                        "common_estimator_digest_stable"
                    ]
                    for execution in executions
                ),
                "per_seed": {
                    str(execution.seed): execution.integrity
                    for execution in executions
                },
            },
        }
    )
    result["diagnostics"]["b_frozen_no_retraining"] = _aggregate_b_frozen(
        executions
    )
    active_watchdog.check("result assembly")
    return result

def run_timing_preflight() -> dict[str, Any]:
    """Run the exact one-seed workload and reveal no scientific metric."""

    configure_deterministic_execution()
    started = time.monotonic()
    execution = _run_registered_seed_unchecked(INTEGRITY_SEED)
    elapsed = time.monotonic() - started
    predicted = elapsed * len(CONFIRMATORY_SEEDS) + 120.0
    return {
        "schema_version": "tear-cpp-preflight-v0.1",
        "seed": INTEGRITY_SEED,
        "scientific_metrics_suppressed": True,
        "elapsed_seconds": elapsed,
        "prediction_rule": "single_seed_seconds_times_5_plus_120",
        "predicted_confirmatory_seconds": predicted,
        "watchdog_seconds": WATCHDOG_ABORT_SECONDS,
        "pass": predicted < WATCHDOG_ABORT_SECONDS,
        "integrity": {
            "c_d_frozen_digests_equal": execution.integrity[
                "c_d_frozen_digests_equal"
            ],
            "frozen_digest_stable_after_all_policy_cells": execution.integrity[
                "frozen_digest_stable_after_all_policy_cells"
            ],
            "common_estimator_digest_stable": execution.integrity[
                "common_estimator_digest_stable"
            ],
            "posterior_row_count_matches_registered_budget": (
                execution.integrity["posterior_rows"]
                == execution.integrity["decision_rows"]
            ),
        },
    }
