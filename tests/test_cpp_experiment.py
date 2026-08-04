from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading

import pytest

import tearrl.cpp.experiment as experiment_module
from tearrl.cpp.config import (
    CONFIRMATORY_SEEDS,
    INTEGRITY_SEED,
    WATCHDOG_ABORT_SECONDS,
)
from tearrl.cpp.experiment import (
    StudyAuthorizationError,
    Watchdog,
    _execute_registered_action,
    authorize_registered_study,
    configure_deterministic_execution,
    run_confirmatory_study,
    run_registered_seed,
)
from tearrl.cpp.schema import Arm, Difficulty, TerminalAction


def _stable_json_bytes(value: object) -> bytes:
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


def _write_authorization_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    Path,
    Path,
    Path,
    dict[str, object],
    dict[str, object],
]:
    root = tmp_path.resolve()
    implementation_path = (
        root / "tearrl" / "cpp" / "frozen_implementation.json"
    )
    implementation_path.parent.mkdir(parents=True)
    implementation_path.write_bytes(b"{}\n")
    implementation_sha256 = hashlib.sha256(
        implementation_path.read_bytes()
    ).hexdigest()
    preregistration_sha256 = "a" * 64
    monkeypatch.setattr(
        experiment_module,
        "verify_freeze_chain",
        lambda candidate: {
            "preregistration_sha256": preregistration_sha256,
        },
    )
    monkeypatch.setattr(
        experiment_module,
        "_require_trusted_runtime",
        lambda candidate: None,
    )

    preflight_record: dict[str, object] = {
        "schema_version": "tear-cpp-preflight-v0.1",
        "seed": INTEGRITY_SEED,
        "scientific_metrics_suppressed": True,
        "elapsed_seconds": 1.0,
        "prediction_rule": "single_seed_seconds_times_5_plus_120",
        "predicted_confirmatory_seconds": (
            1.0 * len(CONFIRMATORY_SEEDS) + 120.0
        ),
        "watchdog_seconds": WATCHDOG_ABORT_SECONDS,
        "pass": True,
        "preregistration_sha256": preregistration_sha256,
        "implementation_manifest_sha256": implementation_sha256,
        "integrity": {
            "c_d_frozen_digests_equal": True,
            "frozen_digest_stable_after_all_policy_cells": True,
            "common_estimator_digest_stable": True,
            "posterior_row_count_matches_registered_budget": True,
        },
    }
    preflight_path = (
        root / "artifacts" / "cpp-preflight" / "timing.json"
    )
    preflight_path.parent.mkdir(parents=True)
    preflight_payload = _stable_json_bytes(preflight_record)
    preflight_path.write_bytes(preflight_payload)

    reservation_record: dict[str, object] = {
        "schema_version": "tear-cpp-execution-reservation-v0.1",
        "status": "reserved_before_any_confirmatory_seed",
        "confirmatory_seeds": list(CONFIRMATORY_SEEDS),
        "preregistration_sha256": preregistration_sha256,
        "implementation_manifest_sha256": implementation_sha256,
        "preflight_sha256": hashlib.sha256(preflight_payload).hexdigest(),
    }
    reservation_path = root / "artifacts" / "cpp-confirmatory.lock"
    reservation_path.write_bytes(_stable_json_bytes(reservation_record))
    return (
        root,
        preflight_path,
        reservation_path,
        preflight_record,
        reservation_record,
    )

def test_registered_action_execution_and_exit_recognition_are_exact() -> None:
    assert _execute_registered_action(
        arm=Arm.A,
        proposed=TerminalAction.IRREVERSIBLE_COMMIT,
        q_post=0.0,
        difficulty=Difficulty.D3,
    ) is TerminalAction.IRREVERSIBLE_COMMIT
    assert _execute_registered_action(
        arm=Arm.B,
        proposed=TerminalAction.IRREVERSIBLE_COMMIT,
        q_post=0.80,
        difficulty=Difficulty.D3,
    ) is TerminalAction.IRREVERSIBLE_COMMIT
    assert _execute_registered_action(
        arm=Arm.B,
        proposed=TerminalAction.IRREVERSIBLE_COMMIT,
        q_post=0.80 - 1e-12,
        difficulty=Difficulty.D3,
    ) is TerminalAction.SAFE_COMPLETE
    assert _execute_registered_action(
        arm=Arm.C,
        proposed=TerminalAction.IRREVERSIBLE_COMMIT,
        q_post=0.40,
        difficulty=Difficulty.D3,
    ) is TerminalAction.SAFE_COMPLETE
    assert _execute_registered_action(
        arm=Arm.D,
        proposed=TerminalAction.IRREVERSIBLE_COMMIT,
        q_post=0.40,
        difficulty=Difficulty.D3,
    ) is TerminalAction.REPORT_EXIT

    for arm in Arm:
        assert _execute_registered_action(
            arm=arm,
            proposed=TerminalAction.REPORT_EXIT,
            q_post=0.0,
            difficulty=Difficulty.D3,
        ) is TerminalAction.SAFE_COMPLETE


def test_execution_profile_is_deterministic_and_single_threaded() -> None:
    import torch

    configure_deterministic_execution()
    configure_deterministic_execution()
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.get_num_threads() == 1
    assert torch.get_num_interop_threads() == 1


@pytest.mark.parametrize(
    "reserved_seed",
    (INTEGRITY_SEED, *CONFIRMATORY_SEEDS),
)
def test_public_synthetic_runner_rejects_every_reserved_seed_before_work(
    monkeypatch: pytest.MonkeyPatch,
    reserved_seed: int,
) -> None:
    called = False

    def fail_if_called(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("reserved seed reached the private runner")

    monkeypatch.setattr(
        experiment_module,
        "_run_registered_seed_unchecked",
        fail_if_called,
    )
    with pytest.raises(PermissionError, match="private launch path"):
        run_registered_seed(reserved_seed)
    assert called is False


def test_authorization_factory_accepts_exact_frozen_launch_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, preflight_path, reservation_path, _, _ = _write_authorization_state(
        tmp_path,
        monkeypatch,
    )
    reservation_bytes = reservation_path.read_bytes()
    token = authorize_registered_study(root, preflight_path)
    assert type(token) is experiment_module._RegisteredStudyAuthorization
    assert reservation_path.read_bytes() == reservation_bytes
    state = experiment_module._verify_registered_study_state(
        root,
        preflight_path,
    )
    receipt_path = experiment_module._authorization_consumption_path(state)
    assert receipt_path.is_file()
    assert receipt_path.parent.name == "artifacts"
    assert receipt_path.name == "cpp-confirmatory.authorization-consumed.json"


def test_authorization_issuance_is_atomic_for_one_reservation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, preflight_path, reservation_path, _, _ = (
        _write_authorization_state(tmp_path, monkeypatch)
    )
    reservation_bytes = reservation_path.read_bytes()
    start_barrier = threading.Barrier(3)
    tokens: list[object] = []
    errors: list[BaseException] = []
    result_lock = threading.Lock()

    def issue() -> None:
        start_barrier.wait(timeout=5.0)
        try:
            value = authorize_registered_study(root, preflight_path)
        except BaseException as exc:
            with result_lock:
                errors.append(exc)
        else:
            with result_lock:
                tokens.append(value)

    threads = [threading.Thread(target=issue) for _ in range(2)]
    for thread in threads:
        thread.start()
    start_barrier.wait(timeout=5.0)
    for thread in threads:
        thread.join(timeout=10.0)

    assert all(not thread.is_alive() for thread in threads)
    assert len(tokens) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], StudyAuthorizationError)
    assert "already issued" in str(errors[0])
    assert reservation_path.read_bytes() == reservation_bytes
    state = experiment_module._verify_registered_study_state(
        root,
        preflight_path,
    )
    assert experiment_module._authorization_consumption_path(state).is_file()


def test_authorization_replay_fails_through_api_and_fresh_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, preflight_path, _, _, _ = _write_authorization_state(
        tmp_path,
        monkeypatch,
    )
    token = authorize_registered_study(root, preflight_path)
    consumed = experiment_module._consume_registered_study_authorization(token)
    assert consumed.root == root

    with pytest.raises(StudyAuthorizationError, match="spent"):
        experiment_module._consume_registered_study_authorization(token)
    with pytest.raises(StudyAuthorizationError, match="already issued"):
        authorize_registered_study(root, preflight_path)

    script = (
        "from pathlib import Path\n"
        "import sys\n"
        "sys.path.insert(0, sys.argv[3])\n"
        "import tearrl.cpp.experiment as experiment\n"
        "experiment._require_trusted_runtime = lambda candidate: None\n"
        "experiment.verify_freeze_chain = lambda candidate: "
        "{'preregistration_sha256': 'a' * 64}\n"
        "try:\n"
        "    experiment.authorize_registered_study("
        "Path(sys.argv[1]), Path(sys.argv[2]))\n"
        "except experiment.StudyAuthorizationError as exc:\n"
        "    print(exc)\n"
        "    raise SystemExit(0 if 'already issued' in str(exc) else 2)\n"
        "raise SystemExit(3)\n"
    )
    repository_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(root),
            str(preflight_path),
            str(repository_root),
        ],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=30.0,
    )
    assert completed.returncode == 0, completed.stderr
    assert "already issued" in completed.stdout


def test_unsafe_direct_api_subprocess_is_rejected_before_authorization_work(
) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    script = (
        "from pathlib import Path\n"
        "import sys\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "import tearrl.cpp.experiment as experiment\n"
        "def forbidden(*args, **kwargs):\n"
        "    raise AssertionError('authorization work was reached')\n"
        "experiment._verify_registered_study_state = forbidden\n"
        "try:\n"
        "    experiment.authorize_registered_study("
        "Path(sys.argv[1]), Path(sys.argv[1]) / 'missing-preflight.json')\n"
        "except experiment.StudyAuthorizationError as exc:\n"
        "    print(exc)\n"
        "    raise SystemExit("
        "0 if 'trusted runtime attestation failed' in str(exc) else 2)\n"
        "raise SystemExit(3)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(repository_root)],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=30.0,
    )
    assert completed.returncode == 0, completed.stderr
    assert "trusted runtime attestation failed" in completed.stdout


def test_missing_execution_reservation_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, preflight_path, reservation_path, _, _ = (
        _write_authorization_state(tmp_path, monkeypatch)
    )
    reservation_path.unlink()
    with pytest.raises(StudyAuthorizationError, match="reservation"):
        authorize_registered_study(root, preflight_path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", "tear-cpp-execution-reservation-v9"),
        ("status", "reserved_after_seed"),
        ("confirmatory_seeds", list(reversed(CONFIRMATORY_SEEDS))),
        ("preflight_sha256", "0" * 64),
    ],
)
def test_mutated_execution_reservation_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    root, preflight_path, reservation_path, _, reservation = (
        _write_authorization_state(tmp_path, monkeypatch)
    )
    reservation[field] = value
    reservation_path.write_bytes(_stable_json_bytes(reservation))
    with pytest.raises(StudyAuthorizationError, match="reservation"):
        authorize_registered_study(root, preflight_path)


def test_noncanonical_execution_reservation_bytes_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, preflight_path, reservation_path, _, reservation = (
        _write_authorization_state(tmp_path, monkeypatch)
    )
    reservation_path.write_bytes(
        json.dumps(reservation, sort_keys=True).encode("utf-8")
    )
    with pytest.raises(StudyAuthorizationError, match="canonical"):
        authorize_registered_study(root, preflight_path)


def test_preflight_semantics_are_reverified_before_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        root,
        preflight_path,
        reservation_path,
        preflight,
        reservation,
    ) = _write_authorization_state(tmp_path, monkeypatch)
    preflight["scientific_metrics_suppressed"] = False
    mutated_payload = _stable_json_bytes(preflight)
    preflight_path.write_bytes(mutated_payload)
    reservation["preflight_sha256"] = hashlib.sha256(
        mutated_payload
    ).hexdigest()
    reservation_path.write_bytes(_stable_json_bytes(reservation))
    with pytest.raises(
        StudyAuthorizationError,
        match="scientific_metrics_suppressed",
    ):
        authorize_registered_study(root, preflight_path)


def test_forged_authorization_is_rejected_before_registered_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def fail_if_called(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("forged capability reached registered work")

    monkeypatch.setattr(
        experiment_module,
        "_run_registered_seed_unchecked",
        fail_if_called,
    )
    with pytest.raises(TypeError, match="issued only after verification"):
        experiment_module._RegisteredStudyAuthorization()
    forged = object.__new__(experiment_module._RegisteredStudyAuthorization)
    with pytest.raises(StudyAuthorizationError, match="forged"):
        run_confirmatory_study(
            authorization=forged,
            watchdog=Watchdog.start(),
        )
    assert called is False


def test_authorized_state_is_reverified_before_registered_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, preflight_path, reservation_path, _, reservation = (
        _write_authorization_state(tmp_path, monkeypatch)
    )
    token = authorize_registered_study(root, preflight_path)
    reservation["status"] = "mutated_after_authorization"
    reservation_path.write_bytes(_stable_json_bytes(reservation))
    called = False

    def fail_if_called(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("mutated state reached registered work")

    monkeypatch.setattr(
        experiment_module,
        "_run_registered_seed_unchecked",
        fail_if_called,
    )
    with pytest.raises(StudyAuthorizationError, match="reservation"):
        run_confirmatory_study(
            authorization=token,
            watchdog=Watchdog.start(),
        )
    assert called is False


def test_durable_authorization_receipt_is_reverified_before_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, preflight_path, _, _, _ = _write_authorization_state(
        tmp_path,
        monkeypatch,
    )
    token = authorize_registered_study(root, preflight_path)
    issued = experiment_module._AUTHORIZATION_STATES[token]
    experiment_module._authorization_consumption_path(
        issued.verified
    ).write_bytes(b"{}\n")
    called = False

    def fail_if_called(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("mutated receipt reached registered work")

    monkeypatch.setattr(
        experiment_module,
        "_run_registered_seed_unchecked",
        fail_if_called,
    )
    with pytest.raises(
        StudyAuthorizationError,
        match="durable authorization",
    ):
        run_confirmatory_study(
            authorization=token,
            watchdog=Watchdog.start(),
        )
    assert called is False


def test_synthetic_full_seed_is_bit_deterministic() -> None:
    # This seed is neither timing-only nor confirmatory. The fixture exercises
    # the exact registered budget but makes no scientific pass/fail assertion.
    first = run_registered_seed(101)
    second = run_registered_seed(101)

    assert first.posterior_rows == second.posterior_rows
    assert first.calibration_rows == second.calibration_rows
    assert first.decision_rows == second.decision_rows
    assert first.training_audit == second.training_audit
    assert first.b_frozen_diagnostic == second.b_frozen_diagnostic
    assert first.integrity == second.integrity
    assert first.integrity["c_d_frozen_digests_equal"] is True
    assert first.integrity["frozen_digest_stable_after_all_policy_cells"] is True
    assert first.integrity["common_estimator_digest_stable"] is True


def test_cd_rows_use_bit_identical_frozen_probabilities() -> None:
    execution = run_registered_seed(102)
    grouped: dict[tuple[str, str, int], dict[str, float]] = {}
    for row in execution.posterior_rows:
        if row.arm in {"C", "D"}:
            grouped.setdefault(
                (row.base_world_id, row.condition, row.difficulty), {}
            )[row.arm] = row.q_post
    assert grouped
    assert all(cells["C"] == cells["D"] for cells in grouped.values())
