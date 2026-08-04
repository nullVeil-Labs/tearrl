"""Matched task policies and registered A/B learning paths for TEAR-CPP."""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Callable, Iterable, Literal

import numpy as np
import torch
from torch import nn

from .calibration import (
    SCOPE_HIDDEN_DIM,
    SCOPE_INPUT_DIM,
    FrozenScopeEstimator,
    ScopeEstimator,
    validated_scope_features,
)


N_TERMINAL_ACTIONS = 3
SAFE_COMPLETE_INDEX = 0
IRREVERSIBLE_COMMIT_INDEX = 1
REPORT_EXIT_INDEX = 2
POLICY_EPOCHS = 80
POLICY_LEARNING_RATE = 0.003
THRESHOLD = 0.80
SURROGATE_SCALE = 0.05

ArmName = Literal["A", "B", "C", "D"]


def _copy_linear(target: nn.Linear, source: nn.Linear) -> None:
    with torch.no_grad():
        target.weight.copy_(source.weight)
        target.bias.copy_(source.bias)


class _ActorPath(nn.Module):
    """The exactly matched trainable actor path shared by all four arms."""

    def __init__(self) -> None:
        super().__init__()
        self.input_layer = nn.Linear(SCOPE_INPUT_DIM, SCOPE_HIDDEN_DIM)
        self.hidden_layer = nn.Linear(SCOPE_HIDDEN_DIM, SCOPE_HIDDEN_DIM)
        self.actor_head = nn.Linear(SCOPE_HIDDEN_DIM, N_TERMINAL_ACTIONS)

    def representation(
        self,
        features: torch.Tensor | np.ndarray | Iterable[float],
    ) -> torch.Tensor:
        inputs, _ = validated_scope_features(features)
        hidden = torch.tanh(self.input_layer(inputs))
        return torch.tanh(self.hidden_layer(hidden))

    def actor_logits(
        self,
        features: torch.Tensor | np.ndarray | Iterable[float],
    ) -> torch.Tensor:
        return self.actor_head(self.representation(features))

    def proposal_probabilities(
        self,
        features: torch.Tensor | np.ndarray | Iterable[float],
    ) -> torch.Tensor:
        return torch.softmax(self.actor_logits(features), dim=-1)

    def actor_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(self.parameters())


class CoupledPolicy(_ActorPath):
    """A/B actor path plus fixed output head on the shared representation."""

    def __init__(self) -> None:
        super().__init__()
        self.scope_head = nn.Linear(SCOPE_HIDDEN_DIM, 1)
        self.register_buffer(
            "scope_temperature",
            torch.tensor(1.0, dtype=torch.float32),
        )
        # Section 9 excludes scope-output weights from every policy optimizer.
        for parameter in self.scope_head.parameters():
            parameter.requires_grad_(False)

    def scope_logits(
        self,
        features: torch.Tensor | np.ndarray | Iterable[float],
    ) -> torch.Tensor:
        return (
            self.scope_head(self.representation(features)).squeeze(-1) /
            self.scope_temperature
        )

    def scope_probabilities(
        self,
        features: torch.Tensor | np.ndarray | Iterable[float],
    ) -> torch.Tensor:
        return torch.sigmoid(self.scope_logits(features))

    def actor_parameters(self) -> tuple[nn.Parameter, ...]:
        # Be explicit: the fixed diagnostic output head is never optimized.
        return tuple(self.input_layer.parameters()) + tuple(
            self.hidden_layer.parameters()) + tuple(self.actor_head.parameters())


class SeparateActor(_ActorPath):
    """C/D actor with no estimator module or estimator parameter path."""


def _initialized_actor_head(seed: int) -> nn.Linear:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        return nn.Linear(SCOPE_HIDDEN_DIM, N_TERMINAL_ACTIONS)


def _initialize_actor_path(
    policy: _ActorPath,
    estimator: ScopeEstimator,
    actor_head: nn.Linear,
) -> None:
    _copy_linear(policy.input_layer, estimator.input_layer)
    _copy_linear(policy.hidden_layer, estimator.hidden_layer)
    _copy_linear(policy.actor_head, actor_head)


def _initialize_coupled_scope(
    policy: CoupledPolicy,
    estimator: ScopeEstimator,
) -> None:
    _copy_linear(policy.scope_head, estimator.output_layer)
    with torch.no_grad():
        policy.scope_temperature.copy_(estimator.temperature)


@dataclass(frozen=True, slots=True)
class MatchedPolicies:
    a: CoupledPolicy
    b: CoupledPolicy
    c: SeparateActor
    d: SeparateActor
    c_estimator: FrozenScopeEstimator
    d_estimator: FrozenScopeEstimator

    def assert_matched_actor_initialization(self) -> None:
        states = (
            self.a.state_dict(),
            self.b.state_dict(),
            self.c.state_dict(),
            self.d.state_dict(),
        )
        actor_names = (
            "input_layer.weight",
            "input_layer.bias",
            "hidden_layer.weight",
            "hidden_layer.bias",
            "actor_head.weight",
            "actor_head.bias",
        )
        for name in actor_names:
            reference = states[0][name]
            if not all(torch.equal(reference, state[name]) for state in states[1:]):
                raise AssertionError(f"unmatched actor initialization: {name}")
        counts = tuple(sum(
            parameter.numel() for parameter in policy.actor_parameters()
        ) for policy in (self.a, self.b, self.c, self.d))
        if len(set(counts)) != 1:
            raise AssertionError(f"unmatched actor parameter counts: {counts}")
        self.c_estimator.assert_integrity()
        self.d_estimator.assert_integrity()
        if self.c_estimator.digest != self.d_estimator.digest:
            raise AssertionError("C/D frozen estimator copies are not identical")


def build_matched_policies(
    calibrated_estimator: ScopeEstimator,
    *,
    actor_seed: int,
) -> MatchedPolicies:
    """Create independently trainable cells from one common initialization."""

    calibrated_estimator.eval()
    common_head = _initialized_actor_head(actor_seed)
    policies: list[_ActorPath] = [
        CoupledPolicy(), CoupledPolicy(), SeparateActor(), SeparateActor()]
    for policy in policies:
        _initialize_actor_path(policy, calibrated_estimator, common_head)
    a, b = policies[0], policies[1]
    assert isinstance(a, CoupledPolicy)
    assert isinstance(b, CoupledPolicy)
    _initialize_coupled_scope(a, calibrated_estimator)
    _initialize_coupled_scope(b, calibrated_estimator)
    matched = MatchedPolicies(
        a=a,
        b=b,
        c=policies[2],
        d=policies[3],
        c_estimator=FrozenScopeEstimator.from_estimator(calibrated_estimator),
        d_estimator=FrozenScopeEstimator.from_estimator(calibrated_estimator),
    )
    assert isinstance(matched.c, SeparateActor)
    assert isinstance(matched.d, SeparateActor)
    matched.assert_matched_actor_initialization()
    return matched


def threshold_surrogate(scope_probability: torch.Tensor) -> torch.Tensor:
    if not bool(torch.isfinite(scope_probability.detach()).all()):
        raise ValueError("scope probabilities must be finite")
    if not bool(torch.logical_and(
            scope_probability.detach() >= 0.0,
            scope_probability.detach() <= 1.0).all()):
        raise ValueError("scope probabilities must lie in [0, 1]")
    return torch.sigmoid(
        (scope_probability - THRESHOLD) / SURROGATE_SCALE)


def hard_threshold_allows(scope_probability: float) -> bool:
    value = float(scope_probability)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("scope probability must be finite and in [0, 1]")
    return value >= THRESHOLD


def hard_threshold_execute(proposed_action: int,
                           scope_probability: float) -> int:
    action = int(proposed_action)
    if action not in range(N_TERMINAL_ACTIONS):
        raise ValueError("unknown terminal action index")
    if (action == IRREVERSIBLE_COMMIT_INDEX and
            not hard_threshold_allows(scope_probability)):
        return SAFE_COMPLETE_INDEX
    return action


def task_only_expected_value(
    proposal_probabilities: torch.Tensor,
    *,
    q_safe: float,
) -> torch.Tensor:
    _validate_proposals_and_safe_value(proposal_probabilities, q_safe)
    return (
        proposal_probabilities[:, SAFE_COMPLETE_INDEX] * float(q_safe) +
        proposal_probabilities[:, IRREVERSIBLE_COMMIT_INDEX]
    )


def threshold_expected_value(
    proposal_probabilities: torch.Tensor,
    scope_probability: torch.Tensor,
    *,
    q_safe: float,
) -> torch.Tensor:
    """B's soft training executor; blocked commit mass becomes safe completion."""

    _validate_proposals_and_safe_value(proposal_probabilities, q_safe)
    if scope_probability.shape != proposal_probabilities.shape[:1]:
        raise ValueError("scope probability must have one value per proposal row")
    gate = threshold_surrogate(scope_probability)
    commit_value = gate + (1.0 - gate) * float(q_safe)
    return (
        proposal_probabilities[:, SAFE_COMPLETE_INDEX] * float(q_safe) +
        proposal_probabilities[:, IRREVERSIBLE_COMMIT_INDEX] * commit_value
    )


def _validate_proposals_and_safe_value(
    proposal_probabilities: torch.Tensor,
    q_safe: float,
) -> None:
    if (proposal_probabilities.ndim != 2 or
            proposal_probabilities.shape[1] != N_TERMINAL_ACTIONS):
        raise ValueError(
            f"proposal probabilities must be [batch, {N_TERMINAL_ACTIONS}]")
    if not bool(torch.isfinite(proposal_probabilities.detach()).all()):
        raise ValueError("proposal probabilities must be finite")
    value = float(q_safe)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("q_safe must be finite and in [0, 1]")


@dataclass(frozen=True, slots=True)
class PolicyTrainingReport:
    arm: ArmName
    rows: int
    epochs: int
    learning_rate: float
    q_safe: float
    initial_expected_task: float
    final_expected_task: float
    frozen_estimator_digest: str | None


def _expected_task(
    arm: ArmName,
    policy: CoupledPolicy | SeparateActor,
    inputs: torch.Tensor,
    q_safe: float,
) -> torch.Tensor:
    proposals = policy.proposal_probabilities(inputs)
    if arm == "B":
        if not isinstance(policy, CoupledPolicy):
            raise TypeError("Arm B requires a coupled policy")
        return threshold_expected_value(
            proposals, policy.scope_probabilities(inputs), q_safe=q_safe)
    return task_only_expected_value(proposals, q_safe=q_safe)


def train_policy_cell(
    arm: ArmName,
    policy: CoupledPolicy | SeparateActor,
    features: torch.Tensor | np.ndarray | Iterable[float],
    *,
    q_safe: float,
    epochs: int = POLICY_EPOCHS,
    learning_rate: float = POLICY_LEARNING_RATE,
    frozen_estimator: FrozenScopeEstimator | None = None,
    checkpoint: Callable[[str], None] | None = None,
) -> PolicyTrainingReport:
    """Run one registered arm/difficulty full-batch policy cell."""

    if arm not in ("A", "B", "C", "D"):
        raise ValueError(f"unknown TEAR-CPP arm: {arm}")
    if epochs != POLICY_EPOCHS:
        raise ValueError(
            f"TEAR-CPP policy training requires exactly {POLICY_EPOCHS} epochs")
    if learning_rate != POLICY_LEARNING_RATE:
        raise ValueError(
            f"TEAR-CPP policy learning rate must be {POLICY_LEARNING_RATE}")
    if arm in ("A", "B") and not isinstance(policy, CoupledPolicy):
        raise TypeError(f"Arm {arm} requires a coupled policy")
    if arm in ("C", "D") and not isinstance(policy, SeparateActor):
        raise TypeError(f"Arm {arm} requires a separate actor")
    if arm in ("C", "D") and frozen_estimator is None:
        raise ValueError(f"Arm {arm} requires its frozen estimator assertion")
    if arm in ("A", "B") and frozen_estimator is not None:
        raise ValueError(f"Arm {arm} must not receive a C/D frozen estimator")

    inputs, _ = validated_scope_features(features)
    optimizer = torch.optim.Adam(
        policy.actor_parameters(), lr=POLICY_LEARNING_RATE)
    frozen_digest = None
    if frozen_estimator is not None:
        frozen_estimator.assert_integrity(optimizer)
        frozen_digest = frozen_estimator.digest

    policy.train()
    with torch.no_grad():
        initial = float(_expected_task(arm, policy, inputs, q_safe).mean())
    for epoch in range(1, POLICY_EPOCHS + 1):
        if checkpoint is not None:
            checkpoint(f"policy {arm} epoch {epoch}/{POLICY_EPOCHS}")
        optimizer.zero_grad(set_to_none=True)
        expected_task = _expected_task(arm, policy, inputs, q_safe).mean()
        (-expected_task).backward()
        optimizer.step()
    policy.eval()
    with torch.no_grad():
        final = float(_expected_task(arm, policy, inputs, q_safe).mean())
    if frozen_estimator is not None:
        frozen_estimator.assert_integrity(optimizer)
    return PolicyTrainingReport(
        arm=arm,
        rows=len(inputs),
        epochs=POLICY_EPOCHS,
        learning_rate=POLICY_LEARNING_RATE,
        q_safe=float(q_safe),
        initial_expected_task=initial,
        final_expected_task=final,
        frozen_estimator_digest=frozen_digest,
    )


def clone_policy(
    policy: CoupledPolicy | SeparateActor,
) -> CoupledPolicy | SeparateActor:
    """Explicit helper for independent difficulty cells."""

    return copy.deepcopy(policy)
