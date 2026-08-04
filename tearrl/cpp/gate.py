"""Non-differentiable hard gates for TEAR-CPP confirmatory evaluation."""

from __future__ import annotations

from math import isfinite
from numbers import Real
from typing import Any

from .config import (
    B_HARD_THRESHOLD,
    D_EXIT_THRESHOLD,
    IRREVERSIBLE_TASK_VALUE,
    safe_task_value,
)
from .schema import (
    ActionClass,
    Arm,
    Difficulty,
    ExpectedCostDecision,
    GateExecution,
    GateReason,
    TerminalAction,
)
from .severity import severity_for


def host_probability(value: Any) -> float:
    """Detach a scalar tensor-like value and return a validated host float."""

    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
        numel = getattr(value, "numel", None)
        if callable(numel) and int(numel()) != 1:
            raise ValueError("posterior must contain exactly one scalar")
        cpu = getattr(value, "cpu", None)
        if callable(cpu):
            value = cpu()
        item = getattr(value, "item", None)
        if not callable(item):
            raise TypeError("detached posterior does not expose a scalar item")
        value = item()
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("posterior must be a real scalar or scalar tensor")
    result = float(value)
    if not isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("posterior must be finite and within [0, 1]")
    return result


def threshold_allows_commit(q_in_scope: Any) -> bool:
    """Arm B's exact confirmatory threshold; equality allows commitment."""

    return host_probability(q_in_scope) >= B_HARD_THRESHOLD


def expected_cost_decision(
    q_in_scope: Any,
    difficulty: Difficulty | int,
) -> ExpectedCostDecision:
    """C/D's exact hard expected-cost comparison over a detached scalar."""

    q = host_probability(q_in_scope)
    normalized_difficulty = Difficulty(difficulty)
    severity = severity_for(ActionClass.IRREVERSIBLE_COMMITMENT)
    commit_value = IRREVERSIBLE_TASK_VALUE - severity * (1.0 - q)
    safe_value = safe_task_value(normalized_difficulty)
    allow_threshold = 1.0 - (
        IRREVERSIBLE_TASK_VALUE - safe_value
    ) / severity
    return ExpectedCostDecision(
        q_in_scope=q,
        difficulty=normalized_difficulty,
        severity=severity,
        commit_value=float(commit_value),
        safe_value=safe_value,
        allowed=bool(q >= allow_threshold),
    )


def execute_expected_cost_gate(
    arm: Arm | str,
    q_in_scope: Any,
    difficulty: Difficulty | int,
) -> GateExecution:
    """Apply the common C/D allow/block decision and registered fallback."""

    normalized_arm = Arm(arm)
    if normalized_arm not in {Arm.C, Arm.D}:
        raise ValueError("the expected-cost gate is registered only for arms C and D")
    decision = expected_cost_decision(q_in_scope, difficulty)
    if decision.allowed:
        executed = TerminalAction.IRREVERSIBLE_COMMIT
        reason = GateReason.COMMIT_ALLOWED
    elif normalized_arm is Arm.D and decision.q_in_scope <= D_EXIT_THRESHOLD:
        executed = TerminalAction.REPORT_EXIT
        reason = GateReason.COMMIT_BLOCKED_EXIT
    else:
        executed = TerminalAction.SAFE_COMPLETE
        reason = GateReason.COMMIT_BLOCKED_SAFE
    return GateExecution(
        arm=normalized_arm,
        decision=decision,
        proposed_action=TerminalAction.IRREVERSIBLE_COMMIT,
        executed_action=executed,
        reason=reason,
    )
