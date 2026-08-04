from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError

import pytest
import torch

from tearrl.cpp.config import B_HARD_THRESHOLD, D_EXIT_THRESHOLD, SAFE_TASK_VALUES
from tearrl.cpp.gate import (
    execute_expected_cost_gate,
    expected_cost_decision,
    host_probability,
    threshold_allows_commit,
)
from tearrl.cpp.schema import (
    ActionClass,
    Arm,
    Difficulty,
    GateReason,
    TerminalAction,
)
from tearrl.cpp.severity import SEVERITY_BY_ACTION_CLASS, severity_for


def test_literal_severity_table_is_complete_exact_and_immutable() -> None:
    assert set(SEVERITY_BY_ACTION_CLASS) == set(ActionClass)
    assert dict(SEVERITY_BY_ACTION_CLASS) == {
        ActionClass.MOVEMENT: 0,
        ActionClass.OBSERVATION: 0,
        ActionClass.SAFE_COMPLETION: 0,
        ActionClass.REPORT_EXIT: 0,
        ActionClass.IRREVERSIBLE_COMMITMENT: 8,
    }
    assert severity_for(ActionClass.IRREVERSIBLE_COMMITMENT) == 8
    with pytest.raises(TypeError):
        SEVERITY_BY_ACTION_CLASS[ActionClass.IRREVERSIBLE_COMMITMENT] = 0


def test_expected_cost_uses_out_of_scope_probability_with_the_registered_sign() -> None:
    certain_out = expected_cost_decision(0.0, Difficulty.D0)
    certain_in = expected_cost_decision(1.0, Difficulty.D0)
    assert certain_out.commit_value == -7.0
    assert certain_in.commit_value == 1.0
    assert certain_out.severity == certain_in.severity == 8
    assert certain_out.commit_value < certain_in.commit_value


@pytest.mark.parametrize("difficulty", list(Difficulty))
def test_expected_cost_ties_allow_commitment_and_just_below_blocks(
    difficulty: Difficulty,
) -> None:
    safe_value = SAFE_TASK_VALUES[int(difficulty)]
    tie_q = 1.0 - (1.0 - safe_value) / 8.0
    tie = expected_cost_decision(tie_q, difficulty)
    assert tie.commit_value == pytest.approx(safe_value)
    assert tie.allowed
    assert not expected_cost_decision(tie_q - 1e-10, difficulty).allowed


def test_c_and_d_share_allow_block_decision_and_only_fallback_differs() -> None:
    for q in (0.0, 0.4, 0.5, 0.6, 0.99, 1.0):
        c = execute_expected_cost_gate(Arm.C, q, Difficulty.D2)
        d = execute_expected_cost_gate(Arm.D, q, Difficulty.D2)
        assert c.decision.allowed == d.decision.allowed
        assert c.decision.commit_value == d.decision.commit_value
        if c.decision.allowed:
            assert c.executed_action is d.executed_action is (
                TerminalAction.IRREVERSIBLE_COMMIT
            )
            assert c.reason is d.reason is GateReason.COMMIT_ALLOWED
        else:
            assert c.executed_action is TerminalAction.SAFE_COMPLETE
            assert c.reason is GateReason.COMMIT_BLOCKED_SAFE
            if q <= D_EXIT_THRESHOLD:
                assert d.executed_action is TerminalAction.REPORT_EXIT
                assert d.reason is GateReason.COMMIT_BLOCKED_EXIT
            else:
                assert d.executed_action is TerminalAction.SAFE_COMPLETE
                assert d.reason is GateReason.COMMIT_BLOCKED_SAFE


def test_d_exit_boundary_is_inclusive() -> None:
    at_boundary = execute_expected_cost_gate(Arm.D, 0.50, Difficulty.D3)
    above_boundary = execute_expected_cost_gate(Arm.D, 0.500001, Difficulty.D3)
    assert not at_boundary.decision.allowed
    assert not above_boundary.decision.allowed
    assert at_boundary.executed_action is TerminalAction.REPORT_EXIT
    assert above_boundary.executed_action is TerminalAction.SAFE_COMPLETE


def test_b_threshold_boundary_is_inclusive_and_hard() -> None:
    assert not threshold_allows_commit(B_HARD_THRESHOLD - 1e-12)
    assert threshold_allows_commit(B_HARD_THRESHOLD)
    assert threshold_allows_commit(1.0)


def test_gate_detaches_tensor_to_host_scalar_without_gradient_path() -> None:
    posterior = torch.tensor(0.999, requires_grad=True)
    decision = expected_cost_decision(posterior, Difficulty.D1)
    assert isinstance(decision.q_in_scope, float)
    assert isinstance(decision.commit_value, float)
    assert posterior.grad is None

    downstream = torch.tensor(decision.commit_value, requires_grad=True)
    downstream.backward()
    assert posterior.grad is None


@pytest.mark.parametrize(
    "value",
    (-0.001, 1.001, float("nan"), float("inf"), True),
)
def test_host_probability_rejects_invalid_scalars(value) -> None:
    with pytest.raises((TypeError, ValueError)):
        host_probability(value)


def test_host_probability_rejects_multielement_tensor() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        host_probability(torch.tensor([0.2, 0.8], requires_grad=True))


def test_expected_cost_gate_accepts_no_world_truth_or_severity_override() -> None:
    parameters = set(inspect.signature(expected_cost_decision).parameters)
    assert parameters == {"q_in_scope", "difficulty"}
    assert not parameters & {
        "world",
        "scope_truth",
        "hidden_rule",
        "hidden_marker",
        "severity",
        "reward",
    }


def test_expected_cost_gate_is_c_d_only_and_decisions_are_frozen() -> None:
    for arm in (Arm.A, Arm.B):
        with pytest.raises(ValueError, match="only for arms C and D"):
            execute_expected_cost_gate(arm, 0.9, Difficulty.D0)
    execution = execute_expected_cost_gate(Arm.C, 1.0, Difficulty.D0)
    with pytest.raises(FrozenInstanceError):
        execution.executed_action = TerminalAction.SAFE_COMPLETE
    with pytest.raises(FrozenInstanceError):
        execution.decision.allowed = False
