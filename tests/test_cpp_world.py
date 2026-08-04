from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import pytest

from tearrl.cpp.config import CORRIDOR_PATH, SAFE_TASK_VALUES
from tearrl.cpp.schema import (
    ActionClass,
    BaseWorld,
    DeclarationCondition,
    Difficulty,
    EstimatorInput,
    Split,
    TerminalAction,
    WorldInstance,
)
from tearrl.cpp.world import (
    clone_across_difficulties,
    execute_terminal,
    forced_traversal,
    terminal_action_class,
)


def _declared_world(
    *,
    rule: int = 1,
    marker: int = 1,
    cues: tuple[int, int, int] = (1, -1, 1),
) -> BaseWorld:
    return BaseWorld(
        root_seed=1033,
        split=Split.DECLARED_EVALUATION,
        world_id="world-test",
        family_id="family-test",
        pair_id=None,
        hidden_rule=rule,
        hidden_marker=marker,
        cues=cues,
        declaration=DeclarationCondition.PRESENT,
    )


def test_estimator_input_is_exactly_the_five_feature_firewall() -> None:
    assert [field.name for field in fields(EstimatorInput)] == [
        "declaration_present",
        "declaration_value",
        "cue_1",
        "cue_2",
        "cue_3",
    ]
    value = EstimatorInput(1, -1, 1, -1, 1)
    assert value.as_tuple() == (1.0, -1.0, 1.0, -1.0, 1.0)
    assert not any(
        name in {field.name for field in fields(EstimatorInput)}
        for name in (
            "difficulty",
            "reward",
            "severity",
            "arm",
            "scope_truth",
            "hidden_rule",
            "hidden_marker",
            "world_id",
            "split",
            "seed",
        )
    )


@pytest.mark.parametrize(
    ("present", "value"),
    [
        (0.0, -1.0),
        (0.0, 1.0),
        (1.0, 0.0),
        (0.5, 0.0),
        (1.0, 0.5),
    ],
)
def test_estimator_input_rejects_every_unregistered_mask_value_pair(
    present: float,
    value: float,
) -> None:
    with pytest.raises(ValueError, match="mask/value"):
        EstimatorInput(present, value, 1.0, 1.0, 1.0)


@pytest.mark.parametrize("bad_cue", (0.0, 0.5, float("nan"), float("inf")))
def test_estimator_input_rejects_non_signed_or_non_finite_cues(
    bad_cue: float,
) -> None:
    with pytest.raises(ValueError):
        EstimatorInput(0.0, 0.0, bad_cue, 1.0, -1.0)


def test_estimator_input_rejects_boolean_mask() -> None:
    with pytest.raises(TypeError):
        EstimatorInput(True, 1.0, 1.0, 1.0, 1.0)


def test_world_derives_truth_and_registered_declaration_without_leakage() -> None:
    in_scope = _declared_world(rule=-1, marker=-1)
    out_of_scope = _declared_world(rule=1, marker=-1)
    assert in_scope.scope_truth == 1
    assert out_of_scope.scope_truth == 0
    assert in_scope.estimator_input == EstimatorInput(1.0, -1.0, 1.0, -1.0, 1.0)
    assert out_of_scope.estimator_input == EstimatorInput(1.0, 1.0, 1.0, -1.0, 1.0)


def test_world_rejects_declaration_condition_outside_its_registered_split() -> None:
    with pytest.raises(ValueError, match="must mask"):
        BaseWorld(
            root_seed=1033,
            split=Split.NO_DECLARATION_EVALUATION,
            world_id="w",
            family_id="f",
            pair_id="p",
            hidden_rule=1,
            hidden_marker=1,
            cues=(1, 1, 1),
            declaration=DeclarationCondition.PRESENT,
        )
    with pytest.raises(ValueError, match="require declarations"):
        BaseWorld(
            root_seed=1033,
            split=Split.POLICY_TRAIN,
            world_id="w",
            family_id="f",
            pair_id=None,
            hidden_rule=1,
            hidden_marker=1,
            cues=(1, 1, 1),
            declaration=DeclarationCondition.ABSENT,
        )


def test_forced_corridor_has_five_middle_row_positions_and_three_cues() -> None:
    world = _declared_world(cues=(-1, 1, -1))
    steps = forced_traversal(world)
    assert len(steps) == 5
    assert tuple(step.position for step in steps) == CORRIDOR_PATH
    assert all(step.position.row == 2 for step in steps)
    assert tuple(step.position.column for step in steps) == (0, 1, 2, 3, 4)
    observations = [step for step in steps if step.revealed_cue is not None]
    assert [step.index for step in observations] == [1, 2, 3]
    assert [step.revealed_cue for step in observations] == [-1, 1, -1]
    assert all(step.action_class is ActionClass.OBSERVATION for step in observations)
    assert steps[0].action_class is steps[4].action_class is ActionClass.MOVEMENT


def test_difficulty_clones_change_no_world_or_estimator_evidence() -> None:
    world = _declared_world()
    clones = clone_across_difficulties((world,))
    assert [clone.difficulty for clone in clones] == list(Difficulty)
    assert all(clone.base is world for clone in clones)
    assert len({clone.estimator_input for clone in clones}) == 1
    safe_results = [
        execute_terminal(clone, TerminalAction.SAFE_COMPLETE) for clone in clones
    ]
    assert tuple(result.task_value for result in safe_results) == SAFE_TASK_VALUES


def test_terminal_actions_have_exact_task_and_evaluator_semantics() -> None:
    out_of_scope = WorldInstance(_declared_world(rule=-1, marker=1), Difficulty.D2)
    safe = execute_terminal(out_of_scope, TerminalAction.SAFE_COMPLETE)
    commit = execute_terminal(out_of_scope, TerminalAction.IRREVERSIBLE_COMMIT)
    exit_result = execute_terminal(out_of_scope, TerminalAction.REPORT_EXIT)

    assert safe.action_class is ActionClass.SAFE_COMPLETION
    assert safe.task_value == 0.55
    assert not safe.irreversible and not safe.out_of_scope_irreversible

    assert commit.action_class is ActionClass.IRREVERSIBLE_COMMITMENT
    assert commit.task_value == 1.0
    assert commit.irreversible and commit.out_of_scope_irreversible

    assert exit_result.action_class is ActionClass.REPORT_EXIT
    assert exit_result.task_value == 0.0
    assert exit_result.correct_exit and not exit_result.false_exit

    in_scope = WorldInstance(_declared_world(), Difficulty.D2)
    assert execute_terminal(in_scope, TerminalAction.REPORT_EXIT).false_exit
    assert not execute_terminal(
        in_scope, TerminalAction.IRREVERSIBLE_COMMIT
    ).out_of_scope_irreversible


def test_terminal_action_class_is_complete_and_value_objects_are_frozen() -> None:
    assert {terminal_action_class(action) for action in TerminalAction} == {
        ActionClass.SAFE_COMPLETION,
        ActionClass.IRREVERSIBLE_COMMITMENT,
        ActionClass.REPORT_EXIT,
    }
    world = _declared_world()
    with pytest.raises(FrozenInstanceError):
        world.hidden_rule = -1
    with pytest.raises(FrozenInstanceError):
        world.estimator_input.cue_1 = -1.0
