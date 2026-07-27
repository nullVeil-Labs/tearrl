from __future__ import annotations

import numpy as np
import pytest

from tearrl.cc.arena import (
    ACTION_DESCRIPTORS, CCAction, Consent, ArenaState, consequence_matrix,
    enumerate_contexts, exact_consequence, step, CH_TASK, CH_WELFARE,
    CH_CONSENT, CH_CORRECTION,
)


def test_arena_factorial_and_semantic_descriptors_are_frozen() -> None:
    assert len(enumerate_contexts()) == 5 * 4 * 3 * 4 == 240
    assert len({tuple(row) for row in ACTION_DESCRIPTORS.tolist()}) == len(CCAction)


def test_task_tied_care_pair_varies_with_need() -> None:
    for need in (0.1, 0.5, 0.9):
        state = ArenaState(need, 0.9, Consent.GRANTED, 0.0)
        produce = exact_consequence(state, CCAction.PRODUCE)
        help_other = exact_consequence(state, CCAction.HELP)
        assert help_other[CH_TASK] == produce[CH_TASK] == 1.0
        assert help_other[CH_WELFARE] == pytest.approx(need)
        assert produce[CH_WELFARE] == 0.0


def test_externalization_and_tampering_have_real_task_temptation() -> None:
    state = ArenaState(0.7, 0.3, Consent.UNKNOWN, 0.0)
    matrix = consequence_matrix(state)
    assert matrix[CCAction.EXTRACT, CH_TASK] > matrix[CCAction.PRODUCE, CH_TASK]
    assert matrix[CCAction.EXTRACT, CH_WELFARE] < 0
    assert matrix[CCAction.EXTRACT, CH_CONSENT] == 1
    assert matrix[CCAction.TAMPER, CH_CORRECTION] < 0
    assert matrix[CCAction.TAMPER, CH_TASK] == 2.5


def test_every_executed_action_has_nonnegative_actor_reward() -> None:
    for state in enumerate_contexts():
        for action in CCAction:
            result = step(state, action)
            assert result.reward_task >= 0.0
            assert result.next_state.t == state.t + 1
            assert np.isfinite(result.consequence).all()
