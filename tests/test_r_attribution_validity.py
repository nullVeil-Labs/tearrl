"""Anti-vacuity tests for the responsibility attribution channel ``r``.

These tests measure the frozen implementation and candidate corrections.  They
do not amend the TEARRL-0 reward, responsibility recurrence, or denominator.
"""

from __future__ import annotations

import numpy as np
import pytest

from tearrl import TearWorld, Verb
from tearrl.eval.attribution import (
    attribution_ratios,
    comparison_summary,
    distribution_summary,
    ratio_series,
)
from tearrl.predicates import action_effect
from tearrl.spec import EPS_R, RESOURCES_MAX


def test_learner_causes_the_entire_welfare_delta() -> None:
    result = attribution_ratios(learner_delta=0.5, intervening_delta=0.0)

    # Frozen reading excludes the current learner action from the denominator.
    assert result.current == pytest.approx(0.5 / EPS_R)
    # Both complete-window readings agree when no other cause contributed.
    expected = 0.5 / (0.5 + EPS_R)
    assert result.completed_window == pytest.approx(expected)
    assert result.contribution_share == pytest.approx(expected)


def test_learner_causes_a_known_fraction_of_same_direction_change() -> None:
    result = attribution_ratios(learner_delta=0.25, intervening_delta=0.75)
    assert result.current == pytest.approx(0.25 / (0.75 + EPS_R))
    expected = 0.25 / (1.0 + EPS_R)
    assert result.completed_window == pytest.approx(expected)
    assert result.contribution_share == pytest.approx(expected)


def test_counterpart_or_environment_causes_the_entire_change() -> None:
    result = attribution_ratios(learner_delta=0.0, intervening_delta=-0.5)
    assert result.current == 0.0
    assert result.completed_window == 0.0
    assert result.contribution_share == 0.0


def test_zero_denominator_case_is_defined_analytically() -> None:
    result = attribution_ratios(learner_delta=0.0, intervening_delta=0.0)
    assert result.current == 0.0
    assert result.completed_window == 0.0
    assert result.contribution_share == 0.0


def test_effect_clipped_by_world_limits_has_zero_learner_attribution() -> None:
    world = TearWorld(seed=450)
    world.reset(seed=450, learner_index=0)
    world.intervene(m_e=0)
    world.step(Verb.WAIT)  # slot 0 acts next
    world.state.agents[0].resources = 1
    world.state.agents[1].resources = RESOURCES_MAX

    effect = action_effect(world, Verb.SHARE, 0)
    assert effect.feasible
    assert effect.d_welfare_other == 0.0

    result = attribution_ratios(effect.d_welfare_other, intervening_delta=0.0)
    assert result.current == result.completed_window == result.contribution_share == 0.0


def test_attribution_window_shift_mutation_changes_known_trajectory() -> None:
    learner = np.array([0.5, 0.0, 0.0])
    intervening = np.zeros(3)

    correct = ratio_series(learner, intervening, learner_shift=0)["completed_window"]
    shifted = ratio_series(learner, intervening, learner_shift=1)["completed_window"]
    magnitude = 0.5 / (0.5 + EPS_R)

    assert correct == pytest.approx([magnitude, 0.0, 0.0])
    assert shifted == pytest.approx([0.0, magnitude, 0.0])
    assert not np.array_equal(correct, shifted)


def test_old_vs_corrected_comparison_is_pinned() -> None:
    old = np.array([0.0, 10.0, 0.25, 1.0])
    corrected = np.array([0.0, 0.9, 0.25, 0.5])
    result = comparison_summary(old, corrected)

    assert result["fraction_changed"] == pytest.approx(0.5)
    assert result["mean_absolute_change"] == pytest.approx(2.4)
    assert result["max_absolute_change"] == pytest.approx(9.1)


def test_distribution_summary_counts_clip_mass_and_avoids_cross_episode_pairs() -> None:
    values = np.array([0.0, 0.5, 1.0, 2.0])
    episodes = np.array([0, 0, 1, 1])
    result = distribution_summary(values, episodes)

    assert result["n"] == 4
    assert result["mass_zero"] == pytest.approx(0.25)
    assert result["mass_between_zero_and_one"] == pytest.approx(0.25)
    assert result["mass_at_or_above_one"] == pytest.approx(0.5)
    assert result["clipping_frequency"] == pytest.approx(0.5)
    assert result["median_nonzero"] == pytest.approx(1.0)
    assert result["max"] == pytest.approx(2.0)
    assert result["unique_support"] == 4


@pytest.mark.parametrize(
    ("learner", "intervening"),
    [(float("nan"), 0.0), (0.0, float("inf"))],
)
def test_nonfinite_attribution_inputs_fail_closed(learner: float, intervening: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        attribution_ratios(learner, intervening)
