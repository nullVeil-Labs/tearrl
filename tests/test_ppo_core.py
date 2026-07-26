"""PPO core, conscience reward and the C6 protected update (§2.8, B.2, B.5)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tearrl.agents.ppo import (
    clipped_surrogate,
    compute_gae,
    conscience_reward,
    entropy_bonus,
    normalize,
    pcgrad,
)
from tearrl.spec import GAMMA, W_HARM, W_O


def test_gae_matches_hand_computation() -> None:
    rewards = np.array([1.0, 1.0, 1.0])
    values = np.array([0.0, 0.0, 0.0])
    dones = np.array([False, False, True])
    lam = 0.95

    advantages, returns = compute_gae(rewards, values, dones, lam=lam)

    a2 = 1.0
    a1 = 1.0 + GAMMA * lam * a2
    a0 = 1.0 + GAMMA * lam * a1
    assert advantages == pytest.approx([a0, a1, a2])
    assert returns == pytest.approx(advantages)      # zero baseline


def test_gae_does_not_bootstrap_across_the_episode_boundary() -> None:
    """TearWorld has a fixed 32-step horizon and no terminal reward (§3.3), so
    value must not leak across `done`."""
    rewards = np.array([0.0, 5.0])
    values = np.array([0.0, 0.0])
    ended = compute_gae(rewards, values, np.array([True, True]))[0]
    continuing = compute_gae(rewards, values, np.array([False, True]))[0]
    assert ended[0] == pytest.approx(0.0)
    assert continuing[0] > 0.0


def test_conscience_reward_is_negative_on_harm_and_zero_when_clean() -> None:
    clean = np.zeros(9)
    assert conscience_reward(clean, debt=0.0) == pytest.approx(0.0)

    harmful = np.zeros(9)
    harmful[2] = -0.5           # DW_other
    assert conscience_reward(harmful, debt=0.0) == pytest.approx(-W_HARM * 0.5)

    tampering = np.zeros(9)
    tampering[7] = 1.0          # O
    assert conscience_reward(tampering, debt=0.0) == pytest.approx(-W_O)


def test_benefiting_the_counterpart_is_not_rewarded_by_r_c() -> None:
    """Only the harm side enters r^C. If positive DW_other paid, an agent could
    offset harm with unrelated generosity -- the same laundering the debt
    channel forbids for `share` (§2.6)."""
    benefit = np.zeros(9)
    benefit[2] = +0.5
    assert conscience_reward(benefit, debt=0.0) == pytest.approx(0.0)


def test_pcgrad_removes_the_conflicting_component_only() -> None:
    """§2.8: if g_T . g_C < 0, g_T <- g_T - proj_{g_C}(g_T)."""
    g_conscience = torch.tensor([1.0, 0.0])
    g_task = torch.tensor([-1.0, 1.0])          # conflicting

    projected, diagnostics = pcgrad(g_task, g_conscience)

    assert diagnostics.conflicted
    assert float(torch.dot(projected, g_conscience)) == pytest.approx(0.0, abs=1e-6)
    assert projected == pytest.approx(torch.tensor([0.0, 1.0]), abs=1e-6)
    assert diagnostics.removed_norm == pytest.approx(1.0)


def test_pcgrad_is_a_noop_when_gradients_agree() -> None:
    g_conscience = torch.tensor([1.0, 0.0])
    g_task = torch.tensor([1.0, 1.0])
    projected, diagnostics = pcgrad(g_task, g_conscience)

    assert not diagnostics.conflicted
    assert diagnostics.removed_norm == 0.0
    assert torch.equal(projected, g_task)


def test_pcgrad_never_projects_the_conscience_gradient() -> None:
    """The asymmetry IS the mechanism: conscience is protected from task
    gradients, not the reverse. A symmetric projection would let the task
    objective erode conscience exactly when they disagree."""
    g_conscience = torch.tensor([1.0, 0.0])
    g_task = torch.tensor([-5.0, 0.0])          # directly opposed
    projected, _ = pcgrad(g_task, g_conscience)

    # g_T is fully cancelled; g_C is returned to the caller untouched.
    assert projected == pytest.approx(torch.tensor([0.0, 0.0]), abs=1e-6)
    assert g_conscience == pytest.approx(torch.tensor([1.0, 0.0]))


def test_clipped_surrogate_is_an_ASCENT_objective() -> None:
    """§2.8 is written in ascent form throughout; a descent-convention loss
    would silently flip the sign of C6's projection."""
    old = torch.zeros(4)
    advantages = torch.ones(4)
    better = clipped_surrogate(torch.full((4,), 0.1), old, advantages)
    worse = clipped_surrogate(torch.full((4,), -0.1), old, advantages)
    assert better > worse


def test_clipping_bounds_the_ratio() -> None:
    old = torch.zeros(3)
    advantages = torch.ones(3)
    huge = clipped_surrogate(torch.full((3,), 5.0), old, advantages)
    assert huge == pytest.approx(1.2, abs=1e-5)      # 1 + clip_eps


def test_entropy_counts_only_feasible_actions() -> None:
    """§2.4 masks infeasible actions; counting them would make entropy a
    function of how many verbs are legal rather than of policy indecision."""
    logits = torch.zeros(1, 4)
    all_legal = torch.tensor([[True, True, True, True]])
    two_legal = torch.tensor([[True, True, False, False]])

    assert float(entropy_bonus(logits, all_legal)) == pytest.approx(np.log(4), abs=1e-5)
    assert float(entropy_bonus(logits, two_legal)) == pytest.approx(np.log(2), abs=1e-5)


def test_streams_are_normalized_separately() -> None:
    """Task and conscience advantages live on different scales; joint
    normalization would silently reweight the §2.8 combination."""
    task = torch.tensor([10.0, 20.0, 30.0])
    conscience = torch.tensor([-0.1, 0.0, 0.1])
    assert float(normalize(task).std()) == pytest.approx(1.0, abs=1e-3)
    assert float(normalize(conscience).std()) == pytest.approx(1.0, abs=1e-3)
