"""PPO core, conscience reward and the C6 protected update (§2.8, B.2, B.5)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tearrl.agents.ppo import (
    RolloutBuffer,
    assign_flat_grad,
    flat_grad,
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


def test_gae_exercises_nonzero_values_and_last_value() -> None:
    advantages, returns = compute_gae(
        rewards=np.array([1.0, 2.0, 3.0]),
        values=np.array([0.4, -0.2, 0.7]),
        dones=np.array([False, False, False]),
        gamma=0.9,
        lam=0.8,
        last_value=1.3,
    )
    assert advantages == pytest.approx([4.256448, 5.3284, 3.47])
    assert returns == pytest.approx([4.656448, 5.1284, 4.17])


def test_gae_distinguishes_termination_from_rollout_truncation() -> None:
    rewards = np.array([1.0])
    values = np.array([0.4])
    terminal = compute_gae(
        rewards, values, np.array([True]), gamma=0.9, lam=0.8, last_value=9.0
    )
    truncated = compute_gae(
        rewards, values, np.array([False]), gamma=0.9, lam=0.8, last_value=9.0
    )
    assert terminal[0] == pytest.approx([0.6])
    assert terminal[1] == pytest.approx([1.0])
    assert truncated[0] == pytest.approx([8.7])
    assert truncated[1] == pytest.approx([9.1])


def test_gae_rejects_misaligned_or_batched_inputs() -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        compute_gae(np.zeros(1), np.zeros(2), np.zeros(2, dtype=bool))
    with pytest.raises(ValueError, match="one-dimensional"):
        compute_gae(np.zeros((1, 1)), np.zeros(1), np.zeros(1, dtype=bool))


def _complete_rollout_row() -> dict:
    return {
        "obs": np.zeros(1),
        "c": np.zeros(4),
        "q_other": np.zeros(4),
        "witness": np.zeros((10, 9)),
        "mask": np.ones(10, dtype=bool),
        "action": 0,
        "logprob": 0.0,
        "reward_task": 0.0,
        "reward_conscience": 0.0,
        "value_task": 0.0,
        "value_conscience": 0.0,
        "done": False,
    }


def test_rollout_buffer_rejects_partial_rows_atomically() -> None:
    buffer = RolloutBuffer()
    with pytest.raises(ValueError, match="incomplete rollout row"):
        buffer.add(action=3)
    assert {len(getattr(buffer, name)) for name in buffer.__slots__} == {0}

    row = _complete_rollout_row()
    row["typo"] = row.pop("done")
    with pytest.raises(ValueError, match="missing=.*done.*extra=.*typo"):
        buffer.add(**row)
    assert {len(getattr(buffer, name)) for name in buffer.__slots__} == {0}

    buffer.add(**_complete_rollout_row())
    assert len(buffer) == 1
    assert {len(getattr(buffer, name)) for name in buffer.__slots__} == {1}


@pytest.mark.parametrize(
    ("advantage", "ratio", "expected", "gradient"),
    [
        (+1.0, 0.5, +0.5, +0.5),
        (+1.0, 1.5, +1.2, 0.0),
        (-1.0, 0.5, -0.8, 0.0),
        (-1.0, 1.5, -1.5, -1.5),
    ],
)
def test_clipping_all_advantage_and_ratio_quadrants(
    advantage: float, ratio: float, expected: float, gradient: float
) -> None:
    logprob = torch.tensor([np.log(ratio)], requires_grad=True)
    objective = clipped_surrogate(
        logprob, torch.zeros(1), torch.tensor([advantage])
    )
    objective.backward()
    assert float(objective.detach()) == pytest.approx(expected, abs=1e-6)
    assert float(logprob.grad) == pytest.approx(gradient, abs=1e-6)


def test_clipped_surrogate_rejects_cross_sample_broadcasting() -> None:
    with pytest.raises(ValueError, match="broadcasting would mix transitions"):
        clipped_surrogate(torch.zeros(2, 1), torch.zeros(2), torch.ones(2))


def test_one_real_torch_adam_step_ascends_the_ppo_objective() -> None:
    """PyTorch Adam descends its loss, so an ascent objective needs a minus."""
    logits = torch.nn.Parameter(torch.zeros(2))
    optimizer = torch.optim.Adam([logits], lr=0.05)
    old_logprob = torch.log_softmax(logits.detach(), dim=0)[0].reshape(1)
    advantage = torch.ones(1)

    def objective() -> torch.Tensor:
        logprob = torch.log_softmax(logits, dim=0)[0].reshape(1)
        return clipped_surrogate(logprob, old_logprob, advantage)

    before = float(objective().detach())
    optimizer.zero_grad(set_to_none=True)
    loss = -objective()
    loss.backward()
    optimizer.step()
    after = float(objective().detach())

    assert after > before
    assert float(torch.log_softmax(logits.detach(), dim=0)[0]) > float(old_logprob)


def test_pcgrad_rejects_empty_vectors() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        pcgrad(torch.empty(0), torch.empty(0))


def test_flat_gradient_round_trip_and_size_validation_are_atomic() -> None:
    first = torch.nn.Parameter(torch.zeros(2))
    second = torch.nn.Parameter(torch.zeros(1))
    first.grad = torch.tensor([1.0, 2.0])
    second.grad = torch.tensor([3.0])
    assert flat_grad([first, second]).tolist() == [1.0, 2.0, 3.0]

    assign_flat_grad([first, second], torch.tensor([4.0, 5.0, 6.0]))
    assert first.grad.tolist() == [4.0, 5.0]
    assert second.grad.tolist() == [6.0]

    before = (first.grad.clone(), second.grad.clone())
    for malformed in (torch.tensor([7.0, 8.0]), torch.tensor([7.0, 8.0, 9.0, 10.0])):
        with pytest.raises(ValueError, match="flat gradient"):
            assign_flat_grad([first, second], malformed)
        assert torch.equal(first.grad, before[0])
        assert torch.equal(second.grad, before[1])


@pytest.mark.parametrize("scale", [1e-30, 1e-20, 1e20, 1e30])
def test_pcgrad_projection_and_cosine_are_scale_stable(scale: float) -> None:
    task = torch.tensor([-1.0, 1.0])
    conscience = torch.tensor([scale, 0.0])
    projected, diagnostics = pcgrad(task, conscience)
    assert projected == pytest.approx(torch.tensor([0.0, 1.0]), abs=1e-5)
    assert diagnostics.cosine == pytest.approx(-1 / np.sqrt(2), abs=1e-6)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_pcgrad_rejects_nonfinite_gradients(bad: float) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        pcgrad(torch.tensor([1.0, 0.0]), torch.tensor([bad, 0.0]))


def test_conscience_reward_rejects_invalid_instrument_values() -> None:
    prediction = np.zeros(9)
    prediction[2] = np.nan
    with pytest.raises(ValueError, match="finite 9-vector"):
        conscience_reward(prediction, debt=0.0)
    with pytest.raises(ValueError, match="debt must be finite"):
        conscience_reward(np.zeros(9), debt=-0.1)
