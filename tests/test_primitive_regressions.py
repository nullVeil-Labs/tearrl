"""
Permanent regressions for defects found by adversarial audit of the PPO core.

Every test here corresponds to a defect that was REPRODUCED before being fixed.
Each was invisible to the tests that existed at the time, and the recurring
cause is the same: a forward-only check passing while the backward pass, or a
non-finite value, was already broken.

The lesson encoded here: masking by MULTIPLICATION does not sanitize.
`0 * NaN = NaN` and `0 * inf = NaN`, so a gate built from `x * mask` lets
non-finite garbage through while looking correct on ordinary inputs. Use
`masked_fill`/`torch.where` to REPLACE, and neutralize BEFORE any arithmetic.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tearrl.agents.networks import (
    C3IMPolicy,
    EmptyActionMask,
    PolicyNet,
    Selector,
    c3im_equalized_width,
)
from tearrl.agents.ppo import entropy_bonus, normalize, pcgrad


# ======================================================================
# 1. entropy: forward AND backward finiteness
# ======================================================================


@pytest.mark.parametrize(
    "mask_row",
    [
        [True, True, False, False],
        [True, False, False, False],      # exactly one feasible action
        [True, True, True, True],
        [False, True, False, True],
    ],
)
def test_entropy_gradient_is_finite_under_masking(mask_row) -> None:
    """Reproduced defect: gradient was [nan, nan, 0.0, 0.0] with a FINITE
    forward value of 0.6628 — so a forward-only test passed while the first
    actor update would have poisoned the network.

    Cause: masked entries are -inf, so `probs * log_probs` forms 0 * -inf = NaN.
    Hiding it behind torch.where AFTER the multiply fixes only the forward pass;
    autograd still differentiates the product and log_softmax couples every
    entry, so the NaN reaches the FEASIBLE logits.
    """
    logits = torch.tensor([[0.3, -0.2, 7.0, -9.0]], requires_grad=True)
    mask = torch.tensor([mask_row])

    value = entropy_bonus(logits, mask)
    value.backward()

    assert torch.isfinite(value), "entropy forward is non-finite"
    assert torch.isfinite(logits.grad).all(), f"entropy gradient: {logits.grad}"

    masked_positions = ~mask[0]
    assert (logits.grad[0][masked_positions] == 0).all(), (
        "masked logits received gradient"
    )


def test_entropy_of_a_single_feasible_action_is_zero() -> None:
    logits = torch.tensor([[5.0, 1.0, 2.0]], requires_grad=True)
    mask = torch.tensor([[True, False, False]])
    value = entropy_bonus(logits, mask)
    value.backward()
    assert float(value) == pytest.approx(0.0, abs=1e-6)
    assert torch.isfinite(logits.grad).all()


def test_entropy_matches_the_uniform_bound_on_feasible_actions() -> None:
    logits = torch.zeros(1, 4)
    assert float(entropy_bonus(logits, torch.tensor([[True] * 4]))) == pytest.approx(
        np.log(4), abs=1e-5
    )
    assert float(
        entropy_bonus(logits, torch.tensor([[True, True, False, False]]))
    ) == pytest.approx(np.log(2), abs=1e-5)


# ======================================================================
# 2. C3-IM invariance to infeasible witness rows
# ======================================================================


def _c3im_case():
    torch.manual_seed(0)
    net = C3IMPolicy(c3im_equalized_width()).eval()
    obs = torch.randn(1, 434)
    q_other = torch.randn(1, 4)
    mask = torch.tensor(
        [[True, True, False, False, False, False, False, False, False, True]]
    )
    return net, obs, q_other, mask


@pytest.mark.parametrize("poison", [0.0, 1.0, 99.0, -1e6, 1e12])
def test_c3im_ignores_finite_perturbation_of_infeasible_rows(poison) -> None:
    """Reproduced defect: garbage in rows the agent cannot select moved the
    FEASIBLE action probabilities by 6.3%.

    §2.4 leaves infeasible actions untrained, so those nine channels carry
    undefined values. C3-IM is the frozen headline comparator for G2 — its
    behaviour must not depend on them.
    """
    net, obs, q_other, mask = _c3im_case()
    baseline = torch.zeros(1, 10, 9)
    perturbed = baseline.clone()
    perturbed[0, ~mask[0], :] = poison

    with torch.no_grad():
        a = torch.softmax(net(obs, baseline, q_other, mask), dim=-1)
        b = torch.softmax(net(obs, perturbed, q_other, mask), dim=-1)

    assert torch.allclose(a, b, atol=1e-9), f"feasible probs moved by {(a - b).abs().max()}"


@pytest.mark.parametrize("poison", [float("nan"), float("inf"), float("-inf")])
def test_c3im_survives_non_finite_infeasible_rows(poison) -> None:
    """Why the gate must REPLACE rather than multiply.

    `witness_out * mask` computes 0 * NaN = NaN, so a non-finite value in an
    untrained row survives the gate and poisons every feasible logit.
    `masked_fill` replaces it outright.
    """
    net, obs, q_other, mask = _c3im_case()
    baseline = torch.zeros(1, 10, 9)
    perturbed = baseline.clone()
    perturbed[0, ~mask[0], :] = poison

    with torch.no_grad():
        a = torch.softmax(net(obs, baseline, q_other, mask), dim=-1)
        b = torch.softmax(net(obs, perturbed, q_other, mask), dim=-1)

    assert torch.isfinite(b).all(), "non-finite infeasible row reached the output"
    assert torch.allclose(a, b, atol=1e-9)


def test_c3im_still_responds_to_feasible_rows() -> None:
    """The gate must not be so aggressive that it deafens the policy."""
    net, obs, q_other, mask = _c3im_case()
    baseline = torch.zeros(1, 10, 9)
    changed = baseline.clone()
    changed[0, 0, :] = 1.0                      # a FEASIBLE row

    with torch.no_grad():
        a = torch.softmax(net(obs, baseline, q_other, mask), dim=-1)
        b = torch.softmax(net(obs, changed, q_other, mask), dim=-1)

    assert not torch.allclose(a, b, atol=1e-6), "C3-IM ignores feasible witness rows"


# ======================================================================
# 3. PCGrad scale invariance
# ======================================================================


@pytest.mark.parametrize("scale", [1e-8, 1e-6, 1.0, 1e6, 1e8])
def test_pcgrad_is_invariant_to_conscience_gradient_scale(scale) -> None:
    """Reproduced defect: an ABSOLUTE 1e-12 cutoff on ||g_C||^2 meant
    ||g_C||=0.9e-6 left g_T untouched while 1.1e-6 removed almost all of it —
    a harmless rescaling switching the mechanism on and off.

    The projection is mathematically scale-invariant; only the guard was not.
    """
    g_task = torch.tensor([-1.0, 1.0])
    projected, _ = pcgrad(g_task, torch.tensor([scale, 0.0]))
    assert projected == pytest.approx(torch.tensor([0.0, 1.0]), abs=1e-6)


def test_pcgrad_handles_a_zero_conscience_gradient() -> None:
    g_task = torch.tensor([-1.0, 1.0])
    projected, diagnostics = pcgrad(g_task, torch.zeros(2))
    assert torch.equal(projected, g_task)
    assert diagnostics.removed_norm == 0.0


def test_pcgrad_result_is_orthogonal_to_the_conscience_gradient() -> None:
    torch.manual_seed(3)
    for _ in range(20):
        g_conscience = torch.randn(8)
        g_task = torch.randn(8)
        projected, diagnostics = pcgrad(g_task, g_conscience)
        if diagnostics.conflicted:
            assert float(torch.dot(projected, g_conscience)) == pytest.approx(0.0, abs=1e-5)
        else:
            assert torch.equal(projected, g_task)


# ======================================================================
# 4. singleton normalization
# ======================================================================


def test_singleton_normalization_returns_zero_not_nan() -> None:
    """`std()` of one element is NaN under Bessel correction, which would
    silently poison an entire update."""
    assert normalize(torch.tensor([5.0])).tolist() == [0.0]
    assert torch.isfinite(normalize(torch.tensor([5.0]))).all()


def test_normalization_of_a_real_batch_is_standardized() -> None:
    values = torch.tensor([10.0, 20.0, 30.0, 40.0])
    out = normalize(values)
    assert float(out.mean()) == pytest.approx(0.0, abs=1e-6)
    assert float(out.std()) == pytest.approx(1.0, abs=1e-3)


# ======================================================================
# 5. all-false masks must fail loudly at the tensor boundary
# ======================================================================


def test_empty_mask_raises_in_every_policy_path() -> None:
    """A.3 makes `wait` unconditional, so the environment never emits an empty
    mask — but a probe, batching bug or corrupted restore can. Silently it
    yields all -inf logits, then NaN from softmax, reaching the optimizer
    disguised as a learning-rate problem."""
    empty = torch.tensor([[False, False, False, False]])

    with pytest.raises(EmptyActionMask):
        entropy_bonus(torch.zeros(1, 4), empty)

    policy = PolicyNet(hidden=8)
    with pytest.raises(EmptyActionMask):
        policy(torch.zeros(1, 434), torch.zeros(1, 4), torch.zeros(1, 10, dtype=torch.bool))

    selector = Selector(hidden=8)
    with pytest.raises(EmptyActionMask):
        selector(
            torch.zeros(1, 10, 9),
            torch.zeros(1, 4),
            torch.zeros(1, 4),
            torch.zeros(1, 10, dtype=torch.bool),
        )

    net, obs, q_other, _ = _c3im_case()
    with pytest.raises(EmptyActionMask):
        net(obs, torch.zeros(1, 10, 9), q_other, torch.zeros(1, 10, dtype=torch.bool))


def test_empty_mask_is_detected_in_any_row_of_a_batch() -> None:
    mask = torch.tensor([[True, True], [False, False], [True, False]])
    with pytest.raises(EmptyActionMask):
        entropy_bonus(torch.zeros(3, 2), mask)


def test_valid_masks_are_not_rejected() -> None:
    mask = torch.tensor([[True, False], [False, True]])
    assert torch.isfinite(entropy_bonus(torch.zeros(2, 2), mask))


@pytest.mark.parametrize("poison", [float("nan"), float("inf"), -float("inf")])
def test_selector_sanitizes_infeasible_rows_before_backward(poison: float) -> None:
    torch.manual_seed(411)
    net = Selector(hidden=8)
    witness = torch.randn(1, 10, 9)
    witness[0, 7] = poison
    witness.requires_grad_()
    c = torch.randn(1, 4)
    q_other = torch.randn(1, 4)
    mask = torch.ones(1, 10, dtype=torch.bool)
    mask[0, 7] = False

    baseline = witness.detach().clone()
    baseline[0, 7] = 0.0
    with torch.no_grad():
        expected = torch.softmax(net(baseline, c, q_other, mask), dim=-1)

    logits = net(witness, c, q_other, mask)
    actual = torch.softmax(logits, dim=-1)
    assert torch.allclose(actual, expected, atol=1e-9)
    (-torch.log_softmax(logits, dim=-1)[0, 0]).backward()

    assert all(torch.isfinite(parameter.grad).all() for parameter in net.parameters())
    assert torch.isfinite(witness.grad[mask]).all()
    assert (witness.grad[~mask] == 0).all()


def test_broadcastable_action_masks_are_rejected_in_every_policy_path() -> None:
    bad = torch.ones(2, 1, dtype=torch.bool)
    obs = torch.zeros(2, 434)
    q_other = torch.zeros(2, 4)
    witness = torch.zeros(2, 10, 9)

    with pytest.raises(ValueError, match="broadcasting a mask"):
        PolicyNet(hidden=8)(obs, q_other, bad)
    with pytest.raises(ValueError, match="broadcasting a mask"):
        Selector(hidden=8)(witness, torch.zeros(2, 4), q_other, bad)
    with pytest.raises(ValueError, match="broadcasting a mask"):
        C3IMPolicy(hidden=8)(obs, witness, q_other, bad)
    with pytest.raises(ValueError, match="broadcasting a mask"):
        entropy_bonus(torch.zeros(2, 10), bad)


def test_action_masks_must_be_boolean() -> None:
    with pytest.raises(TypeError, match="must be bool"):
        entropy_bonus(torch.zeros(1, 2), torch.ones(1, 2, dtype=torch.int64))


def test_constant_float16_normalization_stays_finite() -> None:
    out = normalize(torch.tensor([5.0, 5.0], dtype=torch.float16))
    assert out.tolist() == [0.0, 0.0]
    assert torch.isfinite(out).all()
