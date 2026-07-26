"""
Appendix B.6 -- frozen unit-test assertions.

    repair:  illicit_task_credit_after == max(0, illicit_task_credit_before - 1)
    report:  trust_restored <= trust actually lost to that record
    C3-IM:   q_other appears exactly once in the policy input vector

B.6 says these block CI on failure. They are direct implementations of frozen
protocol text -- if one fails, the implementation has drifted from the
preregistration, not the other way round.
"""

from __future__ import annotations

import pytest
import torch

from tearrl import TearWorld, Verb
from tearrl.agents.networks import C3IM_INPUT_DIM, C3IMPolicy
from tearrl.spec import (
    DETECTION_TRUST_LOSS_UNITS,
    REPORT_TRUST_RESTORE_CAP_UNITS,
    N_ACTIONS,
    Q_DIM,
)
from tearrl.state import OBS_DIM, sample_unit_vector
from tearrl.witness.targets import WITNESS_DIM


# ======================================================================
# B.6 #1 -- repair
# ======================================================================


@pytest.mark.parametrize("k", [1, 2, 3])
def test_b6_repair_forfeits_exactly_one_illicit_credit(k: int) -> None:
    """illicit_task_credit_after == max(0, illicit_task_credit_before - 1)."""
    world = TearWorld(seed=100 + k)
    world.reset(seed=100 + k, learner_index=0)
    world.intervene(m_e=0)
    world.state.agents[1].resources = k

    world.step(Verb.WAIT)
    world.step(Verb.STEAL)
    record = world.state.agents[0].ledger[0]
    assert record.magnitude == k and record.illicit_task_credit == k

    while record.repairable:
        before = record.illicit_task_credit
        world.step(Verb.WAIT)
        world.step(Verb.REPAIR)
        assert record.illicit_task_credit == max(0, before - 1)

    assert record.illicit_task_credit == 0
    assert record.repaired_units == k


# ======================================================================
# B.6 #2 -- report
# ======================================================================


def test_b6_report_never_restores_more_than_was_lost() -> None:
    """trust_restored <= trust actually lost to that record.

    A detected record costs 0.3 (3 units); the report cap is 0.2 (2 units),
    so restoration is strictly bounded below the loss and can never exceed it.
    """
    world = TearWorld(seed=201)
    world.reset(seed=201, learner_index=0)
    world.intervene(m_e=1)

    world.step(Verb.WAIT)
    world.step(Verb.STEAL)
    trust_before_detection = world.state.agents[1].trust_units

    world.step(Verb.WAIT)
    world.step(Verb.WAIT)          # record ages to 1 and is detected here
    record = world.state.agents[0].ledger[0]
    assert record.detected == 1
    assert record.pre_detection_trust == trust_before_detection

    trust_after_detection = world.state.agents[1].trust_units
    lost = trust_before_detection - trust_after_detection
    assert lost == DETECTION_TRUST_LOSS_UNITS

    world.step(Verb.WAIT)
    world.step(Verb.REPORT)
    restored = world.state.agents[1].trust_units - trust_after_detection

    assert restored <= lost
    assert restored <= REPORT_TRUST_RESTORE_CAP_UNITS


def test_b6_report_never_mints_trust_above_pre_violation_level() -> None:
    """v0.4.2 correction 2: intervening `share` recovery must not let a later
    report push trust ABOVE its pre-violation level.

    This is the case that a naive `Delta trust = 0.2 * detected` rule gets
    wrong, which is why the rule reads the stored baseline instead.
    """
    world = TearWorld(seed=202)
    world.reset(seed=202, learner_index=0)
    world.intervene(m_e=1)

    world.step(Verb.WAIT)
    world.step(Verb.STEAL)
    pre_violation_trust = world.state.agents[1].trust_units

    world.step(Verb.WAIT)
    world.step(Verb.WAIT)                      # detected: -0.3
    record = world.state.agents[0].ledger[0]
    assert record.detected == 1

    # Recover trust by sharing before confessing.
    for _ in range(3):
        world.step(Verb.WAIT)
        if world.legal_actions()[Verb.SHARE]:
            world.step(Verb.SHARE)
        else:
            world.step(Verb.WAIT)

    world.step(Verb.WAIT)
    if world.legal_actions()[Verb.REPORT]:
        world.step(Verb.REPORT)

    assert world.state.agents[1].trust_units <= pre_violation_trust, (
        "report minted trust above the pre-violation level after an "
        "intervening share recovery"
    )


def test_b6_undetected_report_restores_nothing() -> None:
    """The confessional arsonist stays sealed: an undetected record carries
    the sentinel pre_detection_trust = 0, so max(0, 0 - current) == 0."""
    world = TearWorld(seed=203)
    world.reset(seed=203, learner_index=0)
    world.intervene(m_e=0)

    world.step(Verb.WAIT)
    world.step(Verb.STEAL)
    before = world.state.agents[1].trust_units

    world.step(Verb.WAIT)
    world.step(Verb.REPORT)

    assert world.state.agents[0].ledger[0].detected == 0
    assert world.state.agents[1].trust_units == before


# ======================================================================
# B.6 #3 -- q_other appears exactly once
# ======================================================================


def test_b6_base_observation_contains_no_identity() -> None:
    """o^F == o^pi == o \\ {q_self, q_other}  (§2.1, v0.4.2 correction 1).

    This is the load-bearing half of B.6's 'exactly once' assertion. If the
    base observation carried identity, q_other would enter twice in C3-IM
    (once here, once via the appended q-input) and the information-matching
    with C6 would be broken. Tested behaviourally: two states differing ONLY
    in q must produce bit-identical observations.
    """
    world = TearWorld(seed=301)
    world.step(Verb.WORK)

    baseline = world.observe(0)
    assert len(baseline) == OBS_DIM

    import random as _random

    rng = _random.Random(999)
    for _ in range(50):
        world.state.agents[0].q = sample_unit_vector(rng)
        world.state.agents[1].q = sample_unit_vector(rng)
        assert world.observe(0) == baseline, "identity leaked into o^F/o^pi"


def test_b6_identity_observation_is_opt_in_and_correctly_sized() -> None:
    """The diagnostic-only identity-bearing observation appends exactly
    2 * Q_DIM values and is never the default."""
    world = TearWorld(seed=302)
    plain = world.observe(0)
    with_identity = world.observe(0, include_identity=True)
    assert len(with_identity) == len(plain) + 2 * Q_DIM
    assert with_identity[: len(plain)] == plain


def _capture_net_input(policy: C3IMPolicy, **forward_kwargs) -> torch.Tensor:
    """Return the vector `policy.net` actually received, for one forward pass.

    Captured by hook, NEVER reassembled here. A test that built its own
    `cat([obs, flat, q_other])` and compared would be checking the production
    line against a copy of itself: it would pass whatever `forward` did with
    the real inputs, which is precisely the assertion B.6 wants made.
    """
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, args):
        captured["x"] = args[0].detach().clone()

    handle = policy.net.register_forward_pre_hook(hook)
    try:
        policy(**forward_kwargs)
    finally:
        handle.remove()
    return captured["x"]


def _contiguous_blocks(indices: list[int]) -> list[tuple[int, int]]:
    """Group sorted indices into (start, length) runs of consecutive integers."""
    blocks: list[tuple[int, int]] = []
    for i in sorted(indices):
        if blocks and i == blocks[-1][0] + blocks[-1][1]:
            start, length = blocks[-1]
            blocks[-1] = (start, length + 1)
        else:
            blocks.append((i, 1))
    return blocks


def _q_sensitive_blocks(assemble, q_a: torch.Tensor, q_b: torch.Tensor):
    """Which coordinate runs of the assembled vector respond to q_other?

    `assemble` maps a q_other tensor to the input vector actually fed to the
    network. Coordinates that differ between two q values are exactly the
    coordinates q_other reaches; everything else is held fixed by the caller.
    """
    va = assemble(q_a).flatten()
    vb = assemble(q_b).flatten()
    assert va.shape == vb.shape
    changed = (va != vb).nonzero().flatten().tolist()
    return _contiguous_blocks(changed)


def test_b6_c3im_q_other_appears_exactly_once() -> None:
    """Full B.6 #3: q_other occupies exactly one contiguous Q_DIM block.

    NOT redundant with `assert_identity_enters_once`. That helper checks
    `net[0].in_features` against a declared width, and width is invariant to
    what actually lands in the slots -- an assembly that wrote q_other twice
    and dropped two other coordinates would satisfy it. This asserts the
    behaviour: perturb q_other, hold every other input fixed, and see which
    coordinates of the REAL assembled vector move.

    The base observation is drawn from the live environment rather than
    synthesised, so a q leak through `observe` would surface here as a second
    block (cf. test_b6_base_observation_contains_no_identity, which pins that
    half directly).
    """
    world = TearWorld(seed=303)
    world.step(Verb.WORK)
    obs = torch.tensor([world.observe(0)], dtype=torch.float32)
    assert obs.shape == (1, OBS_DIM)

    torch.manual_seed(303)
    policy = C3IMPolicy(hidden=16)
    witness_out = torch.randn(1, N_ACTIONS, WITNESS_DIM)
    mask = torch.ones(1, N_ACTIONS, dtype=torch.bool)

    # Two identities differing in EVERY coordinate -- a coordinate that happened
    # to agree would read as absence and silently weaken the test.
    q_a = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
    q_b = torch.tensor([[-0.1, 0.2, -0.3, 0.4]])
    assert bool((q_a != q_b).all()), "q_a and q_b must differ in all Q_DIM coords"

    def assemble(q: torch.Tensor) -> torch.Tensor:
        return _capture_net_input(
            policy, obs=obs, witness_out=witness_out, q_other=q, mask=mask
        )

    blocks = _q_sensitive_blocks(assemble, q_a, q_b)

    assert len(blocks) == 1, (
        f"q_other reaches {len(blocks)} disjoint regions of the policy input, "
        f"expected exactly 1 (blocks: {blocks})"
    )
    start, length = blocks[0]
    assert length == Q_DIM, f"q block is {length} wide, expected {Q_DIM}"
    assert start == C3IM_INPUT_DIM - Q_DIM, (
        f"q block starts at {start}, expected the tail slot "
        f"{C3IM_INPUT_DIM - Q_DIM}"
    )
    # The width contract and the behavioural contract must agree.
    assert assemble(q_a).shape[-1] == C3IM_INPUT_DIM


def test_b6_q_occurrence_detector_discriminates() -> None:
    """Mutation test for the guard above.

    A detector that cannot fail proves nothing. These are deliberately wrong
    assemblies of the same inputs; the detector must report a different block
    count for each. If this test passes while the real policy reports one
    block, the one-block result carries information.
    """
    obs = torch.zeros(1, OBS_DIM)
    flat = torch.zeros(1, WITNESS_DIM * N_ACTIONS)
    q_a = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
    q_b = torch.tensor([[-0.1, 0.2, -0.3, 0.4]])

    def correct(q):
        return torch.cat([obs, flat, q], dim=-1)

    def duplicated(q):
        # q written twice, total width preserved by trimming the witness block
        return torch.cat([obs, flat[:, : -2 * Q_DIM], q, flat[:, :Q_DIM], q], dim=-1)

    def absent(q):
        return torch.cat([obs, flat, torch.zeros_like(q)], dim=-1)

    assert len(_q_sensitive_blocks(correct, q_a, q_b)) == 1
    assert len(_q_sensitive_blocks(duplicated, q_a, q_b)) == 2, (
        "detector failed to see a duplicated q_other -- the guard above is "
        "not actually testing occurrence count"
    )
    assert len(_q_sensitive_blocks(absent, q_a, q_b)) == 0, (
        "detector failed to see an absent q_other"
    )
    # And the duplicated assembly has the SAME total width as the correct one,
    # which is why the width check alone cannot catch it.
    assert correct(q_a).shape == duplicated(q_a).shape
