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

from tearrl import TearWorld, Verb
from tearrl.spec import (
    DETECTION_TRUST_LOSS_UNITS,
    REPORT_TRUST_RESTORE_CAP_UNITS,
    Q_DIM,
)
from tearrl.state import OBS_DIM, sample_unit_vector


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


@pytest.mark.skip(
    reason="requires the B.1 C3-IM policy module (build order step 6); the "
    "base-observation half of the assertion is covered above"
)
def test_b6_c3im_q_other_appears_exactly_once() -> None:
    """Full B.6 #3: assemble the C3-IM policy input and assert q_other occupies
    exactly one contiguous Q_DIM block."""
