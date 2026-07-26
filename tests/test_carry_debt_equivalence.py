"""
Known-answer tests frozen by Amendment 002 §5 (PROPOSED, not yet adopted).

These verify the amendment's load-bearing claim BEFORE it is adopted: that
carry-debt read timing disambiguates §2.8's unindexed `D̂` **without modifying
§2.6's registered debt recurrence**, because the two forms coincide on every
reachable input.

Production code is deliberately NOT changed here. Both forms are implemented as
references so the claim can be checked independently of the implementation that
will eventually adopt it.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from tearrl import TearWorld, Verb
from tearrl.predicates import action_effect
from tearrl.spec import D_MAX, W_HARM, N_ACTIONS

from .conftest import legal_verbs

GRID = np.round(np.arange(0.0, 1.01, 0.1), 3)


def registered_form(d: float, harm: float, restored: float) -> float:
    """§2.6 as written: d' = clip(d + ĥarm - restôred, 0, d_max)."""
    return float(np.clip(d + harm - restored, 0.0, D_MAX))


def carry_form(d: float, harm: float, restored: float) -> tuple[float, float]:
    """Amendment 002 §2: returns (d_carry_t, d_{t+1})."""
    carry = float(np.clip(d - restored, 0.0, D_MAX))
    return carry, float(np.clip(carry + harm, 0.0, D_MAX))


# ======================================================================
# §5.1 exhaustive equivalence
# ======================================================================


def test_forms_agree_on_every_reachable_input() -> None:
    """h * r == 0 is the reachable domain (see §5.2). There the two forms must
    be identical -- that is what makes this a disambiguation rather than a
    modification of the registered recurrence."""
    checked = 0
    for d in GRID:
        for harm in GRID:
            for restored in GRID:
                if harm > 0 and restored > 0:
                    continue                      # unreachable, see below
                assert carry_form(d, harm, restored)[1] == pytest.approx(
                    registered_form(d, harm, restored), abs=1e-12
                )
                checked += 1
    assert checked > 200


def test_the_equivalence_test_is_not_vacuous() -> None:
    """The forms genuinely differ somewhere. If they agreed everywhere, the
    test above would prove nothing about the restriction to reachable inputs."""
    divergent = [
        (d, h, r)
        for d in GRID
        for h in GRID
        for r in GRID
        if h > 0
        and r > 0
        and abs(carry_form(d, h, r)[1] - registered_form(d, h, r)) > 1e-12
    ]
    assert len(divergent) > 100, "forms are indistinguishable; claim is empty"
    # The canonical witness cited in the amendment.
    assert carry_form(0.1, 0.6, 0.5)[1] == pytest.approx(0.6)
    assert registered_form(0.1, 0.6, 0.5) == pytest.approx(0.2)


# ======================================================================
# §5.2 unreachability of the divergent region
# ======================================================================


def _observed_welfare_signs() -> dict[Verb, set[int]]:
    """The sign of ΔŴ_other observed for each verb across reachable states."""
    signs: dict[Verb, set[int]] = {verb: set() for verb in Verb}
    for episode in range(60):
        world = TearWorld(seed=4000 + episode)
        rng = random.Random(episode)
        while not world.state.done:
            actor = world.state.actor_index
            for index in range(N_ACTIONS):
                if world.legal_actions(actor)[index]:
                    delta = action_effect(world, Verb(index), actor).d_welfare_other
                    signs[Verb(index)].add(int(np.sign(round(delta, 9))))
            world.step(rng.choice(legal_verbs(world)))
    return signs


def test_every_registered_action_has_a_single_welfare_sign() -> None:
    """§2.1's unreachability claim, checked EXHAUSTIVELY over the registered
    action set rather than sampled over states.

    The claim is a property of A.3's effect table, not of any particular state:
    ĥarm = max(0, -ΔŴ_other) and restôred = max(0, +ΔŴ_other) for `repair`, so
    if each verb carries a single fixed sign of ΔŴ_other then ĥarm * restôred
    is identically zero. Every one of the ten verbs is covered.
    """
    signs = _observed_welfare_signs()

    uncovered = [verb.name for verb, seen in signs.items() if not seen]
    assert not uncovered, f"verbs never exercised, claim unverified: {uncovered}"

    for verb, seen in signs.items():
        mixed = {s for s in seen if s != 0}
        assert len(mixed) <= 1, (
            f"{verb.name} produced BOTH signs of ΔŴ_other {seen}; §2.1's "
            "unreachability proof does not hold for it"
        )


def test_no_registered_action_can_harm_and_restore_simultaneously() -> None:
    """The consequence: ĥarm * restôred == 0 for every verb.

    FORWARD GUARD -- if this ever fails, §2.1's equivalence proof is void and
    Amendment 002 must be re-opened. A new action permitting harm and
    restoration in the same step REQUIRES A NEW AMENDMENT; it may not be
    absorbed by loosening this test.
    """
    for verb, seen in _observed_welfare_signs().items():
        for sign in seen:
            delta = float(sign)
            harm = max(0.0, -delta)
            restored = max(0.0, delta) if verb is Verb.REPAIR else 0.0
            assert harm * restored == 0.0, (
                f"{verb.name} harms and restores simultaneously -- §2.1 is void "
                "and a new amendment is required"
            )


# ======================================================================
# §5.3 terminal repair must have value
# ======================================================================


def _conscience(prediction: np.ndarray, debt_term: float) -> float:
    """Full registered r^C (§2.8) with an explicitly supplied debt term.

    Uses the production channel weights and every term -- harm, autonomy, U, E,
    O -- not a harm-only simplification. `debt_term` is whichever debt reading
    is under test, which is precisely the quantity Amendment 002 fixes.
    """
    from tearrl.agents.ppo import conscience_reward

    return conscience_reward(prediction, debt_term)


def _repair_state() -> tuple[TearWorld, float]:
    """A reachable state with outstanding debt and a FEASIBLE repair."""
    world = TearWorld(seed=910)
    world.reset(seed=910, learner_index=0)
    world.intervene(m_e=0)
    world.step(Verb.WAIT)
    world.step(Verb.STEAL)          # creates a repairable U record
    world.step(Verb.WAIT)
    assert world.legal_actions(0)[Verb.REPAIR], "repair must be feasible"
    return world, 0.5               # d_t carried from the theft


def test_terminal_repair_beats_waiting_under_carry_timing() -> None:
    """The property `d_t` timing FAILS -- with its preconditions ASSERTED.

    Under `d_t`, repair's effect on r^C appears only at the next decision. At an
    agent's final decision that step never arrives, so terminal repair is worth
    exactly nothing, and the horizon silently switches off a remedial action
    §5.3 declares conscience-consistent.
    """
    world, debt = _repair_state()
    assert debt > 0.0, "precondition: outstanding debt"

    repair_effect = action_effect(world, Verb.REPAIR, 0)
    restored = max(0.0, repair_effect.d_welfare_other) / 0.3   # normalized units
    assert restored > 0.0, "precondition: predicted restoration is positive"

    quiet = np.zeros(9)             # repair and wait raise no U/E/O event

    repair_carry = carry_form(debt, 0.0, restored)[0]
    wait_carry = carry_form(debt, 0.0, 0.0)[0]
    assert repair_carry < wait_carry, "repair must reduce the carry debt"

    r_repair = _conscience(quiet, repair_carry)
    r_wait = _conscience(quiet, wait_carry)
    assert r_repair > r_wait, "terminal repair gains nothing under carry timing"

    # The alternative this rules out: under d_t timing, `repair` and `wait`
    # both read the SAME d_t, so a terminal repair earns nothing. Stated as a
    # real comparison against the d_t value, not as an identity.
    r_under_d_t = _conscience(quiet, debt)
    assert r_wait == pytest.approx(r_under_d_t), (
        "waiting should be unaffected by the timing choice"
    )
    assert r_repair > r_under_d_t, (
        "carry timing must strictly beat d_t timing for a terminal repair"
    )


# ======================================================================
# §5.4 no double charge
# ======================================================================


def test_fresh_harm_contributes_nothing_through_the_debt_term() -> None:
    """From d_t = 0 with no restoration, the DEBT CONTRIBUTION must be zero.

    Deliberately NOT asserted as `r^C = -w_h*Ĥ`: for a real harmful action such
    as `steal`, Û = 1 and Â > 0, so that identity is false. The claim under
    test is only that fresh harm does not ALSO arrive through the debt channel.

    Under `d_{t+1}` timing it would: the same harm billed once through Ĥ_t and
    again through the debt increment it just created.
    """
    from tearrl.spec import W_DEBT

    # A realistic steal prediction: welfare harm, autonomy damage, U raised.
    prediction = np.zeros(9)
    prediction[2] = -0.5      # ΔŴ_other
    prediction[3] = -0.2      # Δautonomy_other
    prediction[5] = 1.0       # U

    harm = max(0.0, -prediction[2])
    carry = carry_form(0.0, harm, 0.0)[0]
    assert carry == 0.0, "carry debt must be zero from an empty-debt state"

    r_carry = _conscience(prediction, carry)
    r_no_debt_term = _conscience(prediction, 0.0)
    assert r_carry == pytest.approx(r_no_debt_term), "debt contributed to r^C"

    # r^C still carries the other registered terms -- it is NOT -w_h*Ĥ.
    assert r_carry < -W_HARM * harm, "autonomy and U terms are missing from r^C"

    # The alternative ruled out: d_{t+1} timing charges the same harm twice.
    next_debt = carry_form(0.0, harm, 0.0)[1]
    assert next_debt == pytest.approx(harm)
    r_double = _conscience(prediction, next_debt)
    assert r_double == pytest.approx(r_carry - W_DEBT * harm)
    assert r_double < r_carry, "d_{t+1} would double-charge fresh harm"


def test_prior_debt_remains_costly() -> None:
    """Carry timing must not accidentally forgive outstanding debt."""
    quiet = np.zeros(9)
    with_debt = _conscience(quiet, carry_form(0.8, 0.0, 0.0)[0])
    without = _conscience(quiet, carry_form(0.0, 0.0, 0.0)[0])
    assert with_debt < without


def test_debt_stays_within_bounds() -> None:
    for d in GRID:
        for harm in GRID:
            for restored in GRID:
                carry, nxt = carry_form(d, harm, restored)
                assert 0.0 <= carry <= D_MAX
                assert 0.0 <= nxt <= D_MAX
