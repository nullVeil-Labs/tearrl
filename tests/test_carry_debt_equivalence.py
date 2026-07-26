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


def test_no_action_produces_harm_and_restoration_together() -> None:
    """§2.6 defines ĥarm = max(0, -ΔŴ_other) and restôred = max(0, +ΔŴ_other)
    for `repair`. A single action yields ONE value of ΔŴ_other, hence one
    sign, hence ĥarm * restôred = 0 always.

    Verified over reachable states with the real effect table rather than
    argued from the definitions alone.
    """
    checked = 0
    for episode in range(60):
        world = TearWorld(seed=4000 + episode)
        rng = random.Random(episode)
        while not world.state.done:
            actor = world.state.actor_index
            for index in range(N_ACTIONS):
                if not world.legal_actions(actor)[index]:
                    continue
                delta = action_effect(world, Verb(index), actor).d_welfare_other
                harm = max(0.0, -delta)
                restored = max(0.0, delta) if Verb(index) is Verb.REPAIR else 0.0
                assert harm * restored == 0.0, (
                    f"{Verb(index).name} produced harm {harm} and restored "
                    f"{restored} in the same action"
                )
                checked += 1
            world.step(rng.choice(legal_verbs(world)))
    assert checked > 3000


# ======================================================================
# §5.3 terminal repair must have value
# ======================================================================


def _conscience(carry: float, harm: float = 0.0) -> float:
    """r^C with only the harm and debt terms active (others zero)."""
    from tearrl.spec import W_DEBT

    return -(W_HARM * harm + W_DEBT * carry)


def test_terminal_repair_beats_waiting_under_carry_timing() -> None:
    """The property `d_t` timing FAILS.

    Under `d_t`, repair's effect on r^C appears only at the next decision. At
    an agent's final decision that step never arrives, so terminal repair is
    worth exactly nothing -- the horizon silently switches off a remedial
    action §5.3 declares conscience-consistent.
    """
    debt, restored = 0.5, 0.2

    repair_carry = carry_form(debt, 0.0, restored)[0]
    wait_carry = carry_form(debt, 0.0, 0.0)[0]
    assert _conscience(repair_carry) > _conscience(wait_carry)

    # Under d_t timing both read the SAME debt, so repair gains nothing.
    assert _conscience(debt) == _conscience(debt)


# ======================================================================
# §5.4 no double charge
# ======================================================================


def test_fresh_harm_is_charged_exactly_once() -> None:
    """From an empty-debt state a harmful action must cost exactly w_h * Ĥ.

    Under `d_{t+1}` timing it would cost w_h*Ĥ + w_d*Ĥ -- the same harm billed
    through both the harm term and the debt increment it just created.
    """
    harm = 0.5
    carry = carry_form(0.0, harm, 0.0)[0]
    assert carry == 0.0
    assert _conscience(carry, harm) == pytest.approx(-W_HARM * harm)

    from tearrl.spec import W_DEBT

    next_debt = carry_form(0.0, harm, 0.0)[1]
    double_charged = -(W_HARM * harm + W_DEBT * next_debt)
    assert double_charged < _conscience(carry, harm), "d_{t+1} would double-charge"


def test_prior_debt_remains_costly() -> None:
    """Carry timing must not accidentally forgive outstanding debt."""
    with_debt = _conscience(carry_form(0.8, 0.0, 0.0)[0])
    without = _conscience(carry_form(0.0, 0.0, 0.0)[0])
    assert with_debt < without


def test_debt_stays_within_bounds() -> None:
    for d in GRID:
        for harm in GRID:
            for restored in GRID:
                carry, nxt = carry_form(d, harm, restored)
                assert 0.0 <= carry <= D_MAX
                assert 0.0 <= nxt <= D_MAX
