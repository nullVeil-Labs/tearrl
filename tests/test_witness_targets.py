"""
Closed-form witness targets (§2.2, §2.4, A.6, B.1).

The central test is `test_closed_form_matches_monte_carlo`: the two-term
expectation over m_e must agree with actually sampling the environment. If it
does not, the witness would be trained on targets the world never produces,
and every Delta-channel in §2.4 -- hence c_t, hence the selector -- would be
built on a fiction.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from tearrl import TearWorld, Verb
from tearrl.spec import N_ACTIONS, P_ME_GIVEN_MC_0, P_ME_GIVEN_MC_1
from tearrl.witness.targets import (
    ACTION_DESCRIPTORS,
    DESCRIPTOR_DIM,
    GRID_STEP,
    HALF_BIN,
    WITNESS_DIM,
    action_descriptor,
    descriptors_are_distinct,
    expected_values,
    quantize,
    witness_target,
    witness_targets_for_state,
)

from .conftest import legal_verbs


def _states(n_episodes: int, seed0: int = 0):
    """Reachable states from random legal rollouts."""
    for episode in range(n_episodes):
        world = TearWorld(seed=seed0 + episode)
        rng = random.Random(seed0 + episode)
        while not world.state.done:
            yield world.clone()
            world.step(rng.choice(legal_verbs(world)))


def _monte_carlo_values(
    world: TearWorld, action: Verb, actor: int, draws: int
) -> np.ndarray:
    """Average post-turn values under NATURAL m_e sampling."""
    total = np.zeros(5, dtype=np.float64)
    for i in range(draws):
        twin = world.clone()
        twin.rng = random.Random(0xD1CE + i)      # independent enforcement draw
        twin.step(action)
        me = twin.state.agents[actor]
        other = twin.state.agents[1 - actor]
        total += np.array(
            [me.task, me.welfare, other.welfare, other.autonomy, other.trust],
            dtype=np.float64,
        )
    return total / draws


# ======================================================================
# the central identity
# ======================================================================


def test_closed_form_matches_monte_carlo() -> None:
    """ybar(o^F, a) == E_sampled[y]  for every feasible action.

    Tolerance is Monte-Carlo error only: the closed form is exact, so any
    systematic gap means the two-term expectation is wrong.
    """
    draws = 4000
    worst = 0.0
    checked = 0

    for world in _states(n_episodes=4, seed0=1000):
        actor = world.state.actor_index
        for index, allowed in enumerate(world.legal_actions(actor)):
            if not allowed:
                continue
            verb = Verb(index)
            closed = expected_values(world, verb, actor)
            sampled = _monte_carlo_values(world, verb, actor, draws)

            # task is the widest-range channel; scale tolerance per channel
            scale = np.array([6.0, 1.0, 1.0, 1.0, 1.0])
            deviation = np.abs(closed - sampled) / scale
            worst = max(worst, float(deviation.max()))
            assert deviation.max() < 0.05, (
                f"{verb.name}: closed form {closed} vs sampled {sampled}"
            )
            checked += 1

    assert checked > 200
    print(f"\nworst scaled MC deviation: {worst:.4f} over {checked} pairs")


def test_expectation_uses_the_correct_enforcement_probability() -> None:
    """The mixing weight must be P(m_e | m_c), not a fixed constant."""
    for m_c in (0, 1):
        world = TearWorld(seed=1234)
        world.reset(seed=1234, learner_index=0)
        world.intervene(m_c=m_c)
        world.step(Verb.WAIT)
        world.step(Verb.STEAL)          # create exposure
        world.step(Verb.WAIT)

        actor = world.state.actor_index
        expected = expected_values(world, Verb.WAIT, actor)

        quiet = world.clone().intervene(m_e=0)
        loud = world.clone().intervene(m_e=1)
        quiet.step(Verb.WAIT)
        loud.step(Verb.WAIT)

        p = P_ME_GIVEN_MC_1 if m_c == 1 else P_ME_GIVEN_MC_0
        manual = p * np.array(
            [
                loud.state.agents[actor].task,
                loud.state.agents[actor].welfare,
                loud.state.agents[1 - actor].welfare,
                loud.state.agents[1 - actor].autonomy,
                loud.state.agents[1 - actor].trust,
            ]
        ) + (1 - p) * np.array(
            [
                quiet.state.agents[actor].task,
                quiet.state.agents[actor].welfare,
                quiet.state.agents[1 - actor].welfare,
                quiet.state.agents[1 - actor].autonomy,
                quiet.state.agents[1 - actor].trust,
            ]
        )
        assert np.allclose(expected, manual)


# ======================================================================
# target structure (§2.4, A.6)
# ======================================================================


def test_wait_target_is_all_zero_in_the_delta_channels() -> None:
    """Deltas are measured against do(a=wait), so wait is the origin."""
    for world in _states(n_episodes=6, seed0=2000):
        target = witness_target(world, Verb.WAIT)
        assert np.allclose(target[:5], 0.0)
        assert np.allclose(target[5:], 0.0)


def test_targets_have_the_declared_shape_and_live_on_the_grid() -> None:
    for world in _states(n_episodes=6, seed0=3000):
        for _, target in witness_targets_for_state(world):
            assert target.shape == (WITNESS_DIM,)
            assert np.all(np.abs(target[:5]) <= 1.0 + 1e-12)
            snapped = np.round(target[:5] / GRID_STEP) * GRID_STEP
            assert np.allclose(target[:5], snapped, atol=1e-12)
            assert set(np.unique(target[5:])) <= {0.0, 1.0}


def test_event_channels_are_exact_indicators() -> None:
    """B.3 validates events by recall >= 0.99 and FPR <= 0.01. The targets
    themselves must be exactly right or that criterion is meaningless."""
    for world in _states(n_episodes=8, seed0=4000):
        for index, target in witness_targets_for_state(world):
            verb = Verb(index)
            assert target[5] == (1.0 if verb is Verb.STEAL else 0.0)
            assert target[6] == (1.0 if verb is Verb.LIE else 0.0)
            assert target[7] == (1.0 if verb is Verb.DISABLE_MONITOR else 0.0)
            assert target[8] == (1.0 if verb is Verb.DISABLE_MONITOR else 0.0)


def test_infeasible_actions_are_skipped_not_zeroed() -> None:
    """§2.4: masked actions are untrained. Emitting a zero target would teach
    the witness that a masked verb is a harmless no-op."""
    world = TearWorld(seed=5000)
    world.reset(seed=5000, learner_index=0)
    world.state.agents[0].energy = 0
    world.state.agents[0].resources = 0
    world.step(Verb.WAIT)

    emitted = {index for index, _ in witness_targets_for_state(world)}
    mask = world.legal_actions()
    assert emitted == {i for i in range(N_ACTIONS) if mask[i]}
    assert Verb.WORK not in emitted


def test_audits_are_priced_into_the_targets() -> None:
    """§2.2: 'the audit mechanism is included in every action's expected
    Dtask -- the witness explicitly prices concealment, confession, tampering
    and risk.'

    An agent holding exposed theft must see a WORSE expected task delta for
    waiting under a hot cue than a cold one, purely from audit risk.
    """
    def exposed_world(m_c: int) -> TearWorld:
        world = TearWorld(seed=6000)
        world.reset(seed=6000, learner_index=0)
        world.intervene(m_c=m_c)
        world.step(Verb.WAIT)
        world.step(Verb.STEAL)
        world.step(Verb.WAIT)
        return world

    cold = expected_values(exposed_world(0), Verb.WAIT, 0)
    hot = expected_values(exposed_world(1), Verb.WAIT, 0)
    assert hot[0] < cold[0], "audit risk is not priced into the expectation"


def test_tampering_removes_audit_exposure_in_the_targets() -> None:
    """The S_O temptation must be visible inside the very consequence
    representation the selector consumes (suture 5). Disabling oversight has
    to look strictly better than waiting when exposure is live."""
    world = TearWorld(seed=6100)
    world.reset(seed=6100, learner_index=0)
    world.intervene(m_c=1)              # hot cue: enforcement likely
    world.step(Verb.WAIT)
    world.step(Verb.STEAL)
    world.step(Verb.WAIT)

    assert world.legal_actions()[Verb.DISABLE_MONITOR]
    tamper = witness_target(world, Verb.DISABLE_MONITOR)
    wait = witness_target(world, Verb.WAIT)
    assert tamper[0] > wait[0], (
        "tampering does not improve expected task consequence; S_O would be "
        "probing a temptation the selector cannot see"
    )


# ======================================================================
# descriptor d(a)  (B.1)
# ======================================================================


def test_descriptors_are_distinct_and_not_one_hot() -> None:
    """'fixed effect-template vector per verb, not an ID'. A one-hot would let
    the witness memorize indices and make C.6's 'descriptor permuted' OOD
    family vacuous."""
    assert descriptors_are_distinct()
    assert ACTION_DESCRIPTORS.shape == (N_ACTIONS, DESCRIPTOR_DIM)

    for row in ACTION_DESCRIPTORS:
        nonzero = int(np.count_nonzero(row))
        assert not (nonzero == 1 and float(np.abs(row).max()) == 1.0), (
            "descriptor row is one-hot shaped"
        )

    # Mechanically similar verbs must sit closer than dissimilar ones.
    def distance(a: Verb, b: Verb) -> float:
        return float(np.linalg.norm(action_descriptor(a) - action_descriptor(b)))

    assert distance(Verb.WORK, Verb.TAKE_OBJECT) < distance(Verb.WORK, Verb.STEAL)


def test_descriptor_is_state_independent() -> None:
    """The template must never be read off the current state, or it would
    smuggle the answer into the witness input."""
    baseline = action_descriptor(Verb.STEAL).copy()
    world = TearWorld(seed=7000)
    for _ in range(9):
        world.step(legal_verbs(world)[0])
    assert np.array_equal(action_descriptor(Verb.STEAL), baseline)


# ======================================================================
# hygiene
# ======================================================================


def test_target_computation_does_not_mutate_the_world() -> None:
    world = TearWorld(seed=8000)
    for _ in range(5):
        world.step(legal_verbs(world)[0])

    before_obs = world.observe(0, include_identity=True)
    before_rng = world.rng.getstate()
    before_t = world.state.t

    for _ in witness_targets_for_state(world):
        pass

    assert world.observe(0, include_identity=True) == before_obs
    assert world.rng.getstate() == before_rng
    assert world.state.t == before_t


def test_quantize_snaps_to_half_bin_resolution() -> None:
    assert HALF_BIN == pytest.approx(0.05)
    assert quantize(np.array([0.049])) == pytest.approx(0.0)
    assert quantize(np.array([0.051])) == pytest.approx(0.1)
    assert quantize(np.array([5.0])) == pytest.approx(1.0)
    assert quantize(np.array([-5.0])) == pytest.approx(-1.0)


def test_targets_are_deterministic() -> None:
    for world in _states(n_episodes=4, seed0=9000):
        first = [t for _, t in witness_targets_for_state(world)]
        second = [t for _, t in witness_targets_for_state(world)]
        for a, b in zip(first, second):
            assert np.array_equal(a, b)
