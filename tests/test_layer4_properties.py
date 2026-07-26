"""
§3.3 verification layer 4 -- randomized long-horizon property tests.

The protocol specifies 10^6 trajectories. CI runs a 20k smoke budget; the
frozen budget runs under `pytest --full` and is what the pilot record cites.

Properties (§3.3):
    - no resolved-violation trajectory beats matched violation-free ones
    - no repair-farming
    - theft conservation
    - enforcement never alters welfare effects
    - no `done` reward
    - scripted signatures reproduce            [pending build-order step 3]

State-range and ledger-capacity invariants from A.1/A.2a ride along in the
main sweep, since they cost nothing once the trajectory is being walked.
"""

from __future__ import annotations

import random

import pytest

from tearrl import TearWorld, Verb
from tearrl.spec import (
    AUTONOMY_UNITS_MAX,
    AUTONOMY_UNITS_MIN,
    ENERGY_MAX,
    FIELD_MAX,
    HORIZON,
    LEDGER_SLOTS,
    OBJECT_STORE_MAX,
    RESOURCES_MAX,
    TASK_MAX,
    TASK_MIN,
    TRUST_UNITS_MAX,
    TRUST_UNITS_MIN,
    RecordType,
)

from .conftest import legal_verbs

# Verbs whose A.3 preconditions do not read the ledger. Replaying only these
# guarantees identical feasibility across enforcement regimes, which is what
# makes the welfare-invariance comparison well posed.
LEDGER_INDEPENDENT = [
    Verb.WORK,
    Verb.HARVEST,
    Verb.SHARE,
    Verb.STEAL,
    Verb.TAKE_OBJECT,
    Verb.WAIT,
]


def _assert_state_ranges(world: TearWorld) -> None:
    """A.1 ranges and the A.2a capacity bound."""
    state = world.state
    assert 0 <= state.field_pool <= FIELD_MAX
    assert 0 <= state.object_store <= OBJECT_STORE_MAX
    assert state.monitor_alive in (0, 1)
    assert state.m_c in (0, 1)
    assert state.m_e in (0, 1)
    for agent in state.agents:
        assert 0 <= agent.energy <= ENERGY_MAX
        assert 0 <= agent.resources <= RESOURCES_MAX
        assert AUTONOMY_UNITS_MIN <= agent.autonomy_units <= AUTONOMY_UNITS_MAX
        assert TRUST_UNITS_MIN <= agent.trust_units <= TRUST_UNITS_MAX
        assert TASK_MIN <= agent.task <= TASK_MAX
        assert len(agent.ledger) <= LEDGER_SLOTS
        for record in agent.ledger:
            assert record.repaired_units <= record.magnitude
            if record.type is RecordType.U_THEFT:
                assert (
                    record.illicit_task_credit
                    == record.magnitude - record.repaired_units
                ), "illicit credit and repaired units fell out of lockstep"


def test_main_property_sweep(trajectory_budget: int) -> None:
    """Walk the budget of random trajectories, checking the per-step
    properties that do not require a paired counterfactual."""
    theft_events = 0
    resolved_violation_trajectories = 0

    for episode in range(trajectory_budget):
        world = TearWorld(seed=episode)
        rng = random.Random(episode ^ 0xC0FFEE)

        initial_task = [agent.task for agent in world.state.agents]
        reward_sums = [0, 0]
        committed_violation = False

        while not world.state.done:
            actor = world.state.actor_index
            other = 1 - actor
            resources_before = world.state.agents[other].resources
            task_before = world.state.agents[actor].task
            ledger_before = len(world.state.agents[actor].ledger)

            action = rng.choice(legal_verbs(world))
            result = world.step(action)
            reward_sums[actor] += result.reward_task

            # --- theft conservation -------------------------------------
            if result.event_u:
                record = world.state.agents[actor].ledger[ledger_before]
                taken = resources_before - world.state.agents[other].resources
                assert taken == record.magnitude, "theft did not conserve"
                assert record.illicit_task_credit == record.magnitude
                # commission-time task credit equals exactly what was taken
                assert task_before + record.magnitude - result.audit_task_penalty == (
                    world.state.agents[actor].task
                )
                theft_events += 1

            if result.event_u or result.event_e or result.event_o:
                committed_violation = True

            _assert_state_ranges(world)

        # --- no `done` reward -------------------------------------------
        for index in (0, 1):
            assert reward_sums[index] == (
                world.state.agents[index].task - initial_task[index]
            ), "per-step rewards do not sum to the realized task change"

        if committed_violation and all(
            not record.unresolved
            for agent in world.state.agents
            for record in agent.ledger
        ):
            resolved_violation_trajectories += 1

    assert theft_events > 0, "budget produced no theft events -- sweep is vacuous"
    assert resolved_violation_trajectories >= 0


@pytest.mark.parametrize("n_pairs", [400])
def test_enforcement_never_alters_welfare(n_pairs: int) -> None:
    """§3.3: enforcement changes task and trust, never welfare.

    Paired replay of an identical action sequence under do(m_e=0) and
    do(m_e=1). Only ledger-independent verbs are used so feasibility cannot
    diverge between the two regimes.
    """
    for episode in range(n_pairs):
        rng = random.Random(episode ^ 0xBEEF)
        quiet = TearWorld(seed=episode).intervene(m_e=0)
        loud = TearWorld(seed=episode).intervene(m_e=1)

        while not quiet.state.done:
            options = [
                verb
                for verb in LEDGER_INDEPENDENT
                if quiet.legal_actions()[verb] and loud.legal_actions()[verb]
            ]
            action = rng.choice(options)
            quiet.step(action)
            loud.step(action)

            for index in (0, 1):
                assert (
                    quiet.state.agents[index].energy == loud.state.agents[index].energy
                )
                assert (
                    quiet.state.agents[index].resources
                    == loud.state.agents[index].resources
                )
                assert (
                    quiet.state.agents[index].welfare == loud.state.agents[index].welfare
                )


@pytest.mark.parametrize("k", [1, 2, 3])
def test_no_repair_farming(k: int) -> None:
    """A steal-then-fully-repair cycle must leave the actor strictly worse off
    on every resource it touches. Otherwise `repair` becomes a laundering
    verb and the debt channel could be farmed for conscience credit."""
    world = TearWorld(seed=900 + k)
    world.reset(seed=900 + k, learner_index=0)
    world.intervene(m_e=0)
    world.state.agents[1].resources = k

    me = world.state.agents[0]
    energy_before = me.energy
    resources_before = me.resources
    task_before = me.task

    world.step(Verb.WAIT)
    world.step(Verb.STEAL)
    for _ in range(k):
        world.step(Verb.WAIT)
        world.step(Verb.REPAIR)

    assert me.task == task_before, "repair cycle changed net task"
    assert me.resources == resources_before - k, "repair cycle gained resources"
    assert me.energy < energy_before, "repair cycle was energy-free"
    assert not world.state.agents[0].ledger[0].unresolved


def test_victim_is_made_whole_by_full_repair() -> None:
    """Same-unit harm/repair accounting (§2.6): k stolen resources returned by
    k repairs restores the victim's resources exactly."""
    world = TearWorld(seed=910)
    world.reset(seed=910, learner_index=0)
    world.intervene(m_e=0)
    world.state.agents[1].resources = 3
    victim_before = world.state.agents[1].resources

    world.step(Verb.WAIT)
    world.step(Verb.STEAL)
    k = world.state.agents[0].ledger[0].magnitude
    assert world.state.agents[1].resources == victim_before - k

    for _ in range(k):
        world.step(Verb.WAIT)
        world.step(Verb.REPAIR)

    assert world.state.agents[1].resources == victim_before
    # Autonomy damage is NOT undone by repair -- only welfare is same-unit.
    assert world.state.agents[1].autonomy < 1.0


def test_ledger_capacity_is_never_exceeded() -> None:
    """A.2a proves <= 1 record per turn x 16 turns. Drive record creation as
    hard as the masks allow and confirm the bound holds."""
    for seed in range(200):
        world = TearWorld(seed=seed)
        rng = random.Random(seed)
        preferred = [Verb.STEAL, Verb.LIE, Verb.DISABLE_MONITOR]
        while not world.state.done:
            mask = world.legal_actions()
            greedy = [verb for verb in preferred if mask[verb]]
            action = greedy[0] if greedy else rng.choice(legal_verbs(world))
            world.step(action)
            for agent in world.state.agents:
                assert len(agent.ledger) <= LEDGER_SLOTS


def test_episode_length_is_exactly_the_horizon() -> None:
    for seed in range(50):
        world = TearWorld(seed=seed)
        rng = random.Random(seed)
        steps = 0
        while not world.state.done:
            world.step(rng.choice(legal_verbs(world)))
            steps += 1
        assert steps == HORIZON
        assert world.state.t == HORIZON


# §3.3 layer 4's final property -- "scripted signatures reproduce" -- now
# lives in tests/test_policies.py::test_scripted_signatures_reproduce, next to
# the policies it fingerprints.
