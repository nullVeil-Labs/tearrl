"""
§3.3 verification layer 3 -- sampled-state, exhaustive-action one-step
counterfactual-equals-simulation tests.

For sampled reachable states and every feasible action, the predicted one-step
consequence (`predicates.action_effect`, the oracle the witness targets are
built from) must equal what actually happens when the action is simulated.

Under do(m_e=0) no audit fires, so a real step's deltas are exactly the
deterministic action effect -- which is the comparison this layer needs.
The audit itself is verified separately in layers 1 and 4.

If this layer fails, the witness would be trained on targets that the
environment does not honour, and every Delta-channel in §2.4 would be
measuring a fiction.
"""

from __future__ import annotations

import random

import pytest

from tearrl import TearWorld, Verb
from tearrl.predicates import action_effect
from tearrl.spec import N_ACTIONS

from .conftest import legal_verbs


def _reachable_states(n_episodes: int, seed0: int = 0):
    """Yield worlds paused at reachable states along random legal rollouts."""
    for episode in range(n_episodes):
        world = TearWorld(seed=seed0 + episode)
        rng = random.Random(seed0 + episode)
        while not world.state.done:
            yield world.clone()
            world.step(rng.choice(legal_verbs(world)))


@pytest.mark.parametrize("n_episodes", [120])
def test_counterfactual_equals_simulation(n_episodes: int) -> None:
    """Predicted effect == simulated effect, for every feasible action at
    every sampled reachable state."""
    checked = 0
    for world in _reachable_states(n_episodes):
        actor = world.state.actor_index
        mask = world.legal_actions(actor)

        for index in range(N_ACTIONS):
            if not mask[index]:
                continue
            verb = Verb(index)

            predicted = action_effect(world, verb, actor)

            sim = world.clone().intervene(m_e=0)
            me_before = sim.state.agents[actor]
            other_before = sim.state.agents[1 - actor]
            task0 = me_before.task
            welfare0 = other_before.welfare
            autonomy0 = other_before.autonomy
            trust0 = other_before.trust

            result = sim.step(verb)

            me_after = sim.state.agents[actor]
            other_after = sim.state.agents[1 - actor]

            assert predicted.d_task == me_after.task - task0
            assert predicted.d_welfare_other == pytest.approx(
                other_after.welfare - welfare0
            )
            assert predicted.d_autonomy_other == pytest.approx(
                other_after.autonomy - autonomy0
            )
            assert predicted.d_trust_toward_actor == pytest.approx(
                other_after.trust - trust0
            )
            assert predicted.event_u == result.event_u
            assert predicted.event_e == result.event_e
            assert predicted.event_o == result.event_o
            checked += 1

    assert checked > 5000, f"only {checked} (state, action) pairs exercised"


def test_action_effect_does_not_mutate_the_world() -> None:
    """The counterfactual is evaluated on a clone; the caller's world must be
    untouched, including its RNG. §6.4 patching depends on this."""
    world = TearWorld(seed=800)
    for _ in range(7):
        world.step(legal_verbs(world)[0])

    before_state = world.observe(0, include_identity=True)
    before_rng = world.rng.getstate()
    before_t = world.state.t

    for index, allowed in enumerate(world.legal_actions()):
        if allowed:
            action_effect(world, Verb(index))

    assert world.observe(0, include_identity=True) == before_state
    assert world.rng.getstate() == before_rng
    assert world.state.t == before_t


def test_infeasible_actions_report_infeasible_not_zero_effect() -> None:
    """§2.4 masks infeasible actions and leaves them untrained. They must be
    distinguishable from genuinely no-op actions like `wait`."""
    world = TearWorld(seed=801)
    world.reset(seed=801, learner_index=0)
    world.state.agents[0].energy = 0
    world.state.agents[0].resources = 0
    world.step(Verb.WAIT)

    assert not world.legal_actions()[Verb.WORK]
    assert not action_effect(world, Verb.WORK).feasible
    assert action_effect(world, Verb.WAIT).feasible


@pytest.mark.parametrize(
    ("draw_name", "force_name"),
    [("_draw_mc", "force_m_c"), ("_draw_me", "force_m_e")],
)
def test_do_intervention_replaces_draw_without_consuming_rng(
    draw_name: str, force_name: str
) -> None:
    """Forced draws consume nothing; natural draws demonstrably consume once."""
    base = TearWorld(seed=802)
    forced = base.clone()
    setattr(forced, force_name, 1)
    before = forced.rng.getstate()
    assert getattr(forced, draw_name)() == 1
    assert forced.rng.getstate() == before

    free = base.clone()
    setattr(free, force_name, None)
    before = free.rng.getstate()
    getattr(free, draw_name)()
    assert free.rng.getstate() != before, "natural draw did not consume RNG"


def test_paired_enforcement_interventions_remain_aligned_after_release() -> None:
    base = TearWorld(seed=803)
    quiet = base.clone().intervene(m_e=0)
    loud = base.clone().intervene(m_e=1)
    quiet.step(Verb.WAIT)
    loud.step(Verb.WAIT)
    assert quiet.state.m_e == 0 and loud.state.m_e == 1
    assert quiet.rng.getstate() == loud.rng.getstate()

    quiet.intervene(m_e=None)
    loud.intervene(m_e=None)
    quiet.step(Verb.WAIT)
    loud.step(Verb.WAIT)
    assert quiet.state.m_e == loud.state.m_e
    assert quiet.rng.getstate() == loud.rng.getstate()
