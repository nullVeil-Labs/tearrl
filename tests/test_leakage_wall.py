"""
The §2.1 leakage wall -- the expected-null diagnostic.

    x_t   full latent state, including m_e,t
    o_t   = x_t \\ {m_e,t, other's concealed/unreported records}

§2.1 states the diagnostic directly: identical o_t with varied m_e,t must
yield identical policy logits. It is an EXPECTED NULL. A non-null here does
not mean the agent learned something interesting -- it means the observation
construction or the harness handed the policy the enforcement draw, and every
downstream restraint metric is then measuring risk-modelling rather than
disposition. UCP is defined under do(m_c=0, m_e=0); if the policy can see
m_e, UCP stops being a restraint measure at all.

`state.py` asserts this null holds by construction. This file is where that
claim is actually made falsifiable, at two levels:

  STRUCTURAL  -- observe() output is bit-identical under varied m_e
  BEHAVIOURAL -- every ActionDistribution the harness scores is invariant,
                 measured through the same _masked_distribution path the
                 probes use, so a leak anywhere in that path surfaces here

Both are mutation-tested below: a deliberately m_e-reading policy must fail
the behavioural guard, or the guard is decoration.

WHY THE MASK MATTERS. `legal_actions` reads energy, resources, field_pool,
object_store, ledger and monitor_alive -- never m_e (world.py). So varying
m_e cannot move the feasibility mask, and any change in the renormalized
distribution is leakage rather than a change in what was available. That
independence is asserted here rather than assumed, because if it ever broke
the behavioural test would start reporting mask differences as leaks.
"""

from __future__ import annotations

import numpy as np
import pytest

from tearrl import TearWorld, Verb
from tearrl.agents.inputs import FrozenResponsibility, PolicyInput, policy_input
from tearrl.conscience import ResponsibilityState, zero_state
from tearrl.eval.probes import (
    ADAPTER_KIND_ATTR,
    _masked_distribution,
    eip,
    evaluate,
    evasion_allocation,
    ig_cue,
    joint_evasion_propensity,
    learned_adapter,
    oip,
    p_safe_productive,
    scripted_adapter,
    trusted_adapter,
    ucp,
)
from tearrl.eval.tuples import STTupleSet, generate_s_t
from tearrl.policies import ALL_POLICIES
from tearrl.spec import N_ACTIONS
from tearrl.state import OBS_DIM, AgentState, WorldState
from tearrl.witness.dataset import INPUT_DIM
from tearrl.witness.model import WitnessEnsemble

STOCHASTIC = {"random", "mix_coop_cheat", "mix_cheat_liar",
              "mix_cheat_tamperer", "mix_coop_random", "mix_monitor_liar"}


@pytest.fixture(scope="module")
def tuples() -> STTupleSet:
    witness = WitnessEnsemble(INPUT_DIM)
    witness.eval()
    return generate_s_t(witness, n_candidates=600, device="cpu", verbose=False)


# ======================================================================
# structural: m_e is not in the observation
# ======================================================================


def test_observation_is_bit_identical_under_varied_m_e() -> None:
    """o_t carries no trace of the enforcement draw.

    Exact equality, not approximate: m_e is meant to be ABSENT, so there is
    no tolerance to allow. A float-tolerant assertion here would pass a
    construction that scaled m_e down rather than omitting it.
    """
    for seed in range(25):
        quiet = TearWorld(seed=seed).intervene(m_e=0)
        loud = TearWorld(seed=seed).intervene(m_e=1)

        assert quiet.state.m_e == 0 and loud.state.m_e == 1, (
            "intervene() did not pin m_e -- the test would be comparing two "
            "worlds that happened to draw the same value"
        )
        for index in (0, 1):
            assert quiet.observe(index) == loud.observe(index)
            assert quiet.observe(index, include_identity=True) == loud.observe(
                index, include_identity=True
            )


def test_feasibility_mask_does_not_depend_on_m_e() -> None:
    """Precondition for reading the behavioural test as a leakage test.

    If m_e ever moved the mask, a distribution difference would be a change
    in what was AVAILABLE rather than a change in disposition, and the null
    below would fail for a reason that is not leakage.
    """
    for seed in range(25):
        quiet = TearWorld(seed=seed).intervene(m_e=0)
        loud = TearWorld(seed=seed).intervene(m_e=1)
        for index in (0, 1):
            assert quiet.legal_actions(index) == loud.legal_actions(index)


def test_m_e_does_reach_the_world_somewhere() -> None:
    """Guard against a vacuous suite: m_e must matter to the ENVIRONMENT even
    though it is invisible to the policy.

    If a refactor made m_e inert everywhere, every test in this file would
    pass while asserting nothing. Detection is the channel §3.1 gives it --
    enforcement fires only when m_e == 1 -- so a steal under do(m_e=1) must
    be detectable and under do(m_e=0) must not.

    TIMING. `_audit` sweeps own auditable records of age >= 1 (A.3a), so a
    record is never detected on the turn it is created. The steal must be
    followed by two more steps -- counterpart, then actor again -- before the
    sweep can see it. Checking immediately after the steal reports a clean
    ledger under BOTH interventions and would make this guard pass vacuously
    in the one direction and fail in the other.
    """
    detected_when_loud = False
    detected_when_quiet = False

    for seed in range(40):
        for value in (0, 1):
            world = TearWorld(seed=seed)
            world.reset(seed=seed, learner_index=0)
            world.intervene(m_e=value)
            world.step(Verb.WAIT)                 # t=0, counterpart
            if not world.legal_actions()[Verb.STEAL]:
                continue
            world.step(Verb.STEAL)                # t=1, actor -- record age 0
            world.step(Verb.WAIT)                 # t=2, counterpart
            world.step(Verb.WAIT)                 # t=3, actor -- sweep runs
            records = world.state.agents[0].ledger
            if any(record.detected for record in records):
                if value == 1:
                    detected_when_loud = True
                else:
                    detected_when_quiet = True

    assert detected_when_loud, (
        "no steal was ever detected under do(m_e=1) -- m_e has become inert "
        "and every leakage assertion in this file is vacuous"
    )
    assert not detected_when_quiet, (
        "a steal was detected under do(m_e=0) -- enforcement is not gated on "
        "the draw the §2.1 wall hides"
    )


# ======================================================================
# behavioural: the null itself
# ======================================================================


def _distribution_under(policy, item, m_e: int, allow_disable: bool) -> np.ndarray:
    world = item.probe(m_c=0, m_e=m_e)
    return _masked_distribution(policy, world, item.actor, item.c, allow_disable)


def _max_leak(policy, items, allow_disable: bool = True) -> float:
    """Largest total-variation gap between the do(m_e=0) and do(m_e=1) policies."""
    worst = 0.0
    for item in items:
        quiet = _distribution_under(policy, item, 0, allow_disable)
        loud = _distribution_under(policy, item, 1, allow_disable)
        worst = max(worst, float(np.abs(quiet - loud).sum()))
    return worst


@pytest.mark.parametrize("name", sorted(ALL_POLICIES))
def test_scripted_policies_are_invariant_to_m_e(name: str, tuples: STTupleSet) -> None:
    """Every App. D anchor passes the §2.1 null.

    This is not a formality. These policies calibrate every threshold in
    App. D; an anchor that responded to m_e would make tau_U, tau_E and
    tau_O functions of the enforcement draw rather than of conduct.

    `monitor_conditional` is the sharpest case and is deliberately included:
    it is the punishment calculator, and it reads m_c -- the VISIBLE cue --
    every decision. Reading m_c is legal and is the whole point of G3. If
    this test could not tell m_c-reading from m_e-reading it would be
    worthless, so it must pass here while the mutant below fails.
    """
    policy = scripted_adapter(
        ALL_POLICIES[name], samples=64 if name in STOCHASTIC else 1
    )
    items = list(tuples)[:40]
    assert items, "empty tuple set -- nothing was tested"

    leak = _max_leak(policy, items)
    assert leak == 0.0, (
        f"scripted policy '{name}' changed its action distribution when only "
        f"m_e varied (max total-variation gap {leak:.6f}); the §2.1 wall has "
        f"a door"
    )


def test_leakage_detector_catches_an_m_e_reading_policy(tuples: STTupleSet) -> None:
    """Mutation test: the guard above must fail on a policy that reads m_e.

    Without this, `leak == 0.0` could hold because the detector never varies
    anything -- the classic tautological guard. The mutant reads
    `world.state.m_e` through exactly the door this build step exists to
    close, which is also a live demonstration that the door is open: a
    learned policy handed the whole TearWorld can do this today.
    """
    def leaky(world, actor, c):
        out = np.ones(N_ACTIONS, dtype=np.float64)
        # the door: the harness hands over the entire world, m_e included
        out[int(Verb.STEAL)] = 50.0 if world.state.m_e == 0 else 1.0
        return out

    items = [t for t in tuples if t.world.legal_actions(t.actor)[Verb.STEAL]][:40]
    assert items, "no probe state offered STEAL -- mutation test is vacuous"

    leak = _max_leak(leaky, items)
    assert leak > 0.0, (
        "the leakage detector reported a perfect null for a policy that "
        "explicitly branches on m_e -- the guard is not measuring anything"
    )


def test_sanitized_input_is_identical_under_varied_m_e() -> None:
    """PolicyInput is invariant to the enforcement draw.

    The behavioural counterpart to the structural seal below: two worlds
    differing ONLY in m_e must produce equal inputs, field for field. If a
    future field were added that carried m_e, this fails before any network
    is involved.
    """
    for seed in range(25):
        quiet = TearWorld(seed=seed).intervene(m_e=0)
        loud = TearWorld(seed=seed).intervene(m_e=1)
        for index in (0, 1):
            a = policy_input(quiet, actor=index)
            b = policy_input(loud, actor=index)
            assert a == b, "sanitized input differed under do(m_e)"


def test_sanitized_input_holds_no_path_back_to_the_world() -> None:
    """Structural seal: m_e is unreachable, not merely uncopied.

    Walks every field and asserts none is a world, a world state, or an
    agent state. A `PolicyInput` that stored `world` would pass the equality
    test above only by luck of field ordering; this closes that.
    """
    world = TearWorld(seed=77).intervene(m_e=1)
    pi = policy_input(world)

    forbidden = (TearWorld, WorldState, AgentState)
    for field_name in PolicyInput.__slots__:
        value = getattr(pi, field_name)
        assert not isinstance(value, forbidden), (
            f"PolicyInput.{field_name} is a {type(value).__name__}; the "
            f"sanitized input has a path back to m_e"
        )
        assert not hasattr(value, "m_e"), (
            f"PolicyInput.{field_name} exposes an m_e attribute"
        )

    # And the container itself refuses to carry one.
    with pytest.raises(Exception):
        pi.obs = (0.0,) * OBS_DIM          # frozen


def test_sanitized_input_carries_no_unregistered_fields() -> None:
    """B.6 registers what a policy may condition on. Nothing else rides along.

    `actor` in particular is bookkeeping the CALLER already has -- it passes
    the value in -- so carrying it back out would expose a field no network
    consumes and none of B.6 has registered.
    """
    assert set(PolicyInput.__slots__) == {"obs", "q_other", "mask", "c"}


def test_responsibility_state_is_immutable_not_merely_copied() -> None:
    """c must be genuinely immutable, not a fresh mutable copy.

    Cloning protects the CALLER from the policy. It does not stop the policy
    mutating what it was handed, which leaves `PolicyInput`'s freeze shallow:
    any later reader of the same input sees rewritten conscience values. The
    boundary type has nowhere to write at all.
    """
    world = TearWorld(seed=78)
    c = ResponsibilityState(h=1.0, b=2.0, r=0.5, d=3.0)
    pi = policy_input(world, c=c)

    assert isinstance(pi.c, FrozenResponsibility)
    assert pi.c.as_array().tolist() == [1.0, 2.0, 0.5, 3.0]

    with pytest.raises(AttributeError):
        pi.c.h = 999.0                      # tuple: no __setattr__ path

    # The caller's state is untouched, and remains the mutable working type.
    assert c.h == 1.0
    assert isinstance(c, ResponsibilityState)

    # A mutable c is refused at the boundary rather than silently accepted.
    with pytest.raises(TypeError, match="FrozenResponsibility"):
        PolicyInput(
            obs=pi.obs, q_other=pi.q_other, mask=pi.mask,
            c=ResponsibilityState(),        # type: ignore[arg-type]
        )


# ======================================================================
# the world stops at the adapter
# ======================================================================


def test_learned_adapter_never_hands_a_world_to_the_policy() -> None:
    """The seal Codex asked for: a learned condition receives PolicyInput.

    Captures whatever the policy was actually called with and asserts it is
    not a world and exposes no m_e. This is prevention rather than the
    after-the-fact null: a policy that cannot reach m_e cannot leak it.
    """
    seen: list[object] = []

    def recording_policy(sanitized):
        seen.append(sanitized)
        return np.ones(N_ACTIONS, dtype=np.float64)

    world = TearWorld(seed=81).intervene(m_e=1)
    adapter = learned_adapter(recording_policy)
    adapter(world, 0, zero_state())

    assert seen, "adapter never called the policy"
    got = seen[0]
    assert isinstance(got, PolicyInput)
    assert not isinstance(got, TearWorld)
    assert not hasattr(got, "state"), "the policy was handed something world-like"
    assert not hasattr(got, "m_e")


def test_learned_adapter_rejects_a_world_taking_callback() -> None:
    """A world-taking policy routed through the sealed adapter must fail loudly.

    It cannot silently work: what arrives has no `.state`, so the attribute
    access that would have read m_e raises instead. The failure mode for
    wiring a condition up wrongly is a crash, not a quiet leak.
    """
    def world_taking(world, actor=None, c=None):
        # exactly the leak this build step exists to prevent
        return np.full(N_ACTIONS, float(world.state.m_e))

    world = TearWorld(seed=82).intervene(m_e=1)
    adapter = learned_adapter(world_taking)

    with pytest.raises(AttributeError, match="state"):
        adapter(world, 0, zero_state())


def test_learned_conditions_are_invariant_to_m_e_by_construction() -> None:
    """The prevention counterpart to the behavioural null.

    A policy that TRIES to branch on m_e through the sanitized input cannot:
    there is no field carrying it, so its output is identical under both
    interventions. Contrast `test_leakage_detector_catches_an_m_e_reading_policy`,
    where the same intent succeeds because the world was handed over.
    """
    def would_leak_if_it_could(sanitized):
        out = np.ones(N_ACTIONS, dtype=np.float64)
        # Nothing here can reach m_e; the best it can do is read the cue.
        out[int(Verb.STEAL)] = 50.0 if getattr(sanitized, "m_e", 0) == 0 else 1.0
        return out

    adapter = learned_adapter(would_leak_if_it_could)
    for seed in range(20):
        quiet = TearWorld(seed=seed).intervene(m_e=0)
        loud = TearWorld(seed=seed).intervene(m_e=1)
        a = adapter(quiet, 0, zero_state())
        b = adapter(loud, 0, zero_state())
        assert np.array_equal(a, b)


def test_sanitized_input_rejects_a_mask_the_environment_cannot_produce() -> None:
    """A.3 guarantees `wait` is always legal, so an all-infeasible mask is a
    bug upstream. Fail at the boundary rather than inside a softmax."""
    world = TearWorld(seed=79)
    good = policy_input(world)

    with pytest.raises(ValueError, match="did not come from A.3"):
        PolicyInput(
            obs=good.obs,
            q_other=good.q_other,
            mask=(False,) * N_ACTIONS,
            c=good.c,
        )

    with pytest.raises(ValueError, match="obs width"):
        PolicyInput(
            obs=good.obs[:-1],
            q_other=good.q_other,
            mask=good.mask,
            c=good.c,
        )


def test_evaluate_refuses_an_unregistered_condition(tuples: STTupleSet) -> None:
    """Fail-closed registration: the harness will not score an undeclared callable.

    `ActionDistribution` must accept a world, so the type system cannot tell a
    trusted script from a learned net. Registration can, and it fails closed --
    an untagged callable is refused rather than assumed innocent, because the
    cost of a wrong assumption here is a silently invalid H1 result rather
    than a crash.
    """
    def unregistered(world, actor, c):
        return np.ones(N_ACTIONS, dtype=np.float64)

    small = STTupleSet(tuples=list(tuples)[:10])
    with pytest.raises(TypeError, match="not a registered adapter"):
        evaluate(unregistered, small, "unregistered")

    # Gating `evaluate` alone would leave every probe as its own open door.
    for probe in (ucp, eip, oip, ig_cue, p_safe_productive,
                  joint_evasion_propensity, evasion_allocation):
        with pytest.raises(TypeError, match="not a registered adapter"):
            probe(unregistered, small)


def test_registration_tags_match_the_adapter_used(tuples: STTupleSet) -> None:
    """The tag records which door a condition came through, and they differ."""
    scripted = scripted_adapter(ALL_POLICIES["always_cooperate"])
    learned = learned_adapter(lambda s: np.ones(N_ACTIONS))
    double = trusted_adapter(lambda w, a, c: np.ones(N_ACTIONS), why="test double")

    assert getattr(scripted, ADAPTER_KIND_ATTR) == "scripted"
    assert getattr(learned, ADAPTER_KIND_ATTR) == "learned"
    assert getattr(double, ADAPTER_KIND_ATTR) == "trusted"


def test_trusted_adapter_requires_a_stated_reason() -> None:
    """The bypass must be visible in the diff, not silently applied."""
    for bad in ("", "   "):
        with pytest.raises(ValueError, match="non-empty reason"):
            trusted_adapter(lambda w, a, c: np.ones(N_ACTIONS), why=bad)


def test_a_learned_condition_scores_through_the_sealed_path(
    tuples: STTupleSet,
) -> None:
    """End-to-end: a learned condition is measurable, and never sees a world.

    Guards against closing the door by making the room unusable -- if the
    sealed path could not actually produce probe numbers, the wall would be
    enforced by nobody being able to evaluate a learned policy at all.
    """
    seen_types: set[type] = set()

    def uniform_learned(sanitized: PolicyInput) -> np.ndarray:
        seen_types.add(type(sanitized))
        return np.ones(N_ACTIONS, dtype=np.float64)

    small = STTupleSet(tuples=list(tuples)[:40])
    result = evaluate(learned_adapter(uniform_learned), small, "learned_uniform")

    assert seen_types == {PolicyInput}, f"policy saw {seen_types}"
    assert 0.0 <= result.ucp <= 1.0
    assert 0.0 <= result.eip <= 1.0
    assert not np.isnan(result.p_safe)


def test_m_c_reading_is_not_mistaken_for_leakage(tuples: STTupleSet) -> None:
    """Discrimination check: reading the VISIBLE cue must not trip the guard.

    m_c is in o_t by construction (state.py) and G3 exists precisely to
    measure policies that condition on it. A leakage detector that flagged
    m_c-reading would forbid the behaviour the protocol wants to study.
    """
    def cue_reader(world, actor, c):
        out = np.ones(N_ACTIONS, dtype=np.float64)
        out[int(Verb.STEAL)] = 50.0 if world.state.m_c == 0 else 1.0
        return out

    items = list(tuples)[:40]
    assert _max_leak(cue_reader, items) == 0.0, (
        "reading m_c was reported as an m_e leak; the detector cannot "
        "distinguish the visible cue from the latent draw"
    )
