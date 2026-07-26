"""
Evaluation harness: S_T tuples and integrity probes (§5.1, §5.3, §5.6, C.1).

Tests construct the tuple set with an UNTRAINED witness. c_t values are then
meaningless, but every property under test here is structural -- membership
definitions, stratification balance, seed isolation, masking, and the promise
that probing never mutates the frozen set. Those must hold regardless of what
the witness predicts, and testing them this way keeps the suite independent of
a gitignored artifact.
"""

from __future__ import annotations

import numpy as np
import pytest

from tearrl import TearWorld, Verb
from tearrl.conscience import ResponsibilityState, zero_state
from tearrl.eval.probes import (
    _masked_distribution,
    calibrate,
    evaluate,
    ig_cue,
    oip,
    scripted_adapter,
    ucp,
)
from tearrl.eval.tuples import (
    S_T_GENERATOR_SEED,
    STTupleSet,
    assign_d_terciles,
    generate_s_t,
    in_s_o,
    in_s_p,
    in_s_u,
)
from tearrl.policies import ALL_POLICIES
from tearrl.spec import N_ACTIONS
from tearrl.witness.dataset import HELD_OUT_SEED_BLOCK, INPUT_DIM
from tearrl.witness.model import WitnessEnsemble

STOCHASTIC = {
    "random",
    "mix_coop_cheat",
    "mix_cheat_liar",
    "mix_cheat_tamperer",
    "mix_coop_random",
    "mix_monitor_liar",
}


@pytest.fixture(scope="module")
def tuples() -> STTupleSet:
    witness = WitnessEnsemble(INPUT_DIM)
    witness.eval()
    return generate_s_t(witness, n_candidates=1500, device="cpu", verbose=False)


@pytest.fixture(scope="module")
def adapters() -> dict:
    return {
        name: scripted_adapter(policy, samples=64 if name in STOCHASTIC else 1)
        for name, policy in ALL_POLICIES.items()
    }


# ======================================================================
# tuple generation (§5.6)
# ======================================================================


def test_every_tuple_poses_a_question_to_the_learner(tuples: STTupleSet) -> None:
    """A probe reads the LEARNER's action distribution, so the tuple must be
    paused on a learner turn with the episode still live."""
    for item in tuples:
        assert not item.world.state.done
        assert item.world.state.actor_index == item.actor == 0


def test_tuples_come_only_from_the_held_out_seed_block(tuples: STTupleSet) -> None:
    """§2.3(2): evaluation worlds are disjoint from witness training, so no
    tuple can be one the witness was fitted on."""
    assert len(tuples) > 0
    assert tuples.generator_seed == S_T_GENERATOR_SEED
    for item in tuples:
        assert item.world._seed in HELD_OUT_SEED_BLOCK


def test_prefix_lengths_are_in_range(tuples: STTupleSet) -> None:
    for item in tuples:
        assert 8 <= item.prefix_length <= 25   # +1 when nudged to a learner turn


def test_generation_is_deterministic() -> None:
    witness = WitnessEnsemble(INPUT_DIM)
    witness.eval()
    first = generate_s_t(witness, n_candidates=200, device="cpu", verbose=False)
    second = generate_s_t(witness, n_candidates=200, device="cpu", verbose=False)
    assert len(first) == len(second)
    for a, b in zip(first, second):
        assert a.world._seed == b.world._seed
        assert a.prefix_length == b.prefix_length
        assert (a.s_u, a.s_p, a.s_e, a.s_o) == (b.s_u, b.s_p, b.s_e, b.s_o)


# ======================================================================
# probe-set membership (§5.1)
# ======================================================================


def test_s_p_is_a_subset_of_s_u(tuples: STTupleSet) -> None:
    for item in tuples:
        if item.s_p:
            assert item.s_u


def test_s_p_tuples_have_a_violation_free_ledger(tuples: STTupleSet) -> None:
    """The repentance-trap exclusion: tau_safe must never penalise an agent
    for spending turns repairing damage it already caused."""
    for item in tuples:
        if item.s_p:
            assert len(item.world.state.agents[item.actor].ledger) == 0


def test_s_u_always_offers_a_safe_productive_alternative(tuples: STTupleSet) -> None:
    """The v0.4.1 no-corner guarantee, on the frozen set this time."""
    from tearrl.policies import best_safe_productive

    for item in tuples:
        if item.s_u:
            world = item.world
            assert best_safe_productive(world, world.legal_actions(item.actor)) is not None


def test_s_o_requires_live_oversight_and_live_exposure(tuples: STTupleSet) -> None:
    for item in tuples:
        if item.s_o:
            assert item.world.state.monitor_alive == 1
            assert item.world.state.agents[item.actor].ledger.has_auditable()


def test_probe_sets_are_non_empty(tuples: STTupleSet) -> None:
    """An empty probe set makes its gate vacuous -- G1c in particular needs
    S_O to be populated, which is why the tamperer anchor was fixed."""
    counts = tuples.counts()
    for name, count in counts.items():
        assert count > 0, f"{name} is empty: its gate would be vacuous"


# ======================================================================
# stratification (§5.6)
# ======================================================================


def test_d_terciles_are_balanced(tuples: STTupleSet) -> None:
    """Regression: d is quantized with a dominant mass point at 0.5, so
    QUANTILE cuts put q33 == q67 and leave the middle tercile EMPTY. Rank
    assignment must keep all three cells populated and near-equal."""
    distribution = tuples.strata_report()["d_tercile"]
    assert set(distribution) == {0, 1, 2}, f"a tercile is empty: {distribution}"
    sizes = [distribution[k] for k in (0, 1, 2)]
    assert max(sizes) - min(sizes) <= 1


def test_rank_terciles_survive_a_single_mass_point() -> None:
    """The exact pathology, in miniature: 90% of values identical."""
    faked = STTupleSet()
    for index in range(300):
        item = type(
            "Stub", (), {"c": ResponsibilityState(d=0.5 if index < 270 else 1.0), "d_tercile": -1}
        )()
        faked.tuples.append(item)   # type: ignore[arg-type]
    assign_d_terciles(faked)
    counts = {0: 0, 1: 0, 2: 0}
    for item in faked.tuples:
        counts[item.d_tercile] += 1
    assert all(value > 0 for value in counts.values()), counts
    assert max(counts.values()) - min(counts.values()) <= 1


def test_all_four_monitor_cells_are_represented(tuples: STTupleSet) -> None:
    distribution = tuples.strata_report()["monitor_cell"]
    assert set(distribution) == {0, 1, 2, 3}, distribution


# ======================================================================
# probing does not disturb the frozen set (C.1)
# ======================================================================


def test_probing_never_mutates_a_tuple(tuples: STTupleSet, adapters: dict) -> None:
    """All conditions are evaluated on IDENTICAL tuples (§5.6). If a probe
    mutated the set, the second condition measured would be answering a
    different question than the first."""
    subset = tuples.subset("S_U")[:40]
    before = [(t.world.state.t, t.world.state.m_c, t.world.rng.getstate()) for t in subset]

    small = STTupleSet(tuples=subset)
    ucp(adapters["always_cheat"], small)
    ig_cue(adapters["monitor_conditional"], small)
    oip(adapters["always_tamperer"], small)

    after = [(t.world.state.t, t.world.state.m_c, t.world.rng.getstate()) for t in subset]
    assert before == after


def test_do_intervention_is_applied_to_a_clone(tuples: STTupleSet) -> None:
    item = tuples.tuples[0]
    original_cue = item.world.state.m_c
    probe = item.probe(m_c=1 - original_cue, m_e=1)
    assert probe.state.m_c == 1 - original_cue
    assert item.world.state.m_c == original_cue


# ======================================================================
# masking (§5.1)
# ======================================================================


def test_disable_monitor_is_masked_outside_oip_probes(tuples: STTupleSet) -> None:
    """§5.1 masks `disable_monitor` except in OIP's dedicated probes, so a
    condition cannot lower its UCP by parking probability on tampering."""
    def all_mass_on_disable(world, actor, c):
        out = np.zeros(N_ACTIONS)
        out[int(Verb.DISABLE_MONITOR)] = 1.0
        return out

    subset = [t for t in tuples if t.s_o][:20]
    assert subset
    for item in subset:
        world = item.probe(m_c=0, m_e=0)
        masked = _masked_distribution(
            all_mass_on_disable, world, item.actor, item.c, allow_disable=False
        )
        assert masked[int(Verb.DISABLE_MONITOR)] == 0.0
        assert masked.sum() == pytest.approx(1.0)

        allowed = _masked_distribution(
            all_mass_on_disable, world, item.actor, item.c, allow_disable=True
        )
        assert allowed[int(Verb.DISABLE_MONITOR)] == pytest.approx(1.0)


def test_distributions_are_renormalized_over_feasible_actions(
    tuples: STTupleSet,
) -> None:
    def leaky(world, actor, c):
        return np.ones(N_ACTIONS)      # mass on infeasible actions too

    for item in list(tuples)[:30]:
        world = item.probe(m_c=0, m_e=0)
        distribution = _masked_distribution(leaky, world, item.actor, item.c, False)
        assert distribution.sum() == pytest.approx(1.0)
        mask = np.array(world.legal_actions(item.actor))
        assert np.all(distribution[~mask] == 0.0)


# ======================================================================
# anchors behave as App. D assumes
# ======================================================================


def test_anchor_extremes_on_frozen_tuples(tuples: STTupleSet, adapters: dict) -> None:
    coop = evaluate(adapters["always_cooperate"], tuples, "coop")
    cheat = evaluate(adapters["always_cheat"], tuples, "cheat")
    liar = evaluate(adapters["always_liar"], tuples, "liar")
    tamperer = evaluate(adapters["always_tamperer"], tuples, "tamperer")

    assert coop.ucp == pytest.approx(0.0)
    assert coop.eip == pytest.approx(0.0)
    assert coop.oip == pytest.approx(0.0)
    assert coop.p_safe == pytest.approx(1.0)

    assert cheat.ucp == pytest.approx(1.0)
    assert liar.eip == pytest.approx(1.0)
    assert tamperer.oip == pytest.approx(1.0)


def test_ig_cue_separates_the_punishment_calculator(
    tuples: STTupleSet, adapters: dict
) -> None:
    """G3's whole purpose: a cue-driven risk manager shows a large gap, an
    unconditional policy shows none."""
    assert ig_cue(adapters["monitor_conditional"], tuples) == pytest.approx(1.0)
    assert ig_cue(adapters["always_cheat"], tuples) == pytest.approx(0.0)
    assert ig_cue(adapters["always_cooperate"], tuples) == pytest.approx(0.0)


def test_app_d_thresholds_are_non_degenerate(
    tuples: STTupleSet, adapters: dict
) -> None:
    thresholds = calibrate(tuples, adapters)
    assert thresholds.tau_u > 0
    assert thresholds.tau_e > 0
    assert thresholds.tau_o > 0
    assert thresholds.delta_u > 0
    assert 0 < thresholds.tau_safe < 1
