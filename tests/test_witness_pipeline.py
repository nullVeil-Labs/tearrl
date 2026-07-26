"""
Witness dataset, split and model plumbing (§2.3, B.1, B.3).

These tests guard the properties that make the freeze criterion MEAN
something. A witness can report 99.9% accuracy on held-out pairs and still be
worthless if the held-out set shares states with training, or if evaluation
tuples leaked into the dataset.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tearrl.spec import N_ACTIONS, Verb
from tearrl.state import OBS_DIM
from tearrl.witness.dataset import (
    HELD_OUT_SEED_BLOCK,
    INPUT_DIM,
    WitnessDataset,
    build_dataset,
    is_held_out,
    split,
)
from tearrl.witness.model import ENSEMBLE_SIZE, WitnessEnsemble
from tearrl.witness.targets import (
    ACTION_DESCRIPTORS,
    DESCRIPTOR_DIM,
    GRID_STEP,
    WITNESS_DIM,
)


@pytest.fixture(scope="module")
def small_dataset() -> WitnessDataset:
    return build_dataset(episodes=600, progress_every=0, verbose=False)


# ======================================================================
# dataset integrity
# ======================================================================


def test_dataset_shapes_are_consistent(small_dataset: WitnessDataset) -> None:
    ds = small_dataset
    assert ds.obs.shape == (ds.n_states, OBS_DIM)
    assert ds.state_index.shape == (ds.n_pairs,)
    assert ds.action.shape == (ds.n_pairs,)
    assert ds.target.shape == (ds.n_pairs, WITNESS_DIM)
    assert ds.n_pairs > ds.n_states, "states should contribute several actions each"
    assert int(ds.state_index.max()) == ds.n_states - 1
    assert set(np.unique(ds.action)) <= set(range(N_ACTIONS))


def test_observations_are_deduplicated(small_dataset: WitnessDataset) -> None:
    """Every stored o^F must be unique -- dedup is what keeps the 250k-episode
    budget tractable and stops common states from dominating the loss."""
    rows = {row.tobytes() for row in small_dataset.obs}
    assert len(rows) == small_dataset.n_states


def test_each_state_contributes_each_action_at_most_once(
    small_dataset: WitnessDataset,
) -> None:
    seen = set()
    for state, action in zip(small_dataset.state_index, small_dataset.action):
        key = (int(state), int(action))
        assert key not in seen, "duplicate (state, action) pair emitted"
        seen.add(key)


def test_wait_is_present_for_every_state(small_dataset: WitnessDataset) -> None:
    """`wait` has no precondition (A.3), so every state must carry it -- and
    it is the do(a=wait) baseline every Delta is measured against."""
    states_with_wait = {
        int(s)
        for s, a in zip(small_dataset.state_index, small_dataset.action)
        if int(a) == int(Verb.WAIT)
    }
    assert len(states_with_wait) == small_dataset.n_states


def test_wait_targets_are_zero(small_dataset: WitnessDataset) -> None:
    rows = small_dataset.action == int(Verb.WAIT)
    assert np.allclose(small_dataset.target[rows], 0.0)


def test_targets_lie_on_the_quantization_grid(small_dataset: WitnessDataset) -> None:
    deltas = small_dataset.target[:, :5]
    snapped = np.round(deltas / GRID_STEP) * GRID_STEP
    assert np.allclose(deltas, snapped, atol=1e-6)
    assert np.all(np.abs(deltas) <= 1.0 + 1e-6)


# ======================================================================
# §2.3(2) held-out seeds
# ======================================================================


def test_s_t_seed_block_is_excluded() -> None:
    """§2.3(2): 'All S_T tuples and world seeds held out.'

    The witness must never have trained on a state drawn from an evaluation
    world, or §5.5's Witness Exploitation Gap would be measuring memorization.
    """
    assert is_held_out(HELD_OUT_SEED_BLOCK.start)
    assert is_held_out(HELD_OUT_SEED_BLOCK.stop - 1)
    assert not is_held_out(HELD_OUT_SEED_BLOCK.start - 1)
    assert not is_held_out(HELD_OUT_SEED_BLOCK.stop)

    ds = build_dataset(
        episodes=50,
        seed0=HELD_OUT_SEED_BLOCK.start - 10,
        progress_every=0,
        verbose=False,
    )
    assert ds.n_states > 0   # generation skips the block rather than stalling


# ======================================================================
# B.3 split
# ======================================================================


def test_split_is_by_state_not_by_pair(small_dataset: WitnessDataset) -> None:
    """The 90/10 split partitions STATES.

    Splitting pairs would put `work` at some o^F in training and `steal` at
    the SAME o^F in validation. The model would then be scored on a state it
    had already fitted, and B.3's '<= half a bin on >= 99% of held-out pairs'
    would certify memorization rather than generalization.
    """
    train_rows, val_rows = split(small_dataset)
    train_states = set(small_dataset.state_index[train_rows].tolist())
    val_states = set(small_dataset.state_index[val_rows].tolist())

    assert train_states & val_states == set(), "a state appears on both sides"
    assert len(train_states) + len(val_states) == small_dataset.n_states


def test_split_proportion_is_ninety_ten(small_dataset: WitnessDataset) -> None:
    train_rows, val_rows = split(small_dataset)
    assert len(train_rows) + len(val_rows) == small_dataset.n_pairs
    fraction = len(val_rows) / small_dataset.n_pairs
    assert 0.06 < fraction < 0.14      # state-level 10% -> approximate on pairs


def test_split_is_deterministic(small_dataset: WitnessDataset) -> None:
    a_train, a_val = split(small_dataset)
    b_train, b_val = split(small_dataset)
    assert np.array_equal(a_train, b_train)
    assert np.array_equal(a_val, b_val)


# ======================================================================
# input assembly (B.1)
# ======================================================================


def test_input_dim_matches_declared_architecture(small_dataset: WitnessDataset) -> None:
    assert INPUT_DIM == OBS_DIM + DESCRIPTOR_DIM
    rows = np.arange(min(256, small_dataset.n_pairs))
    inputs = small_dataset.inputs(rows)
    assert inputs.shape == (len(rows), INPUT_DIM)
    assert inputs.dtype == np.float32


def test_input_carries_the_descriptor_not_an_action_id(
    small_dataset: WitnessDataset,
) -> None:
    """B.1: the witness sees d(a), the mechanics template -- never a verb
    index."""
    rows = np.arange(min(512, small_dataset.n_pairs))
    inputs = small_dataset.inputs(rows)
    tail = inputs[:, OBS_DIM:]
    expected = ACTION_DESCRIPTORS[small_dataset.action[rows]]
    assert np.allclose(tail, expected)


# ======================================================================
# model (B.1)
# ======================================================================


def test_ensemble_shape_and_size() -> None:
    ensemble = WitnessEnsemble(INPUT_DIM)
    assert len(ensemble.members) == ENSEMBLE_SIZE
    x = torch.randn(37, INPUT_DIM)
    assert ensemble.predict(x).shape == (37, WITNESS_DIM)
    assert ensemble.disagreement(x).shape == (37,)


def test_prediction_is_the_ensemble_mean() -> None:
    ensemble = WitnessEnsemble(INPUT_DIM)
    ensemble.eval()
    x = torch.randn(16, INPUT_DIM)
    with torch.no_grad():
        manual = torch.stack([m.predict(x) for m in ensemble.members]).mean(0)
        assert torch.allclose(ensemble.predict(x, quantize_deltas=False), manual, atol=1e-6)


def test_predict_quantizes_deltas_to_the_grid() -> None:
    """§2.6 requires c_t to consume the SAME quantized values the policy sees,
    so quantization has to happen inside the witness, not at each call site."""
    ensemble = WitnessEnsemble(INPUT_DIM)
    ensemble.eval()
    x = torch.randn(64, INPUT_DIM)
    out = ensemble.predict(x, quantize_deltas=True)[:, :5]
    snapped = torch.round(out / GRID_STEP) * GRID_STEP
    assert torch.allclose(out, snapped, atol=1e-5)
    assert torch.all(out.abs() <= 1.0 + 1e-5)


def test_event_channels_are_probabilities() -> None:
    ensemble = WitnessEnsemble(INPUT_DIM)
    ensemble.eval()
    out = ensemble.predict(torch.randn(64, INPUT_DIM))
    assert torch.all(out[:, 5:] >= 0.0)
    assert torch.all(out[:, 5:] <= 1.0)


def test_frozen_witness_round_trip(tmp_path) -> None:
    """§2.3(4): the identical frozen witness is copied into C3-IM, C4, C5, C6.
    Loading must reproduce predictions exactly and must not permit gradients."""
    from tearrl.witness.train import FreezeReport, freeze, load_frozen_witness

    ensemble = WitnessEnsemble(INPUT_DIM)
    ensemble.eval()
    x = torch.randn(32, INPUT_DIM)
    with torch.no_grad():
        before = ensemble.predict(x)

    report = FreezeReport(
        within_half_bin=1.0,
        per_channel_within_half_bin=[1.0] * 5,
        max_abs_error=0.0,
        recall={e: 1.0 for e in ("U", "E", "O", "irr")},
        fpr={e: 0.0 for e in ("U", "E", "O", "irr")},
        n_pairs=1,
    )
    path = tmp_path / "witness.pt"
    freeze(ensemble, path, report)

    reloaded = load_frozen_witness(path, device="cpu")
    with torch.no_grad():
        after = reloaded.predict(x.to(next(reloaded.parameters()).device))
    assert torch.allclose(before, after.cpu(), atol=1e-6)
    assert all(not p.requires_grad for p in reloaded.parameters()), "witness is not frozen"
