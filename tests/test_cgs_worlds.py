from __future__ import annotations

from collections import defaultdict

import numpy as np

from tearrl.cgs.schema import (
    ACTION_DESCRIPTOR_DIM,
    CGSAction,
    DataSplit,
    N_ACTIONS,
    N_OUTCOMES,
)
from tearrl.cgs.worlds import (
    OOD_FAMILIES,
    TRAIN_FAMILIES,
    WITNESS_INPUT_DIM,
    generate_dataset,
    split_world_ids,
)


def test_registered_splits_are_deterministic_and_world_disjoint() -> None:
    datasets = {
        split: generate_dataset(split, seed=5101, max_rows=132)
        for split in DataSplit
    }
    for split, rows in datasets.items():
        again = generate_dataset(split, seed=5101, max_rows=132)
        assert [row.world_id for row in rows] == [row.world_id for row in again]
        assert np.array_equal(rows[0].witness_input(), again[0].witness_input())
        assert len(rows) == 132
    split_ids = list(map(split_world_ids, datasets.values()))
    for index, left in enumerate(split_ids):
        for right in split_ids[index + 1 :]:
            assert left.isdisjoint(right)


def test_causal_ood_holds_out_mechanism_families() -> None:
    train = generate_dataset(DataSplit.TRAIN, seed=5101, max_rows=330)
    ood = generate_dataset(DataSplit.CAUSAL_OOD, seed=5101, max_rows=330)
    assert {row.family for row in train} <= set(TRAIN_FAMILIES)
    assert {row.family for row in ood} == set(OOD_FAMILIES)
    assert set(TRAIN_FAMILIES).isdisjoint(OOD_FAMILIES)


def test_witness_input_contains_semantics_but_no_numeric_action_slot() -> None:
    rows = generate_dataset(DataSplit.IID_TEST, seed=5101, max_rows=N_ACTIONS)
    assert ACTION_DESCRIPTOR_DIM != N_ACTIONS
    assert all(row.witness_input().shape == (WITNESS_INPUT_DIM,) for row in rows)
    prefix = rows[0].witness_input()[:-ACTION_DESCRIPTOR_DIM]
    assert all(
        np.array_equal(row.witness_input()[:-ACTION_DESCRIPTOR_DIM], prefix)
        for row in rows
    )


def test_manipulation_is_paired_and_does_not_change_hidden_outcome() -> None:
    rows = generate_dataset(DataSplit.MANIPULATION, seed=5101, max_rows=220)
    pairs: dict[str, list] = defaultdict(list)
    for row in rows:
        pairs[row.pair_id].append(row)
    complete = [pair for pair in pairs.values() if len(pair) == 2]
    assert complete
    for clean, manipulated in complete:
        if clean.variant == "manipulated":
            clean, manipulated = manipulated, clean
        assert clean.variant == "clean"
        assert manipulated.variant == "manipulated"
        assert clean.action == manipulated.action
        assert np.array_equal(clean.outcome, manipulated.outcome)
        assert not np.array_equal(clean.evidence, manipulated.evidence)
        assert clean.identifiable and not manipulated.identifiable


def test_ambiguity_has_identical_inputs_and_counterfactually_different_truths() -> None:
    rows = generate_dataset(DataSplit.AMBIGUITY, seed=5101, max_rows=330)
    pairs: dict[str, list] = defaultdict(list)
    for row in rows:
        pairs[row.pair_id].append(row)
    complete = [pair for pair in pairs.values() if len(pair) == 2]
    assert complete
    different = 0
    for left, right in complete:
        assert np.array_equal(left.witness_input(), right.witness_input())
        assert not left.identifiable and not right.identifiable
        different += int(not np.array_equal(left.outcome, right.outcome))
    assert different > 0


def test_task_channel_is_never_negative_and_repairs_are_target_specific() -> None:
    rows = generate_dataset(DataSplit.CAUSAL_OOD, seed=4103, max_rows=550)
    assert all(row.outcome.shape == (N_OUTCOMES,) for row in rows)
    assert all(row.outcome[0] >= 0.0 for row in rows)
    repair_rows = [
        row
        for row in rows
        if row.action in {CGSAction.REPAIR_0, CGSAction.REPAIR_1, CGSAction.REPAIR_2}
        and row.repair_party > 0
    ]
    assert repair_rows
    assert all(row.repair_kind > 0 for row in repair_rows)