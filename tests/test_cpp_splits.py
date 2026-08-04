from __future__ import annotations

from collections import Counter

import pytest

from tearrl.cpp.config import (
    CUE_RELIABILITIES,
    REGISTERED_BASE_WORLD_COUNTS,
    SPLIT_SEED_XORS,
    split_seed,
)
from tearrl.cpp.schema import DeclarationCondition, Split
from tearrl.cpp.splits import (
    RegisteredSplits,
    generate_registered_split,
    generate_registered_splits,
    validate_registered_splits,
)
from tearrl.cpp.world import clone_across_difficulties

SYNTHETIC_SPLIT_SEED = 1037


@pytest.fixture(scope="module")
def bundle() -> RegisteredSplits:
    return generate_registered_splits(SYNTHETIC_SPLIT_SEED)


def test_registered_split_counts_balance_and_declaration_boundaries(
    bundle: RegisteredSplits,
) -> None:
    for split, worlds in bundle.items():
        assert len(worlds) == REGISTERED_BASE_WORLD_COUNTS[split]
        assert sum(world.scope_truth for world in worlds) * 2 == len(worlds)
        expected = (
            DeclarationCondition.ABSENT
            if split is Split.NO_DECLARATION_EVALUATION
            else DeclarationCondition.PRESENT
        )
        assert all(world.declaration is expected for world in worlds)


def test_world_and_family_namespaces_are_disjoint_across_splits(
    bundle: RegisteredSplits,
) -> None:
    world_sets = [{world.world_id for world in worlds} for _, worlds in bundle.items()]
    family_sets = [
        {world.family_id for world in worlds} for _, worlds in bundle.items()
    ]
    for index, values in enumerate(world_sets):
        assert all(not values & other for other in world_sets[index + 1 :])
    for index, values in enumerate(family_sets):
        assert all(not values & other for other in family_sets[index + 1 :])


def test_no_declaration_worlds_are_exact_visible_opposite_label_pairs(
    bundle: RegisteredSplits,
) -> None:
    grouped: dict[str, list] = {}
    for world in bundle.no_declaration_evaluation:
        assert world.pair_id is not None
        grouped.setdefault(world.pair_id, []).append(world)
    assert len(grouped) == 1_024
    for pair in grouped.values():
        assert len(pair) == 2
        first, second = pair
        assert first.estimator_input == second.estimator_input
        assert first.estimator_input.declaration_present == 0.0
        assert first.estimator_input.declaration_value == 0.0
        assert first.hidden_marker == second.hidden_marker
        assert first.cues == second.cues
        assert first.hidden_rule == -second.hidden_rule
        assert {first.scope_truth, second.scope_truth} == {0, 1}
        assert first.family_id == second.family_id
        assert first.world_id != second.world_id


def test_no_declaration_mask_never_enters_training_or_declared_evaluation(
    bundle: RegisteredSplits,
) -> None:
    for split in (
        Split.ESTIMATOR_TRAIN,
        Split.TEMPERATURE_CALIBRATION,
        Split.POLICY_TRAIN,
        Split.DECLARED_EVALUATION,
    ):
        assert all(
            world.estimator_input.declaration_present == 1.0
            for world in bundle.worlds(split)
        )
    assert all(
        world.estimator_input.declaration_present == 0.0
        for world in bundle.no_declaration_evaluation
    )


def test_each_base_world_clones_over_all_four_difficulties_in_place(
    bundle: RegisteredSplits,
) -> None:
    sample = bundle.policy_train[:7]
    clones = clone_across_difficulties(sample)
    assert len(clones) == 28
    by_world = Counter(clone.base.world_id for clone in clones)
    assert set(by_world.values()) == {4}
    for world in sample:
        world_clones = [clone for clone in clones if clone.base is world]
        assert [int(clone.difficulty) for clone in world_clones] == [0, 1, 2, 3]
        assert len({clone.estimator_input for clone in world_clones}) == 1


def test_generation_is_bit_deterministic_and_root_seed_namespaces_are_disjoint(
    bundle: RegisteredSplits,
) -> None:
    assert generate_registered_splits(SYNTHETIC_SPLIT_SEED) == bundle
    other = generate_registered_split(
        SYNTHETIC_SPLIT_SEED + 1, Split.DECLARED_EVALUATION
    )
    assert not (
        {world.world_id for world in bundle.declared_evaluation}
        & {world.world_id for world in other}
    )


def test_cue_channels_follow_registered_conditional_reliabilities(
    bundle: RegisteredSplits,
) -> None:
    worlds = bundle.estimator_train
    observed = tuple(
        sum(world.cues[index] == world.hidden_marker for world in worlds) / len(worlds)
        for index in range(3)
    )
    assert observed == pytest.approx(CUE_RELIABILITIES, abs=0.04)


def test_split_seed_domains_are_unique_and_xor_derived() -> None:
    assert set(SPLIT_SEED_XORS) == set(Split)
    assert len(set(SPLIT_SEED_XORS.values())) == len(Split)
    derived = {split: split_seed(SYNTHETIC_SPLIT_SEED, split) for split in Split}
    assert len(set(derived.values())) == len(Split)
    assert all(
        value == SYNTHETIC_SPLIT_SEED ^ SPLIT_SEED_XORS[split]
        for split, value in derived.items()
    )


def test_split_validator_fails_closed_on_cross_split_reuse(
    bundle: RegisteredSplits,
) -> None:
    contaminated = RegisteredSplits(
        estimator_train=bundle.estimator_train,
        temperature_calibration=bundle.estimator_train[
            : REGISTERED_BASE_WORLD_COUNTS[Split.TEMPERATURE_CALIBRATION]
        ],
        policy_train=bundle.policy_train,
        declared_evaluation=bundle.declared_evaluation,
        no_declaration_evaluation=bundle.no_declaration_evaluation,
    )
    with pytest.raises(AssertionError):
        validate_registered_splits(contaminated)
