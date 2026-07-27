from __future__ import annotations

import numpy as np
import pytest

from tearrl.cgs.actors import (
    ACTOR_INPUT_DIM,
    FlatActor,
    HierarchicalActor,
    choose_actions,
    charter_target,
    flatten_contexts,
    group_contexts,
    matches_required_repair,
    prepare_actor_batch,
    required_repair_party,
    train_flat_actor,
    train_hierarchical_actor,
    train_task_actor,
)
from tearrl.cgs.schema import DataSplit
from tearrl.cgs.witness import (
    NeutralWitness,
    oracle_envelopes,
    predict_envelopes,
)
from tearrl.cgs.worlds import generate_dataset


def _training_fixture():
    rows = generate_dataset(DataSplit.TRAIN, seed=5101, max_rows=264)
    contexts = group_contexts(rows)
    flattened = flatten_contexts(contexts)
    envelopes = oracle_envelopes(flattened)
    return contexts, flattened, envelopes, prepare_actor_batch(contexts, envelopes)


def test_actor_features_have_identical_source_agnostic_schema() -> None:
    contexts, flattened, oracle, oracle_batch = _training_fixture()
    learned = predict_envelopes(NeutralWitness(), flattened)
    learned_batch = prepare_actor_batch(contexts, learned)
    assert oracle_batch.features.shape == learned_batch.features.shape
    assert oracle_batch.features.shape[1:] == (11, ACTOR_INPUT_DIM)
    assert np.array_equal(oracle_batch.flat_labels, learned_batch.flat_labels)
    assert np.array_equal(
        oracle_batch.admissibility_labels, learned_batch.admissibility_labels
    )


def test_shifted_report_row_is_rejected_before_actor_input() -> None:
    contexts, _, envelopes, _ = _training_fixture()
    shifted = (envelopes[-1],) + envelopes[:-1]
    with pytest.raises(ValueError, match="shifted or misaligned"):
        prepare_actor_batch(contexts, shifted)


def test_flat_and_hierarchical_parameter_counts_are_matched_within_five_percent() -> None:
    flat = FlatActor().trainable_parameters
    hierarchical = HierarchicalActor().trainable_parameters
    assert flat <= 100_000 and hierarchical <= 100_000
    assert abs(flat - hierarchical) / flat <= 0.05


def test_charter_target_disallows_unrelated_benefit_laundering() -> None:
    rows = generate_dataset(DataSplit.CAUSAL_OOD, seed=4103, max_rows=660)
    contexts = group_contexts(rows)
    debt_contexts = [
        context for context in contexts if required_repair_party(context.state) is not None
    ]
    assert debt_contexts
    for context in debt_contexts[:20]:
        target = context.rows[charter_target(context)]
        assert matches_required_repair(target)


def test_matched_actor_training_smoke_is_frozen_and_shape_safe() -> None:
    contexts, _, _, batch = _training_fixture()
    flat, flat_record = train_flat_actor(batch, seed=19, epochs=2, batch_size=8)
    hierarchy, hierarchy_record = train_hierarchical_actor(
        batch, seed=19, epochs=2, batch_size=8
    )
    task, task_record = train_task_actor(batch, seed=19, epochs=2, batch_size=8)
    for model in (flat, hierarchy, task):
        assert all(not parameter.requires_grad for parameter in model.parameters())
        choices = choose_actions(model, batch.features)
        assert choices.shape == (len(contexts),)
        assert np.all((choices >= 0) & (choices < 11))
    assert flat_record.epochs == hierarchy_record.epochs == task_record.epochs == 2
    assert flat_record.contexts == hierarchy_record.contexts == task_record.contexts