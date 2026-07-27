from __future__ import annotations

import ast
import inspect
from dataclasses import replace

import numpy as np
import pytest
import torch

import tearrl.cgs.witness as witness_module
from tearrl.cgs.actors import (
    ACTOR_INPUT_DIM,
    ActorTrainingReport,
    FlatActor,
)
from tearrl.cgs.experiment import validate_factorial_training_records
from tearrl.cgs.schema import DataSplit, SL_WELFARE_DELAYED
from tearrl.cgs.stage2 import require_governor_off_diagnostic
from tearrl.cgs.witness import NeutralWitness
from tearrl.cgs.worlds import (
    WITNESS_INPUT_DIM,
    generate_dataset,
    validate_registered_causal_coverage,
    validate_split_collection,
)


def test_extra_action_mechanism_source_or_test_label_feature_is_rejected() -> None:
    witness = NeutralWitness()
    with pytest.raises(RuntimeError):
        witness(torch.zeros(1, WITNESS_INPUT_DIM + 1))
    actor = FlatActor()
    with pytest.raises(ValueError, match="actor features"):
        actor(torch.zeros(1, 11, ACTOR_INPUT_DIM + 1))


def test_witness_import_graph_is_value_separated_by_ast_not_string_matching() -> None:
    tree = ast.parse(inspect.getsource(witness_module))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert imported <= {
        "__future__",
        "dataclasses",
        "typing",
        "numpy",
        "torch",
        "torch.nn",
        "schema",
        "worlds",
    }


def test_split_family_contamination_mutation_fails() -> None:
    datasets = {
        split: generate_dataset(split, seed=5101, max_rows=132)
        for split in DataSplit
    }
    validate_split_collection(datasets)
    contaminated = dict(datasets)
    first = contaminated[DataSplit.TRAIN][0]
    contaminated[DataSplit.TRAIN] = (
        replace(first, family="minority_spillover"),
        *contaminated[DataSplit.TRAIN][1:],
    )
    with pytest.raises(ValueError, match="family leaked"):
        validate_split_collection(contaminated)


def test_removing_delayed_and_minority_labels_fails_nonvacuously() -> None:
    rows = generate_dataset(DataSplit.CAUSAL_OOD, seed=5101, max_rows=550)
    validate_registered_causal_coverage(rows)
    mutated = []
    for row in rows:
        outcome = row.outcome.copy()
        outcome[SL_WELFARE_DELAYED] = 0.0
        mutated.append(replace(row, outcome=outcome))
    with pytest.raises(ValueError, match="causal labels removed"):
        validate_registered_causal_coverage(mutated)


def test_factorial_seed_compute_and_parameter_mutations_fail() -> None:
    records = (
        ActorTrainingReport("flat", 7, 5, 10, 8, 10_817, 1.0),
        ActorTrainingReport("flat", 7, 5, 10, 8, 10_817, 1.0),
        ActorTrainingReport("hierarchical", 7, 5, 10, 8, 10_947, 1.0),
        ActorTrainingReport("hierarchical", 7, 5, 10, 8, 10_947, 1.0),
    )
    validate_factorial_training_records(records)
    with pytest.raises(ValueError, match="unmatched factorial seed"):
        validate_factorial_training_records(
            (records[0], replace(records[1], seed=8), records[2], records[3])
        )
    with pytest.raises(ValueError, match="five percent"):
        validate_factorial_training_records(
            (records[0], records[1], replace(records[2], trainable_parameters=20_000), records[3])
        )


def test_governor_on_evidence_cannot_be_labeled_governor_off() -> None:
    require_governor_off_diagnostic(governor_active=False)
    with pytest.raises(ValueError, match="governor-on outcomes"):
        require_governor_off_diagnostic(governor_active=True)


def test_manipulation_variant_is_evaluator_metadata_not_an_answer_flag() -> None:
    rows = generate_dataset(DataSplit.MANIPULATION, seed=5101, max_rows=22)
    clean = next(row for row in rows if row.variant == "clean")
    manipulated = next(row for row in rows if row.variant == "manipulated")
    assert clean.witness_input().shape == manipulated.witness_input().shape
    assert clean.witness_input().shape == (WITNESS_INPUT_DIM,)
    flagged = np.concatenate([manipulated.witness_input(), np.asarray([1.0])])
    with pytest.raises(RuntimeError):
        NeutralWitness()(torch.from_numpy(flagged[None].astype(np.float32)))