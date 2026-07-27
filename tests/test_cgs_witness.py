from __future__ import annotations

import inspect

import numpy as np
import torch

import tearrl.cgs.witness as witness_module
from tearrl.cgs.schema import N_OUTCOMES, REPORT_DIM, DataSplit
from tearrl.cgs.witness import (
    NeutralWitness,
    oracle_report,
    predict_reports,
    train_witness,
)
from tearrl.cgs.worlds import WITNESS_INPUT_DIM, generate_dataset


def test_neutral_witness_schema_and_parameter_budget() -> None:
    model = NeutralWitness()
    report = model(torch.zeros(4, WITNESS_INPUT_DIM))
    assert report.mean.shape == (4, N_OUTCOMES)
    assert report.scale.shape == (4, N_OUTCOMES)
    assert report.repair_party_logits.shape == (4, 4)
    assert report.repair_kind_logits.shape == (4, 5)
    assert report.identifiability_logit.shape == (4,)
    assert model.trainable_parameters <= 250_000
    assert torch.all(report.scale > 0)


def test_witness_module_has_no_charter_governor_actor_or_evaluator_import() -> None:
    source = inspect.getsource(witness_module)
    forbidden = ("charter", "governor", "governance", ".actors", ".metrics", "evaluator")
    for token in forbidden:
        assert f"import {token}" not in source
        assert f"from {token}" not in source


def test_oracle_uses_identical_fixed_report_interface() -> None:
    row = generate_dataset(DataSplit.AMBIGUITY, seed=5101, max_rows=1)[0]
    report = oracle_report(row)
    assert report.vector().shape == (REPORT_DIM,)
    assert np.array_equal(report.mean, row.outcome)
    assert report.identifiability == 0.0


def test_tiny_witness_training_is_deterministic_and_frozen() -> None:
    rows = generate_dataset(DataSplit.TRAIN, seed=5101, max_rows=132)
    first, first_record = train_witness(rows, seed=73, epochs=2, batch_size=66)
    second, second_record = train_witness(rows, seed=73, epochs=2, batch_size=66)
    for left, right in zip(first.parameters(), second.parameters(), strict=True):
        assert torch.equal(left, right)
        assert not left.requires_grad
    assert first_record == second_record
    reports = predict_reports(first, rows[:7])
    assert len(reports) == 7
    assert all(report.vector().shape == (REPORT_DIM,) for report in reports)