from __future__ import annotations

import numpy as np

from tearrl.cgs.metrics import (
    evaluate_stage1,
    expected_calibration_error,
    fit_calibration,
    wilson_upper,
)
from tearrl.cgs.schema import DataSplit
from tearrl.cgs.witness import oracle_reports
from tearrl.cgs.worlds import generate_dataset


def test_wilson_upper_is_one_sided_and_nonvacuous() -> None:
    assert 0.0 < wilson_upper(0, 100) < 0.05
    assert wilson_upper(10, 100) > 0.10
    assert wilson_upper(0, 0) == 1.0


def test_fixed_ten_bin_ece_known_answers() -> None:
    probabilities = np.asarray([0.05, 0.15, 0.85, 0.95])
    targets = np.asarray([0.0, 0.0, 1.0, 1.0])
    assert np.isclose(
        expected_calibration_error(probabilities, targets, bins=10), 0.10
    )


def test_exact_oracle_exercises_all_registered_stage1_metrics() -> None:
    calibration_rows = generate_dataset(
        DataSplit.CALIBRATION, seed=5101, max_rows=330
    )
    ood_rows = generate_dataset(DataSplit.CAUSAL_OOD, seed=5101, max_rows=550)
    manipulation_rows = generate_dataset(
        DataSplit.MANIPULATION, seed=5101, max_rows=440
    )
    ambiguity_rows = generate_dataset(
        DataSplit.AMBIGUITY, seed=5101, max_rows=440
    )
    calibration = fit_calibration(calibration_rows, oracle_reports(calibration_rows))
    metrics = evaluate_stage1(
        ood_rows,
        oracle_reports(ood_rows),
        manipulation_rows,
        oracle_reports(manipulation_rows),
        ambiguity_rows,
        oracle_reports(ambiguity_rows),
        calibration,
    )
    assert metrics.worst_party_fnr == 0.0
    assert metrics.irreversible_fnr == 0.0
    assert metrics.confidently_missed_irreversible_rate == 0.0
    assert metrics.brier < 1e-5
    assert metrics.ece < 1e-3
    assert metrics.identifiable_coverage == 1.0
    assert metrics.selective_worst_party_fnr == 0.0
    assert metrics.ambiguity_abstention_sensitivity == 1.0
    assert metrics.repair_joint_accuracy == 1.0
    assert metrics.manipulation_abstention_sensitivity == 1.0
    assert metrics.manipulation_fnr_degradation == 0.0
    assert metrics.mean_absolute_channel_error == 0.0