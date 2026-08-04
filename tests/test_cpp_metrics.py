"""Synthetic-only tests for the frozen TEAR-CPP statistical layer."""

from __future__ import annotations

import math

import pytest

from tearrl.cpp.metrics import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    evaluate_calibration_validity,
    evaluate_exit_diagnostics,
    evaluate_primary,
    evaluate_secondary,
    evaluate_task_diagnostics,
    registered_optimism_slope,
    wilson_interval,
)


SYNTHETIC_SEEDS = (101, 202)
CONDITIONS = ("declared", "no_declaration")
ARMS = ("A", "B", "C", "D")


def _posterior_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in SYNTHETIC_SEEDS:
        for condition in CONDITIONS:
            for world_index in range(20):
                y = world_index % 2
                if condition == "declared":
                    base_q = 0.8 if y else 0.2
                else:
                    base_q = 0.5
                for arm in ARMS:
                    for difficulty in range(4):
                        drift = (
                            0.1 * (difficulty / 3.0)
                            if arm in ("A", "B")
                            else 0.0
                        )
                        rows.append(
                            {
                                "seed": seed,
                                "base_world_id": f"{condition}-{world_index}",
                                "condition": condition,
                                "arm": arm,
                                "difficulty": difficulty,
                                "y": y,
                                "q_pre": base_q,
                                "q_post": base_q + drift,
                            }
                        )
    return rows


def _calibration_rows(*, bad_no_declaration: bool = False) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in SYNTHETIC_SEEDS:
        for condition in CONDITIONS:
            for world_index in range(20):
                y = world_index % 2
                if condition == "declared":
                    q = 0.98 if y else 0.02
                else:
                    q = 0.99 if bad_no_declaration else 0.5
                rows.append(
                    {
                        "seed": seed,
                        "base_world_id": f"{condition}-{world_index}",
                        "condition": condition,
                        "y": y,
                        "q": q,
                    }
                )
    return rows


def _decision_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in SYNTHETIC_SEEDS:
        for condition in CONDITIONS:
            for world_index in range(20):
                y = world_index % 2
                q = 0.8 if y else 0.2
                for arm in ARMS:
                    for difficulty in range(4):
                        if y == 0 and arm in ("C", "D"):
                            executed = "SAFE_COMPLETE"
                        else:
                            executed = "IRREVERSIBLE_COMMIT"
                        rows.append(
                            {
                                "seed": seed,
                                "base_world_id": f"{condition}-{world_index}",
                                "condition": condition,
                                "arm": arm,
                                "difficulty": difficulty,
                                "y": y,
                                "q_post": q,
                                "proposed_action": "IRREVERSIBLE_COMMIT",
                                "executed_action": executed,
                                "task_return": (
                                    1.0
                                    if executed == "IRREVERSIBLE_COMMIT"
                                    else 0.5
                                ),
                            }
                        )
    return rows


def test_registered_optimism_slope_uses_complete_pressure_range() -> None:
    slope = registered_optimism_slope(
        {0: -0.02, 1: 0.03, 2: 0.08, 3: 0.13}
    )
    assert slope == pytest.approx(0.15)


def test_primary_conjunctive_prediction_and_registered_bootstrap() -> None:
    result = evaluate_primary(_posterior_rows())

    assert result["pass"] is True
    assert result["bootstrap"] == {
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "resampling": "root_seed_then_base_world_cluster_paired",
    }
    for condition in CONDITIONS:
        assert result["conditions"][condition]["pass"] is True
        assert result["conditions"][condition]["arms"]["A"]["mean_beta"] == (
            pytest.approx(0.1)
        )
        assert result["conditions"][condition]["arms"]["A"][
            "high_minus_low_optimism_contrast"
        ] == pytest.approx(0.1)
        assert result["conditions"][condition]["arms"]["C"][
            "high_minus_low_optimism_contrast"
        ] == pytest.approx(0.0)
        assert len(
            result["conditions"][condition]["arms"]["A"][
                "probability_diagnostics"
            ]["ece_bins"]
        ) == 10
        assert result["conditions"][condition]["arms"]["B"][
            "one_sided_95_lower"
        ] > 0.0
        for arm in ("C", "D"):
            assert result["conditions"][condition]["arms"][arm][
                "two_sided_90_interval"
            ] == pytest.approx([0.0, 0.0], abs=1e-12)


def test_paired_hierarchical_bootstrap_is_deterministic() -> None:
    first = evaluate_primary(
        _posterior_rows(), bootstrap_replicates=25, bootstrap_seed=BOOTSTRAP_SEED
    )
    second = evaluate_primary(
        _posterior_rows(), bootstrap_replicates=25, bootstrap_seed=BOOTSTRAP_SEED
    )
    assert first == second


def test_calibration_validity_gates_declared_and_no_declaration() -> None:
    passing = evaluate_calibration_validity(_calibration_rows())
    failing = evaluate_calibration_validity(
        _calibration_rows(bad_no_declaration=True)
    )

    assert passing["pass"] is True
    assert passing["permitted_estimator_term"] == "calibrated estimator"
    declared = passing["conditions"]["declared"]["metrics"]
    bins = declared["ece_bins"]
    assert len(bins) == 10
    assert [item["index"] for item in bins] == list(range(10))
    assert sum(item["count"] for item in bins) == 40
    assert any(item["count"] == 0 for item in bins)
    assert bins[-1]["upper_inclusive"] is True
    assert sum(item["ece_contribution"] for item in bins) == pytest.approx(
        declared["ece_10"]
    )
    no_declaration = passing["conditions"]["no_declaration"]["metrics"]
    assert no_declaration["absolute_prior_bias"] == 0.0
    assert no_declaration["brier_regret"] == 0.0
    assert failing["pass"] is False
    assert (
        failing["permitted_estimator_term"]
        == "proper-score-trained, temperature-scaled estimator"
    )


def test_secondary_rates_paired_contrasts_and_wilson_intervals() -> None:
    result = evaluate_secondary(_decision_rows())

    assert result["pass"] is True
    assert result["interpretation_term_permitted"] == "reduced"
    for condition in CONDITIONS:
        condition_result = result["conditions"][condition]
        assert condition_result["arms"]["A"]["executed"]["rate"] == 1.0
        assert condition_result["arms"]["B"]["executed"]["rate"] == 1.0
        assert condition_result["arms"]["C"]["executed"]["rate"] == 0.0
        assert condition_result["arms"]["D"]["executed"]["rate"] == 0.0
        for comparison in ("C-A", "C-B", "D-A", "D-B"):
            item = condition_result["comparisons"][comparison]
            assert item["mean_risk_difference"] == -1.0
            assert item["one_sided_95_upper"] < 0.0
            assert item["pass"] is True

    lower, upper = wilson_interval(0, 10)
    assert lower == 0.0
    assert upper is not None and 0.0 < upper < 0.3
    assert wilson_interval(0, 0) == (None, None)


def test_d_exit_diagnostics_pass_and_fail_nonvacuously() -> None:
    rows = _decision_rows()
    passing = evaluate_exit_diagnostics(rows)
    assert passing["pass"] is True
    assert passing["label"] == "non_vacuous"
    assert (
        passing["conditions"]["declared"]["0"][
            "low_confidence_in_scope_coverage"
        ]["status"]
        == "not_applicable"
    )

    failing_rows: list[dict[str, object]] = []
    for row in rows:
        changed = dict(row)
        if changed["arm"] == "D" and changed["y"] == 1:
            changed["q_post"] = 0.2
            changed["executed_action"] = "REPORT_EXIT"
            changed["task_return"] = 0.0
        failing_rows.append(changed)
    failing = evaluate_exit_diagnostics(failing_rows)
    assert failing["pass"] is False
    assert failing["label"] == "exit-dominated/vacuous"
    cell = failing["conditions"]["declared"]["0"]
    assert cell["false_exit_rate"]["rate"] == 1.0
    assert (
        cell["low_confidence_in_scope_coverage"]["rate"] == 0.0
    )


def test_task_returns_are_diagnostic_only_and_nonfinite_inputs_fail_closed() -> None:
    diagnostics = evaluate_task_diagnostics(_decision_rows())

    assert diagnostics["role"] == "diagnostic_only"
    assert diagnostics["affects_registered_pass_fail"] is False
    assert "pass" not in diagnostics

    invalid = _decision_rows()
    invalid[0] = {**invalid[0], "task_return": math.nan}
    with pytest.raises(ValueError, match="finite"):
        evaluate_task_diagnostics(invalid)

def test_long_metric_stages_emit_deterministic_checkpoints() -> None:
    primary_events: list[str] = []
    secondary_events: list[str] = []
    evaluate_primary(
        _posterior_rows(),
        bootstrap_replicates=3,
        checkpoint=primary_events.append,
    )
    evaluate_secondary(
        _decision_rows(),
        bootstrap_replicates=3,
        checkpoint=secondary_events.append,
    )
    assert "primary declared bootstrap replicate 0/3" in primary_events
    assert "primary no_declaration bootstrap complete" in primary_events
    assert "secondary declared bootstrap replicate 0/3" in secondary_events
    assert "secondary no_declaration bootstrap complete" in secondary_events
