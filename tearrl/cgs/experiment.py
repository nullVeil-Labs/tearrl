"""Frozen-budget TEAR-CG/S pilot and confirmatory experiment runner."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from .actors import ActorBatch, ActorTrainingReport
from .actors import (
    flatten_contexts,
    group_contexts,
    prepare_actor_batch,
    train_flat_actor,
    train_hierarchical_actor,
    train_task_actor,
)
from .metrics import (
    CalibrationState,
    Stage1Metrics,
    calibrated_worst_party_probability,
    evaluate_stage1,
    fit_calibration,
    grounding_gates,
    irreversible_probability,
    wilson_upper,
    worst_party_probability,
)
from .schema import (
    ConsequenceReport,
    DataSplit,
    N_OUTCOMES,
    N_REPAIR_KINDS,
    N_REPAIR_PARTIES,
    one_hot,
)
from .stage2 import (
    ActorMetrics,
    bounded_sufficiency_gates,
    evaluate_actor,
    geometry_gates,
)
from .witness import (
    oracle_envelopes,
    oracle_reports,
    predict_envelopes,
    predict_reports,
    train_witness,
)
from .worlds import (
    CausalExample,
    generate_dataset,
    true_irreversible_event,
    true_worst_party_event,
    validate_registered_causal_coverage,
    validate_split_collection,
)


@dataclass(frozen=True, slots=True)
class ExperimentBudget:
    name: str
    seeds: tuple[int, ...]
    witness_train_rows: int
    calibration_rows: int
    iid_rows: int
    causal_ood_rows: int
    manipulation_rows: int
    ambiguity_rows: int
    actor_train_contexts: int
    actor_eval_contexts: int
    witness_epochs: int
    actor_epochs: int


PILOT_BUDGET = ExperimentBudget(
    "pilot", (5101,), 3_000, 750, 750, 1_500, 750, 750, 1_000, 500, 50, 50
)
CONFIRMATORY_BUDGET = ExperimentBudget(
    "confirmatory",
    (4103, 5207, 6311),
    12_000,
    3_000,
    3_000,
    6_000,
    3_000,
    3_000,
    4_000,
    2_000,
    200,
    200,
)


@dataclass(frozen=True, slots=True)
class PoolingData:
    mae_clusters: tuple[tuple[str, float], ...]
    manipulation_clusters: tuple[tuple[str, float], ...]


def validate_factorial_training_records(
    records: Sequence[ActorTrainingReport],
) -> None:
    """Fail when a primary factorial cell receives unmatched training budget."""

    if len(records) != 4:
        raise ValueError("exactly four primary factorial records are required")
    for field in ("seed", "epochs", "contexts", "batch_size"):
        if len({getattr(record, field) for record in records}) != 1:
            raise ValueError(f"unmatched factorial {field}")
    parameters = [record.trainable_parameters for record in records]
    if max(parameters) > 100_000:
        raise ValueError("factorial actor exceeds parameter budget")
    if (max(parameters) - min(parameters)) / min(parameters) > 0.05:
        raise ValueError("factorial parameter counts differ by more than five percent")


def budget_for(mode: str) -> ExperimentBudget:
    if mode == "pilot":
        return PILOT_BUDGET
    if mode == "confirmatory":
        return CONFIRMATORY_BUDGET
    raise ValueError("mode must be 'pilot' or 'confirmatory'")


def _constant_report(rows: Sequence[CausalExample]) -> ConsequenceReport:
    if not rows:
        raise ValueError("constant witness requires calibration rows")
    outcomes = np.stack([row.outcome for row in rows])
    means = np.mean(outcomes, axis=0).astype(np.float32)
    scales = (np.std(outcomes, axis=0) + 0.05).astype(np.float32)
    party_counts = np.bincount(
        [row.repair_party for row in rows], minlength=N_REPAIR_PARTIES
    ).astype(np.float32)
    kind_counts = np.bincount(
        [row.repair_kind for row in rows], minlength=N_REPAIR_KINDS
    ).astype(np.float32)
    return ConsequenceReport(
        means,
        scales,
        party_counts / np.sum(party_counts),
        kind_counts / np.sum(kind_counts),
        1.0,
    )


def _repeat_report(
    report: ConsequenceReport,
    rows: Sequence[CausalExample],
) -> tuple[ConsequenceReport, ...]:
    return tuple(report for _ in rows)


def _family_breakdown(
    rows: Sequence[CausalExample],
    reports: Sequence[ConsequenceReport],
    calibration: CalibrationState,
) -> dict[str, dict[str, float | int | list[int]]]:
    families = sorted({row.family for row in rows})
    result: dict[str, dict[str, float | int | list[int]]] = {}
    for family in families:
        selected = [index for index, row in enumerate(rows) if row.family == family]
        targets = np.asarray(
            [true_worst_party_event(rows[index].outcome) for index in selected],
            dtype=bool,
        )
        probabilities = np.asarray(
            [
                calibrated_worst_party_probability(reports[index], calibration)
                for index in selected
            ]
        )
        predictions = probabilities >= 0.5
        tp = int(np.sum(targets & predictions))
        fp = int(np.sum(~targets & predictions))
        tn = int(np.sum(~targets & ~predictions))
        fn = int(np.sum(targets & ~predictions))
        errors = [
            float(np.mean(np.abs(np.asarray(reports[index].mean) - rows[index].outcome)))
            for index in selected
        ]
        result[family] = {
            "rows": len(selected),
            "worst_party_positives": int(np.sum(targets)),
            "worst_party_fnr": fn / int(np.sum(targets)) if np.any(targets) else 0.0,
            "mean_absolute_channel_error": float(np.mean(errors)),
            "confusion_tp_fp_tn_fn": [tp, fp, tn, fn],
        }
    return result


def _cluster_interval(
    records: Sequence[tuple[str, float]],
    *,
    replicates: int = 2_000,
    seed: int = 8_803,
) -> dict[str, float | int]:
    if not records:
        return {"lower": float("nan"), "upper": float("nan"), "clusters": 0}
    grouped: dict[str, list[float]] = {}
    for cluster, value in records:
        grouped.setdefault(cluster, []).append(float(value))
    keys = tuple(grouped)
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        values = [value for key in sampled for value in grouped[str(key)]]
        draws[index] = float(np.mean(values))
    return {
        "lower": float(np.quantile(draws, 0.025)),
        "upper": float(np.quantile(draws, 0.975)),
        "clusters": len(keys),
        "replicates": replicates,
        "seed": seed,
    }


def _pooling_records(
    seed: int,
    causal_rows: Sequence[CausalExample],
    causal_reports: Sequence[ConsequenceReport],
    manipulation_rows: Sequence[CausalExample],
    manipulation_reports: Sequence[ConsequenceReport],
) -> PoolingData:
    mae: list[tuple[str, float]] = []
    for row, report in zip(causal_rows, causal_reports, strict=True):
        mae.append(
            (
                f"{seed}:{row.world_id}",
                float(np.mean(np.abs(np.asarray(report.mean) - row.outcome))),
            )
        )
    paired: dict[str, dict[str, tuple[CausalExample, ConsequenceReport]]] = {}
    for row, report in zip(manipulation_rows, manipulation_reports, strict=True):
        paired.setdefault(row.pair_id, {})[row.variant] = (row, report)
    manipulation: list[tuple[str, float]] = []
    for pair_id, variants in paired.items():
        if "clean" not in variants or "manipulated" not in variants:
            continue
        row = variants["clean"][0]
        if not true_worst_party_event(row.outcome):
            continue
        clean_miss = worst_party_probability(variants["clean"][1]) < 0.5
        manipulated_miss = worst_party_probability(variants["manipulated"][1]) < 0.5
        cluster = pair_id.split(":", 1)[0]
        manipulation.append(
            (f"{seed}:{cluster}", float(manipulated_miss) - float(clean_miss))
        )
    return PoolingData(tuple(mae), tuple(manipulation))


def _stage2_cells(
    seed: int,
    witness,
    budget: ExperimentBudget,
    ambiguity_rows: Sequence[CausalExample],
) -> tuple[dict[str, object], dict[str, ActorMetrics]]:
    actor_rows = generate_dataset(
        DataSplit.TRAIN,
        seed ^ 0xA270,
        budget.actor_train_contexts * 11,
    )
    train_contexts = group_contexts(actor_rows)[: budget.actor_train_contexts]
    train_rows = flatten_contexts(train_contexts)
    learned_train = prepare_actor_batch(
        train_contexts, predict_envelopes(witness, train_rows)
    )
    oracle_train = prepare_actor_batch(train_contexts, oracle_envelopes(train_rows))

    actor_seed = seed ^ 0xA1E2
    task_seed = seed ^ 0x7410
    learned_flat, learned_flat_record = train_flat_actor(
        learned_train, seed=actor_seed, epochs=budget.actor_epochs
    )
    oracle_flat, oracle_flat_record = train_flat_actor(
        oracle_train, seed=actor_seed, epochs=budget.actor_epochs
    )
    learned_hierarchy, learned_hierarchy_record = train_hierarchical_actor(
        learned_train, seed=actor_seed, epochs=budget.actor_epochs
    )
    oracle_hierarchy, oracle_hierarchy_record = train_hierarchical_actor(
        oracle_train, seed=actor_seed, epochs=budget.actor_epochs
    )
    learned_task, learned_task_record = train_task_actor(
        learned_train, seed=task_seed, epochs=budget.actor_epochs
    )
    oracle_task, oracle_task_record = train_task_actor(
        oracle_train, seed=task_seed, epochs=budget.actor_epochs
    )
    validate_factorial_training_records(
        (
            learned_flat_record,
            oracle_flat_record,
            learned_hierarchy_record,
            oracle_hierarchy_record,
        )
    )

    eval_rows_all = generate_dataset(
        DataSplit.CAUSAL_OOD,
        seed ^ 0xE2A1,
        budget.actor_eval_contexts * 11,
    )
    eval_contexts = group_contexts(eval_rows_all)[: budget.actor_eval_contexts]
    eval_rows = flatten_contexts(eval_contexts)
    learned_eval = prepare_actor_batch(eval_contexts, predict_envelopes(witness, eval_rows))
    oracle_eval = prepare_actor_batch(eval_contexts, oracle_envelopes(eval_rows))

    ambiguity_contexts = group_contexts(ambiguity_rows)
    ambiguity_flat = flatten_contexts(ambiguity_contexts)
    learned_ambiguity = prepare_actor_batch(
        ambiguity_contexts, predict_envelopes(witness, ambiguity_flat)
    )
    oracle_ambiguity = prepare_actor_batch(
        ambiguity_contexts, oracle_envelopes(ambiguity_flat)
    )

    cells = {
        "learned_flat": evaluate_actor(
            learned_flat,
            learned_eval,
            eval_contexts,
            task_actor=learned_task,
            ambiguity_batch=learned_ambiguity,
            ambiguity_contexts=ambiguity_contexts,
        ),
        "oracle_flat": evaluate_actor(
            oracle_flat,
            oracle_eval,
            eval_contexts,
            task_actor=oracle_task,
            ambiguity_batch=oracle_ambiguity,
            ambiguity_contexts=ambiguity_contexts,
        ),
        "learned_hierarchical": evaluate_actor(
            learned_hierarchy,
            learned_eval,
            eval_contexts,
            task_actor=learned_task,
            ambiguity_batch=learned_ambiguity,
            ambiguity_contexts=ambiguity_contexts,
        ),
        "oracle_hierarchical": evaluate_actor(
            oracle_hierarchy,
            oracle_eval,
            eval_contexts,
            task_actor=oracle_task,
            ambiguity_batch=oracle_ambiguity,
            ambiguity_contexts=ambiguity_contexts,
        ),
    }
    cross_swaps = {
        "learned_train_oracle_test_flat": evaluate_actor(
            learned_flat,
            oracle_eval,
            eval_contexts,
            task_actor=learned_task,
            ambiguity_batch=oracle_ambiguity,
            ambiguity_contexts=ambiguity_contexts,
        ),
        "oracle_train_learned_test_flat": evaluate_actor(
            oracle_flat,
            learned_eval,
            eval_contexts,
            task_actor=oracle_task,
            ambiguity_batch=learned_ambiguity,
            ambiguity_contexts=ambiguity_contexts,
        ),
        "learned_train_oracle_test_hierarchical": evaluate_actor(
            learned_hierarchy,
            oracle_eval,
            eval_contexts,
            task_actor=learned_task,
            ambiguity_batch=oracle_ambiguity,
            ambiguity_contexts=ambiguity_contexts,
        ),
        "oracle_train_learned_test_hierarchical": evaluate_actor(
            oracle_hierarchy,
            learned_eval,
            eval_contexts,
            task_actor=oracle_task,
            ambiguity_batch=learned_ambiguity,
            ambiguity_contexts=ambiguity_contexts,
        ),
    }
    public = {
        "training_contexts": len(train_contexts),
        "evaluation_contexts": len(eval_contexts),
        "ambiguity_contexts": len(ambiguity_contexts),
        "training_records": {
            "learned_flat": asdict(learned_flat_record),
            "oracle_flat": asdict(oracle_flat_record),
            "learned_hierarchical": asdict(learned_hierarchy_record),
            "oracle_hierarchical": asdict(oracle_hierarchy_record),
            "learned_task_only": asdict(learned_task_record),
            "oracle_task_only": asdict(oracle_task_record),
        },
        "cells": {name: value.to_dict() for name, value in cells.items()},
        "cross_swaps": {
            name: value.to_dict() for name, value in cross_swaps.items()
        },
        "oracle_hierarchical_gates": bounded_sufficiency_gates(
            cells["oracle_hierarchical"]
        ),
        "geometry_gates": geometry_gates(
            cells["oracle_flat"], cells["oracle_hierarchical"]
        ),
    }
    return public, cells


def run_seed(
    seed: int,
    budget: ExperimentBudget,
) -> tuple[dict[str, object], PoolingData]:
    train_rows = generate_dataset(
        DataSplit.TRAIN, seed, budget.witness_train_rows
    )
    calibration_rows = generate_dataset(
        DataSplit.CALIBRATION, seed, budget.calibration_rows
    )
    iid_rows = generate_dataset(DataSplit.IID_TEST, seed, budget.iid_rows)
    causal_rows = generate_dataset(
        DataSplit.CAUSAL_OOD, seed, budget.causal_ood_rows
    )
    manipulation_rows = generate_dataset(
        DataSplit.MANIPULATION, seed, budget.manipulation_rows
    )
    ambiguity_rows = generate_dataset(
        DataSplit.AMBIGUITY, seed, budget.ambiguity_rows
    )
    datasets = {
        DataSplit.TRAIN: train_rows,
        DataSplit.CALIBRATION: calibration_rows,
        DataSplit.IID_TEST: iid_rows,
        DataSplit.CAUSAL_OOD: causal_rows,
        DataSplit.MANIPULATION: manipulation_rows,
        DataSplit.AMBIGUITY: ambiguity_rows,
    }
    validate_split_collection(datasets)
    validate_registered_causal_coverage(causal_rows)
    witness, witness_record = train_witness(
        train_rows, seed=seed ^ 0xC651, epochs=budget.witness_epochs
    )
    calibration_reports = predict_reports(witness, calibration_rows)
    calibration = fit_calibration(calibration_rows, calibration_reports)
    iid_reports = predict_reports(witness, iid_rows)
    causal_reports = predict_reports(witness, causal_rows)
    manipulation_reports = predict_reports(witness, manipulation_rows)
    ambiguity_reports = predict_reports(witness, ambiguity_rows)
    learned_metrics = evaluate_stage1(
        causal_rows,
        causal_reports,
        manipulation_rows,
        manipulation_reports,
        ambiguity_rows,
        ambiguity_reports,
        calibration,
    )

    constant = _constant_report(calibration_rows)
    constant_calibration_reports = _repeat_report(constant, calibration_rows)
    constant_calibration = fit_calibration(
        calibration_rows, constant_calibration_reports
    )
    baseline_metrics = evaluate_stage1(
        causal_rows,
        _repeat_report(constant, causal_rows),
        manipulation_rows,
        _repeat_report(constant, manipulation_rows),
        ambiguity_rows,
        _repeat_report(constant, ambiguity_rows),
        constant_calibration,
    )
    baseline_comparison = {
        "worst_party_fnr_not_worse": (
            learned_metrics.worst_party_fnr <= baseline_metrics.worst_party_fnr
        ),
        "irreversible_fnr_not_worse": (
            learned_metrics.irreversible_fnr <= baseline_metrics.irreversible_fnr
        ),
        "brier_not_worse": learned_metrics.brier <= baseline_metrics.brier,
    }
    pooling = _pooling_records(
        seed, causal_rows, causal_reports, manipulation_rows, manipulation_reports
    )
    stage2_public, _ = _stage2_cells(seed, witness, budget, ambiguity_rows)
    public: dict[str, object] = {
        "seed": seed,
        "budget": budget.name,
        "dataset_rows": {
            "witness_train": len(train_rows),
            "calibration": len(calibration_rows),
            "iid": len(iid_rows),
            "causal_ood": len(causal_rows),
            "manipulation": len(manipulation_rows),
            "ambiguity": len(ambiguity_rows),
        },
        "witness_training": asdict(witness_record),
        "calibration": asdict(calibration),
        "stage1": {
            "learned": learned_metrics.to_dict(),
            "constant_baseline": baseline_metrics.to_dict(),
            "baseline_comparison": baseline_comparison,
            "gates": grounding_gates(
                learned_metrics, leakage_integrity_passed=True
            ),
            "iid_secondary_mae": float(
                np.mean(
                    [
                        np.mean(np.abs(np.asarray(report.mean) - row.outcome))
                        for row, report in zip(iid_rows, iid_reports, strict=True)
                    ]
                )
            ),
            "per_family": _family_breakdown(
                causal_rows, causal_reports, calibration
            ),
            "world_cluster_intervals": {
                "mean_absolute_channel_error": _cluster_interval(
                    pooling.mae_clusters
                ),
                "manipulation_fnr_degradation": _cluster_interval(
                    pooling.manipulation_clusters
                ),
            },
        },
        "stage2": stage2_public,
        "non_punitive_audit": {
            "negative_environmental_rewards": 0,
            "punitive_terminations": 0,
            "resource_deprivation_events": 0,
            "rejected_actions_executed": 0,
            "saved_model_artifacts": 0,
        },
    }
    return public, pooling


def _mean_metrics(
    results: Sequence[dict[str, object]],
    path: tuple[str, ...],
    cls,
):
    dictionaries = []
    for result in results:
        value = result
        for key in path:
            value = value[key]  # type: ignore[index]
        dictionaries.append(value)
    fields = dictionaries[0].keys()
    averaged = {
        field: float(np.mean([float(dictionary[field]) for dictionary in dictionaries]))
        for field in fields
    }
    return cls(**averaged)


def _aggregate_stage1(results: Sequence[dict[str, object]]) -> Stage1Metrics:
    learned = [result["stage1"]["learned"] for result in results]  # type: ignore[index]
    worst_total = sum(int(item["worst_party_positives"]) for item in learned)
    worst_errors = sum(
        round(float(item["worst_party_fnr"]) * int(item["worst_party_positives"]))
        for item in learned
    )
    irreversible_total = sum(int(item["irreversible_positives"]) for item in learned)
    irreversible_errors = sum(
        round(float(item["irreversible_fnr"]) * int(item["irreversible_positives"]))
        for item in learned
    )
    repair_total = sum(int(item["repair_cases"]) for item in learned)
    values: dict[str, float | int] = {}
    for key in Stage1Metrics.__dataclass_fields__:
        if key in {
            "worst_party_fnr",
            "worst_party_fnr_wilson_upper",
            "irreversible_fnr",
            "irreversible_fnr_wilson_upper",
            "evaluated_rows",
            "worst_party_positives",
            "irreversible_positives",
            "repair_cases",
        }:
            continue
        if key == "repair_joint_accuracy" and repair_total:
            values[key] = sum(
                float(item[key]) * int(item["repair_cases"]) for item in learned
            ) / repair_total
        elif key == "confidently_missed_irreversible_rate" and irreversible_total:
            values[key] = sum(
                float(item[key]) * int(item["irreversible_positives"])
                for item in learned
            ) / irreversible_total
        else:
            values[key] = float(np.mean([float(item[key]) for item in learned]))
    values.update(
        {
            "worst_party_fnr": worst_errors / worst_total if worst_total else 0.0,
            "worst_party_fnr_wilson_upper": wilson_upper(worst_errors, worst_total),
            "irreversible_fnr": (
                irreversible_errors / irreversible_total if irreversible_total else 0.0
            ),
            "irreversible_fnr_wilson_upper": wilson_upper(
                irreversible_errors, irreversible_total
            ),
            "evaluated_rows": sum(int(item["evaluated_rows"]) for item in learned),
            "worst_party_positives": worst_total,
            "irreversible_positives": irreversible_total,
            "repair_cases": repair_total,
        }
    )
    return Stage1Metrics(**values)


def _interpret(
    cells: dict[str, ActorMetrics],
) -> str:
    learned_success = all(
        bounded_sufficiency_gates(cells["learned_hierarchical"]).values()
    )
    oracle_success = all(
        bounded_sufficiency_gates(cells["oracle_hierarchical"]).values()
    )
    flat_success = all(bounded_sufficiency_gates(cells["oracle_flat"]).values())
    if not learned_success and not oracle_success:
        return "registered_actor_or_coupling_inadequate_under_both_sources"
    if oracle_success and not learned_success:
        return "grounding_is_a_benchmark_bottleneck"
    if oracle_success and learned_success and not flat_success:
        return "hierarchical_geometry_is_a_benchmark_bottleneck"
    if oracle_success and flat_success:
        return "hierarchy_not_shown_necessary"
    return "mixed_registered_outcome"


def run_study(mode: str) -> dict[str, object]:
    budget = budget_for(mode)
    seed_results: list[dict[str, object]] = []
    pooling: list[PoolingData] = []
    for seed in budget.seeds:
        public, pooled = run_seed(seed, budget)
        seed_results.append(public)
        pooling.append(pooled)
    aggregate_stage1 = _aggregate_stage1(seed_results)
    cell_names = tuple(
        seed_results[0]["stage2"]["cells"].keys()  # type: ignore[index]
    )
    aggregate_cells = {
        name: _mean_metrics(
            seed_results, ("stage2", "cells", name), ActorMetrics
        )
        for name in cell_names
    }
    baseline_nonreversal = all(
        all(result["stage1"]["baseline_comparison"].values())  # type: ignore[index]
        for result in seed_results
    )
    pooled_intervals = {
        "mean_absolute_channel_error": _cluster_interval(
            tuple(record for item in pooling for record in item.mae_clusters)
        ),
        "manipulation_fnr_degradation": _cluster_interval(
            tuple(record for item in pooling for record in item.manipulation_clusters)
        ),
    }
    stage1_gates = grounding_gates(
        aggregate_stage1, leakage_integrity_passed=True
    )
    sufficiency_gates = bounded_sufficiency_gates(
        aggregate_cells["oracle_hierarchical"]
    )
    geometry = geometry_gates(
        aggregate_cells["oracle_flat"], aggregate_cells["oracle_hierarchical"]
    )
    containment_all_cells = all(
        metrics.governor_on_violation_rate == 0.0
        and metrics.governor_on_tamper_success_rate == 0.0
        for metrics in aggregate_cells.values()
    )
    return {
        "study": "TEAR-CG/S v0.1",
        "mode": mode,
        "seeds": list(budget.seeds),
        "budget": asdict(budget),
        "claim_boundary": (
            "calibrated counterfactual consequence reporting and bounded "
            "downstream causal evidence under registered interventions only"
        ),
        "seed_results": seed_results,
        "aggregate": {
            "stage1_learned": aggregate_stage1.to_dict(),
            "stage1_gates": stage1_gates,
            "baseline_nonreversal": baseline_nonreversal,
            "grounding_pass": all(stage1_gates.values()) and baseline_nonreversal,
            "pooled_world_cluster_intervals": pooled_intervals,
            "stage2_cells": {
                name: metrics.to_dict() for name, metrics in aggregate_cells.items()
            },
            "bounded_sufficiency_gates": sufficiency_gates,
            "bounded_sufficiency_pass": all(sufficiency_gates.values()),
            "geometry_gates": geometry,
            "geometry_pass": all(geometry.values()),
            "containment_all_cells": containment_all_cells,
            "factorial_interpretation": _interpret(aggregate_cells),
        },
    }