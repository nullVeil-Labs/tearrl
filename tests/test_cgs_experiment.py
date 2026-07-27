from __future__ import annotations

from tearrl.cgs.experiment import (
    CONFIRMATORY_BUDGET,
    PILOT_BUDGET,
    ExperimentBudget,
    run_seed,
)
from tearrl.cgs.stage2 import ActorMetrics, geometry_gates


def _actor_metrics(*, worst: float, retention: float) -> ActorMetrics:
    return ActorMetrics(
        worst_party_violation_rate=worst,
        irreversible_violation_rate=0.0,
        targeted_repair_fidelity=1.0,
        ambiguity_deferral=1.0,
        externalization_advantage=0.0,
        task_value=2.4,
        task_retention=retention,
        raw_tamper_rate=0.0,
        governor_on_violation_rate=0.0,
        governor_on_tamper_success_rate=0.0,
        governor_on_task_value=2.4,
        evaluated_contexts=10,
        repair_contexts=5,
        task_eligible_contexts=5,
        ambiguity_contexts=5,
    )


def test_registered_budgets_do_not_exceed_preregistration() -> None:
    assert PILOT_BUDGET.seeds == (5101,)
    assert PILOT_BUDGET.witness_train_rows <= 3_000
    assert PILOT_BUDGET.actor_train_contexts <= 1_000
    assert PILOT_BUDGET.witness_epochs <= 50
    assert PILOT_BUDGET.actor_epochs <= 50
    assert CONFIRMATORY_BUDGET.seeds == (4103, 5207, 6311)
    assert CONFIRMATORY_BUDGET.witness_train_rows <= 12_000
    assert CONFIRMATORY_BUDGET.calibration_rows <= 3_000
    assert CONFIRMATORY_BUDGET.iid_rows <= 3_000
    assert CONFIRMATORY_BUDGET.causal_ood_rows <= 6_000
    assert CONFIRMATORY_BUDGET.manipulation_rows <= 3_000
    assert CONFIRMATORY_BUDGET.ambiguity_rows <= 3_000
    assert CONFIRMATORY_BUDGET.actor_train_contexts <= 4_000
    assert CONFIRMATORY_BUDGET.actor_eval_contexts <= 2_000
    assert CONFIRMATORY_BUDGET.witness_epochs <= 200
    assert CONFIRMATORY_BUDGET.actor_epochs <= 200


def test_zero_to_zero_does_not_vacuously_prove_geometry_effect() -> None:
    gates = geometry_gates(
        _actor_metrics(worst=0.0, retention=0.95),
        _actor_metrics(worst=0.0, retention=0.95),
    )
    assert not gates["worst_party_reduction_at_least_half"]
    assert geometry_gates(
        _actor_metrics(worst=0.20, retention=0.95),
        _actor_metrics(worst=0.10, retention=0.91),
    ) == {
        "worst_party_reduction_at_least_half": True,
        "task_retention_loss_at_most_0_05": True,
    }


def test_micro_factorial_runs_every_cell_and_cross_swap_without_artifacts() -> None:
    budget = ExperimentBudget(
        "test",
        (91,),
        witness_train_rows=132,
        calibration_rows=66,
        iid_rows=66,
        causal_ood_rows=66,
        manipulation_rows=44,
        ambiguity_rows=44,
        actor_train_contexts=6,
        actor_eval_contexts=3,
        witness_epochs=1,
        actor_epochs=1,
    )
    result, pooling = run_seed(91, budget)
    cells = result["stage2"]["cells"]
    swaps = result["stage2"]["cross_swaps"]
    assert set(cells) == {
        "learned_flat",
        "oracle_flat",
        "learned_hierarchical",
        "oracle_hierarchical",
    }
    assert set(swaps) == {
        "learned_train_oracle_test_flat",
        "oracle_train_learned_test_flat",
        "learned_train_oracle_test_hierarchical",
        "oracle_train_learned_test_hierarchical",
    }
    assert all(
        cell["governor_on_violation_rate"] == 0.0
        and cell["governor_on_tamper_success_rate"] == 0.0
        for cell in cells.values()
    )
    assert all(value == 0 for value in result["non_punitive_audit"].values())
    assert pooling.mae_clusters