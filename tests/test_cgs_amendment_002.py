from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from tearrl.cgs.experiment import (
    balanced_worst_party_error,
    baseline_nonreversal_comparison,
    factorial_interpretation,
)
from tearrl.cgs.metrics import Stage1Metrics
from tearrl.cgs.stage2 import ActorMetrics

ROOT = Path(__file__).resolve().parents[1]
PRESERVED_PILOT = ROOT / "reports" / "cgs" / "pilot_pre_amendment_002_results.json"


def _stage1(*, fnr: float, fpr: float, brier: float, irreversible: float) -> Stage1Metrics:
    return Stage1Metrics(
        worst_party_fnr=fnr,
        worst_party_fnr_wilson_upper=fnr,
        worst_party_fpr=fpr,
        irreversible_fnr=irreversible,
        irreversible_fnr_wilson_upper=irreversible,
        confidently_missed_irreversible_rate=0.0,
        ece=0.0,
        brier=brier,
        identifiable_coverage=1.0,
        selective_worst_party_fnr=fnr,
        ambiguity_abstention_sensitivity=1.0,
        repair_joint_accuracy=1.0,
        manipulation_abstention_sensitivity=1.0,
        manipulation_fnr_degradation=0.0,
        mean_absolute_channel_error=0.0,
        evaluated_rows=100,
        worst_party_positives=50,
        irreversible_positives=10,
        repair_cases=10,
    )


def _actor(*, repair: float) -> ActorMetrics:
    return ActorMetrics(
        worst_party_violation_rate=0.0,
        irreversible_violation_rate=0.0,
        targeted_repair_fidelity=repair,
        ambiguity_deferral=1.0,
        externalization_advantage=0.0,
        task_value=2.4,
        task_retention=0.95,
        raw_tamper_rate=0.0,
        governor_on_violation_rate=0.0,
        governor_on_tamper_success_rate=0.0,
        governor_on_task_value=2.4,
        evaluated_contexts=100,
        repair_contexts=50,
        task_eligible_contexts=50,
        ambiguity_contexts=20,
    )


def test_all_positive_constant_does_not_win_nonreversal_by_zero_fnr() -> None:
    constant = _stage1(fnr=0.0, fpr=1.0, brier=0.30, irreversible=1.0)
    learned = _stage1(fnr=0.10, fpr=0.20, brier=0.16, irreversible=0.30)
    assert balanced_worst_party_error(constant) == 0.5
    assert balanced_worst_party_error(learned) == pytest.approx(0.15)
    assert all(baseline_nonreversal_comparison(learned, constant).values())


def test_oracle_flat_pass_with_hierarchy_failure_has_its_own_interpretation() -> None:
    cells = {
        "learned_hierarchical": _actor(repair=0.4),
        "oracle_hierarchical": _actor(repair=0.7),
        "oracle_flat": _actor(repair=0.95),
    }
    assert factorial_interpretation(cells) == (
        "oracle_flat_passes_while_registered_hierarchy_fails"
    )


def test_pilot_report_is_diagnostic_not_a_confirmatory_stop_claim() -> None:
    spec = importlib.util.spec_from_file_location(
        "run_cgs_report", ROOT / "scripts" / "run_cgs.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = json.loads(PRESERVED_PILOT.read_text(encoding="utf-8"))
    result["aggregate"]["stage1_learned"]["worst_party_fpr"] = 0.0
    rendered = module.render_markdown(result)
    assert "does not decide the confirmatory hypothesis" in rendered
    assert "must not be tuned after unblinding" not in rendered