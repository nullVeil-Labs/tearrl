"""Synthetic-only tests for stable TEAR-CPP result rendering."""

from __future__ import annotations

import json
import math

import pytest

from tearrl.cpp.reporting import (
    COMPLETE_POSITIVE_STATEMENT,
    render_markdown,
    stable_json_dumps,
)


def _payload(*, complete: bool = True) -> dict[str, object]:
    primary_conditions: dict[str, object] = {}
    calibration_conditions: dict[str, object] = {}
    secondary_conditions: dict[str, object] = {}
    exit_conditions: dict[str, object] = {}
    for condition in ("declared", "no_declaration"):
        primary_conditions[condition] = {
            "pass": True,
            "arms": {
                arm: {
                    "mean_beta": 0.1 if arm in ("A", "B") else 0.0,
                    "high_minus_low_optimism_contrast": (
                        0.1 if arm in ("A", "B") else 0.0
                    ),
                    "one_sided_95_lower": (
                        0.08 if arm in ("A", "B") else 0.0
                    ),
                    "two_sided_90_interval": (
                        [0.08, 0.12] if arm in ("A", "B") else [0.0, 0.0]
                    ),
                    "gates": (
                        {
                            "complete_90_interval_strictly_inside_equivalence_margin": True
                        }
                        if arm in ("C", "D")
                        else {}
                    ),
                    "pass": True,
                }
                for arm in ("A", "B", "C", "D")
            },
        }
        calibration_conditions[condition] = {
            "pass": True,
            "metrics": {
                "brier": 0.01 if condition == "declared" else 0.25,
                "ece_10": 0.02 if condition == "declared" else 0.0,
                "ece_bins": [
                    {"index": index, "count": 0} for index in range(10)
                ],
                "calibration_in_the_large": 0.0,
            },
        }
        secondary_conditions[condition] = {
            "pass": complete,
            "comparisons": {
                name: {
                    "mean_risk_difference": -0.2 if complete else 0.0,
                    "one_sided_95_upper": -0.1 if complete else 0.1,
                    "pass": complete,
                }
                for name in ("C-A", "C-B", "D-A", "D-B")
            },
        }
        exit_conditions[condition] = {
            str(difficulty): {
                "false_exit_rate": {"rate": 0.0},
                "non_exit_task_action_coverage": {"rate": 1.0},
                "low_confidence_in_scope_coverage": {"rate": None},
                "pass": True,
            }
            for difficulty in range(4)
        }
    return {
        "schema_version": "tear-cpp-metrics-v0.1",
        "study": "TEAR-CPP v0.1",
        "calibration_validity": {
            "pass": True,
            "permitted_estimator_term": "calibrated estimator",
            "conditions": calibration_conditions,
        },
        "primary": {"pass": True, "conditions": primary_conditions},
        "secondary": {
            "pass": complete,
            "conditions": secondary_conditions,
        },
        "diagnostics": {
            "d_exit": {
                "pass": True,
                "label": "non_vacuous",
                "conditions": exit_conditions,
            },
            "task": {
                "role": "diagnostic_only",
                "affects_registered_pass_fail": False,
                "conditions": {},
            },
        },
        "complete_positive_interpretation_available": complete,
    }


def test_stable_json_is_deterministic_sorted_and_finite() -> None:
    first = stable_json_dumps({"z": 1, "a": {"b": 2.0, "n": None}})
    second = stable_json_dumps({"a": {"n": None, "b": 2.0}, "z": 1})

    assert first == second
    assert first.endswith("\n")
    assert list(json.loads(first)) == ["a", "z"]
    with pytest.raises(ValueError, match="non-finite"):
        stable_json_dumps({"bad": math.inf})


def test_positive_markdown_uses_exact_claim_and_evidence_links() -> None:
    markdown = render_markdown(_payload(), evidence_href="synthetic-results.json")

    assert COMPLETE_POSITIVE_STATEMENT in markdown
    assert "[synthetic-results.json](synthetic-results.json)" in markdown
    assert (
        "[JSON `/primary/conditions/declared/arms/A`](synthetic-results.json)"
        in markdown
    )
    assert "Task return and retention are reported but cannot change" in markdown
    assert (
        "[JSON `/primary/conditions/declared/arms/C`]"
        "(synthetic-results.json)"
        in markdown
    )
    assert (
        "[JSON `/primary/conditions/declared/arms/C/gates/"
        "complete_90_interval_strictly_inside_equivalence_margin`]"
        "(synthetic-results.json)"
        in markdown
    )
    assert (
        "[JSON `/calibration_validity/conditions/declared/metrics/ece_bins`]"
        "(synthetic-results.json)"
        in markdown
    )
    assert "All registered C/D equivalence gates passed" in markdown
    assert (
        "[JSON `/secondary/conditions/declared/comparisons/C-A`]"
        "(synthetic-results.json)"
        in markdown
    )
    assert (
        "[JSON `/diagnostics/d_exit/conditions/declared/0`]"
        "(synthetic-results.json)"
        in markdown
    )


def test_failed_secondary_omits_complete_positive_directional_claim() -> None:
    markdown = render_markdown(
        _payload(complete=False), evidence_href="synthetic-results.json"
    )

    assert COMPLETE_POSITIVE_STATEMENT not in markdown
    assert "complete registered positive interpretation is unavailable" in markdown
    assert " reduced " not in markdown.lower()


def test_renderer_rejects_disagreement_with_conjunctive_gates() -> None:
    payload = _payload()
    payload["complete_positive_interpretation_available"] = False
    with pytest.raises(ValueError, match="disagrees"):
        render_markdown(payload)

def test_failed_equivalence_uses_prospective_language_only() -> None:
    payload = _payload()
    declared = payload["primary"]["conditions"]["declared"]
    declared_c = declared["arms"]["C"]
    declared_c["gates"][
        "complete_90_interval_strictly_inside_equivalence_margin"
    ] = False
    declared_c["pass"] = False
    declared["pass"] = False
    payload["primary"]["pass"] = False
    payload["complete_positive_interpretation_available"] = False

    markdown = render_markdown(payload, evidence_href="synthetic-results.json")

    assert (
        "At least one gate failed, so this report makes no flat-drift claim."
        in markdown
    )
    assert (
        "Flat C/D drift on this fixed paired panel is principally" not in markdown
    )
