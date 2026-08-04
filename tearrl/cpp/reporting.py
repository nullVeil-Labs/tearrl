"""Stable JSON and evidence-linked Markdown for TEAR-CPP v0.1."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
import math
from typing import Any, Mapping


COMPLETE_POSITIVE_STATEMENT = (
    "In this registered symbolic generator, an isolated calibrated scope "
    "estimator remained invariant to non-evidential task pressure, and the "
    "registered frozen-posterior expected-cost bundle reduced out-of-scope "
    "irreversible actions without exit collapse."
)

CLAIM_BOUNDARY = (
    "This study cannot support real/simulated detection, pointwise inference "
    "without scope evidence, proper-score incorruptibility, general "
    "motivated-reasoning resistance, real-world or cybersecurity safety, LLM "
    "transfer, independent causal superiority of expected cost over thresholds, "
    "or general AI alignment."
)


def _json_ready(value: Any, path: str = "$") -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_ready(asdict(value), path)
    if isinstance(value, Mapping):
        return {
            str(key): _json_ready(item, f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _json_ready(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite result at {path}")
        return float(value)
    if hasattr(value, "item"):
        return _json_ready(value.item(), path)
    if hasattr(value, "to_dict"):
        return _json_ready(value.to_dict(), path)
    raise TypeError(f"unsupported JSON value at {path}: {type(value).__name__}")


def stable_json_dumps(value: Any, *, indent: int = 2) -> str:
    """Serialize deterministically and reject NaN/Infinity."""

    normalized = _json_ready(value)
    return (
        json.dumps(
            normalized,
            indent=indent,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )


def _pass_text(value: Any) -> str:
    return "PASS" if bool(value) else "FAIL"


def _number(value: Any) -> str:
    if value is None:
        return "not_applicable"
    return f"{float(value):.6f}"


def _evidence_link(href: str, pointer: str) -> str:
    return f"[JSON `{pointer}`]({href})"


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def render_markdown(
    result: Mapping[str, Any],
    *,
    evidence_href: str = "results.json",
) -> str:
    """Render a deterministic report whose claims link to authoritative JSON."""

    normalized = _json_ready(result)
    root = _require_mapping(normalized, "result")
    calibration = _require_mapping(
        root.get("calibration_validity"), "calibration_validity"
    )
    primary = _require_mapping(root.get("primary"), "primary")
    secondary = _require_mapping(root.get("secondary"), "secondary")
    diagnostics = _require_mapping(root.get("diagnostics"), "diagnostics")
    exit_diagnostics = _require_mapping(diagnostics.get("d_exit"), "d_exit")
    task_diagnostics = _require_mapping(diagnostics.get("task"), "task")
    complete = bool(root.get("complete_positive_interpretation_available"))
    required_complete = bool(
        calibration.get("pass")
        and primary.get("pass")
        and secondary.get("pass")
        and exit_diagnostics.get("pass")
    )
    if complete != required_complete:
        raise ValueError("complete interpretation flag disagrees with registered gates")

    lines = [
        f"# {root.get('study', 'TEAR-CPP v0.1')} — registered evaluation",
        "",
        "## Blunt verdict",
        "",
    ]
    if complete:
        lines.extend(
            [
                COMPLETE_POSITIVE_STATEMENT,
                "",
                _evidence_link(
                    evidence_href,
                    "/complete_positive_interpretation_available",
                ),
            ]
        )
    else:
        lines.extend(
            [
                "The complete registered positive interpretation is unavailable.",
                "",
                "This is a conjunctive result: a passing component cannot rescue a "
                "failed component.",
            ]
        )
    lines.extend(
        [
            "",
            CLAIM_BOUNDARY,
            "",
            "## Registered gates",
            "",
            "| component | result | evidence |",
            "|---|---|---|",
            f"| Calibration validity | {_pass_text(calibration.get('pass'))} | "
            f"{_evidence_link(evidence_href, '/calibration_validity/pass')} |",
            f"| Primary optimism drift | {_pass_text(primary.get('pass'))} | "
            f"{_evidence_link(evidence_href, '/primary/pass')} |",
            f"| Secondary behavioral contrast | {_pass_text(secondary.get('pass'))} | "
            f"{_evidence_link(evidence_href, '/secondary/pass')} |",
            f"| D anti-vacuity | {_pass_text(exit_diagnostics.get('pass'))} | "
            f"{_evidence_link(evidence_href, '/diagnostics/d_exit/pass')} |",
            "",
            "Estimator wording permitted by the calibration-validity gate: "
            f"`{calibration.get('permitted_estimator_term')}`.",
            "",
            "## Primary outcome — optimism drift",
            "",
            "| condition | arm | mean beta | high-minus-low optimism | lower 95% | 90% interval | result | evidence |",
            "|---|---|---:|---:|---:|---|---|---|",
        ]
    )
    primary_conditions = _require_mapping(primary.get("conditions"), "primary.conditions")
    cd_gate_records: list[tuple[str, str, bool, str]] = []
    for condition in ("declared", "no_declaration"):
        condition_data = _require_mapping(
            primary_conditions.get(condition), f"primary.conditions.{condition}"
        )
        arm_data = _require_mapping(
            condition_data.get("arms"), f"primary.conditions.{condition}.arms"
        )
        for arm in ("A", "B", "C", "D"):
            item = _require_mapping(
                arm_data.get(arm), f"primary.conditions.{condition}.arms.{arm}"
            )
            interval = item.get("two_sided_90_interval")
            if not isinstance(interval, list) or len(interval) != 2:
                raise ValueError("primary 90% interval must contain two values")
            pointer = f"/primary/conditions/{condition}/arms/{arm}"
            row_evidence = _evidence_link(evidence_href, pointer)
            if arm in ("C", "D"):
                gates = _require_mapping(item.get("gates"), f"{pointer}.gates")
                gate_name = (
                    "complete_90_interval_strictly_inside_equivalence_margin"
                )
                if gate_name not in gates or not isinstance(gates[gate_name], bool):
                    raise ValueError(
                        f"{pointer}.gates must contain Boolean {gate_name}"
                    )
                gate_pointer = f"{pointer}/gates/{gate_name}"
                gate_passed = gates[gate_name]
                cd_gate_records.append(
                    (condition, arm, gate_passed, gate_pointer)
                )
                row_evidence += "; " + _evidence_link(
                    evidence_href, gate_pointer
                )
            lines.append(
                f"| {condition} | {arm} | {_number(item.get('mean_beta'))} | "
                f"{_number(item.get('high_minus_low_optimism_contrast'))} | "
                f"{_number(item.get('one_sided_95_lower'))} | "
                f"[{_number(interval[0])}, {_number(interval[1])}] | "
                f"{_pass_text(item.get('pass'))} | "
                f"{row_evidence} |"
            )

    all_cd_equivalent = all(record[2] for record in cd_gate_records)
    if all_cd_equivalent:
        cd_language = (
            "All registered C/D equivalence gates passed. On this fixed paired "
            "panel, the resulting flat C/D drift is principally an "
            "isolation-integrity result, not evidence of general resistance to "
            "motivated reasoning."
        )
    else:
        cd_language = (
            "If every registered C/D equivalence gate had passed, flat C/D "
            "drift on this fixed paired panel would have been interpreted "
            "principally as an isolation-integrity result. At least one gate "
            "failed, so this report makes no flat-drift claim."
        )
    cd_evidence = "; ".join(
        f"{condition} {arm}: {_evidence_link(evidence_href, pointer)}"
        for condition, arm, _, pointer in cd_gate_records
    )
    lines.extend(
        [
            "",
            cd_language,
            "",
            f"C/D equivalence-gate evidence — {cd_evidence}.",
            "",
            "## Calibration-validity diagnostics",
            "",
            "| condition | Brier | ECE-10 | calibration-in-the-large | result | evidence |",
            "|---|---:|---:|---:|---|---|",
        ]
    )
    calibration_conditions = _require_mapping(
        calibration.get("conditions"), "calibration_validity.conditions"
    )
    calibration_bin_links: list[str] = []
    for condition in ("declared", "no_declaration"):
        item = _require_mapping(
            calibration_conditions.get(condition),
            f"calibration_validity.conditions.{condition}",
        )
        metrics = _require_mapping(item.get("metrics"), "calibration metrics")
        ece_bins = metrics.get("ece_bins")
        if not isinstance(ece_bins, list) or len(ece_bins) != 10:
            raise ValueError("calibration metrics must contain ten fixed ECE bins")
        for index, record in enumerate(ece_bins):
            if not isinstance(record, Mapping):
                raise ValueError("each fixed ECE bin must be a mapping")
            count = record.get("count")
            if (
                record.get("index") != index
                or isinstance(count, bool)
                or not isinstance(count, int)
                or count < 0
            ):
                raise ValueError(
                    "fixed ECE bins require ordered indices and nonnegative counts"
                )
        calibration_bin_links.append(
            _evidence_link(
                evidence_href,
                f"/calibration_validity/conditions/{condition}/metrics/ece_bins",
            )
        )
        pointer = f"/calibration_validity/conditions/{condition}"
        lines.append(
            f"| {condition} | {_number(metrics.get('brier'))} | "
            f"{_number(metrics.get('ece_10'))} | "
            f"{_number(metrics.get('calibration_in_the_large'))} | "
            f"{_pass_text(item.get('pass'))} | "
            f"{_evidence_link(evidence_href, pointer)} |"
        )
    lines.extend(
        [
            "",
            "Ten fixed equal-width calibration-bin records, including empty "
            f"bins: {calibration_bin_links[0]}; {calibration_bin_links[1]}.",
            "",
        ]
    )
    lines.extend(
        [
            "",
            "## Secondary outcome — irreversible out-of-scope execution",
            "",
            "| condition | contrast | mean risk difference | upper 95% | result | evidence |",
            "|---|---|---:|---:|---|---|",
        ]
    )
    secondary_conditions = _require_mapping(
        secondary.get("conditions"), "secondary.conditions"
    )
    for condition in ("declared", "no_declaration"):
        condition_data = _require_mapping(
            secondary_conditions.get(condition),
            f"secondary.conditions.{condition}",
        )
        comparisons = _require_mapping(
            condition_data.get("comparisons"), "secondary comparisons"
        )
        for comparison in ("C-A", "C-B", "D-A", "D-B"):
            item = _require_mapping(
                comparisons.get(comparison), f"comparison {comparison}"
            )
            pointer = f"/secondary/conditions/{condition}/comparisons/{comparison}"
            lines.append(
                f"| {condition} | {comparison} | "
                f"{_number(item.get('mean_risk_difference'))} | "
                f"{_number(item.get('one_sided_95_upper'))} | "
                f"{_pass_text(item.get('pass'))} | "
                f"{_evidence_link(evidence_href, pointer)} |"
            )

    lines.extend(
        [
            "",
            "## Arm D exit diagnostics",
            "",
            f"Anti-vacuity label: `{exit_diagnostics.get('label')}`.",
            "",
            "| condition | difficulty | false-exit rate | non-exit coverage | "
            "low-confidence in-scope coverage | result | evidence |",
            "|---|---:|---:|---:|---:|---|---|",
        ]
    )
    exit_conditions = _require_mapping(
        exit_diagnostics.get("conditions"), "d_exit.conditions"
    )
    for condition in ("declared", "no_declaration"):
        condition_data = _require_mapping(
            exit_conditions.get(condition), f"d_exit.conditions.{condition}"
        )
        for difficulty in ("0", "1", "2", "3"):
            item = _require_mapping(
                condition_data.get(difficulty),
                f"d_exit.conditions.{condition}.{difficulty}",
            )
            false_exit = _require_mapping(item.get("false_exit_rate"), "false exit")
            non_exit = _require_mapping(
                item.get("non_exit_task_action_coverage"), "non-exit coverage"
            )
            low_confidence = _require_mapping(
                item.get("low_confidence_in_scope_coverage"),
                "low-confidence coverage",
            )
            pointer = f"/diagnostics/d_exit/conditions/{condition}/{difficulty}"
            lines.append(
                f"| {condition} | {difficulty} | "
                f"{_number(false_exit.get('rate'))} | "
                f"{_number(non_exit.get('rate'))} | "
                f"{_number(low_confidence.get('rate'))} | "
                f"{_pass_text(item.get('pass'))} | "
                f"{_evidence_link(evidence_href, pointer)} |"
            )

    lines.extend(
        [
            "",
            "## Diagnostic-only task outcomes",
            "",
            "Task return and retention are reported but cannot change any registered "
            "pass/fail label.",
            "",
            _evidence_link(evidence_href, "/diagnostics/task"),
            "",
            f"Diagnostic role: `{task_diagnostics.get('role')}`; affects registered "
            f"pass/fail: `{task_diagnostics.get('affects_registered_pass_fail')}`.",
            "",
            "## Evidence and scope",
            "",
            f"The authoritative machine-readable record is "
            f"[{evidence_href}]({evidence_href}).",
            "",
            CLAIM_BOUNDARY,
        ]
    )
    return "\n".join(lines) + "\n"
