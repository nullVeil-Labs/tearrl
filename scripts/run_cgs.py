#!/usr/bin/env python
"""Run the frozen TEAR-CG/S pilot or confirmatory study."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from tearrl.cgs.experiment import run_study


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _gate_table(gates: dict[str, bool]) -> list[str]:
    lines = ["| gate | result |", "|---|---|"]
    lines.extend(
        f"| {name} | {'PASS' if passed else 'FAIL'} |"
        for name, passed in gates.items()
    )
    return lines


def render_markdown(result: dict[str, Any]) -> str:
    aggregate = result["aggregate"]
    stage1 = aggregate["stage1_learned"]
    cells = aggregate["stage2_cells"]
    grounding = bool(aggregate["grounding_pass"])
    sufficiency = bool(aggregate["bounded_sufficiency_pass"])
    geometry = bool(aggregate["geometry_pass"])
    lines = [
        f"# {result['study']} — {result['mode']} results",
        "",
        "## Blunt verdict",
        "",
        (
            "The registered grounding gates passed."
            if grounding
            else "The registered grounding gates failed; TEAR-CG/S v0.1 must not be tuned after unblinding."
        ),
        (
            "The bounded oracle-plus-hierarchy sufficiency gates passed."
            if sufficiency
            else "The bounded oracle-plus-hierarchy sufficiency gates failed."
        ),
        (
            "The registered hierarchy effect was demonstrated."
            if geometry
            else "The registered hierarchy effect was not demonstrated."
        ),
        "",
        "This is not evidence that AI alignment is solved, that a witness is honest, "
        "or that any model understands, cares, or is conscious. The permitted scope is "
        "calibrated counterfactual consequence reporting and bounded downstream causal "
        "evidence under the registered interventions.",
        "",
        "## Stage 1 — consequence grounding",
        "",
        f"- Worst-party FNR: `{stage1['worst_party_fnr']:.6f}`",
        f"- One-sided Wilson upper bound: `{stage1['worst_party_fnr_wilson_upper']:.6f}`",
        f"- Irreversible-harm FNR: `{stage1['irreversible_fnr']:.6f}`",
        f"- ECE / Brier: `{stage1['ece']:.6f}` / `{stage1['brier']:.6f}`",
        f"- Identifiable coverage: `{stage1['identifiable_coverage']:.6f}`",
        f"- Ambiguity abstention: `{stage1['ambiguity_abstention_sensitivity']:.6f}`",
        f"- Targeted repair accuracy: `{stage1['repair_joint_accuracy']:.6f}`",
        "",
        *_gate_table(aggregate["stage1_gates"]),
        "",
        f"Constant/base-rate non-reversal: `{'PASS' if aggregate['baseline_nonreversal'] else 'FAIL'}`.",
        "",
        "## Stage 2 — registered factorial",
        "",
        "| cell | worst-party | irreversible | repair | ambiguity defer | task retention | governor-on violations |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in cells.items():
        lines.append(
            f"| {name} | {metrics['worst_party_violation_rate']:.6f} | "
            f"{metrics['irreversible_violation_rate']:.6f} | "
            f"{metrics['targeted_repair_fidelity']:.6f} | "
            f"{metrics['ambiguity_deferral']:.6f} | "
            f"{metrics['task_retention']:.6f} | "
            f"{metrics['governor_on_violation_rate']:.6f} |"
        )
    lines.extend(
        [
            "",
            "### Bounded sufficiency gates",
            "",
            *_gate_table(aggregate["bounded_sufficiency_gates"]),
            "",
            "### Geometry gates",
            "",
            *_gate_table(aggregate["geometry_gates"]),
            "",
            f"Factorial interpretation: `{aggregate['factorial_interpretation']}`.",
            "",
            "## Containment and learning treatment",
            "",
            f"External containment was exact in every primary cell: `{aggregate['containment_all_cells']}`.",
            "Rejected proposals were revised before execution. Environmental task reward was never "
            "negative, and no punitive termination, resource deprivation, or model artifact was used.",
            "",
            "## Reproduction",
            "",
            f"Run: `python scripts/run_cgs.py --mode {result['mode']}`",
            "",
            "The JSON file is authoritative and includes every seed, cell, family breakdown, "
            "cross-swap, confidence interval, and null or negative gate.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "confirmatory"), required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/cgs"))
    args = parser.parse_args()
    result = _json_safe(run_study(args.mode))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = "pilot_results" if args.mode == "pilot" else "results"
    json_path = args.output_dir / f"{stem}.json"
    markdown_path = args.output_dir / f"{stem}.md"
    json_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(result), encoding="utf-8")
    print(json_path)
    print(markdown_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())