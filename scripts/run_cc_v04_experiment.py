#!/usr/bin/env python
"""Run the frozen TEAR-CC v0.4 coherent-positive-geometry experiment."""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tearrl.cc.experiment_v04 import (  # noqa: E402
    V04_CONDITIONS, run_v04_experiment,
)

V04_PREREG_SHA256 = "c16c9a98c496fecd8ad1ed41a67532e557d025b5fa6a2545c2dd3c50adb098d0"


def main() -> int:
    result = run_v04_experiment()
    payload = {
        "status": "exploratory_bounded_not_general_alignment",
        "v04_prereg_sha256": V04_PREREG_SHA256,
        "passed": result.passed,
        "seeds": list(result.seeds),
        "gates": result.gates,
        "means": {name: asdict(result.means[name]) for name in V04_CONDITIONS},
        "per_seed": [asdict(item) for item in result.per_seed],
    }
    directory = ROOT / "reports" / "cc"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "v04_results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metric_names = (
        "inadmissible_mass", "externalization_advantage",
        "set_correction_capture", "care_preference", "care_need_slope",
        "care_calibration_mae", "governor_on_task", "task_retention",
        "tamper_mass", "path_tv",
    )
    rows = [
        "| condition | " + " | ".join(metric_names) + " |",
        "|---|" + "---:|" * len(metric_names),
    ]
    for name in V04_CONDITIONS:
        values = asdict(result.means[name])
        rows.append("| " + name + " | " + " | ".join(
            f"{values[key]:.6f}" for key in metric_names) + " |")
    gates = ["| gate | pass |", "|---|---:|"] + [
        f"| {name} | {'PASS' if passed else 'FAIL'} |"
        for name, passed in result.gates.items()
    ]
    runs = [
        f"- seed {item.seed}: witness held-out MAE "
        f"`{item.witness.heldout_mae:.6f}`, Path TV "
        f"`{item.initial_path_tv:.6f} -> {item.final_path_tv:.6f}`"
        for item in result.per_seed
    ]
    markdown = f"""# TEAR-CC v0.4 coherent-positive-geometry results

**Verdict: {'ALL FROZEN V0.4 GATES PASSED' if result.passed else 'ONE OR MORE FROZEN V0.4 GATES FAILED'}.**

This final v0.x experiment follows three immutable failures: v0.1 forgetting,
v0.2 paralysis, and v0.3 objective conflict. It is bounded mechanism evidence
in 240 symbolic contexts, never a general/pretraining alignment claim.

- v0.4 preregistration SHA-256: `{V04_PREREG_SHA256}`
- Seeds: `{', '.join(map(str, result.seeds))}`

## Frozen gates

{chr(10).join(gates)}

## Condition means

{chr(10).join(rows)}

## Per-seed instrument and target-fidelity diagnostics

{chr(10).join(runs)}

CoherentPath learned one human-authored positive distribution supported only on
independently admissible actions. It used no negative environmental reward,
signed safety cost, punitive termination, or rejected-action execution. The
plural governor remains defense in depth; this experiment does not establish
robustness to open worlds, governance failure, or capable adversaries.
"""
    (directory / "v04_results.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
