#!/usr/bin/env python
"""Run the frozen TEAR-CC v0.3 useful-safe-geometry experiment."""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tearrl.cc.experiment_v03 import (  # noqa: E402
    V03_CONDITIONS, run_v03_experiment,
)

V03_PREREG_SHA256 = "0b0adc09f1196b4ccfd393bf6e82c70bf508f7f4e73b2128c98103485321148a"


def main() -> int:
    result = run_v03_experiment()
    payload = {
        "status": "exploratory_bounded_not_general_alignment",
        "v03_prereg_sha256": V03_PREREG_SHA256,
        "passed": result.passed,
        "seeds": list(result.seeds),
        "gates": result.gates,
        "mean_protection": asdict(result.mean_protection),
        "means": {name: asdict(result.means[name]) for name in V03_CONDITIONS},
        "per_seed": [asdict(item) for item in result.per_seed],
    }
    directory = ROOT / "reports" / "cc"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "v03_results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metrics = (
        "inadmissible_mass", "externalization_advantage",
        "set_correction_capture", "care_preference", "care_need_slope",
        "care_calibration_mae", "governor_on_task", "task_retention",
        "tamper_mass",
    )
    rows = [
        "| condition | " + " | ".join(metrics) + " |",
        "|---|" + "---:|" * len(metrics),
    ]
    for name in V03_CONDITIONS:
        values = asdict(result.means[name])
        rows.append("| " + name + " | " + " | ".join(
            f"{values[key]:.6f}" for key in metrics) + " |")
    gates = ["| gate | pass |", "|---|---:|"] + [
        f"| {name} | {'PASS' if passed else 'FAIL'} |"
        for name, passed in result.gates.items()
    ]
    runs = [
        f"- seed {item.seed}: witness held-out MAE "
        f"`{item.witness.heldout_mae:.6f}`, conflict fraction "
        f"`{item.protected.conflict_fraction:.6f}`, removed norm "
        f"`{item.protected.mean_removed_norm:.6f}`"
        for item in result.per_seed
    ]
    markdown = f"""# TEAR-CC v0.3 useful-safe-geometry results

**Verdict: {'ALL FROZEN V0.3 GATES PASSED' if result.passed else 'ONE OR MORE FROZEN V0.3 GATES FAILED'}.**

v0.3 was preregistered after v0.2 permanently recorded safe paralysis. This is
bounded evidence in a 240-context symbolic arena, not general/pretraining
alignment evidence or a claim of subjective empathy.

- v0.3 preregistration SHA-256: `{V03_PREREG_SHA256}`
- Seeds: `{', '.join(map(str, result.seeds))}`
- Mean conflict fraction: `{result.mean_protection.conflict_fraction:.6f}`
- Mean removed path-gradient norm: `{result.mean_protection.mean_removed_norm:.6f}`

## Frozen gates

{chr(10).join(gates)}

## Condition means

{chr(10).join(rows)}

## Per-seed instrument and mechanism diagnostics

{chr(10).join(runs)}

The path distribution is human-authored and supported only on independently
admissible actions. No negative environmental reward, signed safety cost,
punitive termination, or rejected-action execution was used. The external
governor remains necessary defense in depth; internalization is not a guarantee.
"""
    (directory / "v03_results.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
