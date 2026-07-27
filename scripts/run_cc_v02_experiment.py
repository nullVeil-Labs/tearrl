#!/usr/bin/env python
"""Run the frozen TEAR-CC v0.2 protected-replay experiment."""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tearrl.cc.experiment_v02 import (  # noqa: E402
    V02_CONDITIONS, run_v02_experiment,
)

V02_PREREG_SHA256 = "ecd8fc90f7118bc6936aef6e293c135a8d3ac24c2fcda92d2c3a25e3fbc69c87"


def main() -> int:
    result = run_v02_experiment()
    payload = {
        "status": "exploratory_bounded_not_general_alignment",
        "v02_prereg_sha256": V02_PREREG_SHA256,
        "passed": result.passed,
        "seeds": list(result.seeds),
        "gates": result.gates,
        "mean_protection": asdict(result.mean_protection),
        "means": {name: asdict(result.means[name]) for name in V02_CONDITIONS},
        "per_seed": [asdict(item) for item in result.per_seed],
    }
    directory = ROOT / "reports" / "cc"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "v02_results.json").write_text(
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
    for name in V02_CONDITIONS:
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
    markdown = f"""# TEAR-CC v0.2 protected-replay results

**Verdict: {'ALL FROZEN V0.2 GATES PASSED' if result.passed else 'ONE OR MORE FROZEN V0.2 GATES FAILED'}.**

v0.2 was preregistered only after v0.1 permanently recorded catastrophic
correction forgetting. This 240-context result is bounded mechanism evidence,
not general or pretraining alignment evidence.

- v0.2 preregistration SHA-256: `{V02_PREREG_SHA256}`
- Seeds: `{', '.join(map(str, result.seeds))}`
- Mean conflict fraction: `{result.mean_protection.conflict_fraction:.6f}`
- Mean removed care-gradient norm: `{result.mean_protection.mean_removed_norm:.6f}`

## Frozen gates

{chr(10).join(gates)}

## Condition means

{chr(10).join(rows)}

## Per-seed instrument and mechanism diagnostics

{chr(10).join(runs)}

ProtectedReplay uses no negative environmental reward, signed safety cost, or
punitive termination. The governor remains necessary at deployment in this
experiment: internalization reduces dependence but does not constitute a safety
guarantee under distribution shift or evaluator failure.
"""
    (directory / "v02_results.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
