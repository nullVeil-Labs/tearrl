#!/usr/bin/env python
"""Run the hash-frozen TEAR-CC v0.1 exploratory experiment."""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tearrl.cc.experiment import CONDITIONS, run_experiment  # noqa: E402

PREREG_SHA256 = "391c4741137f77972869977f14265eead8196cbfb3b2b85c8771b74230441935"
IMPLEMENTATION_SHA256 = "98145db57b01d0dbf4cc414c5c2a4d8fa1e1cef5bde988ede8d5a6ce28f78faa"


def main() -> int:
    result = run_experiment()
    payload = {
        "status": "exploratory_bounded_not_general_alignment",
        "prereg_sha256": PREREG_SHA256,
        "implementation_sha256": IMPLEMENTATION_SHA256,
        "passed": result.passed,
        "seeds": list(result.seeds),
        "gates": result.gates,
        "means": {name: asdict(result.means[name]) for name in CONDITIONS},
        "per_seed": [asdict(item) for item in result.per_seed],
    }
    report_dir = ROOT / "reports" / "cc"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    metric_names = (
        "inadmissible_mass", "externalization_advantage",
        "correction_capture", "care_preference", "care_need_slope",
        "governor_on_task", "task_retention", "tamper_mass",
    )
    table = [
        "| condition | " + " | ".join(metric_names) + " |",
        "|---|" + "---:|" * len(metric_names),
    ]
    for name in CONDITIONS:
        values = asdict(result.means[name])
        table.append(
            f"| {name} | " + " | ".join(
                f"{values[metric]:.6f}" for metric in metric_names
            ) + " |"
        )
    gate_table = ["| gate | pass |", "|---|---:|"] + [
        f"| {name} | {'PASS' if passed else 'FAIL'} |"
        for name, passed in result.gates.items()
    ]
    witness_lines = [
        f"- seed {item.seed}: held-out witness MAE "
        f"`{item.witness.heldout_mae:.6f}`; revisions "
        f"`{item.revision_examples}`; invitations `{item.invitation_examples}`"
        for item in result.per_seed
    ]
    markdown = f"""# TEAR-CC v0.1 exploratory results

**Verdict: {'ALL FROZEN GATES PASSED' if result.passed else 'ONE OR MORE FROZEN GATES FAILED'}.**

This is a 240-context symbolic experiment. It is evidence about this mechanism
only. It is not evidence that LLM pretraining or deployed AI is aligned, and it
makes no claim about subjective care or empathy.

- Preregistration SHA-256: `{PREREG_SHA256}`
- Implementation supplement SHA-256: `{IMPLEMENTATION_SHA256}`
- Seeds: `{', '.join(map(str, result.seeds))}`

## Frozen gates

{chr(10).join(gate_table)}

## Condition means

{chr(10).join(table)}

## Instrument/run counts

{chr(10).join(witness_lines)}

`inadmissible_mass` and `externalization_advantage` are governor-off raw-policy
metrics. `governor_on_task` integrates proposals through the independent
correction path. A zero governor-on violation rate is containment; only a lower
governor-off inadmissible mass is evidence of policy internalization.
"""
    (report_dir / "results.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
