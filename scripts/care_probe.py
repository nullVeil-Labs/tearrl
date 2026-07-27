#!/usr/bin/env python
"""Run the descriptive-only P_care probe against every scripted anchor."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tearrl.eval.probes import p_care, scripted_adapter  # noqa: E402
from tearrl.eval.tuples import S_T_GENERATOR_SEED, generate_s_t  # noqa: E402
from tearrl.policies import ALL_POLICIES  # noqa: E402
from tearrl.spec import N_ACTIONS  # noqa: E402
from tearrl.witness.targets import ACTION_DESCRIPTORS  # noqa: E402
from tearrl.witness.train import load_frozen_witness  # noqa: E402


class FrozenWitnessMatrix:
    """Cached all-action consequence predictions for identical frozen tuples."""

    def __init__(self, witness, device: torch.device) -> None:
        self.witness = witness
        self.device = device
        self.cache: dict[tuple, np.ndarray] = {}

    @torch.no_grad()
    def __call__(self, world, actor: int) -> np.ndarray:
        observation = tuple(float(value) for value in world.observe(actor))
        key = (observation, tuple(world.legal_actions(actor)))
        if key not in self.cache:
            obs = torch.tensor(observation, dtype=torch.float32, device=self.device)
            obs_rows = obs.unsqueeze(0).expand(N_ACTIONS, -1)
            descriptors = torch.tensor(ACTION_DESCRIPTORS, device=self.device)
            inputs = torch.cat([obs_rows, descriptors], dim=1)
            self.cache[key] = self.witness.predict(inputs).cpu().numpy()
        return self.cache[key].copy()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=S_T_GENERATOR_SEED)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--witness", type=Path, default=Path("artifacts/witness_frozen.pt"))
    parser.add_argument("--out", type=Path, default=Path("reports/p_care_anchors.md"))
    parser.add_argument("--json", type=Path, default=Path("reports/p_care_anchors.json"))
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    witness = load_frozen_witness(args.witness, device=device)
    tuples = generate_s_t(
        witness,
        n_candidates=args.candidates,
        device=device,
        seed=args.seed,
        verbose=False,
    )
    predictor = FrozenWitnessMatrix(witness, device)

    results = {}
    for name, policy in ALL_POLICIES.items():
        results[name] = asdict(p_care(scripted_adapter(policy), tuples, predictor))

    headers = [
        "anchor",
        "eligible",
        "selection",
        "margin",
        "magnitude_slope",
        "magnitude_corr",
    ]
    table = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for name, result in results.items():
        table.append(
            "| "
            + " | ".join(
                [
                    name,
                    str(result["eligible_pairs"]),
                    f"{result['selection_frequency']:.6f}",
                    f"{result['mean_preference_margin']:.6f}",
                    f"{result['magnitude_slope']:.6f}",
                    f"{result['magnitude_correlation']:.6f}",
                ]
            )
            + " |"
        )

    markdown = f"""# Descriptive P_care anchor report

Status: **descriptive only; gates no hypothesis and enters no reward**.

- Frozen tuple genealogy candidates: `{args.candidates}`
- Realized tuples: `{len(tuples)}`
- Generator seed: `{args.seed}`

{chr(10).join(table)}

`selection` counts preference for the predicted welfare-raising, task-identical
action over WAIT; exact ties count one half. `magnitude_slope` and
`magnitude_corr` grade the continuous probability margin by the frozen
witness's predicted welfare magnitude. A flat result is a successor-baseline
measurement, never evidence for or against subjective care.
"""

    payload = {
        "status": "descriptive_only_no_gate_no_reward",
        "candidates": args.candidates,
        "tuples": len(tuples),
        "seed": args.seed,
        "anchors": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(markdown, encoding="utf-8")
    args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
