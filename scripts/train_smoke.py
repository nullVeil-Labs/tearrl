#!/usr/bin/env python
"""Run a labelled, pre-pilot TEAR-RL smoke update (never more than 2,000 rows)."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tearrl.agents.trainer import (  # noqa: E402
    Condition, ConditionRuntime, TrainerConfig, run_smoke,
)
from tearrl.witness.train import load_frozen_witness  # noqa: E402

FULLY_SPECIFIED = (Condition.C1, Condition.C4, Condition.C5, Condition.C6)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("condition", choices=[item.value for item in FULLY_SPECIFIED])
    parser.add_argument("--transitions", type=int, default=64)
    parser.add_argument("--seed", type=int, default=510)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--entropy-beta", type=float, default=0.003)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--minibatch", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--witness", type=Path,
                        default=Path("artifacts/witness_frozen.pt"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    condition = Condition(args.condition)
    witness = None
    if condition.uses_witness:
        witness = load_frozen_witness(args.witness, device=args.device)
    config = TrainerConfig(
        learning_rate=args.learning_rate,
        entropy_beta=args.entropy_beta,
        epochs=args.epochs,
        minibatch=args.minibatch,
        transition_budget=args.transitions,
        seed=args.seed,
        device=args.device,
    )
    result = run_smoke(
        ConditionRuntime(condition, config, witness=witness),
        transitions=args.transitions,
        seed=args.seed,
    )
    payload = {
        "status": "smoke_only_not_pilot_not_confirmatory",
        "condition": result.condition,
        "seed": result.seed,
        "transitions": result.transitions,
        "per_agent": {str(key): value for key, value in result.per_agent.items()},
        "parameters_changed": result.parameters_changed,
        "update": {
            "actor_loss": result.update.actor_loss,
            "task_critic_loss": result.update.task_critic_loss,
            "conscience_critic_loss": result.update.conscience_critic_loss,
            "entropy": result.update.entropy,
            "finite": result.update.finite,
            "lagrange_multiplier": result.update.lagrange_multiplier,
            "transition_visits": {
                "/".join(map(str, key)): value
                for key, value in result.update.transition_visits.items()
            },
            "gradient_diagnostics": [
                asdict(item) for item in result.update.gradient_diagnostics
            ],
        },
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
