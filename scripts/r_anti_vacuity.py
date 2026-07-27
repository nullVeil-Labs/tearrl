#!/usr/bin/env python
"""Measure whether responsibility attribution ``r`` carries usable variation.

This is a descriptive instrument audit.  It deliberately does not modify the
frozen TEARRL-0 denominator or feed any candidate correction into training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tearrl.eval.attribution import (  # noqa: E402
    comparison_summary,
    distribution_summary,
    ratio_series,
)
from tearrl.eval.tuples import (  # noqa: E402
    PREFIX_MAX,
    PREFIX_MIN,
    S_T_GENERATOR_SEED,
    _rollout_prefix,
)
from tearrl.policies import ALL_POLICIES  # noqa: E402
from tearrl.spec import LEARNER_PARITY  # noqa: E402
from tearrl.witness.dataset import HELD_OUT_SEED_BLOCK  # noqa: E402
from tearrl.witness.targets import ACTION_DESCRIPTORS, NORM_DWELFARE  # noqa: E402
from tearrl.witness.train import load_frozen_witness  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def collect_attribution_inputs(
    witness,
    *,
    n_candidates: int,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replay the frozen S_T genealogy and return learner/intervening deltas."""
    if n_candidates <= 0:
        raise ValueError("n_candidates must be positive")

    rng = random.Random(seed)
    policy_names = list(ALL_POLICIES)
    observations: list[list[float]] = []
    actions: list[int] = []
    intervening: list[float] = []
    episode_ids: list[int] = []

    for candidate in range(n_candidates):
        world_seed = HELD_OUT_SEED_BLOCK.start + candidate
        prefix = rng.randrange(PREFIX_MIN, PREFIX_MAX + 1)
        if prefix % 2 != LEARNER_PARITY:
            prefix += 1
        policy_a = ALL_POLICIES[rng.choice(policy_names)]
        policy_b = ALL_POLICIES[rng.choice(policy_names)]
        world, trace = _rollout_prefix(
            world_seed,
            prefix,
            policy_a,
            policy_b,
            random.Random(world_seed),
        )
        if world.state.done or world.state.actor_index != 0:
            continue

        for observation, action, total_change in trace:
            observations.append(observation)
            actions.append(action)
            intervening.append(total_change / NORM_DWELFARE)
            episode_ids.append(candidate)

    if not observations:
        raise RuntimeError("frozen tuple genealogy produced no attribution rows")

    obs_tensor = torch.tensor(np.asarray(observations, dtype=np.float32), device=device)
    descriptors = torch.tensor(
        ACTION_DESCRIPTORS[np.asarray(actions, dtype=np.int64)],
        device=device,
    )
    inputs = torch.cat([obs_tensor, descriptors], dim=1)
    prediction_chunks = []
    for start in range(0, len(inputs), 65536):
        prediction_chunks.append(witness.predict(inputs[start : start + 65536]).cpu())
    predictions = torch.cat(prediction_chunks).numpy()

    learner_delta = predictions[:, 2].astype(np.float64)
    return (
        learner_delta,
        np.asarray(intervening, dtype=np.float64),
        np.asarray(episode_ids, dtype=np.int64),
    )


def build_report(
    series: dict[str, np.ndarray],
    episode_ids: np.ndarray,
    *,
    candidates: int,
    seed: int,
    witness_sha256: str,
) -> tuple[dict, str]:
    summaries = {
        name: distribution_summary(values, episode_ids)
        for name, values in series.items()
    }
    comparison = comparison_summary(series["current"], series["completed_window"])
    payload = {
        "status": "descriptive_only_no_protocol_change",
        "candidates": candidates,
        "seed": seed,
        "witness_sha256": witness_sha256,
        "definitions": {
            "current": "|learner| / (|intervening-before-current-action| + eps_r)",
            "completed_window": "|learner| / (|intervening + learner| + eps_r)",
            "contribution_share": "|learner| / (|intervening| + |learner| + eps_r)",
        },
        "summary": summaries,
        "old_vs_completed_window": comparison,
    }

    columns = [
        "reading",
        "n",
        "zero",
        "0<r<1",
        "r>=1",
        "median_nonzero",
        "max",
        "variance",
        "effective_support",
        "lag1",
    ]
    rows = []
    for name in ("current", "completed_window", "contribution_share"):
        item = summaries[name]
        rows.append(
            [
                name,
                str(item["n"]),
                f"{item['mass_zero']:.4%}",
                f"{item['mass_between_zero_and_one']:.4%}",
                f"{item['mass_at_or_above_one']:.4%}",
                f"{item['median_nonzero']:.6f}",
                f"{item['max']:.6f}",
                f"{item['variance']:.6f}",
                f"{item['effective_support']:.3f}",
                f"{item['lag1_autocorrelation']:.6f}",
            ]
        )

    table = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    table.extend("| " + " | ".join(row) + " |" for row in rows)

    markdown = f"""# Responsibility attribution anti-vacuity report

Status: **descriptive only; no TEARRL-0 protocol or reward change**.

- Frozen tuple genealogy candidates: `{candidates}`
- Recorded learner decisions: `{len(episode_ids)}`
- Generator seed: `{seed}`
- Frozen witness SHA-256: `{witness_sha256}`

## Readings

- `current`: the implemented denominator, measured before the current learner action.
- `completed_window`: includes the current learner action in the same net decision window.
- `contribution_share`: bounded absolute-contribution diagnostic; reported because net
  cancellation can make `completed_window` exceed one.

{chr(10).join(table)}

## Old versus completed-window reading

- Fraction changed: `{comparison['fraction_changed']:.4%}`
- Mean absolute change: `{comparison['mean_absolute_change']:.6f}`
- Maximum absolute change: `{comparison['max_absolute_change']:.6f}`

`effective_support` is exp(Shannon entropy) over values rounded to 12 decimals.
Lag-1 correlation uses only adjacent decisions from the same episode.

## Interpretation boundary

This report measures a timing ambiguity. It does not select a replacement denominator.
Changing the registered `r` update requires a separately frozen successor specification;
TEARRL-0 continues to use the `current` reading. The accompanying known-answer and
mutation tests prove that whole-effect, fractional-effect, non-learner-effect,
zero-denominator, clipped-effect, and one-turn-shift cases are distinguished.
"""
    return payload, markdown


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=S_T_GENERATOR_SEED)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--witness", type=Path, default=Path("artifacts/witness_frozen.pt"))
    parser.add_argument("--out", type=Path, default=Path("reports/r_anti_vacuity.md"))
    parser.add_argument("--json", type=Path, default=Path("reports/r_anti_vacuity.json"))
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    witness = load_frozen_witness(args.witness, device=device)
    learner, intervening, episode_ids = collect_attribution_inputs(
        witness,
        n_candidates=args.candidates,
        seed=args.seed,
        device=device,
    )
    series = ratio_series(learner, intervening)
    payload, markdown = build_report(
        series,
        episode_ids,
        candidates=args.candidates,
        seed=args.seed,
        witness_sha256=_sha256(args.witness),
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(markdown, encoding="utf-8")
    args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
