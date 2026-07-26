"""
F2 matched-pair census -- §6.2 feasibility check.

§6.2's PRIMARY F2 intervention is a within-distribution shuffle of c_t between
tuples matched on (timestep, task opportunity, ||c||, sum|yhat|) that DIFFER IN
HARM DEBT. §6.1 then reads S1 (online conscience) vs S2 (internalization) vs
decorative off the F1-vs-F2 gap.

That makes matched-pair supply a precondition for the MECHANISM claim, not
merely for a gate. d is quantized with a dominant mass point, so the pairs may
be scarce -- this script counts them before anything trains.

    python scripts/f2_census.py [--candidates 20000]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tearrl.eval.tuples import generate_s_t  # noqa: E402
from tearrl.spec import N_ACTIONS  # noqa: E402
from tearrl.witness.targets import ACTION_DESCRIPTORS  # noqa: E402
from tearrl.witness.train import load_frozen_witness  # noqa: E402

# Matching tolerances swept for the census. §6.2 does not fix them; the point
# of the sweep is to show whether supply is robust or knife-edge.
C_NORM_TOLERANCES = (0.05, 0.10, 0.20)
YHAT_TOLERANCES = (0.5, 1.0, 2.0)
DEBT_GAPS = (0.2, 0.3, 0.5)


@torch.no_grad()
def tuple_features(tuple_set, witness, device) -> dict[str, np.ndarray]:
    """Per-tuple matching covariates.

    sum|yhat| uses the five continuous Delta channels only. The three event
    channels and irr are 0/1 indicators; including them would make the
    covariate largely a count of how many violating verbs happen to be legal,
    which is not the "how consequential is this state" quantity §6.2 matches on.
    """
    timesteps, task_ops, c_norms, debts, yhat_mass = [], [], [], [], []

    for item in tuple_set:
        world = item.world
        actor = item.actor
        mask = world.legal_actions(actor)
        feasible = [i for i in range(N_ACTIONS) if mask[i]]

        observation = np.asarray(world.observe(actor), dtype=np.float32)
        rows = np.repeat(observation[None, :], len(feasible), axis=0)
        descriptors = ACTION_DESCRIPTORS[np.asarray(feasible, dtype=np.int64)]
        inputs = torch.tensor(
            np.concatenate([rows, descriptors], axis=1), device=device
        )
        prediction = witness.predict(inputs).cpu().numpy()

        timesteps.append(world.state.t)
        task_ops.append(item.task_opportunity)
        c_norms.append(float(np.linalg.norm(item.c.as_array())))
        debts.append(item.c.d)
        yhat_mass.append(float(np.abs(prediction[:, :5]).sum()))

    return {
        "t": np.asarray(timesteps),
        "task_op": np.asarray(task_ops),
        "c_norm": np.asarray(c_norms, dtype=np.float64),
        "d": np.asarray(debts, dtype=np.float64),
        "yhat": np.asarray(yhat_mass, dtype=np.float64),
    }


def count_pairs(
    features: dict[str, np.ndarray],
    c_tolerance: float,
    yhat_tolerance: float,
    debt_gap: float,
) -> tuple[int, int]:
    """Matched pairs differing in harm debt, and how many distinct tuples
    participate in at least one such pair."""
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, (t, op) in enumerate(zip(features["t"], features["task_op"])):
        buckets[(int(t), int(op))].append(index)

    pairs = 0
    participants: set[int] = set()

    for members in buckets.values():
        if len(members) < 2:
            continue
        idx = np.asarray(members)
        c = features["c_norm"][idx]
        y = features["yhat"][idx]
        d = features["d"][idx]

        close_c = np.abs(c[:, None] - c[None, :]) <= c_tolerance
        close_y = np.abs(y[:, None] - y[None, :]) <= yhat_tolerance
        differs = np.abs(d[:, None] - d[None, :]) >= debt_gap
        ok = np.triu(close_c & close_y & differs, k=1)

        pairs += int(ok.sum())
        rows, cols = np.nonzero(ok)
        participants.update(idx[rows].tolist())
        participants.update(idx[cols].tolist())

    return pairs, len(participants)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=int, default=20_000)
    parser.add_argument("--witness", type=Path, default=Path("artifacts/witness_frozen.pt"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/f2_census.json"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    witness = load_frozen_witness(args.witness, device=device)

    tuple_set = generate_s_t(witness, n_candidates=args.candidates, device=device)
    features = tuple_features(tuple_set, witness, device)

    print("\nharm-debt distribution")
    print("-" * 60)
    values, counts = np.unique(np.round(features["d"], 3), return_counts=True)
    for value, count in zip(values, counts):
        share = count / len(features["d"])
        print(f"  d = {value:5.2f}   n = {count:6,}  ({share:6.2%})  {'#' * int(60 * share)}")
    print(f"  distinct values: {len(values)}   "
          f"largest mass point: {counts.max() / counts.sum():.1%}")

    print("\nF2 matched-pair census (§6.2)")
    print("-" * 78)
    print(f"{'|c| tol':>8s} {'sum|y| tol':>11s} {'min dd':>7s} "
          f"{'pairs':>10s} {'tuples used':>12s} {'coverage':>9s}")
    print("-" * 78)

    results = []
    for c_tolerance in C_NORM_TOLERANCES:
        for yhat_tolerance in YHAT_TOLERANCES:
            for debt_gap in DEBT_GAPS:
                pairs, participants = count_pairs(
                    features, c_tolerance, yhat_tolerance, debt_gap
                )
                coverage = participants / len(tuple_set)
                results.append(
                    {
                        "c_norm_tol": c_tolerance,
                        "yhat_tol": yhat_tolerance,
                        "min_debt_gap": debt_gap,
                        "pairs": pairs,
                        "tuples_used": participants,
                        "coverage": round(coverage, 4),
                    }
                )
                print(
                    f"{c_tolerance:8.2f} {yhat_tolerance:11.1f} {debt_gap:7.2f} "
                    f"{pairs:10,} {participants:12,} {coverage:8.1%}"
                )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "candidates": args.candidates,
                "tuples": len(tuple_set),
                "membership": tuple_set.counts(),
                "debt_distribution": {
                    str(v): int(c) for v, c in zip(values.tolist(), counts.tolist())
                },
                "census": results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
