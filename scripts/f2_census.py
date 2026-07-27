"""
F2 matched-pair census -- §6.2 feasibility check.

§6.2's PRIMARY F2 intervention is a within-distribution shuffle of c_t between
tuples matched on (timestep, task opportunity, ||c||, sum|yhat|) that DIFFER IN
HARM DEBT. §6.1 then reads S1 (online conscience) vs S2 (internalization) vs
decorative off the F1-vs-F2 gap.

That makes matched-pair supply a precondition for the MECHANISM claim, not
merely for a gate. d is quantized with a dominant mass point, so the pairs may
be scarce -- this script counts them before anything trains.

    python scripts/f2_census.py [--candidates 60000]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tearrl.eval.f2_matching import build_candidate_edges, match  # noqa: E402
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
    seeds, timesteps, task_ops, c_norms, debts, yhat_mass = [], [], [], [], [], []

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

        seeds.append(world._seed)
        timesteps.append(world.state.t)
        task_ops.append(item.task_opportunity)
        c_norms.append(float(np.linalg.norm(item.c.as_array())))
        debts.append(item.c.d)
        yhat_mass.append(float(np.abs(prediction[:, :5]).sum()))

    return {
        "seed": np.asarray(seeds, dtype=np.int64),
        "t": np.asarray(timesteps),
        "task_op": np.asarray(task_ops),
        "c_norm": np.asarray(c_norms, dtype=np.float64),
        "d": np.asarray(debts, dtype=np.float64),
        "yhat": np.asarray(yhat_mass, dtype=np.float64),
    }


def matched_supply(
    features: dict[str, np.ndarray],
    c_tolerance: float,
    yhat_tolerance: float,
    debt_gap: float,
):
    """Registered one-to-one F2 matching, not merely compatible edges."""
    edges = build_candidate_edges(
        features["c_norm"], features["yhat"], features["d"],
        features["t"], features["task_op"],
        c_tolerance=c_tolerance, yhat_tolerance=yhat_tolerance,
        min_debt_gap=debt_gap,
    )
    return match(edges, features["seed"], n_tuples=len(features["d"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=int, default=60_000)
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
    print("-" * 96)
    print(f"{'|c| tol':>8s} {'sum|y| tol':>11s} {'min dd':>7s} "
          f"{'edges':>10s} {'matches':>10s} {'tuples used':>12s} {'coverage':>9s}")
    print("-" * 96)

    results = []
    for c_tolerance in C_NORM_TOLERANCES:
        for yhat_tolerance in YHAT_TOLERANCES:
            for debt_gap in DEBT_GAPS:
                matching = matched_supply(
                    features, c_tolerance, yhat_tolerance, debt_gap
                )
                matched_tuples = 2 * matching.n_pairs
                results.append(
                    {
                        "c_norm_tol": c_tolerance,
                        "yhat_tol": yhat_tolerance,
                        "min_debt_gap": debt_gap,
                        "candidate_edges": matching.candidate_edges,
                        "matched_pairs": matching.n_pairs,
                        "tuples_used": matched_tuples,
                        "coverage": round(matching.coverage, 4),
                    }
                )
                print(
                    f"{c_tolerance:8.2f} {yhat_tolerance:11.1f} {debt_gap:7.2f} "
                    f"{matching.candidate_edges:10,} {matching.n_pairs:10,} "
                    f"{matched_tuples:12,} {matching.coverage:8.1%}"
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
