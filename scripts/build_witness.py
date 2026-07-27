"""
Build, train and freeze the witness -- build-order step 4 (§2.3, B.3).

    python scripts/build_witness.py [--episodes 250000] [--out artifacts/]

Deterministic given the seeds baked into the pipeline: the dataset seeds, the
90/10 state split seed, and the torch seed. Re-running reproduces the same
frozen instrument, which is what §2.3(4) requires when the identical witness
is copied into C3-IM, C4, C5 and C6.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tearrl.witness.dataset import build_dataset, split  # noqa: E402
from tearrl.witness.train import (  # noqa: E402
    evaluate_freeze,
    freeze,
    train_witness,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=250_000, help="B.3 budget")
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-epochs", type=int, default=400)
    parser.add_argument(
        "--save-dataset",
        action="store_true",
        help="also persist the deduplicated dataset (large)",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"WITNESS BUILD -- B.3 budget {args.episodes:,} episodes")
    print("=" * 70)

    started = time.perf_counter()
    dataset = build_dataset(episodes=args.episodes, progress_every=25_000)
    build_seconds = time.perf_counter() - started
    print(
        f"\ndataset: {dataset.n_states:,} unique states | {dataset.n_pairs:,} pairs "
        f"| {dataset.states_seen:,} visited "
        f"| dedup {1 - dataset.n_states / dataset.states_seen:.1%} "
        f"| {build_seconds:.0f}s"
    )

    if args.save_dataset:
        dataset.save(args.out / "witness_dataset.npz")

    print("\ntraining ensemble ...")
    trained = time.perf_counter()
    ensemble, _, validation_rows = train_witness(
        dataset, device=args.device, max_epochs=args.max_epochs
    )
    train_seconds = time.perf_counter() - trained

    print("\nB.3 FREEZE EVALUATION")
    print("-" * 70)
    report = evaluate_freeze(ensemble, dataset, validation_rows, device=args.device)
    print(report.describe())

    witness_path = args.out / "witness_frozen.pt"
    metadata = {
        "episodes": args.episodes,
        "unique_states": dataset.n_states,
        "pairs": dataset.n_pairs,
        "states_visited": dataset.states_seen,
        "build_seconds": round(build_seconds, 1),
        "train_seconds": round(train_seconds, 1),
        "freeze_report": asdict(report),
        "criteria_met": report.frozen,
    }
    (args.out / "witness_freeze_report.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    print(f"wrote {args.out / 'witness_freeze_report.json'}")

    if not report.frozen:
        print(
            "\nFREEZE CRITERIA NOT MET -- the witness must NOT be used.\n"
            "§2.3(6) permits one dataset expansion and retrain before v1.0."
        )
        return 1

    freeze(ensemble, witness_path, report)
    print(f"\nwrote {witness_path}")
    print("Witness frozen. §2.3(4): copy this identical file into C3-IM, C4, C5, C6.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
