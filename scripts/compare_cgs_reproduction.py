#!/usr/bin/env python
"""Compare byte equality and registered verdict equality for CG/S reruns."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

COUNT_FIELDS = (
    "evaluated_rows",
    "worst_party_positives",
    "irreversible_positives",
    "repair_cases",
)
ACTOR_COUNT_FIELDS = (
    "evaluated_contexts",
    "repair_contexts",
    "task_eligible_contexts",
    "ambiguity_contexts",
)
VERDICT_FIELDS = (
    "grounding_pass",
    "bounded_sufficiency_pass",
    "geometry_pass",
    "containment_all_cells",
    "baseline_nonreversal",
)


class ComparisonStructureError(ValueError):
    """The artifacts do not have the same JSON container structure."""


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verdict_projection(result: dict[str, Any]) -> dict[str, Any]:
    aggregate = result["aggregate"]
    stage1 = aggregate["stage1_learned"]
    cells = aggregate["stage2_cells"]
    return {
        "study": result["study"],
        "mode": result["mode"],
        "seeds": result["seeds"],
        "aggregate_verdicts": {
            key: aggregate[key] for key in VERDICT_FIELDS
        },
        "stage1_gates": aggregate["stage1_gates"],
        "bounded_sufficiency_gates": aggregate["bounded_sufficiency_gates"],
        "geometry_gates": aggregate["geometry_gates"],
        "stage1_counts": {key: stage1[key] for key in COUNT_FIELDS},
        "stage2_counts": {
            name: {key: metrics[key] for key in ACTOR_COUNT_FIELDS}
            for name, metrics in cells.items()
        },
        "per_seed_registered_results": [
            {
                "seed": seed_result["seed"],
                "non_punitive_audit": seed_result["non_punitive_audit"],
                "stage1_gates": seed_result["stage1"]["gates"],
                "stage2_geometry_gates": seed_result["stage2"][
                    "geometry_gates"
                ],
                "stage2_oracle_hierarchical_gates": seed_result["stage2"][
                    "oracle_hierarchical_gates"
                ],
            }
            for seed_result in result["seed_results"]
        ],
    }


def _numeric_deltas(
    left: Any,
    right: Any,
    *,
    path: str = "$",
) -> tuple[list[float], list[int]]:
    floating: list[float] = []
    integer: list[int] = []
    if isinstance(left, dict) or isinstance(right, dict):
        if not isinstance(left, dict) or not isinstance(right, dict):
            raise ComparisonStructureError(
                f"{path} container type differs: "
                f"{type(left).__name__} != {type(right).__name__}"
            )
        reference_keys = set(left)
        candidate_keys = set(right)
        if reference_keys != candidate_keys:
            reference_only = sorted(str(key) for key in reference_keys - candidate_keys)
            candidate_only = sorted(str(key) for key in candidate_keys - reference_keys)
            raise ComparisonStructureError(
                f"{path} object keys differ: "
                f"reference_only={reference_only}, candidate_only={candidate_only}"
            )
        for key in left:
            child_floating, child_integer = _numeric_deltas(
                left[key],
                right[key],
                path=f"{path}.{key}",
            )
            floating.extend(child_floating)
            integer.extend(child_integer)
    elif isinstance(left, list) or isinstance(right, list):
        if not isinstance(left, list) or not isinstance(right, list):
            raise ComparisonStructureError(
                f"{path} container type differs: "
                f"{type(left).__name__} != {type(right).__name__}"
            )
        if len(left) != len(right):
            raise ComparisonStructureError(
                f"{path} list length differs: {len(left)} != {len(right)}"
            )
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            child_floating, child_integer = _numeric_deltas(
                a,
                b,
                path=f"{path}[{index}]",
            )
            floating.extend(child_floating)
            integer.extend(child_integer)
    elif (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    ):
        if isinstance(left, int) and isinstance(right, int):
            delta = abs(left - right)
            if delta > 0:
                integer.append(delta)
        else:
            delta = abs(float(left) - float(right))
            if math.isfinite(delta) and delta > 0.0:
                floating.append(delta)
    return floating, integer

def compare_results(reference_path: Path, candidate_path: Path) -> dict[str, Any]:
    reference = _load(reference_path)
    candidate = _load(candidate_path)
    floating_deltas, integer_deltas = _numeric_deltas(reference, candidate)
    reference_projection = verdict_projection(reference)
    candidate_projection = verdict_projection(candidate)
    reference_cells = reference["aggregate"]["stage2_cells"]
    candidate_cells = candidate["aggregate"]["stage2_cells"]
    return {
        "schema_version": 2,
        "comparison_kind": "cgs_byte_and_registered_verdict_reproduction",
        "reference_sha256": _sha256(reference_path),
        "candidate_sha256": _sha256(candidate_path),
        "byte_equal": reference_path.read_bytes() == candidate_path.read_bytes(),
        "registered_verdict_projection_equal": (
            reference_projection == candidate_projection
        ),
        "registered_verdict_projection": candidate_projection,
        "structure_equal": True,
        "numeric_fields_differ": len(floating_deltas) + len(integer_deltas),
        "floating_fields_differ": len(floating_deltas),
        "floating_delta_min": min(floating_deltas) if floating_deltas else 0.0,
        "floating_delta_max": max(floating_deltas) if floating_deltas else 0.0,
        "integer_fields_differ": len(integer_deltas),
        "integer_delta_min": min(integer_deltas) if integer_deltas else 0,
        "integer_delta_max": max(integer_deltas) if integer_deltas else 0,
        "aggregate_cell_bit_equality": {
            name: reference_cells[name] == candidate_cells[name]
            for name in reference_cells
        },
        "interpretation": {
            "G7_in_frozen_projection_is_not_a_runtime_measurement": True,
            "non_punitive_audit_fields_in_projection_are_frozen_literals_not_runtime_counters": True,
            "verdict_equality_does_not_imply_byte_equality": True,
        },
    }


def _write_new_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


def _relative_or_name(path: Path, root: Path) -> str:
    """Return a useful output label without exposing an absolute caller path."""

    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("reports/cgs/results.json"),
    )
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("refusing to overwrite existing comparison output")
    try:
        payload = compare_results(args.reference, args.candidate)
    except ComparisonStructureError as exc:
        print(f">>> CG/S COMPARISON ERROR: {exc}", file=sys.stderr)
        return 2
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        try:
            _write_new_text(args.output, text)
        except FileExistsError:
            parser.error("refusing to overwrite existing comparison output")
        print(_relative_or_name(args.output, Path.cwd()))
    return 0 if payload["registered_verdict_projection_equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())