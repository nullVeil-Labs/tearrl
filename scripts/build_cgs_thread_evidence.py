#!/usr/bin/env python
"""Build strict evidence for a bundled CG/S numerical-thread-profile contrast."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "8568edb329e3accdeab5f16c763d42853bef30fd"
REFERENCE_RESULTS_SHA256 = (
    "1051c9b1b8a5a4f34d332a594836854cfb447f6f2bfb77f3e880b834950a9fec"
)
REFERENCE_MARKDOWN_SHA256 = (
    "064ecd9632240b3e5fab05312aa65eb8123674a46a4e3f5025f5458260a081ad"
)
CANDIDATE_RESULTS_SHA256 = (
    "5cad306367a83f4f6147618af89a433e22a2bdc1a780b59f9d70ade9ae3b033c"
)
CANDIDATE_MARKDOWN_SHA256 = (
    "29f2f9c2b1858283f04cfc681f1abfa68d3011b56c8b3991dbd925e4422037dd"
)
SEEDS = (4103, 5207, 6311)
EXPECTED_RUNTIME = {
    "python": {"implementation": "CPython", "version": "3.12.10"},
    "os_family": "Windows",
    "packages": {
        "networkx": "3.6.1",
        "numpy": "2.5.1",
        "pytest": "9.1.1",
        "scipy": "1.18.0",
        "torch": "2.11.0+cu128",
    },
    "torch_module_version": "2.11.0+cu128",
    "torch_cuda_build": "12.8",
}
SELECTED_STAGE1_METRICS = (
    "worst_party_fnr",
    "irreversible_fnr",
    "ece",
    "brier",
    "identifiable_coverage",
)
EXPECTED_AGGREGATE_CELL_BIT_EQUALITY = {
    "learned_flat": False,
    "oracle_flat": True,
    "learned_hierarchical": False,
    "oracle_hierarchical": True,
}
SIDECAR_KEYS = {
    "schema_version",
    "study",
    "mode",
    "profile_name",
    "source_commit",
    "freeze_chain",
    "launcher",
    "requested",
    "observed",
    "runtime",
    "outputs",
}


@dataclass(frozen=True, slots=True)
class EvidenceConstraints:
    source_commit: str = SOURCE_COMMIT
    reference_results_sha256: str = REFERENCE_RESULTS_SHA256
    reference_markdown_sha256: str = REFERENCE_MARKDOWN_SHA256
    candidate_results_sha256: str = CANDIDATE_RESULTS_SHA256
    candidate_markdown_sha256: str = CANDIDATE_MARKDOWN_SHA256
    numeric_fields_differ: int = 314
    floating_fields_differ: int = 262
    floating_delta_min: float = 7.421476766467094e-10
    floating_delta_max: float = 1.475
    integer_fields_differ: int = 52
    integer_delta_min: int = 1
    integer_delta_max: int = 341


DEFAULT_CONSTRAINTS = EvidenceConstraints()


def _load_companion(filename: str) -> ModuleType:
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(f"_cgs_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load companion script: {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path.name}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.name,
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _expected_profile(
    *,
    environment_threads: int,
    intraop_threads: int,
    interop_threads: int,
) -> dict[str, Any]:
    return {
        "environment": {
            "OMP_NUM_THREADS": str(environment_threads),
            "MKL_NUM_THREADS": str(environment_threads),
            "OPENBLAS_NUM_THREADS": str(environment_threads),
        },
        "torch": {
            "num_threads": intraop_threads,
            "num_interop_threads": interop_threads,
        },
    }


def _validate_runtime(runtime: Any) -> dict[str, Any]:
    if runtime != EXPECTED_RUNTIME:
        raise RuntimeError("profile runtime does not match the frozen environment")
    return runtime


def _validate_output_binding(
    record: Any,
    artifact: Path,
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(record, dict) or set(record) != {"path", "sha256", "bytes"}:
        raise RuntimeError(f"{label} output binding schema drift")
    actual = _file_record(artifact)
    if record != actual:
        raise RuntimeError(f"{label} output binding does not match artifact")
    return actual


def _validate_sidecar(
    sidecar_path: Path,
    *,
    role: str,
    results_path: Path,
    markdown_path: Path,
    expected_profile: dict[str, Any],
    expected_profile_name: str,
    expected_freeze_chain: dict[str, Any],
    constraints: EvidenceConstraints,
) -> dict[str, Any]:
    sidecar = _load_json(sidecar_path)
    if set(sidecar) != SIDECAR_KEYS:
        raise RuntimeError(f"{role} profile sidecar schema drift")
    if sidecar["schema_version"] != 1:
        raise RuntimeError(f"{role} profile sidecar schema version is not 1")
    if sidecar["study"] != "TEAR-CG/S v0.1" or sidecar["mode"] != "confirmatory":
        raise RuntimeError(f"{role} profile sidecar has the wrong study or mode")
    if sidecar["profile_name"] != expected_profile_name:
        raise RuntimeError(f"{role} profile name is invalid")
    if sidecar["source_commit"] != {
        "value": constraints.source_commit,
        "provenance": "caller_supplied",
    }:
        raise RuntimeError(f"{role} profile has the wrong source commit")
    if sidecar["freeze_chain"] != expected_freeze_chain:
        raise RuntimeError(f"{role} profile does not bind the exact freeze chain")
    expected_launcher = {
        "path": "scripts/run_cgs.py",
        "sha256": expected_freeze_chain["verified_files"]["scripts/run_cgs.py"],
    }
    if sidecar["launcher"] != expected_launcher:
        raise RuntimeError(f"{role} profile launcher binding is invalid")
    if sidecar["requested"] != expected_profile:
        raise RuntimeError(f"{role} requested thread profile is invalid")
    if sidecar["observed"] != {
        "before": expected_profile,
        "after": expected_profile,
    }:
        raise RuntimeError(f"{role} observed thread profile is invalid")
    _validate_runtime(sidecar["runtime"])
    outputs = sidecar["outputs"]
    if not isinstance(outputs, dict) or set(outputs) != {"json", "markdown"}:
        raise RuntimeError(f"{role} profile output set is invalid")
    _validate_output_binding(
        outputs["json"],
        results_path,
        label=f"{role} JSON",
    )
    _validate_output_binding(
        outputs["markdown"],
        markdown_path,
        label=f"{role} Markdown",
    )
    return {
        "sha256": _sha256(sidecar_path),
        "payload": sidecar,
    }


def _pair(reference: float, candidate: float) -> dict[str, float]:
    return {"reference": float(reference), "candidate": float(candidate)}


def _seed_map(result: dict[str, Any]) -> dict[int, dict[str, Any]]:
    mapped = {int(item["seed"]): item for item in result["seed_results"]}
    if tuple(sorted(mapped)) != SEEDS:
        raise RuntimeError("result does not contain the frozen seed set")
    return mapped


def _reject_absolute_paths(value: Any) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _reject_absolute_paths(item)
    elif isinstance(value, list):
        for item in value:
            _reject_absolute_paths(item)
    elif isinstance(value, str):
        normalized = value.replace("\\", "/")
        if (
            re.search(r"(?i)[A-Z]:/", normalized)
            or re.search(r"(^|\s)//[^/]", normalized)
            or re.search(r"(?i)/users/|/home/|file://", normalized)
        ):
            raise RuntimeError("thread evidence contains a private path or URI")


def build_thread_evidence(
    *,
    reference_results: Path,
    reference_markdown: Path,
    reference_sidecar: Path,
    candidate_results: Path,
    candidate_markdown: Path,
    candidate_sidecar: Path,
    source_root: Path,
    constraints: EvidenceConstraints = DEFAULT_CONSTRAINTS,
) -> dict[str, Any]:
    canonical_basenames = {
        reference_results: "results.json",
        reference_markdown: "results.md",
        candidate_results: "results.json",
        candidate_markdown: "results.md",
    }
    for path, expected_name in canonical_basenames.items():
        if path.name != expected_name:
            raise RuntimeError(f"non-canonical thread artifact basename: {path.name}")
    expected_hashes = {
        reference_results: constraints.reference_results_sha256,
        reference_markdown: constraints.reference_markdown_sha256,
        candidate_results: constraints.candidate_results_sha256,
        candidate_markdown: constraints.candidate_markdown_sha256,
    }
    for path, expected_hash in expected_hashes.items():
        if not path.is_file() or _sha256(path) != expected_hash:
            raise RuntimeError(f"thread evidence artifact hash mismatch: {path.name}")

    profiled = _load_companion("run_cgs_profiled.py")
    expected_freeze_chain = profiled._verify_freeze_chain(source_root.resolve())
    reference_profile = _expected_profile(
        environment_threads=8,
        intraop_threads=8,
        interop_threads=8,
    )
    candidate_profile = _expected_profile(
        environment_threads=1,
        intraop_threads=1,
        interop_threads=8,
    )
    reference_sidecar_record = _validate_sidecar(
        reference_sidecar,
        role="reference",
        results_path=reference_results,
        markdown_path=reference_markdown,
        expected_profile=reference_profile,
        expected_profile_name="env8_torch8_8",
        expected_freeze_chain=expected_freeze_chain,
        constraints=constraints,
    )
    candidate_sidecar_record = _validate_sidecar(
        candidate_sidecar,
        role="candidate",
        results_path=candidate_results,
        markdown_path=candidate_markdown,
        expected_profile=candidate_profile,
        expected_profile_name="env1_torch1_8",
        expected_freeze_chain=expected_freeze_chain,
        constraints=constraints,
    )
    if (
        reference_sidecar_record["payload"]["runtime"]
        != candidate_sidecar_record["payload"]["runtime"]
    ):
        raise RuntimeError("thread profiles were not run in the same software runtime")

    comparator = _load_companion("compare_cgs_reproduction.py")
    comparison = comparator.compare_results(reference_results, candidate_results)
    exact_comparison = {
        "schema_version": 2,
        "structure_equal": True,
        "reference_sha256": constraints.reference_results_sha256,
        "candidate_sha256": constraints.candidate_results_sha256,
        "byte_equal": False,
        "registered_verdict_projection_equal": True,
        "numeric_fields_differ": constraints.numeric_fields_differ,
        "floating_fields_differ": constraints.floating_fields_differ,
        "floating_delta_min": constraints.floating_delta_min,
        "floating_delta_max": constraints.floating_delta_max,
        "integer_fields_differ": constraints.integer_fields_differ,
        "integer_delta_min": constraints.integer_delta_min,
        "integer_delta_max": constraints.integer_delta_max,
    }
    for key, expected in exact_comparison.items():
        actual = comparison.get(key)
        if isinstance(expected, float):
            if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=0.0):
                raise RuntimeError(f"thread comparison mismatch: {key}")
        elif actual != expected:
            raise RuntimeError(f"thread comparison mismatch: {key}")
    if (
        comparison["aggregate_cell_bit_equality"]
        != EXPECTED_AGGREGATE_CELL_BIT_EQUALITY
    ):
        raise RuntimeError("aggregate cell bit-equality mapping is invalid")

    reference = _load_json(reference_results)
    candidate = _load_json(candidate_results)
    reference_seeds = _seed_map(reference)
    candidate_seeds = _seed_map(candidate)
    selected_stage1_metrics = {
        name: _pair(
            reference["aggregate"]["stage1_learned"][name],
            candidate["aggregate"]["stage1_learned"][name],
        )
        for name in SELECTED_STAGE1_METRICS
    }
    witness_final_loss = {
        str(seed): _pair(
            reference_seeds[seed]["witness_training"]["final_loss"],
            candidate_seeds[seed]["witness_training"]["final_loss"],
        )
        for seed in SEEDS
    }
    calibration = {
        str(seed): {
            name: _pair(
                reference_seeds[seed]["calibration"][name],
                candidate_seeds[seed]["calibration"][name],
            )
            for name in ("event_temperature", "uncertainty_threshold")
        }
        for seed in SEEDS
    }
    manipulation = {
        str(seed): _pair(
            reference_seeds[seed]["stage1"]["learned"][
                "manipulation_fnr_degradation"
            ],
            candidate_seeds[seed]["stage1"]["learned"][
                "manipulation_fnr_degradation"
            ],
        )
        for seed in SEEDS
    }
    oracle_training_loss = {
        str(seed): {
            cell: _pair(
                reference_seeds[seed]["stage2"]["training_records"][cell][
                    "final_loss"
                ],
                candidate_seeds[seed]["stage2"]["training_records"][cell][
                    "final_loss"
                ],
            )
            for cell in ("oracle_flat", "oracle_hierarchical", "oracle_task_only")
        }
        for seed in SEEDS
    }
    oracle_training_equality = {
        seed: {
            cell: pair["reference"] == pair["candidate"]
            for cell, pair in cells.items()
        }
        for seed, cells in oracle_training_loss.items()
    }
    oracle_training_all_equal = all(
        equal
        for cells in oracle_training_equality.values()
        for equal in cells.values()
    )
    seed_6311_component_bound_status_same = (
        manipulation["6311"]["reference"] <= 0.05
    ) == (manipulation["6311"]["candidate"] <= 0.05)
    if seed_6311_component_bound_status_same:
        raise RuntimeError("seed 6311 did not cross its component manipulation bound")
    evidence = {
        "schema_version": 1,
        "evidence_kind": "post_result_bundled_numerical_thread_profile_contrast",
        "source_commit": {
            "value": constraints.source_commit,
            "provenance": "caller_supplied",
        },
        "freeze_chain": expected_freeze_chain,
        "reference": {
            "profile_name": "env8_torch8_8",
            "intraop_threads": 8,
            "interop_threads": 8,
            "results_sha256": constraints.reference_results_sha256,
            "results_md_sha256": constraints.reference_markdown_sha256,
            "profile_sidecar_sha256": reference_sidecar_record["sha256"],
        },
        "candidate": {
            "profile_name": "env1_torch1_8",
            "intraop_threads": 1,
            "interop_threads": 8,
            "results_sha256": constraints.candidate_results_sha256,
            "results_md_sha256": constraints.candidate_markdown_sha256,
            "profile_sidecar_sha256": candidate_sidecar_record["sha256"],
        },
        "profile_sidecars": {
            "reference": reference_sidecar_record,
            "candidate": candidate_sidecar_record,
        },
        "comparison": comparison,
        "selected_stage1_metrics": selected_stage1_metrics,
        "witness_final_loss_by_seed": witness_final_loss,
        "calibration_by_seed": calibration,
        "manipulation_degradation_by_seed": manipulation,
        "oracle_actor_training_final_loss_by_seed": oracle_training_loss,
        "oracle_actor_training_final_loss_bit_equal_by_seed": oracle_training_equality,
        "oracle_actor_training_final_losses_all_bit_equal": oracle_training_all_equal,
        "aggregate_cell_bit_equality": EXPECTED_AGGREGATE_CELL_BIT_EQUALITY,
        "oracle_primary_cells_bit_equal": True,
        "study_level_gate_dictionaries_and_final_verdicts_unchanged": True,
        "seed_6311_component_bound_status_same": False,
    }
    _reject_absolute_paths(evidence)
    return evidence


def _relative_or_name(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-results",
        type=Path,
        default=Path("reports/cgs/results.json"),
    )
    parser.add_argument(
        "--reference-markdown",
        type=Path,
        default=Path("reports/cgs/results.md"),
    )
    parser.add_argument("--reference-sidecar", type=Path, required=True)
    parser.add_argument("--candidate-results", type=Path, required=True)
    parser.add_argument("--candidate-markdown", type=Path, required=True)
    parser.add_argument("--candidate-sidecar", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite existing thread evidence")
    payload = build_thread_evidence(
        reference_results=args.reference_results,
        reference_markdown=args.reference_markdown,
        reference_sidecar=args.reference_sidecar,
        candidate_results=args.candidate_results,
        candidate_markdown=args.candidate_markdown,
        candidate_sidecar=args.candidate_sidecar,
        source_root=args.source_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(_relative_or_name(args.output, ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
