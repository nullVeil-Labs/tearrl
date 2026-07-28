from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_cgs_thread_evidence.py"
REFERENCE_RESULTS = ROOT / "reports" / "cgs" / "results.json"
REFERENCE_MARKDOWN = ROOT / "reports" / "cgs" / "results.md"


def _module():
    spec = importlib.util.spec_from_file_location("build_cgs_thread_evidence", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _profile(
    environment_threads: int,
    intraop_threads: int,
    interop_threads: int,
) -> dict:
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


def _sidecar(
    *,
    module,
    commit: str,
    profile_name: str,
    profile: dict,
    freeze_chain: dict,
    results: Path,
    markdown: Path,
) -> dict:
    return {
        "schema_version": 1,
        "study": "TEAR-CG/S v0.1",
        "mode": "confirmatory",
        "profile_name": profile_name,
        "source_commit": {
            "value": commit,
            "provenance": "caller_supplied",
        },
        "freeze_chain": freeze_chain,
        "launcher": {
            "path": "scripts/run_cgs.py",
            "sha256": freeze_chain["verified_files"]["scripts/run_cgs.py"],
        },
        "requested": profile,
        "observed": {"before": profile, "after": profile},
        "runtime": {
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
        },
        "outputs": {
            "json": module._file_record(results),
            "markdown": module._file_record(markdown),
        },
    }


def _fixture(tmp_path: Path):
    module = _module()
    reference_results = tmp_path / "reference" / "results.json"
    reference_markdown = tmp_path / "reference" / "results.md"
    candidate_results = tmp_path / "candidate" / "results.json"
    candidate_markdown = tmp_path / "candidate" / "results.md"
    reference_results.parent.mkdir()
    candidate_results.parent.mkdir()
    reference = json.loads(REFERENCE_RESULTS.read_text(encoding="utf-8"))
    candidate = deepcopy(reference)
    candidate["aggregate"]["stage1_learned"]["ece"] += 0.01
    candidate["seed_results"][0]["stage2"]["training_records"][
        "oracle_hierarchical"
    ]["final_loss"] += 1e-8
    for cell in ("learned_flat", "learned_hierarchical"):
        candidate["aggregate"]["stage2_cells"][cell]["task_retention"] += 1e-8
    candidate["seed_results"][2]["stage1"]["learned"][
        "manipulation_fnr_degradation"
    ] = 0.032
    _write_json(reference_results, reference)
    _write_json(candidate_results, candidate)
    reference_markdown.write_text(
        REFERENCE_MARKDOWN.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    candidate_markdown.write_text("# candidate\n", encoding="utf-8")

    comparator = module._load_companion("compare_cgs_reproduction.py")
    comparison = comparator.compare_results(reference_results, candidate_results)
    constraints = module.EvidenceConstraints(
        source_commit="a" * 40,
        reference_results_sha256=_sha256(reference_results),
        reference_markdown_sha256=_sha256(reference_markdown),
        candidate_results_sha256=_sha256(candidate_results),
        candidate_markdown_sha256=_sha256(candidate_markdown),
        numeric_fields_differ=comparison["numeric_fields_differ"],
        floating_fields_differ=comparison["floating_fields_differ"],
        floating_delta_min=comparison["floating_delta_min"],
        floating_delta_max=comparison["floating_delta_max"],
        integer_fields_differ=comparison["integer_fields_differ"],
        integer_delta_min=comparison["integer_delta_min"],
        integer_delta_max=comparison["integer_delta_max"],
    )
    profiled = module._load_companion("run_cgs_profiled.py")
    freeze_chain = profiled._verify_freeze_chain(ROOT)
    reference_sidecar = tmp_path / "reference" / "profile.json"
    candidate_sidecar = tmp_path / "candidate" / "profile.json"
    _write_json(
        reference_sidecar,
        _sidecar(
            module=module,
            commit=constraints.source_commit,
            profile_name="env8_torch8_8",
            profile=_profile(8, 8, 8),
            freeze_chain=freeze_chain,
            results=reference_results,
            markdown=reference_markdown,
        ),
    )
    _write_json(
        candidate_sidecar,
        _sidecar(
            module=module,
            commit=constraints.source_commit,
            profile_name="env1_torch1_8",
            profile=_profile(1, 1, 8),
            freeze_chain=freeze_chain,
            results=candidate_results,
            markdown=candidate_markdown,
        ),
    )
    paths = {
        "reference_results": reference_results,
        "reference_markdown": reference_markdown,
        "reference_sidecar": reference_sidecar,
        "candidate_results": candidate_results,
        "candidate_markdown": candidate_markdown,
        "candidate_sidecar": candidate_sidecar,
        "source_root": ROOT,
    }
    return module, paths, constraints


def test_builder_binds_profiles_outputs_freeze_chain_and_comparison(
    tmp_path: Path,
) -> None:
    module, paths, constraints = _fixture(tmp_path)

    evidence = module.build_thread_evidence(**paths, constraints=constraints)

    assert evidence["reference"]["intraop_threads"] == 8
    assert evidence["candidate"]["interop_threads"] == 8
    assert evidence["comparison"]["registered_verdict_projection_equal"] is True
    assert evidence["aggregate_cell_bit_equality"] == {
        "learned_flat": False,
        "oracle_flat": True,
        "learned_hierarchical": False,
        "oracle_hierarchical": True,
    }
    assert evidence["seed_6311_component_bound_status_same"] is False
    assert evidence["profile_sidecars"]["candidate"]["sha256"] == _sha256(
        paths["candidate_sidecar"]
    )
    assert evidence["candidate"]["profile_sidecar_sha256"] == _sha256(
        paths["candidate_sidecar"]
    )
    assert evidence["selected_stage1_metrics"]["ece"]["reference"] != (
        evidence["selected_stage1_metrics"]["ece"]["candidate"]
    )
    assert evidence["oracle_actor_training_final_losses_all_bit_equal"] is False
    assert evidence["oracle_actor_training_final_loss_bit_equal_by_seed"]["4103"][
        "oracle_hierarchical"
    ] is False
    assert str(tmp_path) not in json.dumps(evidence)


def test_builder_rejects_a_sidecar_output_hash_mismatch(tmp_path: Path) -> None:
    module, paths, constraints = _fixture(tmp_path)
    candidate_sidecar = json.loads(
        paths["candidate_sidecar"].read_text(encoding="utf-8")
    )
    candidate_sidecar["outputs"]["json"]["sha256"] = "0" * 64
    _write_json(paths["candidate_sidecar"], candidate_sidecar)

    with pytest.raises(RuntimeError, match="output binding"):
        module.build_thread_evidence(**paths, constraints=constraints)


def test_builder_rejects_observed_profile_drift(tmp_path: Path) -> None:
    module, paths, constraints = _fixture(tmp_path)
    candidate_sidecar = json.loads(
        paths["candidate_sidecar"].read_text(encoding="utf-8")
    )
    candidate_sidecar["observed"]["after"]["torch"]["num_interop_threads"] = 1
    _write_json(paths["candidate_sidecar"], candidate_sidecar)

    with pytest.raises(RuntimeError, match="observed thread profile"):
        module.build_thread_evidence(**paths, constraints=constraints)

@pytest.mark.parametrize(
    "private_value",
    (
        "prefix C:/Users/name/file.txt",
        "prefix \\\\server\\share\\file.txt",
        "note /Users/name/file.txt",
        "note /home/name/file.txt",
        "file://private/path",
    ),
)
def test_private_string_scan_rejects_embedded_paths(private_value: str) -> None:
    module = _module()
    with pytest.raises(RuntimeError, match="private path or URI"):
        module._reject_absolute_paths({"value": private_value})