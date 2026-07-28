from __future__ import annotations

from copy import deepcopy
from importlib import metadata
import importlib.util
import json
from pathlib import Path
import platform
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_cgs_publication.py"


def _module():
    name = "audit_cgs_publication"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _Threads:
    def __init__(self, intra: int, inter: int) -> None:
        self.intra = intra
        self.inter = inter

    def get_num_threads(self) -> int:
        return self.intra

    def get_num_interop_threads(self) -> int:
        return self.inter


def _set_frozen_environment(monkeypatch) -> None:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        monkeypatch.setenv(name, "8")


def _frozen_result() -> dict:
    return json.loads((ROOT / "reports/cgs/results.json").read_text(encoding="utf-8"))


def test_frozen_thread_profile_is_fail_closed(monkeypatch) -> None:
    module = _module()
    _set_frozen_environment(monkeypatch)
    assert module.require_frozen_thread_profile(_Threads(8, 8)) == {
        "OMP_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "8",
        "torch_intraop_threads": 8,
        "torch_interop_threads": 8,
    }
    with pytest.raises(module.AuditError, match="requires torch.*8/8"):
        module.require_frozen_thread_profile(_Threads(1, 8))
    monkeypatch.delenv("OMP_NUM_THREADS")
    with pytest.raises(module.AuditError, match="OMP/MKL/OpenBLAS"):
        module.require_frozen_thread_profile(_Threads(8, 8))


def test_ece_bins_publish_recomputable_sufficient_statistics() -> None:
    module = _module()
    probabilities = np.asarray([0.05, 0.15, 0.85, 0.95])
    targets = np.asarray([False, True, False, True])
    bins = module._ece_bins(probabilities, targets)
    assert sum(item["count"] for item in bins) == 4
    assert sum(item["ece_contribution"] for item in bins) == pytest.approx(0.45)
    assert bins[0]["probability_sum"] == 0.05
    assert bins[9]["upper_inclusive"] is True


def _profile_sidecar(
    module,
    freeze_chain: dict,
    *,
    role: str,
    results_sha256: str,
    markdown_sha256: str,
) -> dict:
    reference = role == "reference"
    environment_threads = 8 if reference else 1
    intraop_threads = 8 if reference else 1
    profile_name = "env8_torch8_8" if reference else "env1_torch1_8"
    requested = {
        "environment": {
            "OMP_NUM_THREADS": str(environment_threads),
            "MKL_NUM_THREADS": str(environment_threads),
            "OPENBLAS_NUM_THREADS": str(environment_threads),
        },
        "torch": {"num_threads": intraop_threads, "num_interop_threads": 8},
    }
    return {
        "schema_version": 1,
        "study": "TEAR-CG/S v0.1",
        "mode": "confirmatory",
        "profile_name": profile_name,
        "source_commit": {
            "value": module.SOURCE_COMMIT,
            "provenance": "caller_supplied",
        },
        "freeze_chain": freeze_chain,
        "launcher": {
            "path": "scripts/run_cgs.py",
            "sha256": freeze_chain["verified_files"]["scripts/run_cgs.py"],
        },
        "requested": requested,
        "observed": {"before": requested, "after": requested},
        "runtime": deepcopy(module.EXPECTED_THREAD_RUNTIME),
        "outputs": {
            "json": {
                "path": "results.json",
                "sha256": results_sha256,
                "bytes": 1,
            },
            "markdown": {
                "path": "results.md",
                "sha256": markdown_sha256,
                "bytes": 1,
            },
        },
    }


def _thread_payload(module, frozen: dict) -> dict:
    freeze_chain = module._expected_frozen_implementation_chain(ROOT)
    reference_sidecar_sha = "a" * 64
    candidate_sidecar_sha = "b" * 64
    reference_sidecar = _profile_sidecar(
        module,
        freeze_chain,
        role="reference",
        results_sha256=module.REFERENCE_RESULTS_SHA256,
        markdown_sha256=module.EXPECTED_SOURCE_HASHES["reports/cgs/results.md"],
    )
    candidate_sidecar = _profile_sidecar(
        module,
        freeze_chain,
        role="candidate",
        results_sha256=module.EXPECTED_CANDIDATE_RESULTS_SHA256,
        markdown_sha256=module.EXPECTED_CANDIDATE_RESULTS_MD_SHA256,
    )
    comparison = {
        "schema_version": 2,
        "comparison_kind": "cgs_byte_and_registered_verdict_reproduction",
        "structure_equal": True,
        "reference_sha256": module.REFERENCE_RESULTS_SHA256,
        "candidate_sha256": module.EXPECTED_CANDIDATE_RESULTS_SHA256,
        "byte_equal": False,
        "registered_verdict_projection_equal": True,
        "registered_verdict_projection": module._registered_verdict_projection(frozen),
        "numeric_fields_differ": 314,
        "floating_fields_differ": 262,
        "floating_delta_min": 7.421476766467094e-10,
        "floating_delta_max": 1.475,
        "integer_fields_differ": 52,
        "integer_delta_min": 1,
        "integer_delta_max": 341,
        "aggregate_cell_bit_equality": deepcopy(
            module.EXPECTED_AGGREGATE_CELL_BIT_EQUALITY
        ),
        "interpretation": {
            "G7_in_frozen_projection_is_not_a_runtime_measurement": True,
            "verdict_equality_does_not_imply_byte_equality": True,
        },
    }
    return {
        "schema_version": 1,
        "evidence_kind": "post_result_bundled_numerical_thread_profile_contrast",
        "source_commit": {
            "value": module.SOURCE_COMMIT,
            "provenance": "caller_supplied",
        },
        "freeze_chain": freeze_chain,
        "reference": {
            "profile_name": "env8_torch8_8",
            "intraop_threads": 8,
            "interop_threads": 8,
            "results_sha256": module.REFERENCE_RESULTS_SHA256,
            "results_md_sha256": module.EXPECTED_SOURCE_HASHES[
                "reports/cgs/results.md"
            ],
            "profile_sidecar_sha256": reference_sidecar_sha,
        },
        "candidate": {
            "profile_name": "env1_torch1_8",
            "intraop_threads": 1,
            "interop_threads": 8,
            "results_sha256": module.EXPECTED_CANDIDATE_RESULTS_SHA256,
            "results_md_sha256": module.EXPECTED_CANDIDATE_RESULTS_MD_SHA256,
            "profile_sidecar_sha256": candidate_sidecar_sha,
        },
        "profile_sidecars": {
            "reference": {
                "sha256": reference_sidecar_sha,
                "payload": reference_sidecar,
            },
            "candidate": {
                "sha256": candidate_sidecar_sha,
                "payload": candidate_sidecar,
            },
        },
        "comparison": comparison,
        "selected_stage1_metrics": deepcopy(
            module.EXPECTED_SELECTED_STAGE1_METRICS
        ),
        "witness_final_loss_by_seed": deepcopy(
            module.EXPECTED_WITNESS_FINAL_LOSS_BY_SEED
        ),
        "calibration_by_seed": deepcopy(module.EXPECTED_CALIBRATION_BY_SEED),
        "manipulation_degradation_by_seed": deepcopy(
            module.EXPECTED_MANIPULATION_DEGRADATION_BY_SEED
        ),
        "oracle_actor_training_final_loss_by_seed": deepcopy(
            module.EXPECTED_ORACLE_TRAINING_FINAL_LOSS_BY_SEED
        ),
        "oracle_actor_training_final_loss_bit_equal_by_seed": deepcopy(
            module.EXPECTED_ORACLE_TRAINING_EQUALITY_BY_SEED
        ),
        "oracle_actor_training_final_losses_all_bit_equal": False,
        "aggregate_cell_bit_equality": deepcopy(
            module.EXPECTED_AGGREGATE_CELL_BIT_EQUALITY
        ),
        "oracle_primary_cells_bit_equal": True,
        "study_level_gate_dictionaries_and_final_verdicts_unchanged": True,
        "seed_6311_component_bound_status_same": False,
    }


def test_thread_evidence_is_exact_and_fail_closed() -> None:
    module = _module()
    frozen = _frozen_result()
    valid = _thread_payload(module, frozen)
    normalized = module.validate_thread_sensitivity_evidence(valid, ROOT, frozen)
    assert normalized["comparison"]["structure_equal"] is True
    assert (
        normalized[
            "study_level_gate_dictionaries_and_final_verdicts_unchanged"
        ]
        is True
    )
    assert normalized["seed_6311_component_bound_status_same"] is False

    mutations = []
    changed = deepcopy(valid)
    changed["comparison"]["structure_equal"] = False
    mutations.append(changed)
    changed = deepcopy(valid)
    changed["comparison"]["registered_verdict_projection"] = {}
    mutations.append(changed)
    changed = deepcopy(valid)
    changed["selected_stage1_metrics"]["ece"]["candidate"] += 0.01
    mutations.append(changed)
    changed = deepcopy(valid)
    changed["profile_sidecars"]["candidate"]["payload"]["observed"]["after"][
        "torch"
    ]["num_threads"] = 8
    mutations.append(changed)
    for changed in mutations:
        with pytest.raises(module.AuditError):
            module.validate_thread_sensitivity_evidence(changed, ROOT, frozen)


def _environment_payload(module, reference_sidecar: dict) -> dict:
    runtime = reference_sidecar["runtime"]
    return {
        "schema_version": 1,
        "profile_kind": "retrospective_exact_reproduction_environment",
        "python": deepcopy(runtime["python"]),
        "platform": {"os_family": runtime["os_family"]},
        "packages": deepcopy(runtime["packages"]),
        "numeric_runtime": {
            "device_used": "cpu",
            "torch_module_version": runtime["torch_module_version"],
            "torch_cuda_build": runtime["torch_cuda_build"],
            "torch_intraop_threads": 8,
            "torch_interop_threads": 8,
            "OMP_NUM_THREADS": "8",
            "MKL_NUM_THREADS": "8",
            "OPENBLAS_NUM_THREADS": "8",
        },
        "limitations": [
            "This records the environment that reproduced the artifacts; it is not an installation lock.",
            "Cross-platform and cross-version bit-exactness have not been established.",
            "Hardware, host, executable, working-directory, and user identifiers are intentionally omitted.",
        ],
    }


def test_environment_evidence_is_cross_bound_and_exact() -> None:
    module = _module()
    thread = _thread_payload(module, _frozen_result())
    reference_sidecar = thread["profile_sidecars"]["reference"]["payload"]
    payload = _environment_payload(module, reference_sidecar)
    normalized = module.validate_environment_evidence(payload, reference_sidecar)
    assert normalized["numeric_runtime"]["OMP_NUM_THREADS"] == "8"

    changed = deepcopy(payload)
    changed["hostname"] = "private"
    with pytest.raises(module.AuditError, match="schema drift"):
        module.validate_environment_evidence(changed, reference_sidecar)
    changed = deepcopy(payload)
    changed["numeric_runtime"]["OMP_NUM_THREADS"] = "1"
    with pytest.raises(module.AuditError, match="OMP/MKL/OpenBLAS"):
        module.validate_environment_evidence(changed, reference_sidecar)


def _integrity_payload(module) -> dict:
    test_hashes = {
        relative: module._sha256(module._source_blob(ROOT, relative))
        for relative in module.EXPECTED_INTEGRITY_TEST_FILES
    }
    support_hashes = {
        relative: module._sha256(module._source_blob(ROOT, relative))
        for relative in module.EXPECTED_INTEGRITY_SUPPORT_FILES
    }
    required_nodes = sorted(
        {
            nodeid
            for nodeids in module.EXPECTED_INTEGRITY_REQUIREMENTS.values()
            for nodeid in nodeids
        }
    )
    executed = {nodeid: "passed" for nodeid in required_nodes}
    index = 0
    while len(executed) < 34:
        executed[
            f"tests/test_cgs_mutations.py::test_publication_filler_{index}"
        ] = "passed"
        index += 1
    return {
        "schema_version": 1,
        "evidence_kind": "post_result_integrity_test_execution",
        "source_commit": module.SOURCE_COMMIT,
        "execution_source": "clean_git_archive",
        "plugin_autoload_disabled": True,
        "repository_addopts_disabled": True,
        "runtime": {"python": platform.python_version(), "pytest": metadata.version("pytest")},
        "command": [
            "python",
            "-m",
            "pytest",
            "-q",
            "-o",
            "addopts=",
            "--strict-config",
            "--strict-markers",
            "--junitxml=<temporary-file>",
            *module.EXPECTED_INTEGRITY_TEST_FILES,
        ],
        "test_file_sha256": test_hashes,
        "support_file_sha256": support_hashes,
        "executed_tests": executed,
        "requirement_results": {
            requirement: {"status": "pass", "tests": list(nodeids)}
            for requirement, nodeids in module.EXPECTED_INTEGRITY_REQUIREMENTS.items()
        },
        "result": {
            "tests": 34,
            "failures": 0,
            "errors": 0,
            "skipped": 0,
            "passed": 34,
            "all_selected_tests_passed": True,
        },
        "frozen_g7_relationship": {
            "consumed_by_frozen_run": False,
            "publication_status": (
                "not_evaluated_in_frozen_run; separately_verified_post_result"
            ),
        },
    }


def test_integrity_evidence_requires_exact_suite_and_requirements() -> None:
    module = _module()
    valid = _integrity_payload(module)
    normalized = module.validate_integrity_evidence(valid, ROOT)
    assert len(normalized["executed_tests"]) == 34
    assert len(normalized["requirement_results"]) == 14

    changed = deepcopy(valid)
    changed["test_file_sha256"].pop(next(iter(changed["test_file_sha256"])))
    with pytest.raises(module.AuditError, match="exact eight"):
        module.validate_integrity_evidence(changed, ROOT)
    changed = deepcopy(valid)
    changed["result"]["skipped"] = 1
    changed["result"]["passed"] = 33
    with pytest.raises(module.AuditError, match="exact full pass"):
        module.validate_integrity_evidence(changed, ROOT)
    changed = deepcopy(valid)
    changed["requirement_results"].pop(next(iter(changed["requirement_results"])))
    with pytest.raises(module.AuditError, match="exact 14"):
        module.validate_integrity_evidence(changed, ROOT)


def test_current_result_manifest_artifacts_reject_worktree_drift(
    tmp_path: Path,
) -> None:
    module = _module()
    relative = "reports/cgs/result.json"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    source = b'{"sealed":true}\n'
    path.write_bytes(source)
    manifest = {"result_files": {relative: module._sha256(source)}}
    module._verify_current_result_manifest_artifacts(
        tmp_path,
        manifest,
        source_reader=lambda _: source,
    )
    path.write_bytes(b"tampered\n")
    with pytest.raises(module.AuditError, match="artifact changed"):
        module._verify_current_result_manifest_artifacts(
            tmp_path,
            manifest,
            source_reader=lambda _: source,
        )


def test_project_module_origin_must_be_inside_frozen_archive(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _module()
    archive = tmp_path / "archive"
    archive.mkdir()
    sentinel = SimpleNamespace(__file__=str(tmp_path / "current" / "tampered.py"))
    monkeypatch.setitem(sys.modules, "tearrl.audit_sentinel", sentinel)
    with pytest.raises(module.AuditError, match="outside frozen archive"):
        module._assert_frozen_module_origins(archive)
    assert "from tearrl" not in SCRIPT.read_text(encoding="utf-8")


def test_append_only_writer_refuses_existing_output(tmp_path: Path) -> None:
    module = _module()
    payload = {
        "corrected_pooling": {
            "stored_seed_mean_values": {
                "ece": 0.17,
                "selective_worst_party_fnr": 0.10,
                "manipulation_fnr_degradation": 0.003,
            },
            "registered_pooled_values": {
                "ece": 0.16,
                "selective_worst_party_fnr": 0.11,
                "selective_errors": 1,
                "selective_positive_accepted": 9,
                "manipulation_fnr_degradation": 0.0,
                "manipulation_difference_sum": 0,
                "manipulation_eligible_pairs": 10,
            },
        },
        "publication_corrections": {
            "seed_6311_manipulation": {"frozen_eight_thread_value": 0.055},
            "irreversible_wilson_omission": {"one_sided_wilson_upper": 0.286},
            "mae_interval_estimand": {"causal_only_rows": 18000},
        },
        "post_result_evidence": {
            "integrity": {"result": {"passed": 34}},
            "environment": {
                "python": {"version": "3.12.10"},
                "packages": {"numpy": "2.5.1", "torch": "2.11.0"},
            },
            "thread_sensitivity": {
                "reference": {"results_sha256": "1" * 64},
                "candidate": {"results_sha256": "2" * 64},
            },
        },
        "claim_boundary": "No frozen result changes.",
    }
    json_path = tmp_path / "erratum.json"
    markdown_path = tmp_path / "erratum.md"
    module.write_append_only(json_path, markdown_path, payload)
    assert json_path.is_file()
    assert "not evaluable" in markdown_path.read_text(encoding="utf-8")
    with pytest.raises(module.AuditError, match="already exists"):
        module.write_append_only(json_path, markdown_path, payload)


def test_private_local_paths_are_rejected_from_keys_and_values() -> None:
    module = _module()
    for value in (
        {"detail": r"C:\Users\example\file"},
        {r"\\server\private": "value"},
        {"detail": "/home/example/file"},
    ):
        with pytest.raises(module.AuditError, match="private local path"):
            module._reject_private_strings(value)