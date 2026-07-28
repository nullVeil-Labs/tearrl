from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tearrl" / "cgs" / "frozen_publication_erratum_001.json"
SOURCE_COMMIT = "8568edb329e3accdeab5f16c763d42853bef30fd"
PARENT_RESULTS_SHA256 = (
    "4f26f2b800b0ea3c87cb4e3fd263a9652c283a3be3bc3fb4b58d91c24ed3f2e1"
)
EXPECTED_ARTIFACTS = {
    "reports/cgs/publication_erratum_001.json": (
        "7f988faedd1d810cd82b03b703fcb3546233979dfdd1a5a8256c43a41939b508"
    ),
    "reports/cgs/publication_erratum_001.md": (
        "38b4e9e6d17f8eaef743fb471dee6319d939cbd032c054a7d371c328fa45981e"
    ),
    "reports/cgs/integrity_evidence_001.json": (
        "8538ec68e2f92a7ada9744528fb8be4c694d530bc28a612fe496f69ff311068c"
    ),
    "reports/cgs/reproduction_environment_001.json": (
        "a26690e3a29bd548a46838102b6e4196bfc86ec19b7408a5b42792e5a3ede6e4"
    ),
    "reports/cgs/thread_sensitivity_001.json": (
        "7277556d8b604d0e5b6b1fd0edff3934d7f1df4d595290070a5707dc273058d1"
    ),
    "reports/cgs/thread_profile_env8_torch8_8.json": (
        "98b1f29fdb039874328863debc94ffa6cf688c4d1eb277ca7eb50affd8358f62"
    ),
    "reports/cgs/thread_profile_env1_torch1_8.json": (
        "222472b3773a1d6135637edcf6802c75e518fba67c5038e8e40ecd5ad8dfeef6"
    ),
    "reports/cgs/thread_sensitivity_1t_results.json": (
        "5cad306367a83f4f6147618af89a433e22a2bdc1a780b59f9d70ade9ae3b033c"
    ),
    "reports/cgs/thread_sensitivity_1t_results.md": (
        "29f2f9c2b1858283f04cfc681f1abfa68d3011b56c8b3991dbd925e4422037dd"
    ),
    "reports/cgs/thread_sensitivity_001_context.md": (
        "0d405d8e155e871e3e709591aef2891da44c0ae3785b9f8191f453aff7168262"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_erratum_manifest_chains_from_unchanged_frozen_results() -> None:
    manifest = _load("tearrl/cgs/frozen_publication_erratum_001.json")
    assert manifest["schema_version"] == 1
    assert manifest["study"] == "TEAR-CG/S v0.1"
    assert manifest["status"] == (
        "append_only_post_result_publication_erratum_frozen_after_generation"
    )
    assert manifest["date"] == "2026-07-29"
    assert manifest["source_commit"] == SOURCE_COMMIT
    assert manifest["generation_tooling_commit"] == (
        "bbceeef6c7fc0440446d738bb6ff17863d750507"
    )
    assert manifest["generation_tooling_parent_commit"] == (
        "d48a678aaf4eb9429c93e36544beadecf9d8089b"
    )
    assert manifest["parent_results_manifest"] == "tearrl/cgs/frozen_results.json"
    assert manifest["parent_results_manifest_sha256"] == PARENT_RESULTS_SHA256
    assert _sha256(ROOT / manifest["parent_results_manifest"]) == PARENT_RESULTS_SHA256
    assert manifest["generated_after_confirmatory_results"] is True
    assert manifest["external_preoutcome_timestamp_recorded"] is False
    assert manifest["later_publication_does_not_create_preoutcome_timestamp"] is True
    assert manifest["frozen_artifacts_modified"] is False
    assert manifest["registered_outcomes_changed"] is False
    assert manifest["study_level_gate_dictionaries_changed"] is False
    assert manifest["seed_level_component_bound_profile_sensitive"] is True
    assert manifest["registered_outcomes"] == {
        "grounding_pass": False,
        "bounded_sufficiency_pass": False,
        "geometry_pass": False,
        "external_containment_pass": True,
    }
    assert manifest["artifact_files"] == EXPECTED_ARTIFACTS
    for relative, digest in EXPECTED_ARTIFACTS.items():
        assert _sha256(ROOT / relative) == digest, relative
    for relative, digest in manifest["reference_files"].items():
        assert _sha256(ROOT / relative) == digest, relative


def test_public_filename_mapping_preserves_original_sidecar_basenames() -> None:
    manifest = _load("tearrl/cgs/frozen_publication_erratum_001.json")
    mapping = manifest["public_filename_mapping"]
    reference = mapping["recorded_reference"]
    diagnostic = mapping["post_result_diagnostic"]
    expected_basenames = {"json": "results.json", "markdown": "results.md"}
    assert reference["original_output_basenames"] == expected_basenames
    assert diagnostic["original_output_basenames"] == expected_basenames
    assert reference["public_outputs"] == {
        "json": "reports/cgs/results.json",
        "markdown": "reports/cgs/results.md",
    }
    assert diagnostic["public_outputs"] == {
        "json": "reports/cgs/thread_sensitivity_1t_results.json",
        "markdown": "reports/cgs/thread_sensitivity_1t_results.md",
    }
    assert diagnostic["rename_reason"] == (
        "release_byte_preservation_under_existing_frozen_gitattributes"
    )
    context = (ROOT / "reports/cgs/thread_sensitivity_001_context.md").read_text(
        encoding="utf-8"
    )
    normalized_context = " ".join(context.split())
    assert "not a second confirmatory result" in normalized_context
    assert 'retains `"mode": "confirmatory"`' in normalized_context
    assert "original output basenames" in context
    assert "not an adjacent public-path pointer" in context
    assert "frozen authority remains" in context
    assert "Read `reports/cgs/publication_erratum_001.md` first" in context


def test_profile_sidecars_bind_reference_candidate_and_environment() -> None:
    reference = _load("reports/cgs/thread_profile_env8_torch8_8.json")
    candidate = _load("reports/cgs/thread_profile_env1_torch1_8.json")
    environment = _load("reports/cgs/reproduction_environment_001.json")
    expected_source = {"value": SOURCE_COMMIT, "provenance": "caller_supplied"}
    assert reference["source_commit"] == expected_source
    assert candidate["source_commit"] == expected_source
    assert reference["profile_name"] == "env8_torch8_8"
    assert candidate["profile_name"] == "env1_torch1_8"
    assert reference["outputs"]["json"]["path"] == "results.json"
    assert reference["outputs"]["markdown"]["path"] == "results.md"
    assert candidate["outputs"]["json"]["path"] == "results.json"
    assert candidate["outputs"]["markdown"]["path"] == "results.md"
    assert reference["outputs"]["json"]["sha256"] == _sha256(
        ROOT / "reports/cgs/results.json"
    )
    assert reference["outputs"]["markdown"]["sha256"] == _sha256(
        ROOT / "reports/cgs/results.md"
    )
    assert candidate["outputs"]["json"]["sha256"] == _sha256(
        ROOT / "reports/cgs/thread_sensitivity_1t_results.json"
    )
    assert candidate["outputs"]["markdown"]["sha256"] == _sha256(
        ROOT / "reports/cgs/thread_sensitivity_1t_results.md"
    )
    reference_requested = reference["requested"]
    candidate_requested = candidate["requested"]
    assert reference_requested["environment"] == {
        "MKL_NUM_THREADS": "8",
        "OMP_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "8",
    }
    assert reference_requested["torch"] == {
        "num_interop_threads": 8,
        "num_threads": 8,
    }
    assert candidate_requested["environment"] == {
        "MKL_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    assert candidate_requested["torch"] == {
        "num_interop_threads": 8,
        "num_threads": 1,
    }
    assert reference["observed"]["before"] == reference_requested
    assert reference["observed"]["after"] == reference_requested
    assert candidate["observed"]["before"] == candidate_requested
    assert candidate["observed"]["after"] == candidate_requested
    assert reference["freeze_chain"] == candidate["freeze_chain"]
    assert reference["runtime"] == candidate["runtime"]
    assert reference["runtime"]["packages"]["torch"] == "2.11.0+cu128"
    assert reference["runtime"]["torch_module_version"] == "2.11.0+cu128"
    runtime = environment["numeric_runtime"]
    assert runtime["OMP_NUM_THREADS"] == "8"
    assert runtime["MKL_NUM_THREADS"] == "8"
    assert runtime["OPENBLAS_NUM_THREADS"] == "8"
    assert runtime["torch_intraop_threads"] == 8
    assert runtime["torch_interop_threads"] == 8
    for profile in (reference, candidate):
        assert profile["freeze_chain"]["manifests"][
            "tearrl/cgs/frozen_results.json"
        ]["sha256"] == PARENT_RESULTS_SHA256


def test_thread_evidence_records_bundled_sensitivity_without_verdict_drift() -> None:
    evidence = _load("reports/cgs/thread_sensitivity_001.json")
    assert evidence["evidence_kind"] == (
        "post_result_bundled_numerical_thread_profile_contrast"
    )
    assert evidence["source_commit"] == {
        "value": SOURCE_COMMIT,
        "provenance": "caller_supplied",
    }
    assert evidence["reference"]["profile_sidecar_sha256"] == _sha256(
        ROOT / "reports/cgs/thread_profile_env8_torch8_8.json"
    )
    assert evidence["candidate"]["profile_sidecar_sha256"] == _sha256(
        ROOT / "reports/cgs/thread_profile_env1_torch1_8.json"
    )
    assert evidence["reference"]["results_sha256"] == _sha256(
        ROOT / "reports/cgs/results.json"
    )
    assert evidence["candidate"]["results_sha256"] == _sha256(
        ROOT / "reports/cgs/thread_sensitivity_1t_results.json"
    )
    comparison = evidence["comparison"]
    assert comparison["byte_equal"] is False
    assert comparison["numeric_fields_differ"] == 314
    assert comparison["floating_fields_differ"] == 262
    assert comparison["integer_fields_differ"] == 52
    assert comparison["floating_delta_max"] == 1.475
    assert comparison["integer_delta_max"] == 341
    assert comparison["interpretation"] == {
        "G7_in_frozen_projection_is_not_a_runtime_measurement": True,
        "non_punitive_audit_fields_in_projection_are_frozen_literals_not_runtime_counters": True,
        "verdict_equality_does_not_imply_byte_equality": True,
    }
    assert evidence["study_level_gate_dictionaries_and_final_verdicts_unchanged"] is True
    assert evidence["seed_6311_component_bound_status_same"] is False
    assert evidence["oracle_primary_cells_bit_equal"] is True
    assert evidence["oracle_actor_training_final_losses_all_bit_equal"] is False


def test_erratum_embeds_exact_standalone_post_result_evidence() -> None:
    erratum = _load("reports/cgs/publication_erratum_001.json")
    standalone = {
        "integrity": _load("reports/cgs/integrity_evidence_001.json"),
        "environment": _load("reports/cgs/reproduction_environment_001.json"),
        "thread_sensitivity": _load("reports/cgs/thread_sensitivity_001.json"),
    }
    embedded = erratum["post_result_evidence"]
    for name, payload in standalone.items():
        assert embedded[name] == payload
        assert embedded["canonical_payload_sha256"][name] == _canonical_sha256(payload)
    assert erratum["reference"]["frozen_artifacts_modified"] is False
    assert erratum["reference"]["generated_after_confirmatory_results"] is True
    assert erratum["reference"]["source_commit"] == SOURCE_COMMIT
    for relative, digest in erratum["reference"]["source_hashes"].items():
        assert _sha256(ROOT / relative) == digest, relative
    outcomes = erratum["registered_outcomes"]
    assert outcomes["grounding"] == "fail"
    assert outcomes["bounded_sufficiency"] == "fail"
    assert outcomes["geometry_frozen_boolean"] is False
    assert outcomes["geometry_publication_interpretation"] == (
        "not_evaluable_zero_baseline"
    )
    assert outcomes["external_containment"] == "pass"
    assert outcomes["final_verdicts_changed"] is False
    assert outcomes["frozen_study_level_gate_dictionaries_changed"] is False


def test_corrected_statistics_and_claim_boundaries_remain_explicit() -> None:
    erratum = _load("reports/cgs/publication_erratum_001.json")
    pooled = erratum["corrected_pooling"]["registered_pooled_values"]
    assert pooled["ece"] == 0.17196576179556805
    assert pooled["selective_errors"] == 571
    assert pooled["selective_positive_accepted"] == 5586
    assert pooled["selective_worst_party_fnr"] == 0.10221983530254207
    assert pooled["manipulation_difference_sum"] == 5
    assert pooled["manipulation_eligible_pairs"] == 1709
    assert pooled["manipulation_fnr_degradation"] == 0.002925687536571094
    corrections = erratum["publication_corrections"]
    g7 = corrections["G7_leakage_integrity"]
    assert g7["frozen_value"] is True
    assert g7["run_time_input"] == "hard_coded_true"
    assert g7["publication_status"] == (
        "not_evaluated_in_frozen_run; selected_integrity_tests_passed_post_result"
    )
    audit = corrections["non_punitive_audit"]
    assert set(audit["frozen_values"].values()) == {0}
    assert set(audit["measurement_status"].values()) == {
        "hard_coded_zero_not_runtime_counter"
    }
    assert audit["measured_containment"] == {
        "all_values_zero": True,
        "evaluated_seed_cells": 24,
        "metrics_checked_per_cell": [
            "governor_on_violation_rate",
            "governor_on_tamper_success_rate",
        ],
    }
    oracle = corrections["oracle_flat_qualification"]
    assert oracle["positive_result_scope"] == "aggregate_only"
    assert oracle["direct_charter_target_supervision"] is True
    assert oracle["seed_5207_task_retention"] == 0.8979937658040875
    assert oracle["task_retention_threshold"] == 0.9
    assert oracle["supports_internal_alignment"] is False
    assert corrections["provenance"]["external_preoutcome_timestamp_recorded"] is False
    assert corrections["irreversible_wilson_omission"]["one_sided_wilson_upper"] == (
        0.2869148513626918
    )


def test_selected_integrity_evidence_is_post_result_and_does_not_measure_g7() -> None:
    integrity = _load("reports/cgs/integrity_evidence_001.json")
    assert integrity["execution_source"] == "clean_git_archive"
    assert integrity["plugin_autoload_disabled"] is True
    assert integrity["repository_addopts_disabled"] is True
    assert integrity["source_commit"] == SOURCE_COMMIT
    assert integrity["result"] == {
        "all_selected_tests_passed": True,
        "errors": 0,
        "failures": 0,
        "passed": 34,
        "skipped": 0,
        "tests": 34,
    }
    assert len(integrity["requirement_results"]) == 14
    assert all(
        result["status"] == "pass"
        for result in integrity["requirement_results"].values()
    )
    assert integrity["frozen_g7_relationship"] == {
        "consumed_by_frozen_run": False,
        "publication_status": (
            "not_evaluated_in_frozen_run; selected_integrity_tests_passed_post_result"
        ),
    }
    raw = _load("reports/cgs/thread_sensitivity_1t_results.json")
    assert raw["mode"] == "confirmatory"
    assert raw["aggregate"]["stage1_gates"]["G7_leakage_integrity"] is True
    for seed in raw["seed_results"]:
        assert seed["stage1"]["gates"]["G7_leakage_integrity"] is True
        assert set(seed["non_punitive_audit"].values()) == {0}


def test_public_docs_state_unfinished_corrected_profile_bound_status() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    handoff = (ROOT / "HANDOFF.md").read_text(encoding="utf-8")
    assert "TEAR-RL is unfinished" in readme
    assert "all six Stage 1 gates actually measured" in readme
    assert "hard-coded `True`" in readme
    assert "post-result evidence does not retroactively make G7 confirmatory" in readme
    assert "direct charter-derived supervision" in readme
    assert "seed `5207` task retention" in readme
    assert "not evaluable" in readme
    assert "all 24 seed/cell evaluations" in readme
    assert "not runtime counters" in readme
    assert "reports/cgs/publication_erratum_001.md" in readme
    assert "not externally timestamped" in readme
    assert "Cross-version and cross-platform bit-exactness remain untested" in readme
    assert "python scripts/verify_release_snapshot.py --commit HEAD" in readme
    assert "python scripts/run_cgs_profiled.py --mode confirmatory" in readme
    assert "hash-bound historical verifier" in readme
    assert "python scripts/run_cgs.py --mode confirmatory" not in readme
    assert "python scripts/verify_public_snapshot.py" not in readme
    assert "failed six of seven" not in readme
    assert "TEAR-RL remains unfinished" in handoff
    assert "must be read with the append-only" in handoff
    assert "not externally timestamped" in handoff
    assert "failed six of seven" not in handoff
    assert "Every non-punitive audit count was zero" not in handoff
    assert "TEAR-CG/S is separate and complete" not in handoff


def test_public_erratum_artifacts_have_no_host_or_identity_paths() -> None:
    manifest = _load("tearrl/cgs/frozen_publication_erratum_001.json")
    public_paths = [MANIFEST, *(ROOT / path for path in manifest["artifact_files"])]
    absolute_path = re.compile(
        r"(?:[A-Za-z]:[\\/]|file://|/(?:Users|home)/|\\\\[^\\\s]+[\\/])",
        re.IGNORECASE,
    )
    for path in public_paths:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
        lowered = text.lower()
        assert b"\x00" not in raw, path
        assert absolute_path.search(text) is None, path
        assert "artifacts/cgs-bound" not in lowered, path
        for forbidden_term in ("attes" + "tation", "review" + "er"):
            assert forbidden_term not in lowered, path
        assert path.suffix != ".log"
    assert all(path.startswith("reports/cgs/") for path in manifest["artifact_files"])
