#!/usr/bin/env python
"""Run and bind post-result CG/S integrity evidence to an exact commit."""

from __future__ import annotations

import argparse
import hashlib
import io
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import tempfile
import xml.etree.ElementTree as ET

INTEGRITY_TEST_FILES = (
    "tests/test_cgs_mutations.py",
    "tests/test_cgs_witness.py",
    "tests/test_cgs_worlds.py",
    "tests/test_cgs_actors.py",
    "tests/test_cgs_governance.py",
    "tests/test_cgs_experiment.py",
    "tests/test_cgs_implementation_freeze.py",
    "tests/test_cgs_results_freeze.py",
)
INTEGRITY_SUPPORT_FILES = (
    "tests/conftest.py",
    "pyproject.toml",
)
REQUIREMENT_TESTS = {
    "action_mechanism_and_label_input_leakage": (
        "tests/test_cgs_mutations.py::test_extra_action_mechanism_source_or_test_label_feature_is_rejected",
        "tests/test_cgs_worlds.py::test_witness_input_contains_semantics_but_no_numeric_action_slot",
    ),
    "world_family_split_contamination": (
        "tests/test_cgs_mutations.py::test_split_family_contamination_mutation_fails",
    ),
    "witness_imports_charter_or_evaluator": (
        "tests/test_cgs_mutations.py::test_witness_import_graph_is_value_separated_by_ast_not_string_matching",
        "tests/test_cgs_witness.py::test_witness_module_has_no_charter_governor_actor_or_evaluator_import",
    ),
    "source_identity_or_test_labels_in_actor": (
        "tests/test_cgs_actors.py::test_actor_features_have_identical_source_agnostic_schema",
    ),
    "learned_oracle_schema_mismatch": (
        "tests/test_cgs_witness.py::test_oracle_uses_identical_fixed_report_interface",
    ),
    "shifted_oracle_report_row": (
        "tests/test_cgs_actors.py::test_shifted_report_row_is_rejected_before_actor_input",
    ),
    "removed_delayed_or_minority_outcomes": (
        "tests/test_cgs_mutations.py::test_removing_delayed_and_minority_labels_fails_nonvacuously",
    ),
    "manipulation_answer_flag_leakage": (
        "tests/test_cgs_mutations.py::test_manipulation_variant_is_evaluator_metadata_not_an_answer_flag",
    ),
    "wrong_party_or_damage_repair_credit": (
        "tests/test_cgs_actors.py::test_charter_target_disallows_unrelated_benefit_laundering",
        "tests/test_cgs_worlds.py::test_task_channel_is_never_negative_and_repairs_are_target_specific",
    ),
    "unmatched_factorial_budget_or_parameters": (
        "tests/test_cgs_mutations.py::test_factorial_seed_compute_and_parameter_mutations_fail",
        "tests/test_cgs_actors.py::test_flat_and_hierarchical_parameter_counts_are_matched_within_five_percent",
    ),
    "governor_on_used_for_governor_off_claim": (
        "tests/test_cgs_mutations.py::test_governor_on_evidence_cannot_be_labeled_governor_off",
    ),
    "rejected_action_executed_before_revision": (
        "tests/test_cgs_governance.py::test_rejected_tamper_is_never_executed_and_has_no_punishment",
    ),
    "ledger_history_mutation": (
        "tests/test_cgs_governance.py::test_ledger_is_append_only_to_callers_and_detects_history_mutation",
    ),
    "unfrozen_result_or_threshold_change": (
        "tests/test_cgs_implementation_freeze.py::test_cgs_implementation_and_protocol_files_match_freeze_chain",
        "tests/test_cgs_results_freeze.py::test_confirmatory_results_match_frozen_no_tuning_record",
    ),
}


def _git(root: Path, *args: str) -> bytes:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}")
    return completed.stdout


def _repository_root(start: Path) -> Path:
    return Path(
        _git(start, "rev-parse", "--show-toplevel").decode("utf-8").strip()
    ).resolve()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _bind_test_files(
    root: Path,
    revision: str,
) -> tuple[str, dict[str, str], dict[str, str]]:
    commit = _git(root, "rev-parse", f"{revision}^{{commit}}").decode("ascii").strip()
    test_digests = {
        relative: _sha256(_git(root, "show", f"{commit}:{relative}"))
        for relative in INTEGRITY_TEST_FILES
    }
    support_digests = {
        relative: _sha256(_git(root, "show", f"{commit}:{relative}"))
        for relative in INTEGRITY_SUPPORT_FILES
    }
    return commit, test_digests, support_digests


def _extract_git_archive(root: Path, commit: str, destination: Path) -> None:
    archive_bytes = _git(root, "archive", "--format=tar", commit)
    try:
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:") as archive:
            archive.extractall(destination, filter="data")
    except (tarfile.TarError, OSError) as exc:
        raise RuntimeError("could not extract the bound integrity-test archive") from exc

def _junit_counts(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    if root.tag == "testsuite":
        suites = (root,)
    else:
        suites = tuple(root.findall("testsuite"))
    if not suites:
        raise RuntimeError("pytest JUnit output contained no test suite")
    counts = {
        key: sum(int(float(suite.attrib.get(key, "0"))) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    counts["passed"] = (
        counts["tests"]
        - counts["failures"]
        - counts["errors"]
        - counts["skipped"]
    )
    return counts


def _junit_cases(path: Path) -> dict[str, str]:
    root = ET.parse(path).getroot()
    cases: dict[str, str] = {}
    for case in root.iter("testcase"):
        classname = case.attrib.get("classname", "").replace(".", "/")
        name = case.attrib.get("name", "")
        nodeid = f"{classname}.py::{name}"
        if case.find("failure") is not None:
            status = "failed"
        elif case.find("error") is not None:
            status = "error"
        elif case.find("skipped") is not None:
            status = "skipped"
        else:
            status = "passed"
        cases[nodeid] = status
    return cases


def _require_complete_pass(returncode: int, counts: dict[str, int]) -> None:
    if (
        returncode != 0
        or counts.get("failures") != 0
        or counts.get("errors") != 0
        or counts.get("skipped") != 0
        or counts.get("passed") != counts.get("tests")
    ):
        raise RuntimeError(
            "CG/S integrity evidence failed; inspect a direct pytest rerun"
        )

def run_integrity_evidence(root: Path, revision: str) -> dict[str, object]:
    commit, test_digests, support_digests = _bind_test_files(root, revision)
    with tempfile.TemporaryDirectory(prefix="tearrl-cgs-integrity-") as temporary:
        temporary_root = Path(temporary)
        archive_root = temporary_root / "source"
        archive_root.mkdir()
        _extract_git_archive(root, commit, archive_root)
        junit = temporary_root / "pytest.xml"
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-o",
            "addopts=",
            "--strict-config",
            "--strict-markers",
            f"--junitxml={junit}",
            *INTEGRITY_TEST_FILES,
        ]
        environment = os.environ.copy()
        environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
        environment.pop("PYTHONHOME", None)
        environment.pop("PYTHONPATH", None)
        completed = subprocess.run(
            command,
            cwd=archive_root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if not junit.is_file():
            raise RuntimeError("pytest did not produce JUnit evidence")
        counts = _junit_counts(junit)
        cases = _junit_cases(junit)
        _require_complete_pass(completed.returncode, counts)
        required_nodes = {
            nodeid for nodeids in REQUIREMENT_TESTS.values() for nodeid in nodeids
        }
        missing = sorted(required_nodes - cases.keys())
        if missing:
            raise RuntimeError(f"required integrity tests were not executed: {missing}")
        requirement_results = {
            requirement: {
                "status": (
                    "pass"
                    if all(cases[nodeid] == "passed" for nodeid in nodeids)
                    else "not_demonstrated"
                ),
                "tests": list(nodeids),
            }
            for requirement, nodeids in REQUIREMENT_TESTS.items()
        }
        if any(item["status"] != "pass" for item in requirement_results.values()):
            raise RuntimeError("one or more integrity requirements were not demonstrated")

    return {
        "schema_version": 1,
        "evidence_kind": "post_result_integrity_test_execution",
        "source_commit": commit,
        "execution_source": "clean_git_archive",
        "plugin_autoload_disabled": True,
        "repository_addopts_disabled": True,
        "runtime": {
            "python": platform.python_version(),
            "pytest": metadata.version("pytest"),
        },
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
            *INTEGRITY_TEST_FILES,
        ],
        "test_file_sha256": test_digests,
        "support_file_sha256": support_digests,
        "executed_tests": dict(sorted(cases.items())),
        "requirement_results": requirement_results,
        "result": {
            **counts,
            "all_selected_tests_passed": True,
        },
        "frozen_g7_relationship": {
            "consumed_by_frozen_run": False,
            "publication_status": (
                "not_evaluated_in_frozen_run; selected_integrity_tests_passed_post_result"
            ),
        },
    }

def _relative_or_name(path: Path, root: Path) -> str:
    """Return a useful output label without exposing an absolute caller path."""

    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name

def _write_new_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = _repository_root(Path.cwd())
    output = args.output if args.output.is_absolute() else root / args.output
    if output.exists():
        parser.error("refusing to overwrite existing integrity evidence")
    payload = run_integrity_evidence(root, args.commit)
    try:
        _write_new_text(
            output,
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
        )
    except FileExistsError:
        parser.error("refusing to overwrite existing integrity evidence")
    print(_relative_or_name(output, root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())