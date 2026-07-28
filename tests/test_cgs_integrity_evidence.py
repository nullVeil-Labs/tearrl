from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_cgs_integrity_evidence.py"


def _module():
    spec = importlib.util.spec_from_file_location("run_cgs_integrity_evidence", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_junit_counts_exclude_failures_errors_and_skips(tmp_path: Path) -> None:
    module = _module()
    junit = tmp_path / "results.xml"
    junit.write_text(
        '<testsuites><testsuite tests="8" failures="1" errors="2" skipped="1"/>'
        '</testsuites>',
        encoding="utf-8",
    )
    assert module._junit_counts(junit) == {
        "tests": 8,
        "failures": 1,
        "errors": 2,
        "skipped": 1,
        "passed": 4,
    }


def test_integrity_file_inventory_is_explicit_and_public() -> None:
    module = _module()
    assert len(module.INTEGRITY_TEST_FILES) == 8
    assert all(path.startswith("tests/test_cgs_") for path in module.INTEGRITY_TEST_FILES)
    assert len(set(module.INTEGRITY_TEST_FILES)) == len(module.INTEGRITY_TEST_FILES)
    assert len(module.REQUIREMENT_TESTS) == 14
    assert all(nodeids for nodeids in module.REQUIREMENT_TESTS.values())


def test_junit_cases_retain_node_ids_and_status(tmp_path: Path) -> None:
    module = _module()
    junit = tmp_path / "results.xml"
    junit.write_text(
        '<testsuites><testsuite>'
        '<testcase classname="tests.test_cgs_mutations" name="test_pass"/>'
        '<testcase classname="tests.test_cgs_mutations" name="test_skip">'
        '<skipped/></testcase>'
        '</testsuite></testsuites>',
        encoding="utf-8",
    )
    assert module._junit_cases(junit) == {
        "tests/test_cgs_mutations.py::test_pass": "passed",
        "tests/test_cgs_mutations.py::test_skip": "skipped",
    }

def test_output_label_is_relative_or_basename_only(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path / "repo"
    root.mkdir()
    inside = root / "reports" / "evidence.json"
    outside = tmp_path / "private" / "evidence.json"

    assert module._relative_or_name(inside, root) == "reports/evidence.json"
    assert module._relative_or_name(outside, root) == "evidence.json"
    assert str(tmp_path) not in module._relative_or_name(outside, root)

def test_complete_pass_rejects_skips_and_inconsistent_counts() -> None:
    module = _module()
    module._require_complete_pass(
        0,
        {"tests": 34, "failures": 0, "errors": 0, "skipped": 0, "passed": 34},
    )
    with pytest.raises(RuntimeError, match="integrity evidence failed"):
        module._require_complete_pass(
            0,
            {"tests": 34, "failures": 0, "errors": 0, "skipped": 1, "passed": 33},
        )
    with pytest.raises(RuntimeError, match="integrity evidence failed"):
        module._require_complete_pass(
            0,
            {"tests": 34, "failures": 0, "errors": 0, "skipped": 0, "passed": 33},
        )
def test_exclusive_writer_preserves_existing_evidence(tmp_path: Path) -> None:
    module = _module()
    output = tmp_path / "integrity.json"
    output.write_text("sealed\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        module._write_new_text(output, "replacement\n")
    assert output.read_text(encoding="utf-8") == "sealed\n"