from __future__ import annotations

import json
from pathlib import Path

from tearrl.cpp.freeze import (
    matches_committed_blob_sha256,
    verify_freeze_chain,
)


ROOT = Path(__file__).resolve().parents[1]
CPP = ROOT / "tearrl" / "cpp"
MANIFEST = CPP / "frozen_implementation.json"
PREREG = CPP / "frozen_prereg.json"
ANCESTORS = CPP / "frozen_ancestors.json"


def _expected_implementation_paths() -> set[str]:
    paths = {
        "docs/cpp_implementation.md",
        "docs/cpp_prereg.md",
        "scripts/build_cpp_freeze.py",
        "scripts/run_cpp.py",
        "tearrl/cpp/frozen_ancestors.json",
        "tearrl/cpp/frozen_prereg.json",
    }
    paths.update(
        path.relative_to(ROOT).as_posix()
        for path in CPP.glob("*.py")
    )
    paths.update(
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "tests").glob("test_cpp_*.py")
        if path.name != "test_cpp_results_freeze.py"
    )
    return paths


def test_implementation_manifest_binds_complete_preoutcome_surface() -> None:
    frozen = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert frozen["status"] == "frozen_before_preflight_and_outcomes"
    assert frozen["hash_domain"] == "committed_git_blob_bytes"
    assert frozen["parent_commit"] == (
        "2f507dcee27b22952d9c35d9fa2655b642753a04"
    )
    assert frozen["preregistration_commit"] == frozen["parent_commit"]
    assert frozen["preregistration_sha256"] == (
        "9a2cbf6d26cf847762eeb489ab83784c2e640a70ec98b6f4090a48b0d3e6364d"
    )
    assert matches_committed_blob_sha256(
        PREREG, frozen["preregistration_manifest_sha256"]
    )
    assert matches_committed_blob_sha256(
        ANCESTORS, frozen["ancestor_manifest_sha256"]
    )
    assert set(frozen["files"]) == _expected_implementation_paths()
    assert frozen["implementation_file_count"] == len(frozen["files"])
    for relative, expected in frozen["files"].items():
        assert matches_committed_blob_sha256(ROOT / relative, expected), relative


def test_complete_freeze_chain_verifies() -> None:
    receipt = verify_freeze_chain(ROOT)
    assert receipt["preregistration_sha256"] == (
        "9a2cbf6d26cf847762eeb489ab83784c2e640a70ec98b6f4090a48b0d3e6364d"
    )
    assert receipt["ancestor_files"] == 179
    assert receipt["implementation_files"] > 20
