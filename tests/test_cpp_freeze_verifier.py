from __future__ import annotations

import json
from pathlib import Path

import pytest

from tearrl.cpp.freeze import (
    FreezeVerificationError,
    sha256_file,
    verify_ancestors,
    verify_preregistration,
)


ROOT = Path(__file__).resolve().parents[1]


def test_current_preregistration_and_ancestor_freezes_verify() -> None:
    prereg = verify_preregistration(ROOT)
    ancestors = verify_ancestors(ROOT)
    assert prereg["parent_commit"] == ancestors["base_commit"]
    assert len(ancestors["files"]) == 179


def test_preregistration_verifier_fails_closed_on_mutation(tmp_path: Path) -> None:
    cpp = tmp_path / "tearrl" / "cpp"
    docs = tmp_path / "docs"
    cpp.mkdir(parents=True)
    docs.mkdir()
    document = docs / "cpp_prereg.md"
    document.write_text("frozen\n", encoding="utf-8")
    manifest = {
        "document": "docs/cpp_prereg.md",
        "sha256": sha256_file(document),
        "hash_domain": "committed_git_blob_bytes",
        "status": "frozen_before_implementation_and_outcomes",
        "parent_commit": "0" * 40,
    }
    (cpp / "frozen_prereg.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    assert verify_preregistration(tmp_path)["sha256"] == sha256_file(document)

    document.write_text("mutated\n", encoding="utf-8")
    with pytest.raises(FreezeVerificationError, match="digest mismatch"):
        verify_preregistration(tmp_path)


def test_freeze_path_cannot_escape_repository(tmp_path: Path) -> None:
    cpp = tmp_path / "tearrl" / "cpp"
    cpp.mkdir(parents=True)
    manifest = {
        "document": "../outside.md",
        "sha256": "0" * 64,
        "hash_domain": "committed_git_blob_bytes",
        "status": "frozen_before_implementation_and_outcomes",
        "parent_commit": "0" * 40,
    }
    (cpp / "frozen_prereg.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with pytest.raises(FreezeVerificationError, match="unsafe frozen path"):
        verify_preregistration(tmp_path)
