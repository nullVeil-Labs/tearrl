from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from tearrl.cpp.freeze import matches_committed_blob_sha256


ROOT = Path(__file__).resolve().parents[1]
CPP = ROOT / "tearrl" / "cpp"
ANCESTORS = CPP / "frozen_ancestors.json"
FORBIDDEN_MODULES = {
    "tearrl.agents",
    "tearrl.cc",
    "tearrl.cgs",
    "tearrl.spec",
    "tearrl.witness",
    "tearrl.world",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_every_pre_cpp_file_remains_byte_identical() -> None:
    manifest = json.loads(ANCESTORS.read_text(encoding="utf-8"))
    assert manifest["hash_domain"] == "committed_git_blob_bytes"
    assert len(manifest["files"]) == 179
    for relative, expected in manifest["files"].items():
        path = ROOT.joinpath(*relative.split("/"))
        assert path.is_file(), relative
        assert matches_committed_blob_sha256(path, expected), relative


def test_cpp_is_a_clean_room_package() -> None:
    sources = sorted(CPP.glob("*.py"))
    assert sources
    for path in sources:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                assert node.level <= 1, f"parent-package import in {path.name}"
                imported = [node.module or ""]
            else:
                continue
            for module in imported:
                assert not any(
                    module == forbidden or module.startswith(f"{forbidden}.")
                    for forbidden in FORBIDDEN_MODULES
                ), f"{path.name} imports frozen predecessor module {module}"


def test_high_visibility_predecessor_documents_are_bound() -> None:
    manifest = json.loads(ANCESTORS.read_text(encoding="utf-8"))
    for relative in ("README.md", "HANDOFF.md", "reports/research_status.md"):
        assert relative in manifest["files"]
        assert matches_committed_blob_sha256(
            ROOT / relative, manifest["files"][relative]
        )
