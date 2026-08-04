from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_cpp_freeze.py"
SPEC = importlib.util.spec_from_file_location("build_cpp_freeze_for_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
BUILDER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BUILDER
SPEC.loader.exec_module(BUILDER)


def _git(root: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def _write(root: Path, relative: str, payload: bytes) -> None:
    path = root.joinpath(*relative.split("/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@pytest.fixture()
def staged_cpp_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    _git(tmp_path, "init", "--quiet")
    _git(tmp_path, "config", "user.name", "CPP Freeze Test")
    _git(tmp_path, "config", "user.email", "cpp-freeze@example.invalid")

    baseline_payload = b"frozen predecessor\n"
    _write(tmp_path, "README.md", baseline_payload)
    _git(tmp_path, "add", "--", "README.md")
    _git(tmp_path, "commit", "--quiet", "-m", "baseline")
    base_commit = _git(tmp_path, "rev-parse", "HEAD").decode("ascii").strip()

    preregistration_payload = b"# synthetic frozen preregistration\n"
    preregistration_sha256 = _sha256(preregistration_payload)
    preregistration_manifest = {
        "document": BUILDER.PREREGISTRATION_PATH,
        "sha256": preregistration_sha256,
        "hash_domain": "committed_git_blob_bytes",
        "status": "frozen_before_implementation_and_outcomes",
        "parent_commit": base_commit,
    }
    ancestor_manifest = {
        "algorithm": "sha256",
        "base_commit": base_commit,
        "files": {"README.md": _sha256(baseline_payload)},
        "hash_domain": "committed_git_blob_bytes",
        "tracked_file_count": 1,
    }
    _write(
        tmp_path,
        BUILDER.PREREGISTRATION_PATH,
        preregistration_payload,
    )
    _write(
        tmp_path,
        BUILDER.PREREGISTRATION_MANIFEST_PATH,
        (json.dumps(preregistration_manifest, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )
    _write(
        tmp_path,
        BUILDER.ANCESTOR_MANIFEST_PATH,
        (json.dumps(ancestor_manifest, sort_keys=True) + "\n").encode("utf-8"),
    )
    _git(
        tmp_path,
        "add",
        "--",
        BUILDER.PREREGISTRATION_PATH,
        BUILDER.PREREGISTRATION_MANIFEST_PATH,
        BUILDER.ANCESTOR_MANIFEST_PATH,
    )
    _git(tmp_path, "commit", "--quiet", "-m", "preregister")
    parent_commit = _git(tmp_path, "rev-parse", "HEAD").decode("ascii").strip()

    monkeypatch.setattr(BUILDER, "EXPECTED_PARENT_COMMIT", parent_commit)
    monkeypatch.setattr(
        BUILDER,
        "EXPECTED_ANCESTOR_BASE_COMMIT",
        base_commit,
    )
    monkeypatch.setattr(BUILDER, "EXPECTED_ANCESTOR_FILE_COUNT", 1)
    monkeypatch.setattr(
        BUILDER,
        "EXPECTED_PREREGISTRATION_SHA256",
        preregistration_sha256,
    )

    for index, relative in enumerate(sorted(BUILDER.EXPECTED_ADDITION_PATHS)):
        _write(tmp_path, relative, f"indexed-{index}-{relative}\n".encode("utf-8"))
    _git(
        tmp_path,
        "add",
        "--",
        *sorted(BUILDER.EXPECTED_ADDITION_PATHS),
    )
    return tmp_path


def test_write_hashes_exact_index_blobs_and_never_changes_index(
    staged_cpp_repo: Path,
) -> None:
    diverged = "scripts/run_cpp.py"
    staged_bytes = _git(staged_cpp_repo, "show", f":{diverged}")
    _write(staged_cpp_repo, diverged, b"working tree differs after staging\n")
    index_before = _git(staged_cpp_repo, "ls-files", "--stage", "-z")

    manifest = BUILDER.write_manifest(staged_cpp_repo)

    assert manifest["manifest_construction_source"] == (
        "git_index_stage_0_blob_bytes"
    )
    assert set(manifest["files"]) == set(BUILDER.FROZEN_IMPLEMENTATION_PATHS)
    assert BUILDER.MANIFEST_PATH not in manifest["files"]
    assert manifest["files"][diverged] == _sha256(staged_bytes)
    assert manifest["files"][diverged] != _sha256(
        (staged_cpp_repo / diverged).read_bytes()
    )
    assert _git(staged_cpp_repo, "ls-files", "--stage", "-z") == index_before
    assert (
        BUILDER.MANIFEST_PATH
        not in _git(staged_cpp_repo, "diff", "--cached", "--name-only")
        .decode("utf-8")
        .splitlines()
    )


def test_write_refuses_overwrite(staged_cpp_repo: Path) -> None:
    first = BUILDER.write_manifest(staged_cpp_repo)
    manifest_path = staged_cpp_repo / BUILDER.MANIFEST_PATH
    frozen_bytes = manifest_path.read_bytes()

    with pytest.raises(BUILDER.FreezeBuildError, match="refusing to overwrite"):
        BUILDER.write_manifest(staged_cpp_repo)

    assert manifest_path.read_bytes() == frozen_bytes
    assert json.loads(frozen_bytes) == first


def test_verify_staged_requires_canonical_manifest_from_current_index(
    staged_cpp_repo: Path,
) -> None:
    expected = BUILDER.write_manifest(staged_cpp_repo)
    _git(staged_cpp_repo, "add", "--", BUILDER.MANIFEST_PATH)
    assert BUILDER.verify_staged_manifest(staged_cpp_repo) == expected

    manifest_path = staged_cpp_repo / BUILDER.MANIFEST_PATH
    mutated = dict(expected)
    mutated["implementation_file_count"] = 0
    manifest_path.write_text(json.dumps(mutated), encoding="utf-8")
    _git(staged_cpp_repo, "add", "--", BUILDER.MANIFEST_PATH)
    with pytest.raises(BUILDER.FreezeBuildError, match="canonical index-derived"):
        BUILDER.verify_staged_manifest(staged_cpp_repo)


def test_unrelated_or_missing_staged_paths_fail_closed(
    staged_cpp_repo: Path,
) -> None:
    _write(staged_cpp_repo, "unrelated.txt", b"must not be frozen\n")
    _git(staged_cpp_repo, "add", "--", "unrelated.txt")
    with pytest.raises(BUILDER.FreezeBuildError, match="unrelated staged paths"):
        BUILDER.write_manifest(staged_cpp_repo)
    assert not (staged_cpp_repo / BUILDER.MANIFEST_PATH).exists()

    _git(staged_cpp_repo, "reset", "--quiet", "HEAD", "--", "unrelated.txt")
    missing = sorted(BUILDER.EXPECTED_ADDITION_PATHS)[0]
    _git(staged_cpp_repo, "reset", "--quiet", "HEAD", "--", missing)
    with pytest.raises(BUILDER.FreezeBuildError, match="missing staged paths"):
        BUILDER.write_manifest(staged_cpp_repo)
