"""Release scanner regressions: binary blobs and caches must never be skipped."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCANNER = ROOT / "scripts" / "scan_staged_snapshot.py"


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _scan(repo: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCANNER), "--terms", ".leakpatterns"],
        cwd=repo,
        text=True,
        capture_output=True,
    )


def _repo(tmp_path: Path, term: str = "PRIVATE_SENTINEL") -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    _git(repo, "config", "user.name", "Release Scanner Test")
    _git(repo, "config", "user.email", "scanner@example.invalid")
    (repo / ".leakpatterns").write_text(term + "\n", encoding="utf-8")
    return repo


def test_raw_binary_blob_is_scanned(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "artifact.bin").write_bytes(b"prefix\x00PRIVATE_SENTINEL\x00suffix")
    _git(repo, "add", "artifact.bin")
    result = _scan(repo)
    assert result.returncode == 1
    assert "forbidden bytes: artifact.bin:1" in result.stderr


def test_python_bytecode_cache_is_rejected_even_without_a_term_hit(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    cache = repo / "pkg" / "__pycache__"
    cache.mkdir(parents=True)
    target = cache / "module.cpython-313.pyc"
    target.write_bytes(b"otherwise clean bytes")
    _git(repo, "add", "-f", str(target.relative_to(repo)))
    result = _scan(repo)
    assert result.returncode == 1
    assert "cache artifact:" in result.stderr
    assert "__pycache__" in result.stderr


def test_host_path_inside_binary_blob_is_detected(tmp_path: Path) -> None:
    repo = _repo(tmp_path, r"C:\\Users\\")
    (repo / "trace.bin").write_bytes(b"\x00C:\\Users\\VisibleName\\project.py\x00")
    _git(repo, "add", "trace.bin")
    result = _scan(repo)
    assert result.returncode == 1
    assert "forbidden bytes: trace.bin:1" in result.stderr


def test_clean_text_snapshot_passes(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "clean.txt").write_text("public scientific result\n", encoding="utf-8")
    _git(repo, "add", "clean.txt")
    result = _scan(repo)
    assert result.returncode == 0, result.stderr
    assert "CLEAN" in result.stdout

def test_invalid_regex_term_is_scanned_as_a_literal(tmp_path: Path) -> None:
    repo = _repo(tmp_path, "unterminated[")
    (repo / "binary.bin").write_bytes(b"prefix\x00unterminated[\x00suffix")
    _git(repo, "add", "binary.bin")
    result = _scan(repo)
    assert result.returncode == 1
    assert "forbidden bytes: binary.bin:1" in result.stderr


def test_force_added_local_session_file_is_rejected(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    target = repo / "GOAL-private-session.md"
    target.write_text("local agent notes\n", encoding="utf-8")
    archive = repo / "results.zip"
    archive.write_bytes(b"opaque archive placeholder")
    _git(repo, "add", "-f", target.name, archive.name)
    result = _scan(repo)
    assert result.returncode == 1
    assert "local-only artifact: GOAL-private-session.md" in result.stderr
    assert "local-only artifact: results.zip" in result.stderr
