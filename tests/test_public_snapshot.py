from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERIFY = ROOT / "scripts" / "verify_public_snapshot.py"


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    _git(repo, "config", "user.name", "Public Snapshot Test")
    _git(repo, "config", "user.email", "snapshot@example.invalid")
    (repo / ".leakpatterns").write_text("PRIVATE_SENTINEL\n", encoding="utf-8")
    return repo


def _commit(repo: Path, *paths: str) -> None:
    _git(repo, "add", "-f", "--", *paths)
    _git(repo, "commit", "--quiet", "-m", "snapshot")


def _verify(repo: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(VERIFY), "--commit", "HEAD"],
        cwd=repo,
        text=True,
        capture_output=True,
    )


def test_clean_committed_snapshot_passes(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "result.json").write_text('{"result":"negative"}\n', encoding="utf-8")
    _commit(repo, "result.json")
    result = _verify(repo)
    assert result.returncode == 0, result.stderr
    assert "CLEAN" in result.stdout


def test_force_committed_local_path_and_archive_are_rejected(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "GOAL-session.md").write_text("local\n", encoding="utf-8")
    (repo / "results.zip").write_bytes(b"opaque")
    _commit(repo, "GOAL-session.md", "results.zip")
    result = _verify(repo)
    assert result.returncode == 1
    assert "local-only artifact: GOAL-session.md" in result.stderr
    assert "local-only artifact: results.zip" in result.stderr


def test_forbidden_bytes_in_committed_binary_are_rejected(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "trace.bin").write_bytes(b"prefix\x00PRIVATE_SENTINEL\x00suffix")
    _commit(repo, "trace.bin")
    result = _verify(repo)
    assert result.returncode == 1
    assert "forbidden bytes: trace.bin:1" in result.stderr