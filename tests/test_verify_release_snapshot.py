from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VERIFY = ROOT / "scripts" / "verify_release_snapshot.py"


def _module():
    spec = importlib.util.spec_from_file_location("verify_release_snapshot", VERIFY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _git_output(
    repo: Path,
    *args: str,
    input_bytes: bytes | None = None,
) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        input=input_bytes,
        check=True,
        capture_output=True,
    )
    return completed.stdout.decode("ascii").strip()


def _repo(tmp_path: Path, *, with_terms: bool = True) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    _git(repo, "config", "user.name", "Release Snapshot Test")
    _git(repo, "config", "user.email", "snapshot@example.invalid")
    if with_terms:
        (repo / ".leakpatterns").write_text(
            "PRIVATE_SENTINEL\n", encoding="utf-8"
        )
    return repo


def _commit(repo: Path, *paths: str) -> None:
    _git(repo, "add", "-f", "--", *paths)
    _git(repo, "commit", "--quiet", "-m", "snapshot")


def _verify_commit(
    repo: Path,
    *,
    terms: str | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(VERIFY), "--commit", "HEAD"]
    if terms is not None:
        command.extend(["--terms", terms])
    return subprocess.run(
        command,
        cwd=repo,
        text=True,
        capture_output=True,
    )


def _verify_tree(
    tree: Path,
    *,
    terms: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(VERIFY), "--tree", str(tree)]
    if terms is not None:
        command.extend(["--terms", str(terms)])
    return subprocess.run(
        command,
        cwd=tree.parent,
        text=True,
        capture_output=True,
    )


def test_clean_commit_passes_without_private_terms(tmp_path: Path) -> None:
    repo = _repo(tmp_path, with_terms=False)
    (repo / "result.json").write_text('{"result":"negative"}\n', encoding="utf-8")
    _commit(repo, "result.json")

    result = _verify_commit(repo)
    assert result.returncode == 0, result.stderr
    assert "STRUCTURALLY CLEAN" in result.stdout
    assert "no content-term list supplied" in result.stdout


def test_explicit_missing_term_list_fails_closed(tmp_path: Path) -> None:
    repo = _repo(tmp_path, with_terms=False)
    (repo / "result.json").write_text('{"result":"negative"}\n', encoding="utf-8")
    _commit(repo, "result.json")

    result = _verify_commit(repo, terms="missing-patterns.txt")
    assert result.returncode == 2
    assert "terms file missing" in result.stderr
    assert str(tmp_path) not in result.stderr


def test_commit_rejects_local_paths_archives_and_nested_caches(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    local = repo / "docs" / ".codex" / "session.md"
    local.parent.mkdir(parents=True)
    local.write_text("local\n", encoding="utf-8")
    cache = repo / "pkg" / "__pycache__" / "module.pyc"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"compiled")
    archive = repo / "results.zip"
    archive.write_bytes(b"opaque")
    _commit(
        repo,
        str(local.relative_to(repo)),
        str(cache.relative_to(repo)),
        archive.name,
    )

    result = _verify_commit(repo)
    assert result.returncode == 1
    assert "local-only artifact: docs/.codex/session.md" in result.stderr
    assert "cache artifact: pkg/__pycache__/module.pyc" in result.stderr
    assert "local-only artifact: results.zip" in result.stderr


def test_commit_accepts_caller_terms_and_scans_binary_bytes(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "trace.bin").write_bytes(b"prefix\x00PRIVATE_SENTINEL\x00suffix")
    _commit(repo, "trace.bin")

    structural = _verify_commit(repo)
    supplied = _verify_commit(repo, terms=".leakpatterns")
    assert structural.returncode == 0
    assert "no content-term list supplied" in structural.stdout
    assert supplied.returncode == 1
    assert "forbidden bytes: trace.bin:1" in supplied.stderr


def test_extracted_tree_passes_without_git_or_private_terms(tmp_path: Path) -> None:
    tree = tmp_path / "extracted"
    tree.mkdir()
    (tree / "result.json").write_text('{"result":"negative"}\n', encoding="utf-8")

    result = _verify_tree(tree)
    assert result.returncode == 0, result.stderr
    assert "extracted public tree: STRUCTURALLY CLEAN" in result.stdout
    assert "no content-term list supplied" in result.stdout


def test_extracted_tree_checks_local_cache_and_caller_terms(tmp_path: Path) -> None:
    tree = tmp_path / "extracted"
    local = tree / "docs" / ".codex"
    local.mkdir(parents=True)
    (local / "session.md").write_text("local\n", encoding="utf-8")
    cache = tree / "pkg" / ".pytest_cache"
    cache.mkdir(parents=True)
    (cache / "state").write_text("cache\n", encoding="utf-8")
    (tree / "trace.bin").write_bytes(b"prefix\x00PRIVATE_SENTINEL\x00suffix")
    terms = tmp_path / "reviewer-patterns.txt"
    terms.write_text("PRIVATE_SENTINEL\n", encoding="utf-8")

    result = _verify_tree(tree, terms=terms)
    assert result.returncode == 1
    assert "local-only artifact: docs/.codex/" in result.stderr
    assert "cache artifact: pkg/.pytest_cache/" in result.stderr
    assert "forbidden bytes: trace.bin:1" in result.stderr


def test_tree_mode_refuses_git_metadata(tmp_path: Path) -> None:
    tree = tmp_path / "extracted"
    (tree / ".git").mkdir(parents=True)
    (tree / "result.json").write_text("{}\n", encoding="utf-8")

    result = _verify_tree(tree)
    assert result.returncode == 2
    assert "requires an extracted tree without Git metadata" in result.stderr


def test_tree_scan_rejects_symbolic_links_without_following_them(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _module()
    tree = tmp_path / "extracted"
    tree.mkdir()
    monkeypatch.setattr(
        module.os,
        "walk",
        lambda *args, **kwargs: iter([(str(tree), [], ["linked.txt"])]),
    )
    monkeypatch.setattr(
        module.Path,
        "is_symlink",
        lambda self: self.name == "linked.txt",
    )

    assert module.scan_tree(tree) == ["symbolic link: linked.txt"]


def test_empty_extracted_tree_fails_closed(tmp_path: Path) -> None:
    tree = tmp_path / "extracted"
    tree.mkdir()

    result = _verify_tree(tree)
    assert result.returncode == 2
    assert "tree snapshot is empty" in result.stderr

def test_tree_rejects_reserved_directory_name_used_as_nested_file(
    tmp_path: Path,
) -> None:
    tree = tmp_path / "extracted"
    nested = tree / "docs"
    nested.mkdir(parents=True)
    (nested / ".git").write_text("not metadata\n", encoding="utf-8")

    result = _verify_tree(tree)
    assert result.returncode == 1
    assert "local-only artifact: docs/.git" in result.stderr

def test_commit_rejects_symbolic_link_and_gitlink_modes(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "base.txt").write_text("base\n", encoding="utf-8")
    _commit(repo, "base.txt")
    symbolic_link_blob = _git_output(
        repo,
        "hash-object",
        "-w",
        "--stdin",
        input_bytes=b"outside.txt",
    )
    head = _git_output(repo, "rev-parse", "HEAD")
    _git(
        repo,
        "update-index",
        "--add",
        "--cacheinfo",
        "120000",
        symbolic_link_blob,
        "linked.txt",
    )
    _git(
        repo,
        "update-index",
        "--add",
        "--cacheinfo",
        "160000",
        head,
        "nested-module",
    )
    _git(repo, "commit", "--quiet", "-m", "nonregular entries")

    result = _verify_commit(repo)
    assert result.returncode == 1
    assert "symbolic link: linked.txt" in result.stderr
    assert "gitlink: nested-module" in result.stderr


def test_tree_scan_rejects_root_junction_before_traversal(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _module()
    tree = tmp_path / "extracted"
    tree.mkdir()
    monkeypatch.setattr(
        module.Path,
        "is_junction",
        lambda self: self.name == "extracted",
        raising=False,
    )

    with pytest.raises(module.ScanError, match="tree root must not be a junction"):
        module.scan_tree(tree)


def test_tree_scan_rejects_nested_junction_without_following_it(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _module()
    tree = tmp_path / "extracted"
    tree.mkdir()
    monkeypatch.setattr(
        module.os,
        "walk",
        lambda *args, **kwargs: iter([(str(tree), ["linked-dir"], [])]),
    )
    monkeypatch.setattr(module.Path, "is_symlink", lambda self: False)
    monkeypatch.setattr(
        module.Path,
        "is_junction",
        lambda self: self.name == "linked-dir",
        raising=False,
    )

    assert module.scan_tree(tree) == ["junction: linked-dir"]
