#!/usr/bin/env python
"""Verify a Git commit or extracted tree under the current release policy.

The historical ``scripts/verify_public_snapshot.py`` is frozen in the TEAR-CG/S
implementation chain. This versioned successor carries the current policy
without altering that historical artifact.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import fnmatch
import os
from pathlib import Path
import re
import subprocess
import sys

CACHE_PARTS = frozenset({"__pycache__", ".pytest_cache"})
BYTECODE_SUFFIXES = frozenset({".pyc", ".pyo"})
REGULAR_BLOB_MODES = frozenset({"100644", "100755"})
LOCAL_ONLY_DIR_NAMES = frozenset(
    {
        ".agents",
        ".claude",
        ".codex",
        ".git",
        ".idea",
        ".venv",
        ".vscode",
        "artifacts",
        "checkpoints",
        "local",
        "notes",
        "outputs",
        "private",
        "runs",
        "wandb",
    }
)
LOCAL_ONLY_FILE_NAMES = frozenset(
    {".leakpatterns", ".netrc", ".npmrc", ".pypirc", "claude.md"}
)
LOCAL_ONLY_SUFFIXES = frozenset(
    {
        ".7z",
        ".ckpt",
        ".gz",
        ".key",
        ".log",
        ".npz",
        ".p12",
        ".pem",
        ".pfx",
        ".pt",
        ".tar",
        ".tgz",
        ".zip",
    }
)


class ScanError(RuntimeError):
    """The requested release snapshot could not be inspected completely."""


@dataclass(frozen=True, slots=True)
class GitTreeEntry:
    mode: str
    object_type: str
    object_id: str
    path: str


@dataclass(frozen=True, slots=True)
class TermMatcher:
    """Independent expressions cannot escape into or corrupt their neighbours."""

    patterns: tuple[re.Pattern[bytes], ...]

    def search(self, blob: bytes):
        for pattern in self.patterns:
            match = pattern.search(blob)
            if match is not None:
                return match
        return None


def compile_terms(path: Path) -> TermMatcher:
    """Compile a caller-owned term list without publishing its path or contents."""

    if not path.is_file():
        raise ScanError("terms file missing")
    terms = [
        line.strip()
        for line in path.read_bytes().splitlines()
        if line.strip() and not line.lstrip().startswith(b"#")
    ]
    if not terms:
        raise ScanError("term list is empty")
    compiled: list[re.Pattern[bytes]] = []
    for term in terms:
        try:
            compiled.append(re.compile(term, re.IGNORECASE))
        except re.error:
            compiled.append(re.compile(re.escape(term), re.IGNORECASE))
    matcher = TermMatcher(tuple(compiled))
    probe = next(
        (term for term in terms if re.fullmatch(rb"[A-Za-z0-9 _.-]+", term)),
        None,
    )
    if probe is not None:
        lower = matcher.search(probe.lower())
        upper = matcher.search(probe.upper())
    else:
        engine_probe = re.compile(rb"release_case_probe", re.IGNORECASE)
        lower = engine_probe.search(b"release_case_probe")
        upper = engine_probe.search(b"RELEASE_CASE_PROBE")
    if lower is None or upper is None:
        raise ScanError("scanner self-test failed (case-insensitive match is broken)")
    return matcher


def _git(root: Path, *args: str) -> bytes:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ScanError(f"git {' '.join(args)} failed: {detail}")
    return completed.stdout


def repository_root(start: Path) -> Path:
    raw = _git(start, "rev-parse", "--show-toplevel")
    return Path(os.fsdecode(raw.strip())).resolve()


def commit_entries(root: Path, commit: str) -> tuple[GitTreeEntry, ...]:
    raw = _git(root, "ls-tree", "-r", "-z", "--full-tree", commit)
    entries: list[GitTreeEntry] = []
    for record in (item for item in raw.split(b"\0") if item):
        try:
            metadata_bytes, path_bytes = record.split(b"\t", 1)
            mode_bytes, type_bytes, object_id_bytes = metadata_bytes.split(b" ", 2)
            mode = mode_bytes.decode("ascii")
            object_type = type_bytes.decode("ascii")
            object_id = object_id_bytes.decode("ascii")
        except (UnicodeDecodeError, ValueError) as exc:
            raise ScanError("git tree contains an unparseable entry") from exc
        entries.append(
            GitTreeEntry(
                mode=mode,
                object_type=object_type,
                object_id=object_id,
                path=os.fsdecode(path_bytes),
            )
        )
    return tuple(entries)


def _git_entry_violation(entry: GitTreeEntry) -> str | None:
    if entry.mode == "120000":
        return f"symbolic link: {entry.path}"
    if entry.mode == "160000" or entry.object_type == "commit":
        return f"gitlink: {entry.path}"
    if entry.object_type != "blob" or entry.mode not in REGULAR_BLOB_MODES:
        return (
            f"unsupported git entry: {entry.path} "
            f"(mode {entry.mode}, type {entry.object_type})"
        )
    return None


def is_cache_artifact(path: str) -> bool:
    normalized = path.replace("\\", "/")
    parts = {part.lower() for part in normalized.split("/") if part}
    return bool(parts & CACHE_PARTS) or Path(normalized).suffix.lower() in BYTECODE_SUFFIXES


def is_local_only_directory(path: str) -> bool:
    normalized = path.replace("\\", "/")
    parts = tuple(part.lower() for part in normalized.split("/") if part)
    return any(part in LOCAL_ONLY_DIR_NAMES for part in parts)


def is_local_only_artifact(path: str) -> bool:
    """Reject private/local paths even when they were force-added to Git."""

    normalized = path.replace("\\", "/")
    parts = tuple(part.lower() for part in normalized.split("/") if part)
    if not parts:
        return False
    basename = parts[-1]
    suffix = Path(basename).suffix.lower()
    return (
        is_local_only_directory(normalized)
        or basename in LOCAL_ONLY_FILE_NAMES
        or suffix in LOCAL_ONLY_SUFFIXES
        or basename == ".env"
        or basename.startswith(".env.")
        or basename.startswith("id_rsa")
        or basename.startswith("id_ed25519")
        or fnmatch.fnmatch(basename, "goal-*.md")
        or fnmatch.fnmatch(basename, "tearrl-0-*.md")
        or ".private." in basename
        or ".local." in basename
        or fnmatch.fnmatch(basename, "*.code-workspace")
    )


def _content_violation(
    path: str,
    blob: bytes,
    matcher: TermMatcher | None,
) -> str | None:
    if matcher is None:
        return None
    match = matcher.search(blob)
    if match is None:
        return None
    line = blob.count(b"\n", 0, match.start()) + 1
    return f"forbidden bytes: {path}:{line}"


def scan_commit(
    root: Path,
    commit: str,
    terms_path: Path | None = None,
) -> list[str]:
    """Inspect public paths always and caller-supplied content terms optionally."""

    matcher = compile_terms(terms_path) if terms_path is not None else None
    entries = commit_entries(root, commit)
    if not entries:
        raise ScanError("commit snapshot is empty")
    violations: list[str] = []
    for entry in entries:
        entry_violation = _git_entry_violation(entry)
        if entry_violation is not None:
            violations.append(entry_violation)
            continue
        path = entry.path
        if is_local_only_artifact(path):
            violations.append(f"local-only artifact: {path}")
            continue
        if is_cache_artifact(path):
            violations.append(f"cache artifact: {path}")
            continue
        if matcher is None:
            continue
        violation = _content_violation(
            path,
            _git(root, "cat-file", "blob", entry.object_id),
            matcher,
        )
        if violation is not None:
            violations.append(violation)
    return violations


def _is_junction(path: Path) -> bool:
    checker = getattr(path, "is_junction", None)
    return bool(checker is not None and checker())


def scan_tree(
    root: Path,
    terms_path: Path | None = None,
) -> list[str]:
    """Inspect an extracted source tree that deliberately has no Git metadata."""

    if root.is_symlink():
        raise ScanError("tree root must not be a symbolic link")
    if _is_junction(root):
        raise ScanError("tree root must not be a junction")
    if not root.is_dir():
        raise ScanError("tree directory is missing or is not a directory")
    if (root / ".git").exists():
        raise ScanError(
            "tree mode requires an extracted tree without Git metadata; "
            "use commit mode for a clone"
        )
    matcher = compile_terms(terms_path) if terms_path is not None else None
    violations: list[str] = []
    saw_entry = False

    def raise_walk_error(error: OSError) -> None:
        raise error

    for directory, dirnames, filenames in os.walk(
        root, topdown=True, onerror=raise_walk_error, followlinks=False
    ):
        directory_path = Path(directory)
        retained: list[str] = []
        for dirname in sorted(dirnames):
            candidate = directory_path / dirname
            relative = candidate.relative_to(root).as_posix()
            if candidate.is_symlink():
                saw_entry = True
                violations.append(f"symbolic link: {relative}")
            elif _is_junction(candidate):
                saw_entry = True
                violations.append(f"junction: {relative}")
            elif is_local_only_directory(relative):
                saw_entry = True
                violations.append(f"local-only artifact: {relative}/")
            elif is_cache_artifact(f"{relative}/__tree_probe__.txt"):
                saw_entry = True
                violations.append(f"cache artifact: {relative}/")
            else:
                retained.append(dirname)
        dirnames[:] = retained

        for filename in sorted(filenames):
            candidate = directory_path / filename
            relative = candidate.relative_to(root).as_posix()
            saw_entry = True
            if candidate.is_symlink():
                violations.append(f"symbolic link: {relative}")
                continue
            if is_local_only_artifact(relative):
                violations.append(f"local-only artifact: {relative}")
                continue
            if is_cache_artifact(relative):
                violations.append(f"cache artifact: {relative}")
                continue
            if not candidate.is_file():
                violations.append(f"unsupported file type: {relative}")
                continue
            if matcher is not None:
                violation = _content_violation(
                    relative,
                    candidate.read_bytes(),
                    matcher,
                )
                if violation is not None:
                    violations.append(violation)

    if not saw_entry:
        raise ScanError("tree snapshot is empty")
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--commit",
        help="Git commit to inspect (default: HEAD when --tree is omitted)",
    )
    source.add_argument(
        "--tree",
        type=Path,
        help="extracted source tree to inspect without requiring Git metadata",
    )
    parser.add_argument(
        "--terms",
        type=Path,
        help=(
            "optional caller-supplied forbidden-term list; omit for the "
            "publicly reproducible structural scan"
        ),
    )
    args = parser.parse_args()
    try:
        if args.tree is None:
            root = repository_root(Path.cwd())
            commit = args.commit or "HEAD"
            terms = (
                None
                if args.terms is None
                else args.terms
                if args.terms.is_absolute()
                else root / args.terms
            )
            violations = scan_commit(root, commit, terms)
            source_label = f"public commit snapshot {commit}"
        else:
            cwd = Path.cwd()
            tree = args.tree if args.tree.is_absolute() else cwd / args.tree
            terms = (
                None
                if args.terms is None
                else args.terms
                if args.terms.is_absolute()
                else cwd / args.terms
            )
            tree = Path(os.path.abspath(os.fspath(tree)))
            violations = scan_tree(tree, terms)
            source_label = "extracted public tree"
    except (OSError, ScanError) as exc:
        print(f">>> RELEASE SNAPSHOT ERROR: {exc}", file=sys.stderr)
        return 2
    if violations:
        for violation in violations:
            print(violation, file=sys.stderr)
        print(">>> RELEASE SNAPSHOT REJECTED.", file=sys.stderr)
        return 1
    scope = (
        "CLEAN (structural and supplied-term scan)"
        if terms is not None
        else "STRUCTURALLY CLEAN (no content-term list supplied)"
    )
    print(f"{source_label}: {scope}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())