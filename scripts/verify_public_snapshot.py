#!/usr/bin/env python
"""Fail closed when a committed public snapshot contains local/private artifacts."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from scan_staged_snapshot import (
    ScanError,
    compile_terms,
    is_cache_artifact,
    is_local_only_artifact,
    repository_root,
)


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


def commit_paths(root: Path, commit: str) -> tuple[str, ...]:
    raw = _git(root, "ls-tree", "-r", "-z", "--name-only", commit)
    return tuple(os.fsdecode(item) for item in raw.split(b"\0") if item)


def scan_commit(root: Path, commit: str, terms_path: Path) -> list[str]:
    matcher = compile_terms(terms_path)
    paths = commit_paths(root, commit)
    if not paths:
        raise ScanError("commit snapshot is empty")
    violations: list[str] = []
    for path in paths:
        if is_local_only_artifact(path):
            violations.append(f"local-only artifact: {path}")
            continue
        if is_cache_artifact(path):
            violations.append(f"cache artifact: {path}")
            continue
        blob = _git(root, "show", f"{commit}:{path}")
        match = matcher.search(blob)
        if match is not None:
            line = blob.count(b"\n", 0, match.start()) + 1
            violations.append(f"forbidden bytes: {path}:{line}")
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", default="HEAD")
    parser.add_argument("--terms", type=Path, default=Path(".leakpatterns"))
    args = parser.parse_args()
    try:
        root = repository_root(Path.cwd())
        terms = args.terms if args.terms.is_absolute() else root / args.terms
        violations = scan_commit(root, args.commit, terms)
    except (OSError, ScanError) as exc:
        print(f">>> PUBLIC SNAPSHOT ERROR: {exc}", file=sys.stderr)
        return 2
    if violations:
        for violation in violations:
            print(violation, file=sys.stderr)
        print(">>> PUBLIC SNAPSHOT REJECTED.", file=sys.stderr)
        return 1
    print(f"public commit snapshot {args.commit}: CLEAN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())