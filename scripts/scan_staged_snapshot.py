#!/usr/bin/env python
"""Fail-closed scanner for every blob in Git's staged snapshot.

The scanner reads raw bytes, so compiled Python and other binary artifacts are
not skipped.  Forbidden expressions stay in the local, gitignored terms file.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

CACHE_PARTS = frozenset({"__pycache__", ".pytest_cache"})
BYTECODE_SUFFIXES = frozenset({".pyc", ".pyo"})
LOCAL_ONLY_ROOT_DIRS = frozenset(
    {
        ".agents",
        ".claude",
        ".codex",
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
    {".ckpt", ".key", ".log", ".npz", ".p12", ".pem", ".pfx", ".pt"}
)


class ScanError(RuntimeError):
    """The staged snapshot could not be inspected completely."""


def _git(root: Path, *args: str) -> bytes:
    completed = subprocess.run(
        ["git", *args], cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ScanError(f"git {' '.join(args)} failed: {detail}")
    return completed.stdout


def repository_root(start: Path) -> Path:
    raw = _git(start, "rev-parse", "--show-toplevel")
    return Path(os.fsdecode(raw.strip())).resolve()


def staged_paths(root: Path) -> tuple[str, ...]:
    raw = _git(
        root,
        "diff",
        "--cached",
        "--name-only",
        "-z",
        "--diff-filter=ACMR",
    )
    return tuple(os.fsdecode(item) for item in raw.split(b"\0") if item)


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
    if not path.is_file():
        raise ScanError(f"terms file missing: {path}")
    terms = [
        line.strip()
        for line in path.read_bytes().splitlines()
        if line.strip() and not line.lstrip().startswith(b"#")
    ]
    if not terms:
        raise ScanError("term list is empty")
    compiled = []
    for term in terms:
        try:
            compiled.append(re.compile(term, re.IGNORECASE))
        except re.error:
            # Local term files often contain raw host paths. A trailing slash or
            # bracket can be invalid regex syntax while still being a valid
            # literal leak marker. Fail closed by scanning it literally rather
            # than disabling the whole gate.
            compiled.append(re.compile(re.escape(term), re.IGNORECASE))
    matcher = TermMatcher(tuple(compiled))
    literal_probe = next(
        (term for term in terms if re.fullmatch(rb"[A-Za-z0-9 _.-]+", term)), None
    )
    if literal_probe is not None:
        lower = matcher.search(literal_probe.lower())
        upper = matcher.search(literal_probe.upper())
    else:
        engine_probe = re.compile(rb"release_case_probe", re.IGNORECASE)
        lower = engine_probe.search(b"release_case_probe")
        upper = engine_probe.search(b"RELEASE_CASE_PROBE")
    if lower is None or upper is None:
        raise ScanError("scanner self-test failed (case-insensitive match is broken)")
    return matcher
def is_cache_artifact(path: str) -> bool:
    normalized = path.replace("\\", "/")
    parts = {part.lower() for part in normalized.split("/")}
    return bool(parts & CACHE_PARTS) or Path(normalized).suffix.lower() in BYTECODE_SUFFIXES


def is_local_only_artifact(path: str) -> bool:
    """Reject private/local paths even when they were force-added to Git."""

    normalized = path.replace("\\", "/")
    parts = tuple(part.lower() for part in normalized.split("/") if part)
    if not parts:
        return False
    basename = parts[-1]
    suffix = Path(basename).suffix.lower()
    return (
        parts[0] in LOCAL_ONLY_ROOT_DIRS
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


def scan_staged(root: Path, terms_path: Path) -> list[str]:
    pattern = compile_terms(terms_path)
    paths = staged_paths(root)
    if not paths:
        raise ScanError("nothing staged")

    violations: list[str] = []
    for path in paths:
        if is_local_only_artifact(path):
            violations.append(f"local-only artifact: {path}")
            continue
        if is_cache_artifact(path):
            violations.append(f"cache artifact: {path}")
            continue
        blob = _git(root, "show", f":{path}")
        match = pattern.search(blob)
        if match is not None:
            line = blob.count(b"\n", 0, match.start()) + 1
            violations.append(f"forbidden bytes: {path}:{line}")
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terms", type=Path, default=Path(".leakpatterns"))
    args = parser.parse_args()
    try:
        root = repository_root(Path.cwd())
        terms_path = args.terms if args.terms.is_absolute() else root / args.terms
        violations = scan_staged(root, terms_path)
    except (OSError, ScanError) as exc:
        print(f">>> STAGED SCAN ERROR: {exc}", file=sys.stderr)
        return 2

    if violations:
        for violation in violations:
            print(violation, file=sys.stderr)
        print(">>> STAGED SNAPSHOT REJECTED. NOTHING COMMITTED.", file=sys.stderr)
        return 1

    print("staged byte/text scan: CLEAN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
