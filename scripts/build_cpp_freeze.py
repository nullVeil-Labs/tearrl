#!/usr/bin/env python
"""Build or verify TEAR-CPP's implementation freeze from Git stage-0 blobs.

This tool deliberately reads the Git index rather than the working tree.  It
never stages files.  ``--write`` requires the exact pre-manifest addition set
and creates the manifest exclusively.  After the caller stages that new file,
``--verify-staged`` requires the exact post-manifest addition set and verifies
the canonical manifest bytes against every indexed implementation blob.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PARENT_COMMIT = "2f507dcee27b22952d9c35d9fa2655b642753a04"
EXPECTED_ANCESTOR_BASE_COMMIT = "0d241f02e63a35009aee77b1754f394570a45d13"
EXPECTED_ANCESTOR_FILE_COUNT = 179
EXPECTED_PREREGISTRATION_SHA256 = (
    "9a2cbf6d26cf847762eeb489ab83784c2e640a70ec98b6f4090a48b0d3e6364d"
)

MANIFEST_PATH = "tearrl/cpp/frozen_implementation.json"
PREREGISTRATION_PATH = "docs/cpp_prereg.md"
PREREGISTRATION_MANIFEST_PATH = "tearrl/cpp/frozen_prereg.json"
ANCESTOR_MANIFEST_PATH = "tearrl/cpp/frozen_ancestors.json"

# This is intentionally an explicit allowlist.  Adding another implementation
# or test file requires a reviewed edit here; a glob would silently bless an
# unrelated staged file.
FROZEN_IMPLEMENTATION_PATHS = (
    "docs/cpp_implementation.md",
    PREREGISTRATION_PATH,
    "scripts/build_cpp_freeze.py",
    "scripts/run_cpp.py",
    "tearrl/cpp/__init__.py",
    "tearrl/cpp/calibration.py",
    "tearrl/cpp/config.py",
    "tearrl/cpp/experiment.py",
    "tearrl/cpp/freeze.py",
    ANCESTOR_MANIFEST_PATH,
    PREREGISTRATION_MANIFEST_PATH,
    "tearrl/cpp/gate.py",
    "tearrl/cpp/metrics.py",
    "tearrl/cpp/policy.py",
    "tearrl/cpp/reporting.py",
    "tearrl/cpp/schema.py",
    "tearrl/cpp/severity.py",
    "tearrl/cpp/splits.py",
    "tearrl/cpp/world.py",
    "tests/test_cpp_calibration.py",
    "tests/test_cpp_experiment.py",
    "tests/test_cpp_freeze.py",
    "tests/test_cpp_freeze_generator.py",
    "tests/test_cpp_freeze_verifier.py",
    "tests/test_cpp_gate.py",
    "tests/test_cpp_launcher.py",
    "tests/test_cpp_lineage_isolation.py",
    "tests/test_cpp_metrics.py",
    "tests/test_cpp_policy.py",
    "tests/test_cpp_reporting.py",
    "tests/test_cpp_splits.py",
    "tests/test_cpp_world.py",
)

EXISTING_BINDING_PATHS = frozenset(
    (
        PREREGISTRATION_PATH,
        PREREGISTRATION_MANIFEST_PATH,
        ANCESTOR_MANIFEST_PATH,
    )
)
EXPECTED_ADDITION_PATHS = frozenset(FROZEN_IMPLEMENTATION_PATHS) - (
    EXISTING_BINDING_PATHS
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ALLOWED_INDEX_MODES = frozenset(("100644", "100755"))


class FreezeBuildError(RuntimeError):
    """Raised when the staged CPP implementation cannot be frozen safely."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git(root: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise FreezeBuildError(
            f"git {' '.join(arguments)} failed"
            + (f": {detail}" if detail else "")
        )
    return completed.stdout


def _git_path_exists(root: Path, revision: str, relative: str) -> bool:
    completed = subprocess.run(
        ["git", "cat-file", "-e", f"{revision}:{relative}"],
        cwd=root,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return completed.returncode == 0


def _decode_path(payload: bytes, *, label: str) -> str:
    try:
        value = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FreezeBuildError(f"{label} is not UTF-8") from exc
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise FreezeBuildError(f"unsafe {label}: {value!r}")
    return pure.as_posix()


def _index_entries(root: Path) -> dict[str, list[tuple[str, str, int]]]:
    entries: dict[str, list[tuple[str, str, int]]] = {}
    for record in _git(root, "ls-files", "--stage", "-z").split(b"\0"):
        if not record:
            continue
        metadata, separator, raw_path = record.partition(b"\t")
        if not separator:
            raise FreezeBuildError("malformed git index entry")
        fields = metadata.split(b" ")
        if len(fields) != 3:
            raise FreezeBuildError("malformed git index metadata")
        try:
            mode = fields[0].decode("ascii")
            object_id = fields[1].decode("ascii")
            stage = int(fields[2].decode("ascii"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise FreezeBuildError("malformed git index metadata") from exc
        path = _decode_path(raw_path, label="Git index path")
        entries.setdefault(path, []).append((mode, object_id, stage))
    return entries


def _require_stage_zero(
    entries: Mapping[str, Sequence[tuple[str, str, int]]],
    relative: str,
) -> tuple[str, str, int]:
    matches = list(entries.get(relative, ()))
    if len(matches) != 1 or matches[0][2] != 0:
        raise FreezeBuildError(
            f"expected exactly one stage-0 Git index entry: {relative}"
        )
    mode, object_id, stage = matches[0]
    if mode not in _ALLOWED_INDEX_MODES:
        raise FreezeBuildError(
            f"unsupported Git index mode {mode!r} for frozen file: {relative}"
        )
    return mode, object_id, stage


def _index_blob(
    root: Path,
    entries: Mapping[str, Sequence[tuple[str, str, int]]],
    relative: str,
) -> bytes:
    _require_stage_zero(entries, relative)
    # The :path revision is Git's exact stage-0 blob, including normalized EOLs.
    return _git(root, "show", f":{relative}")


def _head_blob(root: Path, relative: str) -> bytes:
    return _git(root, "show", f"HEAD:{relative}")


def _changed_index_paths(root: Path) -> frozenset[str]:
    raw = _git(
        root,
        "diff",
        "--cached",
        "--name-only",
        "--no-renames",
        "-z",
        "HEAD",
        "--",
    )
    return frozenset(
        _decode_path(item, label="staged path")
        for item in raw.split(b"\0")
        if item
    )


def _require_exact_changed_paths(
    root: Path,
    expected: frozenset[str],
) -> None:
    actual = _changed_index_paths(root)
    if actual == expected:
        return
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    details: list[str] = []
    if missing:
        details.append(f"missing staged paths: {', '.join(missing)}")
    if extra:
        details.append(f"unrelated staged paths: {', '.join(extra)}")
    raise FreezeBuildError("staged inventory mismatch; " + "; ".join(details))


def _json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise FreezeBuildError(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FreezeBuildError(f"invalid JSON in {label}") from exc
    if not isinstance(value, dict):
        raise FreezeBuildError(f"{label} must contain a JSON object")
    return value


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise FreezeBuildError(f"{label} must be a lowercase SHA-256")
    return value


def _require_repository(root: Path) -> None:
    resolved = root.resolve()
    try:
        discovered = Path(
            _git(resolved, "rev-parse", "--show-toplevel")
            .decode("utf-8")
            .strip()
        ).resolve()
    except UnicodeDecodeError as exc:
        raise FreezeBuildError("Git repository root is not UTF-8") from exc
    if discovered != resolved:
        raise FreezeBuildError("freeze root is not the Git repository root")
    head = _git(resolved, "rev-parse", "--verify", "HEAD").decode("ascii").strip()
    if head != EXPECTED_PARENT_COMMIT:
        raise FreezeBuildError(
            "CPP implementation freeze requires HEAD "
            f"{EXPECTED_PARENT_COMMIT}, got {head}"
        )


def _require_add_only_parent(root: Path) -> None:
    for relative in sorted(EXISTING_BINDING_PATHS):
        if not _git_path_exists(root, "HEAD", relative):
            raise FreezeBuildError(f"missing frozen parent binding: {relative}")
    for relative in sorted(EXPECTED_ADDITION_PATHS | {MANIFEST_PATH}):
        if _git_path_exists(root, "HEAD", relative):
            raise FreezeBuildError(
                f"CPP implementation path already exists in frozen parent: {relative}"
            )


def _require_unchanged_bindings(
    root: Path,
    entries: Mapping[str, Sequence[tuple[str, str, int]]],
) -> None:
    for relative in sorted(EXISTING_BINDING_PATHS):
        indexed = _index_blob(root, entries, relative)
        if indexed != _head_blob(root, relative):
            raise FreezeBuildError(
                f"frozen parent binding differs between HEAD and index: {relative}"
            )


def _validate_binding_manifests(
    root: Path,
    entries: Mapping[str, Sequence[tuple[str, str, int]]],
) -> tuple[str, str]:
    preregistration_bytes = _index_blob(root, entries, PREREGISTRATION_PATH)
    preregistration_manifest_bytes = _index_blob(
        root, entries, PREREGISTRATION_MANIFEST_PATH
    )
    ancestor_manifest_bytes = _index_blob(
        root, entries, ANCESTOR_MANIFEST_PATH
    )
    preregistration = _json_object(
        preregistration_manifest_bytes,
        label=PREREGISTRATION_MANIFEST_PATH,
    )
    ancestors = _json_object(
        ancestor_manifest_bytes,
        label=ANCESTOR_MANIFEST_PATH,
    )

    expected_preregistration = _require_sha256(
        preregistration.get("sha256"),
        label="frozen preregistration digest",
    )
    if expected_preregistration != EXPECTED_PREREGISTRATION_SHA256:
        raise FreezeBuildError("unexpected frozen preregistration digest")
    if _sha256(preregistration_bytes) != expected_preregistration:
        raise FreezeBuildError(
            "staged preregistration document does not match its frozen digest"
        )
    if preregistration.get("document") != PREREGISTRATION_PATH:
        raise FreezeBuildError("unexpected preregistration document binding")
    if (
        preregistration.get("status")
        != "frozen_before_implementation_and_outcomes"
    ):
        raise FreezeBuildError("unexpected preregistration freeze status")
    if preregistration.get("hash_domain") != "committed_git_blob_bytes":
        raise FreezeBuildError("unexpected preregistration hash domain")
    if preregistration.get("parent_commit") != EXPECTED_ANCESTOR_BASE_COMMIT:
        raise FreezeBuildError("unexpected preregistration parent commit")

    if ancestors.get("algorithm") != "sha256":
        raise FreezeBuildError("unexpected ancestor digest algorithm")
    if ancestors.get("hash_domain") != "committed_git_blob_bytes":
        raise FreezeBuildError("unexpected ancestor hash domain")
    if ancestors.get("base_commit") != EXPECTED_ANCESTOR_BASE_COMMIT:
        raise FreezeBuildError("unexpected ancestor base commit")
    files = ancestors.get("files")
    if not isinstance(files, dict) or len(files) != EXPECTED_ANCESTOR_FILE_COUNT:
        raise FreezeBuildError("unexpected ancestor file inventory")
    if ancestors.get("tracked_file_count") != EXPECTED_ANCESTOR_FILE_COUNT:
        raise FreezeBuildError("ancestor tracked-file count is inconsistent")
    for relative, digest in files.items():
        if not isinstance(relative, str):
            raise FreezeBuildError("ancestor file path must be a string")
        _decode_path(relative.encode("utf-8"), label="ancestor path")
        _require_sha256(digest, label=f"ancestor digest for {relative}")

    return (
        _sha256(preregistration_manifest_bytes),
        _sha256(ancestor_manifest_bytes),
    )


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _build_expected_manifest(
    root: Path,
    entries: Mapping[str, Sequence[tuple[str, str, int]]],
) -> dict[str, Any]:
    preregistration_manifest_sha256, ancestor_manifest_sha256 = (
        _validate_binding_manifests(root, entries)
    )
    files = {
        relative: _sha256(_index_blob(root, entries, relative))
        for relative in sorted(FROZEN_IMPLEMENTATION_PATHS)
    }
    if MANIFEST_PATH in files:
        raise FreezeBuildError("implementation manifest cannot hash itself")
    return {
        "algorithm": "sha256",
        "ancestor_manifest_sha256": ancestor_manifest_sha256,
        "files": files,
        "hash_domain": "committed_git_blob_bytes",
        "implementation_file_count": len(files),
        "manifest_construction_source": "git_index_stage_0_blob_bytes",
        "parent_commit": EXPECTED_PARENT_COMMIT,
        "preregistration_commit": EXPECTED_PARENT_COMMIT,
        "preregistration_manifest_sha256": preregistration_manifest_sha256,
        "preregistration_sha256": EXPECTED_PREREGISTRATION_SHA256,
        "status": "frozen_before_preflight_and_outcomes",
    }


def _prepare(
    root: Path,
    *,
    require_manifest_staged: bool,
) -> tuple[dict[str, list[tuple[str, str, int]]], bytes]:
    resolved = root.resolve()
    _require_repository(resolved)
    _require_add_only_parent(resolved)
    expected_changes = EXPECTED_ADDITION_PATHS | (
        frozenset((MANIFEST_PATH,)) if require_manifest_staged else frozenset()
    )
    _require_exact_changed_paths(resolved, expected_changes)
    entries = _index_entries(resolved)
    for relative in sorted(FROZEN_IMPLEMENTATION_PATHS):
        _require_stage_zero(entries, relative)
    if require_manifest_staged:
        _require_stage_zero(entries, MANIFEST_PATH)
    elif MANIFEST_PATH in entries:
        raise FreezeBuildError(
            "implementation manifest is already staged before --write"
        )
    _require_unchanged_bindings(resolved, entries)
    return entries, _git(resolved, "ls-files", "--stage", "-z")


def write_manifest(root: Path = ROOT) -> dict[str, Any]:
    """Create the unstaged implementation manifest without changing the index."""

    resolved = root.resolve()
    manifest_path = resolved.joinpath(*PurePosixPath(MANIFEST_PATH).parts)
    if os.path.lexists(manifest_path):
        raise FreezeBuildError(
            f"refusing to overwrite existing manifest: {MANIFEST_PATH}"
        )
    entries, index_before = _prepare(
        resolved,
        require_manifest_staged=False,
    )
    manifest = _build_expected_manifest(resolved, entries)
    payload = _canonical_json_bytes(manifest)
    try:
        descriptor = os.open(
            manifest_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
            0o644,
        )
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise FreezeBuildError(
            f"refusing to overwrite existing manifest: {MANIFEST_PATH}"
        ) from exc
    if _git(resolved, "ls-files", "--stage", "-z") != index_before:
        raise FreezeBuildError("Git index changed while writing the freeze manifest")
    return manifest


def verify_staged_manifest(root: Path = ROOT) -> dict[str, Any]:
    """Verify the canonical staged manifest and its exact stage-0 blob map."""

    resolved = root.resolve()
    entries, index_before = _prepare(
        resolved,
        require_manifest_staged=True,
    )
    expected = _build_expected_manifest(resolved, entries)
    staged_manifest = _index_blob(resolved, entries, MANIFEST_PATH)
    expected_bytes = _canonical_json_bytes(expected)
    if staged_manifest != expected_bytes:
        raise FreezeBuildError(
            "staged implementation manifest is not the canonical index-derived manifest"
        )
    if _git(resolved, "ls-files", "--stage", "-z") != index_before:
        raise FreezeBuildError("Git index changed while verifying the freeze manifest")
    return expected


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--write",
        action="store_true",
        help="exclusively create the unstaged manifest from exact staged blobs",
    )
    mode.add_argument(
        "--verify-staged",
        action="store_true",
        help="verify the staged manifest and exact staged implementation inventory",
    )
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    if arguments.write:
        manifest = write_manifest(ROOT)
        print(
            f"created {MANIFEST_PATH} from "
            f"{manifest['implementation_file_count']} stage-0 blobs"
        )
        print("manifest remains unstaged; review and stage it explicitly")
        return 0
    manifest = verify_staged_manifest(ROOT)
    print(
        "staged TEAR-CPP implementation freeze: PASS "
        f"({manifest['implementation_file_count']} files)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
