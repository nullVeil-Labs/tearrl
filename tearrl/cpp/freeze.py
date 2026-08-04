"""Fail-closed verification for the TEAR-CPP freeze chain."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any


class FreezeVerificationError(RuntimeError):
    """Raised when a frozen document, source, or lineage byte has changed."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def matches_committed_blob_sha256(path: Path, expected: str) -> bool:
    """Match a Git blob digest without treating checkout EOLs as mutations.

    The manifest hashes committed blob bytes. A historical Windows checkout
    can retain CRLF even when Git's current attributes specify LF. Raw bytes
    are always tried first; only a raw mismatch gets one text-only CRLF-to-LF
    candidate. Any content change still fails against the registered digest.
    """

    payload = path.read_bytes()
    if sha256_bytes(payload) == expected:
        return True
    if b"\0" in payload:
        return False
    return sha256_bytes(payload.replace(b"\r\n", b"\n")) == expected


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FreezeVerificationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FreezeVerificationError(f"cannot read freeze JSON: {path.name}") from exc
    if not isinstance(value, dict):
        raise FreezeVerificationError(f"freeze JSON is not an object: {path.name}")
    return value


def _resolve_frozen_path(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise FreezeVerificationError(f"unsafe frozen path: {relative!r}")
    candidate = root.joinpath(*pure.parts).resolve()
    resolved_root = root.resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise FreezeVerificationError(f"frozen path escapes root: {relative!r}") from exc
    return candidate


def _verify_file_map(
    root: Path,
    entries: dict[str, Any],
    *,
    label: str,
) -> int:
    if not entries:
        raise FreezeVerificationError(f"{label} file map is empty")
    for relative, expected in entries.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise FreezeVerificationError(f"invalid {label} manifest entry")
        path = _resolve_frozen_path(root, relative)
        if not path.is_file():
            raise FreezeVerificationError(f"missing {label} file: {relative}")
        actual = sha256_file(path)
        if not matches_committed_blob_sha256(path, expected):
            raise FreezeVerificationError(
                f"{label} digest mismatch: {relative}; expected {expected}, got {actual}"
            )
    return len(entries)


def verify_preregistration(root: Path) -> dict[str, Any]:
    manifest_path = root / "tearrl" / "cpp" / "frozen_prereg.json"
    manifest = _load_json(manifest_path)
    if manifest.get("status") != "frozen_before_implementation_and_outcomes":
        raise FreezeVerificationError("unexpected preregistration freeze status")
    if manifest.get("hash_domain") != "committed_git_blob_bytes":
        raise FreezeVerificationError("unexpected preregistration hash domain")
    document = manifest.get("document")
    expected = manifest.get("sha256")
    if not isinstance(document, str) or not isinstance(expected, str):
        raise FreezeVerificationError("invalid preregistration manifest")
    document_path = _resolve_frozen_path(root, document)
    actual = sha256_file(document_path)
    if not matches_committed_blob_sha256(document_path, expected):
        raise FreezeVerificationError(
            f"preregistration digest mismatch; expected {expected}, got {actual}"
        )
    return manifest


def verify_ancestors(root: Path) -> dict[str, Any]:
    manifest_path = root / "tearrl" / "cpp" / "frozen_ancestors.json"
    manifest = _load_json(manifest_path)
    if manifest.get("algorithm") != "sha256":
        raise FreezeVerificationError("unexpected ancestor digest algorithm")
    if manifest.get("hash_domain") != "committed_git_blob_bytes":
        raise FreezeVerificationError("unexpected ancestor hash domain")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise FreezeVerificationError("invalid ancestor file map")
    _verify_file_map(root, files, label="ancestor")
    return manifest


def verify_implementation(root: Path) -> dict[str, Any]:
    manifest_path = root / "tearrl" / "cpp" / "frozen_implementation.json"
    manifest = _load_json(manifest_path)
    if manifest.get("status") != "frozen_before_preflight_and_outcomes":
        raise FreezeVerificationError("unexpected implementation freeze status")
    if manifest.get("hash_domain") != "committed_git_blob_bytes":
        raise FreezeVerificationError("unexpected implementation hash domain")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise FreezeVerificationError("invalid implementation file map")
    _verify_file_map(root, files, label="implementation")

    prereg_path = root / "tearrl" / "cpp" / "frozen_prereg.json"
    ancestor_path = root / "tearrl" / "cpp" / "frozen_ancestors.json"
    prereg_manifest_digest = manifest.get("preregistration_manifest_sha256")
    if not isinstance(prereg_manifest_digest, str) or not (
        matches_committed_blob_sha256(prereg_path, prereg_manifest_digest)
    ):
        raise FreezeVerificationError("preregistration manifest binding mismatch")
    ancestor_manifest_digest = manifest.get("ancestor_manifest_sha256")
    if not isinstance(ancestor_manifest_digest, str) or not (
        matches_committed_blob_sha256(ancestor_path, ancestor_manifest_digest)
    ):
        raise FreezeVerificationError("ancestor manifest binding mismatch")
    return manifest


def verify_freeze_chain(root: Path) -> dict[str, Any]:
    prereg = verify_preregistration(root)
    ancestors = verify_ancestors(root)
    implementation = verify_implementation(root)
    if prereg.get("parent_commit") != ancestors.get("base_commit"):
        raise FreezeVerificationError("preregistration and ancestor bases disagree")
    if implementation.get("preregistration_sha256") != prereg.get("sha256"):
        raise FreezeVerificationError("implementation does not bind preregistration")
    return {
        "preregistration_sha256": prereg["sha256"],
        "ancestor_files": len(ancestors["files"]),
        "implementation_files": len(implementation["files"]),
    }
