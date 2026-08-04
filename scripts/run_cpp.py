#!/usr/bin/env python
"""Run TEAR-CPP's timing-only preflight or frozen confirmatory study."""

from __future__ import annotations

import sys


_ACTUAL_ENTRY = __name__ == "__main__"
_BANNED_STARTUP_MODULES = {
    "argparse",
    "hashlib",
    "importlib",
    "json",
    "os",
    "pathlib",
    "site",
    "sitecustomize",
    "tempfile",
    "typing",
    "usercustomize",
    "tearrl",
    "numpy",
    "torch",
}
_base_prefix = sys.base_prefix.rstrip("\\/")
_TRUSTED_STDLIB_PATHS = (
    f"{_base_prefix}\\DLLs",
    f"{_base_prefix}\\Lib",
)

if _ACTUAL_ENTRY:
    required_flags = {
        "ignore_environment": sys.flags.ignore_environment,
        "isolated": sys.flags.isolated,
        "no_site": sys.flags.no_site,
        "no_user_site": sys.flags.no_user_site,
        "safe_path": sys.flags.safe_path,
        "dont_write_bytecode": sys.flags.dont_write_bytecode,
    }
    missing_flags = sorted(
        name for name, enabled in required_flags.items() if not enabled
    )
    if missing_flags:
        raise SystemExit(
            "TEAR-CPP launcher requires -I -S -P -B; missing: "
            + ", ".join(missing_flags)
        )
    loaded_banned = sorted(
        name
        for name in sys.modules
        if name.partition(".")[0] in _BANNED_STARTUP_MODULES
    )
    if loaded_banned:
        raise SystemExit(
            "TEAR-CPP launcher rejects preloaded startup modules: "
            + ", ".join(loaded_banned)
        )
    trusted_prefix = (_base_prefix + "\\").casefold()
    untrusted_preloads = []
    for name, module in sys.modules.items():
        spec = getattr(module, "__spec__", None)
        origin = getattr(spec, "origin", None)
        normalized = (
            origin.replace("/", "\\").casefold()
            if type(origin) is str
            else None
        )
        if origin in {"built-in", "frozen"}:
            continue
        if name == "__main__" and origin is None:
            continue
        if normalized is not None and normalized.startswith(trusted_prefix):
            continue
        untrusted_preloads.append(name)
    if untrusted_preloads:
        raise SystemExit(
            "TEAR-CPP launcher rejects untrusted preloaded modules: "
            + ", ".join(sorted(untrusted_preloads))
        )
    if sys.flags.optimize != 0 or sys.warnoptions or sys._xoptions:
        raise SystemExit("TEAR-CPP launcher rejects altered interpreter options")
    sys.path[:] = list(_TRUSTED_STDLIB_PATHS)
    sys.path_importer_cache.clear()


import argparse
import hashlib
import importlib.machinery
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any, Callable, Mapping


ROOT = Path(__file__).resolve().parents[1]
_TRUSTED_STDLIB_PATHS = tuple(
    str(Path(path).resolve()) for path in _TRUSTED_STDLIB_PATHS
)
_EXPECTED_INTERPRETER = (ROOT / ".venv" / "Scripts" / "python.exe").resolve()
_SITE_PACKAGES = (ROOT / ".venv" / "Lib" / "site-packages").resolve()
if Path(sys.executable).resolve() != _EXPECTED_INTERPRETER:
    raise RuntimeError(
        "TEAR-CPP requires the repository's exact audited .venv interpreter"
    )
if not _EXPECTED_INTERPRETER.is_file() or not _SITE_PACKAGES.is_dir():
    raise RuntimeError("TEAR-CPP audited .venv paths are missing")


class PreImportFreezeError(RuntimeError):
    """Raised before project imports when a frozen byte binding is invalid."""


_PREREG_MANIFEST = "tearrl/cpp/frozen_prereg.json"
_ANCESTOR_MANIFEST = "tearrl/cpp/frozen_ancestors.json"
_IMPLEMENTATION_MANIFEST = "tearrl/cpp/frozen_implementation.json"
_ANCESTOR_BASE = "0d241f02e63a35009aee77b1754f394570a45d13"
_PREREG_COMMIT = "2f507dcee27b22952d9c35d9fa2655b642753a04"
_PREREG_SHA256 = (
    "9a2cbf6d26cf847762eeb489ab83784c2e640a70ec98b6f4090a48b0d3e6364d"
)
_PREREG_MANIFEST_SHA256 = (
    "c845b4c8f716f6a1c14459ce4de31cfd968ab74aa9842fbcbe0739f8b377f079"
)
_ANCESTOR_MANIFEST_SHA256 = (
    "93638b054d8483a597c3d4aadf122634baf2a3eef20bb36a649473a524296626"
)
_ANCESTOR_COUNT = 179
_IMPLEMENTATION_PATHS = frozenset(
    """
    docs/cpp_implementation.md
    docs/cpp_prereg.md
    scripts/build_cpp_freeze.py
    scripts/run_cpp.py
    tearrl/cpp/__init__.py
    tearrl/cpp/calibration.py
    tearrl/cpp/config.py
    tearrl/cpp/experiment.py
    tearrl/cpp/freeze.py
    tearrl/cpp/frozen_ancestors.json
    tearrl/cpp/frozen_prereg.json
    tearrl/cpp/gate.py
    tearrl/cpp/metrics.py
    tearrl/cpp/policy.py
    tearrl/cpp/reporting.py
    tearrl/cpp/schema.py
    tearrl/cpp/severity.py
    tearrl/cpp/splits.py
    tearrl/cpp/world.py
    tests/test_cpp_calibration.py
    tests/test_cpp_experiment.py
    tests/test_cpp_freeze.py
    tests/test_cpp_freeze_generator.py
    tests/test_cpp_freeze_verifier.py
    tests/test_cpp_gate.py
    tests/test_cpp_launcher.py
    tests/test_cpp_lineage_isolation.py
    tests/test_cpp_metrics.py
    tests/test_cpp_policy.py
    tests/test_cpp_reporting.py
    tests/test_cpp_splits.py
    tests/test_cpp_world.py
    """.split()
)


def _reject_preimport_duplicates(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PreImportFreezeError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_preimport_json(
    path: Path,
    *,
    label: str,
) -> tuple[dict[str, Any], bytes]:
    try:
        payload = path.read_bytes()
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_preimport_duplicates,
        )
    except PreImportFreezeError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PreImportFreezeError(f"cannot read {label}: {path.name}") from exc
    if type(value) is not dict:
        raise PreImportFreezeError(f"{label} must be a JSON object")
    return value, payload


def _require_schema(
    value: dict[str, Any],
    expected: frozenset[str],
    *,
    label: str,
) -> None:
    if set(value) != expected:
        raise PreImportFreezeError(f"malformed {label} schema")


def _require_exact(value: Any, expected: Any, *, label: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise PreImportFreezeError(f"unexpected {label}")


def _require_hex(value: Any, *, length: int, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PreImportFreezeError(f"{label} must be lowercase hexadecimal")
    return value


def _safe_preimport_path(root: Path, relative: Any, *, label: str) -> Path:
    if (
        type(relative) is not str
        or not relative
        or "\\" in relative
        or "\0" in relative
    ):
        raise PreImportFreezeError(f"unsafe {label} path: {relative!r}")
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or pure.as_posix() != relative
        or not pure.parts
        or any(part in {"", ".", ".."} or ":" in part for part in pure.parts)
    ):
        raise PreImportFreezeError(f"unsafe {label} path: {relative!r}")
    resolved_root = root.resolve()
    try:
        candidate = root.joinpath(*pure.parts).resolve()
        candidate.relative_to(resolved_root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise PreImportFreezeError(f"{label} path escapes root: {relative!r}") from exc
    return candidate


def _read_preimport_bytes(path: Path, *, label: str) -> bytes:
    try:
        if not path.is_file():
            raise PreImportFreezeError(f"missing {label}: {path.name}")
        return path.read_bytes()
    except PreImportFreezeError:
        raise
    except OSError as exc:
        raise PreImportFreezeError(f"cannot read {label}: {path.name}") from exc


def _preimport_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _matches_frozen_bytes(payload: bytes, expected: str) -> bool:
    if _preimport_sha256(payload) == expected:
        return True
    if b"\0" in payload:
        return False
    return _preimport_sha256(payload.replace(b"\r\n", b"\n")) == expected


def _require_frozen_bytes(
    payload: bytes,
    expected: str,
    *,
    label: str,
) -> None:
    if not _matches_frozen_bytes(payload, expected):
        raise PreImportFreezeError(
            f"{label} digest mismatch; expected {expected}, "
            f"got {_preimport_sha256(payload)}"
        )


def _validated_entries(
    root: Path,
    value: Any,
    *,
    label: str,
) -> list[tuple[str, str, Path]]:
    if type(value) is not dict or not value:
        raise PreImportFreezeError(f"invalid {label} file map")
    entries: list[tuple[str, str, Path]] = []
    for relative, digest in value.items():
        path = _safe_preimport_path(root, relative, label=label)
        expected = _require_hex(
            digest,
            length=64,
            label=f"{label} digest for {relative}",
        )
        entries.append((relative, expected, path))
    return entries


def _verify_entries(
    entries: list[tuple[str, str, Path]],
    *,
    label: str,
) -> None:
    for relative, expected, path in sorted(entries):
        payload = _read_preimport_bytes(path, label=f"{label} file {relative}")
        if not _matches_frozen_bytes(payload, expected):
            raise PreImportFreezeError(
                f"{label} digest mismatch: {relative}; expected {expected}, "
                f"got {_preimport_sha256(payload)}"
            )


def _validate_preregistration(
    root: Path,
    manifest: dict[str, Any],
) -> tuple[Path, str, str]:
    _require_schema(
        manifest,
        frozenset(
            {
                "date",
                "document",
                "hash_domain",
                "parent_commit",
                "sha256",
                "status",
                "study",
            }
        ),
        label="preregistration manifest",
    )
    for key, expected in (
        ("date", "2026-07-31"),
        ("document", "docs/cpp_prereg.md"),
        ("hash_domain", "committed_git_blob_bytes"),
        ("status", "frozen_before_implementation_and_outcomes"),
        ("study", "TEAR-CPP v0.1"),
    ):
        _require_exact(
            manifest[key],
            expected,
            label=f"preregistration {key.replace('_', ' ')}",
        )
    parent = _require_hex(
        manifest["parent_commit"],
        length=40,
        label="preregistration parent commit",
    )
    _require_exact(
        parent,
        _ANCESTOR_BASE,
        label="preregistration parent commit",
    )
    digest = _require_hex(
        manifest["sha256"],
        length=64,
        label="preregistration digest",
    )
    _require_exact(
        digest,
        _PREREG_SHA256,
        label="preregistration digest",
    )
    return (
        _safe_preimport_path(
            root,
            manifest["document"],
            label="preregistration",
        ),
        digest,
        parent,
    )


def _validate_ancestors(
    root: Path,
    manifest: dict[str, Any],
) -> tuple[list[tuple[str, str, Path]], str]:
    _require_schema(
        manifest,
        frozenset(
            {
                "algorithm",
                "base_commit",
                "files",
                "hash_domain",
                "tracked_file_count",
            }
        ),
        label="ancestor manifest",
    )
    _require_exact(
        manifest["algorithm"],
        "sha256",
        label="ancestor algorithm",
    )
    _require_exact(
        manifest["hash_domain"],
        "committed_git_blob_bytes",
        label="ancestor hash domain",
    )
    base = _require_hex(
        manifest["base_commit"],
        length=40,
        label="ancestor base commit",
    )
    _require_exact(base, _ANCESTOR_BASE, label="ancestor base commit")
    entries = _validated_entries(root, manifest["files"], label="ancestor")
    _require_exact(
        manifest["tracked_file_count"],
        _ANCESTOR_COUNT,
        label="ancestor tracked-file count",
    )
    if len(entries) != _ANCESTOR_COUNT:
        raise PreImportFreezeError("ancestor tracked-file count is inconsistent")
    return entries, base


def _validate_implementation(
    root: Path,
    manifest: dict[str, Any],
) -> list[tuple[str, str, Path]]:
    _require_schema(
        manifest,
        frozenset(
            {
                "algorithm",
                "ancestor_manifest_sha256",
                "files",
                "hash_domain",
                "implementation_file_count",
                "manifest_construction_source",
                "parent_commit",
                "preregistration_commit",
                "preregistration_manifest_sha256",
                "preregistration_sha256",
                "status",
            }
        ),
        label="implementation manifest",
    )
    for key, expected in (
        ("algorithm", "sha256"),
        ("hash_domain", "committed_git_blob_bytes"),
        ("manifest_construction_source", "git_index_stage_0_blob_bytes"),
        ("status", "frozen_before_preflight_and_outcomes"),
    ):
        _require_exact(
            manifest[key],
            expected,
            label=f"implementation {key.replace('_', ' ')}",
        )
    for key in ("parent_commit", "preregistration_commit"):
        commit = _require_hex(
            manifest[key],
            length=40,
            label=f"implementation {key.replace('_', ' ')}",
        )
        _require_exact(
            commit,
            _PREREG_COMMIT,
            label=f"implementation {key.replace('_', ' ')}",
        )
    for key, expected in (
        ("preregistration_sha256", _PREREG_SHA256),
        ("preregistration_manifest_sha256", _PREREG_MANIFEST_SHA256),
        ("ancestor_manifest_sha256", _ANCESTOR_MANIFEST_SHA256),
    ):
        digest = _require_hex(
            manifest[key],
            length=64,
            label=f"implementation {key.replace('_', ' ')}",
        )
        _require_exact(
            digest,
            expected,
            label=f"implementation {key.replace('_', ' ')}",
        )
    entries = _validated_entries(
        root,
        manifest["files"],
        label="implementation",
    )
    paths = {relative for relative, _, _ in entries}
    if paths != _IMPLEMENTATION_PATHS:
        raise PreImportFreezeError("implementation file inventory is not frozen")
    _require_exact(
        manifest["implementation_file_count"],
        len(_IMPLEMENTATION_PATHS),
        label="implementation file count",
    )
    if len(entries) != len(_IMPLEMENTATION_PATHS):
        raise PreImportFreezeError("implementation file count is inconsistent")
    return entries


def _verify_preimport_freeze(root: Path) -> dict[str, Any]:
    """Verify the complete v0.1 freeze using only standard-library code."""

    resolved_root = root.resolve()
    preregistration, preregistration_bytes = _load_preimport_json(
        _safe_preimport_path(
            resolved_root,
            _PREREG_MANIFEST,
            label="preregistration manifest",
        ),
        label="preregistration manifest",
    )
    ancestors, ancestor_bytes = _load_preimport_json(
        _safe_preimport_path(
            resolved_root,
            _ANCESTOR_MANIFEST,
            label="ancestor manifest",
        ),
        label="ancestor manifest",
    )
    implementation, _ = _load_preimport_json(
        _safe_preimport_path(
            resolved_root,
            _IMPLEMENTATION_MANIFEST,
            label="implementation manifest",
        ),
        label="implementation manifest",
    )
    document_path, preregistration_digest, preregistration_parent = (
        _validate_preregistration(resolved_root, preregistration)
    )
    ancestor_entries, ancestor_base = _validate_ancestors(
        resolved_root,
        ancestors,
    )
    implementation_entries = _validate_implementation(
        resolved_root,
        implementation,
    )
    _require_frozen_bytes(
        preregistration_bytes,
        implementation["preregistration_manifest_sha256"],
        label="preregistration manifest binding",
    )
    _require_frozen_bytes(
        ancestor_bytes,
        implementation["ancestor_manifest_sha256"],
        label="ancestor manifest binding",
    )
    _require_frozen_bytes(
        _read_preimport_bytes(
            document_path,
            label="preregistration document",
        ),
        preregistration_digest,
        label="preregistration document",
    )
    _verify_entries(ancestor_entries, label="ancestor")
    _verify_entries(implementation_entries, label="implementation")
    if preregistration_parent != ancestor_base:
        raise PreImportFreezeError("preregistration and ancestor bases disagree")
    if implementation["preregistration_sha256"] != preregistration_digest:
        raise PreImportFreezeError("implementation does not bind preregistration")
    return {
        "preregistration_sha256": preregistration_digest,
        "ancestor_files": len(ancestor_entries),
        "implementation_files": len(implementation_entries),
    }


def _reject_import_shadows(root: Path) -> None:
    """Reject unbound files that could win over verified project/dependency code."""

    import_suffixes = tuple(
        {
            *importlib.machinery.EXTENSION_SUFFIXES,
            *importlib.machinery.SOURCE_SUFFIXES,
            *importlib.machinery.BYTECODE_SUFFIXES,
        }
    )
    expected_package = (root / "tearrl").resolve()
    if not expected_package.is_dir():
        raise PreImportFreezeError("verified tearrl package directory is missing")
    for module_name in ("tearrl", "numpy", "torch"):
        if module_name != "tearrl" and (root / module_name).exists():
            raise PreImportFreezeError(
                f"unbound root import shadow exists: {module_name}"
            )
        for suffix in import_suffixes:
            candidate = root / f"{module_name}{suffix}"
            if candidate.exists():
                raise PreImportFreezeError(
                    f"unbound root import shadow exists: {candidate.name}"
                )

    extension_suffixes = tuple(
        suffix.casefold() for suffix in importlib.machinery.EXTENSION_SUFFIXES
    )
    for candidate in expected_package.rglob("*"):
        if candidate.is_dir():
            if (
                candidate.name != "__pycache__"
                and candidate.with_suffix(".py").is_file()
            ):
                raise PreImportFreezeError(
                    "unbound package directory shadows verified source: "
                    f"{candidate.relative_to(root).as_posix()}"
                )
            continue
        if candidate.parent.name == "__pycache__":
            continue
        if candidate.suffix.casefold() == ".pyc" or (
            candidate.name.casefold().endswith(extension_suffixes)
        ):
            raise PreImportFreezeError(
                "unbound compiled project shadow exists: "
                f"{candidate.relative_to(root).as_posix()}"
            )


def _reject_site_tearrl_candidates(site_packages: Path) -> None:
    suffixes = tuple(
        suffix.casefold()
        for suffix in (
            *importlib.machinery.EXTENSION_SUFFIXES,
            *importlib.machinery.SOURCE_SUFFIXES,
            *importlib.machinery.BYTECODE_SUFFIXES,
        )
    )
    for candidate in site_packages.iterdir():
        name = candidate.name.casefold()
        if name == "tearrl" or any(
            name == f"tearrl{suffix}" for suffix in suffixes
        ):
            raise PreImportFreezeError(
                "audited site-packages contains a competing tearrl candidate"
            )


_PREIMPORT_FREEZE_RECEIPT = _verify_preimport_freeze(ROOT)
_reject_import_shadows(ROOT)
_reject_site_tearrl_candidates(_SITE_PACKAGES)

# Bind thread libraries only after the source gate, but still before importing
# Torch through tearrl.cpp.experiment.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

_FINAL_IMPORT_PATHS = (
    *_TRUSTED_STDLIB_PATHS,
    str(_SITE_PACKAGES),
    str(ROOT),
)
if len({path.casefold() for path in _FINAL_IMPORT_PATHS}) != len(
    _FINAL_IMPORT_PATHS
):
    raise RuntimeError("TEAR-CPP controlled import paths are not unique")
if _ACTUAL_ENTRY:
    _PYCACHE_OWNER = tempfile.TemporaryDirectory(prefix="tear-cpp-v01-pycache-")
    _PYCACHE_PREFIX: Path | None = Path(_PYCACHE_OWNER.name).resolve()
    if not _PYCACHE_PREFIX.is_dir() or any(_PYCACHE_PREFIX.iterdir()):
        raise RuntimeError("TEAR-CPP temporary bytecode cache is not empty")
    sys.pycache_prefix = str(_PYCACHE_PREFIX)
    sys.dont_write_bytecode = True
    sys.path[:] = list(_FINAL_IMPORT_PATHS)
else:
    _PYCACHE_OWNER = None
    _PYCACHE_PREFIX = None
    sys.path[:] = [
        str(ROOT),
        *_TRUSTED_STDLIB_PATHS,
        str(_SITE_PACKAGES),
    ]
sys.path_importer_cache.clear()
_ENTRY_CAPABILITY = object() if _ACTUAL_ENTRY else None

from tearrl.cpp.config import (
    CONFIRMATORY_SEEDS,
    INTEGRITY_SEED,
    WATCHDOG_ABORT_SECONDS,
)
from tearrl.cpp.experiment import (
    Watchdog,
    authorize_registered_study,
    run_confirmatory_study,
    run_timing_preflight,
)
from tearrl.cpp.freeze import sha256_file, verify_freeze_chain
from tearrl.cpp.reporting import render_markdown, stable_json_dumps


PREFLIGHT_PATH = ROOT / "artifacts" / "cpp-preflight" / "timing.json"
RESULT_DIR = ROOT / "reports" / "cpp"
RESULT_JSON = RESULT_DIR / "results.json"
RESULT_MARKDOWN = RESULT_DIR / "results.md"
FROZEN_RESULTS = ROOT / "tearrl" / "cpp" / "frozen_results.json"
RUN_LOCK = ROOT / "artifacts" / "cpp-confirmatory.lock"
STAGING_DIR = ROOT / "artifacts" / "cpp-confirmatory-staging"

_STAGED_RESULT_JSON = "results.json"
_STAGED_RESULT_MARKDOWN = "results.md"
_STAGED_FROZEN_RESULTS = "frozen_results.json"
_RESULT_DESTINATIONS = (
    (_STAGED_RESULT_JSON, RESULT_JSON),
    (_STAGED_RESULT_MARKDOWN, RESULT_MARKDOWN),
    (_STAGED_FROZEN_RESULTS, FROZEN_RESULTS),
)


def _write_new_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise RuntimeError(f"refusing to overwrite existing output: {path.name}") from exc


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json_object(path: Path, *, label: str = "preflight record") -> dict[str, Any]:
    try:
        payload = path.read_text(encoding="utf-8")
        value = json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except RuntimeError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read {label}: {path.name}") from exc
    if type(value) is not dict:
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _require_preflight(
    path: Path,
    *,
    preregistration_sha256: str,
    implementation_manifest_sha256: str,
) -> Mapping[str, Any]:
    record = _load_json_object(path)
    required = {
        "schema_version": "tear-cpp-preflight-v0.1",
        "seed": INTEGRITY_SEED,
        "scientific_metrics_suppressed": True,
        "prediction_rule": "single_seed_seconds_times_5_plus_120",
        "watchdog_seconds": WATCHDOG_ABORT_SECONDS,
        "pass": True,
        "preregistration_sha256": preregistration_sha256,
        "implementation_manifest_sha256": implementation_manifest_sha256,
    }
    for key, expected in required.items():
        actual = record.get(key)
        if type(actual) is not type(expected) or actual != expected:
            raise RuntimeError(f"preflight record failed binding: {key}")
    allowed_keys = {
        *required,
        "elapsed_seconds",
        "predicted_confirmatory_seconds",
        "integrity",
    }
    if set(record) != allowed_keys:
        raise RuntimeError(
            "preflight record contains unregistered fields or scientific metrics"
        )
    if path.read_text(encoding="utf-8") != stable_json_dumps(record):
        raise RuntimeError("preflight record bytes are not canonical")
    elapsed = record.get("elapsed_seconds")
    predicted = record.get("predicted_confirmatory_seconds")
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or float(elapsed) < 0.0
        or isinstance(predicted, bool)
        or not isinstance(predicted, (int, float))
        or not 0.0 <= float(predicted) < WATCHDOG_ABORT_SECONDS
    ):
        raise RuntimeError("preflight prediction does not clear the watchdog")
    expected_prediction = float(elapsed) * 5.0 + 120.0
    if float(predicted) != expected_prediction:
        raise RuntimeError("preflight prediction does not match frozen formula")
    integrity = record.get("integrity")
    expected_integrity_keys = {
        "c_d_frozen_digests_equal",
        "frozen_digest_stable_after_all_policy_cells",
        "common_estimator_digest_stable",
        "posterior_row_count_matches_registered_budget",
    }
    if (
        not isinstance(integrity, dict)
        or set(integrity) != expected_integrity_keys
        or not all(value is True for value in integrity.values())
    ):
        raise RuntimeError("preflight integrity assertions did not all pass")
    return record


def _freeze_receipt() -> tuple[str, str]:
    receipt = verify_freeze_chain(ROOT)
    implementation_path = ROOT / "tearrl" / "cpp" / "frozen_implementation.json"
    return (
        str(receipt["preregistration_sha256"]),
        sha256_file(implementation_path),
    )


def _assert_bindings_unchanged(
    *,
    preregistration_sha256: str,
    implementation_manifest_sha256: str,
    preflight_sha256: str,
) -> None:
    current_preregistration, current_implementation = _freeze_receipt()
    if current_preregistration != preregistration_sha256:
        raise RuntimeError("preregistration binding changed during execution")
    if current_implementation != implementation_manifest_sha256:
        raise RuntimeError("implementation binding changed during execution")
    _require_preflight(
        PREFLIGHT_PATH,
        preregistration_sha256=preregistration_sha256,
        implementation_manifest_sha256=implementation_manifest_sha256,
    )
    if sha256_file(PREFLIGHT_PATH) != preflight_sha256:
        raise RuntimeError("preflight receipt changed during execution")


def _reservation_record(
    *,
    preregistration_sha256: str,
    implementation_manifest_sha256: str,
    preflight_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": "tear-cpp-execution-reservation-v0.1",
        "status": "reserved_before_any_confirmatory_seed",
        "confirmatory_seeds": list(CONFIRMATORY_SEEDS),
        "preregistration_sha256": preregistration_sha256,
        "implementation_manifest_sha256": implementation_manifest_sha256,
        "preflight_sha256": preflight_sha256,
    }


def _create_execution_reservation(
    *,
    preregistration_sha256: str,
    implementation_manifest_sha256: str,
    preflight_sha256: str,
    lock_path: Path = RUN_LOCK,
    staging_dir: Path = STAGING_DIR,
) -> None:
    if staging_dir.exists():
        raise RuntimeError("staging directory exists without a valid reservation")
    _write_new_text(
        lock_path,
        stable_json_dumps(
            _reservation_record(
                preregistration_sha256=preregistration_sha256,
                implementation_manifest_sha256=implementation_manifest_sha256,
                preflight_sha256=preflight_sha256,
            )
        ),
    )
    try:
        staging_dir.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise RuntimeError(
            "confirmatory execution is reserved but staging creation failed; "
            "do not rerun registered seeds"
        ) from exc


def _require_execution_reservation(
    *,
    preregistration_sha256: str,
    implementation_manifest_sha256: str,
    preflight_sha256: str,
    lock_path: Path = RUN_LOCK,
) -> None:
    actual = _load_json_object(lock_path, label="execution reservation")
    expected = _reservation_record(
        preregistration_sha256=preregistration_sha256,
        implementation_manifest_sha256=implementation_manifest_sha256,
        preflight_sha256=preflight_sha256,
    )
    if actual != expected:
        raise RuntimeError("execution reservation does not match frozen receipts")
    if lock_path.read_text(encoding="utf-8") != stable_json_dumps(expected):
        raise RuntimeError("execution reservation bytes are not canonical")


def _registered_outcome_receipt(result: Mapping[str, Any]) -> dict[str, bool]:
    values = {
        "calibration_validity_pass": result["calibration_validity"]["pass"],
        "primary_pass": result["primary"]["pass"],
        "secondary_pass": result["secondary"]["pass"],
        "d_exit_pass": result["diagnostics"]["d_exit"]["pass"],
        "complete_positive_interpretation_available": result[
            "complete_positive_interpretation_available"
        ],
    }
    if any(type(value) is not bool for value in values.values()):
        raise RuntimeError("registered result gates must be booleans")
    return values


def _frozen_result_record(
    *,
    result: Mapping[str, Any],
    result_json_text: str,
    result_markdown_text: str,
    preregistration_sha256: str,
    implementation_manifest_sha256: str,
    preflight_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": "tear-cpp-frozen-results-v0.1",
        "study": "TEAR-CPP v0.1",
        "status": "confirmatory_complete_stopped_without_tuning",
        "hash_domain": "exact_file_bytes",
        "confirmatory_seeds": list(CONFIRMATORY_SEEDS),
        "preregistration_sha256": preregistration_sha256,
        "implementation_manifest_sha256": implementation_manifest_sha256,
        "preflight_sha256": preflight_sha256,
        "result_files": {
            "reports/cpp/results.json": _sha256_text(result_json_text),
            "reports/cpp/results.md": _sha256_text(result_markdown_text),
        },
        "registered_outcomes": _registered_outcome_receipt(result),
    }


def _watchdog_check(watchdog: Watchdog | None, stage: str) -> None:
    if watchdog is not None:
        watchdog.check(stage)


def _stage_result_bundle(
    result: Mapping[str, Any],
    *,
    preregistration_sha256: str,
    implementation_manifest_sha256: str,
    preflight_sha256: str,
    watchdog: Watchdog | None,
    staging_dir: Path = STAGING_DIR,
    writer: Callable[[Path, str], None] = _write_new_text,
) -> None:
    if not staging_dir.is_dir():
        raise RuntimeError("reserved staging directory is missing")
    if any(staging_dir.iterdir()):
        raise RuntimeError("reserved staging directory is not empty")

    _watchdog_check(watchdog, "result JSON serialization")
    result_json_text = stable_json_dumps(result)
    _watchdog_check(watchdog, "result Markdown rendering")
    result_markdown_text = render_markdown(result, evidence_href="results.json")
    _watchdog_check(watchdog, "frozen result receipt construction")
    frozen_result_text = stable_json_dumps(
        _frozen_result_record(
            result=result,
            result_json_text=result_json_text,
            result_markdown_text=result_markdown_text,
            preregistration_sha256=preregistration_sha256,
            implementation_manifest_sha256=implementation_manifest_sha256,
            preflight_sha256=preflight_sha256,
        )
    )

    for name, text in (
        (_STAGED_RESULT_JSON, result_json_text),
        (_STAGED_RESULT_MARKDOWN, result_markdown_text),
        (_STAGED_FROZEN_RESULTS, frozen_result_text),
    ):
        _watchdog_check(watchdog, f"staging {name}")
        writer(staging_dir / name, text)
        _watchdog_check(watchdog, f"staged {name}")


def _validate_staged_bundle(
    *,
    preregistration_sha256: str,
    implementation_manifest_sha256: str,
    preflight_sha256: str,
    staging_dir: Path = STAGING_DIR,
) -> dict[str, Any]:
    expected_names = {
        _STAGED_RESULT_JSON,
        _STAGED_RESULT_MARKDOWN,
        _STAGED_FROZEN_RESULTS,
    }
    if not staging_dir.is_dir():
        raise RuntimeError("reserved staging directory is missing")
    if {path.name for path in staging_dir.iterdir()} != expected_names:
        raise RuntimeError(
            "reserved result bundle is incomplete; do not rerun confirmatory seeds"
        )

    result_path = staging_dir / _STAGED_RESULT_JSON
    markdown_path = staging_dir / _STAGED_RESULT_MARKDOWN
    receipt_path = staging_dir / _STAGED_FROZEN_RESULTS
    result_text = result_path.read_text(encoding="utf-8")
    markdown_text = markdown_path.read_text(encoding="utf-8")
    receipt_text = receipt_path.read_text(encoding="utf-8")
    result = _load_json_object(result_path, label="staged result")
    receipt = _load_json_object(receipt_path, label="staged frozen result receipt")

    if stable_json_dumps(result) != result_text:
        raise RuntimeError("staged result JSON bytes are not canonical")
    if render_markdown(result, evidence_href="results.json") != markdown_text:
        raise RuntimeError("staged Markdown is not derived from result JSON")
    expected_receipt = _frozen_result_record(
        result=result,
        result_json_text=result_text,
        result_markdown_text=markdown_text,
        preregistration_sha256=preregistration_sha256,
        implementation_manifest_sha256=implementation_manifest_sha256,
        preflight_sha256=preflight_sha256,
    )
    if receipt != expected_receipt or stable_json_dumps(receipt) != receipt_text:
        raise RuntimeError("staged frozen result receipt is invalid")
    if result.get("mode") != "confirmatory":
        raise RuntimeError("staged result mode is not confirmatory")
    if result.get("seeds") != list(CONFIRMATORY_SEEDS):
        raise RuntimeError("staged result seed list is not registered")
    provenance = result.get("provenance")
    if not isinstance(provenance, dict) or provenance != {
        "preregistration_sha256": preregistration_sha256,
        "implementation_manifest_sha256": implementation_manifest_sha256,
    }:
        raise RuntimeError("staged result provenance is not frozen")
    return result


def _publish_staged_bundle(
    *,
    watchdog: Watchdog | None,
    staging_dir: Path = STAGING_DIR,
    destinations: tuple[tuple[str, Path], ...] = _RESULT_DESTINATIONS,
    linker: Callable[[str, str], None] = os.link,
) -> None:
    created: list[Path] = []
    try:
        for name, destination in destinations:
            staged = staging_dir / name
            if not staged.is_file():
                raise RuntimeError(f"missing staged output: {name}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            _watchdog_check(watchdog, f"publishing {name}")
            if destination.exists():
                if (
                    not destination.is_file()
                    or destination.read_bytes() != staged.read_bytes()
                ):
                    raise RuntimeError(
                        "existing published output disagrees with staged bytes: "
                        f"{name}"
                    )
            else:
                linker(str(staged), str(destination))
                created.append(destination)
            _watchdog_check(watchdog, f"published {name}")

        for name, destination in destinations:
            staged = staging_dir / name
            if destination.read_bytes() != staged.read_bytes():
                raise RuntimeError(
                    f"published output failed byte verification: {name}"
                )
        _watchdog_check(watchdog, "published bundle byte verification")
    except BaseException:
        for destination in reversed(created):
            try:
                destination.unlink()
            except FileNotFoundError:
                pass
        raise

def _require_actual_entry(capability: object | None) -> None:
    if capability is None or capability is not _ENTRY_CAPABILITY:
        raise RuntimeError("TEAR-CPP execution requires the trusted direct entry")


def _run_preflight(*, _entry_capability: object | None = None) -> int:
    _require_actual_entry(_entry_capability)
    preregistration_sha256, implementation_sha256 = _freeze_receipt()
    if PREFLIGHT_PATH.exists():
        raise RuntimeError(
            f"refusing to overwrite existing preflight: {PREFLIGHT_PATH.name}"
        )
    record = run_timing_preflight()
    record["preregistration_sha256"] = preregistration_sha256
    record["implementation_manifest_sha256"] = implementation_sha256
    _write_new_text(PREFLIGHT_PATH, stable_json_dumps(record))
    if not record["pass"]:
        raise RuntimeError("timing preflight predicts watchdog failure")
    print("timing-only preflight: PASS; scientific metrics suppressed")
    print(PREFLIGHT_PATH.relative_to(ROOT).as_posix())
    return 0


def _run_study(*, _entry_capability: object | None = None) -> int:
    _require_actual_entry(_entry_capability)
    preregistration_sha256, implementation_sha256 = _freeze_receipt()
    _require_preflight(
        PREFLIGHT_PATH,
        preregistration_sha256=preregistration_sha256,
        implementation_manifest_sha256=implementation_sha256,
    )
    preflight_sha256 = sha256_file(PREFLIGHT_PATH)

    if RUN_LOCK.exists():
        _require_execution_reservation(
            preregistration_sha256=preregistration_sha256,
            implementation_manifest_sha256=implementation_sha256,
            preflight_sha256=preflight_sha256,
        )
        _assert_bindings_unchanged(
            preregistration_sha256=preregistration_sha256,
            implementation_manifest_sha256=implementation_sha256,
            preflight_sha256=preflight_sha256,
        )
        result = _validate_staged_bundle(
            preregistration_sha256=preregistration_sha256,
            implementation_manifest_sha256=implementation_sha256,
            preflight_sha256=preflight_sha256,
        )
        recovery_watchdog = Watchdog.start()
        _publish_staged_bundle(watchdog=recovery_watchdog)
        print("confirmatory bundle recovered without re-executing registered seeds")
    else:
        if STAGING_DIR.exists():
            raise RuntimeError("staging exists without an execution reservation")
        for path in (RESULT_JSON, RESULT_MARKDOWN, FROZEN_RESULTS):
            if path.exists():
                raise RuntimeError(
                    f"refusing unreserved existing output: {path.name}"
                )
        _create_execution_reservation(
            preregistration_sha256=preregistration_sha256,
            implementation_manifest_sha256=implementation_sha256,
            preflight_sha256=preflight_sha256,
        )
        watchdog = Watchdog.start()
        authorization = authorize_registered_study(ROOT, PREFLIGHT_PATH)
        result = run_confirmatory_study(
            authorization=authorization,
            watchdog=watchdog,
        )
        _watchdog_check(watchdog, "post-execution freeze verification")
        _assert_bindings_unchanged(
            preregistration_sha256=preregistration_sha256,
            implementation_manifest_sha256=implementation_sha256,
            preflight_sha256=preflight_sha256,
        )
        _stage_result_bundle(
            result,
            preregistration_sha256=preregistration_sha256,
            implementation_manifest_sha256=implementation_sha256,
            preflight_sha256=preflight_sha256,
            watchdog=watchdog,
        )
        _watchdog_check(watchdog, "pre-publication freeze verification")
        _assert_bindings_unchanged(
            preregistration_sha256=preregistration_sha256,
            implementation_manifest_sha256=implementation_sha256,
            preflight_sha256=preflight_sha256,
        )
        _validate_staged_bundle(
            preregistration_sha256=preregistration_sha256,
            implementation_manifest_sha256=implementation_sha256,
            preflight_sha256=preflight_sha256,
        )
        _watchdog_check(watchdog, "transactional publication")
        _publish_staged_bundle(watchdog=watchdog)
        print(
            "confirmatory study completed; registered execution is permanently reserved"
        )

    print(RESULT_JSON.relative_to(ROOT).as_posix())
    print(RESULT_MARKDOWN.relative_to(ROOT).as_posix())
    print(FROZEN_RESULTS.relative_to(ROOT).as_posix())
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("preflight", "study"),
        help="preflight suppresses science; study executes frozen confirmatory seeds",
    )
    return parser


def main() -> int:
    capability = _ENTRY_CAPABILITY
    _require_actual_entry(capability)
    args = _parser().parse_args()
    if args.mode == "preflight":
        return _run_preflight(_entry_capability=capability)
    return _run_study(_entry_capability=capability)


if __name__ == "__main__":
    raise SystemExit(main())
