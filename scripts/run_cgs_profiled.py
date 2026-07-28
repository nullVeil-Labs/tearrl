#!/usr/bin/env python
"""Run the frozen TEAR-CG/S launcher with a bound bundled numerical-thread profile."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import sys
from types import ModuleType
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
)
_EXPECTED_MANIFEST_SHA256 = {
    "tearrl/cgs/frozen_implementation.json": (
        "76e7063bba9ba090e87d3b6f72a7936dfbd43687ebe5fb285c1232319f2f2c05"
    ),
    "tearrl/cgs/frozen_implementation_amendment_001.json": (
        "909380a2c315e6e62567f20657701da2e4962ce94f7a074a6f052ae7ffcb09da"
    ),
    "tearrl/cgs/frozen_implementation_amendment_002.json": (
        "eab0ee54d100adce596ee9c896ff5be3537b44929c16b1727240b58eff7a3187"
    ),
    "tearrl/cgs/frozen_results.json": (
        "4f26f2b800b0ea3c87cb4e3fd263a9652c283a3be3bc3fb4b58d91c24ed3f2e1"
    ),
}
_EXPECTED_RUNTIME_IMPORT_SHA256 = {
    "tearrl/__init__.py": (
        "8e0c2867813d24a103d7d49c35b23e27defa85b031b0954f4eb6c7691ce168cc"
    ),
    "tearrl/ledger.py": (
        "e2303c7baa2607f2aa37df877a424b1f5b51fe02dca7ed004edeee211237ab75"
    ),
    "tearrl/spec.py": (
        "d143b3d40893e70faac1f473b85af93c04bc19265e77b84d0706d3db4a683672"
    ),
    "tearrl/state.py": (
        "8c20d6836da0d8ae44de4529485e873d03bac96f96ace6cee959dec20dda9734"
    ),
    "tearrl/world.py": (
        "45f09a7a0a21357f37ad1ea991b0ed7feb77f1384bc0df39971f5a4a10cff902"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("thread counts must be positive")
    return parsed


def _source_commit(value: str) -> str:
    if re.fullmatch(r"[0-9a-fA-F]{40}", value) is None:
        raise argparse.ArgumentTypeError("source commit must be a full 40-hex SHA")
    return value.lower()


def _profile_name(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", value) is None:
        raise argparse.ArgumentTypeError(
            "profile name must be 1-64 portable filename characters"
        )
    return value


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def _apply_amendment(
    expected_files: dict[str, str],
    amendment: dict[str, Any],
    *,
    label: str,
) -> None:
    for relative, change in amendment.get("changed_files", {}).items():
        old_hash = str(change["old_sha256"])
        if expected_files.get(relative) != old_hash:
            raise RuntimeError(f"{label} old hash does not continue chain: {relative}")
        expected_files[relative] = str(change["new_sha256"])
    for relative, digest in amendment.get("added_files", {}).items():
        if relative in expected_files:
            raise RuntimeError(f"{label} re-adds an existing path: {relative}")
        expected_files[relative] = str(digest)


def _verify_freeze_chain(
    source_root: Path,
    *,
    expected_manifest_sha256: dict[str, str] | None = None,
    expected_runtime_import_sha256: dict[str, str] | None = None,
) -> dict[str, Any]:
    anchors = (
        _EXPECTED_MANIFEST_SHA256
        if expected_manifest_sha256 is None
        else expected_manifest_sha256
    )
    runtime_imports = (
        _EXPECTED_RUNTIME_IMPORT_SHA256
        if expected_runtime_import_sha256 is None
        else expected_runtime_import_sha256
    )
    manifest_hashes: dict[str, str] = {}
    for relative, expected_hash in anchors.items():
        path = source_root / relative
        if not path.is_file():
            raise RuntimeError(f"freeze-chain file is missing: {relative}")
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise RuntimeError(f"freeze-chain hash mismatch: {relative}")
        manifest_hashes[relative] = actual_hash

    base_path = source_root / "tearrl/cgs/frozen_implementation.json"
    amendment_001_path = (
        source_root / "tearrl/cgs/frozen_implementation_amendment_001.json"
    )
    amendment_002_path = (
        source_root / "tearrl/cgs/frozen_implementation_amendment_002.json"
    )
    results_path = source_root / "tearrl/cgs/frozen_results.json"
    base = _load_json(base_path)
    amendment_001 = _load_json(amendment_001_path)
    amendment_002 = _load_json(amendment_002_path)
    frozen_results = _load_json(results_path)

    base_hash = manifest_hashes["tearrl/cgs/frozen_implementation.json"]
    amendment_001_hash = manifest_hashes[
        "tearrl/cgs/frozen_implementation_amendment_001.json"
    ]
    amendment_002_hash = manifest_hashes[
        "tearrl/cgs/frozen_implementation_amendment_002.json"
    ]
    if amendment_001.get("base_manifest_sha256") != base_hash:
        raise RuntimeError("Amendment 001 does not bind the base manifest")
    if amendment_002.get("previous_amendment_sha256") != amendment_001_hash:
        raise RuntimeError("Amendment 002 does not bind Amendment 001")
    if frozen_results.get("amendment_001_manifest_sha256") != amendment_001_hash:
        raise RuntimeError("frozen-results manifest does not bind Amendment 001")
    if frozen_results.get("amendment_002_manifest_sha256") != amendment_002_hash:
        raise RuntimeError("frozen-results manifest does not bind Amendment 002")

    expected_files = {
        str(relative): str(digest)
        for relative, digest in base["implementation_files"].items()
    }
    _apply_amendment(expected_files, amendment_001, label="Amendment 001")
    _apply_amendment(expected_files, amendment_002, label="Amendment 002")
    for relative, digest in amendment_002.get("preserved_pilot_files", {}).items():
        existing = expected_files.get(relative)
        if existing is not None and existing != digest:
            raise RuntimeError(f"preserved pilot hash conflicts with chain: {relative}")
        expected_files[relative] = str(digest)

    for relative, expected_hash in runtime_imports.items():
        existing = expected_files.get(relative)
        if existing is not None and existing != expected_hash:
            raise RuntimeError(f"runtime import hash conflicts with chain: {relative}")
        expected_files[relative] = expected_hash

    for relative, expected_hash in expected_files.items():
        path = source_root / relative
        if not path.is_file():
            raise RuntimeError(f"frozen implementation file is missing: {relative}")
        if _sha256(path) != expected_hash:
            raise RuntimeError(f"frozen implementation hash mismatch: {relative}")

    return {
        "manifests": {
            relative: {"path": relative, "sha256": digest}
            for relative, digest in sorted(manifest_hashes.items())
        },
        "verified_files": {
            relative: digest for relative, digest in sorted(expected_files.items())
        },
        "runtime_import_closure": {
            relative: digest for relative, digest in sorted(runtime_imports.items())
        },
        "source_tree_policy": "clean_archive_export_of_declared_commit_required",
    }


def _requested_profile(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "environment": {
            "OMP_NUM_THREADS": str(args.omp_threads),
            "MKL_NUM_THREADS": str(args.mkl_threads),
            "OPENBLAS_NUM_THREADS": str(args.openblas_threads),
        },
        "torch": {
            "num_threads": args.torch_intra_threads,
            "num_interop_threads": args.torch_interop_threads,
        },
    }


def _observe_profile(torch: ModuleType) -> dict[str, Any]:
    return {
        "environment": {name: os.environ.get(name) for name in _THREAD_ENVIRONMENT},
        "torch": {
            "num_threads": int(torch.get_num_threads()),
            "num_interop_threads": int(torch.get_num_interop_threads()),
        },
    }


def _require_profile(
    requested: dict[str, Any],
    observed: dict[str, Any],
    *,
    phase: str,
) -> None:
    if observed != requested:
        raise RuntimeError(
            f"{phase} profile mismatch: requested={requested!r}, observed={observed!r}"
        )


def _load_frozen_launcher(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("_tearrl_frozen_run_cgs", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load frozen TEAR-CG/S launcher")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_frozen_launcher(
    launcher: ModuleType,
    *,
    mode: str,
    output_dir: Path,
) -> None:
    previous_argv = sys.argv[:]
    try:
        sys.argv = [
            str(getattr(launcher, "__file__", "scripts/run_cgs.py")),
            "--mode",
            mode,
            "--output-dir",
            str(output_dir),
        ]
        return_code = launcher.main()
    finally:
        sys.argv = previous_argv
    if return_code not in (None, 0):
        raise RuntimeError(f"frozen launcher returned {return_code}")


def _output_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.name,
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _relative_or_name(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "confirmatory"), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-commit", type=_source_commit, required=True)
    parser.add_argument("--profile-name", type=_profile_name, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path)
    parser.add_argument("--omp-threads", type=_positive_integer, required=True)
    parser.add_argument("--mkl-threads", type=_positive_integer, required=True)
    parser.add_argument("--openblas-threads", type=_positive_integer, required=True)
    parser.add_argument("--torch-intra-threads", type=_positive_integer, required=True)
    parser.add_argument("--torch-interop-threads", type=_positive_integer, required=True)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    expected_manifest_sha256: dict[str, str] | None = None,
    expected_runtime_import_sha256: dict[str, str] | None = None,
) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    source_root = args.source_root.resolve()
    launcher_path = source_root / "scripts/run_cgs.py"
    output_dir = args.output_dir
    stem = "pilot_results" if args.mode == "pilot" else "results"
    json_path = output_dir / f"{stem}.json"
    markdown_path = output_dir / f"{stem}.md"
    sidecar_path = args.sidecar or output_dir / f"{args.profile_name}.profile.json"

    frozen_output_dir = (source_root / "reports/cgs").resolve()
    resolved_sidecar = sidecar_path.resolve()
    if resolved_sidecar in (json_path.resolve(), markdown_path.resolve()):
        parser.error("profile sidecar must be distinct from both result outputs")
    if resolved_sidecar in (
        frozen_output_dir / f"{stem}.json",
        frozen_output_dir / f"{stem}.md",
    ):
        parser.error("profile sidecar must not overwrite a frozen result artifact")
    if json_path.resolve() == (frozen_output_dir / f"{stem}.json"):
        parser.error("profiled runs must not overwrite the frozen JSON artifact")
    if markdown_path.resolve() == (frozen_output_dir / f"{stem}.md"):
        parser.error("profiled runs must not overwrite the frozen Markdown artifact")
    if sidecar_path.exists():
        parser.error("refusing to overwrite an existing profile sidecar")
    for candidate in (json_path, markdown_path):
        if candidate.exists():
            parser.error(f"refusing to overwrite an existing output: {candidate.name}")
    forbidden_imports = ("torch", "numpy", "scipy", "tearrl")
    if any(
        name == prefix or name.startswith(f"{prefix}.")
        for name in sys.modules
        for prefix in forbidden_imports
    ):
        raise RuntimeError(
            "run_cgs_profiled.py must start in a fresh Python process"
        )

    freeze_chain = _verify_freeze_chain(
        source_root,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_runtime_import_sha256=expected_runtime_import_sha256,
    )
    launcher_hash = _sha256(launcher_path)
    requested = _requested_profile(args)
    for name, value in requested["environment"].items():
        os.environ[name] = value

    torch = importlib.import_module("torch")
    torch.set_num_threads(args.torch_intra_threads)
    torch.set_num_interop_threads(args.torch_interop_threads)
    observed_before = _observe_profile(torch)
    _require_profile(requested, observed_before, phase="before")

    launcher = _load_frozen_launcher(launcher_path)
    _run_frozen_launcher(launcher, mode=args.mode, output_dir=output_dir)

    observed_after = _observe_profile(torch)
    _require_profile(requested, observed_after, phase="after")
    if _sha256(launcher_path) != launcher_hash:
        raise RuntimeError("frozen launcher changed during the run")
    if not json_path.is_file() or not markdown_path.is_file():
        raise RuntimeError("frozen launcher did not write both expected outputs")

    sidecar = {
        "schema_version": 1,
        "study": "TEAR-CG/S v0.1",
        "mode": args.mode,
        "profile_name": args.profile_name,
        "source_commit": {
            "value": args.source_commit,
            "provenance": "caller_supplied",
        },
        "freeze_chain": freeze_chain,
        "launcher": {
            "path": "scripts/run_cgs.py",
            "sha256": launcher_hash,
        },
        "requested": requested,
        "observed": {
            "before": observed_before,
            "after": observed_after,
        },
        "runtime": {
            "python": {
                "implementation": platform.python_implementation(),
                "version": platform.python_version(),
            },
            "os_family": platform.system(),
            "packages": {
                name: _distribution_version(name)
                for name in ("networkx", "numpy", "pytest", "scipy", "torch")
            },
            "torch_module_version": str(torch.__version__),
            "torch_cuda_build": (
                None if torch.version.cuda is None else str(torch.version.cuda)
            ),
        },
        "outputs": {
            "json": _output_record(json_path),
            "markdown": _output_record(markdown_path),
        },
    }
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(_relative_or_name(sidecar_path, source_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
