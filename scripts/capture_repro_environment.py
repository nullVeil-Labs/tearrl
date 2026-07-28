#!/usr/bin/env python
"""Capture an allowlisted, privacy-clean numerical reproduction profile."""

from __future__ import annotations

import argparse
from importlib import metadata
import json
import os
import platform
from pathlib import Path
from typing import Any

PACKAGE_ALLOWLIST = (
    "networkx",
    "numpy",
    "pytest",
    "scipy",
    "torch",
)
THREAD_ENVIRONMENT_ALLOWLIST = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
)


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in PACKAGE_ALLOWLIST:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"required reproduction package is missing: {name}") from exc
    return versions


def capture_environment(*, numeric_device: str = "cpu") -> dict[str, Any]:
    """Return exact relevant versions without paths, hostnames, or hardware IDs."""

    if numeric_device != "cpu":
        raise ValueError("TEAR-CG/S reproduction evidence requires numeric_device='cpu'")
    import torch

    return {
        "schema_version": 1,
        "profile_kind": "retrospective_exact_reproduction_environment",
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "platform": {
            "os_family": platform.system(),
        },
        "packages": _package_versions(),
        "numeric_runtime": {
            "device_used": str(numeric_device),
            "torch_module_version": str(torch.__version__),
            "torch_cuda_build": torch.version.cuda,
            "torch_intraop_threads": int(torch.get_num_threads()),
            "torch_interop_threads": int(torch.get_num_interop_threads()),
            **{
                name: os.environ.get(name)
                for name in THREAD_ENVIRONMENT_ALLOWLIST
            },
        },
        "limitations": [
            "This records the environment that reproduced the artifacts; it is not an installation lock.",
            "Cross-platform and cross-version bit-exactness have not been established.",
            "Hardware, host, executable, working-directory, and user identifiers are intentionally omitted.",
        ],
    }


def _relative_or_name(path: Path, root: Path) -> str:
    """Return a useful output label without exposing an absolute caller path."""

    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def _write_new_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numeric-device", choices=("cpu",), default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("refusing to overwrite existing environment output")
    payload = capture_environment(numeric_device=args.numeric_device)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        try:
            _write_new_text(args.output, text)
        except FileExistsError:
            parser.error("refusing to overwrite existing environment output")
        print(_relative_or_name(args.output, Path.cwd()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())