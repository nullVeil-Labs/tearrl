from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts" / "run_cgs_profiled.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_wrapper():
    spec = importlib.util.spec_from_file_location("run_cgs_profiled_test", WRAPPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_fake_freeze(source_root: Path) -> dict[str, str]:
    scripts = source_root / "scripts"
    cgs = source_root / "tearrl" / "cgs"
    docs = source_root / "docs"
    scripts.mkdir(parents=True)
    cgs.mkdir(parents=True)
    docs.mkdir(parents=True)

    launcher = scripts / "run_cgs.py"
    launcher.write_text(
        """
import argparse
import json
import os
from pathlib import Path
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("PROFILE_TEST_DRIFT") == "1":
        torch.set_num_threads(torch.get_num_threads() + 1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = "pilot_results" if args.mode == "pilot" else "results"
    payload = {
        "environment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "torch": {
            "num_threads": torch.get_num_threads(),
            "num_interop_threads": torch.get_num_interop_threads(),
        },
    }
    (args.output_dir / f"{stem}.json").write_text(
        json.dumps(payload, sort_keys=True) + "\\n", encoding="utf-8"
    )
    (args.output_dir / f"{stem}.md").write_text("# fake result\\n", encoding="utf-8")
    return 0
""".lstrip(),
        encoding="utf-8",
    )
    payload = source_root / "payload.txt"
    payload.write_text("final\n", encoding="utf-8")
    amendment_001_doc = docs / "amendment-001.txt"
    amendment_001_doc.write_text("amendment one\n", encoding="utf-8")
    amendment_002_doc = docs / "amendment-002.txt"
    amendment_002_doc.write_text("amendment two\n", encoding="utf-8")

    old_hash = hashlib.sha256(b"old\n").hexdigest()
    middle_hash = hashlib.sha256(b"middle\n").hexdigest()
    final_hash = _sha256(payload)
    base = {
        "implementation_files": {
            "scripts/run_cgs.py": _sha256(launcher),
            "payload.txt": old_hash,
        }
    }
    base_path = cgs / "frozen_implementation.json"
    base_path.write_text(json.dumps(base, sort_keys=True) + "\n", encoding="utf-8")
    amendment_001 = {
        "base_manifest_sha256": _sha256(base_path),
        "changed_files": {
            "payload.txt": {
                "old_sha256": old_hash,
                "new_sha256": middle_hash,
            }
        },
        "added_files": {
            "docs/amendment-001.txt": _sha256(amendment_001_doc),
        },
    }
    amendment_001_path = cgs / "frozen_implementation_amendment_001.json"
    amendment_001_path.write_text(
        json.dumps(amendment_001, sort_keys=True) + "\n", encoding="utf-8"
    )
    amendment_002 = {
        "previous_amendment_sha256": _sha256(amendment_001_path),
        "changed_files": {
            "payload.txt": {
                "old_sha256": middle_hash,
                "new_sha256": final_hash,
            }
        },
        "added_files": {
            "docs/amendment-002.txt": _sha256(amendment_002_doc),
        },
    }
    amendment_002_path = cgs / "frozen_implementation_amendment_002.json"
    amendment_002_path.write_text(
        json.dumps(amendment_002, sort_keys=True) + "\n", encoding="utf-8"
    )
    frozen_results = {
        "amendment_001_manifest_sha256": _sha256(amendment_001_path),
        "amendment_002_manifest_sha256": _sha256(amendment_002_path),
    }
    frozen_results_path = cgs / "frozen_results.json"
    frozen_results_path.write_text(
        json.dumps(frozen_results, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        path.relative_to(source_root).as_posix(): _sha256(path)
        for path in (
            base_path,
            amendment_001_path,
            amendment_002_path,
            frozen_results_path,
        )
    }


def _run_fake_profile(
    source_root: Path,
    output_dir: Path,
    expected_manifests: dict[str, str],
    *,
    drift: bool = False,
    sidecar: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    driver = """
import importlib.util
import json
import os
import sys

spec = importlib.util.spec_from_file_location("profiled_child", os.environ["WRAPPER"])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
raise SystemExit(module.main(
    sys.argv[1:],
    expected_manifest_sha256=json.loads(os.environ["EXPECTED_MANIFESTS"]),
    expected_runtime_import_sha256=json.loads(os.environ["EXPECTED_RUNTIME_IMPORTS"]),
))
"""
    environment = os.environ.copy()
    environment["WRAPPER"] = str(WRAPPER)
    environment["EXPECTED_MANIFESTS"] = json.dumps(expected_manifests)
    environment["EXPECTED_RUNTIME_IMPORTS"] = "{}"
    if drift:
        environment["PROFILE_TEST_DRIFT"] = "1"
    command = [
        sys.executable,
        "-c",
        driver,
        "--mode",
        "confirmatory",
        "--source-root",
        str(source_root),
        "--source-commit",
        "a" * 40,
        "--profile-name",
        "env1_torch1_1",
        "--output-dir",
        str(output_dir),
        "--omp-threads",
        "1",
        "--mkl-threads",
        "1",
        "--openblas-threads",
        "1",
        "--torch-intra-threads",
        "1",
        "--torch-interop-threads",
        "1",
    ]
    if sidecar is not None:
        command.extend(("--sidecar", str(sidecar)))
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def test_profiled_runner_binds_threads_freeze_chain_and_outputs(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    output_dir = tmp_path / "output"
    expected_manifests = _write_fake_freeze(source_root)

    completed = _run_fake_profile(source_root, output_dir, expected_manifests)

    assert completed.returncode == 0, completed.stderr
    sidecar_path = output_dir / "env1_torch1_1.profile.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    expected_profile = {
        "environment": {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        },
        "torch": {
            "num_threads": 1,
            "num_interop_threads": 1,
        },
    }
    assert sidecar["requested"] == expected_profile
    assert sidecar["observed"] == {
        "before": expected_profile,
        "after": expected_profile,
    }
    assert sidecar["source_commit"] == {
        "value": "a" * 40,
        "provenance": "caller_supplied",
    }
    assert sidecar["launcher"] == {
        "path": "scripts/run_cgs.py",
        "sha256": _sha256(source_root / "scripts" / "run_cgs.py"),
    }
    assert sidecar["outputs"]["json"]["sha256"] == _sha256(
        output_dir / "results.json"
    )
    assert sidecar["outputs"]["markdown"]["sha256"] == _sha256(
        output_dir / "results.md"
    )
    assert set(sidecar["freeze_chain"]["manifests"]) == set(expected_manifests)
    assert sidecar["runtime"]["packages"]["torch"]
    assert sidecar["runtime"]["torch_module_version"]
    assert str(tmp_path) not in sidecar_path.read_text(encoding="utf-8")


def test_profiled_runner_fails_closed_if_thread_count_drifts(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    output_dir = tmp_path / "output"
    expected_manifests = _write_fake_freeze(source_root)

    completed = _run_fake_profile(
        source_root,
        output_dir,
        expected_manifests,
        drift=True,
    )

    assert completed.returncode != 0
    assert "after profile mismatch" in completed.stderr
    assert not (output_dir / "env1_torch1_1.profile.json").exists()


@pytest.mark.parametrize("existing_name", ("results.json", "results.md"))
def test_profiled_runner_refuses_to_overwrite_existing_outputs(
    tmp_path: Path,
    existing_name: str,
) -> None:
    source_root = tmp_path / "source"
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / existing_name).write_text("preserve me\n", encoding="utf-8")
    expected_manifests = _write_fake_freeze(source_root)

    completed = _run_fake_profile(source_root, output_dir, expected_manifests)

    assert completed.returncode != 0
    assert "refusing to overwrite an existing output" in completed.stderr
    assert (output_dir / existing_name).read_text(encoding="utf-8") == "preserve me\n"
    assert not (output_dir / "env1_torch1_1.profile.json").exists()


@pytest.mark.parametrize("collision_name", ("results.json", "results.md"))
def test_profiled_runner_rejects_sidecar_output_collisions(
    tmp_path: Path,
    collision_name: str,
) -> None:
    source_root = tmp_path / "source"
    output_dir = tmp_path / "output"
    expected_manifests = _write_fake_freeze(source_root)

    completed = _run_fake_profile(
        source_root,
        output_dir,
        expected_manifests,
        sidecar=output_dir / collision_name,
    )

    assert completed.returncode != 0
    assert "sidecar must be distinct" in completed.stderr
    assert not (output_dir / collision_name).exists()


def test_freeze_chain_rejects_a_tampered_current_file(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    expected_manifests = _write_fake_freeze(source_root)
    wrapper = _load_wrapper()
    (source_root / "payload.txt").write_text("tampered\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="frozen implementation hash mismatch"):
        wrapper._verify_freeze_chain(
            source_root,
            expected_manifest_sha256=expected_manifests,
            expected_runtime_import_sha256={},
        )
