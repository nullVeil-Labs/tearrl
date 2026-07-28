from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "capture_repro_environment.py"


def _module():
    spec = importlib.util.spec_from_file_location("capture_repro_environment", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_environment_capture_is_allowlisted_and_privacy_clean(monkeypatch) -> None:
    module = _module()
    for name in module.THREAD_ENVIRONMENT_ALLOWLIST:
        monkeypatch.setenv(name, "8")
    payload = module.capture_environment(numeric_device="cpu")
    assert payload["schema_version"] == 1
    assert payload["profile_kind"] == "retrospective_exact_reproduction_environment"
    assert set(payload["packages"]) == set(module.PACKAGE_ALLOWLIST)
    assert payload["numeric_runtime"]["device_used"] == "cpu"
    assert payload["numeric_runtime"]["torch_module_version"] == str(torch.__version__)
    assert {
        name: payload["numeric_runtime"][name]
        for name in module.THREAD_ENVIRONMENT_ALLOWLIST
    } == {name: "8" for name in module.THREAD_ENVIRONMENT_ALLOWLIST}

    def keys(value):
        if isinstance(value, dict):
            for key, item in value.items():
                yield str(key).lower()
                yield from keys(item)
        elif isinstance(value, list):
            for item in value:
                yield from keys(item)

    observed_keys = set(keys(payload))
    for forbidden in (
        "hostname",
        "processor",
        "executable",
        "working_directory",
        "cwd",
        "user_name",
        "device_name",
    ):
        assert forbidden not in observed_keys

def test_environment_capture_rejects_non_cpu_device() -> None:
    module = _module()
    with pytest.raises(ValueError, match="numeric_device='cpu'"):
        module.capture_environment(numeric_device="gpu-private-text")


def test_output_label_is_relative_or_basename_only(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path / "repo"
    root.mkdir()
    inside = root / "reports" / "evidence.json"
    outside = tmp_path / "private" / "evidence.json"

    assert module._relative_or_name(inside, root) == "reports/evidence.json"
    assert module._relative_or_name(outside, root) == "evidence.json"
    assert str(tmp_path) not in module._relative_or_name(outside, root)

def test_environment_output_exclusive_create_preserves_existing_file(
    tmp_path: Path,
) -> None:
    module = _module()
    output = tmp_path / "environment.json"
    output.write_text("preserve\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        module._write_new_text(output, "replace\n")
    assert output.read_text(encoding="utf-8") == "preserve\n"

def test_main_refuses_existing_environment_before_capture(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _module()
    output = tmp_path / "environment.json"
    output.write_text("preserve\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SCRIPT), "--output", str(output)],
    )
    monkeypatch.setattr(
        module,
        "capture_environment",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("capture must not run")
        ),
    )

    with pytest.raises(SystemExit) as raised:
        module.main()
    assert raised.value.code == 2
    assert output.read_text(encoding="utf-8") == "preserve\n"
