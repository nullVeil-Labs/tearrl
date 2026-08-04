from __future__ import annotations

import ast
import importlib.util
import json
import os
from pathlib import Path
import py_compile
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_cpp.py"
SPEC = importlib.util.spec_from_file_location("run_cpp_for_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def _safe_launcher_command(*arguments: str) -> list[str]:
    return [
        sys.executable,
        "-I",
        "-S",
        "-P",
        "-B",
        str(SCRIPT),
        *arguments,
    ]


def _launcher_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
        }
    )
    return environment


def _copy_frozen_surface(destination: Path) -> Path:
    snapshot = destination / "snapshot"
    manifest_relatives = {
        "tearrl/cpp/frozen_prereg.json",
        "tearrl/cpp/frozen_ancestors.json",
        "tearrl/cpp/frozen_implementation.json",
    }
    ancestors = json.loads(
        (ROOT / "tearrl" / "cpp" / "frozen_ancestors.json").read_text(
            encoding="utf-8"
        )
    )
    implementation = json.loads(
        (ROOT / "tearrl" / "cpp" / "frozen_implementation.json").read_text(
            encoding="utf-8"
        )
    )
    relatives = (
        manifest_relatives
        | set(ancestors["files"])
        | set(implementation["files"])
    )
    for relative in sorted(relatives):
        source = ROOT.joinpath(*relative.split("/"))
        target = snapshot.joinpath(*relative.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return snapshot


def test_preimport_gate_precedes_sys_path_and_all_project_imports() -> None:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"), filename=str(SCRIPT))
    sys_import = next(
        node
        for node in tree.body
        if isinstance(node, ast.Import)
        and [alias.name for alias in node.names] == ["sys"]
    )
    first_shadowable_import = next(
        node
        for node in tree.body
        if isinstance(node, ast.Import)
        and any(alias.name == "argparse" for alias in node.names)
    )
    actual_entry_guard = next(
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "_ACTUAL_ENTRY"
    )
    startup_path_replacement = next(
        node
        for node in ast.walk(actual_entry_guard)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Attribute)
            and target.value.attr == "path"
            and isinstance(target.value.value, ast.Name)
            and target.value.value.id == "sys"
            for target in node.targets
        )
    )
    assert sys_import.lineno < startup_path_replacement.lineno
    assert startup_path_replacement.lineno < first_shadowable_import.lineno

    gate_index = next(
        index
        for index, node in enumerate(tree.body)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "_PREIMPORT_FREEZE_RECEIPT"
            for target in node.targets
        )
    )
    prefix = ast.Module(body=tree.body[:gate_index], type_ignores=[])
    allowed_stdlib = {
        "__future__",
        "argparse",
        "hashlib",
        "importlib",
        "json",
        "os",
        "pathlib",
        "sys",
        "tempfile",
        "typing",
    }
    for node in ast.walk(prefix):
        if isinstance(node, ast.Import):
            assert {
                alias.name.partition(".")[0] for alias in node.names
            } <= allowed_stdlib
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").partition(".")[0] in allowed_stdlib
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "insert"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "path"
            and isinstance(node.func.value.value, ast.Name)
            and node.func.value.value.id == "sys"
        ):
            pytest.fail("ROOT was inserted into sys.path before freeze verification")

    project_import_indices = [
        index
        for index, node in enumerate(tree.body)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and (
            (
                isinstance(node, ast.Import)
                and any(
                    alias.name.partition(".")[0] in {"tearrl", "numpy", "torch"}
                    for alias in node.names
                )
            )
            or (
                isinstance(node, ast.ImportFrom)
                and (node.module or "").partition(".")[0]
                in {"tearrl", "numpy", "torch"}
            )
        )
    ]
    assert project_import_indices
    assert min(project_import_indices) > gate_index
    assert RUNNER._PREIMPORT_FREEZE_RECEIPT == {
        "preregistration_sha256": RUNNER._PREREG_SHA256,
        "ancestor_files": 179,
        "implementation_files": 32,
    }
    assert sys.path[0] == str(ROOT)
    assert tuple(sys.path[1 : 1 + len(RUNNER._TRUSTED_STDLIB_PATHS)]) == (
        RUNNER._TRUSTED_STDLIB_PATHS
    )
    assert sys.path[-1] == str(RUNNER._SITE_PACKAGES)
    assert RUNNER._FINAL_IMPORT_PATHS == (
        *RUNNER._TRUSTED_STDLIB_PATHS,
        str(RUNNER._SITE_PACKAGES),
        str(ROOT),
    )
    assert RUNNER._PYCACHE_PREFIX is None
    assert os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] == "0"


def test_imported_module_cannot_reach_execution_entry_points() -> None:
    for entry in (RUNNER.main, RUNNER._run_preflight, RUNNER._run_study):
        with pytest.raises(RuntimeError, match="trusted direct entry"):
            entry()


def test_unsafe_launch_refuses_before_script_shadow_import(
    tmp_path: Path,
) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    copied_launcher = scripts / "run_cpp.py"
    shutil.copy2(SCRIPT, copied_launcher)
    marker = tmp_path / "shadow-imported.txt"
    (scripts / "json.py").write_text(
        "from pathlib import Path\n"
        f"Path({json.dumps(str(marker))}).write_text("
        "'imported', encoding='utf-8')\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, str(copied_launcher), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=os.environ.copy(),
    )
    assert completed.returncode != 0
    assert "requires -I -S -P -B" in completed.stderr
    assert not marker.exists()


def test_controlled_stdlib_paths_ignore_temp_script_shadow(
    tmp_path: Path,
) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    marker = tmp_path / "shadow-imported.txt"
    (scripts / "json.py").write_text(
        "from pathlib import Path\n"
        f"Path({json.dumps(str(marker))}).write_text("
        "'imported', encoding='utf-8')\n"
        "raise RuntimeError('script-dir json shadow executed')\n",
        encoding="utf-8",
    )
    trusted = json.dumps(list(RUNNER._TRUSTED_STDLIB_PATHS))
    bootstrap = "\n".join(
        (
            "import sys",
            f"sys.path[:] = [{json.dumps(str(scripts))}, *{trusted}]",
            f"sys.path[:] = {trusted}",
            "sys.path_importer_cache.clear()",
            "import json",
            "print(json.__file__)",
        )
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-P", "-B", "-c", bootstrap],
        cwd=scripts,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=_launcher_environment(),
    )
    assert completed.returncode == 0, completed.stderr
    assert str(Path(sys.base_prefix) / "Lib") in completed.stdout
    assert not marker.exists()


def test_module_import_discards_fake_path_and_places_root_before_site(
    tmp_path: Path,
) -> None:
    fake = tmp_path / "fake"
    fake_package = fake / "tearrl"
    fake_package.mkdir(parents=True)
    marker = tmp_path / "fake-tearrl-imported.txt"
    (fake_package / "__init__.py").write_text(
        "from pathlib import Path\n"
        f"Path({json.dumps(str(marker))}).write_text("
        "'imported', encoding='utf-8')\n",
        encoding="utf-8",
    )
    bootstrap = "\n".join(
        (
            "import importlib.util",
            "from pathlib import Path",
            "import sys",
            f"fake = Path({json.dumps(str(fake))})",
            f"root = Path({json.dumps(str(ROOT))})",
            f"script = Path({json.dumps(str(SCRIPT))})",
            "sys.path[:] = [str(fake), *sys.path, str(root)]",
            "spec = importlib.util.spec_from_file_location('cpp_path_probe', script)",
            "assert spec is not None and spec.loader is not None",
            "module = importlib.util.module_from_spec(spec)",
            "spec.loader.exec_module(module)",
            "import tearrl",
            "assert str(fake) not in sys.path",
            "assert sys.path[0] == str(root)",
            "assert tuple(sys.path[1:1 + len(module._TRUSTED_STDLIB_PATHS)]) == module._TRUSTED_STDLIB_PATHS",
            "assert sys.path[-1] == str(module._SITE_PACKAGES)",
            "assert Path(tearrl.__file__).resolve() == root / 'tearrl' / '__init__.py'",
            "print('CONTROLLED_PATHS_OK')",
        )
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-P", "-B", "-c", bootstrap],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=_launcher_environment(),
    )
    assert completed.returncode == 0, completed.stderr
    assert "CONTROLLED_PATHS_OK" in completed.stdout
    assert not marker.exists()


def test_empty_pycache_prefix_ignores_temp_unchecked_hash_pyc(
    tmp_path: Path,
) -> None:
    import_root = tmp_path / "import-root"
    package = import_root / "probe"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    source_marker = tmp_path / "verified-source-imported.txt"
    marker = tmp_path / "poisoned-pyc-imported.txt"
    source = package / "config.py"
    source.write_text(
        "from pathlib import Path\n"
        f"Path({json.dumps(str(source_marker))}).write_text("
        "'source', encoding='utf-8')\n"
        "VALUE = 'VERIFIED_SOURCE'\n",
        encoding="utf-8",
    )
    cache_dir = package / "__pycache__"
    cache_dir.mkdir()
    cache = cache_dir / f"config.{sys.implementation.cache_tag}.pyc"
    poison_source = tmp_path / "poison_config.py"
    poison_source.write_text(
        "from pathlib import Path\n"
        f"Path({json.dumps(str(marker))}).write_text("
        "'imported', encoding='utf-8')\n"
        "raise RuntimeError('poisoned unchecked-hash pyc executed')\n",
        encoding="utf-8",
    )
    py_compile.compile(
        str(poison_source),
        cfile=str(cache),
        doraise=True,
        invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH,
    )
    empty_prefix = tmp_path / "empty-pycache"
    empty_prefix.mkdir()
    bootstrap = "\n".join(
        (
            "import sys",
            f"sys.pycache_prefix = {json.dumps(str(empty_prefix))}",
            "sys.dont_write_bytecode = True",
            f"sys.path.insert(0, {json.dumps(str(import_root))})",
            "from probe import config",
            "print(config.VALUE)",
        )
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-P", "-B", "-c", bootstrap],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=_launcher_environment(),
    )
    assert completed.returncode == 0, completed.stderr
    assert "VERIFIED_SOURCE" in completed.stdout
    assert source_marker.is_file()
    assert not marker.exists()
    assert not list(empty_prefix.iterdir())


def test_shadow_inventory_rejects_direct_pyc_and_casefolded_extension(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    package = root / "tearrl"
    package.mkdir(parents=True)
    (root / "numpy.pyc").write_bytes(b"poison")
    with pytest.raises(RUNNER.PreImportFreezeError, match="root import shadow"):
        RUNNER._reject_import_shadows(root)
    (root / "numpy.pyc").unlink()
    (package / "config.PYD").write_bytes(b"poison")
    with pytest.raises(
        RUNNER.PreImportFreezeError,
        match="compiled project shadow",
    ):
        RUNNER._reject_import_shadows(root)


def test_site_packages_tearrl_candidate_is_rejected(tmp_path: Path) -> None:
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()
    RUNNER._reject_site_tearrl_candidates(site_packages)
    (site_packages / "TEARRL.PYC").write_bytes(b"poison")
    with pytest.raises(RUNNER.PreImportFreezeError, match="competing tearrl"):
        RUNNER._reject_site_tearrl_candidates(site_packages)


def test_preimport_json_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text(
        '{"files":{"same":"first","same":"second"}}',
        encoding="utf-8",
    )
    with pytest.raises(RUNNER.PreImportFreezeError, match="duplicate JSON key"):
        RUNNER._load_preimport_json(path, label="synthetic manifest")


@pytest.mark.parametrize(
    "relative",
    ("../outside", "/absolute", "C:/outside", "a\\b", "a//b", "./a", None),
)
def test_preimport_paths_fail_closed(relative: object, tmp_path: Path) -> None:
    with pytest.raises(RUNNER.PreImportFreezeError, match="unsafe"):
        RUNNER._safe_preimport_path(tmp_path, relative, label="synthetic")


@pytest.mark.parametrize(
    "digest",
    ("", "a" * 63, "A" * 64, "g" * 64, 7, None),
)
def test_preimport_digests_fail_closed(digest: object) -> None:
    with pytest.raises(
        RUNNER.PreImportFreezeError,
        match="lowercase hexadecimal",
    ):
        RUNNER._require_hex(digest, length=64, label="synthetic digest")


def test_preimport_schema_and_crlf_fallback_are_strict() -> None:
    with pytest.raises(RUNNER.PreImportFreezeError, match="malformed"):
        RUNNER._require_schema(
            {"known": True, "extra": False},
            frozenset({"known"}),
            label="synthetic manifest",
        )
    committed = b"first\nsecond\n"
    checkout = b"first\r\nsecond\r\n"
    expected = RUNNER._preimport_sha256(committed)
    assert RUNNER._matches_frozen_bytes(checkout, expected)
    assert not RUNNER._matches_frozen_bytes(checkout + b"\0", expected)


def test_source_mutation_fails_before_project_or_heavy_imports(
    tmp_path: Path,
) -> None:
    snapshot = _copy_frozen_surface(tmp_path)
    marker = tmp_path / "project-imported.txt"
    target = snapshot / "tearrl" / "cpp" / "__init__.py"
    original = target.read_text(encoding="utf-8")
    marker_literal = json.dumps(str(marker))
    target.write_text(
        "from pathlib import Path as _MarkerPath\n"
        f"_MarkerPath({marker_literal}).write_text("
        "'imported', encoding='utf-8')\n"
        + original,
        encoding="utf-8",
    )
    with pytest.raises(
        RUNNER.PreImportFreezeError,
        match=r"implementation digest mismatch: tearrl/cpp/__init__\.py",
    ):
        RUNNER._verify_preimport_freeze(snapshot)
    assert not marker.exists()


def _preflight_record() -> dict[str, object]:
    return {
        "schema_version": "tear-cpp-preflight-v0.1",
        "seed": 1709,
        "scientific_metrics_suppressed": True,
        "elapsed_seconds": 1.0,
        "prediction_rule": "single_seed_seconds_times_5_plus_120",
        "predicted_confirmatory_seconds": 125.0,
        "watchdog_seconds": 870,
        "pass": True,
        "preregistration_sha256": "a" * 64,
        "implementation_manifest_sha256": "b" * 64,
        "integrity": {
            "c_d_frozen_digests_equal": True,
            "frozen_digest_stable_after_all_policy_cells": True,
            "common_estimator_digest_stable": True,
            "posterior_row_count_matches_registered_budget": True,
        },
    }


def test_help_is_read_only_and_exposes_no_tuning_flags() -> None:
    before = sorted((ROOT / "reports" / "cpp").glob("*"))
    completed = subprocess.run(
        _safe_launcher_command("--help"),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=_launcher_environment(),
    )
    after = sorted((ROOT / "reports" / "cpp").glob("*"))
    assert completed.returncode == 0
    assert "--mode" in completed.stdout
    for forbidden in ("--seed", "--epoch", "--threshold", "--severity", "--budget"):
        assert forbidden not in completed.stdout
    assert before == after


def test_write_new_text_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "result.txt"
    RUNNER._write_new_text(output, "first\n")
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        RUNNER._write_new_text(output, "second\n")
    assert output.read_text(encoding="utf-8") == "first\n"


def test_preflight_binding_and_suppression_are_fail_closed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "timing.json"
    record = _preflight_record()
    path.write_text(RUNNER.stable_json_dumps(record), encoding="utf-8")
    loaded = RUNNER._require_preflight(
        path,
        preregistration_sha256="a" * 64,
        implementation_manifest_sha256="b" * 64,
    )
    assert loaded["pass"] is True

    record["primary"] = {"pass": True}
    path.write_text(RUNNER.stable_json_dumps(record), encoding="utf-8")
    with pytest.raises(RuntimeError, match="scientific metrics"):
        RUNNER._require_preflight(
            path,
            preregistration_sha256="a" * 64,
            implementation_manifest_sha256="b" * 64,
        )


def test_preflight_rejects_duplicate_and_noncanonical_json(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema_version":"first","schema_version":"second"}',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="duplicate JSON key"):
        RUNNER._require_preflight(
            duplicate,
            preregistration_sha256="a" * 64,
            implementation_manifest_sha256="b" * 64,
        )

    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_text(
        json.dumps(_preflight_record(), sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="not canonical"):
        RUNNER._require_preflight(
            noncanonical,
            preregistration_sha256="a" * 64,
            implementation_manifest_sha256="b" * 64,
        )

@pytest.mark.parametrize(
    "field,value,match",
    [
        ("seed", 2609, "seed"),
        ("scientific_metrics_suppressed", False, "scientific_metrics_suppressed"),
        ("pass", 1, "pass"),
        ("implementation_manifest_sha256", "c" * 64, "implementation"),
        ("predicted_confirmatory_seconds", 870.0, "watchdog"),
    ],
)
def test_preflight_mutations_are_rejected(
    tmp_path: Path,
    field: str,
    value: object,
    match: str,
) -> None:
    record = _preflight_record()
    record[field] = value
    path = tmp_path / f"{field}.json"
    path.write_text(RUNNER.stable_json_dumps(record), encoding="utf-8")
    with pytest.raises(RuntimeError, match=match):
        RUNNER._require_preflight(
            path,
            preregistration_sha256="a" * 64,
            implementation_manifest_sha256="b" * 64,
        )


def test_only_frozen_modes_are_parseable() -> None:
    parser = RUNNER._parser()
    assert parser.parse_args(["--mode", "preflight"]).mode == "preflight"
    assert parser.parse_args(["--mode", "study"]).mode == "study"
    with pytest.raises(SystemExit):
        parser.parse_args(["--mode", "pilot"])

def _minimal_result() -> dict[str, object]:
    return {
        "mode": "confirmatory",
        "seeds": list(RUNNER.CONFIRMATORY_SEEDS),
        "provenance": {
            "preregistration_sha256": "a" * 64,
            "implementation_manifest_sha256": "b" * 64,
        },
        "calibration_validity": {"pass": True},
        "primary": {"pass": True},
        "secondary": {"pass": True},
        "diagnostics": {
            "d_exit": {"pass": True},
            "task": {
                "role": "diagnostic_only",
                "affects_registered_pass_fail": False,
            },
        },
        "complete_positive_interpretation_available": True,
    }


def test_execution_reservation_is_exclusive_canonical_and_bound(
    tmp_path: Path,
) -> None:
    lock = tmp_path / "cpp-confirmatory.lock"
    staging = tmp_path / "cpp-confirmatory-staging"
    kwargs = {
        "preregistration_sha256": "a" * 64,
        "implementation_manifest_sha256": "b" * 64,
        "preflight_sha256": "c" * 64,
    }

    RUNNER._create_execution_reservation(
        **kwargs,
        lock_path=lock,
        staging_dir=staging,
    )
    RUNNER._require_execution_reservation(**kwargs, lock_path=lock)
    assert staging.is_dir()
    assert not list(staging.iterdir())

    with pytest.raises(RuntimeError, match="staging directory exists"):
        RUNNER._create_execution_reservation(
            **kwargs,
            lock_path=lock,
            staging_dir=staging,
        )

    mutated = json.loads(lock.read_text(encoding="utf-8"))
    mutated["preflight_sha256"] = "d" * 64
    lock.write_text(json.dumps(mutated), encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not match"):
        RUNNER._require_execution_reservation(**kwargs, lock_path=lock)


def test_frozen_result_receipt_hashes_exact_result_bytes() -> None:
    result = _minimal_result()
    result_json = "{\"exact\":true}\n"
    markdown = "# exact\n"
    receipt = RUNNER._frozen_result_record(
        result=result,
        result_json_text=result_json,
        result_markdown_text=markdown,
        preregistration_sha256="a" * 64,
        implementation_manifest_sha256="b" * 64,
        preflight_sha256="c" * 64,
    )

    assert receipt["status"] == "confirmatory_complete_stopped_without_tuning"
    assert receipt["result_files"] == {
        "reports/cpp/results.json": RUNNER._sha256_text(result_json),
        "reports/cpp/results.md": RUNNER._sha256_text(markdown),
    }
    assert receipt["registered_outcomes"]["primary_pass"] is True


@pytest.mark.parametrize("failure_index", [0, 1, 2])
def test_staging_failure_never_reaches_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_index: int,
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    monkeypatch.setattr(
        RUNNER,
        "render_markdown",
        lambda result, evidence_href="results.json": "synthetic report\n",
    )
    calls = 0

    def failing_writer(path: Path, text: str) -> None:
        nonlocal calls
        if calls == failure_index:
            raise OSError(f"injected stage failure {failure_index}")
        calls += 1
        RUNNER._write_new_text(path, text)

    with pytest.raises(OSError, match="injected stage failure"):
        RUNNER._stage_result_bundle(
            _minimal_result(),
            preregistration_sha256="a" * 64,
            implementation_manifest_sha256="b" * 64,
            preflight_sha256="c" * 64,
            watchdog=None,
            staging_dir=staging,
            writer=failing_writer,
        )

    assert len(list(staging.iterdir())) == failure_index


@pytest.mark.parametrize("failure_index", [0, 1, 2])
def test_publication_failure_rolls_back_only_new_links(
    tmp_path: Path,
    failure_index: int,
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    names = ("results.json", "results.md", "frozen_results.json")
    for index, name in enumerate(names):
        (staging / name).write_text(f"stage-{index}\n", encoding="utf-8")
    destinations = tuple(
        (name, tmp_path / "published" / name) for name in names
    )
    calls = 0

    def failing_link(source: str, destination: str) -> None:
        nonlocal calls
        if calls == failure_index:
            raise OSError(f"injected publish failure {failure_index}")
        calls += 1
        os.link(source, destination)

    with pytest.raises(OSError, match="injected publish failure"):
        RUNNER._publish_staged_bundle(
            watchdog=None,
            staging_dir=staging,
            destinations=destinations,
            linker=failing_link,
        )

    assert all(not destination.exists() for _, destination in destinations)
    assert all((staging / name).is_file() for name in names)


def test_publication_recovers_matching_partial_bundle_without_reexecution(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    names = ("results.json", "results.md", "frozen_results.json")
    for index, name in enumerate(names):
        (staging / name).write_text(f"stage-{index}\n", encoding="utf-8")
    destinations = tuple(
        (name, tmp_path / "published" / name) for name in names
    )
    destinations[0][1].parent.mkdir(parents=True)
    os.link(staging / names[0], destinations[0][1])

    RUNNER._publish_staged_bundle(
        watchdog=None,
        staging_dir=staging,
        destinations=destinations,
    )

    for name, destination in destinations:
        assert destination.read_bytes() == (staging / name).read_bytes()
