from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "compare_cgs_reproduction.py"
RESULTS = ROOT / "reports" / "cgs" / "results.json"


def _module():
    spec = importlib.util.spec_from_file_location("compare_cgs_reproduction", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_candidate(tmp_path: Path, candidate: dict) -> Path:
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(candidate), encoding="utf-8")
    return path


def test_metric_drift_can_preserve_registered_verdict_projection(tmp_path: Path) -> None:
    module = _module()
    original = json.loads(RESULTS.read_text(encoding="utf-8"))
    candidate = deepcopy(original)
    candidate["aggregate"]["stage1_learned"]["ece"] += 0.01

    comparison = module.compare_results(RESULTS, _write_candidate(tmp_path, candidate))
    assert comparison["schema_version"] == 2
    assert comparison["structure_equal"] is True
    assert comparison["interpretation"] == {
        "G7_in_frozen_projection_is_not_a_runtime_measurement": True,
        "non_punitive_audit_fields_in_projection_are_frozen_literals_not_runtime_counters": True,
        "verdict_equality_does_not_imply_byte_equality": True,
    }
    assert (
        "byte_equality_observed_only_under_the_recorded_numerical_profile"
        not in comparison["interpretation"]
    )
    assert comparison["byte_equal"] is False
    assert comparison["registered_verdict_projection_equal"] is True
    assert comparison["numeric_fields_differ"] == 1
    assert comparison["floating_fields_differ"] == 1
    assert comparison["floating_delta_min"] == pytest.approx(0.01)
    assert comparison["floating_delta_max"] == pytest.approx(0.01)
    assert comparison["integer_fields_differ"] == 0
    assert comparison["integer_delta_min"] == 0
    assert comparison["integer_delta_max"] == 0


def test_gate_drift_fails_verdict_projection(tmp_path: Path) -> None:
    module = _module()
    original = json.loads(RESULTS.read_text(encoding="utf-8"))
    candidate = deepcopy(original)
    candidate["aggregate"]["stage1_gates"]["G3_calibration"] = True

    comparison = module.compare_results(RESULTS, _write_candidate(tmp_path, candidate))
    assert comparison["registered_verdict_projection_equal"] is False


def test_per_seed_non_punitive_counter_drift_fails_projection(
    tmp_path: Path,
) -> None:
    module = _module()
    original = json.loads(RESULTS.read_text(encoding="utf-8"))
    candidate = deepcopy(original)
    candidate["seed_results"][0]["non_punitive_audit"][
        "punitive_terminations"
    ] = 1

    comparison = module.compare_results(RESULTS, _write_candidate(tmp_path, candidate))
    assert comparison["registered_verdict_projection_equal"] is False


@pytest.mark.parametrize(
    ("section", "gate_group", "gate"),
    (
        ("stage1", "gates", "G3_calibration"),
        (
            "stage2",
            "geometry_gates",
            "worst_party_reduction_at_least_half",
        ),
        (
            "stage2",
            "oracle_hierarchical_gates",
            "task_retention",
        ),
    ),
)
def test_per_seed_gate_drift_fails_projection(
    tmp_path: Path,
    section: str,
    gate_group: str,
    gate: str,
) -> None:
    module = _module()
    original = json.loads(RESULTS.read_text(encoding="utf-8"))
    candidate = deepcopy(original)
    gates = candidate["seed_results"][0][section][gate_group]
    gates[gate] = not gates[gate]

    comparison = module.compare_results(RESULTS, _write_candidate(tmp_path, candidate))
    assert comparison["registered_verdict_projection_equal"] is False


def test_float_and_integer_deltas_are_reported_separately(tmp_path: Path) -> None:
    module = _module()
    original = json.loads(RESULTS.read_text(encoding="utf-8"))
    candidate = deepcopy(original)
    candidate["aggregate"]["stage1_learned"]["ece"] += 1.475
    candidate["aggregate"]["stage1_learned"]["evaluated_rows"] += 341

    comparison = module.compare_results(RESULTS, _write_candidate(tmp_path, candidate))
    assert comparison["numeric_fields_differ"] == 2
    assert comparison["floating_fields_differ"] == 1
    assert comparison["floating_delta_min"] == pytest.approx(1.475)
    assert comparison["floating_delta_max"] == pytest.approx(1.475)
    assert comparison["integer_fields_differ"] == 1
    assert comparison["integer_delta_min"] == 341
    assert comparison["integer_delta_max"] == 341


def test_identical_artifact_is_byte_and_verdict_equal() -> None:
    module = _module()
    comparison = module.compare_results(RESULTS, RESULTS)
    assert comparison["byte_equal"] is True
    assert comparison["registered_verdict_projection_equal"] is True
    assert comparison["numeric_fields_differ"] == 0
    assert comparison["floating_fields_differ"] == 0
    assert comparison["integer_fields_differ"] == 0

def test_output_label_is_relative_or_basename_only(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path / "repo"
    root.mkdir()
    inside = root / "reports" / "evidence.json"
    outside = tmp_path / "private" / "evidence.json"

    assert module._relative_or_name(inside, root) == "reports/evidence.json"
    assert module._relative_or_name(outside, root) == "evidence.json"
    assert str(tmp_path) not in module._relative_or_name(outside, root)

def test_object_key_drift_fails_closed(tmp_path: Path) -> None:
    module = _module()
    original = json.loads(RESULTS.read_text(encoding="utf-8"))
    candidate = deepcopy(original)
    candidate["aggregate"]["stage1_learned"]["unexpected_metric"] = 1

    with pytest.raises(module.ComparisonStructureError, match="object keys differ"):
        module.compare_results(RESULTS, _write_candidate(tmp_path, candidate))


def test_list_length_and_container_drift_fail_closed() -> None:
    module = _module()
    with pytest.raises(module.ComparisonStructureError, match="list length differs"):
        module._numeric_deltas([1], [1, 2], path="$.rows")
    with pytest.raises(module.ComparisonStructureError, match="container type differs"):
        module._numeric_deltas({"row": 1}, [1], path="$.rows")


def test_comparison_output_exclusive_create_preserves_existing_file(
    tmp_path: Path,
) -> None:
    module = _module()
    output = tmp_path / "comparison.json"
    output.write_text("preserve\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        module._write_new_text(output, "replace\n")
    assert output.read_text(encoding="utf-8") == "preserve\n"

def test_main_refuses_existing_comparison_before_computing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _module()
    output = tmp_path / "comparison.json"
    output.write_text("preserve\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--candidate",
            str(RESULTS),
            "--output",
            str(output),
        ],
    )
    monkeypatch.setattr(
        module,
        "compare_results",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("comparison must not run")
        ),
    )

    with pytest.raises(SystemExit) as raised:
        module.main()
    assert raised.value.code == 2
    assert output.read_text(encoding="utf-8") == "preserve\n"
