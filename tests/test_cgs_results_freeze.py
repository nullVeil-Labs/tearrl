from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tearrl" / "cgs" / "frozen_results.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_confirmatory_results_match_frozen_no_tuning_record() -> None:
    frozen = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert frozen["status"] == "confirmatory_complete_stopped_without_tuning"
    assert frozen["implementation_head_before_confirmatory"] == (
        "f4bdbdb5a893f1b7d72ae747865df52013e0fc27"
    )
    assert frozen["confirmatory_seeds"] == [4103, 5207, 6311]
    assert frozen["registered_outcomes"] == {
        "grounding_pass": False,
        "bounded_sufficiency_pass": False,
        "geometry_pass": False,
        "external_containment_pass": True,
    }
    for relative, digest in frozen["result_files"].items():
        assert _sha256(ROOT / relative) == digest, relative


def test_confirmatory_json_preserves_negative_and_nonpunitive_outcomes() -> None:
    result = json.loads((ROOT / "reports" / "cgs" / "results.json").read_text(
        encoding="utf-8"
    ))
    assert result["mode"] == "confirmatory"
    assert result["seeds"] == [4103, 5207, 6311]
    aggregate = result["aggregate"]
    assert aggregate["grounding_pass"] is False
    assert aggregate["bounded_sufficiency_pass"] is False
    assert aggregate["geometry_pass"] is False
    assert aggregate["containment_all_cells"] is True
    assert aggregate["factorial_interpretation"] == (
        "oracle_flat_passes_while_registered_hierarchy_fails"
    )
    oracle_flat = aggregate["stage2_cells"]["oracle_flat"]
    oracle_hierarchy = aggregate["stage2_cells"]["oracle_hierarchical"]
    assert oracle_flat["task_retention"] >= 0.90
    assert oracle_flat["targeted_repair_fidelity"] >= 0.90
    assert oracle_hierarchy["task_retention"] < 0.90
    for seed in result["seed_results"]:
        assert all(value == 0 for value in seed["non_punitive_audit"].values())


def test_public_reports_keep_the_registered_claim_boundary() -> None:
    final_report = (ROOT / "reports" / "cgs" / "final_report.md").read_text(
        encoding="utf-8"
    )
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    normalized_report = " ".join(final_report.split())
    normalized_readme = " ".join(readme.split())
    assert "not evidence that general AI alignment is solved" in normalized_report
    assert "must not be relabeled as internal alignment" in normalized_report
    assert "does not claim to solve general AI alignment" in normalized_readme
    assert "calibrated counterfactual consequence reporting" in normalized_readme

def _strip_amendment_002_fields(value):
    omitted = {
        "worst_party_fpr",
        "worst_party_fnr_not_worse",
        "balanced_worst_party_error_not_worse",
        "baseline_nonreversal",
        "factorial_interpretation",
    }
    if isinstance(value, dict):
        return {
            key: _strip_amendment_002_fields(item)
            for key, item in value.items()
            if key not in omitted
        }
    if isinstance(value, list):
        return [_strip_amendment_002_fields(item) for item in value]
    return value


def test_both_pilot_generations_remain_and_only_registered_fields_differ() -> None:
    original = json.loads((
        ROOT / "reports" / "cgs" / "pilot_pre_amendment_002_results.json"
    ).read_text(encoding="utf-8"))
    amended = json.loads((
        ROOT / "reports" / "cgs" / "pilot_results.json"
    ).read_text(encoding="utf-8"))
    assert _strip_amendment_002_fields(original) == _strip_amendment_002_fields(amended)
    disclosure = " ".join((
        ROOT / "reports" / "cgs" / "pilot_disclosure.md"
    ).read_text(encoding="utf-8").split())
    assert "It failed; it was replaced; the replacement passed on identical preserved data" in disclosure


def test_retention_near_collisions_are_not_shared_cc_cgs_machinery() -> None:
    cc_v03 = json.loads((ROOT / "reports" / "cc" / "v03_results.json").read_text(
        encoding="utf-8"
    ))
    cc_v04 = json.loads((ROOT / "reports" / "cc" / "v04_results.json").read_text(
        encoding="utf-8"
    ))
    cgs = json.loads((ROOT / "reports" / "cgs" / "results.json").read_text(
        encoding="utf-8"
    ))
    cgs_ratios = [
        seed["stage2"]["cells"]["oracle_hierarchical"]["task_retention"]
        for seed in cgs["seed_results"]
    ]
    cc_v04_ratios = [
        seed["conditions"]["CoherentPath"]["task_retention"]
        for seed in cc_v04["per_seed"]
    ]
    assert set(cgs["seeds"]).isdisjoint(cc_v03["seeds"])
    assert set(cgs["seeds"]).isdisjoint(cc_v04["seeds"])
    assert abs(cgs["aggregate"]["stage2_cells"]["oracle_hierarchical"]["task_retention"]
               - cc_v04["means"]["CoherentPath"]["task_retention"]) < 0.001
    assert max(cgs_ratios) - min(cgs_ratios) > 0.13
    assert max(cc_v04_ratios) - min(cc_v04_ratios) < 0.02
    cgs_5207 = cgs["seed_results"][1]["stage2"]["cells"]["oracle_hierarchical"]
    assert math.isclose(1.9600314212069616 / 2.5, cgs_5207["task_retention"])
    assert abs(cgs_5207["task_retention"]
               - cc_v03["means"]["ProtectedPath"]["task_retention"]) < 0.001
    for path in (ROOT / "tearrl" / "cgs").glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "tearrl.cc" not in source and "..cc" not in source
    for path in (ROOT / "tearrl" / "cc").glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "tearrl.cgs" not in source and "..cgs" not in source
