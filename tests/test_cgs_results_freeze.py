from __future__ import annotations

import hashlib
import json
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
