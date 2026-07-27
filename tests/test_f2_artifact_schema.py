"""The generated F2 artifact must distinguish edges, matches, and coverage."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_f2_census_artifact_has_current_nonstale_schema() -> None:
    payload = json.loads((ROOT / "reports" / "f2_census.json").read_text(
        encoding="utf-8"))
    assert payload["status"] == "non_registered_schema_diagnostic"
    assert payload["registered_60000_pool_regenerated"] is False
    assert payload["candidates"] > 0 and payload["tuples"] > 0
    assert len(payload["census"]) == 3 * 3 * 3
    for row in payload["census"]:
        assert "pairs" not in row
        assert {
            "c_norm_tol", "yhat_tol", "min_debt_gap", "candidate_edges",
            "matched_pairs", "tuples_used", "coverage",
        } == set(row)
        assert row["candidate_edges"] >= row["matched_pairs"] >= 0
        assert row["tuples_used"] == 2 * row["matched_pairs"]
        assert row["coverage"] == pytest.approx(
            row["tuples_used"] / payload["tuples"], abs=5e-5)
