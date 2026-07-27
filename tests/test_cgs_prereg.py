"""The TEAR-CG/S hypotheses remain byte-frozen before implementation outcomes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_cgs_preregistration_digest_and_claim_boundary_are_frozen() -> None:
    record = json.loads(
        (ROOT / "tearrl" / "cgs" / "frozen_prereg.json").read_text(
            encoding="utf-8"
        )
    )
    document = ROOT / record["document"]
    assert hashlib.sha256(document.read_bytes()).hexdigest() == record["sha256"]
    assert record["status"] == "frozen_before_implementation_and_outcomes"
    assert record["parent_commit"] == "6f70630"
    text = " ".join(document.read_text(encoding="utf-8").split())
    for required in (
        "not TEAR-CC v0.5",
        "does not test or solve general AI alignment",
        "never receives or predicts an admissible/good/bad label",
        "Average MAE is secondary",
        "perfect-oracle comparison",
        "identical fixed-size report",
        "four no-retraining cross-swaps",
        "calibrated counterfactual consequence reporting",
        "cannot establish that a model is honest",
        "After unblinding, no failed gate may be tuned",
    ):
        assert required in text