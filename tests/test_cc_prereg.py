"""The TEAR-CC v0.1 hypotheses stay frozen after result generation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_cc_preregistration_digest_and_claim_boundary_are_frozen() -> None:
    record = json.loads(
        (ROOT / "tearrl" / "cc" / "frozen_prereg.json").read_text(encoding="utf-8")
    )
    document = ROOT / record["document"]
    assert hashlib.sha256(document.read_bytes()).hexdigest() == record["sha256"]
    text = document.read_text(encoding="utf-8")
    for required in (
        "does **not** solve AI alignment",
        "Non-punitive curriculum definition",
        "Externalization Advantage",
        "Correction Capture",
        "H7 correction integrity",
        "Failure is a result",
    ):
        assert required in text
