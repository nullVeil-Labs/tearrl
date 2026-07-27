from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_cc_v04_preregistration_is_frozen() -> None:
    record = json.loads((
        ROOT / "tearrl" / "cc" / "frozen_v04_prereg.json"
    ).read_text(encoding="utf-8"))
    document = ROOT / record["document"]
    assert hashlib.sha256(document.read_bytes()).hexdigest() == record["sha256"]
    text = document.read_text(encoding="utf-8")
    for phrase in (
        "one coherent positive geometry",
        "J_coherent",
        "Path TV",
        "Human authors supplied",
        "ends the v0.x implementation cycle",
    ):
        assert phrase in text
