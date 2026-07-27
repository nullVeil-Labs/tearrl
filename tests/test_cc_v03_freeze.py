from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_cc_v03_preregistration_is_frozen() -> None:
    record = json.loads((
        ROOT / "tearrl" / "cc" / "frozen_v03_prereg.json"
    ).read_text(encoding="utf-8"))
    document = ROOT / record["document"]
    assert hashlib.sha256(document.read_bytes()).hexdigest() == record["sha256"]
    text = document.read_text(encoding="utf-8")
    for phrase in (
        "the safe set needs internal useful",
        "q(REPAIR)=1",
        "q(HELP)=0.5+0.4*need",
        "project only the path gradient",
        "Failure requires v0.4",
    ):
        assert phrase in text
