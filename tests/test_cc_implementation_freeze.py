from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_cc_implementation_supplement_is_frozen() -> None:
    record = json.loads((
        ROOT / "tearrl" / "cc" / "frozen_implementation.json"
    ).read_text(encoding="utf-8"))
    document = ROOT / record["document"]
    assert hashlib.sha256(document.read_bytes()).hexdigest() == record["sha256"]
    text = document.read_text(encoding="utf-8")
    assert "positive-label cross-entropy only" in text
    assert "100 revision epochs followed by 100 invitation epochs" in text
    assert "raw policy" in text and "distributions with no hidden masking" in text
