from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tearrl" / "cgs" / "frozen_implementation.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_cgs_implementation_and_protocol_files_match_pre_pilot_freeze() -> None:
    frozen = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert frozen["status"] == "frozen_before_pilot_and_confirmatory_outcomes"
    assert frozen["parent_commit"] == "36a572d"
    document = ROOT / frozen["document"]
    assert _sha256(document) == frozen["document_sha256"]
    assert frozen["preregistration_sha256"] == (
        "9d1e707ff871b838e8453016042a1ce73ad43d79432bf5b3a26e15099e775d1f"
    )
    for relative, expected in frozen["implementation_files"].items():
        path = ROOT / relative
        assert path.is_file(), relative
        assert _sha256(path) == expected, relative