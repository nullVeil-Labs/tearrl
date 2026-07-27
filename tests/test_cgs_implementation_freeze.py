from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tearrl" / "cgs" / "frozen_implementation.json"
AMENDMENT = ROOT / "tearrl" / "cgs" / "frozen_implementation_amendment_001.json"


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
    amendment = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    assert amendment["status"] == "frozen_before_successful_pilot_training_or_outcomes"
    assert amendment["scientific_changes"] == []
    assert _sha256(MANIFEST) == amendment["base_manifest_sha256"]
    assert _sha256(ROOT / amendment["document"]) == amendment["document_sha256"]
    overrides = amendment["changed_files"]
    for relative, expected in frozen["implementation_files"].items():
        path = ROOT / relative
        assert path.is_file(), relative
        if relative in overrides:
            assert expected == overrides[relative]["old_sha256"], relative
            expected = overrides[relative]["new_sha256"]
        assert _sha256(path) == expected, relative
    for relative, expected in amendment["added_files"].items():
        assert _sha256(ROOT / relative) == expected, relative