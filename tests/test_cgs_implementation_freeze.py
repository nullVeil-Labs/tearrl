from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tearrl" / "cgs" / "frozen_implementation.json"
AMENDMENTS = (
    ROOT / "tearrl" / "cgs" / "frozen_implementation_amendment_001.json",
    ROOT / "tearrl" / "cgs" / "frozen_implementation_amendment_002.json",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_cgs_implementation_and_protocol_files_match_freeze_chain() -> None:
    frozen = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert frozen["status"] == "frozen_before_pilot_and_confirmatory_outcomes"
    assert frozen["parent_commit"] == "36a572d"
    assert _sha256(ROOT / frozen["document"]) == frozen["document_sha256"]
    assert frozen["preregistration_sha256"] == (
        "9d1e707ff871b838e8453016042a1ce73ad43d79432bf5b3a26e15099e775d1f"
    )
    expected = dict(frozen["implementation_files"])
    previous_hash = _sha256(MANIFEST)
    for index, amendment_path in enumerate(AMENDMENTS, start=1):
        amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
        if index == 1:
            assert amendment["base_manifest_sha256"] == previous_hash
            assert amendment["scientific_changes"] == []
        else:
            assert amendment["previous_amendment_sha256"] == previous_hash
            assert amendment["primary_gates_changed"] is False
            assert amendment["model_or_data_changes"] == []
        assert _sha256(ROOT / amendment["document"]) == amendment["document_sha256"]
        for relative, change in amendment["changed_files"].items():
            assert expected[relative] == change["old_sha256"], relative
            expected[relative] = change["new_sha256"]
        for relative, digest in amendment["added_files"].items():
            assert relative not in expected, relative
            expected[relative] = digest
        for relative, digest in amendment.get("preserved_pilot_files", {}).items():
            assert _sha256(ROOT / relative) == digest, relative
        previous_hash = _sha256(amendment_path)
    for relative, digest in expected.items():
        assert _sha256(ROOT / relative) == digest, relative