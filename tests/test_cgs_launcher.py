from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_cgs.py"


def test_direct_cgs_launcher_imports_project_without_mutating_results() -> None:
    result_path = ROOT / "reports" / "cgs" / "pilot_results.json"
    before = result_path.read_bytes() if result_path.exists() else None
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--mode" in result.stdout
    after = result_path.read_bytes() if result_path.exists() else None
    assert after == before