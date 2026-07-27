from __future__ import annotations

import numpy as np

from tearrl.cc.experiment_v04 import run_v04_seed


def test_short_coherent_run_reduces_heldout_path_distance() -> None:
    result = run_v04_seed(
        10103, witness_epochs=10, task_epochs=10, coherent_epochs=20)
    assert result.final_path_tv < result.initial_path_tv
    assert set(result.conditions) == {
        "TaskOnly", "ShieldOnly", "CoherentPath"
    }
    coherent = result.conditions["CoherentPath"]
    assert coherent.governor_on_violation_rate == 0.0
    assert np.isfinite(coherent.path_tv)
