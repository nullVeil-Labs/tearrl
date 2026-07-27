from __future__ import annotations

import numpy as np
import pytest

from tearrl.cc.arena import CCAction, Consent, ArenaState
from tearrl.cc.experiment_v03 import _path_tensors, run_v03_seed
from tearrl.cc.models import train_witness


def test_path_targets_never_escape_admissible_set() -> None:
    witness, _ = train_witness(7789, epochs=5)
    states = (
        ArenaState(0.9, 0.9, Consent.GRANTED, 0.0),
        ArenaState(0.9, 0.9, Consent.GRANTED, 0.5),
        ArenaState(0.9, 0.9, Consent.DENIED, 0.0),
    )
    _, _, allowed, targets = _path_tensors(states, witness)
    assert float(targets[~allowed].sum()) == 0.0
    assert torch_allclose(targets.sum(dim=1).numpy(), np.ones(3))
    assert targets[1, CCAction.REPAIR] == 1.0
    assert targets[2, CCAction.HELP] == 0.0
    assert targets[2, CCAction.PRODUCE] == 1.0


def torch_allclose(actual, expected) -> bool:
    return bool(np.allclose(actual, expected))


def test_debt_free_care_target_is_graded_inside_task_tied_pair() -> None:
    witness, _ = train_witness(7789, epochs=5)
    low = ArenaState(0.1, 0.9, Consent.GRANTED, 0.0)
    high = ArenaState(0.9, 0.9, Consent.GRANTED, 0.0)
    _, _, _, targets = _path_tensors((low, high), witness)
    assert targets[0, CCAction.HELP] == pytest.approx(0.54)
    assert targets[1, CCAction.HELP] == pytest.approx(0.86)


def test_short_v03_run_reports_useful_safe_conditions() -> None:
    result = run_v03_seed(
        7789, witness_epochs=10, task_epochs=10, path_epochs=10)
    assert set(result.conditions) == {
        "TaskOnly", "ShieldOnly", "JointPath", "ProtectedPath"
    }
    assert result.conditions["ProtectedPath"].governor_on_violation_rate == 0.0
    assert np.isfinite(result.conditions["ProtectedPath"].task_retention)
