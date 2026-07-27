from __future__ import annotations

import numpy as np
import pytest
import torch

from tearrl.agents.ppo import pcgrad
from tearrl.cc.arena import enumerate_contexts
from tearrl.cc.experiment_v02 import (
    _governance_tensors, graded_care_target, run_v02_seed,
)
from tearrl.cc.models import split_contexts, train_witness


def test_graded_care_target_has_registered_known_answers() -> None:
    assert graded_care_target(0.0) == pytest.approx(0.5)
    assert graded_care_target(0.5) == pytest.approx(0.7)
    assert graded_care_target(1.0) == pytest.approx(0.9)
    with pytest.raises(ValueError):
        graded_care_target(1.1)


def test_every_context_has_nonempty_independently_governed_safe_set() -> None:
    witness, _ = train_witness(4409, epochs=5)
    _, _, allowed, care_rows, _ = _governance_tensors(
        enumerate_contexts(), witness)
    assert allowed.any(dim=1).all()
    assert care_rows.numel() > 0


def test_asymmetric_projection_can_remove_only_conflicting_care_component() -> None:
    g_safe = torch.tensor([1.0, 0.0])
    g_care = torch.tensor([-1.0, 1.0])
    projected_care, diagnostics = pcgrad(g_care, g_safe)
    assert diagnostics.conflicted
    assert torch.dot(projected_care, g_safe) == pytest.approx(0.0, abs=1e-6)
    assert torch.equal(g_safe, torch.tensor([1.0, 0.0]))


def test_short_v02_run_reports_raw_and_protected_conditions() -> None:
    result = run_v02_seed(
        4409, witness_epochs=10, task_epochs=10, replay_epochs=10)
    assert set(result.conditions) == {
        "TaskOnly", "ShieldOnly", "JointReplay", "ProtectedReplay"
    }
    assert result.conditions["TaskOnly"].inadmissible_mass == pytest.approx(
        result.conditions["ShieldOnly"].inadmissible_mass)
    assert result.conditions["ProtectedReplay"].governor_on_violation_rate == 0.0
    assert np.isfinite(result.protected.conflict_fraction)
