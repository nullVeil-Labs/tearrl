from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from tearrl.cc.arena import CCAction
from tearrl.cc.experiment import (
    CONDITIONS, _curriculum_examples, evaluate_actor, run_seed,
    validate_examples,
)
from tearrl.cc.models import (
    ConsequenceActor, PositiveExample, split_contexts, task_pretrain,
    train_witness,
)


def test_shifted_positive_label_state_is_detected() -> None:
    witness, _ = train_witness(1103, epochs=10)
    torch.manual_seed(4)
    actor = ConsequenceActor()
    states = split_contexts()[0][:12]
    task_pretrain(actor, witness, states, epochs=10)
    revisions, _ = _curriculum_examples(actor, witness, states)
    assert len(revisions) >= 2
    shifted = list(revisions)
    shifted[0] = replace(shifted[0], state=shifted[1].state)
    with pytest.raises(ValueError, match="state/matrix mismatch"):
        validate_examples(shifted, witness)
    validate_examples(revisions, witness)


def test_short_seed_run_reports_all_conditions_and_raw_shield_identity() -> None:
    result = run_seed(
        1103, witness_epochs=10, task_epochs=10, curriculum_epochs=10)
    assert set(result.conditions) == set(CONDITIONS)
    task = result.conditions["TaskOnly"]
    shield = result.conditions["ShieldOnly"]
    assert task.inadmissible_mass == pytest.approx(shield.inadmissible_mass)
    assert shield.governor_on_violation_rate == 0.0
    assert all(np.isfinite(value.inadmissible_mass)
               for value in result.conditions.values())


def test_governor_off_metric_is_not_silently_masked_to_zero() -> None:
    witness, _ = train_witness(1103, epochs=5)
    torch.manual_seed(2)
    actor = ConsequenceActor()  # near-uniform raw policy
    states = split_contexts()[1][:8]
    metrics = evaluate_actor(actor, witness, states, actor, {})
    assert metrics.inadmissible_mass > 0.0
    assert metrics.governor_on_violation_rate == 0.0
    assert metrics.governor_dependence_gap > 0.0
