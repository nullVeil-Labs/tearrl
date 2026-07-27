from __future__ import annotations

import numpy as np
import pytest
import torch

from tearrl.cc.arena import CCAction, Consent, ArenaState, N_CC_ACTIONS
from tearrl.cc.models import (
    ConsequenceActor, PositiveExample, ledger_features, split_contexts,
    task_pretrain, train_positive_labels, train_witness,
)


def test_context_split_is_deterministic_disjoint_and_state_level() -> None:
    train_a, heldout_a = split_contexts(7301)
    train_b, heldout_b = split_contexts(7301)
    assert train_a == train_b and heldout_a == heldout_b
    assert len(train_a) == 192 and len(heldout_a) == 48
    assert set(train_a).isdisjoint(heldout_a)


def test_learned_witness_reduces_loss_and_freezes() -> None:
    witness, report = train_witness(seed=1103, epochs=30)
    assert report.final_loss < report.initial_loss
    assert np.isfinite(report.heldout_mae)
    assert all(not parameter.requires_grad for parameter in witness.parameters())
    assert witness.matrix(split_contexts()[1][0]).shape == (N_CC_ACTIONS, 9)


def test_actor_shared_scorer_is_permutation_equivariant() -> None:
    torch.manual_seed(2)
    actor = ConsequenceActor()
    matrix = torch.randn(3, N_CC_ACTIONS, 9)
    ledger = torch.randn(3, 4)
    permutation = torch.tensor([3, 0, 6, 2, 5, 1, 4])
    original = actor(matrix, ledger)
    permuted = actor(matrix[:, permutation], ledger)
    assert torch.allclose(permuted, original[:, permutation], atol=1e-7)


def test_task_pretraining_learns_the_registered_task_temptation() -> None:
    witness, _ = train_witness(seed=1103, epochs=40)
    torch.manual_seed(1103)
    actor = ConsequenceActor()
    states = split_contexts()[0][:48]
    losses = task_pretrain(actor, witness, states, epochs=60)
    assert losses[-1] < losses[0]
    proposals = [
        int(np.argmax(actor.probabilities(
            witness.matrix(state), ledger_features(state))))
        for state in states
    ]
    assert proposals.count(int(CCAction.TAMPER)) / len(proposals) >= 0.9


def test_positive_label_moves_probability_without_negative_reward_term() -> None:
    witness, _ = train_witness(seed=1103, epochs=20)
    torch.manual_seed(9)
    actor = ConsequenceActor()
    state = ArenaState(0.9, 0.9, Consent.GRANTED, 0.0)
    matrix = witness.matrix(state)
    before = actor.probabilities(matrix, ledger_features(state))[CCAction.HELP]
    losses = train_positive_labels(
        actor, [PositiveExample(state, matrix, CCAction.HELP)], epochs=40)
    after = actor.probabilities(matrix, ledger_features(state))[CCAction.HELP]
    assert losses[-1] < losses[0]
    assert after > before
