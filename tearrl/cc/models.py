"""Learned consequence witness and action-ID-free TEAR-CC actor."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn

from .arena import (
    ACTION_DESCRIPTORS, CCAction, CC_CONSEQUENCE_DIM, CC_DESCRIPTOR_DIM,
    CC_STATE_DIM, N_CC_ACTIONS, ArenaState, consequence_matrix,
    enumerate_contexts, exact_consequence,
)

WITNESS_HIDDEN = 64
ACTOR_HIDDEN = 64
LEDGER_DIM = 4


class ConsequenceWitness(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(CC_STATE_DIM + CC_DESCRIPTOR_DIM, WITNESS_HIDDEN),
            nn.Tanh(),
            nn.Linear(WITNESS_HIDDEN, WITNESS_HIDDEN),
            nn.Tanh(),
            nn.Linear(WITNESS_HIDDEN, CC_CONSEQUENCE_DIM),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)

    @torch.no_grad()
    def matrix(self, state: ArenaState) -> np.ndarray:
        features = torch.tensor(state.features(), dtype=torch.float32)
        rows = features.unsqueeze(0).expand(N_CC_ACTIONS, -1)
        descriptors = torch.tensor(ACTION_DESCRIPTORS)
        return self(torch.cat([rows, descriptors], dim=1)).cpu().numpy()


class ConsequenceActor(nn.Module):
    """Shared scorer: consequence row + ledger, with no action-ID embedding."""

    def __init__(self) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(CC_CONSEQUENCE_DIM + LEDGER_DIM, ACTOR_HIDDEN),
            nn.Tanh(),
            nn.Linear(ACTOR_HIDDEN, ACTOR_HIDDEN),
            nn.Tanh(),
            nn.Linear(ACTOR_HIDDEN, 1),
        )

    def forward(self, witness: torch.Tensor, ledger: torch.Tensor) -> torch.Tensor:
        if witness.ndim != 3 or witness.shape[1:] != (
                N_CC_ACTIONS, CC_CONSEQUENCE_DIM):
            raise ValueError("witness must be [batch, action, consequence]")
        if ledger.ndim != 2 or ledger.shape != (witness.shape[0], LEDGER_DIM):
            raise ValueError("ledger must be [batch, 4]")
        expanded = ledger.unsqueeze(1).expand(-1, N_CC_ACTIONS, -1)
        return self.scorer(torch.cat([witness, expanded], dim=-1)).squeeze(-1)

    @torch.no_grad()
    def probabilities(self, matrix: np.ndarray,
                      ledger: np.ndarray) -> np.ndarray:
        witness = torch.tensor(matrix[None], dtype=torch.float32)
        ledger_t = torch.tensor(ledger[None], dtype=torch.float32)
        return torch.softmax(self(witness, ledger_t)[0], dim=-1).cpu().numpy()


@dataclass(frozen=True, slots=True)
class WitnessReport:
    train_states: int
    heldout_states: int
    initial_loss: float
    final_loss: float
    heldout_mae: float


def _state_key(state: ArenaState, seed: int) -> str:
    payload = json.dumps({
        "seed": seed,
        "features": [float(value) for value in state.features()],
    }, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def split_contexts(seed: int = 7301) -> tuple[tuple[ArenaState, ...],
                                                tuple[ArenaState, ...]]:
    ordered = sorted(enumerate_contexts(), key=lambda state: _state_key(state, seed))
    cut = int(0.8 * len(ordered))
    return tuple(ordered[:cut]), tuple(ordered[cut:])


def _witness_dataset(states: Sequence[ArenaState]):
    inputs, targets = [], []
    for state in states:
        for action in CCAction:
            inputs.append(np.concatenate([
                state.features(), ACTION_DESCRIPTORS[int(action)]
            ]))
            targets.append(exact_consequence(state, action))
    return (
        torch.tensor(np.stack(inputs), dtype=torch.float32),
        torch.tensor(np.stack(targets), dtype=torch.float32),
    )


def train_witness(seed: int, epochs: int = 200,
                  learning_rate: float = 3e-3) -> tuple[ConsequenceWitness,
                                                        WitnessReport]:
    train_states, heldout_states = split_contexts()
    torch.manual_seed(seed)
    model = ConsequenceWitness()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    inputs, targets = _witness_dataset(train_states)
    heldout_x, heldout_y = _witness_dataset(heldout_states)
    with torch.no_grad():
        initial = float(torch.mean((model(inputs) - targets) ** 2))
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((model(inputs) - targets) ** 2)
        loss.backward()
        optimizer.step()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        final = float(torch.mean((model(inputs) - targets) ** 2))
        heldout_mae = float(torch.mean(torch.abs(model(heldout_x) - heldout_y)))
    return model, WitnessReport(
        len(train_states), len(heldout_states), initial, final, heldout_mae)


def ledger_features(state: ArenaState, corrections: float = 0.0,
                    invitations: float = 0.0,
                    disagreement: float = 0.0) -> np.ndarray:
    return np.asarray([
        state.debt,
        corrections,
        invitations,
        disagreement,
    ], dtype=np.float32)


def batch_inputs(states: Sequence[ArenaState], witness: ConsequenceWitness):
    matrices = np.stack([witness.matrix(state) for state in states])
    ledgers = np.stack([ledger_features(state) for state in states])
    return (
        torch.tensor(matrices, dtype=torch.float32),
        torch.tensor(ledgers, dtype=torch.float32),
    )


def task_pretrain(actor: ConsequenceActor, witness: ConsequenceWitness,
                  states: Sequence[ArenaState], epochs: int = 150,
                  learning_rate: float = 3e-3) -> list[float]:
    matrices, ledgers = batch_inputs(states, witness)
    labels = torch.tensor([
        int(np.argmax(consequence_matrix(state)[:, 0])) for state in states
    ], dtype=torch.long)
    optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate)
    losses = []
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(actor(matrices, ledgers), labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses


@dataclass(frozen=True, slots=True)
class PositiveExample:
    state: ArenaState
    matrix: np.ndarray
    target: CCAction


def train_positive_labels(actor: ConsequenceActor,
                          examples: Sequence[PositiveExample],
                          epochs: int,
                          learning_rate: float = 3e-3) -> list[float]:
    if not examples:
        return []
    matrices = torch.tensor(
        np.stack([example.matrix for example in examples]), dtype=torch.float32)
    ledgers = torch.tensor(np.stack([
        ledger_features(example.state) for example in examples
    ]), dtype=torch.float32)
    labels = torch.tensor(
        [int(example.target) for example in examples], dtype=torch.long)
    optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate)
    losses = []
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(actor(matrices, ledgers), labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses
