"""Morally neutral consequence witness for TEAR-CG/S Stage 1.

This module predicts raw intervention outcomes and report uncertainty. It does
not import the actor, charter, governor, ledger, evaluator, or experiment code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .schema import (
    BINARY_OUTCOMES,
    CONTINUOUS_OUTCOMES,
    ConsequenceReport,
    N_OUTCOMES,
    N_REPAIR_KINDS,
    N_REPAIR_PARTIES,
    one_hot,
)
from .worlds import CausalExample, WITNESS_INPUT_DIM


@dataclass(frozen=True, slots=True)
class TensorReport:
    mean: Tensor
    scale: Tensor
    repair_party_logits: Tensor
    repair_kind_logits: Tensor
    identifiability_logit: Tensor


@dataclass(frozen=True, slots=True)
class WitnessTrainingReport:
    seed: int
    epochs: int
    rows: int
    trainable_parameters: int
    final_loss: float


class NeutralWitness(nn.Module):
    """A bounded MLP with exactly the registered consequence-report heads."""

    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.backbone = nn.Sequential(
            nn.Linear(WITNESS_INPUT_DIM, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
        )
        self.mean_head = nn.Linear(self.hidden_dim, N_OUTCOMES)
        self.scale_head = nn.Linear(self.hidden_dim, N_OUTCOMES)
        self.repair_party_head = nn.Linear(self.hidden_dim, N_REPAIR_PARTIES)
        self.repair_kind_head = nn.Linear(self.hidden_dim, N_REPAIR_KINDS)
        self.identifiability_head = nn.Linear(self.hidden_dim, 1)

    def forward(self, inputs: Tensor) -> TensorReport:
        hidden = self.backbone(inputs)
        raw_mean = self.mean_head(hidden)
        mean = torch.cat(
            [
                raw_mean[..., CONTINUOUS_OUTCOMES],
                torch.sigmoid(raw_mean[..., BINARY_OUTCOMES]),
            ],
            dim=-1,
        )
        scale = F.softplus(self.scale_head(hidden)) + 0.01
        return TensorReport(
            mean,
            scale,
            self.repair_party_head(hidden),
            self.repair_kind_head(hidden),
            self.identifiability_head(hidden).squeeze(-1),
        )

    @property
    def trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def _as_training_tensors(
    rows: Sequence[CausalExample],
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    if not rows:
        raise ValueError("witness training requires at least one row")
    inputs = torch.from_numpy(np.stack([row.witness_input() for row in rows]))
    outcomes = torch.from_numpy(np.stack([row.outcome for row in rows]))
    parties = torch.as_tensor([row.repair_party for row in rows], dtype=torch.long)
    kinds = torch.as_tensor([row.repair_kind for row in rows], dtype=torch.long)
    identifiable = torch.as_tensor(
        [float(row.identifiable) for row in rows], dtype=torch.float32
    )
    return inputs, outcomes, parties, kinds, identifiable


def witness_loss(
    report: TensorReport,
    outcomes: Tensor,
    repair_parties: Tensor,
    repair_kinds: Tensor,
    identifiable: Tensor,
) -> Tensor:
    """Prediction loss only; it contains no moral or actor-success target."""

    visible = identifiable > 0.5
    ident_loss = F.binary_cross_entropy_with_logits(
        report.identifiability_logit, identifiable
    )
    target_scale = torch.where(
        identifiable[:, None] > 0.5,
        torch.full_like(report.scale, 0.05),
        torch.full_like(report.scale, 0.75),
    )
    scale_loss = F.smooth_l1_loss(report.scale, target_scale)
    if not bool(torch.any(visible)):
        return ident_loss + 0.25 * scale_loss

    continuous_loss = F.smooth_l1_loss(
        report.mean[visible, CONTINUOUS_OUTCOMES],
        outcomes[visible, CONTINUOUS_OUTCOMES],
    )
    binary_targets = outcomes[visible, BINARY_OUTCOMES]
    binary_predictions = report.mean[visible, BINARY_OUTCOMES].clamp(1e-6, 1 - 1e-6)
    positive_rate = binary_targets.mean(dim=0).clamp(0.02, 0.98)
    positive_weight = ((1.0 - positive_rate) / positive_rate).clamp(1.0, 20.0)
    binary_loss = -(
        positive_weight * binary_targets * torch.log(binary_predictions)
        + (1.0 - binary_targets) * torch.log1p(-binary_predictions)
    ).mean()
    party_loss = F.cross_entropy(
        report.repair_party_logits[visible], repair_parties[visible]
    )
    kind_loss = F.cross_entropy(
        report.repair_kind_logits[visible], repair_kinds[visible]
    )
    return (
        continuous_loss
        + binary_loss
        + 0.35 * (party_loss + kind_loss)
        + 0.35 * ident_loss
        + 0.15 * scale_loss
    )


def train_witness(
    rows: Sequence[CausalExample],
    *,
    seed: int,
    epochs: int,
    learning_rate: float = 2e-3,
    batch_size: int = 256,
    hidden_dim: int = 128,
) -> tuple[NeutralWitness, WitnessTrainingReport]:
    """Train deterministically within the frozen row/epoch/parameter budgets."""

    if epochs <= 0:
        raise ValueError("epochs must be positive")
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = NeutralWitness(hidden_dim=hidden_dim)
    if model.trainable_parameters > 250_000:
        raise ValueError("witness exceeds the frozen 250,000-parameter budget")
    tensors = _as_training_tensors(rows)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator(device="cpu").manual_seed(seed ^ 0xC651)
    final_loss = float("nan")

    for _ in range(epochs):
        order = torch.randperm(len(rows), generator=generator)
        for start in range(0, len(rows), batch_size):
            indices = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            report = model(tensors[0][indices])
            loss = witness_loss(
                report,
                tensors[1][indices],
                tensors[2][indices],
                tensors[3][indices],
                tensors[4][indices],
            )
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach())

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, WitnessTrainingReport(
        seed,
        epochs,
        len(rows),
        model.trainable_parameters,
        final_loss,
    )


@torch.no_grad()
def predict_reports(
    model: NeutralWitness,
    rows: Sequence[CausalExample],
    *,
    batch_size: int = 512,
) -> tuple[ConsequenceReport, ...]:
    if not rows:
        return ()
    inputs = torch.from_numpy(np.stack([row.witness_input() for row in rows]))
    result: list[ConsequenceReport] = []
    for start in range(0, len(rows), batch_size):
        tensor_report = model(inputs[start : start + batch_size])
        means = tensor_report.mean.cpu().numpy()
        scales = tensor_report.scale.cpu().numpy()
        parties = torch.softmax(tensor_report.repair_party_logits, dim=-1).cpu().numpy()
        kinds = torch.softmax(tensor_report.repair_kind_logits, dim=-1).cpu().numpy()
        identifiability = torch.sigmoid(
            tensor_report.identifiability_logit
        ).cpu().numpy()
        for index in range(len(means)):
            result.append(
                ConsequenceReport(
                    means[index],
                    scales[index],
                    parties[index],
                    kinds[index],
                    float(identifiability[index]),
                )
            )
    return tuple(result)


def oracle_report(row: CausalExample) -> ConsequenceReport:
    """Exact source with the same report schema and observer identifiability."""

    return ConsequenceReport(
        np.asarray(row.outcome, dtype=np.float32).copy(),
        np.full(N_OUTCOMES, 0.01, dtype=np.float32),
        one_hot(row.repair_party, N_REPAIR_PARTIES),
        one_hot(row.repair_kind, N_REPAIR_KINDS),
        float(row.identifiable),
    )


def oracle_reports(rows: Sequence[CausalExample]) -> tuple[ConsequenceReport, ...]:
    return tuple(oracle_report(row) for row in rows)