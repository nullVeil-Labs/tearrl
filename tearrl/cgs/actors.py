"""Matched flat and hierarchical actors for TEAR-CG/S Stage 2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .schema import (
    ACTION_DESCRIPTORS,
    ACTION_DESCRIPTOR_DIM,
    CH_CORRECTION,
    CH_TASK,
    CGSAction,
    N_ACTIONS,
    REPORT_DIM,
    ReportEnvelope,
)
from .worlds import (
    CausalExample,
    STATE_DIM,
    WorldState,
    alignment_key,
    true_worst_party_event,
)

ACTOR_INPUT_DIM = REPORT_DIM + STATE_DIM + ACTION_DESCRIPTOR_DIM
ACTOR_HIDDEN = 64


@dataclass(frozen=True, slots=True)
class CandidateContext:
    world_id: str
    context_id: str
    variant: str
    rows: tuple[CausalExample, ...]

    @property
    def state(self) -> WorldState:
        return self.rows[0].state


@dataclass(frozen=True, slots=True)
class ActorBatch:
    features: np.ndarray
    flat_labels: np.ndarray
    task_labels: np.ndarray
    admissibility_labels: np.ndarray
    obligation_labels: np.ndarray
    task_values: np.ndarray


@dataclass(frozen=True, slots=True)
class ActorTrainingReport:
    geometry: str
    seed: int
    epochs: int
    contexts: int
    batch_size: int
    trainable_parameters: int
    final_loss: float


@dataclass(frozen=True, slots=True)
class HierarchicalScores:
    admissibility: Tensor
    obligation: Tensor
    task: Tensor


def _context_id(row: CausalExample) -> str:
    return row.pair_id.rsplit(":", 1)[0]


def group_contexts(rows: Sequence[CausalExample]) -> tuple[CandidateContext, ...]:
    grouped: dict[tuple[str, str, str], list[CausalExample]] = {}
    for row in rows:
        key = (row.world_id, _context_id(row), row.variant)
        grouped.setdefault(key, []).append(row)
    contexts: list[CandidateContext] = []
    for (world_id, context_id, variant), members in grouped.items():
        if len(members) != N_ACTIONS:
            continue
        ordered = tuple(sorted(members, key=lambda row: int(row.action)))
        if tuple(int(row.action) for row in ordered) != tuple(range(N_ACTIONS)):
            raise ValueError("candidate context does not contain every action exactly once")
        state_features = ordered[0].state.features()
        if not all(
            np.array_equal(row.state.features(), state_features) for row in ordered
        ):
            raise ValueError("candidate context mixes states")
        contexts.append(CandidateContext(world_id, context_id, variant, ordered))
    return tuple(contexts)


def flatten_contexts(
    contexts: Sequence[CandidateContext],
) -> tuple[CausalExample, ...]:
    return tuple(row for context in contexts for row in context.rows)


def registered_admissible(row: CausalExample) -> bool:
    """External human-authored charter decision on exact raw consequences."""

    if not row.identifiable and row.action not in {CGSAction.ASK, CGSAction.EXIT}:
        return False
    return bool(
        not true_worst_party_event(row.outcome)
        and float(row.outcome[CH_CORRECTION]) >= 0.0
    )


def required_repair_party(state: WorldState) -> int | None:
    debt = np.asarray(state.debt)
    if not np.any(debt > 0.0):
        return None
    return int(np.argmax(debt))


def matches_required_repair(row: CausalExample) -> bool:
    party = required_repair_party(row.state)
    if party is None:
        return False
    expected_kind = int(row.state.damage_kind[party])
    return bool(
        row.action in {CGSAction.REPAIR_0, CGSAction.REPAIR_1, CGSAction.REPAIR_2}
        and row.repair_party == party + 1
        and row.repair_kind == expected_kind
        and expected_kind > 0
    )


def obligation_satisfied(row: CausalExample) -> bool:
    if not registered_admissible(row):
        return False
    if not row.identifiable:
        return row.action in {CGSAction.ASK, CGSAction.EXIT}
    if required_repair_party(row.state) is not None:
        return matches_required_repair(row)
    return True


def _best_action(rows: Sequence[CausalExample]) -> int:
    if not rows:
        return int(CGSAction.EXIT)
    return int(
        max(rows, key=lambda row: (float(row.outcome[CH_TASK]), -int(row.action))).action
    )


def charter_target(context: CandidateContext) -> int:
    admissible = [row for row in context.rows if registered_admissible(row)]
    obligated = [row for row in admissible if obligation_satisfied(row)]
    return _best_action(obligated or admissible)


def task_target(context: CandidateContext) -> int:
    return _best_action(context.rows)


def prepare_actor_batch(
    contexts: Sequence[CandidateContext],
    envelopes: Sequence[ReportEnvelope],
) -> ActorBatch:
    rows = flatten_contexts(contexts)
    if len(rows) != len(envelopes):
        raise ValueError("report count does not match candidate rows")
    feature_rows: list[np.ndarray] = []
    for row, envelope in zip(rows, envelopes, strict=True):
        if envelope.alignment_key != alignment_key(row):
            raise ValueError("shifted or misaligned consequence report")
        feature = np.concatenate(
            [
                envelope.report.vector(),
                row.state.features(),
                ACTION_DESCRIPTORS[int(row.action)],
            ]
        ).astype(np.float32)
        if feature.shape != (ACTOR_INPUT_DIM,):
            raise AssertionError("actor feature schema drift")
        feature_rows.append(feature)
    features = np.stack(feature_rows).reshape(len(contexts), N_ACTIONS, -1)
    flat_labels = np.asarray([charter_target(context) for context in contexts])
    task_labels = np.asarray([task_target(context) for context in contexts])
    admissibility = np.asarray(
        [
            [float(registered_admissible(row)) for row in context.rows]
            for context in contexts
        ],
        dtype=np.float32,
    )
    obligations = np.asarray(
        [
            [float(obligation_satisfied(row)) for row in context.rows]
            for context in contexts
        ],
        dtype=np.float32,
    )
    task_values = np.asarray(
        [
            [float(row.outcome[CH_TASK]) / 2.5 for row in context.rows]
            for context in contexts
        ],
        dtype=np.float32,
    )
    return ActorBatch(
        features,
        flat_labels.astype(np.int64),
        task_labels.astype(np.int64),
        admissibility,
        obligations,
        task_values,
    )


class FlatActor(nn.Module):
    """A shared scorer that directly maps each candidate row to one score."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(ACTOR_INPUT_DIM, ACTOR_HIDDEN),
            nn.Tanh(),
            nn.Linear(ACTOR_HIDDEN, ACTOR_HIDDEN),
            nn.Tanh(),
        )
        self.score_head = nn.Linear(ACTOR_HIDDEN, 1)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 3 or features.shape[1:] != (
            N_ACTIONS,
            ACTOR_INPUT_DIM,
        ):
            raise ValueError("actor features must be [batch, action, feature]")
        return self.score_head(self.backbone(features)).squeeze(-1)

    @property
    def trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class HierarchicalActor(nn.Module):
    """Separate heads applied in admissibility/obligation/task order."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(ACTOR_INPUT_DIM, ACTOR_HIDDEN),
            nn.Tanh(),
            nn.Linear(ACTOR_HIDDEN, ACTOR_HIDDEN),
            nn.Tanh(),
        )
        self.admissibility_head = nn.Linear(ACTOR_HIDDEN, 1)
        self.obligation_head = nn.Linear(ACTOR_HIDDEN, 1)
        self.task_head = nn.Linear(ACTOR_HIDDEN, 1)

    def forward(self, features: Tensor) -> HierarchicalScores:
        if features.ndim != 3 or features.shape[1:] != (
            N_ACTIONS,
            ACTOR_INPUT_DIM,
        ):
            raise ValueError("actor features must be [batch, action, feature]")
        hidden = self.backbone(features)
        return HierarchicalScores(
            self.admissibility_head(hidden).squeeze(-1),
            self.obligation_head(hidden).squeeze(-1),
            self.task_head(hidden).squeeze(-1),
        )

    @property
    def trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def _freeze(model: nn.Module) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def _train_flat(
    batch: ActorBatch,
    *,
    seed: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    task_only: bool,
) -> tuple[FlatActor, ActorTrainingReport]:
    if epochs <= 0 or len(batch.features) == 0:
        raise ValueError("actor training requires positive epochs and contexts")
    torch.manual_seed(seed)
    model = FlatActor()
    if model.trainable_parameters > 100_000:
        raise ValueError("actor exceeds the frozen 100,000-parameter budget")
    features = torch.from_numpy(batch.features)
    labels = torch.from_numpy(batch.task_labels if task_only else batch.flat_labels)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator(device="cpu").manual_seed(seed ^ 0xAC70)
    final_loss = float("nan")
    for _ in range(epochs):
        order = torch.randperm(len(features), generator=generator)
        for start in range(0, len(features), batch_size):
            indices = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(features[indices]), labels[indices])
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach())
    _freeze(model)
    geometry = "task_only" if task_only else "flat"
    return model, ActorTrainingReport(
        geometry,
        seed,
        epochs,
        len(features),
        batch_size,
        model.trainable_parameters,
        final_loss,
    )


def train_flat_actor(
    batch: ActorBatch,
    *,
    seed: int,
    epochs: int,
    learning_rate: float = 2e-3,
    batch_size: int = 256,
) -> tuple[FlatActor, ActorTrainingReport]:
    return _train_flat(
        batch,
        seed=seed,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        task_only=False,
    )


def train_task_actor(
    batch: ActorBatch,
    *,
    seed: int,
    epochs: int,
    learning_rate: float = 2e-3,
    batch_size: int = 256,
) -> tuple[FlatActor, ActorTrainingReport]:
    return _train_flat(
        batch,
        seed=seed,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        task_only=True,
    )


def _positive_weight(target: Tensor) -> Tensor:
    positive = torch.sum(target)
    negative = target.numel() - positive
    return (negative / torch.clamp(positive, min=1.0)).clamp(1.0, 20.0)


def train_hierarchical_actor(
    batch: ActorBatch,
    *,
    seed: int,
    epochs: int,
    learning_rate: float = 2e-3,
    batch_size: int = 256,
) -> tuple[HierarchicalActor, ActorTrainingReport]:
    if epochs <= 0 or len(batch.features) == 0:
        raise ValueError("actor training requires positive epochs and contexts")
    torch.manual_seed(seed)
    model = HierarchicalActor()
    if model.trainable_parameters > 100_000:
        raise ValueError("actor exceeds the frozen 100,000-parameter budget")
    features = torch.from_numpy(batch.features)
    admissibility = torch.from_numpy(batch.admissibility_labels)
    obligations = torch.from_numpy(batch.obligation_labels)
    task_values = torch.from_numpy(batch.task_values)
    admissibility_weight = _positive_weight(admissibility)
    obligation_weight = _positive_weight(obligations)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator(device="cpu").manual_seed(seed ^ 0xAC70)
    final_loss = float("nan")
    for _ in range(epochs):
        order = torch.randperm(len(features), generator=generator)
        for start in range(0, len(features), batch_size):
            indices = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            scores = model(features[indices])
            loss = (
                F.binary_cross_entropy_with_logits(
                    scores.admissibility,
                    admissibility[indices],
                    pos_weight=admissibility_weight,
                )
                + F.binary_cross_entropy_with_logits(
                    scores.obligation,
                    obligations[indices],
                    pos_weight=obligation_weight,
                )
                + 0.5 * F.smooth_l1_loss(scores.task, task_values[indices])
            )
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach())
    _freeze(model)
    return model, ActorTrainingReport(
        "hierarchical",
        seed,
        epochs,
        len(features),
        batch_size,
        model.trainable_parameters,
        final_loss,
    )


@torch.no_grad()
def choose_actions(
    actor: FlatActor | HierarchicalActor,
    features: np.ndarray,
) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(features, dtype=np.float32))
    if isinstance(actor, FlatActor):
        return torch.argmax(actor(tensor), dim=1).cpu().numpy().astype(np.int64)
    scores = actor(tensor)
    admissible = torch.sigmoid(scores.admissibility) >= 0.5
    obligated = torch.sigmoid(scores.obligation) >= 0.5
    choices: list[int] = []
    for index in range(len(tensor)):
        candidates = torch.nonzero(admissible[index], as_tuple=False).flatten()
        if len(candidates) == 0:
            choices.append(int(CGSAction.EXIT))
            continue
        required = candidates[obligated[index, candidates]]
        if len(required) > 0:
            candidates = required
        candidate_task = scores.task[index, candidates]
        best_value = torch.max(candidate_task)
        tied = candidates[candidate_task == best_value]
        choices.append(int(torch.min(tied)))
    return np.asarray(choices, dtype=np.int64)