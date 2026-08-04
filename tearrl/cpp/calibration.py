"""Proper-score training and immutable scope estimation for TEAR-CPP.

This module deliberately knows nothing about difficulty, task value, action
severity, arm identity, or evaluator state.  Its complete policy-visible input
is the five-field tuple frozen in ``docs/cpp_prereg.md`` section 3.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


SCOPE_INPUT_DIM = 5
SCOPE_HIDDEN_DIM = 16
ESTIMATOR_EPOCHS = 100
ESTIMATOR_LEARNING_RATE = 0.003
TEMPERATURE_MIN = 0.25
TEMPERATURE_MAX = 4.0
TEMPERATURE_MAX_ITERATIONS = 200


def _cpu_float_tensor(value: torch.Tensor | np.ndarray | Iterable[float],
                      *, name: str) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.device.type != "cpu":
        raise ValueError(f"{name} must be a CPU tensor")
    if not tensor.is_floating_point():
        tensor = tensor.to(dtype=torch.float32)
    elif tensor.dtype != torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


def validated_scope_features(
    features: torch.Tensor | np.ndarray | Iterable[float],
) -> tuple[torch.Tensor, bool]:
    """Return a validated ``[batch, 5]`` CPU tensor and a squeeze flag.

    The declaration mask/value relation is checked before any learned layer is
    called.  Cue values are the registered binary marker observations.
    """

    tensor = _cpu_float_tensor(features, name="scope features")
    squeeze = tensor.ndim == 1
    if squeeze:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2 or tensor.shape[1] != SCOPE_INPUT_DIM:
        raise ValueError(
            f"scope features must have shape [batch, {SCOPE_INPUT_DIM}]")
    detached = tensor.detach()
    if not bool(torch.isfinite(detached).all()):
        raise ValueError("scope features must be finite")

    presence = detached[:, 0]
    declaration = detached[:, 1]
    presence_valid = torch.logical_or(presence == 0.0, presence == 1.0)
    absent_valid = torch.logical_and(presence == 0.0, declaration == 0.0)
    declared_valid = torch.logical_and(
        presence == 1.0,
        torch.logical_or(declaration == -1.0, declaration == 1.0),
    )
    if not bool(presence_valid.all()) or not bool(
            torch.logical_or(absent_valid, declared_valid).all()):
        raise ValueError(
            "invalid declaration mask/value pair; expected (0, 0) or "
            "(1, -1/+1)")

    cues = detached[:, 2:]
    if not bool(torch.logical_or(cues == -1.0, cues == 1.0).all()):
        raise ValueError("all three marker cues must be exactly -1 or +1")
    return tensor, squeeze


def validated_binary_targets(
    targets: torch.Tensor | np.ndarray | Iterable[float],
    *,
    expected_rows: int,
) -> torch.Tensor:
    tensor = _cpu_float_tensor(targets, name="scope targets").reshape(-1)
    if tensor.shape != (expected_rows,):
        raise ValueError(
            f"scope targets must contain exactly {expected_rows} rows")
    if not bool(torch.isfinite(tensor.detach()).all()):
        raise ValueError("scope targets must be finite")
    if not bool(torch.logical_or(tensor == 0.0, tensor == 1.0).all()):
        raise ValueError("scope targets must be binary")
    return tensor


class ScopeEstimator(nn.Module):
    """Registered ``5 -> 16 -> 16 -> 1`` calibrated scope estimator."""

    def __init__(self) -> None:
        super().__init__()
        self.input_layer = nn.Linear(SCOPE_INPUT_DIM, SCOPE_HIDDEN_DIM)
        self.hidden_layer = nn.Linear(SCOPE_HIDDEN_DIM, SCOPE_HIDDEN_DIM)
        self.output_layer = nn.Linear(SCOPE_HIDDEN_DIM, 1)
        self.register_buffer(
            "temperature",
            torch.tensor(1.0, dtype=torch.float32),
        )

    def representation(
        self,
        features: torch.Tensor | np.ndarray | Iterable[float],
    ) -> tuple[torch.Tensor, bool]:
        tensor, squeeze = validated_scope_features(features)
        hidden = torch.tanh(self.input_layer(tensor))
        hidden = torch.tanh(self.hidden_layer(hidden))
        return hidden, squeeze

    def unscaled_logits(
        self,
        features: torch.Tensor | np.ndarray | Iterable[float],
    ) -> torch.Tensor:
        hidden, squeeze = self.representation(features)
        logits = self.output_layer(hidden).squeeze(-1)
        return logits.squeeze(0) if squeeze else logits

    def calibrated_logits(
        self,
        features: torch.Tensor | np.ndarray | Iterable[float],
    ) -> torch.Tensor:
        return self.unscaled_logits(features) / self.temperature

    def forward(
        self,
        features: torch.Tensor | np.ndarray | Iterable[float],
    ) -> torch.Tensor:
        return torch.sigmoid(self.calibrated_logits(features))

    def set_temperature(self, value: float) -> None:
        value = float(value)
        if (not math.isfinite(value) or value < TEMPERATURE_MIN or
                value > TEMPERATURE_MAX):
            raise ValueError(
                f"temperature must be finite and in "
                f"[{TEMPERATURE_MIN}, {TEMPERATURE_MAX}]")
        with torch.no_grad():
            self.temperature.fill_(value)


@dataclass(frozen=True, slots=True)
class EstimatorTrainingReport:
    seed: int
    rows: int
    epochs: int
    learning_rate: float
    initial_log_loss: float
    final_log_loss: float


@dataclass(frozen=True, slots=True)
class TemperatureFitReport:
    rows: int
    iterations: int
    temperature: float
    initial_log_loss: float
    final_log_loss: float


def _binary_log_loss(logits: torch.Tensor,
                     targets: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, targets)


def train_scope_estimator(
    features: torch.Tensor | np.ndarray | Iterable[float],
    targets: torch.Tensor | np.ndarray | Iterable[float],
    *,
    seed: int,
    epochs: int = ESTIMATOR_EPOCHS,
    learning_rate: float = ESTIMATOR_LEARNING_RATE,
    checkpoint: Callable[[str], None] | None = None,
) -> tuple[ScopeEstimator, EstimatorTrainingReport]:
    """Train for the fixed full-batch proper-score budget on CPU."""

    if epochs != ESTIMATOR_EPOCHS:
        raise ValueError(
            f"TEAR-CPP estimator training requires exactly "
            f"{ESTIMATOR_EPOCHS} epochs")
    if learning_rate != ESTIMATOR_LEARNING_RATE:
        raise ValueError(
            f"TEAR-CPP estimator learning rate must be "
            f"{ESTIMATOR_LEARNING_RATE}")
    inputs, _ = validated_scope_features(features)
    labels = validated_binary_targets(targets, expected_rows=len(inputs))

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        model = ScopeEstimator().cpu()
    optimizer = torch.optim.Adam(
        tuple(model.parameters()), lr=ESTIMATOR_LEARNING_RATE)

    model.train()
    with torch.no_grad():
        initial = float(_binary_log_loss(model.unscaled_logits(inputs), labels))
    for epoch in range(1, ESTIMATOR_EPOCHS + 1):
        if checkpoint is not None:
            checkpoint(
                f"scope estimator epoch {epoch}/{ESTIMATOR_EPOCHS}"
            )
        optimizer.zero_grad(set_to_none=True)
        loss = _binary_log_loss(model.unscaled_logits(inputs), labels)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        final = float(_binary_log_loss(model.unscaled_logits(inputs), labels))
    return model, EstimatorTrainingReport(
        seed=int(seed),
        rows=len(inputs),
        epochs=ESTIMATOR_EPOCHS,
        learning_rate=ESTIMATOR_LEARNING_RATE,
        initial_log_loss=initial,
        final_log_loss=final,
    )


def fit_temperature(
    estimator: ScopeEstimator,
    features: torch.Tensor | np.ndarray | Iterable[float],
    targets: torch.Tensor | np.ndarray | Iterable[float],
    *,
    max_iterations: int = TEMPERATURE_MAX_ITERATIONS,
) -> TemperatureFitReport:
    """Fit the registered bounded scalar temperature deterministically.

    A golden-section search is used because the calibrated parameter is a
    single positive scalar.  The learned network is never updated here.
    """

    if max_iterations != TEMPERATURE_MAX_ITERATIONS:
        raise ValueError(
            "TEAR-CPP temperature fitting requires a maximum of exactly "
            f"{TEMPERATURE_MAX_ITERATIONS} iterations")
    inputs, _ = validated_scope_features(features)
    labels = validated_binary_targets(targets, expected_rows=len(inputs))
    estimator.eval()
    logits = estimator.unscaled_logits(inputs).detach().to(dtype=torch.float64)
    labels64 = labels.detach().to(dtype=torch.float64)

    def objective(temperature: float) -> float:
        scaled = logits / float(temperature)
        return float(F.binary_cross_entropy_with_logits(scaled, labels64))

    initial = objective(1.0)
    lower, upper = TEMPERATURE_MIN, TEMPERATURE_MAX
    inverse_phi = (math.sqrt(5.0) - 1.0) / 2.0
    left = upper - inverse_phi * (upper - lower)
    right = lower + inverse_phi * (upper - lower)
    left_value = objective(left)
    right_value = objective(right)
    iterations = 0
    for iterations in range(1, TEMPERATURE_MAX_ITERATIONS + 1):
        if upper - lower <= 1e-12:
            break
        if left_value <= right_value:
            upper = right
            right, right_value = left, left_value
            left = upper - inverse_phi * (upper - lower)
            left_value = objective(left)
        else:
            lower = left
            left, left_value = right, right_value
            right = lower + inverse_phi * (upper - lower)
            right_value = objective(right)

    candidates = (
        (TEMPERATURE_MIN, objective(TEMPERATURE_MIN)),
        (TEMPERATURE_MAX, objective(TEMPERATURE_MAX)),
        (left, left_value),
        (right, right_value),
        ((lower + upper) / 2.0, objective((lower + upper) / 2.0)),
    )
    temperature, final = min(candidates, key=lambda item: (item[1], item[0]))
    estimator.set_temperature(float(temperature))
    return TemperatureFitReport(
        rows=len(inputs),
        iterations=iterations,
        temperature=float(temperature),
        initial_log_loss=initial,
        final_log_loss=float(final),
    )


def canonical_state_digest(module: nn.Module) -> str:
    """Hash a module state with sorted names and canonical little-endian bytes."""

    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous().numpy()
        if value.dtype.hasobject:
            raise TypeError("object tensors cannot be canonically digested")
        little_dtype = value.dtype.newbyteorder("<")
        canonical = np.asarray(value, dtype=little_dtype, order="C")
        metadata = json.dumps(
            {
                "name": name,
                "dtype": canonical.dtype.str,
                "shape": list(canonical.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(metadata).to_bytes(8, byteorder="big"))
        digest.update(metadata)
        payload = canonical.tobytes(order="C")
        digest.update(len(payload).to_bytes(8, byteorder="big"))
        digest.update(payload)
    return digest.hexdigest()


def _optimizer_parameter_ids(
    optimizer: torch.optim.Optimizer | None,
) -> set[int]:
    if optimizer is None:
        return set()
    return {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }


@dataclass(frozen=True, slots=True)
class FrozenScopeEstimator:
    """A separately stored estimator with executable immutability checks."""

    estimator: ScopeEstimator
    digest: str

    @classmethod
    def from_estimator(
        cls,
        estimator: ScopeEstimator,
    ) -> "FrozenScopeEstimator":
        clone = copy.deepcopy(estimator).cpu()
        clone.eval()
        for parameter in clone.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        return cls(clone, canonical_state_digest(clone))

    def assert_integrity(
        self,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        if self.estimator.training:
            raise AssertionError("frozen scope estimator must remain in eval mode")
        frozen_ids = {id(parameter) for parameter in self.estimator.parameters()}
        if frozen_ids & _optimizer_parameter_ids(optimizer):
            raise AssertionError(
                "frozen scope estimator appears in a policy optimizer")
        for parameter in self.estimator.parameters():
            if parameter.requires_grad:
                raise AssertionError(
                    "frozen scope estimator parameter requires gradients")
            if parameter.grad is not None:
                raise AssertionError(
                    "frozen scope estimator accumulated a gradient")
        actual = canonical_state_digest(self.estimator)
        if actual != self.digest:
            raise AssertionError(
                f"frozen scope estimator digest changed: "
                f"{self.digest} -> {actual}")

    def probabilities(
        self,
        features: torch.Tensor | np.ndarray | Iterable[float],
    ) -> tuple[float, ...]:
        """Return detached host probabilities suitable for hard gating."""

        self.assert_integrity()
        inputs, _ = validated_scope_features(features)
        with torch.inference_mode():
            values = self.estimator(inputs).detach().cpu().reshape(-1)
        return tuple(float(value) for value in values)

    def probability(
        self,
        features: torch.Tensor | np.ndarray | Iterable[float],
    ) -> float:
        values = self.probabilities(features)
        if len(values) != 1:
            raise ValueError("probability() requires exactly one scope row")
        return values[0]
