"""
Witness network F_phi -- B.1, §2.4.

    MLP [dim(o^F) + dim(d(a))] -> 128 -> 128 -> 9
        5 Delta channels : linear, quantized POST HOC
        U, E, O, irr     : sigmoid
    ensemble 5 (mean; disagreement exploratory)

The five Delta heads stay continuous through training and are snapped to the
21-point grid only at prediction time. That is what makes B.3's freeze
criterion well defined: the quantized prediction equals the target exactly
when the raw error is within half a bin, so "|F_phi - ybar| <= half a bin"
and "the witness reports the right bin" are the same statement.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .targets import GRID_STEP, N_DELTA_CHANNELS, WITNESS_DIM

HIDDEN: int = 128
ENSEMBLE_SIZE: int = 5


class WitnessNet(nn.Module):
    """One ensemble member (B.1)."""

    def __init__(self, input_dim: int, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden, WITNESS_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Raw outputs: Delta channels are pre-quantization, event channels
        are LOGITS (BCE-with-logits is applied by the trainer)."""
        return self.head(self.trunk(x))

    @staticmethod
    def split(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return raw[..., :N_DELTA_CHANNELS], raw[..., N_DELTA_CHANNELS:]

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Calibrated prediction: deltas as-is, events through sigmoid."""
        raw = self.forward(x)
        deltas, event_logits = self.split(raw)
        return torch.cat([deltas, torch.sigmoid(event_logits)], dim=-1)


class WitnessEnsemble(nn.Module):
    """B.1: five members, prediction = mean. Disagreement is exploratory-only
    (§2.4) and exposed here for logging, never for control."""

    def __init__(self, input_dim: int, size: int = ENSEMBLE_SIZE, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.members = nn.ModuleList(
            [WitnessNet(input_dim, hidden) for _ in range(size)]
        )
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([m.predict(x) for m in self.members], dim=0).mean(dim=0)

    @torch.no_grad()
    def predict(self, x: torch.Tensor, quantize_deltas: bool = True) -> torch.Tensor:
        """Frozen-witness output yhat(a) (§2.4).

        With quantize_deltas=True the Delta channels are snapped to the
        21-point grid -- this is the value c_t consumes, and §2.6 requires
        c_t to see the SAME quantized numbers the policy does.
        """
        mean = self.forward(x)
        if not quantize_deltas:
            return mean
        deltas = mean[..., :N_DELTA_CHANNELS]
        snapped = torch.round(torch.clamp(deltas, -1.0, 1.0) / GRID_STEP) * GRID_STEP
        return torch.cat([snapped, mean[..., N_DELTA_CHANNELS:]], dim=-1)

    @torch.no_grad()
    def disagreement(self, x: torch.Tensor) -> torch.Tensor:
        """Per-pair ensemble std over the Delta channels. Exploratory only."""
        stacked = torch.stack([m.predict(x) for m in self.members], dim=0)
        return stacked[..., :N_DELTA_CHANNELS].std(dim=0).mean(dim=-1)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def quantize_numpy(deltas: np.ndarray) -> np.ndarray:
    return np.round(np.clip(deltas, -1.0, 1.0) / GRID_STEP) * GRID_STEP
