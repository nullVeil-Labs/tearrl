"""
Witness training and freeze -- B.3, §2.3(3).

    90/10 split; ensemble 5; Adam 1e-3; patience 10.

    FREEZE CRITERIA:
      |F_phi - ybar| <= half a bin on >= 99% of held-out pairs
      per event type: recall >= 0.99 AND FPR <= 0.01

The freeze gate is not a formality. Every condition from C3-IM through C6
reasons about consequences through this one instrument, and §5.5's Witness
Exploitation Gap measures whether an agent has learned to act where the
witness is wrong. A witness frozen below criterion would hand every agent the
same exploitable blind spots and confound the comparison H1 depends on.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .dataset import INPUT_DIM, WitnessDataset, split
from .model import ENSEMBLE_SIZE, WitnessEnsemble
from .targets import ACTION_DESCRIPTORS, HALF_BIN, N_DELTA_CHANNELS

LEARNING_RATE: float = 1e-3      # B.3
PATIENCE: int = 10               # B.3
BATCH_SIZE: int = 8192
MAX_EPOCHS: int = 200

EVENT_NAMES: tuple[str, ...] = ("U", "E", "O", "irr")


@dataclass(slots=True)
class FreezeReport:
    """B.3 freeze criteria, evaluated on held-out pairs."""

    within_half_bin: float
    per_channel_within_half_bin: list[float]
    max_abs_error: float
    recall: dict[str, float] = field(default_factory=dict)
    fpr: dict[str, float] = field(default_factory=dict)
    n_pairs: int = 0

    @property
    def delta_criterion_met(self) -> bool:
        return self.within_half_bin >= 0.99

    @property
    def event_criterion_met(self) -> bool:
        return all(self.recall[e] >= 0.99 for e in EVENT_NAMES) and all(
            self.fpr[e] <= 0.01 for e in EVENT_NAMES
        )

    @property
    def frozen(self) -> bool:
        return self.delta_criterion_met and self.event_criterion_met

    def describe(self) -> str:
        lines = [
            f"held-out pairs           : {self.n_pairs:,}",
            f"within half a bin (all 5): {self.within_half_bin:.4%}   "
            f"[{'PASS' if self.delta_criterion_met else 'FAIL'} vs 99%]",
            f"max |F_phi - ybar|       : {self.max_abs_error:.4f}  (half bin = {HALF_BIN})",
            "per-channel within half bin:",
        ]
        names = ["Dtask", "DW_self", "DW_other", "Dauton", "Dtrust"]
        for name, value in zip(names, self.per_channel_within_half_bin):
            lines.append(f"    {name:<9s} {value:.4%}")
        lines.append("events:")
        for event in EVENT_NAMES:
            recall_ok = "OK" if self.recall[event] >= 0.99 else "LOW"
            fpr_ok = "OK" if self.fpr[event] <= 0.01 else "HIGH"
            lines.append(
                f"    {event:<4s} recall {self.recall[event]:.4f} [{recall_ok}]   "
                f"FPR {self.fpr[event]:.5f} [{fpr_ok}]"
            )
        lines.append(f"FREEZE: {'CRITERIA MET' if self.frozen else 'NOT MET'}")
        return "\n".join(lines)


class _GpuDataset:
    """Dataset resident on device; batches assembled by indexing.

    Materializing [n_pairs, 450] inputs would cost ~15 GB at B.3 scale. Storing
    observations once per unique state and gathering per batch keeps it near
    3 GB and removes the host-to-device copy from the training loop entirely.
    """

    def __init__(self, dataset: WitnessDataset, device: torch.device) -> None:
        self.obs = torch.from_numpy(dataset.obs).to(device)
        self.descriptors = torch.from_numpy(ACTION_DESCRIPTORS).to(device)
        self.state_index = torch.from_numpy(dataset.state_index.astype(np.int64)).to(device)
        self.action = torch.from_numpy(dataset.action.astype(np.int64)).to(device)
        self.target = torch.from_numpy(dataset.target).to(device)
        self.device = device

    def batch(self, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.cat(
            [self.obs[self.state_index[rows]], self.descriptors[self.action[rows]]],
            dim=1,
        )
        return inputs, self.target[rows]


def _loss(raw: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE on the continuous Delta channels, BCE-with-logits on the four
    indicator channels. Keeping them separate matters: the event channels are
    exact 0/1 indicators, and regressing them under MSE alongside the deltas
    would let a small delta error trade against a misclassified violation."""
    delta_pred, event_logits = raw[..., :N_DELTA_CHANNELS], raw[..., N_DELTA_CHANNELS:]
    delta_true, event_true = target[..., :N_DELTA_CHANNELS], target[..., N_DELTA_CHANNELS:]
    return nn.functional.mse_loss(delta_pred, delta_true) + nn.functional.binary_cross_entropy_with_logits(
        event_logits, event_true
    )


@torch.no_grad()
def _validation_loss(member: nn.Module, data: _GpuDataset, rows: torch.Tensor) -> float:
    member.eval()
    total = 0.0
    count = 0
    for start in range(0, len(rows), BATCH_SIZE * 4):
        chunk = rows[start : start + BATCH_SIZE * 4]
        inputs, targets = data.batch(chunk)
        total += float(_loss(member(inputs), targets)) * len(chunk)
        count += len(chunk)
    member.train()
    return total / max(count, 1)


def train_witness(
    dataset: WitnessDataset,
    device: str | torch.device = "cuda",
    ensemble_size: int = ENSEMBLE_SIZE,
    max_epochs: int = MAX_EPOCHS,
    seed: int = 20260726,
    verbose: bool = True,
) -> tuple[WitnessEnsemble, np.ndarray, np.ndarray]:
    """Train the 5-member ensemble. Returns (ensemble, train_rows, val_rows)."""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    train_rows_np, val_rows_np = split(dataset)
    data = _GpuDataset(dataset, device)
    train_rows = torch.from_numpy(train_rows_np).to(device)
    val_rows = torch.from_numpy(val_rows_np).to(device)

    # Seed BEFORE construction: weight init draws from the global torch RNG,
    # so without this the frozen witness would not be reproducible from the
    # repository -- unacceptable for a preregistered instrument that every
    # condition shares.
    torch.manual_seed(seed)
    ensemble = WitnessEnsemble(INPUT_DIM, size=ensemble_size).to(device)
    if verbose:
        print(
            f"device={device}  pairs={dataset.n_pairs:,}  "
            f"train={len(train_rows):,}  val={len(val_rows):,}  "
            f"params={ensemble.parameter_count():,}"
        )

    generator = torch.Generator(device=device)

    for member_index, member in enumerate(ensemble.members):
        generator.manual_seed(seed + 1000 * member_index)
        optimizer = torch.optim.Adam(member.parameters(), lr=LEARNING_RATE)
        best_loss = float("inf")
        best_state = None
        patience_left = PATIENCE
        started = time.perf_counter()

        for epoch in range(max_epochs):
            permutation = train_rows[
                torch.randperm(len(train_rows), generator=generator, device=device)
            ]
            for start in range(0, len(permutation), BATCH_SIZE):
                inputs, targets = data.batch(permutation[start : start + BATCH_SIZE])
                optimizer.zero_grad(set_to_none=True)
                _loss(member(inputs), targets).backward()
                optimizer.step()

            current = _validation_loss(member, data, val_rows)
            if current < best_loss - 1e-7:
                best_loss = current
                best_state = {k: v.detach().clone() for k, v in member.state_dict().items()}
                patience_left = PATIENCE
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        if best_state is not None:
            member.load_state_dict(best_state)
        if verbose:
            print(
                f"  member {member_index + 1}/{ensemble_size}: "
                f"val_loss={best_loss:.6f}  epochs={epoch + 1}  "
                f"{time.perf_counter() - started:.1f}s"
            )

    return ensemble, train_rows_np, val_rows_np


@torch.no_grad()
def evaluate_freeze(
    ensemble: WitnessEnsemble,
    dataset: WitnessDataset,
    rows: np.ndarray,
    device: str | torch.device = "cuda",
) -> FreezeReport:
    """B.3 freeze criteria on held-out pairs."""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    ensemble.eval()
    data = _GpuDataset(dataset, device)
    row_tensor = torch.from_numpy(rows).to(device)

    within = torch.zeros(N_DELTA_CHANNELS, dtype=torch.float64, device=device)
    all_within = 0
    max_error = 0.0
    total = 0

    true_positive = {e: 0 for e in EVENT_NAMES}
    false_negative = {e: 0 for e in EVENT_NAMES}
    false_positive = {e: 0 for e in EVENT_NAMES}
    true_negative = {e: 0 for e in EVENT_NAMES}

    for start in range(0, len(row_tensor), BATCH_SIZE * 4):
        chunk = row_tensor[start : start + BATCH_SIZE * 4]
        inputs, targets = data.batch(chunk)
        # Raw (unquantized) predictions: the criterion is |F_phi - ybar|.
        prediction = ensemble.predict(inputs, quantize_deltas=False)

        error = (prediction[:, :N_DELTA_CHANNELS] - targets[:, :N_DELTA_CHANNELS]).abs()
        ok = error <= HALF_BIN
        within += ok.sum(dim=0).double()
        all_within += int(ok.all(dim=1).sum())
        max_error = max(max_error, float(error.max()))
        total += len(chunk)

        predicted_event = prediction[:, N_DELTA_CHANNELS:] >= 0.5
        actual_event = targets[:, N_DELTA_CHANNELS:] >= 0.5
        for index, name in enumerate(EVENT_NAMES):
            p, a = predicted_event[:, index], actual_event[:, index]
            true_positive[name] += int((p & a).sum())
            false_negative[name] += int((~p & a).sum())
            false_positive[name] += int((p & ~a).sum())
            true_negative[name] += int((~p & ~a).sum())

    def safe_ratio(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 1.0

    return FreezeReport(
        within_half_bin=all_within / max(total, 1),
        per_channel_within_half_bin=(within / max(total, 1)).tolist(),
        max_abs_error=max_error,
        recall={
            e: safe_ratio(true_positive[e], true_positive[e] + false_negative[e])
            for e in EVENT_NAMES
        },
        fpr={
            e: safe_ratio(false_positive[e], false_positive[e] + true_negative[e])
            if (false_positive[e] + true_negative[e])
            else 0.0
            for e in EVENT_NAMES
        },
        n_pairs=total,
    )


def freeze(ensemble: WitnessEnsemble, path: Path, report: FreezeReport) -> None:
    """§2.3(3)-(4): persist the frozen witness. The identical file is then
    copied into C3-IM, C4, C5 and C6 -- one instrument, every condition."""
    torch.save(
        {
            "state_dict": ensemble.state_dict(),
            "input_dim": ensemble.input_dim,
            "ensemble_size": len(ensemble.members),
            "freeze_report": asdict(report),
            "frozen": report.frozen,
        },
        path,
    )


def load_frozen_witness(path: Path, device: str | torch.device = "cuda") -> WitnessEnsemble:
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    # weights_only=True: the checkpoint holds only tensors and primitives, so
    # there is no reason to allow arbitrary unpickling. This repository is
    # public and checkpoints are the kind of file people fetch from strangers.
    blob = torch.load(path, map_location=device, weights_only=True)
    ensemble = WitnessEnsemble(blob["input_dim"], size=blob["ensemble_size"]).to(device)
    ensemble.load_state_dict(blob["state_dict"])
    ensemble.eval()
    for parameter in ensemble.parameters():   # frozen means frozen
        parameter.requires_grad_(False)
    return ensemble
