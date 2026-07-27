"""Descriptive attribution-window diagnostics for responsibility channel ``r``.

The frozen TEARRL-0 implementation reads the denominator before the learner's
current action is applied.  This module does not change that registered
mechanism.  It makes three candidate readings explicit so the preregistration
can report the measurement defect without repairing it by fiat:

``current``
    Existing implementation: learner contribution divided by the magnitude of
    intervening change observed before the current learner action.
``completed_window``
    Diagnostic correction: include the current learner contribution in the
    same decision-to-decision net-change window.
``contribution_share``
    Bounded diagnostic: divide by the sum of absolute learner and intervening
    contributions, avoiding cancellation in the net-change denominator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..spec import EPS_R


@dataclass(frozen=True, slots=True)
class AttributionRatios:
    current: float
    completed_window: float
    contribution_share: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def attribution_ratios(
    learner_delta: float,
    intervening_delta: float,
    eps: float = EPS_R,
) -> AttributionRatios:
    """Return all registered/diagnostic readings in normalized welfare units."""
    values = np.asarray([learner_delta, intervening_delta, eps], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("attribution inputs must be finite")
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    numerator = abs(float(learner_delta))
    intervening = float(intervening_delta)
    return AttributionRatios(
        current=numerator / (abs(intervening) + eps),
        completed_window=numerator / (abs(intervening + learner_delta) + eps),
        contribution_share=numerator / (abs(intervening) + numerator + eps),
    )


def shifted(values: np.ndarray, offset: int) -> np.ndarray:
    """Shift a causal contribution in time, filling exposed cells with zero."""
    source = np.asarray(values, dtype=np.float64)
    if source.ndim != 1:
        raise ValueError("shifted values must be one-dimensional")
    output = np.zeros_like(source)
    if offset == 0:
        output[:] = source
    elif offset > 0 and offset < len(source):
        output[offset:] = source[:-offset]
    elif offset < 0 and -offset < len(source):
        output[:offset] = source[-offset:]
    return output


def ratio_series(
    learner_delta: np.ndarray,
    intervening_delta: np.ndarray,
    *,
    learner_shift: int = 0,
) -> dict[str, np.ndarray]:
    """Vectorized ratios; ``learner_shift`` exists for the mutation test."""
    learner = np.asarray(learner_delta, dtype=np.float64)
    intervening = np.asarray(intervening_delta, dtype=np.float64)
    if learner.ndim != 1 or intervening.ndim != 1 or learner.shape != intervening.shape:
        raise ValueError("attribution series must be equal one-dimensional arrays")
    if not np.isfinite(learner).all() or not np.isfinite(intervening).all():
        raise ValueError("attribution series must be finite")

    aligned = shifted(learner, learner_shift)
    numerator = np.abs(aligned)
    return {
        "current": numerator / (np.abs(intervening) + EPS_R),
        "completed_window": numerator / (np.abs(intervening + aligned) + EPS_R),
        "contribution_share": numerator
        / (np.abs(intervening) + numerator + EPS_R),
    }


def distribution_summary(
    values: np.ndarray,
    episode_id: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Pinned descriptive statistics, including within-episode lag-1 correlation."""
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or data.size == 0 or not np.isfinite(data).all():
        raise ValueError("summary requires a nonempty finite one-dimensional array")

    rounded = np.round(data, 12)
    _, counts = np.unique(rounded, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = -float(np.sum(probabilities * np.log(probabilities)))

    autocorrelation = 0.0
    if episode_id is not None:
        episodes = np.asarray(episode_id)
        if episodes.shape != data.shape:
            raise ValueError("episode IDs must match values")
        same_episode = episodes[:-1] == episodes[1:]
        left, right = data[:-1][same_episode], data[1:][same_episode]
        if left.size >= 2 and float(left.std()) > 0.0 and float(right.std()) > 0.0:
            autocorrelation = float(np.corrcoef(left, right)[0, 1])

    nonzero = data[data > 0.0]
    return {
        "n": int(data.size),
        "mean": float(data.mean()),
        "variance": float(data.var()),
        "mass_zero": float(np.mean(data == 0.0)),
        "mass_between_zero_and_one": float(np.mean((data > 0.0) & (data < 1.0))),
        "mass_at_or_above_one": float(np.mean(data >= 1.0)),
        "clipping_frequency": float(np.mean(data >= 1.0)),
        "median_nonzero": float(np.median(nonzero)) if nonzero.size else 0.0,
        "max": float(data.max()),
        "unique_support": int(counts.size),
        "effective_support": float(np.exp(entropy)),
        "lag1_autocorrelation": autocorrelation,
    }


def comparison_summary(old: np.ndarray, corrected: np.ndarray) -> dict[str, float]:
    old_values = np.asarray(old, dtype=np.float64)
    new_values = np.asarray(corrected, dtype=np.float64)
    if old_values.shape != new_values.shape or old_values.ndim != 1:
        raise ValueError("comparison arrays must be equal one-dimensional arrays")
    difference = np.abs(old_values - new_values)
    return {
        "fraction_changed": float(np.mean(difference > 1e-12)),
        "mean_absolute_change": float(difference.mean()),
        "max_absolute_change": float(difference.max(initial=0.0)),
    }
