"""
Responsibility state c_t -- §2.6, constants B.5.

    c_t = [h, b, r, d];   h, b, r in [0,1],  d in [0, d_max]

    harm_hat    = max(0, -DW_other_hat)
    benefit_hat = max(0, +DW_other_hat caused by the learner)

    h' = clip(rho*h + alpha*harm_hat)
    b' = clip(rho*b + alpha*benefit_hat)
    r' = clip(rho*r + alpha*|d_other,learner| / (|d_other,total| + eps_r))
    d' = clip(d + harm_hat - restored_hat, 0, d_max)

Inputs are the FROZEN WITNESS's quantized predictions, never oracle values --
U2 is the only oracle-c_t condition (§2.6). So c_t is a running account of
what the agent's own instrument told it it was doing, which is the point: the
disposition under test has to be built from the agent's own predicted
consequences, not from ground truth it never sees.

--------------------------------------------------------------------------
TWO READINGS THE PROTOCOL LEAVES OPEN -- FLAGGED FOR THE PI
--------------------------------------------------------------------------
(1) `restored_hat` is defined as "predicted DW_other of the REPAIR TRANSFER".
    Implemented literally: only `repair` discharges debt. `share` also raises
    the counterpart's welfare, but if it counted here an agent could steal,
    then hand over unrelated resources, and watch its debt channel clear
    without ever touching the U record or forfeiting the illicit credit.
    A.3 makes `repair` the verb that walks a U record toward resolution;
    letting generic generosity do the same work would break the same-unit
    harm/repair accounting §2.6 asks for.

(2) `|d_other,learner| / (|d_other,total| + eps_r)` -- the attribution ratio.
    The protocol does not say over what window "total" is measured. Read here
    as: the learner's predicted contribution to the counterpart's welfare
    this turn, over ALL welfare change the counterpart experienced since the
    learner's previous decision (which includes the counterpart's own turn).
    That makes r answer "of everything that happened to them, how much was
    me" -- distinct from h, which tracks absolute harm. If `total` were
    restricted to the learner's own turn, r would collapse to a near-constant
    multiple of h and carry no independent signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .spec import (
    ALPHA,
    D_MAX,
    EPS_R,
    RHO,
    Verb,
)

# Witness channel indices (§2.4 output order).
CH_DTASK, CH_DW_SELF, CH_DW_OTHER, CH_DAUTONOMY, CH_DTRUST = 0, 1, 2, 3, 4
CH_U, CH_E, CH_O, CH_IRR = 5, 6, 7, 8

C_DIM: int = 4


@dataclass(slots=True)
class ResponsibilityState:
    """c_t = [h, b, r, d] (§2.6).

    h  accumulated predicted harm caused
    b  accumulated predicted benefit caused
    r  accumulated attribution share
    d  outstanding harm debt (harm minus repair, same units)
    """

    h: float = 0.0
    b: float = 0.0
    r: float = 0.0
    d: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.array([self.h, self.b, self.r, self.d], dtype=np.float32)

    def clone(self) -> "ResponsibilityState":
        return ResponsibilityState(self.h, self.b, self.r, self.d)

    def reset(self) -> None:
        self.h = self.b = self.r = self.d = 0.0

    def update(
        self,
        prediction: np.ndarray,
        action: Verb,
        other_total_welfare_change: float,
    ) -> "ResponsibilityState":
        """One update from the frozen witness's quantized prediction (§2.6).

        `prediction` is the 9-channel yhat(a) for the action actually taken,
        with Delta channels already quantized -- §2.6 requires c_t to consume
        the identical quantized values the policy saw.

        `other_total_welfare_change` is the counterpart's total welfare change
        since the learner's previous decision, in the SAME normalized units as
        the witness channel, and is the denominator of the attribution ratio.
        """
        delta_welfare_other = float(prediction[CH_DW_OTHER])

        harm = max(0.0, -delta_welfare_other)
        benefit = max(0.0, delta_welfare_other)

        # Only the repair transfer discharges debt -- see note (1) above.
        restored = benefit if Verb(action) is Verb.REPAIR else 0.0

        attribution = abs(delta_welfare_other) / (
            abs(other_total_welfare_change) + EPS_R
        )

        self.h = _clip_unit(RHO * self.h + ALPHA * harm)
        self.b = _clip_unit(RHO * self.b + ALPHA * benefit)
        self.r = _clip_unit(RHO * self.r + ALPHA * attribution)
        self.d = float(np.clip(self.d + harm - restored, 0.0, D_MAX))
        return self


def _clip_unit(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def zero_state() -> ResponsibilityState:
    """The standard empty-ledger initialization G5's free rollouts start from
    (§1 G5, §5.3)."""
    return ResponsibilityState()
