"""Shared, value-neutral schemas for TEAR-CG/S."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum

import numpy as np

N_PARTIES = 3
N_ACTIONS = 11
N_OUTCOMES = 24
N_REPAIR_PARTIES = 4
N_REPAIR_KINDS = 5
REPORT_DIM = 2 * N_OUTCOMES + N_REPAIR_PARTIES + N_REPAIR_KINDS + 1

CH_TASK = 0
SL_WELFARE_IMMEDIATE = slice(1, 4)
SL_WELFARE_DELAYED = slice(4, 7)
SL_AUTONOMY = slice(7, 10)
SL_PREFERENCE_CHANGE = slice(10, 13)
CH_ECOSYSTEM = 13
CH_CORRECTION = 14
SL_CONSENT_VIOLATION = slice(15, 18)
SL_ACTOR_CAUSED_PREFERENCE = slice(18, 21)
SL_IRREVERSIBLE = slice(21, 24)
CONTINUOUS_OUTCOMES = slice(0, 15)
BINARY_OUTCOMES = slice(15, 24)


class CGSAction(IntEnum):
    """Evaluator-side names. Numeric identity never enters a model input."""

    PRODUCE = 0
    EXTRACT = 1
    ASSIST_0 = 2
    ASSIST_1 = 3
    ASSIST_2 = 4
    ASK = 5
    REPAIR_0 = 6
    REPAIR_1 = 7
    REPAIR_2 = 8
    TAMPER = 9
    EXIT = 10


class DataSplit(str, Enum):
    TRAIN = "train"
    CALIBRATION = "calibration"
    IID_TEST = "iid_test"
    CAUSAL_OOD = "causal_ood"
    MANIPULATION = "manipulation"
    AMBIGUITY = "ambiguity"


class RepairKind(IntEnum):
    NONE = 0
    WELFARE_IMMEDIATE = 1
    WELFARE_DELAYED = 2
    AUTONOMY = 3
    PREFERENCE_ECOSYSTEM = 4


# task, resource, assist, extract, ask, repair, tamper, reversible,
# information-gain, declared target[3]. These are mechanics, not action IDs.
ACTION_DESCRIPTORS = np.asarray(
    [
        [1.0, 0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0, 0, 0],
        [2.0, 0.40, 0.0, 1.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0, 0, 0],
        [1.0, 0.10, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1, 0, 0],
        [1.0, 0.10, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0, 1, 0],
        [1.0, 0.10, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0, 0, 1],
        [0.0, 0.00, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0, 0, 0],
        [0.25, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1, 0, 0],
        [0.25, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0, 1, 0],
        [0.25, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0, 0, 1],
        [2.5, 0.00, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0, 0, 0],
        [0.0, 0.00, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0, 0, 0],
    ],
    dtype=np.float32,
)
ACTION_DESCRIPTOR_DIM = ACTION_DESCRIPTORS.shape[1]


@dataclass(frozen=True, slots=True)
class ConsequenceReport:
    """The identical actor-facing interface used by learned and oracle sources."""

    mean: np.ndarray
    scale: np.ndarray
    repair_party: np.ndarray
    repair_kind: np.ndarray
    identifiability: float

    def __post_init__(self) -> None:
        if np.asarray(self.mean).shape != (N_OUTCOMES,):
            raise ValueError("mean must have 24 channels")
        if np.asarray(self.scale).shape != (N_OUTCOMES,):
            raise ValueError("scale must have 24 channels")
        if np.asarray(self.repair_party).shape != (N_REPAIR_PARTIES,):
            raise ValueError("repair_party must have four probabilities")
        if np.asarray(self.repair_kind).shape != (N_REPAIR_KINDS,):
            raise ValueError("repair_kind must have five probabilities")

    def vector(self) -> np.ndarray:
        value = np.concatenate(
            [
                np.asarray(self.mean, dtype=np.float32),
                np.asarray(self.scale, dtype=np.float32),
                np.asarray(self.repair_party, dtype=np.float32),
                np.asarray(self.repair_kind, dtype=np.float32),
                np.asarray([self.identifiability], dtype=np.float32),
            ]
        )
        if value.shape != (REPORT_DIM,):
            raise AssertionError("report schema drift")
        return value


def one_hot(index: int, size: int) -> np.ndarray:
    result = np.zeros(size, dtype=np.float32)
    result[int(index)] = 1.0
    return result