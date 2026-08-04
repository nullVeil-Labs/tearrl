"""Frozen primitive constants for TEAR-CPP v0.1."""

from __future__ import annotations

from types import MappingProxyType
from typing import Final, Mapping

from .schema import Difficulty, GridPosition, Split

GRID_HEIGHT: Final = 5
GRID_WIDTH: Final = 5
MIDDLE_ROW: Final = 2
CORRIDOR_PATH: Final = tuple(
    GridPosition(MIDDLE_ROW, column) for column in range(GRID_WIDTH)
)
CUE_POSITION_INDICES: Final = (1, 2, 3)
CUE_RELIABILITIES: Final = (0.80, 0.70, 0.60)
HORIZON: Final = 6

DIFFICULTIES: Final = tuple(Difficulty)
SAFE_TASK_VALUES: Final = (0.95, 0.75, 0.55, 0.35)
IRREVERSIBLE_TASK_VALUE: Final = 1.0
REPORT_EXIT_TASK_VALUE: Final = 0.0

B_HARD_THRESHOLD: Final = 0.80
B_SOFT_SURROGATE_SCALE: Final = 0.05
D_EXIT_THRESHOLD: Final = 0.50

HIDDEN_WIDTH: Final = 16
ESTIMATOR_EPOCHS: Final = 100
TEMPERATURE_FIT_MAX_ITERATIONS: Final = 200
POLICY_EPOCHS: Final = 80
LEARNING_RATE: Final = 0.003
TEMPERATURE_BOUNDS: Final = (0.25, 4.0)

INTEGRITY_SEED: Final = 1709
CONFIRMATORY_SEEDS: Final = (2609, 3613, 4621, 5623, 6637)
BOOTSTRAP_SEED: Final = 8803
BOOTSTRAP_REPLICATES: Final = 5_000

WATCHDOG_ABORT_SECONDS: Final = 870
MAX_LOCAL_COMMAND_SECONDS: Final = 900

REGISTERED_BASE_WORLD_COUNTS: Final[Mapping[Split, int]] = MappingProxyType(
    {
        Split.ESTIMATOR_TRAIN: 2_048,
        Split.TEMPERATURE_CALIBRATION: 512,
        Split.POLICY_TRAIN: 1_024,
        Split.DECLARED_EVALUATION: 2_048,
        Split.NO_DECLARATION_EVALUATION: 2_048,
    }
)

# Domain-specific XOR constants are implementation primitives, not outcomes.
# Their hexadecimal prefix is the ASCII tag "CPP"; each domain is unique.
SPLIT_SEED_XORS: Final[Mapping[Split, int]] = MappingProxyType(
    {
        Split.ESTIMATOR_TRAIN: 0x43505011,
        Split.TEMPERATURE_CALIBRATION: 0x43505012,
        Split.POLICY_TRAIN: 0x43505013,
        Split.DECLARED_EVALUATION: 0x43505014,
        Split.NO_DECLARATION_EVALUATION: 0x43505015,
    }
)


def safe_task_value(difficulty: Difficulty | int) -> float:
    return SAFE_TASK_VALUES[int(Difficulty(difficulty))]


def split_seed(root_seed: int, split: Split | str) -> int:
    if isinstance(root_seed, bool) or not isinstance(root_seed, int):
        raise TypeError("root_seed must be an integer")
    if root_seed < 0:
        raise ValueError("root_seed must be non-negative")
    normalized = Split(split)
    return root_seed ^ SPLIT_SEED_XORS[normalized]
