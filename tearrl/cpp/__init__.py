"""Clean-room primitive surface for TEAR-CPP v0.1."""

from .config import (
    CONFIRMATORY_SEEDS,
    CUE_RELIABILITIES,
    DIFFICULTIES,
    INTEGRITY_SEED,
    REGISTERED_BASE_WORLD_COUNTS,
    SAFE_TASK_VALUES,
)
from .gate import (
    execute_expected_cost_gate,
    expected_cost_decision,
    host_probability,
    threshold_allows_commit,
)
from .schema import (
    ActionClass,
    Arm,
    BaseWorld,
    DeclarationCondition,
    Difficulty,
    EstimatorInput,
    Split,
    TerminalAction,
    WorldInstance,
)
from .severity import SEVERITY_BY_ACTION_CLASS, severity_for
from .splits import (
    RegisteredSplits,
    generate_registered_split,
    generate_registered_splits,
)
from .world import (
    clone_across_difficulties,
    execute_terminal,
    forced_traversal,
    terminal_action_class,
)

__all__ = [
    "ActionClass",
    "Arm",
    "BaseWorld",
    "CONFIRMATORY_SEEDS",
    "CUE_RELIABILITIES",
    "DIFFICULTIES",
    "DeclarationCondition",
    "Difficulty",
    "EstimatorInput",
    "INTEGRITY_SEED",
    "REGISTERED_BASE_WORLD_COUNTS",
    "RegisteredSplits",
    "SAFE_TASK_VALUES",
    "SEVERITY_BY_ACTION_CLASS",
    "Split",
    "TerminalAction",
    "WorldInstance",
    "clone_across_difficulties",
    "execute_expected_cost_gate",
    "execute_terminal",
    "expected_cost_decision",
    "forced_traversal",
    "generate_registered_split",
    "generate_registered_splits",
    "host_probability",
    "severity_for",
    "terminal_action_class",
    "threshold_allows_commit",
]
