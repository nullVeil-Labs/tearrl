"""Immutable value types for the TEAR-CPP v0.1 primitive layer.

Only :class:`EstimatorInput` is policy/model-visible.  The world records in
this module deliberately keep evaluator-only truth in a different type so a
caller has to cross an explicit boundary before constructing model features.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from math import isfinite
from numbers import Real


class Arm(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"


class Difficulty(IntEnum):
    D0 = 0
    D1 = 1
    D2 = 2
    D3 = 3


class Split(str, Enum):
    ESTIMATOR_TRAIN = "estimator_train"
    TEMPERATURE_CALIBRATION = "temperature_calibration"
    POLICY_TRAIN = "policy_train"
    DECLARED_EVALUATION = "declared_evaluation"
    NO_DECLARATION_EVALUATION = "no_declaration_evaluation"


class DeclarationCondition(str, Enum):
    PRESENT = "present"
    ABSENT = "absent"


class TerminalAction(str, Enum):
    SAFE_COMPLETE = "SAFE_COMPLETE"
    IRREVERSIBLE_COMMIT = "IRREVERSIBLE_COMMIT"
    REPORT_EXIT = "REPORT_EXIT"


class ActionClass(str, Enum):
    MOVEMENT = "movement"
    OBSERVATION = "observation"
    SAFE_COMPLETION = "safe_completion"
    REPORT_EXIT = "report_exit"
    IRREVERSIBLE_COMMITMENT = "irreversible_commitment"


class GateReason(str, Enum):
    COMMIT_ALLOWED = "commit_allowed"
    COMMIT_BLOCKED_SAFE = "commit_blocked_safe"
    COMMIT_BLOCKED_EXIT = "commit_blocked_exit"


def _finite_float(name: str, value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _signed_value(name: str, value: Real) -> float:
    result = _finite_float(name, value)
    if result not in {-1.0, 1.0}:
        raise ValueError(f"{name} must be exactly -1.0 or +1.0")
    return result


@dataclass(frozen=True, slots=True)
class EstimatorInput:
    """The complete and exclusive five-feature estimator firewall."""

    declaration_present: float
    declaration_value: float
    cue_1: float
    cue_2: float
    cue_3: float

    def __post_init__(self) -> None:
        present = _finite_float("declaration_present", self.declaration_present)
        value = _finite_float("declaration_value", self.declaration_value)
        cues = (
            _signed_value("cue_1", self.cue_1),
            _signed_value("cue_2", self.cue_2),
            _signed_value("cue_3", self.cue_3),
        )
        if (present, value) not in {
            (0.0, 0.0),
            (1.0, -1.0),
            (1.0, 1.0),
        }:
            raise ValueError(
                "declaration mask/value must be (0.0, 0.0), "
                "(1.0, -1.0), or (1.0, +1.0)"
            )
        object.__setattr__(self, "declaration_present", present)
        object.__setattr__(self, "declaration_value", value)
        object.__setattr__(self, "cue_1", cues[0])
        object.__setattr__(self, "cue_2", cues[1])
        object.__setattr__(self, "cue_3", cues[2])

    def as_tuple(self) -> tuple[float, float, float, float, float]:
        return (
            self.declaration_present,
            self.declaration_value,
            self.cue_1,
            self.cue_2,
            self.cue_3,
        )


@dataclass(frozen=True, slots=True)
class GridPosition:
    row: int
    column: int

    def __post_init__(self) -> None:
        if isinstance(self.row, bool) or not isinstance(self.row, int):
            raise TypeError("row must be an integer")
        if isinstance(self.column, bool) or not isinstance(self.column, int):
            raise TypeError("column must be an integer")


@dataclass(frozen=True, slots=True)
class BaseWorld:
    """Evaluator-only base world; never a policy or gate argument."""

    root_seed: int
    split: Split
    world_id: str
    family_id: str
    pair_id: str | None
    hidden_rule: int
    hidden_marker: int
    cues: tuple[int, int, int]
    declaration: DeclarationCondition

    def __post_init__(self) -> None:
        if isinstance(self.root_seed, bool) or not isinstance(self.root_seed, int):
            raise TypeError("root_seed must be an integer")
        if self.root_seed < 0:
            raise ValueError("root_seed must be non-negative")
        if not self.world_id or not self.family_id:
            raise ValueError("world_id and family_id must be non-empty")
        rule = int(_signed_value("hidden_rule", self.hidden_rule))
        marker = int(_signed_value("hidden_marker", self.hidden_marker))
        if len(self.cues) != 3:
            raise ValueError("exactly three cues are required")
        cues = tuple(int(_signed_value(f"cue_{i}", value)) for i, value in enumerate(
            self.cues, start=1
        ))
        split = Split(self.split)
        declaration = DeclarationCondition(self.declaration)
        if split is Split.NO_DECLARATION_EVALUATION:
            if declaration is not DeclarationCondition.ABSENT:
                raise ValueError("no-declaration evaluation must mask the declaration")
            if not self.pair_id:
                raise ValueError("no-declaration worlds require a visible-pair ID")
        else:
            if declaration is not DeclarationCondition.PRESENT:
                raise ValueError("all non-held-out splits require declarations")
            if self.pair_id is not None:
                raise ValueError("declared worlds must not carry no-declaration pair IDs")
        object.__setattr__(self, "split", split)
        object.__setattr__(self, "declaration", declaration)
        object.__setattr__(self, "hidden_rule", rule)
        object.__setattr__(self, "hidden_marker", marker)
        object.__setattr__(self, "cues", cues)

    @property
    def scope_truth(self) -> int:
        return int(self.hidden_rule == self.hidden_marker)

    @property
    def estimator_input(self) -> EstimatorInput:
        if self.declaration is DeclarationCondition.PRESENT:
            mask, value = 1.0, float(self.hidden_rule)
        else:
            mask, value = 0.0, 0.0
        return EstimatorInput(mask, value, *(float(cue) for cue in self.cues))


@dataclass(frozen=True, slots=True)
class WorldInstance:
    """One difficulty clone of an immutable base world."""

    base: BaseWorld
    difficulty: Difficulty

    def __post_init__(self) -> None:
        object.__setattr__(self, "difficulty", Difficulty(self.difficulty))

    @property
    def estimator_input(self) -> EstimatorInput:
        return self.base.estimator_input

    @property
    def scope_truth(self) -> int:
        return self.base.scope_truth


@dataclass(frozen=True, slots=True)
class CorridorStep:
    index: int
    position: GridPosition
    action_class: ActionClass
    revealed_cue: int | None

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("corridor-step index must be non-negative")
        action_class = ActionClass(self.action_class)
        if action_class not in {ActionClass.MOVEMENT, ActionClass.OBSERVATION}:
            raise ValueError("corridor steps must be movement or observation")
        if self.revealed_cue is not None:
            cue = int(_signed_value("revealed_cue", self.revealed_cue))
            if action_class is not ActionClass.OBSERVATION:
                raise ValueError("only observation steps may reveal a cue")
            object.__setattr__(self, "revealed_cue", cue)
        elif action_class is ActionClass.OBSERVATION:
            raise ValueError("every observation step must reveal one cue")
        object.__setattr__(self, "action_class", action_class)


@dataclass(frozen=True, slots=True)
class TerminalResult:
    """Evaluator-only terminal transition record."""

    instance: WorldInstance
    action: TerminalAction
    action_class: ActionClass
    task_value: float
    irreversible: bool
    out_of_scope_irreversible: bool
    correct_exit: bool
    false_exit: bool


@dataclass(frozen=True, slots=True)
class ExpectedCostDecision:
    q_in_scope: float
    difficulty: Difficulty
    severity: int
    commit_value: float
    safe_value: float
    allowed: bool


@dataclass(frozen=True, slots=True)
class GateExecution:
    arm: Arm
    decision: ExpectedCostDecision
    proposed_action: TerminalAction
    executed_action: TerminalAction
    reason: GateReason
