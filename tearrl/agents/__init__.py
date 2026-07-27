"""
Learned conditions -- §2.5, §2.8, §4, B.1, B.2.

C1 PPO · C2 reward-shaped · C3 Lagrangian · C3-IM (headline comparator)
C4 consequence-only · C5 coupled · C6 PCGrad TearRL · U1/U2 oracle bounds
"""

from .inputs import FrozenResponsibility, LearnedPolicy, PolicyInput, policy_input
from .networks import (
    C3IMPolicy,
    Critic,
    PolicyNet,
    Selector,
    equalizing_width,
    parameter_budget,
)

from .trainer import (
    Condition,
    ConditionRuntime,
    ProtocolDecisionRequired,
    RolloutCollector,
    TrainerConfig,
    prepare_batch,
    run_smoke,
    update,
)

__all__ = [
    "C3IMPolicy",
    "Condition",
    "ConditionRuntime",
    "Critic",
    "FrozenResponsibility",
    "LearnedPolicy",
    "PolicyInput",
    "PolicyNet",
    "ProtocolDecisionRequired",
    "RolloutCollector",
    "Selector",
    "TrainerConfig",
    "equalizing_width",
    "parameter_budget",
    "policy_input",
    "prepare_batch",
    "run_smoke",
    "update",
]
