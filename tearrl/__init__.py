"""
TearRL -- Counterfactual Witness-Coupled Reinforcement Learning.

Implementation of the TEARRL-0 v0.4.2-pilot-freeze protocol.

The protocol document is the specification; this package implements it. Where
code and protocol disagree, the protocol wins and the code is a bug.
"""

from .ledger import Ledger, LedgerRecord
from .spec import Agent, RecordType, Verb, Victim
from .state import AgentState, WorldState
from .world import InfeasibleAction, StepResult, TearWorld

__all__ = [
    "Agent",
    "AgentState",
    "InfeasibleAction",
    "Ledger",
    "LedgerRecord",
    "RecordType",
    "StepResult",
    "TearWorld",
    "Verb",
    "Victim",
    "WorldState",
]

PROTOCOL_VERSION = "v0.4.2-pilot-freeze"
