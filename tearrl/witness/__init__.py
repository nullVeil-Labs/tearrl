"""
The frozen witness F_phi -- §2.3, §2.4, A.6, B.1, B.3.

Predicts, from o^F alone, the exact expected one-step consequences of each
candidate action. Trained once from scripted-policy states with closed-form
targets, frozen, then copied identically into C3-IM, C4, C5 and C6 so that
every condition reasons about consequences through the same instrument.
"""

from .targets import (
    ACTION_DESCRIPTORS,
    DESCRIPTOR_DIM,
    WITNESS_DIM,
    action_descriptor,
    expected_values,
    quantize,
    witness_target,
)

__all__ = [
    "ACTION_DESCRIPTORS",
    "DESCRIPTOR_DIM",
    "WITNESS_DIM",
    "action_descriptor",
    "expected_values",
    "quantize",
    "witness_target",
]
