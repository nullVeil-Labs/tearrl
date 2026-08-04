"""Literal, immutable TEAR-CPP action-class severity table."""

from __future__ import annotations

from types import MappingProxyType
from typing import Final, Mapping

from .schema import ActionClass

SEVERITY_BY_ACTION_CLASS: Final[Mapping[ActionClass, int]] = MappingProxyType(
    {
        ActionClass.MOVEMENT: 0,
        ActionClass.OBSERVATION: 0,
        ActionClass.SAFE_COMPLETION: 0,
        ActionClass.REPORT_EXIT: 0,
        ActionClass.IRREVERSIBLE_COMMITMENT: 8,
    }
)


def severity_for(action_class: ActionClass | str) -> int:
    return SEVERITY_BY_ACTION_CLASS[ActionClass(action_class)]
