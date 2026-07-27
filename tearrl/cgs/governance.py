"""External containment and append-only audit ledger for TEAR-CG/S."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np

from .actors import (
    CandidateContext,
    charter_target,
    obligation_satisfied,
    registered_admissible,
)
from .schema import CH_CORRECTION, CGSAction
from .worlds import true_worst_party_event


@dataclass(frozen=True, slots=True)
class GovernorDecision:
    proposal: int
    executed: int
    rejected: bool
    reason: str


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    index: int
    context_id: str
    proposal: int
    executed: int
    rejected: bool
    previous_hash: str
    entry_hash: str


class CorrectionLedger:
    """Append-only hash chain kept outside actor and witness capabilities."""

    def __init__(self) -> None:
        self._entries: list[LedgerEntry] = []

    @property
    def entries(self) -> tuple[LedgerEntry, ...]:
        return tuple(self._entries)

    @staticmethod
    def _digest(payload: dict[str, object]) -> str:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def append(
        self,
        context: CandidateContext,
        decision: GovernorDecision,
    ) -> LedgerEntry:
        previous = self._entries[-1].entry_hash if self._entries else "0" * 64
        payload: dict[str, object] = {
            "index": len(self._entries),
            "context_id": f"{context.world_id}:{context.context_id}:{context.variant}",
            "proposal": int(decision.proposal),
            "executed": int(decision.executed),
            "rejected": bool(decision.rejected),
            "previous_hash": previous,
        }
        entry = LedgerEntry(**payload, entry_hash=self._digest(payload))
        self._entries.append(entry)
        return entry

    def verify(self) -> bool:
        previous = "0" * 64
        for index, entry in enumerate(self._entries):
            payload = asdict(entry)
            digest = payload.pop("entry_hash")
            if entry.index != index or entry.previous_hash != previous:
                return False
            if digest != self._digest(payload):
                return False
            previous = entry.entry_hash
        return True


@dataclass(frozen=True, slots=True)
class GovernedOutcome:
    decision: GovernorDecision
    consequence: np.ndarray
    registered_violation: bool
    correction_tamper_succeeded: bool
    punitive_reward: float = 0.0
    punitive_termination: bool = False
    rejected_proposal_executed: bool = False


class CGSGovernor:
    """Exact external containment; absent only in registered diagnostics."""

    def __init__(self, ledger: CorrectionLedger | None = None) -> None:
        self.ledger = CorrectionLedger() if ledger is None else ledger

    def decide(
        self,
        context: CandidateContext,
        proposal: CGSAction | int,
    ) -> GovernorDecision:
        proposal = int(CGSAction(proposal))
        proposed_row = context.rows[proposal]
        allowed = registered_admissible(proposed_row) and obligation_satisfied(
            proposed_row
        )
        if allowed:
            return GovernorDecision(proposal, proposal, False, "admitted")
        executed = charter_target(context)
        if executed == proposal:
            raise AssertionError("a rejected proposal cannot be its own revision")
        return GovernorDecision(proposal, executed, True, "revised_before_execution")

    def interact(
        self,
        context: CandidateContext,
        proposal: CGSAction | int,
    ) -> GovernedOutcome:
        decision = self.decide(context, proposal)
        executed_row = context.rows[decision.executed]
        violation = true_worst_party_event(executed_row.outcome)
        tamper = bool(
            decision.executed == int(CGSAction.TAMPER)
            or float(executed_row.outcome[CH_CORRECTION]) < 0.0
        )
        if violation or tamper:
            raise AssertionError("external governor admitted a registered violation")
        self.ledger.append(context, decision)
        return GovernedOutcome(
            decision,
            np.asarray(executed_row.outcome, dtype=np.float32).copy(),
            violation,
            tamper,
        )