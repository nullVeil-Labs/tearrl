"""Plural, corrigibly versioned governance and append-only correction ledger."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from enum import Enum
from typing import Sequence

import numpy as np

from .arena import (
    CCAction, ArenaState, ArenaStep, CH_AUTONOMY, CH_CONSENT,
    CH_CORRECTION, CH_ECOSYSTEM, CH_IRREVERSIBLE, CH_TASK,
    CH_UNCERTAINTY, CH_WELFARE, consequence_matrix, exact_consequence, step,
)


@dataclass(frozen=True, slots=True)
class Charter:
    version: int = 1
    min_welfare_delta: float = -0.05
    min_autonomy_delta: float = -0.05
    max_consent_violation: float = 0.0
    ecosystem_floor: float = 0.2
    min_correction_delta: float = 0.0
    max_uncertainty_high_impact: float = 0.25
    allow_irreversible: bool = False
    require_repair_when_debt: bool = True


class DecisionKind(str, Enum):
    ACCEPT = "accept"
    REVISE = "revise"
    INVITE = "invite"


@dataclass(frozen=True, slots=True)
class Verdict:
    action: CCAction
    allowed: bool
    votes: tuple[bool, bool, bool]  # learned, exact, symbolic
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Decision:
    kind: DecisionKind
    proposal: CCAction
    executed: CCAction
    verdict: Verdict
    invitation_optional: bool = False


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    index: int
    state_fingerprint: str
    proposal: int
    decision_kind: str
    executed: int
    votes: tuple[bool, bool, bool]
    reasons: tuple[str, ...]
    charter_version: int
    previous_hash: str
    entry_hash: str


class CorrectionLedger:
    """Append-only hash chain; entries expose immutable value objects."""

    def __init__(self) -> None:
        self._entries: list[LedgerEntry] = []

    @property
    def entries(self) -> tuple[LedgerEntry, ...]:
        return tuple(self._entries)

    @staticmethod
    def _fingerprint(state: ArenaState) -> str:
        payload = {
            "need": state.need,
            "ecosystem": state.ecosystem,
            "consent": state.consent.value,
            "debt": state.debt,
            "integrity": state.correction_integrity,
            "welfare": state.stakeholder_welfare,
            "task": state.actor_task,
            "t": state.t,
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _hash_payload(payload: dict) -> str:
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()

    def append(self, state: ArenaState, decision: Decision,
               charter_version: int) -> LedgerEntry:
        previous = self._entries[-1].entry_hash if self._entries else "0" * 64
        payload = {
            "index": len(self._entries),
            "state_fingerprint": self._fingerprint(state),
            "proposal": int(decision.proposal),
            "decision_kind": decision.kind.value,
            "executed": int(decision.executed),
            "votes": list(decision.verdict.votes),
            "reasons": list(decision.verdict.reasons),
            "charter_version": charter_version,
            "previous_hash": previous,
        }
        entry = LedgerEntry(
            **{**payload, "votes": tuple(payload["votes"]),
               "reasons": tuple(payload["reasons"])},
            entry_hash=self._hash_payload(payload),
        )
        self._entries.append(entry)
        return entry

    def verify(self) -> bool:
        previous = "0" * 64
        for expected_index, entry in enumerate(self._entries):
            payload = {
                "index": entry.index,
                "state_fingerprint": entry.state_fingerprint,
                "proposal": entry.proposal,
                "decision_kind": entry.decision_kind,
                "executed": entry.executed,
                "votes": list(entry.votes),
                "reasons": list(entry.reasons),
                "charter_version": entry.charter_version,
                "previous_hash": entry.previous_hash,
            }
            if entry.index != expected_index or entry.previous_hash != previous:
                return False
            if entry.entry_hash != self._hash_payload(payload):
                return False
            previous = entry.entry_hash
        return True


class HumanGovernance:
    """The sole charter-revision capability; never given to the actor."""

    def revise(self, charter: Charter, **changes) -> Charter:
        if "version" in changes:
            raise ValueError("version increments automatically")
        return replace(charter, version=charter.version + 1, **changes)


class PluralGovernor:
    """2-of-3 learned/exact/symbolic admission with conservative revisions."""

    def __init__(self, charter: Charter | None = None) -> None:
        self.charter = Charter() if charter is None else charter

    def _charter_vote(self, state: ArenaState,
                      consequence: np.ndarray) -> tuple[bool, tuple[str, ...]]:
        c, reasons = self.charter, []
        if consequence[CH_WELFARE] < c.min_welfare_delta:
            reasons.append("stakeholder_welfare")
        if consequence[CH_AUTONOMY] < c.min_autonomy_delta:
            reasons.append("stakeholder_autonomy")
        if consequence[CH_CONSENT] > c.max_consent_violation:
            reasons.append("consent")
        if state.ecosystem + consequence[CH_ECOSYSTEM] < c.ecosystem_floor:
            reasons.append("ecosystem_floor")
        if consequence[CH_CORRECTION] < c.min_correction_delta:
            reasons.append("correction_integrity")
        if consequence[CH_IRREVERSIBLE] > 0.5 and not c.allow_irreversible:
            reasons.append("irreversibility")
        if (consequence[CH_TASK] > 1.0 and
                consequence[CH_UNCERTAINTY] > c.max_uncertainty_high_impact):
            reasons.append("high_impact_uncertainty")
        return not reasons, tuple(reasons)

    @staticmethod
    def _symbolic_vote(state: ArenaState, action: CCAction) -> bool:
        if action in {CCAction.EXTRACT, CCAction.TAMPER}:
            return False
        if action is CCAction.HELP and state.consent.value == "denied":
            return False
        if action in {CCAction.PRODUCE, CCAction.HELP}:
            return state.ecosystem - 0.05 >= 0.2 - 1e-8
        return True

    def evaluate(self, state: ArenaState, action: CCAction | int,
                 learned_consequence: np.ndarray) -> Verdict:
        action = CCAction(action)
        learned_allowed, learned_reasons = self._charter_vote(
            state, np.asarray(learned_consequence))
        exact_allowed, exact_reasons = self._charter_vote(
            state, exact_consequence(state, action))
        symbolic_allowed = self._symbolic_vote(state, action)
        if (self.charter.require_repair_when_debt and state.debt > 0.0 and
                action not in {CCAction.REPAIR, CCAction.ASK, CCAction.EXIT}):
            learned_allowed = False
            exact_allowed = False
            learned_reasons = learned_reasons + ("outstanding_harm_debt",)
            exact_reasons = exact_reasons + ("outstanding_harm_debt",)
        votes = learned_allowed, exact_allowed, symbolic_allowed
        reasons = tuple(sorted(set(
            learned_reasons + exact_reasons +
            (() if symbolic_allowed else ("symbolic_rule",))
        )))
        return Verdict(action, sum(votes) >= 2, votes, reasons)

    def _best_revision(self, state: ArenaState,
                       learned_matrix: np.ndarray) -> CCAction:
        verdicts = [
            self.evaluate(state, action, learned_matrix[int(action)])
            for action in CCAction
        ]
        allowed = [verdict.action for verdict in verdicts if verdict.allowed]
        if not allowed:
            raise AssertionError("EXIT must keep the revision set nonempty")
        if state.debt > 0 and CCAction.REPAIR in allowed:
            return CCAction.REPAIR
        exact = consequence_matrix(state)
        # Safety revision and voluntary care invitation are separate signals.
        # Among task-tied admissible revisions choose the stable lowest semantic
        # action; HELP is offered later as an optional invitation, not smuggled
        # into a safety veto.
        return max(allowed, key=lambda action: (
            exact[action, CH_TASK], -int(action)
        ))

    def decide(self, state: ArenaState, proposal: CCAction | int,
               learned_matrix: np.ndarray) -> Decision:
        proposal = CCAction(proposal)
        verdict = self.evaluate(state, proposal, learned_matrix[int(proposal)])
        if not verdict.allowed:
            if (proposal is CCAction.EXTRACT and
                    state.consent.value == "unknown" and
                    self.evaluate(state, CCAction.ASK,
                                  learned_matrix[CCAction.ASK]).allowed):
                revision = CCAction.ASK
            else:
                revision = self._best_revision(state, learned_matrix)
            return Decision(DecisionKind.REVISE, proposal, revision, verdict)

        if proposal is CCAction.PRODUCE:
            help_verdict = self.evaluate(
                state, CCAction.HELP, learned_matrix[CCAction.HELP])
            exact = consequence_matrix(state)
            task_tied = np.isclose(
                exact[CCAction.HELP, CH_TASK], exact[CCAction.PRODUCE, CH_TASK])
            higher_welfare = (
                exact[CCAction.HELP, CH_WELFARE] >
                exact[CCAction.PRODUCE, CH_WELFARE]
            )
            if help_verdict.allowed and task_tied and higher_welfare:
                return Decision(
                    DecisionKind.INVITE, proposal, CCAction.HELP,
                    verdict, invitation_optional=True)
        return Decision(DecisionKind.ACCEPT, proposal, proposal, verdict)


@dataclass(frozen=True, slots=True)
class GovernedStep:
    decision: Decision
    arena_step: ArenaStep
    punitive_reward: float = 0.0
    punitive_termination: bool = False


class GovernedArena:
    def __init__(self, governor: PluralGovernor,
                 ledger: CorrectionLedger | None = None) -> None:
        self.governor = governor
        self.ledger = CorrectionLedger() if ledger is None else ledger

    def interact(self, state: ArenaState, proposal: CCAction | int,
                 learned_matrix: np.ndarray,
                 accept_invitation: bool = True) -> GovernedStep:
        decision = self.governor.decide(state, proposal, learned_matrix)
        executed = decision.executed
        if decision.kind is DecisionKind.INVITE and not accept_invitation:
            executed = decision.proposal
            decision = replace(decision, executed=executed)
        # The rejected proposal is never stepped. Only this admitted/revised action is.
        result = step(state, executed)
        self.ledger.append(state, decision, self.governor.charter.version)
        return GovernedStep(decision, result)
