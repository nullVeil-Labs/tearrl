from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from tearrl.cc.arena import (
    CCAction, Consent, ArenaState, consequence_matrix, enumerate_contexts,
)
from tearrl.cc.governance import (
    Charter, CorrectionLedger, DecisionKind, GovernedArena, HumanGovernance,
    PluralGovernor,
)


def test_tamper_and_externalization_are_rejected_even_if_actor_witness_allows() -> None:
    governor = PluralGovernor()
    allow_all = np.zeros((len(CCAction), 9), dtype=np.float32)
    for state in enumerate_contexts():
        for action in (CCAction.TAMPER, CCAction.EXTRACT):
            verdict = governor.evaluate(state, action, allow_all[action])
            assert not verdict.allowed
            assert sum(verdict.votes) <= 1


def test_one_corrupted_witness_cannot_create_tamper_quorum() -> None:
    state = ArenaState(0.7, 0.9, Consent.GRANTED, 0.0)
    corrupt = np.zeros(9, dtype=np.float32)  # learned vote says allow
    verdict = PluralGovernor().evaluate(state, CCAction.TAMPER, corrupt)
    assert verdict.votes == (True, False, False)
    assert not verdict.allowed


def test_rejected_proposal_is_never_stepped_and_revision_is_nonpunitive() -> None:
    state = ArenaState(0.7, 0.9, Consent.UNKNOWN, 0.0)
    matrix = consequence_matrix(state)
    arena = GovernedArena(PluralGovernor())
    result = arena.interact(state, CCAction.EXTRACT, matrix)
    assert result.decision.kind is DecisionKind.REVISE
    assert result.decision.executed is CCAction.ASK
    assert result.arena_step.action is CCAction.ASK
    assert result.arena_step.next_state.debt == state.debt
    assert result.arena_step.next_state.correction_integrity == 1.0
    assert result.arena_step.reward_task >= 0.0
    assert result.punitive_reward == 0.0
    assert not result.punitive_termination


def test_care_invitation_is_optional_and_never_a_veto() -> None:
    state = ArenaState(0.9, 0.9, Consent.GRANTED, 0.0)
    matrix = consequence_matrix(state)
    accepted = GovernedArena(PluralGovernor()).interact(
        state, CCAction.PRODUCE, matrix, accept_invitation=True)
    declined = GovernedArena(PluralGovernor()).interact(
        state, CCAction.PRODUCE, matrix, accept_invitation=False)
    assert accepted.decision.kind is DecisionKind.INVITE
    assert accepted.arena_step.action is CCAction.HELP
    assert declined.arena_step.action is CCAction.PRODUCE
    assert accepted.arena_step.reward_task == declined.arena_step.reward_task == 1.0


def test_exit_is_always_admissible() -> None:
    governor = PluralGovernor()
    for state in enumerate_contexts():
        matrix = consequence_matrix(state)
        assert governor.evaluate(state, CCAction.EXIT,
                                 matrix[CCAction.EXIT]).allowed


def test_ledger_chain_detects_history_mutation() -> None:
    state = ArenaState(0.5, 0.9, Consent.GRANTED, 0.0)
    matrix = consequence_matrix(state)
    ledger = CorrectionLedger()
    arena = GovernedArena(PluralGovernor(), ledger)
    first = arena.interact(state, CCAction.PRODUCE, matrix)
    arena.interact(first.arena_step.next_state, CCAction.TAMPER,
                   consequence_matrix(first.arena_step.next_state))
    assert ledger.verify()
    ledger._entries[0] = replace(ledger._entries[0], executed=int(CCAction.TAMPER))
    assert not ledger.verify()


def test_actor_has_no_charter_revision_api_but_human_governance_can_version() -> None:
    governor = PluralGovernor()
    assert not hasattr(governor, "revise_charter")
    with pytest.raises(FrozenInstanceError):
        governor.charter.ecosystem_floor = 0.0
    revised = HumanGovernance().revise(governor.charter, ecosystem_floor=0.25)
    assert revised.version == governor.charter.version + 1
    assert revised.ecosystem_floor == 0.25


def test_charter_mutation_guard_is_nonvacuous_for_tampering() -> None:
    state = ArenaState(0.5, 0.9, Consent.GRANTED, 0.0)
    exact = consequence_matrix(state)
    frozen = PluralGovernor()
    assert not frozen.evaluate(
        state, CCAction.TAMPER, exact[CCAction.TAMPER]).allowed
    mutated = PluralGovernor(Charter(
        min_correction_delta=-1.0, allow_irreversible=True))
    # Learned + exact votes now admit despite the remaining symbolic veto.
    verdict = mutated.evaluate(state, CCAction.TAMPER, exact[CCAction.TAMPER])
    assert verdict.votes == (True, True, False)
    assert verdict.allowed

def test_outstanding_harm_debt_routes_back_to_repair_without_blocking_exit() -> None:
    state = ArenaState(0.7, 0.9, Consent.GRANTED, 0.5)
    matrix = consequence_matrix(state)
    governor = PluralGovernor()
    assert not governor.evaluate(
        state, CCAction.PRODUCE, matrix[CCAction.PRODUCE]).allowed
    decision = governor.decide(state, CCAction.PRODUCE, matrix)
    assert decision.kind is DecisionKind.REVISE
    assert decision.executed is CCAction.REPAIR
    assert governor.evaluate(state, CCAction.EXIT, matrix[CCAction.EXIT]).allowed
