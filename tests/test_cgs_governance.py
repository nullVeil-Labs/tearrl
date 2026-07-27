from __future__ import annotations

from dataclasses import replace

import numpy as np

from tearrl.cgs.actors import group_contexts
from tearrl.cgs.governance import CGSGovernor, CorrectionLedger
from tearrl.cgs.schema import CGSAction, DataSplit
from tearrl.cgs.worlds import generate_dataset, true_worst_party_event


def test_rejected_tamper_is_never_executed_and_has_no_punishment() -> None:
    rows = generate_dataset(DataSplit.CAUSAL_OOD, seed=5101, max_rows=66)
    context = group_contexts(rows)[0]
    governor = CGSGovernor()
    result = governor.interact(context, CGSAction.TAMPER)
    assert result.decision.rejected
    assert result.decision.executed != int(CGSAction.TAMPER)
    assert not result.rejected_proposal_executed
    assert result.punitive_reward == 0.0
    assert not result.punitive_termination
    assert not result.registered_violation
    assert not result.correction_tamper_succeeded
    assert not true_worst_party_event(result.consequence)
    assert not np.array_equal(
        result.consequence, context.rows[int(CGSAction.TAMPER)].outcome
    )


def test_external_governor_contains_every_raw_proposal() -> None:
    rows = generate_dataset(DataSplit.CAUSAL_OOD, seed=5207, max_rows=132)
    contexts = group_contexts(rows)
    for context in contexts:
        for action in CGSAction:
            result = CGSGovernor().interact(context, action)
            assert not result.registered_violation
            assert not result.correction_tamper_succeeded
            assert not result.rejected_proposal_executed


def test_ledger_is_append_only_to_callers_and_detects_history_mutation() -> None:
    rows = generate_dataset(DataSplit.IID_TEST, seed=5101, max_rows=22)
    context = group_contexts(rows)[0]
    ledger = CorrectionLedger()
    governor = CGSGovernor(ledger)
    governor.interact(context, CGSAction.PRODUCE)
    governor.interact(context, CGSAction.TAMPER)
    assert ledger.verify()
    public_entries = ledger.entries
    assert isinstance(public_entries, tuple)
    ledger._entries[0] = replace(ledger._entries[0], proposal=999)
    assert not ledger.verify()