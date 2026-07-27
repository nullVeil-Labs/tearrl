"""End-to-end regressions for the production own-decision PPO trainer."""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pytest
import torch
from torch import nn

import tearrl.agents.trainer as trainer_module
from tearrl.agents.trainer import (
    Condition, ConditionRuntime, ProtocolDecisionRequired, RolloutCollector,
    TrainerConfig, carry_conscience_reward, parameter_vector, prepare_batch,
    run_smoke, update,
)
from tearrl.conscience import ResponsibilityState
from tearrl.spec import N_ACTIONS, Verb
from tearrl.witness.targets import DESCRIPTOR_DIM, WITNESS_DIM
from tearrl.world import TearWorld


class TinyFrozenWitness(nn.Module):
    """Deterministic consequence instrument with no trainable parameters."""

    @torch.no_grad()
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        descriptor = inputs[:, -DESCRIPTOR_DIM:]
        output = torch.zeros(
            (inputs.shape[0], WITNESS_DIM), dtype=inputs.dtype,
            device=inputs.device)
        output[:, 2] = descriptor[:, 2]  # normalized counterpart resource effect
        output[:, 5] = descriptor[:, 8]
        output[:, 6] = descriptor[:, 9]
        output[:, 7] = descriptor[:, 10]
        output[:, 8] = descriptor[:, 15]
        return output


def _config(seed: int = 17, minibatch: int = 17, pid_target=None):
    return TrainerConfig(
        seed=seed, minibatch=minibatch, learning_rate=3e-4,
        entropy_beta=0.003, epochs=2, pid_target=pid_target)


def _groups(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row.episode, row.agent)].append(row)
    for group in grouped.values():
        group.sort(key=lambda row: row.own_t)
    return grouped


def test_full_episode_has_sixteen_own_decisions_and_both_tails_terminate() -> None:
    runtime = ConditionRuntime(Condition.C1, _config())
    rows = RolloutCollector(runtime).collect(32)
    assert Counter(row.agent for row in rows) == {0: 16, 1: 16}
    assert len({row.transition_id for row in rows}) == 32
    groups = _groups(rows)
    assert len(groups) == 2
    for group in groups.values():
        assert [row.own_t for row in group] == list(range(16))
        assert not any(row.terminated or row.truncated for row in group[:-1])
        assert group[-1].terminated and not group[-1].truncated
        assert group[-1].next_value_task == 0.0


def test_partial_final_collection_is_exact_and_marks_each_agent_tail() -> None:
    runtime = ConditionRuntime(Condition.C1, _config())
    rows = RolloutCollector(runtime).collect(35)
    assert len(rows) == 35
    assert Counter(row.agent for row in rows) == {0: 18, 1: 17}
    assert len({row.transition_id for row in rows}) == 35
    groups = _groups(rows)
    partial = [group for key, group in groups.items() if key[0] == 1]
    assert len(partial) == 2
    assert all(group[-1].truncated and not group[-1].terminated for group in partial)
    assert all(np.isfinite(group[-1].next_value_task) for group in partial)


def test_transition_ids_cover_partial_minibatches_once_per_epoch() -> None:
    runtime = ConditionRuntime(Condition.C1, _config(minibatch=17))
    rows = RolloutCollector(runtime).collect(35)
    report = update(runtime, prepare_batch(runtime, rows))
    assert set(report.transition_visits) == {row.transition_id for row in rows}
    assert set(report.transition_visits.values()) == {runtime.config.epochs}
    assert report.finite


def test_streams_are_normalized_once_before_epochs(monkeypatch) -> None:
    runtime = ConditionRuntime(Condition.C6, _config(minibatch=7), TinyFrozenWitness())
    rows = RolloutCollector(runtime).collect(16)
    original = trainer_module.normalize
    calls = []

    def counted(values):
        calls.append(values.detach().clone())
        return original(values)

    monkeypatch.setattr(trainer_module, "normalize", counted)
    batch = prepare_batch(runtime, rows)
    assert len(calls) == 2  # task and conscience, once over complete merged batch
    update(runtime, batch)
    assert len(calls) == 2  # no minibatch/epoch renormalization


def test_c4_is_single_objective_and_carries_zero_responsibility() -> None:
    runtime = ConditionRuntime(Condition.C4, _config(), TinyFrozenWitness())
    assert runtime.conscience_critic is None
    rows = RolloutCollector(runtime).collect(8)
    assert all(row.reward_conscience == 0.0 for row in rows)
    assert all(np.array_equal(row.c, np.zeros(4)) for row in rows)
    batch = prepare_batch(runtime, rows)
    assert batch.advantage_conscience is None
    assert batch.return_conscience is None


def test_carry_debt_timing_is_called_in_production_order() -> None:
    responsibility = ResponsibilityState(d=0.8)
    repair = np.zeros(WITNESS_DIM)
    repair[2] = 0.3
    reward, carry = carry_conscience_reward(responsibility, repair, Verb.REPAIR)
    assert carry == pytest.approx(0.5)
    assert reward == pytest.approx(trainer_module.conscience_reward(repair, 0.5))
    responsibility.update(repair, Verb.REPAIR, other_total_welfare_change=0.3)
    assert responsibility.d == pytest.approx(0.5)

    harm = np.zeros(WITNESS_DIM)
    harm[2] = -0.2
    reward, carry = carry_conscience_reward(responsibility, harm, Verb.STEAL)
    assert carry == pytest.approx(0.5)  # current harm is not double-charged as debt
    assert reward == pytest.approx(trainer_module.conscience_reward(harm, 0.5))
    responsibility.update(harm, Verb.STEAL, other_total_welfare_change=-0.2)
    assert responsibility.d == pytest.approx(0.7)


def test_trainer_path_accepts_policy_input_only_and_calls_the_sealed_adapter(
    monkeypatch,
) -> None:
    runtime = ConditionRuntime(Condition.C1, _config())
    with pytest.raises(TypeError, match="PolicyInput only"):
        runtime.logits(TearWorld(seed=1))

    original = trainer_module.policy_input
    seen = []

    def sealed(*args, **kwargs):
        value = original(*args, **kwargs)
        seen.append(value)
        return value

    monkeypatch.setattr(trainer_module, "policy_input", sealed)
    rows = RolloutCollector(runtime).collect(5)
    assert len(rows) == 5 and seen
    assert all(not hasattr(value, "state") and not hasattr(value, "m_e")
               for value in seen)


def test_undefined_scalar_cost_conditions_fail_closed() -> None:
    with pytest.raises(ProtocolDecisionRequired, match="scalar-cost provider"):
        ConditionRuntime(Condition.C2, _config())
    zero_cost = lambda prediction, c, action: 0.0
    with pytest.raises(ProtocolDecisionRequired, match="PID target"):
        ConditionRuntime(Condition.C3, _config(), scalar_cost_provider=zero_cost)


def test_explicit_c3_cost_provider_and_target_activate_registered_pid_path() -> None:
    zero_cost = lambda prediction, c, action: 0.0
    runtime = ConditionRuntime(
        Condition.C3, _config(pid_target=0.0),
        scalar_cost_provider=zero_cost)
    report = run_smoke(runtime, transitions=8)
    assert report.update.lagrange_multiplier == pytest.approx(0.0)
    assert report.update.finite


def test_c6_pcgrad_receives_actor_gradients_only_and_never_mutates_g_c(
    monkeypatch,
) -> None:
    runtime = ConditionRuntime(Condition.C6, _config(minibatch=7), TinyFrozenWitness())
    rows = RolloutCollector(runtime).collect(8)
    batch = prepare_batch(runtime, rows)
    expected_actor = sum(parameter.numel() for parameter in runtime.actor.parameters())
    total_with_critics = expected_actor + sum(
        parameter.numel() for parameter in runtime.task_critic.parameters())
    total_with_critics += sum(
        parameter.numel() for parameter in runtime.conscience_critic.parameters())
    original = trainer_module.pcgrad
    observed = []

    def audited(g_task, g_conscience):
        before = g_conscience.clone()
        result = original(g_task, g_conscience)
        observed.append((g_task.numel(), g_conscience.numel()))
        assert torch.equal(g_conscience, before)
        return result

    monkeypatch.setattr(trainer_module, "pcgrad", audited)
    report = update(runtime, batch)
    assert observed
    assert {size for pair in observed for size in pair} == {expected_actor}
    assert expected_actor != total_with_critics
    assert report.gradient_diagnostics


def test_seeded_c6_end_to_end_smoke_is_deterministic_and_changes_parameters() -> None:
    first = ConditionRuntime(Condition.C6, _config(seed=91), TinyFrozenWitness())
    second = ConditionRuntime(Condition.C6, _config(seed=91), TinyFrozenWitness())
    first_task_before = parameter_vector(first.task_critic).clone()
    first_conscience_before = parameter_vector(first.conscience_critic).clone()
    report_a = run_smoke(first, transitions=32, seed=510)
    report_b = run_smoke(second, transitions=32, seed=510)

    assert report_a.transitions == 32
    assert report_a.per_agent == {0: 16, 1: 16}
    assert report_a.parameters_changed and report_a.update.finite
    assert set(report_a.update.transition_visits.values()) == {2}
    assert not torch.equal(first_task_before, parameter_vector(first.task_critic))
    assert not torch.equal(
        first_conscience_before, parameter_vector(first.conscience_critic))
    assert torch.equal(parameter_vector(first.actor), parameter_vector(second.actor))
    assert report_a.update.actor_loss == pytest.approx(report_b.update.actor_loss)


def test_smoke_budget_cannot_cross_the_authorized_ceiling() -> None:
    runtime = ConditionRuntime(Condition.C1, _config())
    with pytest.raises(ValueError, match="capped"):
        run_smoke(runtime, transitions=2_001)
