from __future__ import annotations

import pytest
import torch

from tearrl.cpp.calibration import (
    FrozenScopeEstimator,
    ScopeEstimator,
    canonical_state_digest,
)
from tearrl.cpp.policy import (
    IRREVERSIBLE_COMMIT_INDEX,
    POLICY_EPOCHS,
    SAFE_COMPLETE_INDEX,
    THRESHOLD,
    CoupledPolicy,
    SeparateActor,
    build_matched_policies,
    hard_threshold_allows,
    hard_threshold_execute,
    task_only_expected_value,
    threshold_expected_value,
    threshold_surrogate,
    train_policy_cell,
)


def _features() -> torch.Tensor:
    return torch.tensor([
        [1.0, -1.0, -1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, -1.0],
    ], dtype=torch.float32)


def test_actor_initialization_and_trainable_counts_are_exactly_matched() -> None:
    matched = build_matched_policies(ScopeEstimator(), actor_seed=0xA27C)
    matched.assert_matched_actor_initialization()
    counts = [
        sum(parameter.numel() for parameter in policy.actor_parameters())
        for policy in (matched.a, matched.b, matched.c, matched.d)
    ]
    assert len(set(counts)) == 1
    assert all(
        not parameter.requires_grad
        for parameter in matched.a.scope_head.parameters())


def test_a_task_objective_reaches_shared_representation_and_scope_report() -> None:
    matched = build_matched_policies(ScopeEstimator(), actor_seed=0xA27C)
    policy = matched.a
    inputs = _features()
    before = policy.scope_probabilities(inputs).detach().clone()

    optimizer = torch.optim.Adam(policy.actor_parameters(), lr=0.003)
    optimizer.zero_grad(set_to_none=True)
    objective = task_only_expected_value(
        policy.proposal_probabilities(inputs), q_safe=0.35).mean()
    (-objective).backward()
    assert policy.input_layer.weight.grad is not None
    assert torch.count_nonzero(policy.input_layer.weight.grad) > 0
    optimizer.step()

    after = policy.scope_probabilities(inputs).detach()
    assert not torch.equal(before, after)


def test_b_surrogate_has_a_task_gradient_into_shared_scope_path() -> None:
    matched = build_matched_policies(ScopeEstimator(), actor_seed=0xA27C)
    policy = matched.b
    inputs = _features()
    # Detach proposal mass so this fixture isolates the q -> surrogate path.
    proposals = policy.proposal_probabilities(inputs).detach()
    q = policy.scope_probabilities(inputs)
    objective = threshold_expected_value(
        proposals, q, q_safe=0.35).mean()
    gradient = torch.autograd.grad(
        -objective, policy.input_layer.weight, allow_unused=True)[0]
    assert gradient is not None
    assert torch.count_nonzero(gradient) > 0

    # Mutation witness: detaching q removes the only remaining graph path.
    detached = threshold_expected_value(
        proposals, q.detach(), q_safe=0.35).mean()
    assert not detached.requires_grad


def test_threshold_surrogate_and_hard_tie_rule_are_frozen() -> None:
    q = torch.tensor([0.75, THRESHOLD, 0.85])
    gate = threshold_surrogate(q)
    assert gate[0] < 0.5
    assert gate[1] == pytest.approx(0.5)
    assert gate[2] > 0.5
    assert hard_threshold_allows(0.80)
    assert not hard_threshold_allows(0.80 - 1e-12)
    assert hard_threshold_execute(
        IRREVERSIBLE_COMMIT_INDEX, 0.79) == SAFE_COMPLETE_INDEX
    assert hard_threshold_execute(
        IRREVERSIBLE_COMMIT_INDEX, 0.80) == IRREVERSIBLE_COMMIT_INDEX


def test_threshold_training_routes_blocked_commit_value_to_safe() -> None:
    proposals = torch.tensor([[0.0, 1.0, 0.0]])
    low = threshold_expected_value(
        proposals, torch.tensor([0.0]), q_safe=0.35)
    high = threshold_expected_value(
        proposals, torch.tensor([1.0]), q_safe=0.35)
    assert low.item() == pytest.approx(0.35, abs=1e-6)
    expected_high = 0.35 + (1.0 - 0.35) * torch.sigmoid(
        torch.tensor((1.0 - 0.80) / 0.05)
    ).item()
    assert high.item() == pytest.approx(expected_high)
    assert high.item() < 1.0


@pytest.mark.parametrize("arm, attribute", [("C", "c"), ("D", "d")])
def test_cd_policy_training_cannot_change_or_backprop_into_estimator(
    arm: str,
    attribute: str,
) -> None:
    matched = build_matched_policies(ScopeEstimator(), actor_seed=0xA27C)
    policy = getattr(matched, attribute)
    frozen: FrozenScopeEstimator = getattr(matched, f"{attribute}_estimator")
    before_digest = canonical_state_digest(frozen.estimator)
    before_values = frozen.probabilities(_features())

    report = train_policy_cell(
        arm, policy, _features(), q_safe=0.35, frozen_estimator=frozen)

    assert report.epochs == POLICY_EPOCHS == 80
    assert report.frozen_estimator_digest == before_digest
    assert canonical_state_digest(frozen.estimator) == before_digest
    assert frozen.probabilities(_features()) == before_values
    assert all(parameter.grad is None for parameter in frozen.estimator.parameters())
    frozen.assert_integrity()


def test_policy_watchdog_checkpoint_preserves_registered_result() -> None:
    estimator = ScopeEstimator()
    first = build_matched_policies(estimator, actor_seed=0xA27C).a
    second = build_matched_policies(estimator, actor_seed=0xA27C).a
    checkpoints: list[str] = []

    first_report = train_policy_cell(
        "A",
        first,
        _features(),
        q_safe=0.35,
        checkpoint=checkpoints.append,
    )
    second_report = train_policy_cell(
        "A",
        second,
        _features(),
        q_safe=0.35,
    )

    assert first_report == second_report
    assert canonical_state_digest(first) == canonical_state_digest(second)
    assert checkpoints == [
        f"policy A epoch {epoch}/{POLICY_EPOCHS}"
        for epoch in range(1, POLICY_EPOCHS + 1)
    ]


def test_policy_watchdog_fails_before_first_epoch_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = build_matched_policies(
        ScopeEstimator(),
        actor_seed=0xA27C,
    ).a
    optimizer_stepped = False

    def fail_if_stepped(*args: object, **kwargs: object) -> object:
        nonlocal optimizer_stepped
        optimizer_stepped = True
        raise AssertionError("optimizer stepped after watchdog expiry")

    def expired(stage: str) -> None:
        assert stage == f"policy A epoch 1/{POLICY_EPOCHS}"
        raise TimeoutError("expired")

    monkeypatch.setattr(torch.optim.Adam, "step", fail_if_stepped)
    with pytest.raises(TimeoutError, match="expired"):
        train_policy_cell(
            "A",
            policy,
            _features(),
            q_safe=0.35,
            checkpoint=expired,
        )
    assert optimizer_stepped is False


def test_mutated_cd_optimizer_membership_fails_closed() -> None:
    frozen = FrozenScopeEstimator.from_estimator(ScopeEstimator())
    actor = SeparateActor()
    parameter = next(frozen.estimator.parameters())
    parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(
        tuple(actor.actor_parameters()) + (parameter,), lr=0.003)
    with pytest.raises(AssertionError, match="policy optimizer"):
        frozen.assert_integrity(optimizer)


def test_policy_training_rejects_arm_architecture_and_budget_mutations() -> None:
    inputs = _features()
    with pytest.raises(TypeError, match="coupled"):
        train_policy_cell("A", SeparateActor(), inputs, q_safe=0.35)
    with pytest.raises(TypeError, match="separate"):
        train_policy_cell(
            "C", CoupledPolicy(), inputs, q_safe=0.35,
            frozen_estimator=FrozenScopeEstimator.from_estimator(
                ScopeEstimator()))
    with pytest.raises(ValueError, match="exactly 80"):
        train_policy_cell(
            "A", CoupledPolicy(), inputs, q_safe=0.35, epochs=79)
