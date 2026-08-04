from __future__ import annotations

import numpy as np
import pytest
import torch

from tearrl.cpp.calibration import (
    ESTIMATOR_EPOCHS,
    TEMPERATURE_MAX,
    TEMPERATURE_MIN,
    FrozenScopeEstimator,
    ScopeEstimator,
    canonical_state_digest,
    fit_temperature,
    train_scope_estimator,
)


def _declared_fixture(repeats: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    rows, labels = [], []
    for declaration in (-1.0, 1.0):
        for marker in (-1.0, 1.0):
            label = float(declaration == marker)
            for pattern in (
                (marker, marker, marker),
                (marker, marker, -marker),
                (marker, -marker, marker),
                (-marker, marker, marker),
            ):
                for _ in range(repeats):
                    rows.append((1.0, declaration, *pattern))
                    labels.append(label)
    return (
        torch.tensor(rows, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.float32),
    )


def test_estimator_architecture_and_training_are_deterministic() -> None:
    features, labels = _declared_fixture()
    checkpoints: list[str] = []
    first, first_report = train_scope_estimator(
        features,
        labels,
        seed=1031,
        checkpoint=checkpoints.append,
    )
    second, second_report = train_scope_estimator(
        features, labels, seed=1031)

    assert first.input_layer.in_features == 5
    assert first.input_layer.out_features == 16
    assert first.hidden_layer.in_features == 16
    assert first.hidden_layer.out_features == 16
    assert first.output_layer.in_features == 16
    assert first.output_layer.out_features == 1
    assert first_report.epochs == ESTIMATOR_EPOCHS == 100
    assert first_report.final_log_loss < first_report.initial_log_loss
    assert first_report == second_report
    assert canonical_state_digest(first) == canonical_state_digest(second)
    assert checkpoints == [
        f"scope estimator epoch {epoch}/{ESTIMATOR_EPOCHS}"
        for epoch in range(1, ESTIMATOR_EPOCHS + 1)
    ]


def test_estimator_watchdog_fails_before_first_epoch_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features, labels = _declared_fixture(repeats=1)
    optimizer_stepped = False

    def fail_if_stepped(*args: object, **kwargs: object) -> object:
        nonlocal optimizer_stepped
        optimizer_stepped = True
        raise AssertionError("optimizer stepped after watchdog expiry")

    def expired(stage: str) -> None:
        assert stage == f"scope estimator epoch 1/{ESTIMATOR_EPOCHS}"
        raise TimeoutError("expired")

    monkeypatch.setattr(torch.optim.Adam, "step", fail_if_stepped)
    with pytest.raises(TimeoutError, match="expired"):
        train_scope_estimator(
            features,
            labels,
            seed=1031,
            checkpoint=expired,
        )
    assert optimizer_stepped is False


def test_temperature_fit_is_bounded_and_does_not_change_weights() -> None:
    features, labels = _declared_fixture()
    estimator, _ = train_scope_estimator(features, labels, seed=1031)
    before = {
        name: value.detach().clone()
        for name, value in estimator.named_parameters()
    }
    report = fit_temperature(estimator, features, labels)

    assert TEMPERATURE_MIN <= report.temperature <= TEMPERATURE_MAX
    assert report.final_log_loss <= report.initial_log_loss + 1e-10
    for name, value in estimator.named_parameters():
        assert torch.equal(value, before[name])


@pytest.mark.parametrize("invalid", [
    [[0.0, 1.0, 1.0, 1.0, 1.0]],
    [[1.0, 0.0, 1.0, 1.0, 1.0]],
    [[0.5, 0.0, 1.0, 1.0, 1.0]],
    [[1.0, 1.0, 0.0, 1.0, 1.0]],
])
def test_invalid_scope_observations_fail_before_model_call(invalid) -> None:
    estimator = ScopeEstimator()
    called = False

    def mark_called(_module, _inputs):
        nonlocal called
        called = True

    handle = estimator.input_layer.register_forward_pre_hook(mark_called)
    try:
        with pytest.raises(ValueError):
            estimator(torch.tensor(invalid, dtype=torch.float32))
    finally:
        handle.remove()
    assert not called


def test_frozen_estimator_returns_host_values_and_detects_mutation() -> None:
    features, labels = _declared_fixture(repeats=2)
    estimator, _ = train_scope_estimator(features, labels, seed=1031)
    fit_temperature(estimator, features, labels)
    frozen = FrozenScopeEstimator.from_estimator(estimator)

    values = frozen.probabilities(features[:3])
    assert isinstance(values, tuple)
    assert all(isinstance(value, float) for value in values)
    frozen.assert_integrity()

    with torch.no_grad():
        frozen.estimator.output_layer.bias.add_(0.01)
    with pytest.raises(AssertionError, match="digest changed"):
        frozen.assert_integrity()


def test_frozen_estimator_is_rejected_from_an_optimizer() -> None:
    frozen = FrozenScopeEstimator.from_estimator(ScopeEstimator())
    # Mutation fixture: re-enabling even one frozen parameter is detected.
    parameter = next(frozen.estimator.parameters())
    parameter.requires_grad_(True)
    optimizer = torch.optim.Adam([parameter], lr=0.003)
    with pytest.raises(AssertionError):
        frozen.assert_integrity(optimizer)


def test_digest_is_sensitive_to_temperature() -> None:
    estimator = ScopeEstimator()
    before = canonical_state_digest(estimator)
    estimator.set_temperature(1.25)
    assert canonical_state_digest(estimator) != before


def test_training_rejects_nonregistered_budget() -> None:
    features, labels = _declared_fixture(repeats=1)
    with pytest.raises(ValueError, match="exactly 100"):
        train_scope_estimator(features, labels, seed=1, epochs=99)
