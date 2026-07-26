"""
Deterministic F2 intervention pairing (§6.2, Amendment 001 §5.2).

The distinction these tests enforce: a CANDIDATE EDGE is not an INTERVENTION
PAIR. A tuple may donate or receive c_t once. Reporting edge counts as pair
counts overstated usable supply by a factor of ~46 in the first census, which
is why the one-to-one constraint is tested rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pytest

from tearrl.eval.f2_matching import (
    C_NORM_TOLERANCE,
    MIN_DEBT_GAP,
    YHAT_TOLERANCE,
    build_candidate_edges,
    covariate_balance,
    edge_key,
    match,
)


def _grid(n: int = 60) -> dict[str, np.ndarray]:
    """A synthetic pool in one (timestep, task_op) block, half low-debt."""
    rng = np.random.default_rng(0)
    debt = np.where(np.arange(n) < n // 2, 0.0, 0.5)
    return {
        "c_norm": rng.uniform(0.40, 0.42, n),
        "yhat": rng.uniform(1.00, 1.10, n),
        "d": debt,
        "t": np.full(n, 11),
        "task_op": np.ones(n, dtype=int),
        "seeds": np.arange(7_300_000, 7_300_000 + n),
    }


def _edges(data):
    return build_candidate_edges(
        data["c_norm"], data["yhat"], data["d"], data["t"], data["task_op"]
    )


def test_each_tuple_appears_in_at_most_one_pair() -> None:
    """The one-to-one constraint. Without it, a single well-placed donor tuple
    could be counted as dozens of independent interventions."""
    data = _grid()
    result = match(_edges(data), data["seeds"], n_tuples=len(data["d"]))

    used: list[int] = []
    for a, b in result.pairs:
        used.extend((a, b))
    assert len(used) == len(set(used)), "a tuple was reused across pairs"


def test_pairs_respect_every_registered_tolerance() -> None:
    data = _grid()
    result = match(_edges(data), data["seeds"], n_tuples=len(data["d"]))
    assert result.pairs

    for a, b in result.pairs:
        assert abs(data["c_norm"][a] - data["c_norm"][b]) <= C_NORM_TOLERANCE
        assert abs(data["yhat"][a] - data["yhat"][b]) <= YHAT_TOLERANCE
        assert abs(data["d"][a] - data["d"][b]) >= MIN_DEBT_GAP


def test_pair_count_is_bounded_by_half_the_participants() -> None:
    """The bound that exposed the original overstatement."""
    data = _grid()
    edges = _edges(data)
    result = match(edges, data["seeds"], n_tuples=len(data["d"]))

    assert result.candidate_edges == len(edges)
    assert result.n_pairs <= result.participants // 2
    assert result.n_pairs < result.candidate_edges


def test_matching_is_deterministic() -> None:
    """Frozen SHA-256 tie-breaking must make the selected pairing reproducible
    across runs and processes."""
    data = _grid()
    first = match(_edges(data), data["seeds"], n_tuples=len(data["d"]))
    second = match(_edges(data), data["seeds"], n_tuples=len(data["d"]))
    assert first.pairs == second.pairs


def test_edge_key_is_symmetric_and_stable() -> None:
    assert edge_key(11, 22) == edge_key(22, 11)
    assert 0.0 <= edge_key(11, 22) < 1.0
    assert edge_key(11, 22) != edge_key(11, 23)
    assert edge_key(11, 22, seed=1) != edge_key(11, 22, seed=2)


def test_banded_costs_order_two_edges_lexicographically() -> None:
    """The banding orders any TWO INDIVIDUAL EDGES by level 1 first.

    Note what this does NOT establish: the solver minimizes a SUM, so across a
    whole matching many small level-2 savings can outweigh one level-1 loss.
    That limit is documented in the module and in Amendment 001 §5.2 rather
    than papered over -- see the next test.
    """
    c_norm = np.array([0.400, 0.401, 0.450])
    yhat = np.array([1.000, 1.000, 1.000])
    debt = np.array([0.0, 0.5, 0.5])
    edges = build_candidate_edges(
        c_norm, yhat, debt, np.full(3, 11), np.ones(3, dtype=int)
    )
    result = match(edges, np.arange(7_300_000, 7_300_003), n_tuples=3)
    assert result.pairs == [(0, 1)], "did not prefer the tighter ||c|| match"


def test_banding_does_not_guarantee_matching_level_lexicographic_order() -> None:
    """Pins the honest limit of the frozen rule (Amendment 001 §5.2).

    If this ever starts failing, the weights have been separated enough to make
    level 1 dominate additively -- at which point the amendment's careful
    disclaimer is understating the guarantee and should be revisited, rather
    than the test being deleted.
    """
    from tearrl.eval.f2_matching import _W1, _W2

    per_edge_level1_span = _W1 * C_NORM_TOLERANCE
    level2_per_edge_max = _W2 * YHAT_TOLERANCE
    pairs_at_registered_pool = 1595

    assert level2_per_edge_max * pairs_at_registered_pool > per_edge_level1_span, (
        "summed level-2 cost can no longer exceed a per-edge level-1 difference; "
        "the §5.2 disclaimer now understates the guarantee"
    )


def test_coverage_counts_matched_tuples_not_edges() -> None:
    """Coverage defines the matched-overlap population the F2 estimand applies
    to, so it must be 2 * pairs / n_tuples."""
    data = _grid()
    result = match(_edges(data), data["seeds"], n_tuples=200)
    assert result.coverage == pytest.approx(2 * result.n_pairs / 200)


def test_balance_reports_the_realized_debt_contrast() -> None:
    data = _grid()
    result = match(_edges(data), data["seeds"], n_tuples=len(data["d"]))
    balance = covariate_balance(result, data["c_norm"], data["yhat"], data["d"])

    assert balance["min_debt_gap"] >= MIN_DEBT_GAP
    assert balance["max_abs_delta_c_norm"] <= C_NORM_TOLERANCE
    assert balance["mean_debt_low"] < balance["mean_debt_high"]


def test_empty_candidate_set_is_handled() -> None:
    """No pairs is a reportable outcome, not a crash: it would mean F2 has no
    matched-overlap population at all."""
    c_norm = np.array([0.1, 0.9])
    result = match(
        build_candidate_edges(
            c_norm, np.array([1.0, 1.0]), np.array([0.0, 0.5]),
            np.full(2, 11), np.ones(2, dtype=int),
        ),
        np.array([7_300_000, 7_300_001]),
        n_tuples=2,
    )
    assert result.n_pairs == 0
    assert result.coverage == 0.0
