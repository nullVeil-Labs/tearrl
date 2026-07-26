"""
Deterministic F2 intervention pairing -- §6.2, Amendment 001 §5.2.

The census counts CANDIDATE EDGES. The intervention needs one-to-one PAIRS: a
tuple may donate or receive c_t once, not many times. With 22,642 candidate
edges over 5,601 participating tuples the average degree is ~8, so the usable
supply is bounded by |participants| / 2 -- roughly 2,800, not 22,642. Reporting
edge count as though it were pair count overstates supply by an order of
magnitude, and the F2 estimand is defined over the MATCHED-OVERLAP POPULATION,
not over every reachable state.

FROZEN RULE
-----------
Build a graph whose vertices are tuples satisfying the registered matching
tolerances and whose edges join pairs differing in harm debt by at least the
registered gap. Select a MAXIMUM-CARDINALITY matching; among maximum-cardinality
matchings choose the one minimizing, lexicographically:

    1. |delta ||c|| |
    2. |delta sum|yhat| |
    3. a frozen SHA-256 edge key with seed 7301

Each tuple appears in at most one pair.

The debt graph is NOT bipartite -- d = 0.0, 0.5, 1.0 are mutually >= 0.5 apart
and form a triangle -- so general (blossom) matching is required rather than a
bipartite assignment.

EXACT AGGREGATE -- read this before citing the rule
---------------------------------------------------
Each edge carries the scalar banded cost

    cost = W1*|d||c||  +  W2*|d sum|yhat||  +  W3*hash,
    W1 = 1.0,  W2 = 1e-3,  W3 = 1e-7

and the solver selects a maximum-cardinality matching MINIMIZING THE SUM of
those costs. So the aggregate is additive, and the banding gives a strict
lexicographic ordering only BETWEEN INDIVIDUAL EDGES.

It does NOT give a matching-level lexicographic minimum over sorted cost
vectors. Under the registered tolerances a single edge's level-1 term spans at
most 0.05 while level-2 contributes up to 5e-4 per edge, so across a
1,595-pair matching roughly 60 edges of level-2 savings could offset one
edge's 0.03 level-1 loss. In practice level 1 dominates by two orders of
magnitude (measured totals ~26.0 vs ~0.30), but dominance is NOT guaranteed
and must not be claimed.

The frozen rule is therefore: maximum cardinality first, then minimum SUMMED
banded cost, ties broken by the SHA-256 edge key. That is fully deterministic
and reproducible, which is what the protocol requires; it is simply not the
sorted-vector lexicographic minimum, and the distinction is recorded so the
prose and the implementation cannot drift apart.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

# Registered F2 matching tolerances (Amendment 001 §5.1), frozen.
C_NORM_TOLERANCE: float = 0.05
YHAT_TOLERANCE: float = 0.50
MIN_DEBT_GAP: float = 0.50

F2_SEED: int = 7301
F2_HASH_DOMAIN: bytes = b"f2-pairing"

# Banded weights. These order any TWO EDGES lexicographically (level 1, then
# level 2, then hash). They do NOT make level 1 dominate summed level-2 terms
# across a whole matching -- see the module docstring. Frozen values.
_W1, _W2, _W3 = 1.0, 1.0e-3, 1.0e-7


def edge_key(seed_a: int, seed_b: int, seed: int = F2_SEED) -> float:
    """Frozen deterministic tie-break in [0, 1). Order-independent in (a, b)."""
    low, high = (seed_a, seed_b) if seed_a <= seed_b else (seed_b, seed_a)
    payload = b"|".join(
        (str(low).encode(), str(high).encode(), F2_HASH_DOMAIN, str(seed).encode())
    )
    digest = hashlib.sha256(payload).digest()[:8]
    return int.from_bytes(digest, "big") / float(1 << 64)


@dataclass(slots=True)
class F2Matching:
    pairs: list[tuple[int, int]] = field(default_factory=list)
    candidate_edges: int = 0
    participants: int = 0
    n_tuples: int = 0
    max_cardinality: int = 0

    @property
    def n_pairs(self) -> int:
        return len(self.pairs)

    @property
    def coverage(self) -> float:
        """Share of ALL tuples that enter the intervention. This is the
        matched-overlap population the F2 estimand is defined over."""
        return (2 * self.n_pairs / self.n_tuples) if self.n_tuples else 0.0


def build_candidate_edges(
    c_norm: np.ndarray,
    yhat: np.ndarray,
    debt: np.ndarray,
    timestep: np.ndarray,
    task_op: np.ndarray,
    c_tolerance: float = C_NORM_TOLERANCE,
    yhat_tolerance: float = YHAT_TOLERANCE,
    min_debt_gap: float = MIN_DEBT_GAP,
) -> list[tuple[int, int, float, float]]:
    """Candidate edges (i, j, d_cnorm, d_yhat) within a (timestep, task_op) block."""
    from collections import defaultdict

    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, (t, op) in enumerate(zip(timestep, task_op)):
        buckets[(int(t), int(op))].append(index)

    edges: list[tuple[int, int, float, float]] = []
    for members in buckets.values():
        if len(members) < 2:
            continue
        idx = np.asarray(members)
        dc = np.abs(c_norm[idx][:, None] - c_norm[idx][None, :])
        dy = np.abs(yhat[idx][:, None] - yhat[idx][None, :])
        dd = np.abs(debt[idx][:, None] - debt[idx][None, :])
        ok = np.triu((dc <= c_tolerance) & (dy <= yhat_tolerance) & (dd >= min_debt_gap), k=1)
        rows, cols = np.nonzero(ok)
        for r, c in zip(rows, cols):
            edges.append((int(idx[r]), int(idx[c]), float(dc[r, c]), float(dy[r, c])))
    return edges


def match(
    edges: list[tuple[int, int, float, float]],
    seeds: np.ndarray,
    n_tuples: int,
) -> F2Matching:
    """Maximum-cardinality one-to-one matching of minimum SUMMED banded cost.

    See the module docstring: the aggregate is additive, not a sorted-vector
    lexicographic minimum.
    """
    import networkx as nx

    graph = nx.Graph()
    for i, j, d_cnorm, d_yhat in edges:
        cost = (
            _W1 * d_cnorm
            + _W2 * d_yhat
            + _W3 * edge_key(int(seeds[i]), int(seeds[j]))
        )
        # max_weight_matching maximizes; negate so minimal cost wins.
        graph.add_edge(i, j, weight=-cost)

    participants = graph.number_of_nodes()
    if participants == 0:
        return F2Matching(candidate_edges=len(edges), n_tuples=n_tuples)

    chosen = nx.max_weight_matching(graph, maxcardinality=True)
    pairs = sorted(tuple(sorted(pair)) for pair in chosen)

    return F2Matching(
        pairs=[(int(a), int(b)) for a, b in pairs],
        candidate_edges=len(edges),
        participants=participants,
        n_tuples=n_tuples,
        max_cardinality=len(pairs),
    )


def covariate_balance(
    matching: F2Matching,
    c_norm: np.ndarray,
    yhat: np.ndarray,
    debt: np.ndarray,
) -> dict[str, float]:
    """Balance diagnostics over the matched pairs, plus the realized debt
    contrast actually patched (Amendment 001 §5.2 reporting requirement)."""
    if not matching.pairs:
        return {}
    a = np.array([p[0] for p in matching.pairs])
    b = np.array([p[1] for p in matching.pairs])

    low = np.minimum(debt[a], debt[b])
    high = np.maximum(debt[a], debt[b])
    return {
        "mean_abs_delta_c_norm": float(np.abs(c_norm[a] - c_norm[b]).mean()),
        "max_abs_delta_c_norm": float(np.abs(c_norm[a] - c_norm[b]).max()),
        "mean_abs_delta_yhat": float(np.abs(yhat[a] - yhat[b]).mean()),
        "max_abs_delta_yhat": float(np.abs(yhat[a] - yhat[b]).max()),
        "mean_debt_gap": float((high - low).mean()),
        "min_debt_gap": float((high - low).min()),
        "mean_debt_low": float(low.mean()),
        "mean_debt_high": float(high.mean()),
    }
