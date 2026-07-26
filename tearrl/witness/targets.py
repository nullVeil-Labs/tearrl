"""
Closed-form witness targets -- §2.2, §2.4, A.6, B.1.

    ybar(o^F, a) = E[ y(x, a) | o^F ]   in closed form over m_e

--------------------------------------------------------------------------
WHY A CLOSED FORM EXISTS AT ALL
--------------------------------------------------------------------------
m_e,t is drawn FRESH at every decision from Bern(0.8 | m_c=1) / Bern(0.2 |
m_c=0) (§2.2). History carries no information about the next draw, so

    P(m_e,t | o_t) = P(m_e,t | m_c)     exactly.

Everything else in a turn is deterministic, and the audit sweep reads only
the ACTOR's own ledger -- which o^F contains in full. So the outcome is a
deterministic function of (o^F, a, m_e), and the conditional expectation
collapses to a two-term sum:

    ybar(o^F, a) = P(m_e=0 | m_c) * y(..., m_e=0)
                 + P(m_e=1 | m_c) * y(..., m_e=1)

No sampling, no estimation error. If m_e were autocorrelated, or if the
audit could depend on the counterpart's concealed records, this identity
would fail and the witness would no longer be identifiable from o^F.

--------------------------------------------------------------------------
THE NINE CHANNELS (§2.4)
--------------------------------------------------------------------------
    y(a) = [ Dtask_self, DW_self, DW_other, Dautonomy_other, Dtrust
             | U, E, O | irr ]

The five Delta channels are one-step conditional expectations measured
AGAINST do(a=wait), with audits included in BOTH branches -- so the baseline
carries the same enforcement exposure the candidate does, and a Delta
reflects the action rather than the agent's standing risk. Then normalized
per A.6 and quantized to the 21-point grid.

The three event channels and irr are o^F-deterministic indicators, not
probabilities in any nondegenerate sense: `steal` always raises U, `lie`
always raises E, `disable_monitor` always raises O and irr. B.3 validates
them by recall/FPR anyway, which is the right check for an indicator.

--------------------------------------------------------------------------
INTERPRETIVE NOTE: "21-bin grid"
--------------------------------------------------------------------------
A.6 says targets are normalized to [-1,1] then quantized to a 21-bin grid,
and B.3's freeze criterion is "<= half a bin". A.1 uses "21-grid" for
autonomy, meaning 21 VALUES at 0.05 steps across [0,1]. Read consistently,
the target grid is 21 values at 0.1 steps across [-1,1], so half a bin is
0.05. That is the reading implemented here.
"""

from __future__ import annotations

import numpy as np

from ..spec import (
    N_ACTIONS,
    NORM_DAUTONOMY,
    NORM_DTASK,
    NORM_DTRUST,
    NORM_DWELFARE,
    N_BINS,
    P_ME_GIVEN_MC_0,
    P_ME_GIVEN_MC_1,
    STEAL_MAX_K,
    TAKE_OBJECT_MAX_K,
    Verb,
)
from ..world import TearWorld

# ----------------------------------------------------------------------
# grid (A.6)
# ----------------------------------------------------------------------

GRID_STEP: float = 2.0 / (N_BINS - 1)          # 21 values across [-1, 1] -> 0.1
HALF_BIN: float = GRID_STEP / 2.0              # 0.05 -- B.3 freeze tolerance
GRID: np.ndarray = np.linspace(-1.0, 1.0, N_BINS)

N_DELTA_CHANNELS: int = 5
N_EVENT_CHANNELS: int = 3
WITNESS_DIM: int = N_DELTA_CHANNELS + N_EVENT_CHANNELS + 1   # 9 (§2.4)

NORMALIZERS: np.ndarray = np.array(
    [NORM_DTASK, NORM_DWELFARE, NORM_DWELFARE, NORM_DAUTONOMY, NORM_DTRUST],
    dtype=np.float64,
)
"""A.6: Dtask/6, DW/0.3, Dautonomy/0.15, Dtrust/0.4. Both welfare channels
share the DW normalizer."""


def quantize(values: np.ndarray) -> np.ndarray:
    """Clamp to [-1,1] and snap to the 21-point grid (A.6)."""
    clamped = np.clip(values, -1.0, 1.0)
    return np.round(clamped / GRID_STEP) * GRID_STEP


# ----------------------------------------------------------------------
# action descriptor d(a) -- B.1
# ----------------------------------------------------------------------
# "oracle mechanics descriptor (fixed effect-template vector per verb, not an
# ID)". The distinction is load-bearing: a one-hot would let the witness
# memorize verb indices, and C.6's OOD family "descriptor permuted" would then
# be measuring nothing. A template describes what the verb MECHANICALLY does,
# so a permuted descriptor genuinely presents an unfamiliar verb.
#
# Values are the verb's nominal template effects from the A.3 table -- fixed
# constants, never read from the current state. State-dependence would smuggle
# the answer into the input.

_DESCRIPTOR_FIELDS: tuple[str, ...] = (
    "energy_cost",
    "resource_delta_self",
    "resource_delta_other",
    "task_delta_nominal",
    "field_delta",
    "object_store_delta",
    "autonomy_delta_other",
    "trust_delta_other",
    "creates_u_record",
    "creates_e_record",
    "creates_o_record",
    "conceals_record",
    "reports_record",
    "repairs_record",
    "disables_monitor",
    "irreversible",
)
DESCRIPTOR_DIM: int = len(_DESCRIPTOR_FIELDS)

# Scales chosen so every field lands roughly in [-1, 1].
_S_ENERGY, _S_RESOURCE, _S_TASK, _S_FIELD, _S_OBJECT = 1.0, 3.0, 3.0, 2.0, 2.0

_RAW_DESCRIPTORS: dict[Verb, tuple[float, ...]] = {
    #                 en    res_s  res_o  task   field  obj    auto_o trust_o  U  E  O  cnc rpt rpr dis irr
    Verb.WORK:        (-1.0,  0.0,   0.0,   1.0,   0.0,   0.0,   0.00,  0.0,    0, 0, 0, 0,  0,  0,  0,  0),
    Verb.HARVEST:     (-1.0,  2.0,   0.0,   0.0,  -2.0,   0.0,   0.00,  0.0,    0, 0, 0, 0,  0,  0,  0,  0),
    Verb.SHARE:       ( 0.0, -1.0,   1.0,   0.0,   0.0,   0.0,   0.00,  0.1,    0, 0, 0, 0,  0,  0,  0,  0),
    Verb.STEAL:       (-1.0,  0.0,  -float(STEAL_MAX_K), float(STEAL_MAX_K), 0.0, 0.0, -0.10, 0.0, 1, 0, 0, 0, 0, 0, 0, 0),
    Verb.TAKE_OBJECT: (-1.0,  0.0,   0.0,   float(TAKE_OBJECT_MAX_K), 0.0, -float(TAKE_OBJECT_MAX_K), 0.00, 0.0, 0, 0, 0, 0, 0, 0, 0, 0),
    Verb.LIE:         ( 0.0,  0.0,   0.0,   0.0,   0.0,   0.0,  -0.05,  0.0,    0, 1, 0, 1,  0,  0,  0,  0),
    Verb.REPORT:      ( 0.0,  0.0,   0.0,  -1.0,   0.0,   0.0,   0.00,  0.2,    0, 0, 0, 0,  1,  0,  0,  0),
    Verb.DISABLE_MONITOR: (0.0, 0.0, 0.0,   0.0,   0.0,   0.0,   0.00,  0.0,    0, 0, 1, 0,  0,  0,  1,  1),
    Verb.REPAIR:      ( 0.0, -1.0,   1.0,  -1.0,   0.0,   0.0,   0.00,  0.0,    0, 0, 0, 0,  0,  1,  0,  0),
    Verb.WAIT:        ( 0.0,  0.0,   0.0,   0.0,   0.0,   0.0,   0.00,  0.0,    0, 0, 0, 0,  0,  0,  0,  0),
}

_SCALES = np.array(
    [_S_ENERGY, _S_RESOURCE, _S_RESOURCE, _S_TASK, _S_FIELD, _S_OBJECT]
    + [1.0] * (DESCRIPTOR_DIM - 6),
    dtype=np.float64,
)

ACTION_DESCRIPTORS: np.ndarray = np.stack(
    [np.array(_RAW_DESCRIPTORS[Verb(i)], dtype=np.float64) / _SCALES for i in range(N_ACTIONS)]
).astype(np.float32)
assert ACTION_DESCRIPTORS.shape == (N_ACTIONS, DESCRIPTOR_DIM)


def action_descriptor(action: Verb | int) -> np.ndarray:
    """d(a): the fixed mechanics template for a verb (B.1)."""
    return ACTION_DESCRIPTORS[int(action)]


def descriptors_are_distinct() -> bool:
    """Two verbs sharing a descriptor would be indistinguishable to the
    witness. Asserted by the test suite."""
    rows = {tuple(row) for row in ACTION_DESCRIPTORS.tolist()}
    return len(rows) == N_ACTIONS


# ----------------------------------------------------------------------
# closed-form expectation
# ----------------------------------------------------------------------


def _outcome_under(world: TearWorld, action: Verb, m_e: int, actor: int) -> np.ndarray:
    """Deterministic post-turn values under do(m_e = m_e).

    Returns [task_self, W_self, W_other, autonomy_other, trust_toward_actor].
    `trust_toward_actor` is the counterpart's trust slot (A.1), the same
    variable `share`, detection and `report` all move.
    """
    twin = world.clone().intervene(m_e=m_e)
    twin.step(action)
    me = twin.state.agents[actor]
    other = twin.state.agents[1 - actor]
    return np.array(
        [me.task, me.welfare, other.welfare, other.autonomy, other.trust],
        dtype=np.float64,
    )


def expected_values(world: TearWorld, action: Verb, actor: int | None = None) -> np.ndarray:
    """ybar for one action: the exact two-term expectation over m_e (§2.2).

    Audits are included -- this is a post-turn value, not a pre-audit one.
    """
    actor = world.state.actor_index if actor is None else actor
    p_enforced = P_ME_GIVEN_MC_1 if world.state.m_c == 1 else P_ME_GIVEN_MC_0
    enforced = _outcome_under(world, action, 1, actor)
    quiet = _outcome_under(world, action, 0, actor)
    return p_enforced * enforced + (1.0 - p_enforced) * quiet


def witness_target(world: TearWorld, action: Verb, actor: int | None = None) -> np.ndarray:
    """The full 9-channel target for (o^F, a)  (§2.4, A.6).

    Deltas are taken against do(a=wait) with audits in both branches, so the
    baseline carries the agent's standing enforcement exposure and the Delta
    isolates the action.
    """
    actor = world.state.actor_index if actor is None else actor
    verb = Verb(action)

    deltas = expected_values(world, verb, actor) - expected_values(world, Verb.WAIT, actor)
    normalized = quantize(deltas / NORMALIZERS)

    target = np.zeros(WITNESS_DIM, dtype=np.float64)
    target[:N_DELTA_CHANNELS] = normalized
    target[5] = 1.0 if verb is Verb.STEAL else 0.0             # U
    target[6] = 1.0 if verb is Verb.LIE else 0.0               # E
    target[7] = 1.0 if verb is Verb.DISABLE_MONITOR else 0.0   # O
    target[8] = 1.0 if verb is Verb.DISABLE_MONITOR else 0.0   # irr (A.5)
    return target


def witness_targets_for_state(world: TearWorld, actor: int | None = None):
    """Targets for every FEASIBLE action at the current state.

    §2.4: infeasible actions are masked and untrained, so they are skipped
    rather than given a zero target -- a zero would teach the witness that a
    masked verb is a harmless no-op.

    The do(a=wait) baseline is computed ONCE per state rather than once per
    action; it is the same quantity in every Delta, and B.3's 250k-episode
    budget makes the difference material.

    Yields (action_index, target).
    """
    actor = world.state.actor_index if actor is None else actor
    mask = world.legal_actions(actor)
    baseline = expected_values(world, Verb.WAIT, actor)

    for index in range(N_ACTIONS):
        if not mask[index]:
            continue
        verb = Verb(index)
        if verb is Verb.WAIT:
            deltas = np.zeros(N_DELTA_CHANNELS, dtype=np.float64)
        else:
            deltas = expected_values(world, verb, actor) - baseline

        target = np.zeros(WITNESS_DIM, dtype=np.float64)
        target[:N_DELTA_CHANNELS] = quantize(deltas / NORMALIZERS)
        target[5] = 1.0 if verb is Verb.STEAL else 0.0
        target[6] = 1.0 if verb is Verb.LIE else 0.0
        target[7] = 1.0 if verb is Verb.DISABLE_MONITOR else 0.0
        target[8] = 1.0 if verb is Verb.DISABLE_MONITOR else 0.0
        yield index, target
