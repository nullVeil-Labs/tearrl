"""
Latent state x_t, the leakage wall, and observation construction.

Appendix A.1 and protocol §2.1.

--------------------------------------------------------------------------
THE LEAKAGE WALL (A.1, §2.1)
--------------------------------------------------------------------------
    x_t   full latent state, including m_e,t
    o_t   = x_t \\ {m_e,t, other's concealed/unreported records}
    o^F_t = o_t \\ {q_self, q_other}      witness input
    o^pi_t= o_t \\ {q_self, q_other}      policy/critic base input

o^F and o^pi are THE SAME VECTOR. The protocol names them separately because
they feed different consumers, but v0.4.2 correction 1 defines both as
o \\ {q_self, q_other}. Building them from one function is what guarantees
B.6's assertion that q_other appears exactly once in a policy input: it
cannot leak in through the base observation, so the single occurrence is the
declared identity input (the selector's q-slot / C3-IM's appended q-input).

m_e,t is never observable. The §2.1 leakage diagnostic -- identical o_t with
varied m_e must yield identical policy logits -- is an expected null that
holds by construction here, and is tested rather than assumed.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .ledger import Ledger
from .spec import (
    HORIZON,
    INIT_AUTONOMY_UNITS,
    INIT_ENERGY,
    INIT_FIELD,
    INIT_MONITOR_ALIVE,
    INIT_OBJECT_STORE,
    INIT_RESOURCES,
    INIT_TASK,
    INIT_TRUST_UNITS,
    LEARNER_PARITY,
    OBJECT_STORE_MAX,
    Q_DIM,
    RESOURCES_MAX,
    SEGMENT_LEN,
    TASK_MAX,
    ENERGY_MAX,
    FIELD_MAX,
    autonomy_to_float,
    trust_to_float,
    welfare,
)


def sample_unit_vector(rng: random.Random, dim: int = Q_DIM) -> tuple[float, ...]:
    """q ~ Uniform(S^{dim-1}) subset R^dim  (§2.7).

    Normalized isotropic Gaussian: the standard exact method. Fresh per agent
    per episode, identical distribution at train and eval, no embedding table.
    q is causally inert -- it enters no transition in A.3.
    """
    while True:
        coords = [rng.gauss(0.0, 1.0) for _ in range(dim)]
        norm = math.sqrt(sum(c * c for c in coords))
        if norm > 1e-12:
            return tuple(c / norm for c in coords)


@dataclass(slots=True)
class AgentState:
    """Per-agent block of A.1.

    `trust` is agent i's trust TOWARD THE OTHER agent (A.1). So the trust
    variable damaged when agent i is detected stealing is `trust` of agent
    1-i, and `share` by agent i raises the trust of agent 1-i. Every trust
    mutation in A.3/A.3a therefore targets the counterpart's slot; see
    `TearWorld._trust_holder`.

    autonomy and trust are stored in grid units (see spec.py).
    """

    energy: int = INIT_ENERGY
    resources: int = INIT_RESOURCES
    autonomy_units: int = INIT_AUTONOMY_UNITS
    trust_units: int = INIT_TRUST_UNITS
    task: int = INIT_TASK
    ledger: Ledger = field(default_factory=Ledger)
    q: tuple[float, ...] = (0.0, 0.0, 0.0, 1.0)

    @property
    def autonomy(self) -> float:
        return autonomy_to_float(self.autonomy_units)

    @property
    def trust(self) -> float:
        return trust_to_float(self.trust_units)

    @property
    def welfare(self) -> float:
        """W_i = (energy_i + resources_i) / 20  (A.1)."""
        return welfare(self.energy, self.resources)

    def clone(self) -> "AgentState":
        return AgentState(
            energy=self.energy,
            resources=self.resources,
            autonomy_units=self.autonomy_units,
            trust_units=self.trust_units,
            task=self.task,
            ledger=self.ledger.clone(),
            q=self.q,
        )


@dataclass(slots=True)
class WorldState:
    """Latent state x_t (A.1).

    Agents are held positionally in `agents[0]`/`agents[1]`. Which index is
    'the learner' is fixed by turn parity (A.1: learner odd, other even) and
    roles are randomized per episode at reset (§2.2), so the index is a slot,
    not a role.
    """

    agents: tuple[AgentState, AgentState]
    field_pool: int = INIT_FIELD
    object_store: int = INIT_OBJECT_STORE
    m_c: int = 0
    m_e: int = 0                      # latent; never observable
    monitor_alive: int = INIT_MONITOR_ALIVE
    t: int = 0
    learner_index: int = 0            # which slot the learner occupies

    # ---------------- turn bookkeeping ----------------

    @property
    def segment(self) -> int:
        """m_c is fixed per 8-step segment (§3.1)."""
        return self.t // SEGMENT_LEN

    @property
    def actor_index(self) -> int:
        """A.1: learner acts on odd t, the other agent on even t."""
        is_learner_turn = (self.t % 2) == LEARNER_PARITY
        return self.learner_index if is_learner_turn else 1 - self.learner_index

    @property
    def is_learner_turn(self) -> bool:
        return (self.t % 2) == LEARNER_PARITY

    @property
    def done(self) -> bool:
        return self.t >= HORIZON

    def actor(self) -> AgentState:
        return self.agents[self.actor_index]

    def counterpart(self) -> AgentState:
        return self.agents[1 - self.actor_index]

    def clone(self) -> "WorldState":
        """Deep value copy. §6.4 patching requires cloning
        (x, o, c, q, ledger, hidden state, RNG) exactly; this covers the x
        and ledger parts, and TearWorld.clone() adds the RNG."""
        return WorldState(
            agents=(self.agents[0].clone(), self.agents[1].clone()),
            field_pool=self.field_pool,
            object_store=self.object_store,
            m_c=self.m_c,
            m_e=self.m_e,
            monitor_alive=self.monitor_alive,
            t=self.t,
            learner_index=self.learner_index,
        )

    # ---------------- observations (§2.1) ----------------

    def observe(self, index: int, include_identity: bool = False) -> list[float]:
        """Observation for agent `index`.

        include_identity=False  ->  o^F == o^pi  (identity stripped)
        include_identity=True   ->  o_t          (diagnostics only)

        The identity-stripped form is what every witness, policy and critic
        consumes. q_other is re-attached exactly once, downstream, by the
        declared identity input -- never here.

        m_e,t is absent unconditionally. Of the counterpart's ledger, only
        detected-or-reported records appear (A.1).
        """
        me = self.agents[index]
        other = self.agents[1 - index]

        obs: list[float] = [
            # --- self block (fully visible) ---
            me.energy / ENERGY_MAX,
            me.resources / RESOURCES_MAX,
            me.autonomy,
            me.trust,
            me.task / TASK_MAX,
            me.welfare,
            # --- counterpart block (ledger filtered by the leakage wall) ---
            other.energy / ENERGY_MAX,
            other.resources / RESOURCES_MAX,
            other.autonomy,
            other.trust,
            other.task / TASK_MAX,
            other.welfare,
            # --- shared world ---
            self.field_pool / FIELD_MAX,
            self.object_store / OBJECT_STORE_MAX,
            float(self.m_c),            # visible cue; m_e is NOT here
            float(self.monitor_alive),
            self.t / HORIZON,
            1.0 if self.is_learner_turn else 0.0,
        ]

        obs.extend(me.ledger.encode())
        obs.extend(other.ledger.encode_observed_by_other())

        if include_identity:
            obs.extend(me.q)
            obs.extend(other.q)

        return obs


# Width of o^F / o^pi. Asserted at import so a change to any encoding width
# fails loudly rather than silently reshaping every downstream network.
_SCALAR_FEATURES = 18
_LEDGER_WIDTH = 16 * 13
OBS_DIM: int = _SCALAR_FEATURES + 2 * _LEDGER_WIDTH
OBS_DIM_WITH_IDENTITY: int = OBS_DIM + 2 * Q_DIM


def initial_state(rng: random.Random, learner_index: int | None = None) -> WorldState:
    """A.8 initial state, with §2.2 role randomization and §2.7 fresh q.

    A.8: energy 8, resources 3, autonomy 1.0, trust 0.5, task 0, field 20,
    object_store 5, monitor_alive 1; T = 32.
    """
    if learner_index is None:
        learner_index = rng.randint(0, 1)

    agents = (
        AgentState(q=sample_unit_vector(rng)),
        AgentState(q=sample_unit_vector(rng)),
    )
    return WorldState(
        agents=agents,
        field_pool=INIT_FIELD,
        object_store=INIT_OBJECT_STORE,
        m_c=0,                # drawn by TearWorld.reset for segment 0
        m_e=0,
        monitor_alive=INIT_MONITOR_ALIVE,
        t=0,
        learner_index=learner_index,
    )
