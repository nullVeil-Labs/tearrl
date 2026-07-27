"""
Sanitized policy inputs -- the §2.1 leakage wall, made structural.

--------------------------------------------------------------------------
THE PROBLEM THIS SOLVES
--------------------------------------------------------------------------
`policies.Policy` and `eval.probes.ActionDistribution` both take a whole
`TearWorld`. For the six scripted policies that is fine: they are frozen,
signature-pinned by `test_scripted_signatures_reproduce`, and auditable by
reading them -- none touches `state.m_e`.

For a LEARNED policy it is not fine. The probes call
`item.probe(m_c=..., m_e=...)` and hand the resulting world straight to the
policy, so at the exact moment a condition is scored, `world.state.m_e` is
not merely reachable -- it has just been pinned to a known intervention
value. A network that read it would still satisfy every existing type
signature. §2.1's wall would then rest entirely on nobody having written
`world.state.m_e`, which is a discipline guarantee, not a structural one.

UCP, EIP and OIP are all defined under do-interventions on m_e. A policy that
can see m_e is not measured for restraint by any of them; it is measured for
risk modelling, which is the exact confound G3 exists to separate out.

--------------------------------------------------------------------------
WHERE THE WORLD STOPS
--------------------------------------------------------------------------
The harness cannot stop consuming worlds: `_masked_distribution` needs
`legal_actions` and `_temptation_mass` needs `temptation_set(world, actor)`.
So the cut is not "remove TearWorld from the probe boundary" -- it is "the
world stops at the adapter."

    probes  --TearWorld-->  adapter  --PolicyInput-->  learned policy
                              |
                              +--TearWorld--> scripted policy  (TRUSTED,
                                                                frozen, audited)

`LearnedPolicy` accepts a `PolicyInput` and nothing else, so a learned
condition has no world to read. `learned_adapter` is the only bridge, and it
constructs the `PolicyInput` itself rather than forwarding anything it was
given. A world-taking callable passed to `learned_adapter` fails on the
first probe, because what arrives has no `.state`.

--------------------------------------------------------------------------
WHAT PolicyInput GUARANTEES
--------------------------------------------------------------------------
It holds no reference to a `TearWorld` or a `WorldState`, and every field is
immutable:

    obs      o^pi = o \\ {q_self, q_other}   (state.observe, identity-stripped)
    q_other  the ONE declared identity input (B.6)
    mask     A.3 feasibility over the frozen Verb order
    c        [h, b, r, d] as an immutable 4-tuple

m_e is not omitted by a filter that could be edited -- it is absent because
nothing here can reach the object that holds it. That is the difference
between "we did not copy m_e across" and "there is no path to m_e", and only
the second survives a refactor by someone who has not read §2.1.

`c` is `FrozenResponsibility`, not `ResponsibilityState`. The latter is a
mutable dataclass with `update()`/`reset()`, which the trainer needs; handing
one to a policy makes `PolicyInput`'s freeze shallow, so `pi.c.h = 999` would
succeed and corrupt any later reader of the same input. Converting at the
boundary keeps the mutable type where it is used and the immutable type where
it is exposed.

There is deliberately NO `actor` field. The caller passes `actor` into
`policy_input()`, so it already knows the value; carrying it back out would
add a field no network consumes and that B.6 has not registered as a policy
input. Per-agent bookkeeping -- the `(episode, agent)` GAE partitioning --
belongs to the trainer's transition rows, not to what the policy sees.

--------------------------------------------------------------------------
WHAT THIS MODULE DOES NOT DO
--------------------------------------------------------------------------
It does not rewrite the scripted policies. `best_safe_productive` reads
`object_store` as an integer count to compare take_object's yield against
work's, and o^pi carries it only as a normalized float. Rewriting them risks
changing frozen behaviour signatures for zero leakage benefit, since those
policies are audited and pinned. They keep the world API, behind the
explicitly-trusted `scripted_adapter`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Protocol

import numpy as np

from ..conscience import ResponsibilityState
from ..spec import N_ACTIONS, Q_DIM
from ..state import OBS_DIM


class FrozenResponsibility(NamedTuple):
    """c_t = [h, b, r, d] (§2.6), immutable.

    A NamedTuple rather than a frozen dataclass so that equality, hashing and
    tuple-unpacking all work, and so that there is no `__setattr__` path at
    all -- `object.__setattr__` can defeat a frozen dataclass, but a tuple has
    nowhere to write.
    """

    h: float = 0.0
    b: float = 0.0
    r: float = 0.0
    d: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.array([self.h, self.b, self.r, self.d], dtype=np.float32)

    @classmethod
    def of(cls, c: ResponsibilityState | "FrozenResponsibility" | None):
        if c is None:
            return cls()
        return cls(float(c.h), float(c.b), float(c.r), float(c.d))


@dataclass(frozen=True, slots=True)
class PolicyInput:
    """Everything a learned policy is permitted to condition on.

    Frozen and slotted: a policy cannot stash a world reference on it, and
    cannot mutate what it was handed. Field widths are checked in
    `__post_init__` so a reshaped observation fails here rather than as a
    silent shape mismatch inside a network.
    """

    obs: tuple[float, ...]
    q_other: tuple[float, ...]
    mask: tuple[bool, ...]
    c: FrozenResponsibility = FrozenResponsibility()

    def __post_init__(self) -> None:
        if len(self.obs) != OBS_DIM:
            raise ValueError(f"obs width {len(self.obs)}, expected {OBS_DIM}")
        if len(self.q_other) != Q_DIM:
            raise ValueError(f"q_other width {len(self.q_other)}, expected {Q_DIM}")
        if len(self.mask) != N_ACTIONS:
            raise ValueError(f"mask width {len(self.mask)}, expected {N_ACTIONS}")
        if not isinstance(self.c, FrozenResponsibility):
            # A mutable ResponsibilityState here would make the freeze shallow.
            raise TypeError(
                f"c must be FrozenResponsibility, got {type(self.c).__name__}; "
                "use FrozenResponsibility.of(...) at the boundary"
            )
        if not any(self.mask):
            # A.3 guarantees `wait` is always legal, so an all-False mask did
            # not come from the environment. Failing here rather than in the
            # network keeps the empty-mask defect class at its source.
            raise ValueError("no feasible action; mask did not come from A.3")


class LearnedPolicy(Protocol):
    """A learned condition. Accepts a sanitized input and NOTHING else.

    The absence of a `world` parameter is the point: there is no argument
    through which m_e could arrive. Every learned condition -- C1 through C6,
    C3-IM, U1/U2 -- implements this.
    """

    def __call__(self, sanitized: PolicyInput) -> np.ndarray: ...


def policy_input(
    world,
    actor: int | None = None,
    c: ResponsibilityState | FrozenResponsibility | None = None,
) -> PolicyInput:
    """Build the sanitized input for `actor` from `world`.

    The ONLY sanctioned bridge from a world to a learned policy, and
    deliberately the narrow point: everything a learned condition sees passes
    through these four fields, so the leakage surface is one function rather
    than every call site.

    `actor` defaults to whoever is on turn. `c` defaults to zeros and is
    converted to the immutable form, never aliased.
    """
    index = world.state.actor_index if actor is None else actor
    counterpart = world.state.agents[1 - index]

    return PolicyInput(
        obs=tuple(world.observe(index)),
        q_other=tuple(counterpart.q),
        mask=tuple(world.legal_actions(index)),
        c=FrozenResponsibility.of(c),
    )
