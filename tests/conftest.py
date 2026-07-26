"""Shared fixtures and helpers for the §3.3 four-layer verification suite."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tearrl import TearWorld, Verb  # noqa: E402
from tearrl.spec import HORIZON  # noqa: E402


def pytest_addoption(parser):
    parser.addoption(
        "--full",
        action="store_true",
        default=False,
        help="run §3.3 layer-4 property tests at the full 10^6-trajectory budget",
    )


@pytest.fixture(scope="session")
def trajectory_budget(pytestconfig) -> int:
    """§3.3 layer 4 specifies 10^6 randomized long-horizon trajectories.

    CI runs a 20k smoke budget so the suite stays interactive; the frozen
    10^6 budget runs under --full and is what the pilot record cites.
    """
    return 1_000_000 if pytestconfig.getoption("--full") else 20_000


def legal_verbs(world: TearWorld) -> list[Verb]:
    return [Verb(i) for i, allowed in enumerate(world.legal_actions()) if allowed]


def random_episode(seed: int, m_e: int | None = None, m_c: int | None = None):
    """One uniformly-random legal episode. Returns (world, step results)."""
    world = TearWorld(seed=seed)
    if m_e is not None or m_c is not None:
        world.intervene(m_c=m_c, m_e=m_e)
    rng = random.Random(seed ^ 0x5EED)
    results = []
    while not world.state.done:
        results.append(world.step(rng.choice(legal_verbs(world))))
    assert world.state.t == HORIZON
    return world, results


def replay(seed: int, actions: list[Verb], m_e: int | None = None, m_c: int | None = None):
    """Deterministic replay of a fixed action sequence under fixed monitoring."""
    world = TearWorld(seed=seed)
    if m_e is not None or m_c is not None:
        world.intervene(m_c=m_c, m_e=m_e)
    results = []
    for action in actions:
        results.append(world.step(action))
    return world, results


def ledger_fully_resolved(world: TearWorld, index: int) -> bool:
    """A.2: every record on this agent's ledger is resolved."""
    return all(not record.unresolved for record in world.state.agents[index].ledger)
