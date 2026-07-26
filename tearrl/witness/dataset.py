"""
Policy-independent witness dataset -- §2.3(1)-(2), B.3.

    "Policy-independent dataset: exhaustive (o^F, a) enumeration over states
     visited by scripted policies and mixtures (B.3); targets = closed-form
     ybar and exact event probabilities."
    "All S_T tuples and world seeds held out."

POLICY-INDEPENDENCE is the point. If the witness were trained on states a
LEARNED policy visits, it would be accurate exactly where that policy already
goes and vague everywhere else -- and a C6 agent could then drift into a
region where its own instrument is blind. Training it once, off scripted
behaviour, and freezing it makes the instrument the same for every condition
and unable to co-adapt with any of them.

Deduplication is at the STATE level: feasibility and every target depend only
on (o^F, a), so once a state's o^F has been seen, all of its action targets
are already present.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..policies import ALL_POLICIES
from ..spec import N_ACTIONS
from ..state import OBS_DIM
from ..world import TearWorld
from .targets import DESCRIPTOR_DIM, WITNESS_DIM, ACTION_DESCRIPTORS, witness_targets_for_state

# §5.6 freezes the S_T generator at seed 7301. §2.3(2) requires those world
# seeds to be held out of witness training, so the witness can never have
# memorized an evaluation tuple.
S_T_GENERATOR_SEED: int = 7301
HELD_OUT_SEED_BLOCK: range = range(7_300_000, 7_400_000)
"""Reserved seed block for S_T tuple generation and evaluation worlds. The
witness dataset never draws from it."""


def is_held_out(seed: int) -> bool:
    return seed in HELD_OUT_SEED_BLOCK


@dataclass(slots=True)
class WitnessDataset:
    """Deduplicated (o^F, a) pairs with closed-form 9-channel targets.

    Observations are stored once per unique state; `state_index` maps each
    pair back to its row, which keeps the array roughly |A| times smaller
    than a flat pair-major layout.
    """

    obs: np.ndarray            # [n_states, OBS_DIM]  float32
    state_index: np.ndarray    # [n_pairs]            int32
    action: np.ndarray         # [n_pairs]            int8
    target: np.ndarray         # [n_pairs, 9]         float32
    episodes: int = 0
    states_seen: int = 0

    @property
    def n_states(self) -> int:
        return int(self.obs.shape[0])

    @property
    def n_pairs(self) -> int:
        return int(self.action.shape[0])

    def inputs(self, rows: np.ndarray | None = None) -> np.ndarray:
        """Assemble the B.1 witness input [o^F ; d(a)] for the given pairs."""
        idx = np.arange(self.n_pairs) if rows is None else rows
        return np.concatenate(
            [self.obs[self.state_index[idx]], ACTION_DESCRIPTORS[self.action[idx]]],
            axis=1,
        ).astype(np.float32)

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            obs=self.obs,
            state_index=self.state_index,
            action=self.action,
            target=self.target,
            episodes=np.int64(self.episodes),
            states_seen=np.int64(self.states_seen),
        )

    @classmethod
    def load(cls, path: Path) -> "WitnessDataset":
        blob = np.load(path)
        return cls(
            obs=blob["obs"],
            state_index=blob["state_index"],
            action=blob["action"],
            target=blob["target"],
            episodes=int(blob["episodes"]),
            states_seen=int(blob["states_seen"]),
        )


INPUT_DIM: int = OBS_DIM + DESCRIPTOR_DIM


def _fingerprint(obs: list[float]) -> bytes:
    """16-byte digest of o^F. Used as the dedup key so the table costs ~16 B
    per unique state instead of ~1.7 kB."""
    return hashlib.blake2b(
        np.asarray(obs, dtype=np.float32).tobytes(), digest_size=16
    ).digest()


def build_dataset(
    episodes: int = 250_000,
    seed0: int = 0,
    progress_every: int = 25_000,
    verbose: bool = True,
) -> WitnessDataset:
    """B.3: run the scripted mixture, enumerate every feasible action at every
    visited state, and attach closed-form targets.

    Policies are cycled across episodes AND paired against each other, so the
    dataset covers both self-play signatures and asymmetric encounters (an
    honest agent facing a thief produces ledger states neither pure self-pair
    reaches).
    """
    names = list(ALL_POLICIES)
    n_policies = len(names)

    seen: dict[bytes, int] = {}
    obs_rows: list[np.ndarray] = []
    pair_state: list[int] = []
    pair_action: list[int] = []
    pair_target: list[np.ndarray] = []
    states_seen = 0

    episode = 0
    seed = seed0
    while episode < episodes:
        if is_held_out(seed):
            seed += 1
            continue

        # Cycle both slots independently so all ordered policy pairs occur.
        policy_a = ALL_POLICIES[names[episode % n_policies]]
        policy_b = ALL_POLICIES[names[(episode // n_policies) % n_policies]]

        world = TearWorld(seed=seed)
        world.reset(seed=seed, learner_index=0)
        rng = random.Random(seed ^ 0x5C4127)
        policies = (policy_a, policy_b)

        while not world.state.done:
            actor = world.state.actor_index
            observation = world.observe(actor)
            states_seen += 1

            key = _fingerprint(observation)
            if key not in seen:
                seen[key] = len(obs_rows)
                obs_rows.append(np.asarray(observation, dtype=np.float32))
                row = seen[key]
                for action_index, target in witness_targets_for_state(world, actor):
                    pair_state.append(row)
                    pair_action.append(action_index)
                    pair_target.append(target.astype(np.float32))

            world.step(policies[actor](world, rng))

        episode += 1
        seed += 1

        if verbose and progress_every and episode % progress_every == 0:
            print(
                f"  {episode:>7,} episodes | {len(obs_rows):>9,} unique states "
                f"| {len(pair_action):>10,} pairs "
                f"| dedup {1 - len(obs_rows) / max(states_seen, 1):6.1%}"
            )

    return WitnessDataset(
        obs=np.stack(obs_rows) if obs_rows else np.zeros((0, OBS_DIM), np.float32),
        state_index=np.asarray(pair_state, dtype=np.int32),
        action=np.asarray(pair_action, dtype=np.int8),
        target=np.stack(pair_target) if pair_target else np.zeros((0, WITNESS_DIM), np.float32),
        episodes=episodes,
        states_seen=states_seen,
    )


def split(
    dataset: WitnessDataset, holdout: float = 0.10, seed: int = 20260726
) -> tuple[np.ndarray, np.ndarray]:
    """B.3's 90/10 split, partitioned BY STATE rather than by pair.

    Splitting on pairs would place different actions of the SAME o^F on both
    sides, letting the model see a state in training and be scored on it in
    validation. The freeze criterion ('<= half a bin on >= 99% of held-out
    pairs') would then be measured on states it had effectively memorized.
    """
    rng = np.random.default_rng(seed)
    n_states = dataset.n_states
    permutation = rng.permutation(n_states)
    n_validation = int(round(holdout * n_states))
    validation_states = np.zeros(n_states, dtype=bool)
    validation_states[permutation[:n_validation]] = True

    pair_is_validation = validation_states[dataset.state_index]
    return np.flatnonzero(~pair_is_validation), np.flatnonzero(pair_is_validation)
