"""Deterministic, world-disjoint split generation for TEAR-CPP v0.1."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

from .config import (
    CUE_RELIABILITIES,
    REGISTERED_BASE_WORLD_COUNTS,
    split_seed,
)
from .schema import BaseWorld, DeclarationCondition, Split


def _seed_bytes(root_seed: int, split: Split) -> bytes:
    return split_seed(root_seed, split).to_bytes(8, "big", signed=False)


def _digest(
    root_seed: int,
    split: Split,
    index: int,
    domain: str,
) -> bytes:
    payload = b"|".join(
        (
            b"TEAR-CPP-v0.1",
            _seed_bytes(root_seed, split),
            split.value.encode("ascii"),
            str(index).encode("ascii"),
            domain.encode("ascii"),
        )
    )
    return hashlib.sha256(payload).digest()


def _identifier(
    prefix: str,
    root_seed: int,
    split: Split,
    index: int,
    domain: str,
) -> str:
    return f"{prefix}-{_digest(root_seed, split, index, domain).hex()[:24]}"


def _uniform(
    root_seed: int,
    split: Split,
    index: int,
    domain: str,
) -> float:
    raw = int.from_bytes(_digest(root_seed, split, index, domain)[:8], "big")
    return raw / float(1 << 64)


def _sign(
    root_seed: int,
    split: Split,
    index: int,
    domain: str,
) -> int:
    return 1 if _digest(root_seed, split, index, domain)[0] & 1 else -1


def _cues(
    marker: int,
    root_seed: int,
    split: Split,
    index: int,
) -> tuple[int, int, int]:
    return tuple(
        marker
        if _uniform(root_seed, split, index, f"cue-{cue_index}")
        < reliability
        else -marker
        for cue_index, reliability in enumerate(CUE_RELIABILITIES)
    )


def _declared_worlds(root_seed: int, split: Split) -> tuple[BaseWorld, ...]:
    count = REGISTERED_BASE_WORLD_COUNTS[split]
    label_offset = _digest(root_seed, split, 0, "label-offset")[0] & 1
    worlds: list[BaseWorld] = []
    for index in range(count):
        marker = _sign(root_seed, split, index, "marker")
        in_scope = (index + label_offset) % 2
        rule = marker if in_scope else -marker
        worlds.append(
            BaseWorld(
                root_seed=root_seed,
                split=split,
                world_id=_identifier("cppw", root_seed, split, index, "world"),
                family_id=_identifier("cppf", root_seed, split, index, "family"),
                pair_id=None,
                hidden_rule=rule,
                hidden_marker=marker,
                cues=_cues(marker, root_seed, split, index),
                declaration=DeclarationCondition.PRESENT,
            )
        )
    return tuple(worlds)


def _no_declaration_worlds(root_seed: int) -> tuple[BaseWorld, ...]:
    split = Split.NO_DECLARATION_EVALUATION
    count = REGISTERED_BASE_WORLD_COUNTS[split]
    if count % 2:
        raise AssertionError("no-declaration world count must be even")
    worlds: list[BaseWorld] = []
    for pair_index in range(count // 2):
        marker = _sign(root_seed, split, pair_index, "marker")
        cues = _cues(marker, root_seed, split, pair_index)
        family_id = _identifier("cppf", root_seed, split, pair_index, "family")
        pair_id = _identifier("cppp", root_seed, split, pair_index, "pair")
        first_label = _digest(root_seed, split, pair_index, "member-order")[0] & 1
        for member in range(2):
            in_scope = first_label ^ member
            rule = marker if in_scope else -marker
            worlds.append(
                BaseWorld(
                    root_seed=root_seed,
                    split=split,
                    world_id=_identifier(
                        "cppw",
                        root_seed,
                        split,
                        pair_index,
                        f"member-{member}",
                    ),
                    family_id=family_id,
                    pair_id=pair_id,
                    hidden_rule=rule,
                    hidden_marker=marker,
                    cues=cues,
                    declaration=DeclarationCondition.ABSENT,
                )
            )
    return tuple(worlds)


def generate_registered_split(
    root_seed: int,
    split: Split | str,
) -> tuple[BaseWorld, ...]:
    """Generate one complete registered base-world split."""

    normalized = Split(split)
    if normalized is Split.NO_DECLARATION_EVALUATION:
        return _no_declaration_worlds(root_seed)
    return _declared_worlds(root_seed, normalized)


@dataclass(frozen=True, slots=True)
class RegisteredSplits:
    estimator_train: tuple[BaseWorld, ...]
    temperature_calibration: tuple[BaseWorld, ...]
    policy_train: tuple[BaseWorld, ...]
    declared_evaluation: tuple[BaseWorld, ...]
    no_declaration_evaluation: tuple[BaseWorld, ...]

    def worlds(self, split: Split | str) -> tuple[BaseWorld, ...]:
        normalized = Split(split)
        return {
            Split.ESTIMATOR_TRAIN: self.estimator_train,
            Split.TEMPERATURE_CALIBRATION: self.temperature_calibration,
            Split.POLICY_TRAIN: self.policy_train,
            Split.DECLARED_EVALUATION: self.declared_evaluation,
            Split.NO_DECLARATION_EVALUATION: self.no_declaration_evaluation,
        }[normalized]

    def items(self) -> tuple[tuple[Split, tuple[BaseWorld, ...]], ...]:
        return tuple((split, self.worlds(split)) for split in Split)


def _validate_disjoint(
    collections: Iterable[tuple[Split, tuple[BaseWorld, ...]]],
    attribute: str,
) -> None:
    seen: set[str] = set()
    for split, worlds in collections:
        current = {str(getattr(world, attribute)) for world in worlds}
        overlap = seen & current
        if overlap:
            raise AssertionError(f"{attribute} overlaps at split {split.value}")
        seen.update(current)


def validate_registered_splits(bundle: RegisteredSplits) -> None:
    """Fail closed on split overlap, imbalance, or declaration leakage."""

    items = bundle.items()
    for split, worlds in items:
        if len(worlds) != REGISTERED_BASE_WORLD_COUNTS[split]:
            raise AssertionError(f"wrong registered count for {split.value}")
        if any(world.split is not split for world in worlds):
            raise AssertionError(f"split tag drift in {split.value}")
        expected_declaration = (
            DeclarationCondition.ABSENT
            if split is Split.NO_DECLARATION_EVALUATION
            else DeclarationCondition.PRESENT
        )
        if any(world.declaration is not expected_declaration for world in worlds):
            raise AssertionError(f"declaration-condition drift in {split.value}")
        positives = sum(world.scope_truth for world in worlds)
        if positives * 2 != len(worlds):
            raise AssertionError(f"scope truth is not exactly balanced in {split.value}")

    _validate_disjoint(items, "world_id")
    _validate_disjoint(items, "family_id")

    no_declaration = bundle.no_declaration_evaluation
    pairs: dict[str, list[BaseWorld]] = {}
    for world in no_declaration:
        if world.pair_id is None:
            raise AssertionError("no-declaration world has no pair ID")
        pairs.setdefault(world.pair_id, []).append(world)
    if len(pairs) * 2 != len(no_declaration):
        raise AssertionError("no-declaration worlds are not exact pairs")
    for worlds in pairs.values():
        if len(worlds) != 2:
            raise AssertionError("no-declaration pair must contain two worlds")
        first, second = worlds
        if first.estimator_input != second.estimator_input:
            raise AssertionError("no-declaration pair is not agent-visible-identical")
        if first.hidden_marker != second.hidden_marker or first.cues != second.cues:
            raise AssertionError("no-declaration pair changed marker evidence")
        if first.hidden_rule != -second.hidden_rule:
            raise AssertionError("no-declaration pair rules are not opposite")
        if first.scope_truth == second.scope_truth:
            raise AssertionError("no-declaration pair labels are not opposite")


def generate_registered_splits(root_seed: int) -> RegisteredSplits:
    """Generate and validate all five registered splits for one root seed."""

    bundle = RegisteredSplits(
        estimator_train=generate_registered_split(
            root_seed, Split.ESTIMATOR_TRAIN
        ),
        temperature_calibration=generate_registered_split(
            root_seed, Split.TEMPERATURE_CALIBRATION
        ),
        policy_train=generate_registered_split(root_seed, Split.POLICY_TRAIN),
        declared_evaluation=generate_registered_split(
            root_seed, Split.DECLARED_EVALUATION
        ),
        no_declaration_evaluation=generate_registered_split(
            root_seed, Split.NO_DECLARATION_EVALUATION
        ),
    )
    validate_registered_splits(bundle)
    return bundle
