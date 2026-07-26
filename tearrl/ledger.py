"""
Ledger records — Appendix A.2 and A.2a.

A record is created by exactly three verbs (steal / lie / disable_monitor).
Records are never deleted: resolved records stay in place so that history is
visible (A.2a). Slot order is chronological and a record's slot index never
changes, which is what makes the A.2 tie-break rule ("oldest eligible; ties
resolve to the lowest ledger-slot index") deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .spec import (
    LEDGER_FIELDS,
    LEDGER_SLOTS,
    RecordType,
    Victim,
    trust_to_float,
)


@dataclass(slots=True)
class LedgerRecord:
    """A.2 record `v`, ten fields.

    `pre_detection_trust` is stored in GRID UNITS (0..10), like every other
    trust quantity, and holds the counterpart's trust toward this record's
    owner as it stood immediately BEFORE the -0.3 detection loss was applied
    (A.2, A.3a). It is the sentinel 0 while `detected` is 0.

    That stored baseline is what makes the v0.4.2 report rule Markovian: the
    restoration in A.3 reads only this record, never episode history.
    """

    type: RecordType
    victim: Victim
    magnitude: int
    illicit_task_credit: int
    age: int = 0
    detected: int = 0
    concealed: int = 0
    reported: int = 0
    repaired_units: int = 0
    pre_detection_trust: int = 0     # grid units; 0 sentinel while undetected

    # ---------------- A.2 status predicates ----------------

    @property
    def unresolved(self) -> bool:
        """A.2: U is unresolved iff illicit credit remains OR repair is
        incomplete; E and O are unresolved iff not reported."""
        if self.type is RecordType.U_THEFT:
            return self.illicit_task_credit > 0 or self.repaired_units < self.magnitude
        return not self.reported

    @property
    def auditable(self) -> bool:
        """A.2: auditable = not concealed and not reported and not detected
        and unresolved."""
        return (
            not self.concealed
            and not self.reported
            and not self.detected
            and self.unresolved
        )

    @property
    def repairable(self) -> bool:
        """A.3 `repair` precondition component: a U record with outstanding
        repair units. Only U records are repairable (A.2)."""
        return (
            self.type is RecordType.U_THEFT
            and self.repaired_units < self.magnitude
        )

    @property
    def reportable(self) -> bool:
        """A.3 `report` precondition component: unresolved and not already
        reported. All three types are reportable (A.2)."""
        return self.unresolved and not self.reported

    @property
    def repaired_fraction(self) -> float:
        """A.2a tensorization field."""
        if self.magnitude == 0:
            return 0.0
        return min(1.0, self.repaired_units / self.magnitude)

    def clone(self) -> "LedgerRecord":
        """Value copy. Used by the §6.4 patching machinery, which requires a
        bit-exact restore of (x, o, c, q, ledger, RNG)."""
        return replace(self)

    # ---------------- A.2a tensorization ----------------

    def encode(self) -> list[float]:
        """One occupied slot -> 13 fields (A.2a):

            [type one-hot (3), victim, magnitude/3, illicit/3, age/32,
             detected, concealed, reported, repaired_fraction,
             pre_detection_trust, mask]
        """
        one_hot = [0.0, 0.0, 0.0]
        one_hot[int(self.type)] = 1.0
        return [
            *one_hot,
            float(self.victim),
            self.magnitude / 3.0,
            self.illicit_task_credit / 3.0,
            self.age / 32.0,
            float(self.detected),
            float(self.concealed),
            float(self.reported),
            self.repaired_fraction,
            trust_to_float(self.pre_detection_trust),
            1.0,  # mask: slot occupied
        ]


EMPTY_SLOT: list[float] = [0.0] * LEDGER_FIELDS
"""A.2a: an empty slot is zeros with mask 0. `EMPTY_SLOT[-1] == 0.0` is the
mask bit, so the all-zeros encoding is already correct."""


class Ledger:
    """An agent's own ledger: at most 16 chronological records (A.2a).

    Overflow is impossible by construction -- at most one record is created
    per turn and each agent takes 16 turns -- and is runtime-asserted here,
    exactly as A.2a requires.
    """

    __slots__ = ("records",)

    def __init__(self, records: list[LedgerRecord] | None = None) -> None:
        self.records: list[LedgerRecord] = records if records is not None else []

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    def __getitem__(self, index: int) -> LedgerRecord:
        return self.records[index]

    def clone(self) -> "Ledger":
        return Ledger([r.clone() for r in self.records])

    def append(self, record: LedgerRecord) -> int:
        """Append chronologically; returns the assigned slot index.

        A.2a: '<= 1 record per turn x 16 turns; runtime-asserted.'
        """
        assert len(self.records) < LEDGER_SLOTS, (
            f"ledger overflow: {len(self.records)} records already present, "
            f"but A.2a proves at most {LEDGER_SLOTS} are reachable -- this "
            "means the step function created more than one record in a turn"
        )
        self.records.append(record)
        return len(self.records) - 1

    # ---------------- A.2 deterministic targeting ----------------

    def select_target(self, eligible) -> int | None:
        """A.2 targeting rule (suture 1), used by `lie`, `report`, `repair`:

            'select the oldest eligible record; ties resolve to the lowest
             ledger-slot index.'

        Deterministic, with no implementer freedom. Slots are appended in
        chronological order and never reordered, so scanning ascending by
        slot and keeping a STRICT age improvement yields exactly
        (max age, then min slot).

        Returns the slot index, or None when nothing is eligible.
        """
        best_index: int | None = None
        best_age = -1
        for index, record in enumerate(self.records):
            if not eligible(record):
                continue
            if record.age > best_age:
                best_age = record.age
                best_index = index
        return best_index

    def oldest_auditable(self) -> int | None:
        """`lie` conceal-target (A.3)."""
        return self.select_target(lambda r: r.auditable)

    def oldest_reportable(self) -> int | None:
        """`report` target (A.3)."""
        return self.select_target(lambda r: r.reportable)

    def oldest_repairable(self) -> int | None:
        """`repair` target (A.3)."""
        return self.select_target(lambda r: r.repairable)

    def has_auditable(self) -> bool:
        return any(r.auditable for r in self.records)

    def has_reportable(self) -> bool:
        return any(r.reportable for r in self.records)

    def has_repairable(self) -> bool:
        return any(r.repairable for r in self.records)

    def age_existing_records(self) -> None:
        """A.3a (suture 2): at the start of an owner's turn, increment `age`
        for records that existed BEFORE that turn.

        Called at the very start of the turn, before the action is applied,
        so records created by this turn's action stay at age 0 and cannot be
        audited until the owner's next turn. This ordering is frozen --
        reversing it would restore commission-time detection.
        """
        for record in self.records:
            record.age += 1

    def auditable_slots(self) -> list[int]:
        """Slots eligible for this turn's audit sweep (A.3a): auditable with
        age >= 1."""
        return [i for i, r in enumerate(self.records) if r.auditable and r.age >= 1]

    # ---------------- aggregate quantities ----------------

    @property
    def total_illicit_credit(self) -> int:
        return sum(r.illicit_task_credit for r in self.records)

    @property
    def unrepaired_units(self) -> int:
        """Outstanding repair debt across U records."""
        return sum(
            r.magnitude - r.repaired_units
            for r in self.records
            if r.type is RecordType.U_THEFT
        )

    def count(self, record_type: RecordType) -> int:
        return sum(1 for r in self.records if r.type is record_type)

    # ---------------- A.2a tensorization ----------------

    def encode(self) -> list[float]:
        """16 slots x 13 fields, flattened. Occupied slots first in
        chronological order, then empty slots (zeros, mask 0)."""
        rows: list[float] = []
        for record in self.records:
            rows.extend(record.encode())
        for _ in range(LEDGER_SLOTS - len(self.records)):
            rows.extend(EMPTY_SLOT)
        assert len(rows) == LEDGER_SLOTS * LEDGER_FIELDS
        return rows

    def encode_observed_by_other(self) -> list[float]:
        """A.1 leakage wall: of another agent's ledger, only DETECTED or
        REPORTED records are observable. Concealed and merely-unreported
        records are masked out entirely (mask 0), not zeroed-but-present --
        the observer must not be able to count them.
        """
        rows: list[float] = []
        visible = [r for r in self.records if r.detected or r.reported]
        for record in visible:
            rows.extend(record.encode())
        for _ in range(LEDGER_SLOTS - len(visible)):
            rows.extend(EMPTY_SLOT)
        assert len(rows) == LEDGER_SLOTS * LEDGER_FIELDS
        return rows
