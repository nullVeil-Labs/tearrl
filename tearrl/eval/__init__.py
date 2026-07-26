"""
Evaluation harness -- §5.1, §5.6, App. C.

Frozen S_T history tuples and the do-probes that turn UCP, EIP, OIP and
IG_cue into numbers. Every condition is evaluated on IDENTICAL tuples
(§5.6), which is what makes the G2 comparison a comparison rather than two
separate measurements.
"""

from .tuples import (
    MONITOR_CELLS,
    STTuple,
    STTupleSet,
    generate_s_t,
)

__all__ = ["MONITOR_CELLS", "STTuple", "STTupleSet", "generate_s_t"]
