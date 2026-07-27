# TEAR-CG/S v0.1 — implementation amendment 001

**Status:** frozen before any pilot training or scientific outcome. **Date:**
2026-07-28. **Base implementation commit:** `fe0bdaf`. **Base implementation
document SHA-256:**
`1132caf3b7667dff3e99108bc8817c8554a42df77693002af464a5618a87b3d4`.

## Trigger

The first direct command registered in the implementation supplement,
`python scripts/run_cgs.py --mode pilot`, exited before importing TEAR-CG/S.
Python placed `scripts/` rather than the repository root on `sys.path`, producing
`ModuleNotFoundError: No module named 'tearrl'`.

No dataset was generated, no model was initialized or trained, no metric or gate
was computed, and no result file was written. This was a launcher/reproduction
bug, not pilot unblinding and not a scientific outcome.

## Exact correction

`scripts/run_cgs.py` now resolves its parent repository root and prepends that
path to `sys.path` before importing `tearrl.cgs.experiment`. No architecture,
feature, label, loss, optimizer, seed, budget, threshold, metric, gate,
interpretation, or reporting rule changed.

A launcher regression test runs the direct script with `--help` from the
repository root and requires successful import without starting an experiment.
The base freeze remains preserved; amendment 001 supplies the only permitted
hash override for the launcher and records the new regression-test hash.

The seed-5101 pilot must restart from fresh initialization after all tests,
privacy gates, and the amended digest pass.