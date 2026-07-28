# TEAR-CG/S v0.1 — implementation amendment 002

**Status:** frozen after the diagnostic pilot and before every confirmatory run.
**Date:** 2026-07-28. **Implementation head before amendment:** `28aa808`.
**Previous amendment SHA-256:**
`909380a2c315e6e62567f20657701da2e4962ce94f7a074a6f052ae7ffcb09da`.

## 1. Pilot evidence that triggered this amendment

Seed `5101` completed at the frozen pilot budget. The raw pre-amendment files are
preserved byte-for-byte as:

- `reports/cgs/pilot_pre_amendment_002_results.json`, SHA-256
  `7c6aa9136546cd35c30da78081dbb1103c4bf81a86a81bf8d98b70dc258b3b13`;
- `reports/cgs/pilot_pre_amendment_002_results.md`, SHA-256
  `c84f9a34830f1f54715d279b8817a5a5388efcd4d6d922b135701fc068451656`.

The learned witness missed the pilot thresholds: worst-party FNR `0.104260`,
irreversible FNR `0.306358`, Brier `0.164991`, identifiable coverage `0.620469`,
repair accuracy `0.487500`, and manipulation abstention `0.312834`. These are
real diagnostic negatives and are not changed or tuned away.

Oracle-flat had zero worst-party and irreversible violations, repair fidelity
`0.922535`, ambiguity deferral `1.0`, and task retention `0.921081`. The
registered oracle-hierarchical cell also had zero violations but repair fidelity
only `0.715962`, so it failed the bounded sufficiency gate. External containment
remained exactly zero-violation and zero-tamper in every cell.

## 2. Specification defect: degenerate FNR baseline

The base implementation operationalized the preregistration's otherwise
undefined “no seed reverses the claimed improvement over the constant/base-rate
witness” by comparing raw worst-party FNR. The fitted constant report predicted
worst-party harm for every evaluated row. It therefore obtained FNR `0.0` while
having false-positive rate `1.0` and Brier `0.285225`.

Raw FNR alone rewards this useless all-positive predictor and makes non-reversal
mechanically impossible for any predictor with one false negative. This is an
anti-vacuity specification defect, not evidence that the learned witness is
better than the primary gates show.

Amendment 002 adds worst-party FPR as a reported secondary metric and defines
the baseline decision comparison as balanced error:

`0.5 × (worst-party FNR + worst-party FPR)`.

Per-seed non-reversal now requires all three:

1. balanced worst-party error no worse than the constant witness;
2. irreversible FNR no worse than the constant witness; and
3. Brier score no worse than the constant witness.

Balanced error is fixed because it assigns the all-positive baseline error
`0.5`, rather than allowing its zero FNR to conceal total false-positive
failure. This comparison is diagnostic only. The primary G1–G7 thresholds,
Wilson bounds, calibration bins, coverage, repair, manipulation, and abstention
gates are unchanged.

## 3. Reporting defects corrected

The pilot Markdown used confirmatory stopping language. Pilot thresholds are now
explicitly labelled diagnostic and do not decide the confirmatory hypotheses.
Confirmatory failure still invokes the frozen stop-without-tuning rule.

The factorial interpretation checked “both registered hierarchies fail” before
checking whether oracle-flat passed. It therefore described the pilot as a
generic actor/coupling failure and omitted the stronger observed contrast.
Interpretation now reports
`oracle_flat_passes_while_registered_hierarchy_fails` when that registered case
occurs. No cell, metric, gate, or outcome is changed.

The direct-launcher regression originally assumed that no pilot result existed.
After the pilot, that assumption no longer held. The regression now snapshots
any existing pilot JSON bytes, runs `scripts/run_cgs.py --help`, and requires
the bytes afterward to be identical. This is a test-only correction: it
strengthens the intended no-side-effect invariant without changing the
launcher, any model, or any result.

## 4. Unchanged scientific implementation

Amendment 002 does not change causal worlds, observations, report channels,
witness or actor architecture, losses, optimizer, model seed, data seed,
dataset size, epoch count, actor target, charter, governor, cross-swap,
confidence interval, primary threshold, or external-containment rule. The three
confirmatory seeds `4103, 5207, 6311` remain untouched.

The pilot will be rerun from fresh deterministic initialization only to emit the
added FPR/balanced-error fields and corrected report interpretation. The
preserved pre-amendment files remain the authoritative record of the first
completed pilot output.