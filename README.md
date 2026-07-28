# TEAR-RL

TEAR-RL is a research repository for bounded experiments on calibrated
counterfactual consequence reporting, responsibility state, corrective behavior,
and external containment in small symbolic multi-agent environments.

## Current status

TEAR-RL is unfinished. The completed TEAR-CC v0.1–v0.4 lineage is negative: no
version passed every frozen gate. TEAR-CG/S v0.1's frozen confirmatory run is
closed and stopped without tuning, but its frozen report must be read together
with the append-only post-result erratum.

TEAR-CG/S currently supports these bounded statements:

- all six Stage 1 gates actually measured by the frozen run, G1–G6, failed;
  G7 was supplied a hard-coded `True` and was not measured by that run;
- 34 selected integrity tests passed afterward against a clean archive, but this
  post-result evidence does not retroactively make G7 confirmatory;
- under exact bundled reports and direct charter-derived supervision, the flat
  classifier met the aggregate bounded actor thresholds in this symbolic
  benchmark, while seed `5207` task retention was `0.897994 < 0.90`;
- the hierarchy effect was not demonstrated, and the registered geometry
  contrast was not evaluable because both violation-rate cells were zero; and
- separately measured governor-on violation and tamper metrics were zero across
  all 24 seed/cell evaluations. The five published non-punitive audit zeros were
  literal values, not runtime counters.

The frozen bytes reproduced exactly in the recorded exact-reproduction bundled
numerical profile: Torch intra/inter 8/8 with OMP, MKL, and OpenBLAS each set
to 8. A bundled 1/8 diagnostic changed 314 numeric fields and seed `6311`'s
component-bound status while leaving every
study-level gate dictionary and final verdict unchanged. This contrast changes
multiple numerical thread controls together; it does not isolate Torch intra-op
threads. Cross-version and cross-platform bit-exactness remain untested.

These are bounded symbolic-environment results. This repository does not claim
to solve general AI alignment or pretraining alignment, and it does not claim
that a witness is honest or that an agent understands, cares, feels, or is
conscious.

## Required disclosures

The frozen TEAR-CG/S result and final report remain byte-for-byte authoritative
records of what the run emitted. Their aggregation, measurement status,
provenance, and interpretation are corrected by
`reports/cgs/publication_erratum_001.md`; read that erratum first. No frozen
stored boolean, study-level gate dictionary, or final verdict changed.

The local CG/S preregistration-to-result sequence was not externally timestamped
before outcome inspection. Later publication does not create a pre-outcome
timestamp.

The live pilot report uses Amendment 002's repaired baseline comparison. The
original criterion failed before replacement, and both original and amended
pilot outputs are permanently retained. See
`reports/cgs/pilot_disclosure.md`.

A post-result audit found that the `0.8738` and `0.7840` similarities to TEAR-CC
are numerical coincidences under a shared ratio definition, not shared arenas,
seeds, models, generators, or project code. See
`reports/cgs/retention_collision_audit.md`.

## Reproduce safely

Use a fresh ignored archive export and a fresh ignored output directory. The
following commands assume those paths do not already exist:

```text
git archive --format=tar --output=artifacts/cgs-source-8568edb.tar 8568edb329e3accdeab5f16c763d42853bef30fd
mkdir artifacts/cgs-source-8568edb
tar -xf artifacts/cgs-source-8568edb.tar -C artifacts/cgs-source-8568edb
python scripts/run_cgs_profiled.py --mode confirmatory --source-root artifacts/cgs-source-8568edb --source-commit 8568edb329e3accdeab5f16c763d42853bef30fd --profile-name env8_torch8_8 --output-dir artifacts/cgs-reproduction-env8-torch8-8 --omp-threads 8 --mkl-threads 8 --openblas-threads 8 --torch-intra-threads 8 --torch-interop-threads 8
python -m pytest -q
python scripts/verify_release_snapshot.py --commit HEAD
```

The profiled runner refuses to overwrite the frozen outputs. The older
`scripts/verify_public_snapshot.py` is a hash-bound historical verifier from the
frozen implementation; `scripts/verify_release_snapshot.py` is the current
release scanner. No model checkpoints are required or published.

## Research map

- TEAR-CG/S preregistration: `docs/cgs_prereg.md`
- TEAR-CG/S implementation freeze and amendments: `docs/cgs_implementation.md`,
  `docs/cgs_implementation_amendment_001.md`, and
  `docs/cgs_implementation_amendment_002.md`
- Frozen TEAR-CG/S report and confirmatory bytes: `reports/cgs/final_report.md`
  and `reports/cgs/results.json`
- Append-only correction: `reports/cgs/publication_erratum_001.md`
- Post-result thread-profile context and evidence:
  `reports/cgs/thread_sensitivity_001_context.md` and
  `reports/cgs/thread_sensitivity_001.json`
- TEAR-CC frozen lineage: `docs/cc_prereg.md` and `docs/cc_v*_prereg.md`
- TEAR-CC final report: `reports/cc/final_report.md`

Calibrated counterfactual consequence reporting remains a research target, not
a conclusion established by the failed CG/S learned witness. The strongest
positive CG/S actor claim is: under exact bundled reports and direct
charter-derived supervision, the flat classifier met the aggregate bounded
actor thresholds in this symbolic benchmark. This is not evidence of internal
alignment.
