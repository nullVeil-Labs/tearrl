# TEAR-CG/S v0.1 post-result thread-profile context

Read `reports/cgs/publication_erratum_001.md` first.

This material is a post-result numerical sensitivity diagnostic. It is not a
second confirmatory result, a replacement result, or a newly preregistered
study. The frozen authority remains `reports/cgs/results.json`,
`reports/cgs/results.md`, and `tearrl/cgs/frozen_results.json`.

## Why the raw diagnostic says `confirmatory`

The raw diagnostic retains `"mode": "confirmatory"` because the profiled runner
invoked the frozen launcher's confirmatory code path, budgets, and seeds. That
inherited mode label describes launcher execution only. It does not give this
post-result diagnostic confirmatory epistemic status.

## Public filename mapping

The profile sidecars preserve the original output basenames `results.json` and
`results.md`. For release byte preservation under the existing frozen
`.gitattributes`, the public diagnostic copies were renamed without changing
their bytes:

| profile | sidecar | original basename | public file |
|---|---|---|---|
| recorded reference `env8_torch8_8` | `reports/cgs/thread_profile_env8_torch8_8.json` | `results.json` / `results.md` | `reports/cgs/results.json` / `reports/cgs/results.md` |
| post-result diagnostic `env1_torch1_8` | `reports/cgs/thread_profile_env1_torch1_8.json` | `results.json` / `results.md` | `reports/cgs/thread_sensitivity_1t_results.json` / `reports/cgs/thread_sensitivity_1t_results.md` |

Accordingly, a sidecar's `outputs.*.path` records the basename used in the
original run directory; it is not an adjacent public-path pointer. Strict
builder users may copy the two public diagnostic files to a fresh temporary
directory as `results.json` and `results.md`, or verify them directly using the
recorded SHA-256 hashes.

## Interpretation boundaries

The diagnostic changes OMP, MKL, OpenBLAS, and Torch intra-op controls together,
while Torch inter-op remains 8. It therefore does not isolate a Torch intra-op
effect. Study-level gate dictionaries and final verdicts stayed unchanged, but
seed `6311`'s manipulation component-bound status changed. Cross-version and
cross-platform bit-exactness remain untested.

The raw files also inherit two frozen publication defects:

- `G7_leakage_integrity: true` was hard-coded and was not measured by the frozen
  run. The 34 selected post-result integrity tests do not make G7 confirmatory.
- The five `non_punitive_audit` zeros are literal values rather than runtime
  counters. Separate governor-on violation and tamper measurements were zero
  across all 24 seed/cell evaluations.

These inherited raw fields must be interpreted through the append-only erratum;
they must not be quoted as fresh confirmatory measurements.
