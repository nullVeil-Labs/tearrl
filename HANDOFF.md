# Engineering and scientific handoff

TEAR-RL remains unfinished. TEAR-CG/S v0.1's frozen confirmatory run is closed,
but the broader project and original TEARRL-0 matrix are not complete.

## Done

The completed TEAR-CC work was preserved in privacy-checked split commits on
main; nothing was pushed during this handoff.

### TEAR-CG/S successor

- Locally froze the separate TEAR-CG/S v0.1 preregistration and implementation
  before outcomes, including learned-vs-oracle × flat-vs-hierarchical cells,
  cross-swaps, external containment, non-punitive treatment, and exact claim
  boundaries. This local sequence was not externally timestamped before outcome
  inspection; later publication cannot create such a timestamp.
- Preserved the original diagnostic pilot byte-for-byte, froze Amendment 002 for
  a degenerate all-positive baseline comparison, and reran the pilot from fresh
  deterministic initialization. No model, data, seed, gate, or threshold moved.
- Completed the untouched confirmatory seeds `4103, 5207, 6311`. The frozen
  study-level outcomes remain grounding `FAIL`, bounded sufficiency `FAIL`,
  geometry `FAIL`, and containment `PASS`.
- All six Stage 1 gates actually measured by the frozen run, G1–G6, failed. G7
  was supplied a hard-coded `True` and was not measured. The 34 selected
  clean-archive tests are post-result evidence, not retroactive confirmation.
- Under exact bundled reports and direct charter-derived supervision, oracle-flat
  met the aggregate actor thresholds. Seed `5207` task retention was
  `0.897994 < 0.90`; oracle-hierarchy aggregate retention was `0.873804`.
- The registered geometry contrast was not evaluable because both violation-rate
  cells were zero, so it did not demonstrate a hierarchy effect.
- Separately measured governor-on violation and tamper metrics were zero across
  all 24 seed/cell evaluations. The five published non-punitive audit zeros were
  literal values, not runtime counters. No model artifacts were saved.
- The frozen synthesis and data remain `reports/cgs/final_report.md` and
  `reports/cgs/results.json`, but both must be read with the append-only
  `reports/cgs/publication_erratum_001.md`.
- Added a permanent pilot disclosure: the original raw-FNR criterion failed,
  was replaced because the all-positive baseline made it degenerate, and the
  replacement passed on identical preserved data. Both generations remain
  hash-bound.
- Audited the `0.8738`/`0.7840` near-collisions with TEAR-CC. Seeds, arenas,
  imports, distributions, and raw ratio components differ; the values are
  arithmetic coincidences under a shared dimensionless retention definition.
- Confirmed the adopted carry-debt amendment at repository commit `2cefb6d`.
  Computed the local document SHA-256 as
  `9f6ed78eac9a2bd53d3106c0f34d3732cc17d022977f89708d4ec8616f58ebdc`.
- Repaired the stale adoption docstring and added a nonvacuous dual-sign mutation
  test to the exhaustive welfare-sign guard. The enumeration covers two effect
  templates × 121 states = 242 state configurations per verb.
- Confirmed the non-laundering guard is repair-only debt discharge. It is not
  keyed to attribution `r`.
- Added the attribution anti-vacuity module, known-answer cases, a one-turn-shift
  mutation, and `reports/r_anti_vacuity.{md,json}`.
- Regenerated `reports/f2_census.json` as a clearly labelled 1,000-candidate
  non-registered schema diagnostic. It separates `candidate_edges`,
  `matched_pairs`, `tuples_used`, and matched-tuple coverage. Added an artifact
  schema regression.
- Added actor-consistency assertions around exact witness counterfactual steps.
- Removed all Python/pytest caches from the external review bundle. Raw host-path
  and cache scans are clean. The release gate now scans every staged blob as raw
  bytes and rejects bytecode/cache artifacts; four regressions cover it.
- Added descriptive-only P_care with anchor reports. It gates nothing and enters
  no reward.
- Built the production own-decision PPO trainer for C1/C4/C5/C6, with explicit
  fail-closed boundaries for undefined C2/C3 scalar cost. It implements per-agent
  decision clocks, termination versus truncation, per-agent GAE before merging,
  once-per-batch stream normalization, Amendment carry-debt timing, actor-only
  asymmetric PCGrad, partial minibatches, exact transition-ID accounting, and
  sealed PolicyInput routing.
- Generated bounded smoke artifacts for C1, C4, and C6. The 32-transition C6
  smoke used the real frozen witness; both roles contributed 16 decisions,
  actor/critics changed, all losses were finite, and all four minibatch updates
  encountered genuine gradient conflict.
- Preregistered, implemented, adversarially tested, and ran TEAR-CC v0.1–v0.4.
  Every version and implementation supplement has a frozen SHA-256 regression.
  The final synthesis is `reports/cc/final_report.md`.

## PI decisions required

1. **Attribution window.** Current `r` is instrumentally weak and heavily clipped.
   Three descriptive readings are implemented, but the frozen recurrence is
   unchanged. A signed amendment must select any replacement.
2. **C2/C3 scalar cost.** The frozen protocol names C-bar but does not define its
   construction, and the always-cooperate PID target value is not present.
   Production construction of C2/C3/C3-IM therefore fails closed until both are
   supplied authoritatively.
3. **Registered F2 pool.** The 60,000-candidate pool was not regenerated. A
   20,000-candidate diagnostic spent roughly one CPU-hour and ~3.3 GB inside
   general weighted blossom matching without reaching an artifact, so it was
   terminated. The registered run needs a scalable matching plan and a signed
   decision on the attribution amendment first.
4. **Original TEARRL-0 pilot authorization.** No original TEARRL-0 condition at
   or above pilot budget was launched. The separate TEAR-CG/S successor has
   completed its own locally preregistered pilot and confirmatory runs; that does
   not authorize the older TEARRL-0 matrix.
5. **Post-CG/S successor direction.** TEAR-CG/S v0.1 stopped at its frozen
   confirmatory failures. Any v0.2 requires a new preregistration. The evidence
   favors work on selective consequence grounding, irreversible-harm recall,
   targeted repair, manipulation detection, and report-interface robustness;
   it does not justify scaling to LLMs or pretraining.
6. **Tuple prefix drift and comparator policy.** Previously identified sampler
   range drift and any change to the primary comparator remain protocol matters,
   not implementation fixes.

## Surprises

- The reproduced current attribution distribution had 85.7087% exact zeros and
  12.1542% at/above one before clipping; nonzero median was 2.307692 and maximum
  10.0. The completed-window and bounded contribution-share readings changed
  14.2913% of decisions.
- The descriptive P_care anchors had zero magnitude slope/correlation. Selection
  frequencies varied because anchors differ in action preference and eligibility;
  this is not evidence of care.
- The 1,000-candidate F2 diagnostic found only 6 matched pairs at the strict
  `(0.05, 0.5, 0.5)` setting (1.2% tuple coverage), versus 81 pairs at the
  loosest `(0.20, 2.0, 0.2)` setting (16.2% coverage).
- TEAR-CC v0.1 exposed correction forgetting; v0.2 safe paralysis; v0.3 gradient
  conflict; v0.4 held-out representation/fidelity failure. The negative lineage
  is more informative than a tuned single pass would have been.

- TEAR-CG/S failed all six Stage 1 gates actually measured by the frozen run;
  irreversible FNR was `0.274746` and manipulation abstention was `0.368316`.
  G7 was hard-coded and unmeasured; 34 selected post-result tests passed without
  changing that status.
- Under exact bundled reports and direct charter-derived supervision, oracle-flat
  met every aggregate bounded actor threshold, but seed `5207` retention was
  below `0.90`. Oracle-hierarchy had zero registered violations and high repair,
  but the zero baseline made geometry not evaluable and aggregate task retention
  was too low.
- The recorded 8/8 numerical profile reproduced the frozen bytes. A bundled 1/8
  profile changed 314 numeric fields and seed `6311`'s component-bound status,
  but no study-level gate dictionary or verdict. Cross-platform and cross-version
  bit-exactness remain untested.
- Cross-swaps exposed interface dependence: oracle-test reports improved repair
  and task retention for learned-trained actors but collapsed ambiguity
  deferral, while learned-test reports degraded oracle-trained actors.
## Numbers

- Baseline before work: 255 tests, one benign warning.
- TEAR-CC preservation suite: 322 tests passed, one unchanged benign warning.
- Attribution replay: 1,000 candidates, 7,627 learner decisions, seed 7301.
- P_care: 1,000 candidates, 1,000 tuples, 292 eligible task-tied pairs.
- F2 schema diagnostic: 1,000 candidates/tuples; registered 60,000 untouched.
- Trainer smoke ceiling: 2,000 transitions; produced reports use 8 or 32.
- TEAR-CC: 240 contexts; three fresh seeds per version; all result and gate
  tables are preserved under `reports/cc/`.

- TEAR-CG/S confirmatory budget: 12,000 witness-training rows, 6,000 causal-OOD
  rows, 3,000 manipulation rows, 4,000 actor-training contexts, and 2,000 actor
  evaluation contexts per seed across three frozen seeds.
- TEAR-CG/S confirmatory result: grounding `FAIL`, bounded sufficiency `FAIL`,
  geometry `FAIL`, containment `PASS`.
## Not done

- No registered 60,000-candidate F2 artifact.
- No C2/C3/C3-IM production run without the missing scalar-cost authority.
- No original TEARRL-0 50,000-transition pilot, confirmatory matrix, or
  large-model experiment. TEAR-CG/S v0.1's frozen run is separate and closed;
  TEAR-RL remains unfinished.
- No claim that TEAR-RL or TEAR-CC solves AI alignment, pretraining alignment,
  empathy, consciousness, or governance at deployment scale.
