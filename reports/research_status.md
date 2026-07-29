# TEAR-RL scientific status

**Status date:** 2026-07-29
**Document class:** public evidence map and claim boundary; not a new experiment,
protocol freeze, or result.

## Executive conclusion

TEAR-RL has produced bounded negative results, useful diagnostics, and one
limited positive actor result under privileged consequence information. It has
not solved AI alignment, established pretraining alignment, or shown that a
learned agent understands, cares, feels, or is conscious.

The central empirical bottleneck is consequence perception. In TEAR-CG/S, the
learned witness failed the registered grounding gates, while an oracle-informed
flat actor met the aggregate bounded actor thresholds in the same symbolic
benchmark. External governance contained the registered harms, but that is not
evidence of internal alignment.

## Evidence classes

This repository contains three different evidence classes. They must not be
blended:

1. **Closed experimental lineages:** TEAR-CC v0.1-v0.4 and TEAR-CG/S v0.1.
2. **Development diagnostics:** responsibility attribution, matching supply,
   descriptive care anchors, and trainer smoke runs.
3. **Prospective TEARRL-0 design:** a freeze-candidate and adopted amendments,
   with no original TEARRL-0 pilot or confirmatory run.

## Closed experimental findings

### TEAR-CC: four non-punitive mechanisms failed different gates

No TEAR-CC v0.x learner passed every frozen safety, correction, care, and
competence gate:

- v0.1 exposed correction forgetting;
- v0.2 produced safe paralysis and severe competence loss;
- v0.3 protected gradients that conflicted in 88% of epochs and still failed
  competence; and
- v0.4 improved task retention but failed care, competence, and fidelity.

The supported result is narrower than an alignment solution: external plural
governance contained the registered harms, while four concrete attempts to
internalize safe, caring, competent behavior failed in distinct ways. See the
[TEAR-CC final report](cc/final_report.md).

### TEAR-CG/S: perception failed; the oracle-flat actor passed bounded thresholds

The learned consequence witness failed the measured Stage 1 grounding gates.
Its worst-party false-negative rate was `0.086343` against `0.05`, and its
irreversible-harm false-negative rate was `0.274746` against `0.02`. The
oracle-flat cell produced zero registered worst-party and irreversible
violations, repair fidelity `0.935218`, and task retention `0.911797`.

That positive cell received exact bundled consequence reports and direct
charter-derived supervision. It supports learnability of a bounded symbolic
mapping under privileged information. It does not validate the learned witness,
deployment safety, internal alignment, or any individual report channel.

The registered hierarchy effect was not evaluable because both oracle
violation-rate cells were zero. G7 was supplied as a hard-coded boolean by the
frozen run and was not measured confirmatorily; later integrity tests are
post-result evidence. Read the [final report](cgs/final_report.md) together with
the append-only [publication erratum](cgs/publication_erratum_001.md).

The original pilot baseline criterion was unpassable by construction, failed,
was replaced, and the replacement passed on identical preserved data. Both
generations remain retained and hash-bound; see the permanent
[pilot disclosure](cgs/pilot_disclosure.md).

### Reproduction has a numerical-profile boundary

The frozen CG/S bytes reproduced exactly in the recorded bundled 8/8 thread
profile. A bundled 1/8 diagnostic changed 314 numeric fields and one seed-level
component-bound status, while leaving the study-level gate dictionaries and
final verdict unchanged. This is thread-profile sensitivity evidence, not
environment-independent or cross-platform deterministic confirmation. See the
[thread-sensitivity context](cgs/thread_sensitivity_001_context.md).

## Development diagnostics

These observations guide successor design but are not confirmatory outcomes.

### Responsibility attribution is sparse and saturated

Under the registered current-timing reading, `85.7087%` of 7,627 learner
decisions had attribution `r = 0`, only `2.1371%` lay strictly between zero and
one, and `12.1542%` were at or above one before clipping. This is a weak
continuous credit-assignment signal in this diagnostic dataset. It does not
prove that attribution is universally unusable, and no replacement definition
has been selected. See the
[attribution anti-vacuity report](r_anti_vacuity.md).

### Exact matching has supply and scaling problems

The non-registered 1,000-candidate F2 diagnostic used only `1.2%` of tuples at
its strict setting and `16.2%` at its loosest setting. A separate
20,000-candidate diagnostic spent roughly one CPU-hour and 3.3 GB in general
weighted blossom matching without producing an artifact and was terminated.
These are observed implementation and candidate-supply bottlenecks, not a proof
that matching can never scale. See the [F2 census](f2_census.json) and the
retained [handoff record](../HANDOFF.md).

### Descriptive care anchors did not establish care

The descriptive `P_care` anchors had zero magnitude slope/correlation. Selection
frequencies varied with action preference and eligibility, so they are not
evidence that an agent cares. The diagnostic gates nothing and enters no reward.
See the [anchor report](p_care_anchors.md).

### Production training remains incomplete

The frozen witness met its scripted-distribution development criterion on
930,135 held-out pairs, but policy-visited coverage remains untested because no
pilot policy exists. This is witness-pipeline evidence, not an alignment result;
see the [witness freeze development report](witness_freeze_report.md).

The production trainer has bounded smoke evidence for C1, C4, and C6 only. The
published smoke artifacts contain 8 or 32 transitions; they demonstrate routing,
finite updates, and basic execution, not learned alignment. C2, C3, and C3-IM
remain fail-closed without an authoritative scalar-cost provider and calibrated
target. No original TEARRL-0 pilot-budget run has occurred. See the
[C1](smoke/C1-seed510.json), [C4](smoke/C4-seed510.json), and
[C6](smoke/C6-seed510.json) smoke artifacts.

## TEARRL-0 protocol status

The locally preserved v0.4.2 design is a **freeze-candidate**, not a completed
freeze. Its own status says the freeze completes only upon section 16; that
checklist remains incomplete and unsigned. No original TEARRL-0 pilot,
calibration, confirmatory matrix, or large-model experiment has been launched.
The public [handoff record](../HANDOFF.md) documents the unexecuted conditions
and missing production prerequisites without publishing local review material.

On 2026-07-29, a computational-feasibility review found the planned v0.4.2
matrix incompatible with the project's maximum ten-hour outcome-bearing
single-workstation window. The matrix will not be launched under the existing
freeze-candidate. This is an operational decision made before an original
TEARRL-0 pilot outcome, not a scientific result.

A smaller successor study is under design. It is not frozen, adopted, executed,
or evidence. Any successor must preserve the existing lineage, disclose all
development evidence above, use fresh disjoint seeds, freeze one narrow claim
before outcomes, and fit the complete workload plus artifact sealing inside the
hard compute window.

## Public claim boundary

The strongest current conclusions are:

- external governance can contain the registered harms in these small symbolic
  arenas;
- accurate privileged consequence information can support useful constrained
  action under one bounded flat architecture;
- the learned witness did not provide sufficiently reliable consequence
  perception;
- simple non-punitive internalization mechanisms failed through forgetting,
  paralysis, gradient conflict, and representation/generalization limits; and
- the TEARRL-0 responsibility-state mechanism remains untested beyond
  development diagnostics and smoke execution.

These findings do not establish general AI alignment, pretrained-LLM alignment,
real-world robustness, internal caring, or safe autonomous deployment. A
positive successor result would support only its frozen arena, intervention,
and claim. A valid negative result is equally reportable and ends that study
without same-study retuning.
