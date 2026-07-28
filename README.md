# TEAR-RL

TEAR-RL is a research repository for bounded experiments on consequence
prediction, responsibility state, corrective behavior, and external containment
in small symbolic multi-agent environments.

## Current status

The completed TEAR-CC v0.1–v0.4 lineage is negative: no version passed every
frozen gate. TEAR-CG/S v0.1 is also complete and stopped without tuning after its
confirmatory failures.

TEAR-CG/S found that:

- the learned consequence witness failed six of seven grounding gates;
- perfect-oracle plus flat actor passed the bounded actor thresholds;
- perfect-oracle plus registered hierarchy failed task retention;
- the hierarchy effect was not demonstrated; and
- the external governor achieved exact containment in the registered benchmark.

These are bounded symbolic-environment results. This repository does not claim
to solve general AI alignment or pretraining alignment, and it does not claim
that a witness is honest or that an agent understands, cares, feels, or is
conscious.

## Reproduce

```text
python -m pytest -q
python scripts/run_cgs.py --mode pilot
python scripts/run_cgs.py --mode confirmatory
python scripts/verify_public_snapshot.py
```

The runs are deterministic under the frozen seeds and write transparent Markdown
and JSON artifacts under `reports/`. No model checkpoints are required or
published.

## Research map

- TEAR-CG/S preregistration: `docs/cgs_prereg.md`
- TEAR-CG/S implementation freeze and amendments: `docs/cgs_implementation.md`,
  `docs/cgs_implementation_amendment_001.md`, and
  `docs/cgs_implementation_amendment_002.md`
- TEAR-CG/S final report: `reports/cgs/final_report.md`
- TEAR-CG/S authoritative confirmatory data: `reports/cgs/results.json`
- TEAR-CC frozen lineage: `docs/cc_prereg.md` and `docs/cc_v*_prereg.md`
- TEAR-CC final report: `reports/cc/final_report.md`

The strongest permitted claim is calibrated counterfactual consequence reporting
and bounded downstream causal evidence under registered interventions.
