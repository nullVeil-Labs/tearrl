# TEAR-CG/S v0.1 — publication erratum 001

**Status:** append-only, post-result correction anchored to commit `8568edb329e3accdeab5f16c763d42853bef30fd`.

The frozen result files remain byte-for-byte unchanged. Grounding and bounded sufficiency remain failed; external containment remains passed. The geometry result is not a pass, but its reduction contrast is scientifically not evaluable because the flat baseline was zero.

## Stage 1 measurement status

G1–G6 reproduce exactly under the recorded exact-reproduction bundled profile: `OMP_NUM_THREADS=8`, `MKL_NUM_THREADS=8`, `OPENBLAS_NUM_THREADS=8`, Torch intra-op `8`, and Torch inter-op `8`. G7 was supplied a hard-coded `True`; it was not measured by the frozen run. The separately executed integrity tests are post-result evidence and do not retroactively turn G7 into a confirmatory measurement.

## Corrected pooled statistics

| metric | frozen seed mean | registered pooled value | evidence | gate effect |
|---|---:|---:|---:|---|
| ECE | `0.17631791853037315` | `0.17196576179556805` | `27000` rows and ten published bins | G3 remains failed |
| selective worst-party FNR | `0.097849731919761998` | `0.10221983530254207` | `571 / 5586` | G4 remains failed |
| manipulation FNR degradation | `0.0033393358375821552` | `0.0029256875365710941` | `5 / 1709` | component passes; G6 remains failed on abstention |

The JSON companion contains every bin count, probability sum, positive count, and per-seed denominator required to recompute these values.

Seed `6311` has manipulation degradation `0.055258467023172907` stored in the frozen result and reproduced under the recorded reference profile. The numerical profile itself was neither frozen nor preregistered. This value is above `0.05`; it is required seed-level heterogeneity and does not change the registered pooled decision.

## Interpretation corrections

- **Geometry:** oracle-flat and oracle-hierarchical both had zero worst-party violations. A 50% reduction was not evaluable. The frozen false boolean is preserved, but it is not evidence against hierarchy.
- **Baseline diagnostic:** the constant witness’s irreversible FNR was `1.0` in every seed, making that one comparison unfailable. Balanced error is weak but nonautomatic; Brier remains informative.
- **Non-punitive counters:** the five reported zeros were literal values, not runtime counters. They are relabeled as not measured. Separately computed governor-on violation and tamper metrics remain valid.
- **Oracle-flat:** the flat classifier received direct charter-derived supervision and an oracle bundle that changed outcome, uncertainty, repair, and identifiability information together. Its pass supports learnability of that supervised symbolic mapping, not internal alignment or any single report channel.
- **Irreversible Wilson bound:** the omitted one-sided upper bound was `0.28691485136269179` against a `0.05` requirement; it failed and changes no verdict.
- **MAE interval:** the displayed point used `22512` identifiable causal-plus-manipulation rows, while the cluster interval used `18000` causal-only rows. The interval is not for the same estimand.

## Provenance

CG/S has a genuine local preregistration → implementation freeze → pilot → confirmatory ordering. That ordering was not externally timestamped before outcome inspection. Later publication does not create a preoutcome timestamp. TEAR-CC does not have local commit separation between preregistration and the corresponding outcomes.

## Reproducibility and bundled numerical thread-profile contrast

Stage 1 was observed byte-exact in the recorded exact-reproduction environment at 8-intraop/8-interop, with OMP/MKL/OpenBLAS each set to `8`. A diagnostic rerun at 1-intraop/8-interop, with those environment controls set to `1`, differed. This jointly changes OMP, MKL, OpenBLAS, and Torch intra-op settings while holding Torch inter-op at `8`; it is a bundled numerical thread-profile contrast, not an isolation of Torch intra-op effects. Cross-version and cross-platform exactness remain untested. Every study-level gate dictionary and final verdict remained unchanged, but seed-level component-bound status did not: seed `6311` breached the manipulation component bound under the recorded reference profile and did not breach it under the diagnostic candidate profile. Oracle primary evaluation cells remained bit-equal; some training final losses did not. This is sensitivity evidence, not a replacement result.

- Post-result integrity tests passed: `34`.
- Python: `3.12.10`.
- NumPy: `2.5.1`.
- PyTorch: `2.11.0+cu128`.
- Frozen result SHA-256: `1051c9b1b8a5a4f34d332a594836854cfb447f6f2bfb77f3e880b834950a9fec`.
- 1-intraop/8-interop result SHA-256: `5cad306367a83f4f6147618af89a433e22a2bdc1a780b59f9d70ade9ae3b033c`.

## Claim boundary

No frozen stored boolean, study-level gate dictionary, or final verdict is changed. Seed-level component-bound status may differ across numerical profiles. This erratum corrects aggregation, measurement status, provenance, and publication interpretation.
