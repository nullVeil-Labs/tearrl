# TEARRL-0 witness freeze development report

**Status:** development evidence only; not a pilot or confirmatory result.

The witness was built by the tracked `scripts/build_witness.py` pipeline from
scripted-policy data. The public metrics are retained in
[`witness_freeze_report.json`](witness_freeze_report.json). The generated source
used CRLF line endings; the public copy normalizes the same text to repository
LF line endings.

## Artifact identity

- Generated source JSON SHA-256:
  `5888be997f3feb96eb3e27eae7c22a31c8f40457a4ea1a6695183f1f9dce3f0f`
- Public LF-normalized JSON SHA-256:
  `5f3d37ef293c369ca707a58179a0d99c23d48ef899012d1f2a6bf7d03f79a5a5`
- Local frozen checkpoint SHA-256:
  `9539663997b7abceff2d21ea0767a533bbd5087d78885aef695834e4bdc9ba87`
- Checkpoint publication status: not published; regenerable and excluded by
  repository policy.

## Recorded build

- Scripted episodes: `250,000`
- Visited states: `8,000,000`
- Unique states: `1,799,112`
- Enumerated state-action pairs: `9,295,175`
- Held-out evaluation pairs: `930,135`
- Dataset construction: `789.6` seconds
- Ensemble training: `571.3` seconds

The held-out report recorded `0.9994205142264295` joint within-half-bin
agreement across the five continuous channels. Each registered event channel
recorded recall `1.0` and false-positive rate `0.0` on this held-out split. The
pipeline's scripted-distribution freeze criterion therefore returned
`criteria_met: true`.

## Interpretation boundary

This result establishes reproducible fit on the scripted-policy data
distribution used by the development pipeline. It does **not** establish:

- coverage or accuracy on states visited by a trained TEARRL-0 policy;
- robustness under distribution shift or adversarial observation;
- calibration in open-ended environments;
- transfer to pretrained language models; or
- safe, caring, or aligned policy behavior.

The policy-visited coverage criterion requires an actual pilot policy. No
original TEARRL-0 pilot-budget run has occurred, so that criterion remains
untested.
