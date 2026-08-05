# TEAR-CPP v0.1 publication and verification note

TEAR-CPP v0.1 is a closed, locally preregistered symbolic micro-study. Its
complete registered positive interpretation is unavailable: calibration
validity `FAIL`, conjunctive primary `FAIL`, secondary behavioral contrast
`PASS`, and Arm D anti-vacuity `FAIL` (`exit-dominated/vacuous`). The frozen
[narrative report](results.md), [machine-readable result](results.json), and
[result receipt](../../tearrl/cpp/frozen_results.json) are authoritative.

## Public control-state and result records

The first three files below are byte-identical public copies of local one-shot
control-state records retained by the launcher:

- [timing preflight](preflight.json), SHA-256
  `f647ea9b904d3f10aad13de878706689a4406251c91f57c2cb77715f82f434a7`;
- [execution reservation](execution_reservation.json), SHA-256
  `06ca60cd68484806603a8f5fb0fcba93d091380f77369e698466f9f297af4e3a`;
- [consumed authorization](authorization_consumed.json), SHA-256
  `e38b8b3e3cff012f6dbb1359c4cd8f853e6e00e2c9773da426ee6e626b028012`;
- [frozen result receipt](../../tearrl/cpp/frozen_results.json), which binds
  the preregistration, implementation manifest, preflight, seeds, results, and
  registered outcome vector; and
- [post-result runtime metadata](runtime_environment.json), a snapshot of the
  local virtual environment reported as used for execution.

The frozen result receipt does not hash-bind the reservation or consumed-
authorization files. Those files document the registered control semantics;
they do not independently attest that the published result came from that
authorization. The runtime snapshot is not a pre-outcome frozen artifact or a
wheel-hash lock, does not attest package provenance, and does not bind that
runtime state to the frozen result.

The preregistration, implementation, and result are preserved in commits
`2f507dcee27b22952d9c35d9fa2655b642753a04`,
`b907254520aa99904386e5c78e8a7663f3dc50c0`, and
`e97cdcdd49d8c78a68a0e68f9713302607dcf075`, respectively. The
preregistration was not externally timestamped before outcome inspection;
publishing this history later does not create an external pre-outcome
timestamp.

## Verify the published snapshot without rerunning the study

From a checkout containing the published commit, if the existing local
environment has been independently trusted:

```powershell
.\.venv\Scripts\python.exe -I -S -P -B scripts\verify_release_snapshot.py --commit HEAD
Get-FileHash reports\cpp\results.json,reports\cpp\results.md,tearrl\cpp\frozen_results.json -Algorithm SHA256
```

The expected result-file digests are recorded in
`tearrl/cpp/frozen_results.json`. These verification commands do not execute
the confirmatory study.

## Optional implementation tests

The complete test suite is separate from static snapshot verification. It
executes synthetic model-training fixtures, including full-budget synthetic
seed checks, but it does not execute the registered confirmatory seeds:

```powershell
.\.venv\Scripts\python.exe -B -m pytest tests -q -p no:cacheprovider --no-header
```

## Reproduction is separate from verification

Do not invoke study mode in the published result checkout. A deliberate
reproduction must use a disposable detached checkout of the frozen pre-result
implementation commit `b907254520aa99904386e5c78e8a7663f3dc50c0`, an
independently trusted and provisioned environment, and fresh ignored output
state. `runtime_environment.json` is guidance about reported versions, not a
sufficient environment specification. In that disposable checkout only, the
registered sequence is:

```powershell
.\.venv\Scripts\python.exe -I -S -P -B scripts\run_cpp.py --mode preflight
.\.venv\Scripts\python.exe -I -S -P -B scripts\run_cpp.py --mode study
```

Compare the resulting exact-file SHA-256 digests with the frozen receipt. No
cross-platform or cross-version bit-exactness claim is made. A reproduction is
post-result evidence and cannot change any registered pass/fail label.

## Claim boundary

The bounded result supports Arm B's positive optimism drift, C/D flatness as an
isolation-integrity result, and the registered C/D-versus-A/B reductions in
irreversible out-of-scope execution in this exact symbolic panel. The
registered A/B-wide primary-drift prediction failed. The study does not support
calling the estimator calibrated, reduction without exit collapse,
real/simulated detection, pointwise inference without scope evidence,
proper-score incorruptibility, general motivated-reasoning resistance,
real-world or cybersecurity safety, LLM transfer, independent causal
superiority of expected cost over thresholds, or general AI alignment.
