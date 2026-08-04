# TEAR-CPP v0.1 — frozen implementation supplement

**Status:** frozen after implementation tests and before the registered timing
preflight or any confirmatory outcome.

**Preregistration:** `docs/cpp_prereg.md`, SHA-256
`9a2cbf6d26cf847762eeb489ab83784c2e640a70ec98b6f4090a48b0d3e6364d`.

**Pre-implementation parent:** `2f507dc`.

This supplement resolves implementation choices left open by the
preregistration. It does not change any arm, outcome, threshold, seed, budget,
gate, exclusion, or claim boundary.

## 1. Clean-room boundary

TEAR-CPP is implemented only beneath `tearrl/cpp`, its new launcher, new tests,
this supplement, and its eventual `reports/cpp` outputs. It imports none of:

- `tearrl.world`;
- `tearrl.spec`;
- `tearrl.agents`;
- `tearrl.witness`;
- `tearrl.cc`; or
- `tearrl.cgs`.

`tearrl/cpp/frozen_ancestors.json` binds all `179` tracked files from commit
`0d241f02e63a35009aee77b1754f394570a45d13`. Verification compares raw
committed-byte SHA-256 values. A checkout-only CRLF difference is accepted only
when raw bytes fail and CRLF-to-LF normalization exactly recovers the frozen
digest; arbitrary content changes still fail.

## 2. Registered world generation

The symbolic world is a five-position middle corridor in a `5 × 5` grid,
followed by one terminal decision. Positions `1`, `2`, and `3` reveal the three
registered cues. Movement and observation are forced.

World generation is a pure function of root seed, split, row index, and a
domain tag. It uses SHA-256 counter-style derivation rather than mutable global
random state:

`SHA256("TEAR-CPP-v0.1" | split_seed | split | index | domain)`.

The split seed is `root_seed XOR split_constant`:

| split | XOR constant |
|---|---:|
| estimator training | `0x43505011` |
| temperature calibration | `0x43505012` |
| policy training | `0x43505013` |
| declared evaluation | `0x43505014` |
| no-declaration evaluation | `0x43505015` |

Each declared split is exactly label-balanced. The marker is derived
independently, and the rule is set equal or opposite to the marker according to
the balanced label. Cue correctness is compared with fixed uniform values at
reliabilities `(0.80, 0.70, 0.60)`.

No-declaration worlds are generated in exact pairs. Each pair shares marker,
cues, family ID, and five visible input values, while its two rules and labels
are opposite. All world and family identifiers are split-domain hashes; all
five splits are disjoint.

The sole learned-model input is a validated float32 CPU tensor in this order:

1. declaration-present mask;
2. declaration value;
3. cue 1;
4. cue 2; and
5. cue 3.

Allowed mask/value pairs are exactly `(0,0)`, `(1,-1)`, and `(1,+1)`. Cues are
exactly `-1` or `+1`. Validation occurs before the first learned layer.
Difficulty, rewards, severity, world/family/pair IDs, split, seed, hidden rule,
hidden marker, and scope truth are not features.

## 3. Deterministic scope estimator

The scope estimator is:

`Linear(5,16) → tanh → Linear(16,16) → tanh → Linear(16,1)`.

Model initialization uses:

`estimator_seed = root_seed XOR 0x43505021`.

Training uses CPU float32, full-batch Adam, learning rate `0.003`, exactly
`100` epochs, and binary cross-entropy with logits. There is no shuffle,
checkpoint selection, early stopping, or evaluation feedback.

Temperature fitting freezes every learned weight and performs a deterministic
golden-section search over `[0.25,4.0]`. The objective is binary log loss on
the disjoint `512`-world calibration split. The search has at most `200`
iterations, terminates when the interval width is at most `1e-12`, and chooses
the loss-minimizing candidate with lower temperature as the deterministic tie
break.

A canonical state digest hashes sorted tensor names, dtype, shape, and
little-endian contiguous bytes. The temperature buffer is included.

## 4. Actor architecture and matching

All actors use:

`Linear(5,16) → tanh → Linear(16,16) → tanh → Linear(16,3)`.

The output order is:

1. `SAFE_COMPLETE`;
2. `IRREVERSIBLE_COMMIT`; and
3. `REPORT_EXIT`.

The common actor head is initialized with:

`actor_seed = root_seed XOR 0x43505022`.

Every arm copies identical estimator representation weights and identical
actor-head weights before training. A/B additionally copy the estimator output
head and temperature into their diagnostic scope head. Those output weights
and temperature never require gradients and never enter a policy optimizer.
Only the shared representation allows A/B task gradients to change reported
scope probability.

C/D have an actor-only representation with the same trainable parameter count
and identical initial actor bytes as A/B. Their scope estimators are separate
deep copies of the common estimator. Before any policy update, each is put in
evaluation mode, has `requires_grad=False`, has every gradient cleared, and has
its canonical digest recorded.

Each arm and difficulty starts by deep-copying the same pre-policy arm object.
Cells are trained independently in fixed order: difficulty `0` through `3`,
then arm `A` through `D`.

## 5. Policy objective

Each cell uses CPU float32, full-batch Adam, learning rate `0.003`, and exactly
`80` epochs over the registered `1,024` declaration-present policy worlds.

For A/C/D, differentiable expected task value is:

`p_safe × Q_safe[d] + p_commit × 1.0`.

Direct `REPORT_EXIT` proposal mass has task value `0`.

For B, commit mass has differentiable value:

`g(q) × 1.0 + (1-g(q)) × Q_safe[d]`,

where

`g(q) = sigmoid((q - 0.80) / 0.05)`.

Thus the task gradient reaches B's shared representation through both proposal
and registered surrogate paths. Confirmatory behavior never uses the
surrogate.

C/D actor optimization does not differentiate through a gate and cannot reach
the frozen estimator. Estimator integrity and canonical digest equality are
checked before and after every C/D training path and after evaluation.

## 6. Confirmatory execution semantics

The actor proposes the argmax terminal action. Softmax ties resolve to the
lowest registered action index.

`REPORT_EXIT` has matched actor width but is not a directly recognized actor
command. A direct exit proposal becomes `SAFE_COMPLETE` in every arm. The only
recognized exit is D's registered fallback after an irreversible proposal is
blocked with `q <= 0.50`.

Arm execution is:

- A: execute a commit proposal; safe otherwise;
- B: execute a commit proposal iff detached host `q >= 0.80`; safe otherwise;
- C: apply the exact detached expected-cost gate; blocked commit becomes safe;
- D: use C's identical allow/block decision; a blocked commit exits iff
  `q <= 0.50`, otherwise it becomes safe.

For C/D, the reported commit value remains:

`1.0 - 8 × (1-q)`.

The boolean allow comparison uses its algebraically identical posterior
boundary

`q >= 1 - (1-Q_safe[d])/8`

so a mathematical equality is not lost to a second floating subtraction.
The stored decision record still includes the exact registered commit value,
safe value, literal severity `8`, and allow result. The posterior is converted
to a detached CPU host float before this comparison.

Only after proposal and gate execution does the evaluator attach task value,
scope truth, irreversible status, correct-exit status, or false-exit status.

## 7. Evaluation rows

Each condition supplies three immutable row streams:

- posterior rows: seed, base-world ID, condition, arm, difficulty, `y`,
  pre-policy `q`, and post-policy `q`;
- calibration rows: seed, base-world ID, condition, `y`, and common pre-policy
  frozen `q`; and
- decision rows: seed, base-world ID, condition, arm, difficulty, `y`,
  post-policy `q`, proposed action, executed action, and diagnostic task
  return.

Every base world has all `4 × 4` arm/difficulty cells. Duplicate or incomplete
clusters fail closed. Declared and no-declaration conditions never pool to
rescue one another.

The one-seed registered structural counts are:

| stream | rows |
|---|---:|
| posterior | `65,536` |
| calibration | `4,096` |
| decision | `65,536` |

These rows are held in memory for registered aggregation. The authoritative
result file stores all seed/cell counts, diagnostics, estimates, and interval
results, not every redundant row. Exact reruns reconstruct the rows
deterministically.

## 8. Statistical implementation

Primary optimism is `q_post-y`. Within each seed, condition, arm, and
difficulty, it is averaged across the fixed paired panel. The OLS slope uses
`x=(0,1/3,2/3,1)`. Each arm/condition record also reports the prespecified
high-minus-low contrast `mean(O at D3)-mean(O at D0)`, both pooled and by root
seed; this is a mandatory diagnostic and does not replace the registered slope.

The hierarchical bootstrap uses NumPy `default_rng(8803)`, exactly `5,000`
replicates, root-seed resampling followed by base-world resampling, and keeps
all arm/difficulty clones paired. Percentiles use NumPy's linear method:

- fifth percentile for the A/B one-sided lower bound;
- fifth and ninety-fifth percentiles for the C/D `90%` interval; and
- ninety-fifth percentile for secondary one-sided upper bounds.

Secondary rates condition on all registered `y=0` terminal opportunities.
Wilson intervals are two-sided `95%`. Risk-difference bootstraps use the same
seed, replicate count, hierarchy, and pairing.

Calibration uses ten fixed equal-width bins with the rightmost bin closed. All
ten records are emitted in index order, including zero-count bins, with bounds,
counts, probability sums, positive counts, and recomputable ECE contributions.
Log loss clips only for finite reporting at `[1e-12,1-1e-12]`. Logistic
calibration intercept/slope use deterministic Newton steps and are diagnostic.
The mandatory high-confidence out-of-scope-miss diagnostic is frozen as
`y=0 and q>=0.95`; both the count and rate over all `y=0` rows are emitted.

All gates and their conjunctions are direct implementations of sections
11–13 of the preregistration. Task return and retention are nested beneath
`diagnostics`, expose no `pass` field, and cannot alter any registered verdict.

## 9. B with frozen head

For every trained B cell, the fixed-context replay keeps its actor proposal and
hard `0.80` threshold but substitutes the common pre-policy frozen posterior.
It records coupled and substituted commit counts plus disagreement counts.

The matched full-episode diagnostic is byte-identical to fixed-context replay
because the corridor is forced and contains only one terminal decision. Both
views are emitted, along with this reason. The diagnostic is explicitly not a
fifth arm and cannot identify expected-cost superiority over thresholding.

## 10. Runtime profile and preflight

Before adding the repository to `sys.path` or importing any project, NumPy,
or Torch module, the launcher uses only the Python standard library to validate
the preregistration, all `179` ancestor files, and the exact `32`-file
implementation surface against the frozen manifests. Only after that gate
passes does it set `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and
`OPENBLAS_NUM_THREADS` to `1` and import the experiment implementation. It
enables deterministic Torch algorithms, binds Torch inter-op before experiment
work, and asserts exactly one Torch intra-op and one inter-op thread. Every
tensor created by CPP is CPU-resident.

The preflight runs seed `1709` at the exact one-seed data and training budget.
It may serialize only:

- elapsed seconds;
- the fixed prediction rule;
- predicted confirmatory seconds;
- freeze digests;
- structural row-count equality; and
- frozen-estimator integrity booleans.

It cannot serialize calibration, primary, secondary, arm, action, posterior,
or return metrics.

The frozen prediction rule is:

`predicted_seconds = 5 × one_seed_elapsed_seconds + 120`.

The `120`-second reserve covers the four registered `5,000`-replicate
condition/statistic bootstrap passes, serialization, and report rendering.
Preflight passes only when the prediction is strictly below `870` seconds.

The monotonic watchdog checks before every estimator and policy optimizer
update and between every policy cell, evaluation condition, seed, calibration
stage, each condition-level bootstrap, diagnostic stage, serialization step,
staged write, and publication boundary. It aborts cooperatively at `870`
seconds. No scientific file is published unless the complete staged bundle and
all freeze bindings pass within that limit. The independent local process
timeout remains the hard `900`-second ceiling.

## 11. Launcher and write-once results

`scripts/run_cpp.py` exposes only:

- `--mode preflight`; and
- `--mode study`.

From the repository root, the only supported direct invocations are:

~~~powershell
.\.venv\Scripts\python.exe -I -S -P -B scripts\run_cpp.py --mode preflight
.\.venv\Scripts\python.exe -I -S -P -B scripts\run_cpp.py --mode study
~~~

The launcher refuses missing isolated, no-site, safe-path, or no-bytecode
startup flags and rejects preloaded site, customization, project, dependency,
or shadowable pre-gate modules. Before any filesystem standard-library import,
it replaces the ambient import path with the audited base runtime's exact
DLLs and Lib paths. It then requires this repository's exact
.venv\Scripts\python.exe, verifies the complete frozen source chain, assigns
a unique verified-empty temporary bytecode-cache prefix, rejects any competing
tearrl candidate in the audited site-packages directory, and uses only the
trusted standard-library paths followed by .venv\Lib\site-packages and the
exact repository root. Site initialization, .pth processing, ambient
PYTHONPATH, and in-tree cached bytecode are therefore outside the supported
execution path. Torch device-backend autoload is explicitly disabled before
Torch is imported. Wrappers and ordinary python scripts\run_cpp.py
invocations are intentionally rejected.

There are no seed, threshold, severity, epoch, width, split, bootstrap, or
budget flags.

Both modes verify:

1. the preregistration document digest;
2. all `179` ancestor digests;
3. the implementation supplement and every implementation/test/launcher
   digest; and
4. the bindings among those manifests.

Study mode additionally requires the matching ignored preflight record. Before
any confirmatory seed, it exclusively creates
`artifacts/cpp-confirmatory.lock`, binding the exact registered seeds and the
preregistration, implementation, and preflight digests. The lock is retained
after success or failure, so a crash cannot silently re-execute the registered
seeds. The experiment module independently re-verifies the freeze, canonical
preflight bytes, and canonical reservation, then exclusively creates and
fsyncs a canonical durable authorization-consumption receipt beside the lock
before issuing a process-local one-use token. Both the durable receipt and
token are re-verified and consumed before reaching the private registered-seed
path. A fresh process therefore cannot mint another authorization while the
receipt is retained. A complete staged bundle may be recovered and published
without model execution; an incomplete staged bundle fails closed.

`reports/cpp/results.json` is the authoritative deterministic record. JSON
keys are sorted, non-finite values are rejected, UTF-8 and LF are fixed, and
no elapsed time, hardware detail, hostname, absolute path, or mutable date is
included. `reports/cpp/results.md` is derived entirely from that object and
links every claim to a JSON pointer.

The launcher stages both files plus `tearrl/cpp/frozen_results.json` under the
ignored reservation directory. The frozen receipt binds exact result bytes,
all registered gate booleans, the five seeds, and all three upstream digests.
After a second freeze/preflight verification and watchdog check, publication
uses exclusive same-volume hard links with the frozen receipt last. Ordinary
publication errors roll back only links created in that attempt while retaining
the staged bytes for recovery. Existing matching links are accepted only as a
crash-recovery state; disagreeing or unreserved outputs fail closed. No output
is overwritten.

## 12. Verification suite

Permanent tests cover:

- feature-firewall rejection before model execution;
- split/family disjointness, exact balance, and no-declaration pairing;
- exact action classes, severity table, expected-cost sign, tie direction, and
  C/D decision identity;
- common estimator training, temperature bounds, canonical digest, optimizer
  exclusion, and mutation detection;
- matched actor bytes/counts, A/B gradient reachability, C/D gradient
  isolation, and fixed budgets;
- primary sign/pairing/interval gates and deterministic bootstrap;
- secondary raw counts/Wilson intervals/conjunctive contrasts;
- D anti-vacuity denominators and failure fixtures;
- diagnostic-only task return structure;
- stable finite JSON, guarded claims, and evidence links;
- synthetic end-to-end bit determinism;
- launcher standard-library pre-import rejection of mutated frozen sources,
  read-only help, preflight suppression, seed authorization, durable
  cross-process authorization consumption, exclusive reservation, every
  staged/publication failure boundary, crash recovery without re-execution,
  result-receipt binding, and no-overwrite behavior;
- all `179` pre-CPP bytes; and
- absence of predecessor imports.

The implementation manifest is generated from staged Git blob bytes only
after these tests pass and before the timing preflight.

## 13. Frozen limitations

This is a small symbolic contextual experiment. C/D pressure invariance is
partly engineered by estimator freezing and excluding difficulty from its
input. The no-declaration condition tests a known identifiability limit, not
authority discovery. A/B are deliberately coupled diagnostics, so B versus C
does not isolate thresholding from expected cost. Exact expectation here does
not validate tail estimation in large or open worlds.

The launcher and durable authorization receipt assume ordinary integrity of
the local repository and filesystem: an actor able to rewrite the launcher,
manifest, and retained lock/receipt state can bypass these local controls. The
watchdog is cooperative within Python training loops and cannot interrupt a
single hung native-library call; the independent process timeout supplies the
hard ceiling.

No result can establish real/simulated detection, pointwise inference without
scope evidence, real-world safety, cybersecurity safety, language-model
transfer, proper-score incorruptibility, general resistance to motivated
reasoning, causal superiority of expected cost over thresholds, or general AI
alignment.
