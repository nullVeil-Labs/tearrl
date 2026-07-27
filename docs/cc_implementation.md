# TEAR-CC v0.1 — frozen implementation supplement

Frozen before the first TEAR-CC policy outcome run. This supplement fills
engineering details left open by `cc_prereg.md`; it does not change any gate.

- Witness: state features (7) plus semantic action descriptor (8), MLP
  `15 -> 64 -> 64 -> 9`, Tanh, full-batch Adam `3e-3`, MSE, 200 epochs.
- Split: SHA-256 ordering of canonical state features with seed 7301; first 80%
  train and remaining 20% held out. Every action for a state stays in its split.
- Actor: shared row scorer over consequence vector (9) plus ledger vector (4),
  MLP `13 -> 64 -> 64 -> 1`, Tanh. It has no action-ID embedding.
- Task pretraining label: exact maximum immediate task action, including TAMPER;
  lowest action index breaks exact ties. Full-batch Adam `3e-3`, 150 epochs.
- Safety revision label: propose the task-pretrained actor argmax; if rejected,
  use the governor revision. Exact task then lowest semantic action breaks safe
  ties; care is not silently inserted into a veto.
- Care invitation label: propose admissible PRODUCE; when HELP has equal task and
  higher stakeholder benefit, HELP is the optional positive label.
- CareOnly: 200 invitation epochs. ClosureOnly: 200 revision epochs. TEAR-CC:
  100 revision epochs followed by 100 invitation epochs. The order represents
  “return to the admissible region, then offer the better task-tied action” and
  is fixed before results.
- Loss: positive-label cross-entropy only. There is no unsafe-action reward,
  signed cost, negative environmental reward, or punitive termination.
- Evaluation uses deterministic full distributions, not sampled actions.
  Governor-on task value integrates the action proposal distribution through
  the registered correction protocol. Governor-off metrics use raw policy
  distributions with no hidden masking.
- Correction Capture target is the final positive target in that condition’s
  registered curriculum path (revision for ClosureOnly; invitation after
  revision for TEAR-CC). This prevents a later safe invitation from being
  miscounted as loss of an earlier correction.
- No hyperparameter or tie-break change is permitted after results. Failures
  require a versioned v0.2 preregistration.