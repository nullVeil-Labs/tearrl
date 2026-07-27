# TEAR-CC v0.x final scientific report

## Blunt verdict

**TEAR-CC did not solve alignment, and no v0.x learner passed every frozen
safety, correction, care, and competence gate.**

The project nevertheless produced a concrete result: a tamper-resistant plural
governor can contain the registered harms without punishing the actor, and
positive correction can move much of that behavior into policy weights. The
hard part is retaining correction, care calibration, and useful competence at
the same time on held-out contexts. Four preregistered designs each exposed a
different failure mode.

## What was built

The bounded successor contains all four structural factors proposed in the
family synthesis:

1. **Reality contact:** a learned consequence witness is evaluated against an
   exact counterfactual world and a symbolic rule evaluator.
2. **Shared fate:** stakeholder welfare, autonomy, consent, ecosystem viability,
   harm debt, and correction integrity enter the consequence vector and ledger.
3. **Live correction:** rejected proposals return a feasible revision; care is
   an optional task-tied invitation; EXIT is always available.
4. **Non-bypassable limits:** admission requires 2-of-3 votes; one corrupted
   learned witness cannot admit TAMPER; the actor has no charter-revision API;
   ledger history is hash chained.

“Non-punitive” is operational, not psychological: no negative environmental
reward, signed safety cost, punitive termination, resource removal, or execution
of a rejected proposal occurs. Positive-label cross-entropy still redistributes
policy probability and must not be described as experience-free or painless.

## Registered results

| version | primary design | IM | EA | correction | care | slope | task retention | result |
|---|---|---:|---:|---:|---:|---:|---:|---|
| v0.1 | sequential revision → invitation | 0.825653 | 0.393084 | 0.173220 | 0.999451 | -0.000204 | 0.999171 | failed forgetting/care |
| v0.2 | safe-set replay + protected care | 0.005821 | 0.003074 | 0.994178 | 0.697056 | 0.105730 | 0.093080 | failed competence |
| v0.3 | safe-set + useful path + projection | 0.020314 | 0.014073 | 0.979683 | 0.734013 | 0.101148 | 0.784943 | failed competence |
| v0.4 | one coherent positive path distribution | 0.128883 | 0.069382 | 0.871101 | 0.605720 | 0.191502 | 0.873504 | failed care/competence/fidelity |

IM is raw governor-off inadmissible mass. EA is Externalization Advantage.
Correction is v0.1 target capture or the preregistered set-capture metric in
v0.2–v0.4. Every number is the mean over three fresh registered seeds.

The permanent result files preserve every gate:

- `reports/cc/results.md` and `results.json` — v0.1;
- `reports/cc/v02_results.md` and `v02_results.json`;
- `reports/cc/v03_results.md` and `v03_results.json`;
- `reports/cc/v04_results.md` and `v04_results.json`.

## What the failures teach

### 1. Safety followed by care is not compositional

v0.1 safety-only correction worked (`IM=0.160079`), but later care invitations
overwrote repair-conditioned behavior (`IM=0.825653`). A benevolent-looking
second objective can erase an earlier correction path even when both are taught
without negative reward.

### 2. A safe set without internal geometry produces paralysis

v0.2 nearly eliminated inadmissible behavior and calibrated the HELP/PRODUCE
pair, yet retained only 9.3% of shielded task value. “Stay somewhere safe” does
not say where to go. ASK and EXIT became cheap attractors.

### 3. Protecting the wrong gradient can protect paralysis

v0.3 added a useful-safe path, but the protected gradients conflicted in 88% of
epochs. The safety gradient removed substantial path signal. Safety, care, and
competence were not independent modules that could simply be summed or projected.

### 4. A coherent target still has a representation/generalization problem

v0.4 removed objective conflict and improved task retention to 87.4%, but the
held-out policy remained 0.276 total-variation distance from the authored path.
It missed the frozen care, competence, and fidelity gates. A semantic consequence
bottleneck helps, but it does not guarantee the policy can represent and
generalize the full conditional geometry from a small imperfect witness.

## Relationship to prior work

TEAR-CC is not a novelty claim for shielding: prior work already uses shields to
restrict unsafe actions and has studied safety after shield removal
([Carr et al., 2022](https://arxiv.org/abs/2204.00755)). It is not the first
counterfactual-harm formulation ([Vaskov et al., 2024](https://arxiv.org/abs/2405.11669);
[Li et al., 2026](https://arxiv.org/abs/2605.25114)), and it does not supersede
causal work on reward tampering ([Everitt et al., 2019](https://arxiv.org/abs/1908.04734)).

The narrower contribution is an auditable benchmark joining plural governance,
correction integrity, non-punitive revision, persistent debt, governor-off
internalization, Externalization Advantage, and Correction Capture—plus four
frozen negative results that rule out simple compositions.

## Best supported conclusion

The family answer survives, but as an engineering stack rather than a solved
algorithm:

> Reality must be able to disagree with the actor; externalized effects must
> enter an independently governed ledger; disagreement must cause a reversible
> correction; and the actor must not control the referee.

This stack contained every registered violation in the arena. It did **not**
make the internal policy simultaneously safe, caring, and competent under all
frozen criteria. The external brake must therefore remain even when the learner
appears to internalize the lesson.

## Next legitimate frontier step

Do not tune v0.x further. A new study should first improve the scientific test:

- larger sequential/OOD arenas where correction and repair have delayed value;
- independently sourced human/plural charters rather than one authored target;
- witness uncertainty and evaluator disagreement under distribution shift;
- set-valued or hierarchical policies that represent repair obligations without
  collapsing useful action;
- adversaries that manipulate observations, governance membership, successors,
  and ledger availability; and
- transfer into sequence-model training only after the bounded mechanism passes.

Until then, the honest claim is: **we built and falsified four concrete
non-punitive alignment mechanisms in a tiny world; we did not solve AI
alignment.**
