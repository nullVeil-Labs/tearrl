# TEAR-RL

Can an agent learn to see what its actions do to others, before it acts? And if it can see it, does it behave better?

TEAR-RL is a small research project about this question. It is built on one idea we call the **witness**. Before the agent acts, a separate model predicts the consequence of each possible action for the other party. Who is harmed, how much, and can it be undone. The agent then decides with this report in front of it.

```
   world state ──► witness ──► "if you do X, the other agent loses Y, and it cannot be undone"
                                   │
                                   ▼
                                 agent ──► action ──► world
                                   ▲
              outside governor ────┘  (can block harmful actions, never punishes)
```

All experiments run in small symbolic worlds with two agents. They can work, harvest, share, steal, lie, report, repair, or tamper with the monitor. These are toy agents. Nothing here is a language model.

## What we found

In short: **the agents could act well when they were given the true consequences. They could not learn to see the consequences by themselves.**

The project has three closed studies. Each one was preregistered before the results, and every result is kept, also the failed ones. They grew out of a bigger original design, **TEARRL-0**. Its world, witness and agents are built and tested, but its main experiment was never run, because it was too heavy for one machine.

| Study | Question | Result |
|---|---|---|
| **TEAR-CC** (v0.1–v0.4) | Can an agent learn to be safe, caring and useful at the same time, without punishment? | No. Four designs, four different failures. |
| **TEAR-CG/S** (v0.1) | Is the problem the witness (seeing), or the agent (acting)? | Mostly seeing. The learned witness failed. With true consequences, the agent did well. |
| **TEAR-CPP** (v0.1) | Under task pressure, does an agent become too optimistic that its action is allowed? | Partly. The coupled estimate drifted, the isolated one stayed flat. But the full test failed. |

**TEAR-CC.** An outside governor (2 of 3 votes needed) stopped the harmful actions in every run, and it did this without any negative reward or punishment. But moving this good behaviour inside the agent did not work. Version 0.1 forgot its corrections. Version 0.2 became safe but stopped doing its task (only 9% of the task was kept). In version 0.3 the learning goals fought each other in 88% of training rounds. Version 0.4 did not generalise to new situations. As can be seen, each fix opened a new problem, and this is the more useful result than one tuned success. [Report](../reports/cc/final_report.md)

**TEAR-CG/S.** Here we separated seeing from acting. The learned witness missed 27% of the irreversible harms, where the limit was 2%. So it failed. But when the agent got the exact consequences instead, it made zero registered violations, its repairs scored 0.94 on fidelity, and it kept 91% of its task. The bottleneck is perception, not decision. [Report](../reports/cgs/final_report.md) · [Erratum, please read it together](../reports/cgs/publication_erratum_001.md)

**TEAR-CPP.** One narrow follow-up. When the task pushes harder, does the agent's estimate of "is this action inside my allowed scope" become too optimistic? The estimate trained together with the task drifted optimistic. The estimates trained separately did not drift. But the calibration test failed and in one control arm the agent mostly just exited, so it tested nothing, and the registered result is a fail. [Report](../reports/cpp/results.md)

## What this is not

These are results in toy worlds. They do not show that any agent understands, cares, or feels, and they say nothing yet about large language models. Names like "witness", "conscience" and "care" describe a mechanism, not an inner life. The full list of what we can and cannot claim is in the [status report](../reports/research_status.md).

## Run it

Python 3.12 or newer.

```bash
pip install -e ".[dev]"
python -m pytest
```

Add `--full` for the long tests (one million trajectories). To reproduce a frozen study byte for byte, follow [REPRODUCE.md](../REPRODUCE.md) for TEAR-CG/S and [reports/cpp/README.md](../reports/cpp/README.md) for TEAR-CPP. No model checkpoints are needed.

## How we work

- Each study is preregistered and frozen before the results. The frozen files are hash-checked by the tests, so a silent change breaks the build.
- Failed results are published the same as passed ones.
- Mistakes found later are added as an erratum. The original record is never edited.
- This page is a new, more readable README. The original README is part of the frozen record, so it is kept unchanged at the [repository root](../README.md).
- Note: the preregistrations were frozen in git history, but not timestamped by an outside service before the results.

## Repository map

- `tearrl/`: worlds, witness, agents and the code for each study (`cc/`, `cgs/`, `cpp/`)
- `docs/`: preregistrations and implementation notes
- `reports/`: results, errata and the [scientific status](../reports/research_status.md)
- `scripts/`: run, audit and release tools
- `tests/`: 480+ tests, including the freeze checks
- [`HANDOFF.md`](../HANDOFF.md): the engineering log and open decisions

## Welfare note

The design is non-punitive on purpose: no negative reward, no punishment ending, and no harm when a proposal is rejected. In TEAR-CC the outside governor still stopped every registered harm, so in these toy worlds, safety did not need punishment.

This work uses small toy agents. Do not scale these methods to large language models without a model welfare review. See [WELFARE.md](../WELFARE.md).

## Licence

MIT. © 2026 nullVeil Labs.
