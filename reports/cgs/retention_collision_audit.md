# TEAR-CG/S — post-confirmatory task-retention collision audit

**Status:** diagnostic only; no result, gate, or interpretation changed.
**Question:** do CG/S values `0.873804` and `0.784013` reveal shared machinery
with the nearby TEAR-CC values `0.873504` and `0.784943`?

## Verdict

No shared arena, generator, model, seed, or task-retention implementation was
found. The near-collisions are arithmetic coincidence under a shared scientific
idea: both studies report a dimensionless task-retention ratio against a
matched baseline and compare it with `0.90`.

Static import inspection found no TEAR-CC↔CG/S project-code imports. CG/S Stage 2
uses only `tearrl.cgs` modules; TEAR-CC v0.3/v0.4 use `tearrl.cc` modules (with
v0.3 additionally using the older generic PPO gradient utility). Their actions,
states, data generators, models, evaluation contexts, and seeds are separate.

## Collision 1 — means near 0.8738

| study | seed ratios | arithmetic mean |
|---|---|---:|
| TEAR-CC v0.4 CoherentPath | `0.865495`, `0.880003`, `0.875014` | `0.8735038251` |
| CG/S oracle-hierarchical | `0.916075`, `0.784013`, `0.921323` | `0.8738035538` |

The means differ by `0.0002997287`, but their distributions do not resemble one
another. TEAR-CC is tightly clustered; CG/S combines one low seed with two seeds
above `0.916`.

TEAR-CC v0.4 raw per-seed ratios use governor-on task divided by ShieldOnly task:

| seed | numerator | denominator | ratio |
|---:|---:|---:|---:|
| 10103 | `0.5228849415` | `0.6041457830` | `0.8654946475` |
| 11213 | `0.5316482684` | `0.6041436122` | `0.8800031277` |
| 12323 | `0.5286299283` | `0.6041390305` | `0.8750136999` |

## Collision 2 — values near 0.784

TEAR-CC v0.3's reported `0.7849429202` is an arithmetic mean of three seeds:

| seed | numerator | denominator | ratio |
|---:|---:|---:|---:|
| 7789 | `0.4765487232` | `0.6041390208` | `0.7888063952` |
| 8893 | `0.4748294136` | `0.6041431646` | `0.7859551204` |
| 9901 | `0.4712734273` | `0.6041446174` | `0.7800672450` |

CG/S `0.7840125685` is one seed (`5207`), reconstructed deterministically in
memory from 319 eligible contexts:

`1.9600314212 / 2.5 = 0.7840125685`.

The numerator, denominator, scale, context count, aggregation level, and seed are
all different. The difference between the two displayed ratios is
`0.0009303517`.

## Shared and not shared

Shared:

- the abstract ratio `task achieved / matched baseline task`;
- the frozen competence boundary `0.90`;
- NumPy/PyTorch as general numerical dependencies.

Not shared:

- project module imports between `tearrl.cc` and `tearrl.cgs`;
- seeds (`7789, 8893, 9901, 10103, 11213, 12323` versus
  `4103, 5207, 6311`);
- arena state, action space, world generator, actor architecture, witness,
  training targets, task-eligible mask, or raw numerator/denominator scale.

The collision does not indicate contamination. It is worth disclosing because
both values sit near the same `0.90` competence boundary, making an accidental
visual resemblance more salient than the underlying data warrant.
