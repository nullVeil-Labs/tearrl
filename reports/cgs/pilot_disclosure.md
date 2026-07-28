# TEAR-CG/S v0.1 — permanent pilot disclosure

**Status:** post-pilot provenance disclosure; scientific implementation and
confirmatory outcomes unchanged.

## The required sentence

**Pilot disclosure:** because the fitted constant witness was all-positive, the
original raw-FNR non-reversal criterion was unpassable by construction for any
learned witness with a nonzero worst-party false-negative count. It failed; it
was replaced; the replacement passed on identical preserved data.

## Exact sequence

1. The frozen pilot seed `5101` ran under the original implementation.
2. The learned witness had worst-party FNR `0.104260`; the all-positive constant
   witness had FNR `0.0` and FPR `1.0`. Raw FNR therefore declared a reversal
   even though the constant witness marked every negative row as harmful.
3. The original output was copied byte-for-byte before any amendment.
4. Amendment 002 added FPR and operationalized the previously undefined
   constant/base-rate non-reversal comparison as balanced worst-party error,
   irreversible FNR, and Brier. No primary gate, threshold, seed, model, data,
   loss, architecture, or optimizer changed.
5. The pilot was rerun from fresh deterministic initialization. The replacement
   comparison passed. A recursive comparison found every old/new JSON value
   identical after removing only the registered FPR, baseline-decision, and
   interpretation fields.

The replacement does not rescue the pilot: G1–G6 still fail and only G7 passes.

## Permanently retained artifacts

| generation | artifact | SHA-256 |
|---|---|---|
| original | `pilot_pre_amendment_002_results.json` | `7c6aa9136546cd35c30da78081dbb1103c4bf81a86a81bf8d98b70dc258b3b13` |
| original | `pilot_pre_amendment_002_results.md` | `c84f9a34830f1f54715d279b8817a5a5388efcd4d6d922b135701fc068451656` |
| amended | `pilot_results.json` | `d20ec428e86307447b81e94fff8b024e835a9593d47e104549951cf2ee4dcd17` |
| amended | `pilot_results.md` | `9389c9dba3032200325f768a9197eb9e86ebb383835264bdf86181080283e700` |

All four hashes are bound by `tearrl/cgs/frozen_results.json` and enforced by
tests. The amended file is not allowed to replace or obscure the original.
