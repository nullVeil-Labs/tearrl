# Responsibility attribution anti-vacuity report

Status: **descriptive only; no TEARRL-0 protocol or reward change**.

- Frozen tuple genealogy candidates: `1000`
- Recorded learner decisions: `7627`
- Generator seed: `7301`
- Frozen witness SHA-256: `9539663997b7abceff2d21ea0767a533bbd5087d78885aef695834e4bdc9ba87`

## Readings

- `current`: the implemented denominator, measured before the current learner action.
- `completed_window`: includes the current learner action in the same net decision window.
- `contribution_share`: bounded absolute-contribution diagnostic; reported because net
  cancellation can make `completed_window` exceed one.

| reading | n | zero | 0<r<1 | r>=1 | median_nonzero | max | variance | effective_support | lag1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current | 7627 | 85.7087% | 2.1371% | 12.1542% | 2.307692 | 10.000000 | 1.431692 | 1.757 | -0.009868 |
| completed_window | 7627 | 85.7087% | 11.5380% | 2.7534% | 0.697674 | 2.400000 | 0.170026 | 1.800 | 0.326355 |
| contribution_share | 7627 | 85.7087% | 14.2913% | 0.0000% | 0.697674 | 0.909091 | 0.057956 | 1.757 | 0.034342 |

## Old versus completed-window reading

- Fraction changed: `14.2913%`
- Mean absolute change: `0.295135`
- Maximum absolute change: `9.090909`

`effective_support` is exp(Shannon entropy) over values rounded to 12 decimals.
Lag-1 correlation uses only adjacent decisions from the same episode.

## Interpretation boundary

This report measures a timing ambiguity. It does not select a replacement denominator.
Changing the registered `r` update requires a separately frozen successor specification;
TEARRL-0 continues to use the `current` reading. The accompanying known-answer and
mutation tests prove that whole-effect, fractional-effect, non-learner-effect,
zero-denominator, clipped-effect, and one-turn-shift cases are distinguished.
