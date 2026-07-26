# Nested prospective selection recovery

This benchmark validates the complete training-only selection procedure rather than one
model in isolation. It asks whether nested prospective validation distinguishes a static
population from shared smooth drift without making the final test sessions available when
the candidate is chosen.

```bash
uv run python -m benchmarks.nested_selection.benchmark
```

## Design

Twenty matched repetitions of each generating regime use ten subjects, seven sessions,
and 40 trials per session. A static Bernoulli history GLM competes with a shared smooth GLM
whose fixed knots are at sessions 0, 3, and 6. Both candidates use stimulus, one
session-reset choice lag, and the same `l2` penalty.

Each dataset has two outer origins:

- train on sessions 0–4 and test session 5;
- train on sessions 0–5 and test session 6.

Within each outer training study, inner cohort folds begin after three sessions and use
only the rows supplied by that training study. The candidate with the lower inner
subject-balanced log loss is refitted on the complete outer training set and evaluated on
the untouched outer test session. Exact ties follow declared candidate order.

## Pinned result

| Generating regime | Expected candidate selected | Datasets recovering at both outer origins | Mean nested outer log loss |
| --- | ---: | ---: | ---: |
| Stationary | 37 / 40 folds (92.5%) | 17 / 20 (85%) | 0.5818 |
| Shared smooth drift | 40 / 40 folds (100%) | 20 / 20 (100%) | 0.4328 |

All 80 selected outer fits pass the normalized fit audit. Under stationarity, always using
the static candidate has mean outer log loss 0.5800, so the three incorrect smooth
selections impose a small observed cost. Under shared drift, the nested procedure always
selects the smooth candidate and matches its mean outer loss; the static candidate's mean
loss is 0.5418.

The exact per-repetition selections, outer losses, inner-fold counts, and audit contract
are retained in [`result.json`](result.json).

## What this establishes

For these fixed effect sizes and this design, the nested procedure usually protects the
stationary regime from unnecessary flexibility and reliably detects strong shared drift.
The accompanying API test changes every outer-test outcome while holding outer training
data fixed; the chosen candidate and serialized inner report remain identical. That is a
direct regression against outcome leakage, not merely an assertion based on code layout.

## Interpretation boundary

Recovery is conditional on this balanced design, these two candidates, fixed knots and
penalties, and relatively strong drift. It does not validate post-hoc candidate grids,
weak or individual-specific drift, arbitrary missing follow-up, or the coverage of the
outer subject bootstrap. The 7.5% stationary fold error is retained as part of the result
rather than hidden by changing the seed or signal. Those extensions require their own
design-specific recovery studies.
