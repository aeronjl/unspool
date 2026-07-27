# Replicated IBL prospective hierarchical model comparison

This benchmark asks whether session-varying individual trajectories improve actual future
prediction, first for already-observed animals and then across an entirely unseen lab. It
is the prospective modelling successor to the
[replicated IBL cohort benchmark](../ibl2021_replicated/README.md), not another descriptive
trajectory analysis.

## Fixed design

The cohort is the checksum-pinned 2021 IBL behavioural panel: 78 animals in nine labs and
six outcome-blind endpoint windows per animal. To keep the public regression benchmark
bounded and give each session a comparable maximum contribution, the analysis retains up
to the first 100 source rows in every session. Nine short source sessions contribute all
their available rows. The cap is applied before reading choice direction or choice
validity. Rows with source `choice = 0` are then excluded because the Bernoulli models
require an observed left/right response. The resulting panel contains 46,152 trials.

All source mappings are fixed rather than learned:

- IBL `choice = -1` becomes binary rightward choice `1`;
- signed contrast is `contrastRight - contrastLeft`, treating the absent side as zero;
- the outcome-blind `window_position = 0, ..., 5` becomes the model clock;
- history is a one-trial choice lag that resets at each session.

No scaler, landmark, feature selection, hyperparameter, or trial threshold is learned from
the public outcomes. Model hyperparameters and knots `(0, 2, 5)` were fixed by the earlier
[synthetic trajectory-recovery benchmark](../trajectory_recovery/README.md).

## Two prospective boundaries

The within-animal comparison jointly fits all 78 animals on positions 0–4 and predicts
position 5. It tests whether earlier animal-specific paths help forecast the same animals.

The lab-transfer comparison uses nine folds. For each fold, it fits positions 0–4 from the
other eight labs and predicts position 5 for entirely unseen animals in the held-out lab.
The reusable `leave_one_lab_out_session_forecast_splits` contract rejects subject overlap,
lab overlap, non-common session coordinates, and training positions that do not precede
the test horizon.

Both comparisons use subject-balanced prospective log loss as their primary metric and a
5,000-draw paired subject bootstrap. The lab-transfer report also averages subject losses
within lab and then weights the nine labs equally, with a paired empirical-lab bootstrap.
Nine historical laboratories are not a probability sample of laboratories, so that
interval describes sensitivity to these sites rather than population-of-labs uncertainty.
Scoring is filtered and sequential within the test session: after a choice is observed it
can initialize the one-step-ahead history feature for the next trial. This is not an
open-loop joint forecast of the complete session at position 5.

## Result

| Forecast target | Static partial pooling | Hierarchical smooth drift | Static minus drift (95% interval) |
| --- | ---: | ---: | ---: |
| Same animals, future session | 0.6400 | **0.5549** | +0.0851 (+0.0162, +0.1460) |
| Unseen lab and animals, future session; subject-balanced | 0.6285 | **0.6049** | +0.0236 (−0.0744, +0.1049) |
| Unseen lab; lab-balanced secondary estimand | 0.6304 | **0.6207** | +0.0097 (−0.1017, +0.1176) |

Positive differences favor the drifting model. All 20 fits pass the normalized numerical
audit. The first result is evidence that drift carries prospective information for these
animals under this endpoint-window design. The cross-lab results do not resolve a
transport advantage: both paired intervals include zero, and drift varies substantially
across held-out labs.

This does not establish a continuous learning law. The clock is ordinal, the gap between
positions 2 and 3 varies by animal, and entry is conditioned on reaching the protocol
transition. It establishes the narrower claim that a predeclared drifting model improves
one-step endpoint-window forecasts in the represented animals.

## Reproduce

```bash
uv run --extra ibl python -m benchmarks.ibl2021_prospective.benchmark
```

The exact source tables share the ignored cache used by the replicated-cohort benchmark.
The committed `result.json` pins all scores, fold provenance, audits, and uncertainty.
