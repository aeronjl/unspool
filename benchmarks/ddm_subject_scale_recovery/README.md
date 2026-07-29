# Parameter-specific longitudinal DDM scale recovery

This benchmark asks whether the longitudinal hierarchical Wiener model can distinguish
heterogeneity in stimulus sensitivity from heterogeneity in boundary separation without
receiving either scale from the generator. It also keeps the fourth session strictly
prospective and compares prediction with an oracle given the true scales.

```bash
uv run python -m benchmarks.ddm_subject_scale_recovery.benchmark
```

The pinned design crosses 6 and 12 animals. Each animal contributes three 45-trial
training sessions and one held-out future session. The true natural-scale components are
`0.22` for stimulus drift and `0.07` for boundary; both estimates start at `0.14` within
bounds `(0.03, 0.5)`. Eight repeated datasets are retained at each population size.

The estimator alternates a joint path MAP step with parameter-specific variance updates
under a local Gaussian approximation. Its intervals use the Louis observed-information
identity: the M-step's expected-prior curvature is complete-data information, and the
conditional variance of the complete-data score is subtracted from it. They remain
approximate posterior intervals, not exact ones. Bound hits and iteration-limit exits
remain visible.

## Pinned result

All 16 variance procedures and final joint fits converge. Moving from 6 to 12 animals
reduces joint scale RMSE from `0.06144` to `0.04806`. Mean future-session log loss is
within `0.00081` of the oracle at 6 animals and within `0.00070` at 12 animals.

The two components are not equally easy. Drift-scale RMSE falls from `0.07883` to
`0.06603`, while boundary-scale RMSE falls from `0.03654` to `0.01613`. One 6-animal
boundary estimate hits the lower bound. Interval coverage is 100% for both components at 6
animals, and 100% and 87.5% at 12 animals, against a nominal 95%.

## Two limits this result does not remove

The corrected interval is conservative rather than exact. Its mean standard error against
the Monte Carlo sampling standard deviation of the estimates is `2.07x` and `0.92x` at 6
animals and `1.67x` and `1.45x` at 12 animals, for drift and boundary respectively. Three
of the four cells over-cover; the reported rate is an upper bound on how tight this
procedure is, not a demonstration that it is calibrated.

The drift-scale point estimate is biased low. Its mean is `0.15827` at 6 animals and
`0.17412` at 12, against a truth of `0.22` — 28.1% and 20.9% low. Boundary scale is close
to unbiased (`0.07721` and `0.07558` against `0.07`). This is EM/Laplace shrinkage in the
point estimate; the debiased simulator does not address it, and the wide interval absorbs
rather than corrects it. It is recorded here as a known open gap.

The exact output, including every seed and fit audit, is retained in
[`result.json`](result.json). Eight repetitions per population size establish a regression
target, not a definitive coverage study.
