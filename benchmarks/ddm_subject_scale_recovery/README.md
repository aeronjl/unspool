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
under a local Gaussian approximation. Its intervals use final expected-prior curvature
and should be read as diagnostics, not exact posterior credible intervals. Bound hits and
iteration-limit exits remain visible.

## Pinned result

All 16 variance procedures and final joint fits converge. Moving from 6 to 12 animals
reduces joint scale RMSE from `0.09178` to `0.05138`. Mean future-session log loss is
within `0.00233` of the oracle at 6 animals and within `0.00080` at 12 animals.

The two components are not equally easy. Drift-scale RMSE falls from `0.12741` to
`0.06859`, while boundary-scale RMSE remains near `0.024`. One 6-animal boundary estimate
hits the lower bound. Approximate interval coverage ranges from 50% to 62.5%, which is far
below nominal; the intervals are retained as local curvature diagnostics and must not be
reported as calibrated uncertainty.

The exact output, including every seed and fit audit, is retained in
[`result.json`](result.json). Eight repetitions per population size establish a regression
target, not a definitive coverage study.
