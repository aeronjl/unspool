# Subject-scale recovery and calibration

This benchmark tests the estimated-scale mode of
`hierarchical(model, over="subject", estimate_scale=True)` rather than assuming the
Gaussian subject scale is known.

> **The committed `result.json` is stale.** It was produced by the hand-written
> `HierarchicalBernoulliHistoryGLM`, whose Laplace profile wrote its quadratic forms in
> scalar GLM-specific shorthand. The generic profile is handed a penalty *matrix* and
> computes `0.5 x' P x` and `-0.5 logdet(P)` instead, which is the same objective
> reassociated. A re-run moves the continuous summaries by between `1e-6` and `3.4e-4`
> (largest: `mean_standard_error` at 24 subjects and scale 1.0). Every claim the benchmark
> makes is unchanged: `convergence_rate`, `coverage_95`, `boundary_rate` and
> `more_subjects_reduce_rmse` are bit-identical in all six regimes. The file is left
> untouched rather than re-stamped, because regenerating it here would record a
> `-dirty` provenance revision that nobody could reproduce; re-run it on a clean tree. It asks whether a bounded Laplace
marginal-likelihood estimate recovers low, moderate, and high heterogeneity; whether its
local uncertainty is calibrated in this design; and whether its future-session prediction
approaches a model given the true scale.

```bash
uv run python -m benchmarks.subject_scale_recovery.benchmark
```

The design crosses 8 and 24 subjects with true scales `0.1`, `0.5`, and `1.0`. Every
subject contributes three 35-trial training sessions and one held-out future session. Each
of the six regimes has 20 matched repetitions. The estimator always starts at `0.4` with
bounds `(0.05, 1.5)`; it is not initialized at the truth.

## Pinned result

All 120 estimated fits converge. Moving from 8 to 24 subjects reduces scale RMSE in every
true-scale regime. Approximate 95% log-scale interval coverage is between 95% and 100% in
these finite repetitions. Estimated-scale future-session log loss is within `0.00039` of
the oracle mean in every regime, with similarly small Brier-score differences.

The difficult case remains visible: at true scale `0.1`, the estimate reaches its lower
bound in 40% of 8-subject runs and 20% of 24-subject runs. A boundary estimate means this
design has not resolved smaller heterogeneity; it must not be reported as evidence that the
population variance is exactly the configured lower bound.

The exact output is retained in [`result.json`](result.json). Twenty repetitions give a
coarse interval-calibration check, not a definitive coverage study. The benchmark also
uses one common scale for all coefficients and subjects already represented in training.
