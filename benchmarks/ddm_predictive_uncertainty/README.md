# Longitudinal DDM predictive uncertainty

This benchmark tests two uncertainty contracts for the parameter-specific subject scales
of `HierarchicalSmoothWienerDriftDiffusion`, then scores entirely unseen animals after
integrating over fitted random-effect paths. It uses 20 matched panels, each with eight
training animals, four new animals, three sessions, and 35 trials per session.

The uncorrected interval uses the final expected-prior curvature. The opt-in supplemented
interval numerically differentiates one forced Laplace-EM update and applies the resulting
missing-information correction in log-scale coordinates. It refuses to report an interval
when the EM map is not locally stable. This is the supplemented EM construction of
[Meng and Rubin (1991)](https://doi.org/10.1080/01621459.1991.10475130), applied here to
the library's approximate variance-component update.

## Pinned result

All 20 outer parameter fits converged. The supplemented covariance was locally stable in
18/20 panels; the other two remain explicit failures. Conditional on those 18 finite
results, drift-scale coverage rose from 70% locally to 100%, and boundary-scale coverage
rose from 65% to 88.9%. Mean interval widths increased from `0.12017` to `0.22623` for
drift and from `0.04998` to `0.12221` for boundary. Twenty panels are enough to reject the
known narrow local approximation here, but not to claim universal nominal calibration.

For each of 80 unseen animals, the predictive calculation draws one coherent deviation
trajectory per Monte Carlo draw and scores the subject's trials jointly. Relative to the
population-trajectory plug-in, it improves mean joint log probability by `0.79135`, wins
for 70% of animals, and has median effective draws of `997.67` out of 4,096. The minimum
effective draws (`70.44`) and maximum log-score Monte Carlo standard error (`0.11813`) are
retained rather than hidden. Fitted population and scale uncertainty are not integrated,
so this remains empirical-Bayes random-effect prediction rather than a full posterior
predictive distribution.

An earlier whole-subject bootstrap prototype was not shipped: its pilot intervals
undercovered even the already narrow local intervals. That failure is consistent with the
warning that directly bootstrapping fitted best linear unbiased predictors can suppress
random-effect variation; see [Morris (2002)](https://doi.org/10.1016/S0167-7152(02)00041-X).
The rejected method is recorded here so it is not quietly rediscovered as a default.

```bash
uv run python -m benchmarks.ddm_predictive_uncertainty.benchmark
```

The complete machine-readable result, including every interval, stability decision,
subject score, effective sample size, and Monte Carlo error, is pinned in
[`result.json`](result.json).
