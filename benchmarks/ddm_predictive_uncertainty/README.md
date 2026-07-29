# Longitudinal DDM predictive uncertainty

This benchmark tests two uncertainty contracts for the parameter-specific subject scales
of `HierarchicalSmoothWienerDriftDiffusion`, then scores entirely unseen animals after
integrating over fitted random-effect paths. It uses 20 matched panels, each with eight
training animals, four new animals, three sessions, and 35 trials per session.

The uncorrected interval uses the final expected-prior curvature, which is complete-data
information and therefore systematically too narrow; the library no longer reports it by
default. The opt-in supplemented interval numerically differentiates one forced Laplace-EM
update and applies the resulting missing-information correction in log-scale coordinates.
It refuses to report an interval when the EM map is not locally stable. This is the
supplemented EM construction of
[Meng and Rubin (1991)](https://doi.org/10.1080/01621459.1991.10475130), applied here to
the library's approximate variance-component update.

## Pinned result

All 20 outer parameter fits converged. Once the simulator's Brownian-bridge absorption
test removed its boundary overshoot, the supplemented covariance became locally stable in
all 20 panels; the largest EM spectral radius is `0.89677`, against `0.99020` before. The
refusal path is retained and still fires on an unstable map, but it is no longer exercised
by this pinned design. Drift-scale coverage rose from 50% locally to 100%, and
boundary-scale coverage rose from 85% to 100%. Mean interval widths increased from
`0.12431` to `0.22391` for drift and from `0.05309` to `0.08723` for boundary. Twenty
panels are enough to reject the known narrow local approximation here, but not to claim
universal nominal calibration; the supplemented interval over-covers rather than being
exact.

For each of 80 unseen animals, the predictive calculation draws one coherent deviation
trajectory per Monte Carlo draw and scores the subject's trials jointly. Relative to the
population-trajectory plug-in, it improves mean joint log probability by `0.98299`, wins
for 68.75% of animals, and has median effective draws of `904.79` out of 4,096. The
minimum effective draws (`26.23`) and maximum log-score Monte Carlo standard error
(`0.19463`) are retained rather than hidden; both are worse than before, so the mean
improvement is carried by a heavier tail of animals and should not be read as a uniform
gain. Fitted population and scale uncertainty are not integrated, so this remains
empirical-Bayes random-effect prediction rather than a full posterior predictive
distribution.

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
