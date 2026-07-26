# Threshold-landmark uncertainty and resolution

This benchmark tests whether Unspool's uncertainty-aware threshold landmark distinguishes a
decisive learning transition from a marginal regime that only sometimes satisfies the same
operational criterion.

```bash
uv run python -m benchmarks.landmark_uncertainty.benchmark
```

Each of 30 repetitions contains four 30-trial sessions and a probability change at
cumulative trial 50. Accuracy rises from `0.15` to either `0.95` (decisive) or `0.72`
(marginal). Both regimes use an `0.8` threshold, a 15-trial detection window, three
consecutive qualifying windows, and 200 bootstrap draws.

The bootstrap causally smooths the binary training metric with a seven-trial window and
Jeffreys regularization, samples Bernoulli outcomes at the original chronological
positions, and reapplies the unchanged landmark rule. It assumes conditional independence
given that plug-in trajectory. It is not a Bayesian posterior, and it does not model
residual serial dependence.

## Pinned result

Decisive learning produces a point landmark in all 30 datasets and a landmark in every
bootstrap draw. Its equal-tailed 90% interval is 11.93 trials wide on average. Marginal
learning produces a point landmark in 83.33% of datasets and resolves only 82.05% of
bootstrap draws on average; conditional intervals are 38.44 trials wide when any draw
resolves.

Point landmarks remain the values used by the ordinary transform. Bootstrap draws produce
parallel landmark-relative clock samples, and their unresolved fraction exactly matches
the unresolved landmark fraction in every run. Intervals are calculated only among
resolved draws, so they must always be reported alongside resolution rate.

The exact output is retained in [`result.json`](result.json). These results characterize
this declared criterion, sample size, smoothing rule, and two probability trajectories.
They do not establish universal interval calibration or a unique definition of learning.
