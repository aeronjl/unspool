# Flagship prospective longitudinal study

This benchmark is Behavio's first end-to-end scientific product: the same prospective
question is asked of two checksum-pinned public datasets while their distinct sampling
frames remain explicit. It forecasts each animal's sixth aligned session from its first
five and compares four explanations of longitudinal behavioural structure:

1. one static coefficient vector for every animal;
2. static, partially pooled individual differences;
3. one smooth trajectory shared by every animal; and
4. smooth, partially pooled individual trajectories.

The exercise is a comparative forecast, not a retrospective description. One cohort fold
jointly fits all animals, and no held-out sixth-session choice is available during fitting.
Its scoring, paired uncertainty, audits, and fold provenance are produced by Behavio's
public `ProspectiveComparisonReport` rather than benchmark-local comparison code.

## Fixed analysis contract

Both panels contain six sessions per animal: the first three and final three eligible
sessions in the source dataset. Their source session orders are retained, while
`session_order = 0, ..., 5` supplies the explicitly aligned analysis clock. The fifth
aligned session is the last training session and the sixth is the test session. This
alignment compares trajectory position; it does not assert equal calendar time, training
exposure, or task protocol across datasets.

All models use rightward binary choice, signed stimulus, and one session-reset choice lag.
The smooth models use knots at aligned ranks 0, 2, and 5. The basis, `l2 = 0.02`, smoothness
of 3, subject scale of 0.4, and subject smoothness of 3 were fixed from the preceding
[synthetic trajectory benchmark](../trajectory_recovery/README.md), not selected from
these held-out scores.

The primary metric is log loss averaged first within animal and then equally across
animals. Trial-pooled log loss is retained as a secondary estimand because session sizes
vary substantially. Uncertainty uses 5,000 paired nonparametric subject-bootstrap draws;
all pairwise differences, individual subject losses, complete fit audits, and fitted
stimulus paths are retained in [`result.json`](result.json).

## Source-specific panels

The Cell panel applies the published behavioural exclusions, retains the 30 mice entering
the original early-learning analysis, excludes the paper's `ALK`/`MMM` session groups, and
then selects first-three/final-three canonical sessions. It contains 33,814 trials.

The IBL panel uses the pre-existing outcome-blind nine-lab selection contract: one animal
per lab, with three early and three final pre-transition training sessions. It drops the
303 source rows with invalid (`0`) choices and contains 28,097 trials. IBL ALF rightward
choices (`-1`) are recoded to the same binary-right convention as the Cell panel.

## Result

| Dataset | Lowest point estimate | Subject-balanced log loss | Difference from complete pooling (95% paired bootstrap interval) |
| --- | --- | ---: | ---: |
| Cell 2025, 30 mice | Static partial pooling | 0.5412 | +0.0376 (+0.0114, +0.0678) |
| IBL 2021, 9 mice | Shared smooth drift | 0.6310 | +0.0259 (−0.1949, +0.1909) |

Positive differences mean lower loss than complete pooling. All eight model fits pass the
normalized numerical audit.

The Cell result supports individual heterogeneity, but not a sharp distinction between
stable and smoothly changing differences: static partial pooling beats the hierarchical
smooth model by only 0.0021 log-loss units, with a paired interval from −0.0395 to +0.0475
when expressed as the hierarchical-minus-static loss difference. The smooth hierarchical
fit nevertheless retains positive stimulus-sensitivity changes for every animal; its
descriptive paths should not be confused with a prospectively resolved advantage.

The IBL point ranking is less stable. Shared smooth drift is 0.0052 below static partial
pooling, but that paired difference has an interval spanning roughly −0.184 to +0.251.
Nine subjects are insufficient to distinguish these close structural accounts here. The
trial-pooled and subject-balanced rankings also differ, demonstrating why the primary
aggregation rule had to be declared rather than chosen after inspection.

## Reproduce

```bash
uv run python -m benchmarks.cell2025.fetch_data
uv run python -m benchmarks.ibl2021.fetch_data
uv run --with pyarrow python -m benchmarks.flagship_longitudinal.benchmark
```

The Cell fetcher reads only the required compressed member from the 11.3 GB archive. Both
adapters verify their pinned source hashes before fitting. Raw data stay ignored by Git;
only the compact machine-readable result is committed.

## Interpretation boundary

This is evidence about one six-session alignment, one covariate/history specification,
and fixed regularization choices. It is not a model-recovery result, proof that parameters
are biologically stationary, population inference for either source cohort, or a
cross-dataset effect-size comparison. In particular:

- the Cell and IBL inclusion mechanisms and task protocols differ;
- collapsed session rank discards the unequal source-time gaps that remain recorded in
  `source_session_order`;
- the IBL panel has one animal per lab, so animal and lab variation cannot be separated;
- bootstrap intervals quantify finite-subject sampling variation, not hyperparameter,
  source-selection, or landmark uncertainty; and
- these four Bernoulli GLMs do not exhaust learning, latent-state, reinforcement-learning,
  reaction-time, or drift-diffusion explanations.

The result is therefore a falsifiable baseline and an executable study contract. Its main
scientific contribution is to keep the competing explanations, forecast origin,
aggregation, uncertainty, and failed-fit policy visible enough to be improved.
