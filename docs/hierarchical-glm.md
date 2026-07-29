# Partial pooling with fixed or estimated scale

<figure class="doc-figure" data-figure-kind="Synthetic benchmark">
  <img src="../assets/hierarchical-pooling.svg" alt="Two benchmark plots showing subject-coefficient RMSE and prospective log loss for complete pooling, partial pooling, and independent fits as true between-animal variation increases.">
  <figcaption><strong>Synthetic benchmark · pooling under heterogeneity.</strong> Across the committed fixed-scale simulation, partial pooling has the lowest mean subject-coefficient RMSE and prospective log loss in all three heterogeneity regimes. This validates the declared design, not every population.<span class="doc-figure__meta"><strong>Unit:</strong> simulated subject · <strong>n:</strong> declared subjects across three heterogeneity regimes · <strong>Estimands:</strong> coefficient RMSE and prospective log loss · <a href="../reference/figure-provenance/">provenance</a></span></figcaption>
</figure>

`HierarchicalBernoulliHistoryGLM` is Behavio's first population model. It extends the
static Bernoulli history GLM with an inspectable coefficient vector for every training
subject while retaining a population-level vector:

\[
\operatorname{logit} P(y_{it}=1) = x_{it}^{\top}(\beta + b_i),
\qquad b_i \sim \mathcal{N}(0, \sigma_{subject}^{2}I).
\]

By default, the implementation performs one joint maximum-a-posteriori fit with
`subject_scale` fixed before fitting. Subject deviations receive the corresponding
Gaussian penalty, and the optional `l2` penalty applies only to the non-intercept
population coefficients.

```python
from behavio import HierarchicalBernoulliHistoryGLM

model = HierarchicalBernoulliHistoryGLM(
    covariates=("stimulus",),
    choice_lags=1,
    subject_scale=0.5,
)
fit = model.fit(study)

print(fit.parameters)  # population coefficients
print(fit.subjects)  # stable row order
print(fit.subject_deviations)  # deviations from the population
print(fit.subject_coefficients)  # population + deviation
```

## Estimating the subject scale

Set `estimate_subject_scale=True` to estimate one shared scale from the training data:

```python
model = HierarchicalBernoulliHistoryGLM(
    covariates=("stimulus",),
    choice_lags=1,
    subject_scale=0.4,
    estimate_subject_scale=True,
    subject_scale_bounds=(0.05, 1.5),
)
fit = model.fit(study)

print(fit.subject_scale)
print(fit.subject_scale_standard_error)
print(fit.subject_scale_confidence_interval_95)
print(fit.subject_scale_at_boundary)
```

In this mode, `subject_scale` is the optimizer's initial value, while
`subject_scale_bounds` are declared scientific bounds. The fit maximizes a Laplace-
approximated marginal likelihood: subject deviations are integrated out approximately
rather than treating the scale as another raw joint-MAP coordinate. The latter would be
degenerate because a vanishing scale and vanishing deviations can improve the normalized
joint density without establishing that the population variance is zero.

The estimation flag affects fitting only. If the same model object is used to simulate,
its configured `subject_scale` remains the generative scale; recovery studies should
therefore use separate generator and estimator objects, as the benchmark does.

The returned scale standard error comes from a numerical local Hessian in population-
coefficient and log-scale coordinates. The 95% interval is a delta-method log-scale
interval. A scale at either configured bound sets both `subject_scale_at_boundary` and the
common fit boundary diagnostic. Such a result is unresolved beyond that bound, not an
estimate of exact zero or of the exact upper limit.

## Seen and unseen subjects

For a subject represented in the fitted data, prediction uses its estimated deviation.
For a subject absent from the fitted data, prediction uses the population coefficient
vector. The fit records this policy as `population-mean-plugin`, and
`fit.subject_was_fitted(subject)` makes the distinction queryable. This is a point plug-in,
not integration over a new subject's random-effect distribution.

The policy lets population-held-out folds run without silently manufacturing a subject
effect from test outcomes. It does not make their uncertainty complete: predictive
intervals for a genuinely new subject require integrating both population and random-
effect uncertainty.

## Simulation and recovery

`simulate()` returns an ordinary observed `Study`. `simulate_with_effects()` additionally
returns a `HierarchicalGLMSimulation`, keeping the realized population and subject truth
separate from the observed columns. This prevents recovery metadata from leaking into
fitting code while making subject-level recovery testable.

The [fixed-scale benchmark](https://github.com/aeronjl/behavio/tree/main/benchmarks/hierarchical_glm) compares complete
pooling, independent fits, and partial pooling on the same generated animals and future
sessions. Its scale is fixed to the known generative value, so the benchmark validates the
shrinkage mechanism rather than hyperparameter selection.

The [subject-scale recovery benchmark](https://github.com/aeronjl/behavio/tree/main/benchmarks/subject_scale_recovery)
crosses two population sizes with three true scales, checks approximate interval coverage,
and compares future-session predictions with an oracle given the true scale.

## Full posterior inference with PyMC

The optional [PyMC backend](pymc-backend.md) samples the fixed-scale version of this same
model with NUTS. It preserves the flat intercept, L2-equivalent population priors, Gaussian
subject deviations, filtered-history design, and task denominator while returning labelled
posterior, likelihood, predictive, observed-data, and sampler-diagnostic groups.

This is full posterior inference conditional on the declared fixed `subject_scale`. The
`estimate_subject_scale=True` Laplace path remains empirical Bayes and is rejected by the
adapter because it does not define a full-posterior scale prior.

## Current boundary

The deterministic MAP and empirical-Bayes paths remain deliberately short of a full
hierarchical Bayesian model. The optional fixed-scale PyMC path propagates coefficient and
subject-deviation uncertainty, but the family as a whole still has these boundaries:

- there is one independent, shared scale for every coefficient rather than separate or
  correlated variance components;
- scale estimation uses a Laplace approximation and local-Hessian uncertainty rather than
  a posterior distribution; the PyMC path conditions on a fixed scale;
- estimated-scale subject standard errors are conditional on population coefficients and
  the fitted scale;
- deterministic-path predictions do not integrate parameter or random-effect uncertainty;
  the first PyMC adapter currently retains in-sample posterior predictive draws only;
- subject coefficients are static across sessions;
- there are no nested lab effects, correlated random effects, or missing-data model.

Those omissions keep the first population contract auditable. Multiple variance
components, calibrated posterior prediction, and dynamic subject trajectories require
their own recovery and calibration benchmarks.
