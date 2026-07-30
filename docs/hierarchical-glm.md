# Partial pooling with fixed or estimated scale

<figure class="doc-figure" data-figure-kind="Synthetic benchmark">
  <img src="../assets/hierarchical-pooling.svg" alt="Two benchmark plots showing subject-coefficient RMSE and prospective log loss for complete pooling, partial pooling, and independent fits as true between-animal variation increases.">
  <figcaption><strong>Synthetic benchmark · pooling under heterogeneity.</strong> Across the committed fixed-scale simulation, partial pooling has the lowest mean subject-coefficient RMSE and prospective log loss in all three heterogeneity regimes. This validates the declared design, not every population.<span class="doc-figure__meta"><strong>Unit:</strong> simulated subject · <strong>n:</strong> declared subjects across three heterogeneity regimes · <strong>Estimands:</strong> coefficient RMSE and prospective log loss · <a href="../reference/figure-provenance/">provenance</a></span></figcaption>
</figure>

`hierarchical(model, over="subject")` is Behavio's population combinator. It gives any
composable model an inspectable parameter vector for every training group while retaining
a population-level vector:

\[
\operatorname{logit} P(y_{it}=1) = x_{it}^{\top}(\beta + b_i),
\qquad b_i \sim \mathcal{N}(0, \operatorname{diag}(\sigma^{2})).
\]

By default, the implementation performs one joint maximum-a-posteriori fit with the scales
fixed before fitting. Group deviations receive the corresponding Gaussian penalty, and the
wrapped model's own penalty -- for a GLM, the optional `l2` on non-intercept coefficients
-- continues to apply to the population vector alone.

```python
from behavio import BernoulliHistoryGLM
from behavio.compose import hierarchical

model = hierarchical(
    BernoulliHistoryGLM(predictors=("stimulus",), choice_lags=1),
    over="subject",
    scale=0.5,
)
fit = model.fit(study)

print(fit.parameters)  # population parameters
print(fit.groups)  # stable row order
print(fit.group_deviations)  # deviations from the population
print(fit.group_parameters)  # population + deviation
```

This replaces the deleted `HierarchicalBernoulliHistoryGLM`, which had one
`subject_scale` shared by every coefficient and so could not say that the bias varies
between animals while the stimulus sensitivity does not. Name the varying parameters and
their scales instead:

```python
model = hierarchical(
    BernoulliHistoryGLM(predictors=("stimulus",), choice_lags=1),
    over="subject",
    parameters=("intercept", "choice_lag_1"),
    scale=0.5,
    parameter_scales={"choice_lag_1": 0.15},
)
```

`over=` is any study column, so `over="lab"` is a lab-level model. See
[composing models](composing-models.md).

## Estimating the subject scale

Set `estimate_scale=True` to estimate one common multiplier on the declared scales from
the training data:

```python
model = hierarchical(
    BernoulliHistoryGLM(predictors=("stimulus",), choice_lags=1),
    over="subject",
    scale=0.4,
    estimate_scale=True,
    scale_bounds=(0.05, 1.5),
)
fit = model.fit(study)

print(fit.scales)
print(fit.scale_standard_error)
print(fit.scale_confidence_interval_95)
print(fit.scale_at_boundary)
```

In this mode, `scale` is the optimizer's initial value, while `scale_bounds` are declared
scientific bounds on the first varying parameter's scale. The fit maximizes a Laplace-
approximated marginal likelihood: subject deviations are integrated out approximately
rather than treating the scale as another raw joint-MAP coordinate. The latter would be
degenerate because a vanishing scale and vanishing deviations can improve the normalized
joint density without establishing that the population variance is zero.

The estimation flag affects fitting only. If the same model object is used to simulate,
its configured scales remain the generative ones; recovery studies should
therefore use separate generator and estimator objects, as the benchmark does.

The returned scale standard error comes from a numerical local Hessian in population-
coefficient and log-scale coordinates. The 95% interval is a delta-method log-scale
interval. A scale at either configured bound sets both `scale_at_boundary` and the
common fit boundary diagnostic. Such a result is unresolved beyond that bound, not an
estimate of exact zero or of the exact upper limit.

## Seen and unseen subjects

For a subject represented in the fitted data, prediction uses its estimated deviation.
For a subject absent from the fitted data, prediction uses the population parameter
vector. The fit records this policy as `population-plugin`, and
`fit.group_was_fitted(subject)` makes the distinction queryable. This is a point plug-in,
not integration over a new subject's random-effect distribution.

The policy lets population-held-out folds run without silently manufacturing a subject
effect from test outcomes. It does not make their uncertainty complete: predictive
intervals for a genuinely new subject require integrating both population and random-
effect uncertainty.

## Simulation and recovery

`simulate()` returns an ordinary observed `Study`. `simulate_with_effects()` additionally
returns a `HierarchicalSimulation`, keeping the realized population and group truth
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

This is full posterior inference conditional on the declared fixed scales. The
`estimate_scale=True` Laplace path remains empirical Bayes and is rejected by the adapter
because it does not define a full-posterior scale prior.

## Current boundary

The deterministic MAP and empirical-Bayes paths remain deliberately short of a full
hierarchical Bayesian model. The optional fixed-scale PyMC path propagates coefficient and
subject-deviation uncertainty, but the family as a whole still has these boundaries:

- the per-parameter scales are independent: there are no correlated variance components,
  and `estimate_scale` estimates one common multiplier rather than each scale separately;
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
