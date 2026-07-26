# Fixed-scale partial pooling

`HierarchicalBernoulliHistoryGLM` is Unspool's first population model. It extends the
static Bernoulli history GLM with an inspectable coefficient vector for every training
subject while retaining a population-level vector:

\[
\operatorname{logit} P(y_{it}=1) = x_{it}^{\top}(\beta + b_i),
\qquad b_i \sim \mathcal{N}(0, \sigma_{subject}^{2}I).
\]

The implementation performs one joint maximum-a-posteriori fit. `subject_scale` is
\(\sigma_{subject}\): a positive value fixed before fitting. Subject deviations receive
the corresponding Gaussian penalty, and the optional `l2` penalty applies only to the
non-intercept population coefficients.

```python
from unspool import HierarchicalBernoulliHistoryGLM

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

The [fixed-scale benchmark](../benchmarks/hierarchical_glm/README.md) compares complete
pooling, independent fits, and partial pooling on the same generated animals and future
sessions. Its scale is fixed to the known generative value, so the benchmark validates the
shrinkage mechanism rather than hyperparameter selection.

## Current boundary

This is deliberately not a full hierarchical Bayesian model:

- `subject_scale` is fixed, shared by every coefficient, and not estimated;
- estimates are joint MAP points, with local-Hessian approximate standard errors;
- predictions do not integrate parameter or random-effect uncertainty;
- subject coefficients are static across sessions;
- there are no nested lab effects, correlated random effects, or missing-data model.

Those omissions keep the first population contract auditable. Variance-component
estimation and dynamic subject trajectories should be added only with dedicated recovery
and calibration benchmarks.
