# PyMC hierarchical GLM backend

Unspool's first full-posterior adapter samples the existing
`HierarchicalBernoulliHistoryGLM` with PyMC's established NUTS implementation. It is a
narrow interoperability reference, not a new sampler or a second behavioural model.

The adapter reuses the model's own binary-outcome validation and filtered-history design
matrix. It also requires the same `TaskSpec` used by the deterministic golden path, so an
undeclared covariate or mismatched scored observation fails before PyMC is imported.

```python
from unspool import (
    ChoiceSpec,
    HierarchicalBernoulliHistoryGLM,
    PyMCHierarchicalGLMBackend,
    TaskSpec,
    audit_posterior,
    psis_loo,
)

task = TaskSpec(
    choice=ChoiceSpec(options=(0, 1)),
    predictors=("stimulus",),
)
model = HierarchicalBernoulliHistoryGLM(
    covariates=("stimulus",),
    choice_lags=1,
    l2=0.25,
    subject_scale=0.5,
)
backend = PyMCHierarchicalGLMBackend(
    draws=1_000,
    tune=1_000,
    chains=4,
    cores=4,
    target_accept=0.9,
    seed=2026,
)

posterior = backend.sample(model, study, task=task)
audit = audit_posterior(posterior)
loo = psis_loo(posterior)

if audit.issues:
    for issue in audit.issues:
        print(issue.code, issue.targets)
```

Install the backend only when needed:

```bash
pip install "unspool[bayesian]"
```

PyMC 5.28 is used on Python 3.11 and current PyMC 6 on Python 3.12 and later. Both are
tested with real sampling. PyMC's `sample` function uses independent random streams for
chains and supports convergence diagnostics; the backend fixes the sampler name, initial
method, and complete configuration in result provenance. See the
[PyMC sampling reference](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.sample.html).

## Model and prior identity

The fitted model is unchanged:

\[
\operatorname{logit} P(y_{it}=1)
= x_{it}^{\top}(\beta + b_i),
\qquad b_i \sim \mathcal{N}(0, \sigma_{subject}^{2}I).
\]

The full-posterior graph exactly reflects the fixed-scale MAP path's declared penalties:

- the population intercept has a flat prior;
- non-intercept population coefficients are flat when `l2=0` and otherwise have
  `Normal(0, 1 / sqrt(l2))` priors; and
- subject deviations have `Normal(0, subject_scale)` priors.

Flat priors preserve the existing model definition but do not guarantee posterior
propriety under complete or quasi-complete separation. For routine use, declare a positive
`l2` unless a flat-slope analysis is scientifically required, and treat finite sampler
output as insufficient evidence of a proper posterior.

`estimate_subject_scale=True` is rejected. The empirical-Bayes MAP path defines bounded
Laplace marginal-likelihood estimation, not a prior on the scale. Sampling it as if those
bounds implied a Bayesian prior would change the model while pretending only the backend
changed. A later varying-scale model must declare and recover its prior explicitly.

## Retained evidence

The returned `PosteriorResult` contains:

- `population_coefficient`, `subject_deviation`, and derived `subject_coefficient` draws;
- trial-level `choice_probability` draws using observed past choices only;
- PyMC's per-draw NUTS diagnostics in `sample_stats`;
- pointwise Bernoulli values in `log_likelihood`;
- replicated choices in `posterior_predictive`;
- the observed choices and fixed design matrix; and
- model signature, task denominators, backend version/configuration, scored columns, and
  prior descriptions.

```python
population = posterior["posterior"]["population_coefficient"]
deviations = posterior["posterior"]["subject_deviation"]
log_likelihood = posterior["log_likelihood"]["choice"]
diverging = posterior["sample_stats"]["diverging"]

print(population.dims)  # chain, draw, coefficient
print(deviations.dims)  # chain, draw, subject, coefficient
print(diverging.values.sum())
```

PyMC computes the pointwise likelihood from the graph after sampling, and
`sample_posterior_predictive` generates observations conditional on the retained draws.
The latter also supports out-of-sample prediction when model data and coordinates are
changed; Unspool will expose that through its prospective split contract in a later slice.
See the [PyMC posterior-predictive reference](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.sample_posterior_predictive.html).

Use `posterior.to_arviz()` for the installed ArviZ representation. The standard groups and
labelled axes are described in [labelled posterior results](posterior-results.md); the
backend-neutral convergence policy is described in
[posterior convergence diagnostics](posterior-diagnostics.md), and the pointwise
predictive-fit calculation in [PSIS-LOO predictive evaluation](psis-loo.md).

## Interpretation boundary

The adapter produces samples, not a declaration that a fit is trustworthy. The common
audit screens divergences, maximum-tree-depth saturation, chain mixing, and effective
sample size; users must still inspect prior sensitivity and posterior-predictive mismatch.
Thirty-draw CI smoke tests establish API interoperability only; they are not scientific
sampling defaults or recovery evidence.

The current adapter is limited to a fixed independent scale shared across coefficients,
static subject effects, binary non-omitted choices, and in-sample posterior predictive
draws. It does not yet provide correlated effects, a posterior on variance components,
dynamic trajectories, missing-data models, or prospective new-subject integration.
