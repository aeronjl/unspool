# Smooth change as a competing explanation

`SmoothBernoulliHistoryGLM` is Behavio's first nonstationary reference model. It asks
whether continuously changing coefficients predict later sessions better than one static
coefficient vector. It is a deliberately restrained competitor, not a general claim that
learning is smooth.

## Parameterization

For each base coefficient—intercept, task covariates, and choice-history terms—the model
stores values \(\theta_{jk}\) at fixed temporal knots \(\tau_k\). Coefficients between
knots are linearly interpolated. A time-scaled first-difference penalty is added to the
negative Bernoulli log likelihood:

\[
\frac{\lambda}{2}\sum_j\sum_k
\frac{(\theta_{j,k+1}-\theta_{jk})^2}{\tau_{k+1}-\tau_k}.
\]

This is the MAP penalty induced by a Gaussian random walk over each coefficient path.
Larger `smoothness` means greater precision and therefore less change. Unequal knot spacing
is respected. An optional `l2` penalty regularizes non-intercept coefficient levels.

The path is continuous but not necessarily differentiable at a knot. It is not a full
state-space posterior and does not integrate over trajectory or smoothing-hyperparameter
uncertainty. Those distinctions should remain visible when reporting results.

## The clock is part of the model

```python
from behavio import SmoothBernoulliHistoryGLM

model = SmoothBernoulliHistoryGLM(
    covariates=("stimulus",),
    choice_lags=1,
    time="session_order",
    knots=(0, 2, 4, 6, 8),
    smoothness=10.0,
)
```

Knots are declared in the units of one explicit `Study` column. They must cover every time
at which the model is simulated, fitted, or evaluated. `session_order`, elapsed days,
cumulative trials, and landmark-relative time would define different models; Behavio does
not silently substitute or align them.

The basis and smoothing strength must be specified without looking at held-out outcomes.
When future knots contain no training observations, the random-walk penalty carries the
last supported coefficient level forward. It does not estimate a future trajectory from
future choices. Choosing knots or `smoothness` using test performance would nevertheless
leak, so nested validation or pre-registration is required for comparative claims.

Coefficient paths are subject-specific by default. Passing a multi-subject `Study` raises
an error rather than assuming that equal clock values align individual learning histories.
`shared_trajectory=True` is an explicit opt-in for a scientifically justified common path;
it is not a substitute for the population-plus-subject paths in the
[hierarchical smooth model](hierarchical-smooth-glm.md).

## Simulation and inspection

```python
import numpy as np

truth = model.parameters_from_paths(
    {
        "intercept": np.linspace(-0.3, 0.3, len(model.knots)),
        "stimulus": np.linspace(0.5, 1.8, len(model.knots)),
        "choice_lag_1": np.linspace(0.7, 0.2, len(model.knots)),
    }
)
simulated = model.simulate(design, truth, seed=123)
fit = model.fit(simulated)
trajectory = model.coefficient_trajectory(fit)
```

`parameters_from_paths` avoids hand-constructing knot-qualified parameter names. During
simulation, generated choices recursively update history within each session. Fitted
trajectories can also be evaluated between knots by passing `times=` to
`coefficient_trajectory`.

## Prospective comparison

The smooth and static models satisfy the same simulation, fitting, filtered-prediction,
pointwise-scoring, diagnostics, and recovery contracts:

```python
splits = forward_session_splits(study, min_train_sessions=3)
static_scores = evaluate_splits(static_model, study, splits)
smooth_scores = evaluate_splits(smooth_model, study, splits)
```

A smooth in-sample description is not evidence for a learning trajectory. The relevant
first test is whether a path fitted only to earlier sessions improves pointwise prediction
of complete later sessions. Recovery should then establish which path shapes and change
magnitudes are estimable under the actual design. Static truth must also be included: a
flexible model that always invents drift is not a useful competitor.

The `run_parameter_recovery` API accepts smooth path parameter sets directly and retains
every knot truth, estimate, standard error, seed, convergence flag, and optimizer message.
The complementary [model-recovery API](model-recovery.md) simulates under static and smooth
generators and tabulates which family wins prospective comparison, including ties and
optimization failures.

Run the end-to-end comparison with:

```bash
uv run python examples/smooth_glm.py
```
