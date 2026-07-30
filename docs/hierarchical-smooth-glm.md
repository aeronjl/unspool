# Partially pooled trajectories

`hierarchical(smooth(model))` is Behavio's model of population and individual change. For
coefficient \(c\), subject \(i\), and fixed temporal basis \(B(t)\), it represents

\[
\beta_{ic}(t) = B(t)^{\top}(\theta_c + u_{ic}),
\]

where \(\theta_c\) is the population knot path and \(u_{ic}\) is a subject-deviation
path. Both remain directly inspectable.

```python
import numpy as np

from behavio import BernoulliHistoryGLM
from behavio.compose import hierarchical, smooth

paths = smooth(
    BernoulliHistoryGLM(predictors=("stimulus",), choice_lags=1),
    over="session_order",
    knots=(0.0, 2.0, 4.0),
    smoothness=3.0,
    group_smoothness=3.0,
)
model = hierarchical(paths, over="subject", scale=0.4)
fit = model.fit(study)

population = paths.trajectory_from_knots(fit.estimates)
individual = paths.trajectory_from_knots(np.asarray(list(fit.parameters_for("mouse-1").values())))
```

This used to be a class of its own, `HierarchicalSmoothBernoulliHistoryGLM`, which
subclassed the *base* GLM rather than the smooth one -- so it looked like a fourth sibling
rather than the combination of the second and third. It was the combination all along; see
[composing models](composing-models.md) for the ingredient that makes it one, and for why
the order is `hierarchical(smooth(...))` and never the reverse.

The joint MAP fit uses three declared penalties:

- `smoothness` penalizes first differences in population knot paths;
- `scale` shrinks every subject-deviation knot toward zero;
- `group_smoothness` penalizes first differences within each deviation path, and defaults
  to `smoothness` because a deviation from a smooth path is itself a path.

`l2`, when nonzero, applies to non-intercept population paths. The subject scale is a
penalty scale in this combined prior, not the marginal standard deviation at an individual
knot. All hyperparameters and knots are fixed before fitting.

## Time and prospective prediction

The clock and knot range must cover every fitted or predicted row. Knots may span a known
future session, but they cannot be selected using that session's outcomes. For example,
with knots at sessions 0, 2, and 4, training data through session 3 partly identify the
last linear segment before session 4 is scored.

Prediction is filtered with respect to choice history. A fitted subject uses its estimated
deviation path. A subject absent from training uses the population path, recorded as the
`population-plugin` policy and queryable with `fit.group_was_fitted(subject)`. This is a
point prediction and does not integrate new-subject or trajectory uncertainty.

## Simulation and recovery

`simulate()` returns an observed `Study`. `simulate_with_effects()` additionally returns
the population and subject knot paths outside that study. By default it draws deviations
from the prior the group penalty implies -- correlated across the knots of one coefficient,
because a deviation path is a path. Its optional `group_deviations` argument accepts exact
realized paths for recovery experiments, without adding truth columns to the fitted data.

The [factorial trajectory benchmark](https://github.com/aeronjl/behavio/tree/main/benchmarks/trajectory_recovery) makes
five models compete under stationary identical animals, stable individual differences,
shared drift, and individual drift. The hierarchical smooth model wins only the individual-
drift regime; simpler accounts win the other three under both trajectory recovery and
prospective log loss.

## Current boundary

This composition does not yet estimate its trajectory penalties, integrate posterior
uncertainty, model correlated coefficient deviations, handle irregular subject-specific
knots, or carry latent states. Paths are piecewise linear on a shared declared basis, and
all subjects are assumed comparable on that clock. Fitting currently uses a dense joint
design and Hessian, so it targets moderate cohorts rather than thousands of animals or
knots. These restrictions make alignment an explicit scientific assumption rather than a
hidden consequence of array shape.
