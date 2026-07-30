# Composing models: `smooth()` and `hierarchical()`

Two questions get asked of almost every behavioural model. *Does this coefficient change
over the course of training?* and *does it differ between animals?* Neither is a property
of one model family. Both are transformations you apply to whatever family you already
have.

Behavio used to answer them by writing another class. There was a Bernoulli history GLM, a
smooth one, a hierarchical one, and a hierarchical smooth one; the same four again for
drift diffusion, minus the cells nobody had needed yet. Eleven of the twenty-four
family-by-axis cells existed, they cost four thousand lines between them, and each one
re-implemented `simulate`, `fit`, `predict` and `pointwise_log_prob`. Worse, the two axes
did not compose: `HierarchicalSmoothBernoulliHistoryGLM` extended the *base* GLM, not the
smooth one, so it was a fourth sibling rather than the combination of the second and third.

They are combinators now.

```python
from behavio import BernoulliHistoryGLM
from behavio.compose import hierarchical, smooth

base = BernoulliHistoryGLM(covariates=("stimulus",), choice_lags=1, l2=0.02)

drifting = smooth(base, over="session_order", knots=(0.0, 2.0, 4.0), smoothness=3.0)
pooled = hierarchical(base, over="subject", scale=0.4)
both = hierarchical(drifting, over="subject", scale=0.4)
```

Each returns an ordinary estimator, so `fit_model`, `evaluate_splits`, `compare_models`,
`nested_select_model`, `run_parameter_recovery`, `run_model_recovery` and `describe()` work
on the result without knowing it was composed.

## What each combinator does to the model

`smooth(model, over=..., knots=..., smoothness=...)` replaces every parameter of `model`
with one value per knot, linearly interpolated between knots, and adds a spacing-scaled
first-difference penalty -- the MAP penalty of a Gaussian random walk observed at the
knots. The design matrix becomes the old one multiplied row-wise by the temporal basis;
the penalty becomes the old one lifted onto knots plus that roughness term.

`hierarchical(model, over=..., parameters=..., scale=...)` adds one deviation vector per
group to the parameters named in `parameters`, and fits population and deviations jointly
by maximum a posteriori. The design matrix gains one zero-padded copy of the varying
columns per group; the penalty gains one Gaussian block per group.

## Order matters, and only one order is accepted

**Hierarchy is the outer combinator.** Write `hierarchical(smooth(model))`. The reverse
raises:

```python
smooth(hierarchical(base, over="subject"), over="session_order", knots=(0.0, 4.0))
# TypeError: hierarchy is the outer combinator: write hierarchical(smooth(model)) ...
```

A hierarchical estimator *reports* the population coordinate while *fitting* a joint one
whose width depends on how many groups the study happens to contain, and an outer
combinator cannot expand a coordinate whose width it does not know until it sees the data.
The restriction is real, and it is also the right way round: "which parameters vary by
group" is the last question to ask about a model, not the first.

## A group's deviation inherits the shape of what it deviates from

This is the part that makes `hierarchical(smooth(model))` the correct composition rather
than a plausible-looking wrong one.

A subject's deviation from a smooth population path is itself a path. If the deviation were
given a plain isotropic ridge -- one independent Gaussian per knot -- every subject would
be free to jump between adjacent knots at no cost, and the fitted "individual trajectories"
would be noise wearing a trajectory's clothes. So a smooth model declares, through
[`group_penalty`](#the-contract), that a deviation copy of its parameters carries the
roughness penalty too:

\[
P_{\text{group}} = \operatorname{diag}(\sigma^{-2}) + \lambda_{\text{group}} R,
\]

where \(R\) is the same first-difference precision the population path pays. That is why
`smooth()` takes a `group_smoothness` argument it never uses itself: it is the roughness a
*group's* path is given when hierarchy is applied on top, and it defaults to `smoothness`.

With that in place, `hierarchical(smooth(glm))` reproduces the deleted
`HierarchicalSmoothBernoulliHistoryGLM` bit for bit. The sibling class was the composition
all along; nobody had written down the ingredient that made it one.

## Saying which parameters vary

The deleted hierarchical GLM had a single `subject_scale` shared by every coefficient, so
"the bias differs between animals but the stimulus sensitivity does not" -- the first thing
anybody wants to say -- could not be said. It can now:

```python
pooled = hierarchical(
    base,
    over="subject",
    parameters=("intercept", "choice_lag_1"),
    scale=0.5,
    parameter_scales={"choice_lag_1": 0.15},
)
```

`parameters=None` (the default) lets everything vary, which is the old behaviour.
`over=` is any study column, not only `subject`: `over="lab"` is a lab-level model.

A smooth parameter varies by group as a *whole path*. Naming some of a coefficient's knots
and not others is refused, because a partial path has no roughness prior:

```python
hierarchical(drifting, over="subject", parameters=("intercept[session_order=0]",))
# ValueError: a smooth parameter varies by group as a whole path ...
```

## Parameter names

Naming is mechanical and stable, and it is part of the promise:

| Model | `parameter_names` |
| --- | --- |
| `BernoulliHistoryGLM(covariates=("stimulus",), choice_lags=1)` | `intercept`, `stimulus`, `choice_lag_1` |
| `smooth(base, over="session_order", knots=(0, 4))` | `intercept[session_order=0]`, `intercept[session_order=4]`, `stimulus[session_order=0]`, ... |
| `hierarchical(base, over="subject")` | `intercept`, `stimulus`, `choice_lag_1` |
| `hierarchical(smooth(base, ...), over="subject")` | the smooth names, unchanged |

`smooth()` qualifies each name with its clock and knot, in coefficient-major, knot-minor
order. `hierarchical()` changes no name at all: the coordinate it reports is the
*population* one, so a hierarchical model simulates from the same named parameters as the
model it wraps and a recovery study can compare them directly. Group deviations are not
parameters of that coordinate; they are read off the fit by group label.

## Reading a hierarchical fit

```python
fit = pooled.fit(study)

fit.parameters  # population parameters, by name
fit.groups  # group labels, in first-appearance order
fit.varying_parameters  # which parameters carry deviations
fit.group_deviations  # (groups, varying) deviations from the population
fit.group_parameters  # population + deviation, for the varying parameters
fit.parameters_for("mouse-a")  # one group's full parameter vector
fit.group_was_fitted("new-mouse")  # False -> the population plug-in was used
```

A group absent from training is predicted with the population parameters, recorded as
`unseen_group_policy="population-plugin"`. That is a point plug-in, not integration over a
new group's random-effect distribution.

Reading a smooth fit is unchanged: `coefficient_trajectory(fit, times=...)` evaluates the
fitted paths anywhere in the knot range. For a composed model, the inner smooth model is
reachable as `model.model`, and `trajectory_from_knots` turns any knot vector -- the
population estimate, or one group's `parameters_for(...)` -- into a trajectory.

## Estimating the scale

```python
pooled = hierarchical(
    base, over="subject", scale=0.4, estimate_scale=True, scale_bounds=(0.05, 1.5)
)
fit = pooled.fit(study)
fit.scales  # effective per-parameter scales
fit.scale_standard_error
fit.scale_confidence_interval_95
fit.scale_at_boundary
```

The declared scales become the initial value for a bounded Laplace marginal-likelihood
estimate of one common multiplier on them. This is empirical Bayes, not a posterior: the
deviations are integrated out approximately and the uncertainty is a local Hessian. A scale
resting on either bound sets `scale_at_boundary` and the common boundary diagnostic, and
means the answer is unresolved beyond that bound rather than equal to it.

## From a formula

A formula's group term is honoured by `model_from_formula`:

```python
from behavio import BernoulliHistoryGLM, model_from_formula

model = model_from_formula(
    "choice ~ stimulus + lag(choice, 1) + (1 + stimulus | subject)",
    BernoulliHistoryGLM(),
    scale=0.4,
)
```

The fixed terms become the model's design; the group term becomes
`hierarchical(model, over="subject", parameters=("intercept", "stimulus"))`. A
`DesignSpec` is still one fixed matrix with nowhere to put a varying effect, so
`Formula.to_design()` on a formula with a group term routes you here rather than silently
dropping the declaration. See [design formulas](design-formulas.md).

## The contract

A model is composable if it satisfies
`behavio.contracts.compose.PenalisedLinearEstimator`: its likelihood must see a study only
through a **quadratically penalised linear predictor**. Every generalized linear family in
Behavio is one of those. The drift-diffusion families are not, and neither is the GLM-HMM,
whose likelihood is a mixture over latent states; both are refused rather than silently
mis-composed.

Five things a combinator needs, and the members that carry each:

| Ingredient | Members |
| --- | --- |
| A penalised objective to add to | `design_matrix`, `penalty_matrix`, `outcomes`, `likelihood`, `fit_penalised` |
| Which parameters may vary, and over what | `parameter_names`, plus the `VaryingEffects` declaration |
| A per-group block structure | `group_penalty` on the model, `group_blocks` on the study |
| A way to expand and contract the parameter vector | `expand_group_design`, `expand_group_penalty`, `simulate_rows` |
| A route for the simulator to draw group effects | `draw_group_deviations` |

`fit_penalised` is the model's own solver rather than a re-implementation, which is what
makes a composed fit bit-for-bit the fit the model would have run itself on the expanded
problem. `simulate_rows` takes one coefficient vector *per row*, which is the single
representation both axes collapse to: hierarchy varies coefficients by group, smoothness
varies them by clock value, and the recursion over generated history is the same in both.

`LinearPredictorLikelihood` is separate from the estimator so that a combinator can write a
*new* objective -- the Laplace profile over a group scale, for instance -- without knowing
which family it is profiling.

To make a new family composable, implement those members. See
[extensions](extensions.md).
