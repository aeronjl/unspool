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

The same two verbs apply to `MultinomialLogit`, so a task with three or more actions gets
drifting, per-subject and per-subject-drifting choice models without a line of new
modelling code:

```python
from behavio import ChoiceSpec, DesignSpec, MultinomialLogit, NumericTerm

actions = MultinomialLogit(
    choice=ChoiceSpec(options=("left", "right", "up"), available_options_column="available"),
    design=DesignSpec((NumericTerm("stimulus"),)),
    l2=0.05,
)

drifting_actions = smooth(actions, over="session_order", knots=(0.0, 4.0))
pooled_actions = hierarchical(actions, over="subject", scale=0.4)
both_actions = hierarchical(drifting_actions, over="subject", scale=0.4)
```

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
| `MultinomialLogit(...)` | `category['right']::intercept`, `category['right']::stimulus`, `category['up']::intercept`, ... |
| `smooth(actions, over="session_order", knots=(0, 4))` | `category['right']::intercept[session_order=0]`, ... |
| `hierarchical(actions, over="subject")` | the multinomial names, unchanged |

`smooth()` qualifies each name with its clock and knot, in coefficient-major, knot-minor
order. `hierarchical()` changes no name at all: the coordinate it reports is the
*population* one, so a hierarchical model simulates from the same named parameters as the
model it wraps and a recovery study can compare them directly. Group deviations are not
parameters of that coordinate; they are read off the fit by group label.

A multinomial coefficient is already per-category, and the qualifiers compose with that
rather than replacing it, because per-category structure lives in the *name* and only the
*predictor* is per-category. Naming one category's coefficients in a `hierarchical()` call
does not need `repr` quoting written into a string literal:

```python
per_category = hierarchical(
    actions, over="subject", parameters=actions.category_parameter_names("up")
)
per_category.varying_parameters
# ("category['up']::intercept", "category['up']::stimulus")
```

## Models with more than one number per row

A Bernoulli GLM predicts one number per trial. A multinomial logit predicts one per
category, and it is no less a penalised linear model for it: the coefficients still enter
linearly and the penalty is still quadratic. What is wider is the *shape* of the linear
predictor, which `predictor_cells` declares.

| Model | `predictor_cells` | `design_matrix(study).shape` |
| --- | --- | --- |
| `BernoulliHistoryGLM(...)` | `()` | `(rows, parameters)` |
| `MultinomialLogit(options=("left", "right", "up"))` | `("category['left']", "category['right']", "category['up']")` | `(rows, 3, parameters)` |
| `WienerDriftDiffusion(...)` | `("drift", "boundary", "starting_bias", "nondecision_time")` | `(rows, 4, parameters)` |

`()` is the scalar case, and every array a scalar-predictor model exchanges with a
combinator has exactly the shape it always had -- which is why the fits the deleted
hand-written GLM classes published are still reproduced bit for bit. A cell axis re-orders
the same products rather than changing them, so the deleted drift-diffusion classes are
reproduced to the optimizer's tolerance instead; see
[migration guides](migration-guides.md). Nothing about hierarchy or
smoothness is per-family, so nothing about them turned out to be per-shape either:
grouping partitions *rows*, and a cell axis sits between the row axis and the coordinate
axis it never touches, so `expand_group_design` copies a group's block across every cell of
a row and `expand_group_penalty` -- which lives entirely on the coordinate -- did not change
at all.

### Availability is a modelling statement, not a broadcast

A task with per-trial choice sets offers some categories on some trials. That reaches the
likelihood through `predictor_offsets`, an additive term on the linear predictor that no
parameter multiplies, with `-inf` marking a cell outside the support of a row. A combinator
carries offsets through untouched, and the consequences follow from the arithmetic rather
than from a special case:

- A trial that did not offer a category contributes zero probability, zero gradient and
  zero curvature to that category's coefficients, at population level and at group level
  alike.
- A subject who was **never** offered a category has no likelihood curvature at all on that
  category's deviation. The joint fit leaves that deviation at exactly zero and reports its
  standard error as exactly the prior standard deviation -- the honest answer for a
  quantity the data cannot speak to. Nothing is imputed and nothing is quietly pooled.
- The *population* coefficients for that category stay identified by the subjects who were
  offered it.

Omissions behave the same way in reverse. With `include_omission=True` the omission
category is always available, because it represents failure to emit any action rather than
an offered action, so a lapse rate can be let vary by subject or follow a path in session
time like any other coefficient.

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
through a **quadratically penalised linear predictor**, and its row scores must be
independent given that predictor. Every generalized linear family in Behavio is one of
those, `MultinomialLogit` included, and so is `WienerDriftDiffusion`: there is no link on
its four predictor cells, but each of them is still a design times a coefficient block, and
a trial's joint choice/latency density depends on the study through nothing else. The
GLM-HMM is not, and is refused rather than silently mis-composed.

Five things a combinator needs, and the members that carry each:

| Ingredient | Members |
| --- | --- |
| A penalised objective to add to | `design_matrix`, `predictor_offsets`, `penalty_matrix`, `outcomes`, `likelihood`, `fit_penalised` |
| How wide one row's prediction is | `predictor_cells` |
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
which family it is profiling. Three shape-aware contractions in
`behavio.contracts.compose` -- `linear_predictor`, `parameter_gradient` and
`information_matrix` -- are the only places that know whether a row predicts one number or
several, so a family author writes neither branch.

There is no `boundary_threshold` on the contract. A hierarchical fit has to know whether
*population plus deviation* is at a boundary, and that number is not a coordinate of the
vector the optimizer returns; rather than making the combinator ask for a threshold, the
combinator supplies the quantity as `PenalisedDesign.derived_estimates`, a function of the
solution, and the model's own solver applies its own convention to it. Every number in a
fit's diagnostics is produced by the model that owns the convention behind it.

### Declining the contract

Structural typing answers "does this object have these members", which is not the question
being asked. `BernoulliGLMHMM` subclasses the Bernoulli GLM for its per-state emissions, so
it *inherits* every member above and would satisfy any widening of them -- a
`(rows, states)` linear predictor is exactly what its emissions produce. What it cannot
honour is the half of the contract that is not a shape: a penalised linear model's log
likelihood is a sum of independent row scores, and a GLM-HMM's is a forward recursion in
which each row depends on every row before it, so profiling out a group deviation one block
at a time would optimise something that is not this model's likelihood.

No arrangement of members can be inspected to discover that, so a model in that position
declares it in a sentence:

```python
BernoulliGLMHMM(covariates=("stimulus",)).penalised_linear_refusal
# 'a GLM-HMM is a latent-state mixture, not a penalised linear model: ...'
```

`require_penalised_linear` -- the single check both combinators run -- reads the declaration
before it runs the structural test that would say yes, and reports the sentence in the
`TypeError`.

To make a new family composable, implement those members. See
[extensions](extensions.md).
