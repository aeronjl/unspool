# Composing models: `smooth()`, `hierarchical()` and `mix()`

Three questions get asked of almost every behavioural model. *Does this coefficient change
over the course of training?*, *does it differ between animals?* and *did the animal do the
task on every trial?* None is a property of one model family. All three are transformations
you apply to whatever family you already have.

Behavio used to answer them by writing another class. There was a Bernoulli history GLM, a
smooth one, a hierarchical one, and a hierarchical smooth one; the same four again for
drift diffusion, minus the cells nobody had needed yet. Eleven of the twenty-four
family-by-axis cells existed, they cost four thousand lines between them, and each one
re-implemented `simulate`, `fit`, `predict` and `pointwise_log_prob`. Worse, the two axes
did not compose: `HierarchicalSmoothBernoulliHistoryGLM` extended the *base* GLM, not the
smooth one, so it was a fourth sibling rather than the combination of the second and third.

The third question had it worse. Three unrelated mechanisms existed for one idea: a
`contaminant=` slot on the drift-diffusion model, a whole `LapsePsychometric` class whose
only job was adding a symmetric lapse to a logistic curve, and a bounded lapse inside a
reinforcement-learning policy. None composed with the others, and a lapse on a GLM, on a
multinomial or on a GLM-HMM could not be written at all.

They are combinators now.

```python
from behavio import BernoulliHistoryGLM
from behavio.compose import UniformChoiceGuess, hierarchical, mix, smooth

base = BernoulliHistoryGLM(predictors=("stimulus",), choice_lags=1, l2=0.02)

drifting = smooth(base, over="session_order", knots=(0.0, 2.0, 4.0), smoothness=3.0)
pooled = hierarchical(base, over="subject", scale=0.4)
lapsing = mix(base, UniformChoiceGuess(), weight_bounds=(0.0, 0.2))
everything = hierarchical(smooth(lapsing, over="session_order", knots=(0.0, 4.0)))
```

Each returns an ordinary estimator, so `fit_model`, `evaluate_splits`, `compare_models`,
`nested_select_model`, `run_parameter_recovery`, `run_model_recovery` and `describe()` work
on the result without knowing it was composed.

The same two verbs apply to `MultinomialLogit`, so a task with three or more actions gets
drifting, per-subject and per-subject-drifting choice models without a line of new
modelling code:

```python
from behavio import ChoiceSpec, MultinomialLogit
from behavio.design import DesignSpec, NumericTerm

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

`mix(model, component, weight_bounds=...)` replaces each row's density with a weighted
average of the model's and a declared simpler process's, and adds exactly one parameter --
the weight. For a model composed through a linear predictor the design gains one intercept
column and the predictor gains cells the component fills; for a model composed through a row
objective the row coordinate gains one column and the component's log density is held beside
the objective. Which of the two happens is decided by the model's contract and by nothing
else.

## Order matters, and only one order is accepted

**Hierarchy outermost, mixture innermost.** Write `hierarchical(smooth(mix(model)))`, and
every prefix of it is a model. Both reverses raise:

```python
smooth(hierarchical(base, over="subject"), over="session_order", knots=(0.0, 4.0))
# TypeError: hierarchy is the outer combinator: write hierarchical(smooth(model)) ...

mix(smooth(base, over="session_order", knots=(0.0, 4.0)), UniformChoiceGuess())
# TypeError: mix is the innermost combinator: write smooth(mix(model)) ...
```

The two refusals have different reasons and that is worth knowing.

A hierarchical estimator *reports* the population coordinate while *fitting* a joint one
whose width depends on how many groups the study happens to contain, and an outer
combinator cannot expand a coordinate whose width it does not know until it sees the data.
That restriction is arithmetic, and it is also the right way round: "which parameters vary
by group" is the last question to ask about a model, not the first.

Nothing stops `mix(smooth(model))` arithmetically. It is refused because it is a *second
spelling*: `mix(smooth(model))` is a smooth model with a stationary weight, and
`smooth(mix(model), parameters=<the model's own names>)` is exactly that model, while
`smooth(mix(model))` with the weight named is the one `mix(smooth(model))` could not
express. One order reaches both models; two orders reach the same models twice. Keeping
each combinator wrapping only things more primitive than itself is also what keeps the
number of pass-through members linear in the number of combinators rather than quadratic.

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

## `mix()`: a simpler process alongside the model

A lapse is a mixture with a guessing process. A contaminant is a mixture with a
distribution over the scored outcome. They are one concept with different components, so
`mix()` takes a **component** and there is no vocabulary of processes to learn:

\[
p(y_r) = (1-\omega)\,p_{\text{model}}(y_r) + \omega\,p_{\text{component}}(y_r).
\]

Four components ship, one per shape of observation:

| Component | Scores | Guesses |
| --- | --- | --- |
| `UniformChoiceGuess(outcome=..., probability=0.5)` | one binary column | a coin |
| `UniformCategoryGuess(choice=ChoiceSpec(...))` | a category code | uniformly over the options the trial offered |
| `UniformResponseGuess(time_bounds=..., ...)` | a joint choice and latency | an independent coin and a uniform latency |
| `UniformDurationGuess(duration_bounds=..., outcome=...)` | one bare duration | uniformly over a declared interval |

```python
from behavio import (
    ChoiceSpec,
    MultinomialLogit,
    UniformCategoryGuess,
    UniformChoiceGuess,
    UniformResponseGuess,
    WienerDriftDiffusion,
    mix,
)
from behavio.compose import UniformDurationGuess
from behavio.models.scalar_timing import DurationReproduction

lapsing_glm = mix(base, UniformChoiceGuess(), weight_bounds=(0.0, 0.2))
lapsing_actions = mix(actions, UniformCategoryGuess(choice=actions.choice))
contaminated_ddm = mix(
    WienerDriftDiffusion(predictors=("stimulus",), nondecision_time_bounds=(0.1, 0.6)),
    UniformResponseGuess(time_bounds=(0.05, 3.0)),
    weight_bounds=(0.0, 0.25),
)
contaminated_timing = mix(
    DurationReproduction(),
    UniformDurationGuess(duration_bounds=(0.05, 8.0), outcome="reproduced_duration"),
    weight_bounds=(0.0, 0.25),
)
```

The first two need no interval declared because a coin and a set of options are finite. A
duration is not, so `UniformDurationGuess` takes its bounds and puts them in the model's
signature; [what that costs](#a-continuous-outcome-needed-a-component-not-a-combinator) is
below.

### The weight is estimated; its range is declared

`weight_bounds` is the general form of the `maximum_lapse` and `probability_bounds`
arguments the three deleted mechanisms each carried in their own dialect. It says how large
the mixture is *allowed* to be, and nothing about where in that range it sits.

The estimated coordinate is a single unbounded number, `mixture_logit`, and the reported
weight is the component's own name for it -- `lapse_rate` for a guessing process,
`contaminant_rate` for a distribution over response times:

```python
fit = lapsing_glm.fit(study)

lapsing_glm.weight(fit)  # the natural rate
fit.derived_value("lapse_rate")  # the same number, with a standard error
lapsing_glm.to_natural(fit.estimates)  # every parameter on its reported scale
lapsing_glm.component_responsibility(study, fit)  # posterior P(component) per trial
```

A logit rather than the weight itself, for three reasons that are one reason. A penalised
linear fit is unconstrained and a Bernoulli GLM's solver has no box to put a bound in, so a
natural-scale weight would have forced a box onto every coefficient of every model that
could be mixed. A group deviation on a rate near zero is not Gaussian and a deviation on its
log odds is, so `hierarchical(mix(model))` gets the right prior for free. And a Wald interval
formed on a logit and mapped back cannot leave `[0, 1]`.

The price is that a saturated weight is a large coordinate rather than a coordinate at a
bound, so `|mixture_logit| >= 12` sets the usual `boundary_estimate` diagnostic. A weight
resting there means *the data are consistent with no second process*, not *the weight is
zero*.

### A responsibility is not a label

`component_responsibility(study, fit)` returns the posterior probability that each trial
came from the component. A fast response is evidence for a contaminant; it is not a
contaminant. Nothing in Behavio turns responsibilities into an exclusion rule.

### Asymmetry belongs to the link, not to the mixture

A component has **no estimated parameters**: only the weight is estimated. That line is
where the psychophysical two-gamma form falls on the far side. Writing

\[
\gamma + (1-\gamma-\lambda)F(z)
 = (\gamma+\lambda)\frac{\gamma}{\gamma+\lambda} + \bigl(1-(\gamma+\lambda)\bigr)F(z)
\]

shows it *is* a mixture, with weight \(\gamma+\lambda\) and a Bernoulli guess of
probability \(\gamma/(\gamma+\lambda)\). But two free rates are two estimated numbers,
and a component with an estimated parameter is a second model hiding inside the first: it
would need its own coordinate, its own box, its own group prior and its own place in every
combinator. A **declared** asymmetry costs nothing and is available today --
`UniformChoiceGuess(probability=0.6)`. An **estimated** one is a shape of the curve, and
[`PsychometricFunction`](psychometric-functions.md) already estimates both rates inside the
link, which is where a shape of a curve belongs.

### Identifiability is reported before the fit, not after it

A lapse rate and a shallow slope trade off: the weight is estimated from how flat the
asymptotes are relative to how steep the middle is. At the limit -- a model whose prediction
is the *same on every row* -- the trade-off is exact rather than merely awkward, and any
weight can be traded against the model's own parameters without changing the fit at all.
`describe(study)` says so:

```python
mix(BiasOnly(), UniformChoiceGuess()).describe(study).findings
# [warning] unidentified_mixture: lapse_rate is not identified by this design: the model
# predicts the same thing on every row, so any weight can be traded against the model's
# own parameters without changing the fit
```

The mirror image is reported too: a component that gives zero density to every observation
cannot have produced any of them, so its weight can only rest on the floor of its range.
That is `unreachable_mixture_component`, and it is what a contaminant window declared in
the wrong unit looks like.

Neither is an error, because both are modelling decisions only their author can judge --
the same standing as a knot placed outside the data's support.

**On the row-objective route, one of the two carries over exactly and one does not.**
`unreachable_mixture_component` is a statement about the component and the observed
outcomes; neither of those knows which contract the model satisfies, so it is the same
statement computed by the same code.

`unidentified_mixture` carries over in meaning but not in proof. On the penalised route it is
read off the design matrix, so "the model predicts the same thing on every row" holds at
*every* coordinate and the finding is exact. There is no design matrix on the row route, and
the prediction is a nonlinear function of the coordinate, so what is checked instead is the
prediction at each of the deterministic restarts the model's own solver would use. A design
that does not distinguish two rows is still caught -- a constant design gives a constant
prediction wherever it is evaluated -- but a varying design that happens to look flat at all
of those coordinates would be reported without being degenerate. The message says where it
looked, so the difference is legible in the report rather than only here.

The wrapped model's own findings are forwarded on both routes: a discounting design that
cannot identify a discount rate says so whether or not the model has been mixed with a
lapse.

## What a component must expose

`behavio.contracts.mixture.MixtureComponent`, which sits beside `PenalisedLinearEstimator`
for the same reason: it names what a combinator needs that structural typing cannot check.

| Ingredient | Member |
| --- | --- |
| A density on the model's own outcome | `pointwise_log_density(study, outcomes)` |
| A prediction of the model's own shape | `prediction_probability(study)`, `prediction_width` |
| A simulator, so the weight can be recovered | `simulate_outcomes(study, rows, generator=...)` |
| An identity, so a fit records what it was mixed with | `component_name`, `signature`, `weight_name` |
| A refusal, in a sentence | `mixture_refusal(model)` |

The refusal is a method rather than an inspection because structural typing answers "does
this object have these members", which is not the question: a uniform guess over three
categories and a multinomial over four have identical member sets and cannot be mixed.

A component must be able to score the model's observation, so a mixable model exposes
`outcomes(study)` -- the array in its own outcome coordinate. `PenalisedLinearEstimator`
declares that member already; a bounded-coordinate family that wants to be mixed declares
the same member under the same name.

### Where the weight rides

For a model composed through a linear predictor, everything the mixed likelihood needs
arrives through that predictor. The wrapped model's cells come first, then the mixture logit
-- the only new cell a parameter multiplies -- then the component's log density and its
predicted probability, carried as *offsets*: terms no parameter multiplies. That is the
channel per-trial option availability already travels down, widened by one observation, and
it is why a mixture survives being sliced by group or multiplied by a temporal basis without
either outer combinator knowing what it is carrying.

For a model composed through a `RowObjective` there is no predictor to add a cell to. What
that predictor was standing in for is still there, though: **one coordinate vector per row**,
which is exactly the representation both hierarchy and smoothness collapse to. So the
mixture logit becomes one extra *column* of the row coordinate, and the property is the same
one -- an outer combinator rewrites the coordinate and the weight follows it.

```python
from behavio import TemporalDiscounting  # a family with no design matrix at all

impulsive = mix(TemporalDiscounting(), UniformChoiceGuess(), weight_bounds=(0.0, 0.3))

per_animal_lapse = hierarchical(impulsive, over="subject", parameters=("mixture_logit",))
drifting_lapse = smooth(
    impulsive, over="session_order", knots=(0.0, 5.0), parameters=("mixture_logit",)
)
```

The component's log density does not need an offset on this route because it does not need a
channel at all: it is a function of the study and of the observed outcome and of no
parameter, so the row objective computes it once and holds it. An offset was always the
*representation* of "a per-row term no parameter multiplies", not the thing itself.

The gradient stays analytic on both routes and is the same two terms: the wrapped model's own
gradient scaled by the posterior probability that the row came from the model -- the
responsibility an EM step would compute, here simply the derivative -- and the weight's
derivative through its logit. Neither is a finite difference; a mixture of two differentiable
densities has no reason to become one.

One thing differs and it is contractual. On the penalised route the reported coordinate is
the wrapped model's `parameter_names` verbatim, because that contract declares its
coordinate to be "the one the model is estimated, reported and simulated in". On the row
route the reported coordinate is the wrapped model's `natural_names`, because the
bounded-coordinate contract declares the opposite -- `discount_rate_log` is estimated and
`discount_rate` is reported, and appending a weight to the first would report a logarithm
under the name of the thing it is the logarithm of.

## Models that decline a mixture

`mix()` runs `require_mixable`, which asks one question: are this model's rows independent?
That is the condition a mixture actually needs -- an average of two densities of *one row's*
outcome -- and it is not the same question as "does this model have a linear predictor". A
model whose likelihood recurses declares so in a sentence, `independent_rows_refusal`, and
that sentence is what the `TypeError` reports:

```python
mix(BinaryRLAgent(), UniformChoiceGuess())
# TypeError: mix() cannot be applied to BinaryRLAgent: a value-updating agent is a
# recursion over trials, so there is no per-row density for a mixture to average ...
```

`SoftmaxPolicy(maximum_lapse=...)` therefore stays where it is, and that is a statement
about the model rather than a leftover. A trial's choice probability is a function of a value
trace every earlier trial in the session wrote to, so there is no per-row density to average.
The policy lapse is also in the right place scientifically -- it mixes the *action* the agent
emits while leaving the value update to see the action that was actually taken, which is what
makes the learned trace on a lapse trial the trace the animal's own choice produced. A
mixture applied from outside the recursion could not express that, because from outside there
is no recursion to reach into.

The declaration is eager, so that `mix()` fails at the call rather than at the fit. The exact
form of the same question is `RowObjective.row_blocks`, which only a study can answer, and
`require_independent_rows` asks it as soon as one exists: a model that declared nothing and
turns out to score its rows in blocks is refused there instead.

The same agent is nonetheless smoothed and pooled, because neither of those needs a row to
have its own density -- only a coordinate constant within each block. That is the next
section.

## Models whose coordinate is bounded, not linear

`BinaryQLearning`, `BinaryRLAgent`, `PsychometricFunction`, `TemporalDiscounting`,
`ProspectTheory`, `DurationReproduction`, `TemporalBisection` and `PatchLeaving` are the
families whose likelihood is not a penalised linear one and whose parameters are bounded: a
learning rate in \((0,1)\), an inverse temperature above zero, a width above zero, a lapse
rate below its declared maximum, a discount rate above zero, a Weber fraction above zero, a
giving-up intake rate above zero. All sixteen of their `smooth()` and `hierarchical()` cells
work. The last five were written *after* this contract existed and needed nothing added to
it; see [economic and value-based choice](economic-choice.md), where `mix()` also works,
because a mixture is gated on row independence and a value-based trial's rows are independent.
The two continuous families reach it as well, through
[a component rather than a combinator](#a-continuous-outcome-needed-a-component-not-a-combinator).
The two agents are the families where it does not, and there the refusal is the recursion.
`BernoulliGLMHMM` is a fourth model on this contract, and a partial one: its
[hierarchical cell is open](#a-glm-hmm-which-cell-opened-and-what-still-refuses) on the
emission coefficients, and its smooth cell is refused.

The three most recent families are the first on this contract whose **observation is not a
choice**: `behavio.models.scalar_timing.DurationReproduction` scores a reproduced duration
and `behavio.models.patch_leaving.PatchLeaving` scores a residence time that may be
right-censored. Neither combinator noticed. `smooth()` and `hierarchical()` never look at an
observation — they rewrite a coordinate and hand the model back its own row objective — so a
per-subject Weber fraction and a patch-leaving threshold that drifts across training are
available for no combinator code at all:

```python
from behavio.compose import hierarchical, smooth
from behavio.models.patch_leaving import PatchLeaving
from behavio.models.scalar_timing import DurationReproduction, TemporalBisection

per_animal_weber = hierarchical(
    DurationReproduction(), over="subject", parameters=("weber_fraction_log",), scale=0.5
)
drifting_threshold = smooth(
    PatchLeaving(),
    over="session_order",
    knots=(0.0, 5.0),
    parameters=("giving_up_rate_log",),
)
sharpening_clock = smooth(
    TemporalBisection(), over="session_order", knots=(0.0, 5.0), parameters=("clock_rate_log",)
)
```

### A continuous outcome needed a component, not a combinator

`mix()` used to be refused on the two continuous families, and the refusal was never a limit
of `mix()`: their rows are independent, the weight rides in one extra column of the row
coordinate exactly as it does for a discounting model, and `require_mixable` never reached
the arithmetic because it compares scored columns first. What was missing was a **component**
that scores a bare duration. `UniformDurationGuess` is it, and neither combinator changed:

```python
from behavio.compose import UniformDurationGuess, mix
from behavio.models.patch_leaving import PatchLeaving
from behavio.models.scalar_timing import DurationReproduction

contaminated_reproduction = mix(
    DurationReproduction(),
    UniformDurationGuess(duration_bounds=(0.05, 8.0), outcome="reproduced_duration"),
    weight_bounds=(0.0, 0.25),
)
contaminated_foraging = mix(
    PatchLeaving(censoring_time_column="observation_limit"),
    UniformDurationGuess(
        duration_bounds=(0.0, 3.0),
        outcome="residence_time",
        censoring_time_column="observation_limit",
    ),
)
```

Three things the component has to say that a binary guess did not.

**The interval is declared, in the outcome column's own units.** Uniform over two options is
one half and there is nothing to argue about; uniform over a duration does not exist until an
interval is named. `UniformResponseGuess` met the same question and answered it with
`time_bounds` in canonical seconds — it can, because a drift-diffusion model declares a
`ResponseTimeSpec` and hands components a latency in seconds whatever the column holds. A
scalar-timing or patch-leaving model declares no unit at all and returns its column verbatim,
so `duration_bounds` is in *that column's* units and appears in the composed model's
signature. Reading the interval off the data instead would make the component's normalising
constant a function of the sample, and the widest observations — the ones a contaminant
exists to explain — would be the ones setting the density that explains them.

**A censored row is scored by a survival probability, not by a density.** `PatchLeaving`
scores a visit that was still in progress by \(\log S(c)\). A mixture averages what *each*
process says about the observation that was actually made, so the component contributes the
probability that its own duration exceeds the same \(c\):

\[
S_{\text{mix}}(c) = (1-\omega)\,S_{\text{model}}(c) + \omega\,S_{\text{comp}}(c).
\]

This needed nothing added to the component contract — a component is handed the study
alongside the outcome, which is the channel per-trial option availability already travels
down — but it does mean both processes must read the **same** limit column, and
`mixture_refusal` says so rather than letting them disagree silently. Contributing the
density instead is not a rounding error: a density is one over time and a survival
probability is dimensionless, so every censored row looks like a row the contaminant could
not have produced. On a study with three rows in five censored, a censoring-blind component
recovers less than half the weight the study was simulated with, while the model's own
parameters barely move — which is what makes the mistake hard to see in a fit table.
`tests/test_compose_duration_mixture.py` measures it.

**The mixed prediction is a density.** `predict()` on either family returns a
`DensityPrediction`, so a mixture's prediction is the weighted average of two densities at
every point of the model's own grid rather than of two probabilities. The component's half is
obtained by asking `pointwise_log_density` about each grid point, which is a question it
could already answer, so this needed no new member either — only a second averaging function,
`blended_density`, beside the one that averages probabilities.

```python
from behavio import BinaryQLearning, PsychometricFunction
from behavio.compose import hierarchical, smooth

agent = BinaryQLearning()
curve = PsychometricFunction()

per_animal = hierarchical(agent, over="subject", parameters=("choice_bias",), scale=0.5)
warming_up = smooth(
    agent,
    over="session_order",
    knots=(0.0, 5.0),
    parameters=("inverse_temperature_log",),
)
per_animal_curve = hierarchical(
    curve, over="subject", parameters=("threshold", "log_width"), scale=0.4
)
sharpening = smooth(curve, over="session_order", knots=(0.0, 5.0), parameters=("log_width",))
```

### One combinator, two contracts, a smaller core

`smooth()` and `hierarchical()` are still one function each. What is new is a *sibling*
contract they route through, `behavio.contracts.bounded.BoundedCoordinateEstimator`, chosen
by one check -- `require_composable` -- that both combinators run in place of
`require_penalised_linear`.

The sibling exists because the two contracts turned out to share almost everything. Eight
members are identical -- `parameter_names`, `penalty_matrix`, `coordinate_box`,
`initial_points`, `group_parameter_expansion`, `group_penalty`, `draw_group_deviations`,
`simulate_rows` -- plus the model's own solver. Selecting the varying columns, widening the
penalty and the box, naming the joint coordinate, slicing the estimate back into a
population and one deviation per group: none of that ever looks at a likelihood, so none of
it is duplicated.

What differs is exactly one member, and the package had already named the shape of it.
`simulate_rows` takes **one coefficient vector per row** and its docstring calls that "the
single representation that both hierarchy and smoothness collapse to". A bounded-coordinate
model supplies the *likelihood* counterpart:

```python
objective = agent.row_objective(study)  # a RowObjective
value, gradient = objective.value_and_gradient(rows)  # rows is (trials, parameters)
```

A combinator's contribution is then a **linear map** from the joint coordinate it invented
to that `(rows, parameters)` array -- population plus this row's group deviation, or a knot
vector evaluated at this row's clock value -- so the chain rule is one contraction over
rows. Widening the contract instead would have meant giving a Q-learning agent a
`design_matrix` it does not have; forking the combinators would have meant two
`hierarchical`s. Neither was necessary once the shared core was written down.

### `ParameterSpace` was most of the answer, and what it was missing

`ParameterSpace` already describes the transformed coordinate exactly: per-parameter
transforms (`bounded-logit`, `log`, `identity`), natural bounds, optimizer bounds and
priors, with `OptimizationProblem` switching to MAP when priors are present and an explicit
`PriorMeasure` for the transform Jacobian. `BinaryQLearning` carries one.

Three things it does not carry, all of which hierarchy needs:

- **A prior that is not per-parameter.** `PriorSpec` is a scalar family on one natural
  coordinate. A group deviation is a prior on the *difference* between two coordinates of a
  joint vector, and the number of those coordinates is not known until a study is seen.
- **A study.** Every `ParameterSpace` member is a pure function of a vector. A group
  structure and a clock are properties of the data, and `coordinate_box(study)` is where a
  data-derived bound (a psychometric threshold cannot be far outside the tested range) is
  answered.
- **A likelihood.** A parameter space says what a coordinate *means*, never how to score it.

So the bridge is `BoundedCoordinateEstimator`, and where a model has a `ParameterSpace` the
bridge is thin: `coordinate_box` is `parameter_space.optimizer_bounds` and nothing else, and
`parameter_names` is `parameter_space.optimizer_names`, which is already the unconstrained
coordinate. Two of the three families are not expressible as a `ParameterSpace` at all --
`BinaryRLAgent` assembles its coordinate from swappable components and `PsychometricFunction`
has a link-dependent location name -- so the bridge is a protocol rather than a requirement
to own one.

### Deviations are Gaussian on the transformed scale, and that is checked

A learning rate of 0.1 with a Gaussian deviation of standard deviation 1.5 is a negative
learning rate about a quarter of the time. The same deviation on its **logit** is a rate in
\((0,1)\) every time:

\[
\alpha_g = \operatorname{logit}^{-1}\!\left(\operatorname{logit}(\alpha) + b_g\right),
\qquad b_g \sim \mathcal{N}(0, \sigma^2).
\]

Nothing had to be added for this: `parameter_names` for all three families *is* the
transformed coordinate -- `learning_rate_logit`, `inverse_temperature_log`, `log_width`,
`lapse_logit` -- and the natural coordinate is reached through `NaturalParameterisation` or
`parameter_components`, exactly as recovery already reads it. `require_composable` refuses a
bounded-coordinate model that does not declare a finite `coordinate_box`, because a finite
box on the transformed coordinate is the statement that a transform was applied.

Reading a group's parameters back therefore goes through the model, not through arithmetic
on the fit:

```python
fit = per_animal.fit(study)
vector = fit.parameters_for("mouse-a")
agent.parameter_components(vector).learning_rate  # the natural rate for that animal
```

### Smooth reinforcement learning: which parameter drifts, and how often

`parameters=` says *which* parameters follow a path, and for an RL agent that is a real
scientific choice rather than a formality:

```python
smooth(agent, over="session_order", knots=(0, 5), parameters=("inverse_temperature_log",))
# the policy sharpens across training; the learning rate does not change

smooth(agent, over="session_order", knots=(0, 5), parameters=("learning_rate_logit",))
# the animal updates more or less aggressively; its policy noise does not change
```

The selector is sufficient for *which*. It is **not** sufficient on its own, because a
recursion makes the *clock* part of the model in a way it is not for a GLM. A GLM's
coefficient may drift trial by trial; a learning rate may not, because the value trace that
a trial-varying \(\alpha\) writes cannot say which of its values produced which part of the
trace. So the convention is declared and enforced:

> **The parameters in force on trial \(r\) are the paths evaluated at trial \(r\)'s clock
> value, and the clock must be constant within each block the model's likelihood recurses
> over** -- for these agents, one subject's session.

`session_order` satisfies that. A within-session trial counter does not, and is refused:

```python
smooth(agent, over="trial", knots=(0.0, 39.0), parameters=("learning_rate_logit",)).fit(study)
# ModelDataError: ... a path over 'trial' is only defined if the clock is constant within
# each block that recursion runs over; smooth over a session-level clock instead
```

The same rule applies to `hierarchical()`: `over="subject"` is admissible because a session
lies inside a subject, and a grouping column that cuts through a session is refused rather
than averaged over.

### A rate at its bound is reported, not shrunk

A Gaussian deviation on a logit is the right prior for a rate the data locate in the
*interior* of its range. It is not a repair for a rate the data push to a bound: as a lapse
rate goes to zero its logit goes to \(-\infty\), the group's deviation becomes unbounded, the
conditional mode runs to the edge of the box, and the Laplace curvature there describes the
box rather than the likelihood. Shrinkage then gets reported with a confidence the data do
not support.

Following the precedent `mix()` set for an unidentified weight, this is a `describe()`
finding rather than a silent number. The check is the identifiability statement behind the
hazard, and it is cheap and pre-fit: this curve's guess rate is its value at the lowest
stimulus levels and its lapse rate is one minus its value at the highest, so a group with no
responses of the losing kind at that end has no evidence about that rate at all.

```python
pooled = hierarchical(curve, over="subject", parameters=("lapse_logit",), scale=0.5)
pooled.describe(study).findings
# [warning] unidentified_group_rate: lapse_rate is at the floor of its range for subject b:
# the study has no evidence of it at the asymptote, so a Gaussian deviation on its logit is
# unbounded and its shrinkage will be reported more confidently than the data allow
```

A warning rather than an error, on the same standing as a knot outside the data's support:
fixing the rate at a known value (`PsychometricFunction(fixed_lapse_rate=0.02)`) or dropping
it from `parameters=` is a modelling decision only its author can make. After the fit the
other half of the same hazard shows up where it always did -- a coordinate resting on its
box sets `boundary_estimate`.

### What is different about the fit

The joint problem is no longer a penalised linear solve. It is a MAP fit of the population
coordinate and every group's deviation together, by multi-start L-BFGS-B inside the widened
box, with the analytic row gradient contracted onto the joint coordinate. The covariance is
a numerical Hessian of that gradient, and each group's own Hessian block, inverted, is that
group's Laplace covariance conditional on the population -- which is exactly the second
moment `scale_estimator="laplace-em"` needs, so estimating a group scale works here too:

```python
hierarchical(
    agent,
    over="subject",
    parameters=("choice_bias",),
    scale=0.4,
    estimate_scale=True,
    scale_estimator="laplace-em",
)
```

`scale_estimator="laplace-profile"` is declined for these families, by name and with the
alternative in the message: the profile builds a per-group conditional objective out of a
group's *design matrix*, and there is not one.

## Parameter names

Naming is mechanical and stable, and it is part of the promise:

| Model | `parameter_names` |
| --- | --- |
| `BernoulliHistoryGLM(predictors=("stimulus",), choice_lags=1)` | `intercept`, `stimulus`, `choice_lag_1` |
| `smooth(base, over="session_order", knots=(0, 4))` | `intercept[session_order=0]`, `intercept[session_order=4]`, `stimulus[session_order=0]`, ... |
| `hierarchical(base, over="subject")` | `intercept`, `stimulus`, `choice_lag_1` |
| `hierarchical(smooth(base, ...), over="subject")` | the smooth names, unchanged |
| `mix(base, UniformChoiceGuess())` | the base names, plus `mixture_logit` |
| `mix(TemporalDiscounting(), UniformChoiceGuess())` | `discount_rate_log`, `inverse_temperature_log`, plus `mixture_logit` |
| `smooth(mix(base, ...), over="session_order", knots=(0, 4))` | the smooth names, plus `mixture_logit[session_order=0]`, ... |
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
| `mix(BernoulliHistoryGLM(...), UniformChoiceGuess())` | `("linear_predictor", "mixture_weight", "component_log_density", "component_probability[0]")` | `(rows, 4, parameters + 1)` |

A mixture over a row objective has no row in that table at all: it declares no
`predictor_cells` and builds no design matrix, because there is no predictor for cells to be
cells of. Its weight is a column of the `(rows, parameters)` coordinate instead, and the
component's contribution is held beside the objective rather than carried down a channel.

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

A model is composable if it satisfies one of two sibling contracts, and
`behavio.contracts.bounded.require_composable` -- the single check `smooth()` and
`hierarchical()` run -- says which.

`behavio.contracts.compose.PenalisedLinearEstimator` is the first: its likelihood must see a
study only through a **quadratically penalised linear predictor**, and its row scores must
be independent given that predictor. Every generalized linear family in Behavio is one of
those, `MultinomialLogit` included, and so is `WienerDriftDiffusion`: there is no link on
its four predictor cells, but each of them is still a design times a coefficient block, and
a trial's joint choice/latency density depends on the study through nothing else. The
GLM-HMM is not, and is refused rather than silently mis-composed -- it composes through the
sibling contract instead, and only on the parameters that can carry a Gaussian.

`behavio.contracts.bounded.BoundedCoordinateEstimator` is the second, described
[above](#models-whose-coordinate-is-bounded-not-linear): eight of the same members, and
`row_objective` in place of `design_matrix` + `likelihood`. `mix()` runs
`behavio.contracts.mixture.require_mixable` instead, which routes on the same question and
adds one of its own -- a mixture averages two densities of *one row's* outcome, so it needs
the rows to be independent, and `row_blocks` is where a model says whether they are.

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
which each row depends on every row before it.

No arrangement of members can be inspected to discover that, so a model in that position
declares it in a sentence:

```python
BernoulliGLMHMM(predictors=("stimulus",)).penalised_linear_refusal
# 'a GLM-HMM is a latent-state mixture, not a penalised linear model: ...'
```

`require_penalised_linear` reads the declaration before it runs the structural test that
would say yes, and reports the sentence in the `TypeError`. `BinaryRLAgent` declares one
too, and it is still true -- there is no linear predictor to widen -- but a *recursion* is
not the same obstacle as a *latent-state mixture*: the agent's parameters are still one
vector per session, so it composes through the bounded-coordinate contract.

`mix()` reads a **second** declaration, `independent_rows_refusal`, and both of these models
make one. That separation is the point: the penalised-linear refusal is about a predictor and
the mixture refusal is about a row, and only the second is what closes a mixture. For a
GLM-HMM the second is a modelling statement as much as an arithmetic one. A lapse on a
GLM-HMM is a lapse on the *emission*, inside the recursion. Averaged in from outside, over
the marginal one-step-ahead prediction, the weight would be free to absorb the state
switching it is supposed to be distinguished from -- which is the opposite of what a lapse
competitor is for.

```python
BernoulliGLMHMM(predictors=("stimulus",)).independent_rows_refusal
# 'a GLM-HMM is a latent-state mixture whose rows are not independent: ...'
```

The other two cells are covered next.

### A GLM-HMM: which cell opened, and what still refuses

`hierarchical()` works on a GLM-HMM. `smooth()` and `mix()` do not, and the reasons are
different from each other and from the sentence above.

```python
from behavio import BernoulliGLMHMM
from behavio.compose import hierarchical

switching = BernoulliGLMHMM(predictors=("stimulus",), n_states=2, l2=0.01)
per_animal = hierarchical(switching, over="subject", parameters=("intercept",), scale=0.5)
```

**Row independence was never the obstacle hierarchy faced.** It is the obstacle
`PenalisedLinearEstimator` faces, and `BoundedCoordinateEstimator` was written to relax it:
`row_blocks` names the blocks a recursion runs over, and a coordinate that is constant within
one is scored exactly. A GLM-HMM recurses over a subject's session, which is the same answer
`BinaryQLearning` gives, so `over="subject"` composes for the same reason and a grouping
column that cuts a session is refused by the same check. Almost none of this cell was new
code; the row objective is the model's own forward-backward gradient, evaluated one session
at a time.

**The simplex is a real obstacle, and it closes transitions rather than hierarchy.** A
transition row lives on a simplex and is charted here by reference-category logits. An
isotropic Gaussian on those is a prior on the *chart*: for \(K \ge 3\) it is not invariant to
which state was made the reference, and the reference here is chosen by label
canonicalisation rather than by the user. So `parameters=` on a GLM-HMM admits emission
coefficients and refuses everything else, `parameters=None` is an error rather than a fit,
and "this animal is stickier" is left to `stickiness=` at population level or to per-animal
models. See [the GLM-HMM page](glm-hmm.md#why-transitions-stay-pooled).

**Label switching is what the cell had to earn, and anchoring is what earns it.** Relabelling
one subject's states leaves a GLM-HMM's likelihood exactly where it was; it does not leave
the group prior where it was, because a relabelled subject is far from the population and
pays for it. Per-subject relabelling is therefore not a symmetry of the joint objective and
the label-consistent solution is its global optimum. The only surviving symmetry is the
global one, and `fit_rows` resolves it by the same `label_by` ordering the pooled fit uses --
exactly, because relabelling is a linear map on this coordinate, so the covariance goes
through it unchanged.

That argument is about the global optimum, so it is checked rather than trusted:

```python
fit = per_animal.fit(study)
agreement = switching.group_label_agreement(fit)
agreement.relabelled_groups  # groups whose deviation is a relabelling, not a difference
agreement.margins  # how much worse the next-best matching is, per group
```

`align_latent_states` does not answer this question. It aligns inferred state posteriors
against *known simulated truth*, which is a recovery diagnostic and unavailable on data;
`group_label_agreement` matches two fitted parameter vectors and runs on anything.

**`smooth()` stays refused, and the reason is again labels.** The ordering that names these
states is an ordering of numbers; under `smooth()` `label_by` becomes a path, paths cross, and
no single permutation canonicalises a fit in which "state 0" is one behaviour early in
training and another late. The fit would converge and report knots, which is what makes it
dangerous rather than merely unavailable.

### Saying which parameters may vary at all

`require_penalised_linear` and `require_bounded_coordinate` ask whether a model composes.
`behavio.contracts.compose.require_varying_parameters` asks the question immediately after,
and it is a different one: a model can have a perfectly good row objective and a perfectly
good box while still having parameters no Gaussian deviation and no Gaussian random walk can
sit on.

The case that forced it is a coordinate that is a **chart** rather than a quantity, and no
member can express that -- `penalty_matrix` and `group_penalty` have the same shape whether
or not the coordinate they act on is exchangeable. So a model may declare
`varying_parameter_refusal(parameters, *, combinator=...)`, and both `smooth()` and
`hierarchical()` consult it before building anything. The `combinator` argument is there
because the answer may legitimately differ by axis, which is exactly what a latent-state model
does when its labels are an ordering of one of the parameters in question.

Like `penalised_linear_refusal`, it is read with `getattr` and absent by default: a contract
cannot require a model to announce which parts of itself it declines.

To make a new family composable, implement those members. See
[extensions](extensions.md).
