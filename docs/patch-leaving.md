# Patch leaving

Charnov's (1976) marginal value theorem says that an animal exploiting a depleting patch in
an environment whose patches are separated by a travel time \(\tau\) should leave when the
patch's instantaneous intake rate has fallen to the environment's long-run average rate:

\[
g'(t^{*}) = R^{*} = \max_{t} \frac{g(t)}{\tau + t}.
\]

That is a **normative** statement. It says what an optimal forager would do; it is not a
likelihood, and it has no free parameters — given a gain function and a travel time,
\(t^{*}\) is determined. Meanwhile the most replicated finding in the patch-leaving
literature is that animals **overstay**: they leave at an intake rate below \(R^{*}\).

So this module keeps the theorem and the model apart on purpose.

```python
from behavio.models import PatchLeaving, marginal_value_rate, marginal_value_residence_time

model = PatchLeaving(travel_time_column="travel_time")
marginal_value_rate(patch_yield=8.0, patch_decay=0.3, travel_time=10.0)  # no fitting in it
```

`marginal_value_rate` and `marginal_value_residence_time` are the closed form, with no
fitting anywhere near them. `PatchLeaving` is a behavioural model of when an animal actually
goes, fitted to residence times without assuming that the threshold it estimates is the
optimal one. **A module that fitted the theorem could not measure a departure from it.**

## Patch leaving fits a hazard, not the marginal value theorem

A patch visit produces a **residence time**, not a choice. The animal is in the patch and is,
moment by moment, deciding whether to go, so the fitted object is a leaving decision per unit
residence time — a hazard. The primitive is the instantaneous leaving rate and the density is
derived from it, which is the *opposite* of a response-time model, where a first-passage
density is the primitive and its hazard is a description computed afterwards.

The animal carries a giving-up rate \(\theta\): the intake rate at which the patch stops
being worth staying in. It does not carry it exactly, and the noise goes on the log scale
because a rate's noise is multiplicative:

\[
\log \Theta \sim \text{Logistic}(\log\theta,\ s), \qquad
T = \inf\{t : g'(t) \le \Theta\}.
\]

Because \(g'\) is strictly decreasing, everything is closed form:

\[
S(t) = \frac{\sigma(u(t))}{\sigma(u(0))}, \qquad
u(t) = \frac{\log g'(t) - \log\theta}{s}, \qquad
\lambda(t) = -u'(t)\,\bigl(1 - \sigma(u(t))\bigr).
\]

Two estimated parameters, both positive, both estimated as logarithms: `giving_up_rate` is in
intake units per time, and `decision_noise` is dimensionless — a Weber fraction on the intake
rate. The hazard rises as the patch depletes, which is the shape patch residence data have.

The division by \(\sigma(u(0))\) is a **conditioning, not a normalisation**: an animal that
entered a patch and stayed in it for a positive time had a threshold below that patch's entry
rate, so the model conditions on that rather than placing an atom of probability at zero
residence.

As \(s \to 0\) the leaving time converges to the deterministic threshold crossing, so setting
\(\theta = R^{*}\) recovers Charnov's \(t^{*}\) exactly. That is what the simulator is
validated against.

| Column | Meaning |
| --- | --- |
| `patch_yield` | the patch's asymptotic gain \(A\) |
| `patch_decay` | \(\rho\) for the exponential gain, \(h\) for the hyperbolic |
| `residence_time` | the observed visit duration |
| `travel_time_column=` | optional; changes no likelihood, adds three derived quantities |
| `censoring_time_column=` | optional; changes the likelihood a great deal |

## `overstaying_ratio` is what the theorem is for

Declare `travel_time_column` and a fit gains three derived quantities:

```python
fit = model.fit(study)
fit.derived_value("marginal_value_rate")  # Charnov's R*, a benchmark, not an estimate
fit.derived_value("optimal_residence_time")  # where each patch's rate reaches R*
fit.derived_value("overstaying_ratio")  # R* / fitted giving_up_rate
```

Above one is an animal that leaves at a lower intake rate than optimal, which is to say it
overstays. The number is something the fit *reports*, with a delta-method standard error,
rather than an assumption baked into the model reporting it.

**The environment, not the patch, sets the threshold.** `marginal_value_rate` solves the
fixed point over every patch type present, weighted by the frequency with which it appears,
so a heterogeneous study gets **one rate** and a **different optimal residence time per
type**. `describe()` reports `heterogeneous_environment` so a reader does not take the mean of
several optima for one.

### Two gain functions, and their decay parameters are not interchangeable

| | Gain | Optimum |
| --- | --- | --- |
| `GainFunction.EXPONENTIAL` | \(g(t) = A(1 - e^{-\rho t})\) | solves an implicit equation; independent of \(A\) |
| `GainFunction.HYPERBOLIC` | \(g(t) = At/(t+h)\) (Holling) | exactly \(t^{*} = \sqrt{h\tau}\), with \(R^{*} = A/(\sqrt{h}+\sqrt{\tau})^2\) |

They are different models — the gain appears in `model_name` and in the signature — and the
second exists partly because an exact closed form is what the root finder can be validated
against without validating it against itself. The exponential form's optimum not depending on
\(A\) is itself a discriminating prediction that the hyperbolic form does not make.

## Censoring is declared, and it changes the score

A session that ends while the animal is still in a patch is **not a leaving time**. Declare
`censoring_time_column` — the longest residence each row could have shown — and such a row is
scored by \(\log S(c)\) rather than \(\log f(t)\).

```python
model = PatchLeaving(travel_time_column="travel_time", censoring_time_column="session_left")
```

Leave it undeclared and the model asserts that every duration ran to its event. `describe()`
reports `undeclared_censoring` when the residence times pile up on a common maximum, which is
what that assertion looks like when it is false. Ignoring real censoring biases the threshold
in a direction that is easy to state and easy to miss — a truncated visit read as a departure
looks like a forager that left sooner, so the fitted giving-up rate moves **up** — and a
committed test measures it.

Left truncation and interval censoring are **refused rather than approximated**. An animal
that entered a patch before recording began has a conditional likelihood, not a marginal one,
and scoring it as if it did not would be the exact failure the censoring machinery exists to
prevent.

### One thing does not fit the prediction contract, and it is reported

`predict()` returns a `DensityPrediction` of the **leaving time**, on every row, censored or
not: that is what the model claims about the row. A censored row's *score* is a survival
probability, and no member of `ModelPrediction` can carry "the probability the event is still
to come".

So `pointwise_log_prob` and `DensityPrediction.observed_log_density` **agree on uncensored
rows and deliberately disagree on censored ones**. The first is the likelihood; the second is
the prediction. A consumer that scores the density directly instead of asking the model will
misscore exactly the censored rows, and `describe()` says so through `heavy_censoring` with
the share of the study affected. The gap in the contract is recorded in
[SDR-0063](decisions/0063-defer-the-log-score-only-comparison-and-the-survival-carrying-prediction.md).

The censoring *arithmetic* — which rows are scored by a density and which by a survival, how
the gradient follows the same selection, and whether a duration equal to its limit is an
event — is written once in `behavio.models._kernels.hazard` and is what a censored
response-time model would import. It would not import a hazard family: forcing a Wiener
first-passage density into a parametric-hazard family would be a claim about drift diffusion
that drift diffusion does not make.

## A single patch type cannot test the theorem at all

If every patch in a study depletes identically then \(\log g'(t)\) is one fixed monotone
function of \(t\), and "leave when the intake rate falls to \(\theta\)" predicts **exactly**
what "leave after \(t_\theta\) seconds" predicts. The content of the marginal value theorem
is that *one rate threshold governs patches of different richness and depletion*, and that is
invisible in such a design however many visits it contains.

```python
PatchLeaving().describe(study).findings
# [warning] unidentified_leaving_rule: every patch in this study depletes identically, so the
# intake rate is one fixed function of elapsed time and 'leave when the rate falls to a
# threshold' predicts exactly what 'leave after a fixed time' predicts ...
```

This is the most important finding the family has. A fitted `giving_up_rate` from such a
design is a residence time wearing a rate's units.

| Finding | The design that produces it |
| --- | --- |
| `unidentified_leaving_rule` | one patch type: a rate threshold and a time threshold are the same model |
| `heterogeneous_environment` | more than one type, so `optimal_residence_time` should be read per type |
| `undeclared_censoring` | residence times piled up on a common maximum with no censoring column |
| `heavy_censoring` | a quarter or more of visits still in progress, so the density and the score diverge on that share |
| `all_rows_censored` | nothing ever departed, so the giving-up rate is bounded rather than located |

## Every coordinate is a logarithm

| Reported | Estimated |
| --- | --- |
| `giving_up_rate` | `giving_up_rate_log` |
| `decision_noise` | `decision_noise_log` |

The family sits on the shared log-coordinate estimator alongside the scalar-timing families,
so the natural parameterisation, the multi-start solver, the curvature, the row objective and
the group prior are written once.

The threshold's box is **derived from the study** rather than declared, as
`TemporalDiscounting` derives its rate box from the delays that were used: a giving-up rate
above every patch's entry rate is an animal that never enters, and one far below the rate any
observed visit reached is an animal no residence time in this study can distinguish from a
slightly lower one. Restarts are closed form and deterministic — a threshold model's median
residence time *is* its threshold crossing, so the median observed visit's own intake rate is
the design's own estimate of the giving-up rate, and the restarts vary the noise around it.

## Composition

`smooth()` and `hierarchical()` compose over the family through the
[bounded-coordinate contract](composing-models.md#models-whose-coordinate-is-bounded-not-linear),
over a censored likelihood as readily as an uncensored one.

```python
from behavio.compose import hierarchical, smooth

per_animal = hierarchical(model, over="subject", parameters=("giving_up_rate_log",), scale=0.5)
drifting = smooth(model, over="session_order", knots=(0.0, 5.0), parameters=("giving_up_rate_log",))
```

**`mix()` reaches this family through `UniformDurationGuess`**, the component that scores a
bare duration. `mix()` itself is unchanged; a component that writes a residence time is all
that was ever missing.

```python
from behavio.compose import UniformDurationGuess, mix

contaminated = mix(
    PatchLeaving(censoring_time_column="observation_limit"),
    UniformDurationGuess(
        duration_bounds=(0.0, 3.0),
        outcome="residence_time",
        censoring_time_column="observation_limit",
    ),
    weight_bounds=(0.0, 0.4),
)
contaminated.natural_names
# ("giving_up_rate", "decision_noise", "contaminant_rate")
```

**The component declares the censoring column too, and it must be the model's.** A mixture
averages what each process says about the observation that was actually made, and on a
censored row that observation is "the visit outlasted \(c\)". So the component contributes
the probability *its* duration exceeds the same \(c\) rather than its density there:

\[
S_{\text{mix}}(c) = (1-\omega)\,S_{\text{model}}(c) + \omega\,S_{\text{comp}}(c).
\]

Contributing the density instead is dimensionally wrong — a density is one over time, a
survival probability is dimensionless — so every censored row would look like a row the
contaminant could not have produced and the weight would be dragged towards the floor of its
declared range. On a simulated study with three visits in five censored, a censoring-blind
component recovers less than half the weight, while `giving_up_rate` moves by under ten per
cent. `mix()` refuses a component and a model that read different limit columns rather than
letting the two disagree.

`duration_bounds` is in the residence-time column's own units and appears in the composed
model's signature. The identifiability findings carry over with one shift worth knowing: a
contaminant whose declared interval lies entirely beyond every observed residence time is
**not** reported as `unreachable_mixture_component` here, and correctly so — a process that
always outlasts the session is exactly what a censored row is consistent with.

There is no linear predictor here either — the model divides a log intake rate by an
estimated noise and reads the result through a survival function — so nothing in it is a
design matrix times a coefficient, and `penalised_linear_refusal` says so rather than leaving
it to structural typing.

## Scoring and comparison

`evaluate_splits` works and reports the log score of every row, censored rows included, under
the model's own `pointwise_log_prob`.

`compare_models` **ranks two patch-leaving candidates on a declared log score**. Its default
table carries a Brier score beside the log score, a Brier score needs a discrete margin, and a
residence time has none — so a patch-leaving candidate is refused that column by name, before
anything is fitted, rather than being given an invented number. Declaring the rule that is
defined for it gives the table back:

```python
report = compare_models(
    {"hyperbolic": PatchLeaving(), "exponential": PatchLeaving(gain=GainFunction.EXPONENTIAL)},
    study,
    splits,
    outcome_column="residence_time",
    metrics=(ScoreMetric.LOG_LOSS,),
)
```

The scores are the model's own `pointwise_log_prob`, so censored rows enter as `log S(c)` and
not as a density. See
[declaring which rules the table carries](comparison.md#declaring-which-rules-the-table-carries)
and [SDR-0063](decisions/0063-defer-the-log-score-only-comparison-and-the-survival-carrying-prediction.md).

The constrained model is still reachable as a *comparison*: fix `giving_up_rate` at
`marginal_value_rate` and score it against the free fit.

## What a fitted giving-up rate does not establish

**The marginal value theorem.** A giving-up rate from a single patch type is a residence time
wearing a rate's units; the theorem's content is invisible in that design and `describe()`
reports `unidentified_leaving_rule` before the fit.

**Optimal foraging.** The model does not assume the fitted threshold is \(R^{*}\), which is
precisely what lets `overstaying_ratio` measure the distance between them. A ratio near one
is a result, not a validation of the model.

**A completed visit.** A censored row scored by its density is a visit the model was never
told was truncated, and the bias it produces is upward in the giving-up rate. Declare the
censoring column, and read the likelihood through `pointwise_log_prob`.

## References

- Charnov, E. L. (1976). Optimal foraging, the marginal value theorem. *Theoretical
  Population Biology, 9*(2), 129-136.
- Stephens, D. W., & Krebs, J. R. (1986). *Foraging Theory*. Princeton University Press.

Why the theorem is a benchmark rather than the likelihood, what a per-moment decision noise
would have confounded, and what a censored response-time model would and would not reuse from
here are recorded in
[SDR-0061](decisions/0061-fit-patch-leaving-as-a-hazard-not-as-the-marginal-value-theorem.md).
