# Scalar timing

Two paradigms, one memory. Gibbon's (1977) scalar expectancy theory says that a timed
duration is represented with noise **proportional to the duration itself**, and that single
statement is the content of the theory: a clock whose error grows by a fixed number of
milliseconds is not a scalar clock. So both families here share

\[
\log \hat{T} \sim \mathcal{N}\!\left(\log(\kappa\,T^{\beta}),\ \sigma^2\right),
\qquad \sigma^2 = \log(1 + w^2),
\]

for a clock rate \(\kappa > 0\), a Weber fraction \(w > 0\) and an optional central-tendency
exponent \(\beta > 0\). The parameterisation is chosen so that the **coefficient of variation
of \(\hat{T}\) is exactly \(w\) at every \(T\)** — the scalar property stated as an identity
rather than as an approximation that happens to hold for small \(w\).

```python
from behavio.models import BisectionRule, DurationReproduction, TemporalBisection

reproduction = DurationReproduction()
bisection = TemporalBisection(short_anchor=2.0, long_anchor=8.0)
```

Nothing here is called an *interval*. `behavio.observed.interval_policy` curates externally
discovered annotation bouts and has nothing to do with timing; "interval" already means a
confidence interval and an annotated bout in this package, so the paradigm the literature
calls interval reproduction is `DurationReproduction`, and the thing an animal times is a
**duration**.

## Why lognormal rather than Gibbon's Gaussian

Gibbon writes the representation as a normal variable with standard deviation proportional
to its mean. That has the scalar property exactly, and also assigns positive probability to a
negative duration. At \(w = 0.15\) that mass is \(10^{-11}\) and nobody notices; at
\(w = 0.5\) it is 2.3 %, and an optimizer exploring a box that reaches \(w = 3\) spends time
in a region where the model is not a model of durations at all.

A lognormal is positive everywhere, has constant coefficient of variation everywhere, and is
linear in \(\log T\) — which is what lets the *same* memory produce a reproduction density
and a bisection psychometric function without a second set of assumptions. What it changes
empirically is the shape of the reproduction distribution's right tail, which is the one
place the two forms disagree measurably; a study that can resolve that tail should say which
it is fitting.

```python
from behavio.models import memory_log_sd, weber_fraction_from_log_sd

memory_log_sd(0.25)  # sqrt(log(1 + w**2))
weber_fraction_from_log_sd(0.2462)  # its exact inverse, not a small-noise approximation
```

## Duration reproduction

Each row presents a target duration and records a reproduction, and the model is the shared
memory read as a density on that reproduction.

| Column | Meaning |
| --- | --- |
| `target_duration` | the duration presented, strictly positive |
| `reproduced_duration` | the duration produced, strictly positive |

`clock_rate` scales the **median** reproduction, so \(\kappa = 1\) is an animal whose typical
reproduction is the target. Reporting the median rather than the mean is what keeps
\(\kappa\) orthogonal to \(w\): a lognormal's mean moves with its width, so a
"mean reproduction" parameterisation would make a noisier animal look like a slower clock.

```python
truth = reproduction.parameters_from_components(clock_rate=1.0, weber_fraction=0.2)
study = reproduction.simulate(design, truth, seed=0)
fit = reproduction.fit(study)

reproduction.coefficient_of_variation(fit)  # one number, with no study argument
reproduction.median_reproduction(study, fit.estimates)
```

`coefficient_of_variation` takes no `study` argument, and that is the point: under the scalar
property there is nothing for it to depend on.

`predict()` returns a `DensityPrediction` over the reproduction column, tabulated on one
shared geometric grid. `pointwise_log_prob` returns the **analytic** log density rather than
reading the grid, so a fold's score never depends on `grid_points`; the grid is a tabulation
of the same closed form, which is what makes the two agree to interpolation error rather than
by construction. The density keeps its change-of-variable Jacobian even though it is constant
in the parameters, because a pointwise score is compared *across* models and a density that
dropped its Jacobian would win for a reason that is not about behaviour.

### The central-tendency exponent

\(\beta = 1\) is pure scalar timing: the reproduced duration is proportional to the target.
\(\beta < 1\) is Vierordt's law — short durations over-reproduced, long ones
under-reproduced — which is robust in humans and is a regression towards the centre of the
tested range rather than a property of the clock.

```python
vierordt = DurationReproduction(fixed_central_tendency=None)  # estimate it
```

It is fixed at one by default, because a model whose mean is free is no longer making the
prediction the scalar property is a prediction about. It is also **not estimable from
bisection** at all; see below.

## Temporal bisection

Two anchors \(S < L\) are trained and **declared**, never learned: they are a fact about the
training procedure, so they appear in the model's signature and a fit can never be read
without them, exactly as `value_scale` works for `TemporalDiscounting`. The outcome column is
one when the animal responded *long*.

| Column | Meaning |
| --- | --- |
| `probe_duration` | the probe presented, strictly positive |
| `choice` | one when the animal reported *long* |

Under the ratio rule,

\[
P(\text{long} \mid t)
= \Phi\!\left(\frac{\log(\kappa t) - \tfrac{1}{2}\log(SL)}{\sigma}\right),
\qquad \sigma = \sqrt{\log(1 + w^2)},
\]

which crosses one half at \(t = \sqrt{SL}/\kappa\). **An accurate clock bisects at the
geometric mean of the anchors.** That is Church and Deluty's (1977) result, and it is the
known answer the family is validated against — with no fitting anywhere in the assertion.

```python
from behavio.models import bisection_threshold

bisection_threshold(2.0, 8.0)  # 4.0, the geometric mean
bisection_threshold(2.0, 8.0, rule=BisectionRule.DIFFERENCE)  # 5.0, the arithmetic mean
bisection.bisection_point(fit)  # the fitted probe duration reported long half the time
```

A fit reports `bisection_point` as a derived quantity with a standard error, and its
description names the rule that produced it.

## A single anchor pair cannot separate the two rules

Fitting a bisection study needs a **decision rule** — a statement of which anchor a
remembered probe counts as closer to — and the literature does not agree on one.

| Rule | Respond *long* when | Crosses one half at |
| --- | --- | --- |
| Ratio (Gibbon 1981) | \(\hat{T}/S > L/\hat{T}\) | \(\sqrt{SL}\) |
| Similarity (Wearden 1991) | the smaller/larger ratio favours \(L\) | \(\sqrt{SL}\) |
| Difference (arithmetic) | \(\hat{T} - S > L - \hat{T}\) | \((S+L)/2\) |

The first two are the same rule — "similarity of two durations" *means* the ratio of the
smaller to the larger — so there are two candidate rules, not three, and they disagree about
exactly one number.

Both produce the same psychometric *shape*, a probit in \(\log t\) with slope \(1/\sigma\),
because the probit comes from the memory rather than from the rule. On **one anchor pair**
the two are therefore a reparameterisation: they fit any single study identically, with the
same Weber fraction, the same log likelihood, and clock rates differing by exactly the ratio
of the two comparison durations:

\[
\frac{\kappa_{\text{difference}}}{\kappa_{\text{ratio}}}
= \frac{(S+L)/2}{\sqrt{SL}} = \frac{S + L}{2\sqrt{SL}}
\]

— which is \(5/4\) for a 2 s / 8 s pair, and is asserted by a committed test that also checks
that the two fits place the bisection point at the *same observed duration*. Since a model
instance carries one anchor pair, **no single fit of `TemporalBisection` can test the rule**.

So the rule is part of the model's `signature` and of its `model_name`.
`temporal-bisection-ratio` and `temporal-bisection-difference` are different models to
`compare_models`, to `nested_select_model` and to a frozen protocol, and separating the
accounts means fitting several anchor pairs with the clock rate held common across them —
Church and Deluty's design, and a comparison **between** models rather than a parameter
inside one. This package does not currently offer a multi-pair model.

**A reported bisection point is meaningless without the rule.** That is why the rule is in
the signature rather than a constructor convenience.

### Why the ratio rule is the default

*Empirical.* Church and Deluty trained rats on four anchor pairs — 1 v 4, 2 v 8, 3 v 12 and
4 v 16 seconds — and found the bisection point at the geometric mean of each. The arithmetic
rule predicts 2.5, 5, 7.5 and 10 s where the geometric rule predicts 2, 4, 6 and 8. Varying
the pair is what makes the accounts separable.

*Theoretical.* The clock's noise is multiplicative, which is the whole content of scalar
expectancy theory, and a ratio comparison is the only one that leaves every noise source in
the model multiplicative. This module treats the anchors as remembered exactly, and under
that idealisation the rules differ in one number and nothing else — but any complete account
has to admit that anchors are remembered with noise too, and the moment it does the
idealisation breaks in the ratio rule's favour: the ratio comparison's anchor noise is
additive on the log scale and therefore scalar, the difference comparison's is additive on
the linear scale and therefore is not. The difference rule stops obeying Weber's law at
exactly the point where it stops being a relabelling.

`BisectionRule.DIFFERENCE` is implemented and reachable, because the disagreement is real and
human bisection points do sometimes land near the arithmetic mean.

### What bisection cannot identify

A **central-tendency exponent**. Under this memory the decision variable is
\(\beta \log t / \sigma\), so \(\beta\) and the Weber fraction enter the slope as one number
and are exactly confounded. `DurationReproduction` estimates one; bisection cannot, and the
honest response is to run both paradigms rather than to report an unidentified number.

A **response bias** separate from the clock. A subject who says *long* more often and a
subject whose clock runs fast move the bisection point the same way, so `clock_rate` here is
the two of them together. Reproduction separates them; bisection alone cannot.

## One Weber fraction across both paradigms

The two families estimate the same two parameters from the same memory, so a study that runs
both can ask whether **one \(w\) describes them** — scalar timing's strongest testable claim,
and one neither paradigm can make alone. Nothing in the two likelihoods is shared: one is a
lognormal density on a duration and the other a probit on a binary report, so agreement is
evidence that the memory behind them is one memory. A committed test asserts it.

## What each design can and cannot see

```python
DurationReproduction().describe(study).findings
# [warning] narrow_target_range: the tested durations span a factor of 1.2, so a scalar clock
# and a clock with constant absolute variability predict nearly the same spreads here ...
```

| Finding | The design that produces it |
| --- | --- |
| `narrow_target_range` | tested durations spanning less than half an octave |
| `unidentified_central_tendency` | one target duration, with the exponent estimated |
| `weakly_identified_central_tendency` | two target durations, with the exponent estimated |
| `too_few_probe_durations` | fewer than three distinct probes: a location and a slope cannot be separated |
| `probes_do_not_span_the_comparison` | every probe on one side of the comparison duration, so the bisection point is an extrapolation |
| `probes_outside_the_anchors` | probes outside the trained range, where the model extrapolates a rule the animal was never trained to apply |
| `narrow_anchor_ratio` | \(L/S < 1.5\), so the two rules' comparison durations are within a few per cent of each other |

`narrow_target_range` is the one that matters most, and it is worth restating why: **the
scalar property is a claim about how variability changes with duration**, so a design that
tests one duration cannot see it however many trials it runs. None of these is an error, and
each is a statement about a design only its author can change.

## Every coordinate is a logarithm

| Reported | Estimated |
| --- | --- |
| `clock_rate` | `clock_rate_log` |
| `weber_fraction` | `weber_fraction_log` |
| `central_tendency` (when estimated) | `central_tendency_log` |

Both families sit on the shared log-coordinate estimator, so the natural parameterisation,
the multi-start solver, the numerical curvature, the row objective and the group prior are
written once. The boxes are **declared** rather than derived from the design, unlike the
discounting family's delay-derived box, because these bounds are statements about *timing*:
a clock twenty times fast is not a clock, and a Weber fraction of three is an animal whose
reproductions carry no information about the target.

Restarts are deterministic and closed form. Reproduction starts from the study's own log-log
regression of \(\log R\) on \(\log T\), whose location, slope and residual spread are exactly
the three parameters; bisection starts from the empirical crossing.

## Composition

`smooth()` and `hierarchical()` compose over both families through the
[bounded-coordinate contract](composing-models.md#models-whose-coordinate-is-bounded-not-linear),
with nothing added to either combinator.

```python
from behavio.compose import hierarchical, mix, smooth, UniformChoiceGuess

per_animal = hierarchical(
    reproduction, over="subject", parameters=("weber_fraction_log",), scale=0.5
)
drifting = smooth(
    reproduction, over="session_order", knots=(0.0, 5.0), parameters=("clock_rate_log",)
)
lapsing = mix(bisection, UniformChoiceGuess(), weight_bounds=(0.0, 0.2))
```

A Gaussian group deviation on the Weber fraction itself would be a negative fraction
sometimes; on its logarithm, never. That is the whole reason the coordinate is a logarithm.

**A within-session clock is admissible here.** These rows are independent — `row_blocks` is
`arange(n_rows)` — so a coordinate may vary trial by trial, exactly as it may for a
psychometric curve and unlike a reinforcement-learning agent. Admissible is not estimable: a
knot per trial is not a model of anything, and the roughness penalty is the only thing
standing between the two.

**`mix()` reaches bisection and not reproduction**, and the obstacle is a component rather
than a combinator. A bisection report is a binary choice, so `UniformChoiceGuess` mixes with
it. A reproduction's observation is a **bare duration**, and no shipped mixture component
scores one: `UniformChoiceGuess` writes a binary choice and `UniformResponseGuess` writes a
joint choice and latency, and `require_mixable` refuses both by comparing scored columns
before any arithmetic happens. `mix()` itself is untouched and would work.

Neither family has a linear predictor — one puts a parameter in the *scale* of a density, the
other divides a linear term by an estimated memory width — so neither takes the
penalised-linear route, and both say so through `penalised_linear_refusal` rather than
leaving it to structural typing.

## Scoring and comparison

`evaluate_splits` works for both. `compare_models` ranks a bisection model against any other
choice model, because a bisection report is a binary outcome with a discrete margin.

**It cannot rank two reproduction candidates against each other.** `compare_models` reports a
Brier score beside the log score, a Brier score needs a discrete margin, and an unlabelled
density has none, so it raises `UnscoreableByBrier` rather than inventing a number. The log
score *is* defined, and `evaluate_splits` reports it; what is missing is a way to ask
`compare_models` for that half alone. That gap is in the comparison layer rather than in this
family and is recorded in
[SDR-0063](decisions/0063-defer-the-log-score-only-comparison-and-the-survival-carrying-prediction.md).

## What a fitted Weber fraction does not establish

**A scalar clock.** A Weber fraction from a narrow target range is a spread, not a
demonstration that spread scales with duration; the scalar property is a claim about a
*range* of durations and `describe()` reports `narrow_target_range` before the fit.

**A bisection point on its own.** It is unreadable without its rule, which is why the rule is
in the signature, and it cannot be compared with a bisection point reported under the other
rule without converting the clock rate by the ratio of their comparison durations.

**A property of a subject.** Both parameters are properties of a fitted curve on a declared
duration coordinate. A clock rate estimated from bisection is the clock and any response bias
together.

## References

- Gibbon, J. (1977). Scalar expectancy theory and Weber's law in animal timing.
  *Psychological Review, 84*(3), 279-325.
- Church, R. M., & Deluty, M. Z. (1977). Bisection of temporal intervals. *Journal of
  Experimental Psychology: Animal Behavior Processes, 3*(3), 216-228.
- Gibbon, J. (1981). On the form and location of the psychometric bisection function for
  time. *Journal of Mathematical Psychology, 24*(1), 58-87.
- Wearden, J. H. (1991). Human performance on an analogue of an interval bisection task.
  *Quarterly Journal of Experimental Psychology, 43B*(1), 59-81.

The decision-rule argument, including what a multi-pair model would change, is
[SDR-0060](decisions/0060-bisect-time-by-the-ratio-rule.md).
