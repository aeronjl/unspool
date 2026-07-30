# Economic and value-based choice

Two families, one shape. A trial offers two options, each option gets a subjective value
\(V\), and the choice is a softmax:

\[
P(\text{option 1}) = \sigma\!\left(\beta\,(V_1 - V_0)\right).
\]

`TemporalDiscounting` puts a discount factor between an amount and its value.
`ProspectTheory` puts a value function and a probability weighting function there. Nothing
else differs, and that is not an implementation detail — it is why this pair cost one value
function each rather than one model each.

```python
from behavio.models import ProspectTheory, TemporalDiscounting

impulsivity = TemporalDiscounting(value_scale=100.0)
risk = ProspectTheory(value_scale=100.0)
```

## Temporal discounting

The canonical design is a binary choice between a **smaller-sooner** and a **larger-later**
option, so a trial carries two amounts and two delays:

| Column | Meaning |
| --- | --- |
| `sooner_amount`, `later_amount` | the two options' amounts |
| `sooner_delay`, `later_delay` | the two options' delays, in whatever unit the study uses |
| `choice` | one when the **second** (conventionally later) option was chosen |

\[
V = \frac{A}{S}\,D(t), \qquad
D(t) = \frac{1}{1 + kt} \ \text{(Mazur 1987)} \quad\text{or}\quad D(t) = e^{-kt}.
\]

```python
model = TemporalDiscounting(discount="exponential", value_scale=100.0)
truth = model.parameters_from_components(discount_rate=0.02, inverse_temperature=8.0)
study = model.simulate(design, truth, seed=0)
fit = model.fit(study)

model.parameter_components(fit).discount_rate
fit.derived_value("discount_rate")  # the same number, with a delta-method standard error
```

### The indifference point is closed form, and the package uses it twice

Equating the two options gives, for the hyperbola,
\(k^{*} = (A_L - A_S) / (A_S t_L - A_L t_S)\), and for the exponential
\(k^{*} = \log(A_L/A_S)/(t_L - t_S)\). At \(k = k^{*}\) the model's choice probability is
exactly one half **whatever the inverse temperature is**.

That is used as a *validation* — a committed test asserts it for both discount functions,
which no implementation with the wrong factor can satisfy for both at once — and as the
*restart schedule*: `initial_points` starts from quantiles of the design's own indifference
rates rather than from a constant, so there is no unseeded randomness anywhere in the fit.

```python
from behavio.models import indifference_discount_rate

indifference_discount_rate(50.0, 0.0, 100.0, 30.0)  # 0.0333... per day
model.indifference_rates(study)  # one per trial, read off the model's own columns
```

## Prospect theory

Each option is a two-outcome prospect \((x, p;\, y, 1-p)\):

| Column | Meaning |
| --- | --- |
| `option_0_outcome`, `option_1_outcome` | each option's first outcome |
| `option_0_probability`, `option_1_probability` | the probability of that outcome |
| `other_outcome_columns=` | optional; the complementary outcome, zero when not named |
| `choice` | one when the **second** option was chosen |

\[
V = \pi_h\,v(x_h) + \pi_l\,v(x_l), \qquad
v(z) = \begin{cases} (z/S)^{\alpha} & z \ge 0 \\ -\lambda\,(-z/S)^{\beta} & z < 0 \end{cases}
\]

with Prelec's \(w\) supplying the decision weights \(\pi\).

### Weights are cumulative, and for a two-outcome prospect that is three cases

Cumulative prospect theory (Tversky & Kahneman 1992) rather than the separable weighting of
the 1979 paper, and no numerical work is involved. With the two outcomes ranked so that
\(x_h \ge x_l\) and \(q = P(x_h)\):

| Both weakly positive | Both weakly negative | Mixed |
| --- | --- | --- |
| \(\pi_h = w(q)\), \(\pi_l = 1 - w(q)\) | \(\pi_h = 1 - w(1-q)\), \(\pi_l = w(1-q)\) | \(\pi_h = w(q)\), \(\pi_l = w(1-q)\) |

Two consequences are worth stating. A **sure** prospect is unweighted, \(V = v(x)\), so no
dominated prospect can be preferred — the property the 1979 form lacked. And whenever one
outcome is zero the two theories **agree exactly**, so the four-column design that most risk
studies actually run costs nothing for the choice of cumulative weighting.

### Why Prelec (1998) and not Tversky–Kahneman (1992)

\[
w(p) = \exp\!\left(-\delta(-\log p)^{\gamma}\right)
\qquad\text{rather than}\qquad
w(p) = \frac{p^{\gamma}}{\left(p^{\gamma} + (1-p)^{\gamma}\right)^{1/\gamma}}.
\]

Three reasons, in the order they mattered.

1. **It separates two things that are psychologically separate.** Gonzalez and Wu (1999)
   showed that the *curvature* of the weighting function and its *elevation* vary
   independently across people. TK92's single \(\gamma\) moves both, so "nearly linear but
   pessimistic" is a subject it cannot describe. Prelec's \(\gamma\) is curvature and
   \(\delta\) is elevation.
2. **It is monotone wherever its parameters are admissible.** TK92's form is not increasing
   on \((0,1)\) for \(\gamma\) below about 0.28. An optimizer that wanders there is fitting
   something that is not a weighting function, and it will converge and report a number.
   Prelec's is strictly increasing for every \(\gamma > 0, \delta > 0\), so the box does not
   have to encode a constraint that came from the parameterisation rather than the data.
3. **Its fixed point is readable.** \(w(p) = p\) at \(p = \exp(-\delta^{1/(1-\gamma)})\),
   which is \(1/e \approx 0.368\) whenever \(\delta = 1\) — close to the crossover the data
   keep showing.

**What TK92 would change:** one parameter instead of two, so curvature and elevation would be
locked together; a floor near 0.28 on \(\gamma\)'s box; and a fixed point that moves with
\(\gamma\) instead of being read off \(\delta\). Nothing else. `w` and its two derivatives are
the only thing the likelihood asks the weighting function for, so swapping the form is a
change to one function.

One weighting function is estimated and used in both domains. TK92 reported
\(\gamma^+ = 0.61\) and \(\gamma^- = 0.69\); a second pair doubles the weighting coordinate
for a difference a single subject's data rarely resolves.

### The fourfold pattern is a prediction, so it is asserted

\[
c = x\,w(p)^{1/\alpha}, \qquad \text{risk seeking} \iff c > px.
\]

With TK92's medians the four cells come out as published, and a committed test asserts it
twice — once from the declared parameters, with no fitting at all, and once from parameters
recovered out of simulated choices:

| | likely | unlikely |
| --- | --- | --- |
| **gains** | risk averse | risk seeking |
| **losses** | risk seeking | risk averse |

```python
from behavio.models import ProspectTheoryParameters, certainty_equivalent

tk92 = ProspectTheoryParameters(0.88, 0.88, 2.25, 0.65, 1.0, 8.0)
certainty_equivalent(100.0, 0.05, tk92)  # 9.84 against an expected value of 5
```

Loss aversion cancels out of a within-domain certainty equivalent, which is exactly why the
model warns about a design with no gain-against-loss trial.

## Every coordinate is a logarithm, and that is the whole transform story

| Reported | Estimated | Why |
| --- | --- | --- |
| `discount_rate` | `discount_rate_log` | positive; multiplicative in what it does |
| `gain_exponent`, `loss_exponent` | `..._log` | positive; **not** bounded above by one |
| `loss_aversion` | `loss_aversion_log` | positive |
| `weighting_curvature`, `weighting_elevation` | `..._log` | positive |
| `inverse_temperature` | `inverse_temperature_log` | positive; the same name the RL families use |

A logarithm is where a positive parameter's Wald interval cannot cross zero, and it is where
a Gaussian group deviation is admissible — which is the only requirement
[`BoundedCoordinateEstimator`](composing-models.md#models-whose-coordinate-is-bounded-not-linear)
imposes. Because *every* coordinate happens to land on the same transform, the natural
Jacobian is one diagonal of exponentials and `to_natural` is one dictionary comprehension.

**A curvature exponent is not capped at one.** Several implementations estimate a logit in
\((0,1)\). This one does not: \(\alpha > 1\) is a convex value function and therefore risk
seeking for gains *through curvature*, which is an empirical finding for individual subjects
and one half of a prediction the fourfold pattern is supposed to be able to fail. A bound
that forbids an outcome is not a prior, it is a censored test.

**`value_scale` is declared, never learned.** It divides amounts before the utility, so
\(\beta\) is dimensionless once it is stated. Deriving it from the study would make it
learned preprocessing: it would differ between training folds and the fitted \(\beta\) would
mean a different thing in each one. It appears in the model signature, so a fit can never be
read without it, and a study whose amounts are two orders of magnitude away from it gets a
`value_scale_mismatch` finding before the fit.

## The identifiability hazard, before the fit and after it

The inverse temperature and the value function's curvature trade off. A steeper utility with
a flatter softmax predicts nearly what a flatter utility with a sharper softmax predicts, and
this is the field's best-known identifiability problem. It is a property of the likelihood,
not a defect — but there are designs in which the trade-off is *exact*, and every one of
those is a cheap question about the design that can be answered before anything is fitted.
Following the precedent [`mix()`](composing-models.md#identifiability-is-reported-before-the-fit-not-after-it)
set for an unidentified mixture weight, each is a `describe()` finding:

```python
ProspectTheory().describe(study).findings
# [warning] unidentified_utility_curvature: gain_exponent is not identified by this design:
# every non-zero outcome on that side of zero has the same magnitude, so the exponent enters
# every row as one constant factor and the inverse temperature absorbs it exactly
```

| Finding | The design that produces it |
| --- | --- |
| `unidentified_utility_curvature` | one outcome magnitude on that side of zero |
| `weakly_identified_utility_curvature` | two magnitudes |
| `unobserved_loss_domain` | no negative outcome at all, so the loss branch has no gradient |
| `unidentified_loss_aversion` | no trial placing a gain against a loss |
| `unidentified_probability_weighting` | fewer interior probabilities than weighting parameters |
| `unidentified_discount_rate` | equal delays on every trial |
| `value_scale_mismatch` | amounts two orders of magnitude from the declared unit |

None is an error, because each is a statement about a design only its author can change —
the same standing as a knot placed outside the data's support.

Two things pick up the *soft* version of the hazard after the fit. A coordinate resting on
its box sets the usual `boundary_estimate` diagnostic. And the correlation the trade-off
actually produces is read straight off the fitted covariance:

```python
model.temperature_scale_correlation(fit, parameter="gain_exponent_log")
```

A magnitude near one means the pair is only *jointly* identified and neither number should
be reported alone.

### Fixing a parameter rather than estimating it against nothing

Any of the loss exponent, the loss aversion coefficient and the weighting elevation may be
declared, in which case it leaves the estimated coordinate entirely:

```python
gains_only = ProspectTheory(fixed_loss_exponent=1.0, fixed_loss_aversion=1.0)
one_parameter_prelec = ProspectTheory(fixed_weighting_elevation=1.0)
```

**Tying two parameters together is deliberately not offered.** The common \(\alpha = \beta\)
convention is a constraint on the coordinate, and every combinator that widens the coordinate
would have to be taught to respect it; a declared value is simply a parameter that is not
there.

## Composition: two combinators work, one does not

`smooth()` and `hierarchical()` compose over both families through the
[bounded-coordinate contract](composing-models.md#models-whose-coordinate-is-bounded-not-linear),
with nothing added to either combinator.

```python
from behavio.compose import hierarchical, smooth

per_animal = hierarchical(impulsivity, over="subject", parameters=("discount_rate_log",), scale=0.5)
drifting = smooth(
    impulsivity, over="session_order", knots=(0.0, 5.0), parameters=("discount_rate_log",)
)
per_animal_risk = hierarchical(risk, over="subject", parameters=("loss_aversion_log",), scale=0.4)
```

### A smooth discount rate means something, and the clock-block rule does not apply

"Does this animal get more impulsive across training?" is a real experimental question, and
`smooth(model, over="session_order", ...)` is the model of it.

The restriction the reinforcement-learning families needed **does not carry over**, and the
reason is worth being precise about. `smooth(BinaryQLearning(), over="trial")` is an error
because a learning rate that changed mid-session leaves the value trace unable to say which
of its values wrote which part of the trace. A discounting or prospect-theory trial's
likelihood reads that trial's amounts, delays and probabilities and *nothing else that
happened*, so `row_blocks` is `arange(n_rows)` — every row is its own block, exactly as it is
for a psychometric curve — and a coordinate may vary trial by trial. A within-session clock
is therefore **admissible** here, and a committed test asserts that the fit it produces runs
rather than raises.

Admissible is not the same as estimable. Two knots over a within-session counter is a model
of within-session drift and can be fitted; a knot per trial is not, and the roughness penalty
is the only thing standing between those two, exactly as for any other smooth model.

### A lapse is `mix()`, and it works here for the reason it refuses the agents

```python
from behavio.compose import UniformChoiceGuess, hierarchical, mix, smooth

impulsive_with_lapses = mix(
    TemporalDiscounting(value_scale=100.0), UniformChoiceGuess(), weight_bounds=(0.0, 0.3)
)

fit = impulsive_with_lapses.fit(study)
impulsive_with_lapses.to_natural(fit.estimates)
# {'discount_rate': ..., 'inverse_temperature': ..., 'lapse_rate': ...}
```

A reinforcement-learning agent declines a mixture because a lapse belongs *inside* its
recursion, where it can mix the emitted action while leaving the value update to see the
action that was taken. That argument does not apply here: these rows are independent, and a
lapse on a discounting model is an ordinary thing to want — the same lapse a psychometric
curve estimates.

`mix()` is gated on exactly that: **row independence**, which `row_blocks` reports, and not
on the presence of a linear predictor. This family has no linear predictor and no design
matrix, so the weight is not a cell of a predictor here — it is one extra column of the row
coordinate, `mixture_logit`, appended to the six logarithms. That is the same channel in the
other contract's vocabulary, so it composes with everything:

```python
per_animal_lapse = hierarchical(
    impulsive_with_lapses, over="subject", parameters=("mixture_logit",), scale=0.6
)
drifting_lapse = smooth(
    impulsive_with_lapses,
    over="session_order",
    knots=(0.0, 5.0),
    parameters=("mixture_logit",),
)
```

The reported coordinate is this family's own: `discount_rate`, not `discount_rate_log`, with
`lapse_rate` beside it. The mixture delegates to the model's `to_natural` rather than copying
its estimated names, because a bounded-coordinate model's coordinate is by contract a
transform of what it reports.

The gradient stays analytic. A mixture of two densities differentiates to the wrapped model's
own gradient scaled by the posterior probability that the row came from the model, plus the
weight's derivative through its logit; `tests/test_economic.py` checks both against central
differences, on the joint coordinate the solver searches and on the per-row coordinate an
outer combinator hands down.

One identifiability finding carries over exactly and one carries over in meaning only. See
[composing models](composing-models.md#identifiability-is-reported-before-the-fit-not-after-it)
for which is which and why.

## Ambiguity is not implemented, and here is what it would cost

Risk is implemented; **ambiguity is not**. It does not fall out of this module as cleanly as
the rest, and the reason is the one thing that made everything else cheap. The standard
ambiguity model replaces the stated probability with a subjective one,
\(p_{\text{sub}} = p - \eta A/2\) for an ambiguity level \(A\), and \(\eta\) is
**signed**: ambiguity *seeking* is as real a finding as ambiguity aversion. So it is the one
parameter in this family that cannot live on a logarithm, and the uniform "every coordinate
is a log" arithmetic — one diagonal Jacobian, one encode, one decode — would have to become
per-parameter. That plus an ambiguity-level column per option is the cost, and it is a
deliberate omission rather than an oversight.

## What a fitted discount rate does not establish

A discount rate is a property of a fitted curve on a declared delay coordinate, not a
property of a person. It changes with the delay unit, with the discount function, with the
amounts used, and with whether the softmax was allowed to absorb the amount scale. A fitted
\(\lambda\) above one is not evidence of loss aversion unless the design contained trials
that placed a gain against a loss; the model says so before the fit, and the finding should
be read rather than filtered.

## References

- Mazur, J. E. (1987). An adjusting procedure for studying delayed reinforcement. In
  *Quantitative Analyses of Behavior* (Vol. 5, pp. 55-73). Erlbaum.
- Kahneman, D., & Tversky, A. (1979). Prospect theory: an analysis of decision under risk.
  *Econometrica, 47*(2), 263-291.
- Tversky, A., & Kahneman, D. (1992). Advances in prospect theory: cumulative representation
  of uncertainty. *Journal of Risk and Uncertainty, 5*(4), 297-323.
- Prelec, D. (1998). The probability weighting function. *Econometrica, 66*(3), 497-527.
- Gonzalez, R., & Wu, G. (1999). On the shape of the probability weighting function.
  *Cognitive Psychology, 38*(1), 129-166.
