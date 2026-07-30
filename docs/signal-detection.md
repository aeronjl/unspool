# Signal detection theory

Detection and discrimination is the most common on-ramp into quantitative behaviour, and
its estimators are small enough that everyone reimplements them. The disagreements between
those reimplementations are not in the arithmetic; they are in the conventions. Behavio's
`behavio.models.sdt` therefore states every convention it uses, records the analytic
choices on the objects it returns, and exposes the fitted models through the same
estimator contract as every other family.

| Estimator or summary | Scored event | Question |
| --- | --- | --- |
| `equal_variance_summary` | closed-form summary | d', c, c', beta, A', B''<sub>D</sub> for one table |
| `EqualVarianceSDT` | yes/no response | fitted d' and criterion with standard errors |
| `forced_choice_d_prime` | closed-form summary | d' from m-alternative proportion correct |
| `z_roc_summary` | closed-form summary | z-ROC line, d<sub>a</sub>, slope, area |
| `UnequalVarianceSDT` | confidence rating | maximum-likelihood unequal-variance ROC |
| `MetaSDT` | response and confidence jointly | meta-d', M-ratio, M-diff |

Everything that is *fitted* is a `BehaviourEstimator` with a matching simulator, so it
enters `run_parameter_recovery` like any other model. Everything that is a *closed-form
summary of a table* is a plain function returning a frozen record. A quantity that can be
written down from four counts does not need an optimizer, a signature, or a fit identity,
and pretending otherwise would make the catalogue less honest rather than more uniform.

## Equal-variance yes/no

```python
from behavio import EqualVarianceSDT, equal_variance_summary
from behavio.models import DetectionCounts, RateCorrection

summary = equal_variance_summary(
    DetectionCounts(hits=67, misses=33, false_alarms=16, correct_rejections=84)
)
summary.d_prime  # 1.4344
summary.criterion  # 0.2773
summary.beta  # 1.4884
```

The model places noise at \(-d'/2\) and signal at \(+d'/2\), so

\[
H = \Phi(d'/2 - c), \qquad F = \Phi(-d'/2 - c), \qquad
d' = z(H) - z(F), \qquad c = -\tfrac{1}{2}\bigl(z(H) + z(F)\bigr).
\]

A **positive criterion is conservative**: a bias against responding "yes". The relative
criterion is \(c' = c/d'\), and \(\ln\beta = c\,d'\) exactly.

!!! note "Natural log or base ten"
    Published tables report \(\beta\), \(\ln\beta\) and \(\log_{10}\beta\) interchangeably,
    frequently without saying which. `SignalDetectionSummary` carries `beta`, `log_beta`
    (natural) and `log10_beta` under unambiguous names rather than one ambiguous field.

`A'` follows Grier (1971) and `B''`<sub>D</sub> follows Donaldson (1992). Both are computed
from the same rates as the parametric indices, so one `CorrectedRates` record describes the
whole summary.

## Corrections for extreme rates are declared, never inferred

A hit rate of one or a false-alarm rate of zero makes d' infinite. Two standard repairs
exist and **they disagree**:

| `RateCorrection` | Source | What it does |
| --- | --- | --- |
| `NONE` | — | nothing; an extreme rate yields an infinite d' that stays visible |
| `LOG_LINEAR` | Hautus (1995) | adds 0.5 to all four cells of **every** table |
| `ONE_OVER_2N` | Macmillan & Kaplan (1985) | replaces **only** a rate of 0 or 1 by \(1/2N\) or \(1-1/2N\) |

The difference matters twice over. `LOG_LINEAR` changes estimates for tables that needed no
repair at all, so it is not a fallback that quietly does nothing when unused;
`ONE_OVER_2N` leaves those tables alone but is discontinuous at the boundary.

```python
rates = detection_rates(counts, correction=RateCorrection.LOG_LINEAR)
rates.correction  # RateCorrection.LOG_LINEAR
rates.correction_applied  # True
rates.is_degenerate  # False
```

`CorrectedRates` records both the correction that was *declared* and whether it actually
*changed anything*, so a corrected rate can never be read as an uncorrected one. Fitting a
degenerate table without declaring a correction raises rather than silently repairing:

```python
EqualVarianceSDT().fit(study)
# ModelDataError: an extreme hit or false-alarm rate makes d' infinite; declare
# RateCorrection.LOG_LINEAR ... or RateCorrection.ONE_OVER_2N ... rather than leaving
# the choice implicit
```

This is exactly the kind of analytic choice
[sensitivity analysis](sensitivity-analysis.md) exists to interrogate: run the same study
under both corrections and read the spread.

## Forced choice

Two-alternative forced choice has the exact relation \(d' = \sqrt{2}\,z(P_c)\). That
shortcut is **wrong** for more than two alternatives, which needs the Green and Swets
integral

\[
P_c = \int \varphi(x - d')\,\Phi(x)^{m-1}\,\mathrm{d}x
\]

inverted numerically. `forced_choice_d_prime` dispatches on the number of alternatives and
the two routes agree exactly at \(m = 2\):

```python
forced_choice_d_prime(0.75, n_alternatives=2)  # 0.9539
forced_choice_d_prime(0.75, n_alternatives=3)  # 1.4338
forced_choice_d_prime(0.75, n_alternatives=4)  # 1.6822
```

## Unequal variance and the ROC

From confidence ratings, the z-ROC line is \(z(H) = a + s\,z(F)\) with slope
\(s = \sigma_N/\sigma_S\), so the signal standard deviation is \(1/s\). Then

\[
d_a = \frac{\sqrt{2}\,a}{\sqrt{1 + s^2}}, \qquad A_z = \Phi\!\left(d_a/\sqrt{2}\right).
\]

`z_roc_summary` fits that line by ordinary least squares on the observed operating points,
which is the classical estimator. `UnequalVarianceSDT` fits the same model by maximum
likelihood over the whole rating table, with the criteria parameterised as a first
criterion plus positive gaps so ordering can never be violated during optimization. These
are two different estimators of the same quantities and will not agree exactly; both are
provided, and the difference between them is information.

```python
model = UnequalVarianceSDT(ratings=(1, 2, 3, 4, 5, 6))
fit = model.fit(study)
fit.d_a, fit.z_roc_slope, fit.signal_sd, fit.area_under_curve
```

The reported `empirical_area` on a `ZRocSummary` is the trapezoid over the observed
operating points. It is a chord approximation to a concave curve and therefore always
smaller than the binormal `area_under_curve`; both are reported so the gap stays visible.

## Meta-d'

Meta-d' is the type-1 sensitivity a type-1 observer would have needed in order to produce
the observed confidence data. Getting it right is entirely a matter of getting the
constraints right.

* Type-1 d' and criterion are estimated first, in closed form, and then **held fixed**.
* The criterion is held at a fixed *relative* position, so the criterion used inside the
  meta model is \(c_{\mathrm{meta}} = \text{meta-}d' \times c/d'\). Holding \(c\) rather
  than \(c'\) is a common and consequential error.
* The likelihood is the confidence distribution **conditional on stimulus and type-1
  response**, which is what makes the fit independent of type-1 performance.
* Type-2 criteria are parameterised as positive gaps outward from the criterion, so they
  can never cross it and can never fall out of order.

```python
from behavio import MetaSDT

model = MetaSDT(confidence_levels=(1, 2, 3, 4))
fit = model.fit(study)
fit.derived_value("meta_d_prime"), fit.derived_value("m_ratio"), fit.derived_value("m_diff")
fit.derived_values  # the type-2 criteria arrive as type2_criterion_no_k / _yes_k
```

`MetaSDT` returns a plain `FitResult`. It used to return a `MetaSDTFitResult` whose
properties renamed entries of `derived`; the numbers were always in `derived`, so the
subclass only decided who was allowed to read them by name. The one structured record it
carried, the corrected type-1 `rates`, is a deterministic function of the study and the
model's declared `correction` and now lives on the model as `model.type1_summary(study)`.
The two rates themselves ride along in `derived`, and whether the declared correction
actually fired stays visible in the fit's diagnostic message.

`MetaSDT` scores the **joint** response-and-confidence cell, so its prediction is a
categorical distribution over `2K` cells named by the composite categories
`(response, confidence)` — `(0, 3)` is "responded no at confidence 3" — under the declared
factorisation `("response", "confidence")`, and its `scored_columns` are
`("response", "confidence")`. Those cells used to be labelled with the strings `"no-k"`
and `"yes-k"`, which every caller wanting one margin had to parse apart; ask the
prediction instead:

```python
prediction = model.predict(study, fit)
prediction.marginal("response")  # P(response), summing over confidence
prediction.marginal("confidence")  # P(confidence), summing over response
```

Marginalising is exact rather than approximate: the cells of one row partition that row's
probability. Its simulator draws the response from
the type-1 model and then the confidence bin from the meta model's conditional
distribution, which is precisely the factorisation the likelihood uses, so simulate-and-
refit recovery tests the estimator rather than a different model.

Because the two stages share no information, the reported covariance is block diagonal by
construction. That is a property of the published estimator, not an approximation
introduced here.

!!! warning "Zero confidence cells"
    The released MATLAB implementation pads every cell when any confidence cell is empty.
    Behavio does not: an empty cell contributes nothing to a conditional likelihood and
    needs no repair. The *type-1* table is a different matter, and its treatment is
    governed by the estimator's declared `correction`.

### Validating a meta-d' implementation

The check that separates a correct implementation from an incorrect one is a
metacognitively ideal observer: sample continuous evidence, respond by comparing it to the
criterion, and rate confidence by the distance of the *same* sample from that criterion.
Such an observer has no metacognitive inefficiency at all, so meta-d' must equal type-1 d'
and the M-ratio must be one. Behavio's test suite simulates that observer directly from
continuous evidence -- not from the estimator's own generative model -- and requires an
M-ratio of one to within sampling error, then requires that corrupting only the confidence
read-out lowers meta-d' while leaving type-1 d' untouched.

## What signal detection theory does not establish

d' separates sensitivity from bias *under the model's distributional assumptions*. An
equal-variance d' computed from a task whose evidence distributions are unequal in variance
is not a bias-free sensitivity measure; that is what the ROC is for. A meta-d' below d' is
evidence of metacognitive inefficiency only if the type-1 model is adequate, and an M-ratio
is not a measure of "metacognitive ability" independent of the task, the criterion
placement, or the number of confidence levels the participant was offered.
