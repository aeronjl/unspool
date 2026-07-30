# Psychometric functions

A psychophysicist asks for a **threshold**. `behavio.models.psychometric` therefore makes
the threshold the primary output, in stimulus units, with a confidence interval -- not an
intercept and a coefficient on the logit scale that the reader has to divide.

`PsychometricFunction` is the general family:

\[
\psi(x) = \gamma + (1 - \gamma - \lambda)\,
F\!\left(\frac{t(x) - t(\alpha)}{w} + z_{1/2}\right)
\]

with a **guess** rate \(\gamma\) and a **lapse** rate \(\lambda\) as two separate bounded
parameters -- the two-gamma form -- a threshold \(\alpha\) and a width \(w\) in stimulus
units, and a declared link \(F\).

!!! note "The logistic baseline is unchanged"
    `Psychometric` in [canonical baselines](baselines.md) remains exactly as it was:
    logistic link, intercept and slope. A lapse is added to it by `mix()` rather than by a
    class of its own, and the resulting model has *one* symmetric rate.
    `PsychometricFunction` is a new family beside them, not a replacement: it estimates a
    guess rate and a lapse rate **separately** inside the link, which a mixture cannot do
    with one weight. See
    [composing models](composing-models.md#mix-a-simpler-process-alongside-the-model) for
    why that line falls where it does.

## Links

| `PsychometricLink` | Core | Stimulus scale |
| --- | --- | --- |
| `LOGISTIC` | \(1/(1+e^{-z})\) | linear |
| `GAUSS` | \(\Phi(z)\) | linear |
| `ERF` | \((\operatorname{erf}(z)+1)/2 = \Phi(z\sqrt{2})\) | linear |
| `GUMBEL` | \(1 - \exp(-e^{z})\) | linear |
| `WEIBULL` | \(1 - \exp(-e^{z})\) | log |

```python
from behavio import PsychometricFunction
from behavio.models import PsychometricLink

model = PsychometricFunction(
    stimulus="signed_contrast",
    link=PsychometricLink.GAUSS,
    maximum_guess=0.2,
    maximum_lapse=0.2,
)
fit = model.fit(study)
summary = model.summarize(fit, level=0.95)
summary.threshold, summary.threshold_interval
summary.width, summary.guess_rate, summary.lapse_rate
summary.slope_at_threshold
```

## Conventions, because implementations disagree

**Threshold.** \(\alpha\) is the stimulus level at which the *link* reaches one half, so
\(\psi(\alpha)\) is exactly midway between \(\gamma\) and \(1 - \lambda\). That is what the
constant \(z_{1/2}\) is for: the logistic, Gauss and erf links already satisfy
\(F(0) = 1/2\), while the Gumbel link does not, so its location is shifted by
\(z_{1/2} = \log\log 2\).

Wichmann and Hill (2001) write the Weibull as \(1 - \exp(-(x/\alpha)^{\beta})\), whose
\(\alpha\) sits at \(1 - e^{-1} \approx 0.632\), not at one half. The equivalent
50 %-referenced form used here is \(1 - 2^{-(x/\alpha)^{1/w}}\). **Thresholds from the two
conventions are not interchangeable.**

**Stimulus scale.** The Weibull is the Gumbel on log stimulus. Its threshold is still in
stimulus units and must be positive; its width is in log-stimulus units, and its shape
parameter is \(1/w\).

**erf versus Gauss.** These describe the same curves with widths differing by
\(\sqrt{2}\). Both are provided because the erf parameterisation is the one the
International Brain Laboratory's released code uses.

**Bounds.** `maximum_guess` and `maximum_lapse` bound the rates that are *estimated* and
appear in the model signature, so a fit can never be read without its bounds. Either rate
may instead be **fixed** at a value the task determines:

```python
two_alternative = PsychometricFunction(fixed_guess_rate=0.5, maximum_lapse=0.1)
two_alternative.parameter_names  # ("threshold", "log_width", "lapse_logit")
```

A fixed rate leaves the parameter vector entirely rather than being estimated at a
boundary, and its summary reports a zero standard error and a degenerate interval -- the
honest description of a quantity that was declared rather than estimated.

## The estimated coordinate is not the natural one

The optimizer works on `log_width` and on bounded logits for the two rates, so it is
unconstrained in the interior. The **location** coordinate is the threshold itself for
every linear-stimulus link, so `fit.parameters["threshold"]` and
`fit.standard_error_map["threshold"]` are directly meaningful; for the Weibull it is
`log_threshold`, because a positive scale parameter's Wald interval belongs on the log
scale.

`summarize()` maps the whole coordinate back. Each interval is formed on the coordinate
that was estimated and then transformed, so an interval can never leave the parameter's
admissible range.

Every deterministic restart, its objective, and the selected optimum stay on
`PsychometricFitResult`. There is no unseeded randomness anywhere in the fit: the restart
schedule is derived from the observed stimulus levels and an empirical midpoint crossing.

That restart evidence is the whole reason the subclass exists: it is produced by fitting
and nothing short of refitting recomputes it. The class carried more once — a `link` field
and typed `threshold` / `width` / `guess_rate` / `lapse_rate` readers — and both are gone.
The link is a declared configuration already spelled out by `model_name` and
`model_signature`, and the four natural parameters live in `FitResult.derived`, where they
carry their own standard errors and intervals and where a consumer typed on plain
`FitResult` can see them. Read them with `fit.derived_value("threshold")`, or ask
`summarize()` for the whole coordinate at once.

## Curves that drift, and curves per animal

That unconstrained coordinate is also what makes the curve composable. A group deviation
must be Gaussian *somewhere*, and `log_width` and `lapse_logit` are where it is honest:

```python
from behavio.compose import hierarchical, smooth

sharpening = smooth(model, over="session_order", knots=(0.0, 5.0), parameters=("log_width",))
per_animal = hierarchical(model, over="subject", parameters=("threshold", "log_width"), scale=0.4)
```

Letting a **rate** vary by group carries a hazard that a threshold does not. A lapse rate at
the floor of its range has a logit at minus infinity, so its group deviations are unbounded
and the Laplace curvature at the optimum describes the box rather than the likelihood.
`describe(study)` says so before the fit, per group, whenever a subject shows no errors at
the easiest levels:

```python
hierarchical(model, over="subject", parameters=("lapse_logit",)).describe(study).findings
# [warning] unidentified_group_rate: lapse_rate is at the floor of its range for subject b ...
```

`fixed_lapse_rate=` remains the answer when the rate is known, and dropping the rate from
`parameters=` remains the answer when it is not identified per animal. See
[composing models](composing-models.md#a-rate-at-its-bound-is-reported-not-shrunk).

## Promoting the IBL benchmark model

`benchmarks/ibl2021_psychometrics` contains an independent implementation of the released
`erf_psycho_2gammas` together with the published Nelder-Mead restart schedule and its
penalty box. The published pipeline draws four of its five restarts from an *unseeded*
uniform distribution and so is not reproducible run to run; the benchmark keeps the same
restart schedule but derives it from an explicit seed.

That equation is now a first-class model. `PsychometricLink.ERF` with two gammas *is* the
released form under a different naming of the location:

| Released name | Behavio name |
| --- | --- |
| `bias` | `threshold` |
| `threshold` (the erf scale) | `width` |
| `lapse_low` | `guess_rate` |
| `lapse_high` | `lapse_rate` |

```python
from behavio.models import erf_two_gamma_probability

erf_two_gamma_probability(contrasts, bias=-4.0, threshold=18.0, lapse_low=0.06, lapse_high=0.11)
```

**The benchmark keeps its own implementation.** It reproduces a published *pipeline*,
including the optimiser quirks and penalty box that produced the published numbers; the
package model is a reusable *estimator* with declared bounds, deterministic restarts, and
retained diagnostics. Collapsing the benchmark onto the package would turn an independent
parity check into a tautology. Instead a committed test asserts that
`erf_two_gamma_probability` reproduces the benchmark's `erf_psycho_2gammas` bit for bit
across random parameter draws, so the two can no longer drift apart unnoticed.

## What a threshold does not establish

A threshold is a property of a fitted curve on a declared stimulus coordinate, not a
property of a sensory system. Changing the stimulus units, the link, or the threshold
convention changes the number. A lapse rate absorbs stimulus-independent errors of every
origin -- attention, motor slips, task disengagement -- and is not evidence for any one of
them; where those alternatives are scientifically plausible they should be modelled and
compared, not folded into \(\lambda\).
