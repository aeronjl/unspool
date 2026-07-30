# SDR-0061: Fit patch leaving as a hazard, and read it against the marginal value theorem

- **Status:** Accepted
- **Date:** 2026-07-30
- **Related decisions:** [SDR-0060](0060-bisect-time-by-the-ratio-rule.md)

## Context

`docs/philosophy.md` has said since the package was renamed that "a foraging bout" is in
scope. Nothing implemented it. The obvious content to implement is Charnov's (1976) marginal
value theorem: an animal exploiting a depleting patch in an environment whose patches are
separated by a travel time \(\tau\) should leave when the patch's instantaneous intake rate
has fallen to the environment's long-run average rate,

\[
g'(t^{*}) = R^{*} = \max_{t} \frac{g(t)}{\tau + t}.
\]

Two design questions had to be answered before any of it could be written.

**Is the theorem the model?** The MVT is *normative*: it says what an optimal forager would
do. It is not a likelihood, and it has no free parameters — given a gain function and a
travel time, \(t^{*}\) is determined. A "marginal value theorem model" fitted to data would
therefore have nothing to fit, or would have to smuggle in free parameters that the theorem
does not contain. Meanwhile the most replicated finding in the patch-leaving literature is
that animals **overstay**: they leave at an intake rate below \(R^{*}\).

**What shape is the observable?** A patch visit produces a residence time, not a choice. The
animal is in the patch and is, moment by moment, deciding whether to go, so the fitted object
is a leaving decision per unit residence time — a hazard. A session that ends while the
animal is still in a patch produces a right-censored observation, not a leaving time. The
package's audit had separately flagged a survival/hazard gap for response times, so the
question of whether one hazard machinery serves both had to be answered rather than deferred.

## Decision

**The theorem and the model are separate objects.** `marginal_value_rate` and
`marginal_value_residence_time` are the closed form, with no fitting anywhere near them.
`PatchLeaving` is a behavioural model of when an animal actually goes, fitted to residence
times without assuming that the threshold it estimates is the optimal one. A fit that
declares `travel_time_column` gains three derived quantities —
`marginal_value_rate`, `optimal_residence_time` and `overstaying_ratio` — so overstaying is a
number the fit *reports* rather than an assumption baked into the model reporting it. A
module that fitted the MVT could not measure a departure from it.

**The likelihood is a threshold-crossing hazard.** The animal carries a giving-up rate
\(\theta\) with multiplicative noise, \(\log \Theta \sim \text{Logistic}(\log\theta, s)\), and
leaves at the first moment the intake rate falls below it. Because \(g'\) is strictly
decreasing this is closed form throughout:

\[
S(t) = \frac{\sigma(u(t))}{\sigma(u(0))},\qquad
u(t) = \frac{\log g'(t) - \log\theta}{s},\qquad
\lambda(t) = -u'(t)\bigl(1-\sigma(u(t))\bigr).
\]

Two estimated parameters, both positive, both logarithms: \(\theta\) is in intake units per
time and \(s\) is dimensionless — a Weber fraction on the intake rate. As \(s \to 0\) the
leaving time converges on the deterministic threshold crossing, so setting \(\theta = R^{*}\)
recovers \(t^{*}\) exactly, which is what the simulator is validated against.

**Censoring is declared, and its arithmetic is shared.** `censoring_time_column` names each
row's longest observable residence; such a row is scored by \(\log S(c)\). Leaving it
undeclared asserts that every duration ran to its event, and `describe()` reports
`undeclared_censoring` when the residence times pile up on a common maximum, which is what
that assertion looks like when it is false.

**One hazard machinery serves patch leaving and response times only in part, and the part it
serves is `behavio.models._kernels.hazard`.** What is *not* shared is where the density comes
from: patch leaving is hazard-first, because the leaving rate is the decision and the density
is derived from it, whereas a response-time model is density-first, because a first-passage
density is the primitive and its hazard is a description computed afterwards. Forcing a
Wiener first-passage density into a parametric-hazard family would be a claim about drift
diffusion that drift diffusion does not make. What *is* shared is the censoring selection —
which rows are scored by a density and which by a survival, how the gradient follows the same
selection, and whether a duration equal to its limit is an event — and that is written once
in the kernel. A censored variant of `WienerDriftDiffusion` would import that kernel and
supply a survival function; it would not import a hazard family.

## Consequences

**A patch-leaving fit is falsifiable against the theorem rather than by it.** The
`overstaying_ratio` is \(R^{*}/\hat\theta\); above one is an animal that leaves at a lower
intake rate than optimal.

**The environment, not the patch, sets the threshold.** `marginal_value_rate` solves the
fixed point over every patch type present, weighted by the frequency with which it appears,
so a heterogeneous study gets one rate and a different optimal residence time per type.
`describe()` reports `heterogeneous_environment` so that a reader does not take the mean of
several optima for one.

**A single-patch-type study cannot test the theorem at all.** If every patch depletes
identically then \(\log g'(t)\) is one fixed monotone function of \(t\), and "leave when the
intake rate falls to \(\theta\)" predicts exactly what "leave after \(t_\theta\) seconds"
predicts. The content of the MVT is that *one rate threshold governs patches of different
richness and depletion*, and that is invisible in such a design however many visits it
contains. `describe()` reports `unidentified_leaving_rule`, and it is the most important
finding this family has.

**One thing does not fit the prediction contract, and it is reported rather than worked
around.** `predict` returns a `DensityPrediction` of the leaving time, which is what the
model claims about every row. A censored row's *score* is a survival probability, and no
member of `ModelPrediction` can carry "the probability the event is still to come". So
`pointwise_log_prob` and `DensityPrediction.observed_log_density` agree on uncensored rows
and deliberately disagree on censored ones; the first is the likelihood and the second is the
prediction. A consumer that scores the density directly instead of asking the model will
misscore exactly the censored rows, and `describe()` says so through `heavy_censoring`.

**`compare_models` cannot rank two candidates whose only prediction is an unlabelled
density.** It reports a Brier score beside the log score, a Brier score needs a discrete
margin, and a residence time has none — so it raises `UnscoreableByBrier` rather than
inventing a number. `evaluate_splits` reports the log score, which *is* defined. What is
missing is a way to ask `compare_models` for the log-score half alone, and that is a gap in
the comparison layer rather than in this family.

## Alternatives considered

**Fit the MVT directly, with the giving-up rate constrained to \(R^{*}\).** Rejected. It has
no free parameters, so it is not an estimator; and constraining the threshold to the optimum
makes overstaying unmeasurable, which discards the one result the literature is about. The
constrained model is still reachable as a *comparison*: fix `giving_up_rate` at
`marginal_value_rate` and score it against the free fit.

**Make the noise a per-moment decision noise rather than a threshold noise.** Considered and
rejected as the primitive, though the two are close. A per-moment logistic leaving rule of
the form \(\lambda(t) = \lambda_0 \exp((\log\theta - \log g'(t))/\kappa)\) is a Gompertz
hazard, which is a plausible shape — but \(\theta\) and \(\lambda_0\) enter it only through
\(\log\lambda_0 + \log\theta/\kappa\) and are therefore **exactly** confounded. Repairing that
would mean declaring \(\lambda_0\) as a unit constant, which makes the reported threshold
depend on an arbitrary declaration. The threshold formulation has no such confound, gives the
same rising hazard, and reads directly as "the intake rate at which this animal gives up".

**Score censored rows by their density and ignore the difference.** Rejected. It biases the
threshold in a direction that is easy to state and easy to miss — a truncated visit read as a
departure looks like a forager that left sooner, so the fitted giving-up rate moves up — and
`tests/test_patch_leaving.py::test_ignoring_censoring_biases_the_giving_up_rate_upwards`
measures it.

**Support left truncation and interval censoring.** Deferred, and refused explicitly rather
than approximated. An animal that entered a patch before recording began has a *conditional*
likelihood, not a marginal one; scoring it as if it did not would be the exact failure the
censoring machinery exists to prevent.

**One gain function.** Rejected. Two are offered and their decay parameters are not
interchangeable: the exponential \(g(t) = A(1-e^{-\rho t})\) is the standard depleting-patch
schedule and its optimum solves an implicit equation, and Holling's disc equation
\(g(t) = At/(t+h)\) has the exact optimum \(t^{*} = \sqrt{h\tau}\) with
\(R^{*} = A/(\sqrt{h}+\sqrt{\tau})^2\). The second exists partly because an exact closed form
is what the root finder can be validated against without validating it against itself.

## Revisit trigger

Revisit when a censored response-time model is written, because that is the test of whether
the kernel's boundary was drawn in the right place: if the new model needs the censoring
selection and nothing else from `behavio.models._kernels.hazard`, the split was correct; if
it needs a hazard family, it was not. Revisit sooner if `compare_models` gains a log-score-only
mode, because these families would then flow through it and the `UnscoreableByBrier` refusal
above would stop being a consequence of this decision.
