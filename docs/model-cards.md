# Model cards

These cards summarize the scientific contract of each first-party family. They are not
performance rankings. “Supported” refers to the evidence boundary in the
[capability matrix](methods/capability-matrix.md), not to every possible use of the model.

## At a glance

| Family | Scored event | Change represented | Multi-animal treatment | Main confusions |
| --- | --- | --- | --- | --- |
| Canonical baselines | Binary choice | None; fixed observable effects | Complete pooling | bias, stimulus, history, lapse |
| History GLM | Binary choice | Static coefficients | Complete or partial pooling | RL, latent state, omitted covariates |
| Smooth history GLM | Binary choice | Smooth coefficient paths | Complete or partial pooling | GLM-HMM, RL, clock choice |
| GLM-HMM | Binary choice | Discrete recurrent states | Complete pooling | smooth drift, history, state count |
| Binary RL | Binary choice | Recursive values and policy traces | Complete pooling | history, bias, lapse, reward schedule |
| Normative belief updating | Binary response to an exogenous observation | Recursive belief written by the task, not by the response | Complete or partial pooling | volatility versus coupling, response rule, observation versus response column |
| Scalar timing | A reproduced duration, or a binary long/short report | None by default; a drifting or per-subject clock is a combinator away | Complete or partial pooling | decision rule, target range, clock rate versus response bias |
| Patch leaving | A patch residence time, possibly right-censored | None by default | Complete or partial pooling | one patch type, undeclared censoring, threshold versus the theorem |
| Psychometric family | Binary choice | None; a fixed threshold, width, guess and lapse | Complete pooling | link choice, threshold convention, lapse versus slope |
| Temporal discounting | Binary choice between two delayed amounts | None by default; a smooth or per-subject discount rate is a combinator away | Complete or partial pooling | discount function, delay and amount units, rate versus choice noise |
| Prospect theory | Binary choice between two prospects | None by default; smooth and partially pooled cells are combinator expressions | Complete or partial pooling | curvature versus inverse temperature, weighting form, loss aversion without mixed gambles |
| Signal detection | Yes/no, rating, or response + confidence | None; fixed sensitivity and criteria | Complete pooling | extreme-rate corrections, equal versus unequal variance, meta-d' constraints |
| Multinomial logit | Categorical choice | Static or smooth per-category coefficients | Complete or partial pooling | availability, omissions, coding |
| Wiener DDM | Choice + response time | Within-decision accumulation; parameters fixed across trials | Complete or partial pooling | contaminants, RT origin, scale trade-offs |
| Session-varying Wiener DDM | Choice + response time | Smooth parameter paths between decisions | Single subject, complete, or partial pooling | across-trial paths versus within-decision dynamics |

Every family exposes filtered prediction and pointwise scoring. Generative families also
support simulation and design-specific recovery. Configuration-specific signatures prevent
fits from being reused under a different specification.

## Canonical binary baselines

**Classes:** `BiasOnly`, `Psychometric`, `Perseveration`, `WinStayLoseShift`

**Use when:** the intended explanation is one named observable regularity, or a richer
model needs a strong minimal comparator.

**Requires:** binary `choice`; models add stimulus, prior choice, or prior outcome fields
as named by their task. Histories reset at declared subject/session boundaries.

**Predicts:** one-step filtered binary choice probability.

**Evidence:** generative simulation, fit audit, prospective scoring, and recovery coverage.

**Does not establish:** a latent cognitive mechanism merely because its coefficient is
nonzero. Lapse is a fixed-support random-response mixture, not a general outlier process.

[Detailed assumptions](baselines.md)

## Static Bernoulli history GLM

**Class:** `BernoulliHistoryGLM`

**Use when:** binary choice depends on declared current covariates and finite observed
choice history with coefficients fixed over the study.

**Requires:** binary `choice`, numeric covariates, and explicit chronology.

**Predicts:** one-step filtered choice; current predictions may use earlier observed
choices, never later outcomes.

**Parameters:** intercept, covariate coefficients, and named lag coefficients; optional L2
penalty changes the uncertainty interpretation.

**Evidence:** supported common estimator contract, prospective evaluation, fit audits,
simulation, and parameter/model recovery.

**Does not establish:** stationarity of behaviour, causal effects of covariates, or reward
learning. Unmodeled smooth or discrete change can appear as history dependence.

[Detailed assumptions](modelling.md#static-bernoulli-history-glm)

## Smooth Bernoulli history GLM

**Expression:** `smooth(BernoulliHistoryGLM(...))`

**Use when:** one or more choice coefficients may change continuously over an explicit
learning clock.

**Requires:** a fixed clock, knots, covariates, histories, and smoothness chosen without
test outcomes.

**Predicts:** filtered future choice with the declared future-knot persistence rule.

**Parameters:** coefficient values on fixed knots, named `coefficient[clock=knot]`, plus
configured roughness and L2 penalties.

**Evidence:** prospective and exact-design recovery benchmarks, including weak-signal
confusions with static, GLM-HMM, and RL models.

**Does not establish:** a biological learning rule or discrete strategy transition. A
trajectory is conditional on clock and basis.

[Detailed assumptions](smooth-drift.md) · [Composing models](composing-models.md)

## Hierarchical static and smooth GLMs

**Expressions:** `hierarchical(BernoulliHistoryGLM(...), over="subject")`,
`hierarchical(smooth(BernoulliHistoryGLM(...)), over="subject")`

**Use when:** population structure and individual deviations are both part of the estimand.

**Requires:** multiple labelled subjects; the smooth model also requires a shared fixed
clock and knots.

**Predicts:** fitted deviations for represented subjects and an explicitly recorded
population plug-in for unseen subjects.

**Parameters:** population coefficients or paths in the wrapped model's own coordinate,
plus per-group deviations on the parameters named in `parameters=`, under fixed or
bounded-estimated per-parameter scales.

**Evidence:** pooling, subject-scale, trajectory, and prospective population benchmarks. A
PyMC NUTS backend is available for the fixed-scale static history GLM.

**Does not establish:** exchangeability across arbitrary labs, calibrated uncertainty for
every backend, or reliable person-level measurement.

[Static pooling](hierarchical-glm.md) · [Smooth pooling](hierarchical-smooth-glm.md) ·
[Composing models](composing-models.md) · [PyMC backend](pymc-backend.md)

## Bernoulli GLM-HMM

**Class:** `BernoulliGLMHMM`

**Use when:** binary choice may switch among a small number of recurrent policies with
state-specific input-driven Bernoulli emissions.

**Requires:** binary choice, declared inputs and lags, state count, stationary transition
semantics, and enough trials to support occupancy.

**Predicts:** one-step choice from filtered state probabilities. Smoothed state
probabilities are descriptive and cannot support a future prediction claim.

**Parameters:** state-specific GLM coefficients, initial-state probabilities, transition
matrix, and optional sticky Dirichlet self-transition prior.

**Evidence:** state-label alignment, occupancy and restart diagnostics, state-count nested
selection, prospective IBL example, and model recovery.

**Does not establish:** psychologically discrete states, stable label meaning, or
session-varying transitions. State count is a selected specification, not a discovered
natural kind.

[Detailed assumptions](glm-hmm.md)

## Binary reinforcement learning

**Classes:** `BinaryQLearning`, `BinaryRLAgent`

**Use when:** the claim concerns action values updated by observed rewards. Use the compact
class for a fixed reference specification and the composable class for declared learning,
forgetting, choice-kernel, policy, and reset components.

**Requires:** binary choice, reward in $[0,1]$, explicit reset columns, and action-
contingent reward probabilities for simulation.

**Predicts:** filtered choice before the current reward update.

**Parameters:** learning rate or asymmetric rates, inverse temperature, bias, and optional
forgetting, kernel, perseveration, or lapse quantities.

**Evidence:** recursive gradient checks, deterministic multistart audits, exact-design
recovery, and prospective competition with static and smooth choice models.

**Does not establish:** model-free neural learning, value persistence beyond the declared
reset boundary, or parameter identifiability under an uninformative reward schedule.

[Detailed assumptions](q-learning.md)

## Normative belief updating

**Classes:** `BetaBernoulliObserver`, `HierarchicalGaussianFilter`

**Use when:** the claim concerns a belief an observer holds about a changing world, formed
from what the **task** did rather than from what the subject chose, and read out through a
declared response rule.

**Requires:** an `observation` column of binary task outcomes and a binary response column,
which are different columns; preserved trial order with subject/session reset blocks; a
declared `BeliefResponse` component. The filter additionally declares its initial variances
and initial meta-belief, and a three-level filter declares or estimates a volatility coupling
and a meta-volatility.

**Predicts:** a filtered one-step-ahead binary response probability, formed from a belief
that has seen observations strictly before the trial and has never seen the response.

**Parameters:** the observer estimates `retention_logit` with an optional `prior_mean_logit`
and `prior_strength_log`; the filter estimates `tonic_volatility` and, at three levels,
`meta_volatility` and `volatility_coupling_log`, with `initial_belief` declarable. Response
parameters are the component's own — `inverse_temperature_log` with an optional
`choice_bias`, or `decision_noise_log`. A declared parameter leaves the model entirely and
its value enters the signature.

**Evidence:** clean-room implementation validated against four closed forms rather than
against another package — the exact static Beta-Bernoulli posterior mean asserted bitwise,
the leaky observer's closed-form asymptotic learning rate, the constant-volatility
Rescorla-Wagner reduction asserted to twelve decimal places, and a zero coupling making a
three-level filter's first two levels identical to a two-level filter's; a gradient exact in
the response coordinate and in the chain rule, checked against central differences;
deterministic multistart with every restart retained; the estimator conformance harness; and
design-specific recovery that includes an **asserted failure** for the third level's
volatility beside successful recovery of the two parameters the same study can see. All three
combinators compose, `mix()` included.

**Does not establish:** that a subject tracked volatility. On every reversal design tested,
displacing the third level's tonic volatility by a factor of \(e\) moves the whole belief
vector by under 0.1 in norm over 480 trials, so a number reported for it comes from the
restart and the box rather than from the data; `describe()` names it before the fit and a
two-level filter is the honest model for such a design. Nor a "learning rate": a tonic
volatility is a log step variance, and only where the belief is stationary does it reduce to
a rate. Agreement with another implementation is not validation, because published
implementations disagree about the update equations and all of them converge.

[Detailed assumptions](normative-belief.md)

## Parametric psychometric functions

**Class:** `PsychometricFunction`

**Use when:** the reportable quantity is a threshold and a width in stimulus units, and the
asymptotes on the two sides of the curve are not assumed equal.

**Requires:** a binary outcome, one declared numeric stimulus column, and declared bounds
for whichever of the guess and lapse rates is estimated rather than fixed by the task. The
Weibull link additionally requires strictly positive stimulus levels.

**Predicts:** a filtered binary choice probability at each stimulus level.

**Parameters:** threshold (or its logarithm for the Weibull), log width, and a bounded
logit for each estimated rate. `summarize()` returns the natural threshold, width, guess
rate, and lapse rate with intervals formed on their own scales.

**Evidence:** closed-form link identities checked against independently written
expressions, an analytic gradient checked against central differences, deterministic
multistart with every restart retained, design-specific recovery, and bit-for-bit parity
between `erf_two_gamma_probability` and the independent implementation committed beside the
IBL 2021 benchmark.

**Does not establish:** a sensory limit. A threshold is a property of a fitted curve on a
declared stimulus coordinate under a declared threshold convention; the Weibull's 50 % and
63 % conventions give different numbers for the same data. A lapse rate absorbs
stimulus-independent errors of every origin and is not evidence for any one of them.

[Detailed assumptions](psychometric-functions.md)

## Temporal discounting

**Class:** `TemporalDiscounting`

**Use when:** the trial is a choice between a smaller-sooner and a larger-later amount and
the reportable quantity is a discount rate.

**Requires:** a binary outcome, two amount columns, two non-negative delay columns, and a
declared `value_scale` giving the unit the amounts are in.

**Predicts:** a filtered binary choice probability, one per trial, as a softmax over the two
options' discounted values.

**Parameters:** `discount_rate_log` and `inverse_temperature_log`, both logarithms of
strictly positive quantities, reported as `discount_rate` and `inverse_temperature` with
delta-method standard errors. Mazur's hyperbola and the exponential are separate declared
discount functions and their rates are not interchangeable.

**Evidence:** the closed-form indifference point asserted for both discount functions at
three inverse temperatures, an analytic gradient checked against central differences,
deterministic multistart seeded from the design's own indifference rates with every restart
retained, and design-specific recovery. `smooth()` and `hierarchical()` compose over it.

**Does not establish:** impulsivity as a trait. A discount rate changes with the delay unit,
the discount function, the amounts used, and whether `value_scale` let the inverse
temperature absorb the amount scale. A design whose two delays are equal on every trial
cannot identify the rate at all, and says so before the fit.

[Detailed assumptions](economic-choice.md)

## Prospect theory

**Class:** `ProspectTheory`

**Use when:** the trial is a choice between two prospects and the reportable quantities are
value-function curvature, loss aversion, and probability weighting.

**Requires:** a binary outcome, one outcome and one probability column per option, optional
complementary-outcome columns for genuinely two-outcome prospects, and a declared
`value_scale`.

**Predicts:** a filtered binary choice probability as a softmax over two cumulative
prospect-theory valuations.

**Parameters:** logarithms of the gain exponent, loss exponent, loss aversion, Prelec
curvature, Prelec elevation, and the inverse temperature. Any of the loss exponent, loss
aversion, and weighting elevation may be declared fixed and then leaves the coordinate.
Exponents are not capped at one, because a convex value function is an empirical finding
rather than a modelling error.

**Evidence:** the fourfold pattern of risk attitudes asserted twice — from Tversky and
Kahneman's declared medians with no fitting, and from parameters recovered out of simulated
choices — a weighting function checked for monotonicity, anchoring, and its \(1/e\) fixed
point, an analytic gradient checked against central differences, and six-parameter recovery
on a titration design. `smooth()` and `hierarchical()` compose over it.

**Does not establish:** loss aversion from a design without a gain-against-loss trial, where
\(\lambda\) is the inverse temperature under another name; or a value-function curvature
from a design with one outcome magnitude, where the same is true of the exponent. Both are
reported as `describe()` findings before the fit. Ambiguity is not modelled.

[Detailed assumptions](economic-choice.md)

## Signal detection theory

**Classes:** `EqualVarianceSDT`, `UnequalVarianceSDT`, `MetaSDT`

**Use when:** the trial event is a detection or discrimination judgement and sensitivity
must be separated from response bias, or when confidence ratings are available and the
question is about the ROC or about metacognitive efficiency.

**Requires:** a binary `signal` indicator with a binary yes/no `response`;
`UnequalVarianceSDT` requires an ordered confidence rating with at least three declared
levels; `MetaSDT` requires response and confidence together with at least two declared
levels and a positive, finite type-1 d'. A table with an extreme hit or false-alarm rate
requires an explicitly declared `RateCorrection`.

**Predicts:** filtered yes/no probability (`EqualVarianceSDT`), a filtered probability over
rating categories (`UnequalVarianceSDT`), or a filtered probability over the joint
response-and-confidence cells (`MetaSDT`).

**Parameters:** d' and criterion; signal mean, log signal standard deviation, and ordered
rating criteria; type-1 d' and criterion with meta-d' and ordered type-2 criteria.

**Evidence:** reproduction of the Macmillan and Creelman worked example and of the
published m-alternative forced-choice table, an ideal-observer check in which meta-d'
recovers type-1 d' from evidence simulated outside the estimator, and design-specific
recovery for all three estimators.

**Does not establish:** a bias-free sensitivity measure when the equal-variance assumption
fails -- that is what the ROC is for -- nor "metacognitive ability" independent of the
task, criterion placement, and number of confidence levels offered. `MetaSDT` reports a
block-diagonal covariance because its two stages share no information; that is a property
of the published estimator, not an approximation.

[Detailed assumptions](signal-detection.md)

## Multinomial and omission-aware choice

**Class:** `MultinomialLogit`

**Use when:** the trial event has more than two actions, trial-specific availability, or an
omission category that belongs in the likelihood.

**Requires:** an explicit stable choice coordinate, reference category, design matrix, and
availability or omission semantics in `TaskSpec`.

**Predicts:** a filtered trial-by-category probability matrix.

**Parameters:** treatment-coded category-specific coefficients. `smooth()` gives each of
them a path in clock time and `hierarchical()` lets each of them vary by subject, so
drifting, pooled and drifting-pooled multinomials are expressions rather than classes; see
[composing models](composing-models.md).

**Evidence:** task validation, prospective scoring, simulation, and recovery tests, for the
static model and for all three composed cells.

**Does not establish:** sequential value learning or latent regimes. Categorical
calibration summaries remain limited.

[Detailed assumptions](multinomial.md)

## Static Wiener drift diffusion

**Class:** `WienerDriftDiffusion`

**Use when:** binary choice and response time are one joint observed event under a Wiener
first-passage account.

**Requires:** binary choice, positive eligible response time, explicit time unit and
origin, fixed noise scale, and -- when the model is mixed with a uniform response process --
a declared latency support for that process.

**Predicts:** a joint choice/response-time density and binary choice probability.

**Parameters:** drift regression coefficients, boundary separation, starting bias, and
non-decision time. A contaminant is not a parameter of this model: it is
`mix(model, UniformResponseGuess(...))`, which adds one mixing weight.

**Evidence:** analytic/numerical density checks, multistart fit audit, prospective IBL
example, contaminant and parameter recovery benchmarks.

**Does not establish:** a unique decomposition of drift, boundary, bias, and non-decision
time without adequate design; nor comparability with choice-only log scores.

[Detailed assumptions](drift-diffusion.md)

## Session-varying and partially pooled Wiener drift diffusion

**Expressions:** `smooth(WienerDriftDiffusion(...))`,
`hierarchical(WienerDriftDiffusion(...), over="subject")`,
`hierarchical(smooth(WienerDriftDiffusion(...)), over="subject")`

**Use when:** named DDM parameters may change smoothly across trials or sessions, or may
differ between animals, or both. The middle expression — a cohort with no longitudinal
hypothesis — had no hand-written class and is now an ordinary case.

**Requires:** all static DDM commitments; `smooth()` additionally requires a fixed
across-trial clock and knots, and `hierarchical()` requires at least two labelled groups
and a declared deviation scale.

**Predicts:** joint future choice/response-time outcomes. Represented subjects use their
fitted deviations; unseen-subject prediction plugs in the population by default or
integrates the fitted random effect through `predict_new_groups()`.

**Parameters:** the wrapped model's own coordinate, with each smoothed parameter replaced
by fixed-knot values named `parameter[clock=knot]`. Hierarchy renames nothing: the reported
coordinate is the population one, and deviations are read off the fit by group label under
fixed or bounded-estimated per-parameter scales.

**Evidence:** smooth, hierarchical, contaminant, predictive-uncertainty, and subject-scale
recovery benchmarks, plus a stored replay of the two deleted classes these expressions
replaced.

**Does not establish:** changing within-trial diffusion dynamics. The smooth clock indexes
decisions across the study, not time within a decision.

[Session-varying DDM](smooth-ddm.md) · [Partially pooled DDM](hierarchical-smooth-ddm.md) ·
[Composing models](composing-models.md)

## Scalar timing

**Classes:** `DurationReproduction`, `TemporalBisection`

**Use when:** the reportable quantity is a clock rate and a Weber fraction, from a reproduced
duration or from a binary report of which trained anchor a probe was more like.

**Requires:** `DurationReproduction` needs a strictly positive target column and a strictly
positive reproduction column. `TemporalBisection` needs declared anchors \(S < L\) — a fact
about the training procedure, never learned — a strictly positive probe column, a binary
report, and a declared `BisectionRule`.

**Predicts:** a `DensityPrediction` over the reproduced duration, tabulated on one shared
geometric grid, or a filtered binary report probability. `pointwise_log_prob` is the analytic
log density in both cases, so a fold's score never depends on the grid resolution.

**Parameters:** `clock_rate_log` and `weber_fraction_log`, both logarithms of strictly
positive quantities, reported as `clock_rate` and `weber_fraction`. Reproduction adds an
optional `central_tendency_log` — Vierordt's exponent, fixed at one by default. Bisection
reports `bisection_point` as a derived quantity whose description names the rule that
produced it. The two gain-free paradigms share one memory, so a study that runs both can ask
whether one Weber fraction describes them.

**Evidence:** the coefficient-of-variation identity and its exact inverse; the scalar property
asserted on simulated reproductions, on their regression against the target, and on the
tabulated density itself; Church and Deluty's geometric-mean bisection point with no fitting
anywhere in the assertion; the exact reparameterisation between the two decision rules, whose
clock rates differ by the ratio of their comparison durations and whose fits agree about
everything else; analytic gradients checked against central differences; design-specific
recovery of the clock rate, the Weber fraction and the exponent, and of one Weber fraction
across both paradigms; the estimator conformance harness; and hierarchical, smooth and mixed
recovery over the composed models.

**Does not establish:** a scalar clock. A Weber fraction from a narrow target range is not
evidence of one — the scalar property is a claim about how variability changes *with*
duration, and `describe()` reports `narrow_target_range` before the fit. A bisection point is
unreadable without its rule, which is why the rule is in the signature and why one anchor
pair cannot test it. A bisection clock rate is the clock and any response bias together, and
a central-tendency exponent is not estimable from bisection at all.

[Detailed assumptions](scalar-timing.md)

## Patch leaving

**Class:** `PatchLeaving`

**Use when:** the observation is a patch residence time and the reportable quantity is the
intake rate at which the animal gives up.

**Requires:** a patch yield column, a patch decay column, and a strictly positive residence
time, with a declared gain function. `travel_time_column` is optional and changes no
likelihood. `censoring_time_column` is optional and changes the likelihood a great deal: it
names the longest residence each row could have shown.

**Predicts:** a `DensityPrediction` of the **leaving time** on every row, censored or not,
because that is what the model claims about the row.

**Parameters:** `giving_up_rate_log` and `decision_noise_log`, reported as `giving_up_rate` —
in the study's own intake units per time — and `decision_noise`, a Weber fraction on that
rate. A declared travel time adds `marginal_value_rate`, `optimal_residence_time` and
`overstaying_ratio` as derived quantities: Charnov's prediction and the fitted animal's
departure from it.

**Evidence:** the hyperbolic optimum \(\sqrt{h\tau}\) to machine precision and the exponential
optimum against Charnov's implicit equation, both checked against a brute-force maximisation
of the long-run rate rather than against the root finder itself; a nearly noiseless forager
simulating the theorem's residence time; the predicted density integrating to one and
agreeing with the simulator; an analytic gradient checked against central differences with and
without censoring; a censored row scored against an independently written survival
probability; the upward bias from ignoring censoring measured rather than asserted; recovery
from a censored study; the estimator conformance harness; and hierarchical and smooth recovery
over a censored likelihood.

**Does not establish:** the marginal value theorem. A giving-up rate from a single patch type
is a residence time wearing a rate's units — with one patch type, "leave when the rate falls
to a threshold" and "leave after a fixed time" are the same model, and `describe()` reports
`unidentified_leaving_rule` before the fit. Nor optimal foraging: the model deliberately does
not constrain its threshold to \(R^{*}\), which is what lets `overstaying_ratio` measure the
distance. A censored row scored by the returned density rather than by `pointwise_log_prob`
is misscored, and `heavy_censoring` reports the share affected.

[Detailed assumptions](patch-leaving.md)

## Card-level release rule

A new first-party family should not be added here until its card can name the observed
event, required task fields, filtering semantics, parameters, numerical diagnostics,
simulation contract, exact-design recovery evidence, important competitors, and unsupported
claims. External models can satisfy the same standard without becoming Behavio internals;
see [Extend Behavio](extensions.md).
