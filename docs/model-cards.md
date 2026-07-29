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
| Multinomial logit | Categorical choice | Static coefficients | Complete pooling | availability, omissions, coding |
| Wiener DDM | Choice + response time | Within-decision accumulation; optionally smooth across-trial parameters | Single subject or partial pooling | contaminants, RT origin, scale trade-offs |

Every family exposes filtered prediction and pointwise scoring. Generative families also
support simulation and design-specific recovery. Configuration-specific signatures prevent
fits from being reused under a different specification.

## Canonical binary baselines

**Classes:** `BiasOnly`, `Psychometric`, `Perseveration`, `WinStayLoseShift`,
`LapsePsychometric`

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

**Class:** `SmoothBernoulliHistoryGLM`

**Use when:** one or more choice coefficients may change continuously over an explicit
learning clock.

**Requires:** a fixed clock, knots, covariates, histories, and smoothness chosen without
test outcomes.

**Predicts:** filtered future choice with the declared future-knot persistence rule.

**Parameters:** coefficient values on fixed knots plus configured roughness and L2
penalties.

**Evidence:** prospective and exact-design recovery benchmarks, including weak-signal
confusions with static, GLM-HMM, and RL models.

**Does not establish:** a biological learning rule or discrete strategy transition. A
trajectory is conditional on clock and basis.

[Detailed assumptions](smooth-drift.md)

## Hierarchical static and smooth GLMs

**Classes:** `HierarchicalBernoulliHistoryGLM`,
`HierarchicalSmoothBernoulliHistoryGLM`

**Use when:** population structure and individual deviations are both part of the estimand.

**Requires:** multiple labelled subjects; the smooth model also requires a shared fixed
clock and knots.

**Predicts:** fitted deviations for represented subjects and an explicitly recorded
population plug-in for unseen subjects.

**Parameters:** population coefficients or paths, subject deviations, and fixed or bounded-
estimated subject scales depending on configuration.

**Evidence:** pooling, subject-scale, trajectory, and prospective population benchmarks. A
PyMC NUTS backend is available for the fixed-scale static history GLM.

**Does not establish:** exchangeability across arbitrary labs, calibrated uncertainty for
every backend, or reliable person-level measurement.

[Static pooling](hierarchical-glm.md) · [Smooth pooling](hierarchical-smooth-glm.md) ·
[PyMC backend](pymc-backend.md)

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

## Multinomial and omission-aware choice

**Class:** `MultinomialLogit`

**Use when:** the trial event has more than two actions, trial-specific availability, or an
omission category that belongs in the likelihood.

**Requires:** an explicit stable choice coordinate, reference category, design matrix, and
availability or omission semantics in `TaskSpec`.

**Predicts:** a filtered trial-by-category probability matrix.

**Parameters:** treatment-coded category-specific coefficients.

**Evidence:** task validation, prospective scoring, simulation, and recovery tests.

**Does not establish:** sequential value learning or latent regimes. Current categorical
calibration and hierarchical extensions remain limited.

[Detailed assumptions](multinomial.md)

## Static Wiener drift diffusion

**Class:** `WienerDriftDiffusion`

**Use when:** binary choice and response time are one joint observed event under a Wiener
first-passage account.

**Requires:** binary choice, positive eligible response time, explicit time unit and
origin, fixed noise scale, and declared contaminant support if the mixture is enabled.

**Predicts:** a joint choice/response-time density and binary choice probability.

**Parameters:** drift regression coefficients, boundary separation, starting bias,
non-decision time, and optional contaminant probability.

**Evidence:** analytic/numerical density checks, multistart fit audit, prospective IBL
example, contaminant and parameter recovery benchmarks.

**Does not establish:** a unique decomposition of drift, boundary, bias, and non-decision
time without adequate design; nor comparability with choice-only log scores.

[Detailed assumptions](drift-diffusion.md)

## Smooth and hierarchical Wiener drift diffusion

**Classes:** `SmoothWienerDriftDiffusion`,
`HierarchicalSmoothWienerDriftDiffusion`

**Use when:** selected DDM parameters may change smoothly across trials or sessions, with
optional shrunken animal-specific deviations.

**Requires:** all static DDM commitments plus a fixed across-trial clock, knots, changing
parameter set, and population prediction policy.

**Predicts:** joint future choice/response-time outcomes. Represented subjects use their
paths; unseen-subject prediction can integrate or plug in population structure through
the documented method.

**Parameters:** fixed-knot drift, boundary, or starting-bias paths; stationary non-decision
time; optional subject deviation scales.

**Evidence:** smooth, hierarchical, contaminant, predictive-uncertainty, and subject-scale
recovery benchmarks.

**Does not establish:** changing within-trial diffusion dynamics. The smooth clock indexes
decisions across the study, not time within a decision.

[Smooth DDM](smooth-ddm.md) · [Hierarchical smooth DDM](hierarchical-smooth-ddm.md)

## Card-level release rule

A new first-party family should not be added here until its card can name the observed
event, required task fields, filtering semantics, parameters, numerical diagnostics,
simulation contract, exact-design recovery evidence, important competitors, and unsupported
claims. External models can satisfy the same standard without becoming Behavio internals;
see [Extend Behavio](extensions.md).
