# Choose a model by the claim

Do not begin with “Should I fit an RL model or a GLM-HMM?” Begin with the event you need to
predict, the future in which it must work, and the scientific contrast that could make its
mechanism informative. Model choice comes after those commitments.

<figure class="doc-figure doc-figure--wide" data-figure-kind="Conceptual">
  <img src="assets/model-choice-workflow.svg" alt="A decision workflow starts from whether the scored event is choice alone or joint choice and response time, routes choice models by observable structure, smooth change, discrete regimes, or reward updating, and finishes with a deployment boundary, matched alternatives, and recovery diagnostics.">
  <figcaption><strong>Decision order.</strong> A mechanism earns interpretation only after
  the observed event, deployment boundary, alternatives, and recovery design are fixed.</figcaption>
</figure>

## 1. Declare the observed event

| What is observed and scientifically relevant? | Start with | Do not compare directly with |
| --- | --- | --- |
| Binary choice | Binary baselines or Bernoulli history GLM | Joint choice/RT likelihoods |
| More than two actions or modeled omissions | `MultinomialLogit` | Binary models after silently dropping categories |
| Binary choice and response time | `WienerDriftDiffusion` family | Choice-only log scores |
| A binary response to an exogenous observation sequence | `BetaBernoulliObserver` or `HierarchicalGaussianFilter` | Value-learning agents, whose recursion is written by the response rather than by the observation |
| A reproduced duration | `DurationReproduction` | Choice likelihoods, and any score whose units are not a density on that duration |
| A patch residence time, possibly censored | `PatchLeaving` | Models that treat a session-truncated visit as a departure |
| A neural measurement | A companion neural model outside Behavio | A behavioural likelihood relabelled as neural evidence |

The `scored_columns` contract enforces this boundary. A probability for choice and a joint
density for choice and response time are not commensurable scores.

## 2. Declare where prediction must generalize

Choose the splitter before inspecting candidate performance.

| Intended use | Validation geometry | Meaning of success |
| --- | --- | --- |
| Later trials in the same session | Within-session rolling origin | Filtered near-future prediction after replaying the observed prefix |
| Later complete sessions for represented animals | Forward-session splits | Forecasting the same animals later in training |
| A new animal | Leave-subject-out | Population-level transfer without fitted subject effects |
| A new laboratory | Leave-lab-out | Transfer beyond complete labs in the observed sample |
| A historical deployment sequence | Historical-cohort splits | Performance for later-arriving cohorts under the declared order |

Whole-session leave-one-out is interpolation, not forecasting. A held-out lab does not by
itself imply a population-of-laboratories estimand.

## 3. Name the simplest live explanation

### Observable choice structure

Use `BiasOnly`, `Psychometric`, `Perseveration` or `WinStayLoseShift` when the question can
be expressed as a canonical behavioural summary; add a lapse to any of them with `mix()`.
Use `BernoulliHistoryGLM` when several declared covariates and reset-safe histories must be
estimated together. These models are scientific controls, not disposable preliminaries.

### Smooth change across learning

Use `smooth(model, over=...)` when a coefficient is expected to vary continuously over a
declared clock. Wrap that in `hierarchical(..., over="subject")` when individual paths
should deviate from a shared population path. Knots and clocks must be defined without future
outcomes; more flexible smoothness belongs inside nested training-only selection.

### Discrete latent regimes

Use `BernoulliGLMHMM` when switching among a small number of recurring choice policies is
the claim. Always include observable-history and smooth-change competitors. Select state
count inside training data, align labels for recovery, inspect occupancy, and use filtered
state probabilities for prospective claims.

### Reward-driven updating

Use `BinaryQLearning` for the compact reference agent or `BinaryRLAgent` for explicit
learning, forgetting, choice-kernel, policy, and reset components. A fitted learning rate
is not evidence for reward learning unless the actual reward schedule distinguishes the
agent from history, bias, lapse, and drift accounts under model recovery.

### Choice and decision time

Use `WienerDriftDiffusion` when accuracy and response time form one joint claim. The smooth
and hierarchical variants describe across-trial parameter paths; they do not change the
within-decision diffusion process into a learning model. Response-time origin, eligibility,
units, and contaminant support are part of the task contract.

### More than two actions or explicit omissions

Use `MultinomialLogit` when the stable categorical choice set is itself the target. It can
respect trial-specific availability and retain omissions as a modeled category, and it
composes: `smooth()` and `hierarchical()` supply drifting and per-subject versions of it
without a new class. The current RL, GLM-HMM, and DDM reference families are binary; do not
coerce a richer task into them merely for API convenience.

### A belief about a changing world

Use `behavio.models.belief.BetaBernoulliObserver` or
`behavio.models.belief.HierarchicalGaussianFilter` when the claim is about what the subject
*should* have believed given what the task showed it, and the response is a read-out of that
belief. The task's observation and the subject's response are different columns, and keeping
them apart is the whole distinction from a value-learning agent: the belief recursion is
written by the observation, so every row has a density of its own and a lapse is expressible
where it is not for a Q-learning agent.

A fitted volatility is not evidence that the subject tracked volatility. On reversal designs
the three-level filter's `meta_volatility` is not identified by binary responses at all;
`describe()` measures the study's own belief sensitivity and reports
`belief_insensitive_parameter` before the fit, and a two-level filter is the honest model for
such a design. See
[SDR-0062](decisions/0062-implement-normative-belief-updating-clean-room.md) for why this is
a clean-room implementation and which conventions it declares.

### A timed duration

Use `behavio.models.scalar_timing.DurationReproduction` when the animal reproduces a target
duration and the reportable quantity is a Weber fraction, and
`behavio.models.scalar_timing.TemporalBisection` when it reports which of two trained anchors
a probe was more like. Both estimate the same two parameters — a clock rate and a Weber
fraction — from the same scalar memory, so a study that runs both can ask whether one Weber
fraction describes them, which neither paradigm can ask alone.

A fitted Weber fraction is not evidence of a scalar clock unless the design tests durations
over a *range*: the scalar property is a claim about how variability changes with duration,
and `describe()` reports `narrow_target_range` when the tested durations are too close
together to see it. A bisection point is not readable without the decision rule that produced
it, which is why the rule is in the model's signature; see
[SDR-0060](decisions/0060-bisect-time-by-the-ratio-rule.md) for why the ratio rule is the
default and why one anchor pair cannot test it.

### Leaving a depleting patch

Use `behavio.models.patch_leaving.PatchLeaving` when the observation is a residence time and
the reportable quantity is the intake rate at which the animal gives up. Declare
`travel_time_column` and the fit additionally reports Charnov's optimum beside the threshold
it estimated, as an `overstaying_ratio`. Declare `censoring_time_column` whenever a session
can end while the animal is still in a patch; such a visit is not a leaving time and is
scored by its survival function instead.

A fitted giving-up rate is not evidence for the marginal value theorem unless the patches
*differ*. With one patch type, "leave when the intake rate falls to a threshold" and "leave
after a fixed time" make identical predictions, and `describe()` reports
`unidentified_leaving_rule`. See
[SDR-0061](decisions/0061-fit-patch-leaving-as-a-hazard-not-as-the-marginal-value-theorem.md)
for why the theorem is a benchmark this family is read against rather than the likelihood it
fits.

## 4. Decide whether pooling is part of the claim

| Scientific unit | Suitable starting point | Prediction for unseen subjects |
| --- | --- | --- |
| One subject | Non-hierarchical model | Not applicable |
| Population-average effect | Hierarchical static or smooth GLM | Population plug-in, explicitly labelled |
| Individual trajectories | Hierarchical smooth GLM, multinomial, or DDM | Evaluate represented and unseen subjects separately |
| Stable individual measure | Paired occasion estimates plus reliability analysis | Not a substitute for a joint trial-level reliability model |

Independent per-subject fits can be useful descriptive objects, but they neither pool weak
animals nor propagate estimation error into a population analysis. Conversely, a pooled
group effect does not establish reliable individual differences.

## 5. Build the candidate set around confusions

A useful candidate set contains mechanisms that can plausibly imitate one another in the
actual design:

- history GLM versus RL for reward-correlated choice repetition;
- smooth drift versus GLM-HMM for gradual versus apparently abrupt change;
- lapse versus contaminant DDM for unusual choices or response times;
- complete pooling versus partial pooling for individual variation;
- static versus smooth parameters for longitudinal change.

Candidates must score the same observed event on the same held-out rows. Hyperparameters,
state count, and smoothness are candidates too, so select them inside an inner loop.

## 6. Require the evidence stack

Before interpreting a fitted mechanism, retain:

1. source, cohort, unit, and task validation;
2. numerical fit or posterior diagnostics;
3. prospective predictions and pointwise scores;
4. calibration or posterior-predictive checks;
5. exact-design parameter recovery for reported parameters;
6. exact-design model recovery for the candidate set;
7. sensitivity to defensible preprocessing, prior, and likelihood choices; and
8. explicit failures, exclusions, and claim limits.

Passing one layer does not certify the others. A converged optimizer can estimate an
unrecoverable parameter; a recoverable model can still predict poorly; the best candidate
can still be an inadequate model of the data.

## Unsupported combinations

The current package does not yet provide a first-party multinomial RL agent, collapsing-
bound or race/LBA likelihood, hierarchical GLM-HMM, session-varying GLM-HMM transitions,
or joint trial-level Bayesian reliability model. Use a conforming external estimator when
one exists, and preserve these same task, scoring, prediction, recovery, and provenance
contracts. The [capability matrix](methods/capability-matrix.md) is the authoritative
support boundary.

Next: inspect the common-format [model cards](model-cards.md), then choose a
[worked study](tutorials/index.md) with the closest observed event and deployment geometry.
