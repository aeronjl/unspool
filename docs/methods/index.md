# Methods catalog

Behavio groups methods by the explanation they offer for change. Availability is not the
same as validation: consult the [capability matrix](capability-matrix.md) and each model's
recovery evidence.

If you are starting an analysis rather than browsing implementations, use the
[model-choice guide](../model-choice-guide.md) first. The [model cards](../model-cards.md)
then state each family's required observations, prediction semantics, evidence, and
unsupported uses in a common format.

<figure class="doc-figure doc-figure--wide" data-figure-kind="Conceptual">
  <img src="../assets/model-atlas.svg" alt="Five conceptual trajectories comparing a static GLM, smooth drift, GLM-HMM regimes, Q-learning value updating, and a drift-diffusion tendency across early-to-late learning.">
  <figcaption><strong>Model-family atlas.</strong> Each family encodes a different explanation of behavioural change. The curves are conceptual signatures, not fitted data or evidence that any family is correct.</figcaption>
</figure>

## Fixed threshold, sensitivity, and bias

`PsychometricFunction` estimates a threshold and a width in stimulus units under a declared
link, with a guess and a lapse rate as separate bounded parameters. See
[psychometric functions](../psychometric-functions.md).

The signal detection family separates sensitivity from response bias for yes/no,
forced-choice, rating, and response-with-confidence data, and states every convention it
uses rather than leaving it to the reader. See
[signal detection theory](../signal-detection.md).

Neither family represents change. Both describe a fixed observer over the trials they are
given, which is what makes them the comparators a longitudinal claim has to beat; neither
has a hierarchical or smooth variant, so a multi-animal fit pools completely.

## Stable choice structure

The Bernoulli history GLM estimates stimulus and choice-history effects that remain fixed
through the study. Hierarchical variants share information across animals without treating
trials as independent subjects.

The multinomial logit extends the same stable conditional-choice role to more than two
actions, trial-specific choice sets, and an explicit modeled omission category. See
[multinomial and omission-aware choice](../multinomial.md).

## Smooth longitudinal change

Fixed-knot GLMs and drift-diffusion models allow declared parameters to vary over an
explicit clock. Hierarchical smooth models separate a population trajectory from shrunken
individual deviations.

## Discrete latent regimes

The GLM-HMM represents switching among discrete emission states. Filtered prediction is
kept separate from smoothed description, and latent-state labels are aligned only through
permutation-invariant recovery.

## Reward-driven learning

The binary Q-learning agent represents trial-by-trial value updating. It competes under the
same pointwise predictive contract as the GLM and GLM-HMM families.

## Normative belief updating

`behavio.models.belief.BetaBernoulliObserver` and
`behavio.models.belief.HierarchicalGaussianFilter` describe what an observer should believe
about a changing binary world, and read that belief out through a separately declared
response model. The distinction that matters is which column writes the recursion: a
reinforcement-learning agent's values are written by the action the agent took, whereas a
normative observer's beliefs are written by the task's own observations, which are exogenous.
Every row therefore has a density of its own, which is why a lapse mixture is well defined
here and refused for the agents.

Both are clean-room implementations validated against closed forms rather than against
another package, because implementations of the HGF's update equations disagree with each
other in ways a fitted number cannot reveal. The third level's tonic volatility does not
recover from binary responses on any reversal design tested, and `describe()` measures and
reports that before anything is fitted. See
[normative belief updating](../normative-belief.md).

## Value-based and economic choice

`TemporalDiscounting` and `ProspectTheory` score a binary choice between two options as a
softmax over their subjective values: a discount factor of delay for the first, a
domain-dependent value function and Prelec probability weighting for the second. Both are
static — they describe a fixed decision maker — but both compose with `smooth()` and
`hierarchical()`, so a discount rate that drifts across training or differs between animals
is available without new modelling code. Their inverse temperature trades off against the
value function's curvature, and the designs in which that trade-off is exact are reported by
`describe()` before the fit. See
[economic and value-based choice](../economic-choice.md).

## Choice and response time

Wiener drift-diffusion models jointly score choice and response time. Static, smooth,
hierarchical, and explicit-contaminant variants share physical-unit and fit-audit
contracts.

## Scalar timing

`behavio.models.scalar_timing.DurationReproduction` and
`behavio.models.scalar_timing.TemporalBisection` are the two standard interval-timing
paradigms over one memory: a duration is represented with noise **proportional to the
duration itself**, which is Gibbon's (1977) scalar property and is the whole content of the
theory. Reproduction scores a continuous duration and bisection scores a binary report, and
both estimate the same clock rate and Weber fraction, so a study running both can ask whether
one Weber fraction describes them. A bisection curve crosses one half at the geometric mean
of its anchors (Church & Deluty 1977) under the declared ratio rule; the decision rule is
part of the model's signature because a bisection point cannot be read without it. The
designs that cannot see the scalar property — durations spanning too narrow a range — are
reported by `describe()` before the fit.

Nothing here is called an *interval*. `behavio.observed.interval_policy` curates annotation
bouts and is unrelated; the thing an animal times is a **duration**.

## Patch leaving and the marginal value theorem

`behavio.models.patch_leaving.PatchLeaving` scores a **residence time**, which may be
right-censored when a session ends while the animal is still in a patch. The observable is a
hazard: the animal is deciding, moment by moment, whether to go, and the model estimates the
intake rate at which it gives up together with how noisily it applies that threshold.

Charnov's (1976) theorem is kept beside the model rather than inside it.
`marginal_value_rate` and `marginal_value_residence_time` are the closed form with no fitting
in them, and a fit that declares a travel-time column reports an `overstaying_ratio` against
that benchmark — so the most replicated result in the foraging literature is measured rather
than assumed. A study whose patches all deplete identically cannot test the theorem at all,
because a rate threshold and a time threshold then predict the same residence times, and
`describe()` says so.

## Comparison is part of the method

A model family becomes interpretable only after specifying its alternatives, prospective
target, aggregation unit, fit diagnostics, and design-specific recovery. See
[model comparison](../comparison.md) and [model recovery](../model-recovery.md).
