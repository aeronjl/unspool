# SDR-0063: Record two continuous-outcome scoring gaps rather than patching them under a model wave

- **Status:** Accepted
- **Date:** 2026-07-30
- **Related decisions:** [SDR-0061](0061-fit-patch-leaving-as-a-hazard-not-as-the-marginal-value-theorem.md),
  [SDR-0062](0062-implement-normative-belief-updating-clean-room.md)

## Context

Three separate model waves — the scalar-timing families, `PatchLeaving`, and the
`dynamax` switching-autoregression wrapper — arrived at the same two walls, independently
and in that order. Each wave recorded the wall in its own module docstring or decision
record and worked around it locally, which is exactly how a shared-layer gap gets
rediscovered a fourth time. This record is the one place both are stated, so the next
family finds them before it hits them.

Neither is a defect in a model. Both are consequences of a scoring layer that was written
when every first-party family scored a discrete choice, and both are now blocking a whole
class of model rather than one family.

### Gap one: `compare_models` cannot rank two unlabelled-density candidates

`behavio.compare.compare_models` computes a Brier column beside the log score,
unconditionally. For a `DensityPrediction` with no categories it raises
`UnscoreableByBrier` rather than inventing a number, and that refusal is
*correct*: a Brier score is a squared distance to an indicator, so it needs a discrete
margin, and a residence time, a reproduced duration and a running speed have none.
`_brier_scoreable_margin` states the argument at length and this record does not disturb it.

What is missing is the other half: there is no way to ask `compare_models` for the
**log-score half alone**. The log score is the joint log density of the whole observation
and is perfectly well defined for every one of these predictions —
`behavio.evaluate.evaluate_splits` reports it and works fine — so the refusal removes a
defined metric along with an undefined one.

The consequence is that the prospective *comparison table*, which is the package's own
answer to "how do you decide between two accounts", is unreachable for every
continuous-outcome model, present and future: `DurationReproduction`, `PatchLeaving`,
`DynamaxSwitchingAutoregression`, and any censored response-time model that follows them.
A user who wants two timing models ranked against each other has to reimplement the
aggregation that `compare_models` already contains.

### Gap two: no member of `ModelPrediction` can carry a survival probability

`ModelPrediction` is `Prediction | CategoricalPrediction | DensityPrediction`. A
right-censored row's score is `log S(c)` — the probability that the event is still to come —
and none of the three can express it. A `DensityPrediction` says what the model claims about
the row, which for a truncated patch visit is still a density over the *leaving time*; it
cannot also say "and this row was only observed up to `c`".

`PatchLeaving` therefore has `pointwise_log_prob` and
`DensityPrediction.observed_log_density` **agree on uncensored rows and deliberately
disagree on censored ones**: the first is the likelihood, the second is the prediction. That
disagreement is documented in the module docstring, reported through the `heavy_censoring`
finding, and reachable by a consumer who scores the returned density instead of asking the
model — which will misscore exactly the censored rows, silently.

## Decision

**Record both, fix neither in a model wave.** Each is a change to a shared contract that
every existing family reads, and a change of that size made under a wave whose subject is
one model family gets its design decided by that family's convenience.

Until they are fixed:

- a continuous-outcome model is compared through `evaluate_splits`, whose log score is
  defined for it, and not through `compare_models`;
- a censored family's likelihood is read through `pointwise_log_prob`, never by scoring the
  density `predict()` returns. Every family with censoring must say so in its own docstring
  and report the share of rows affected as a `describe()` finding, as `PatchLeaving` does.

The shapes a fix would take, stated here so the next attempt starts from a position rather
than from scratch:

**For gap one**, a declared metric set on `compare_models` — the caller names which scoring
rules the table carries, and a candidate that cannot support a named rule is refused at
declaration rather than at scoring time. That keeps the current default honest (a comparison
that silently dropped its Brier column would be a different table under the same name) and
makes the log-score-only table an explicit request. It is a change to the report type, so it
touches the evidence bundles and the frozen protocols that read them.

**For gap two**, a fourth `ModelPrediction` member carrying an interval-valued observation,
or an explicit observation-limit channel on `DensityPrediction`. The second is smaller and
the first is more honest; deciding between them wants a second censored family to look at,
which is the same trigger [SDR-0061](0061-fit-patch-leaving-as-a-hazard-not-as-the-marginal-value-theorem.md)
already names for its kernel boundary.

## Consequences

**The comparison layer is now the narrowest part of the stack, and that is visible.** Every
other contract — simulation, fitting, prediction, pointwise scoring, diagnostics,
`smooth()`, `hierarchical()`, `mix()` — accepts a continuous-outcome family without
modification. Comparison does not, and it is the one a falsification-first package can least
afford to have missing.

**A censored row is scored correctly today and read incorrectly by an unwary consumer.** The
package cannot make the second impossible without gap two closed; it can and does make it
loud.

**Nothing is silently approximated.** No Brier score is invented for a density, no censored
row is scored by its density, and no number is reported for a quantity the layer cannot
express. That is the property this deferral protects, and it is worth more than the
convenience it costs.

## Alternatives considered

**Make the Brier column optional with a keyword and ship it now.** Tempting and small, and
rejected as the *first* move rather than on its merits. A boolean that turns off one column
is a metric declaration with one bit in it; the decision worth making is which rules a
comparison table carries and how a candidate declares what it supports, and a boolean added
first is the thing the eventual design has to be compatible with.

**Let a density's Brier score be its integrated-mass version, or `NaN`.** Rejected, and this
is the same refusal `_brier_scoreable_margin` already makes: a rescaled density ranks models
by their grid resolution, and a `NaN` in a comparison table is a number-shaped hole that
aggregation will average over.

**Score censored rows by their density and drop the distinction.** Rejected in
[SDR-0061](0061-fit-patch-leaving-as-a-hazard-not-as-the-marginal-value-theorem.md), where
the bias it produces is measured rather than argued: a truncated visit read as a departure
looks like a forager that left sooner, so the fitted giving-up rate moves up.

**Give `PatchLeaving` a private comparison path.** Rejected. A second implementation of the
aggregation, the paired bootstrap and the fold provenance, reachable only from one family,
is how a shared layer stops being shared.

## Revisit trigger

- A second continuous-outcome family needs a prospective comparison table, which makes the
  metric declaration a design with two users rather than one.
- A censored response-time model is written, which is the same trigger
  [SDR-0061](0061-fit-patch-leaving-as-a-hazard-not-as-the-marginal-value-theorem.md) names
  and which would decide between the two shapes gap two could take.
- Either gap is reported by someone outside this repository, which would make it a usability
  defect rather than a deferred design.
