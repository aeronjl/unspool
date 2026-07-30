# SDR-0060: Bisect time by the ratio rule, and declare it

- **Status:** Accepted
- **Date:** 2026-07-30
- **Related decisions:** [SDR-0049](0049-make-interval-policy-order-explicit-and-auditable.md)

## Context

`behavio.models.scalar_timing.TemporalBisection` fits the standard temporal-bisection
paradigm: two anchor durations \(S < L\) are trained, a probe \(t\) is presented, and the
animal reports whether the probe was more like the short anchor or the long one. Fitting it
requires a **decision rule** — a statement of which anchor a remembered probe counts as
closer to — and the literature does not agree on one.

Three rules are named in the field:

| Rule | Respond *long* when | Crosses one half at |
| --- | --- | --- |
| Ratio (Gibbon 1981) | \(\hat{T}/S > L/\hat{T}\) | \(\sqrt{SL}\) |
| Similarity (Wearden 1991) | the smaller/larger ratio favours \(L\) | \(\sqrt{SL}\) |
| Difference (arithmetic) | \(\hat{T} - S > L - \hat{T}\) | \((S+L)/2\) |

The first two are the same rule: "similarity of two durations" *means* the ratio of the
smaller to the larger, so Wearden's rule and Gibbon's give the same comparison point. There
are therefore two candidate rules, not three, and they disagree about exactly one number.

Under the module's shared scalar memory — \(\log \hat{T} \sim \mathcal{N}(\log(\kappa t),
\sigma^2)\) with \(\sigma^2 = \log(1+w^2)\) — both rules produce the same psychometric
*shape*: a probit in \(\log t\) whose slope is \(1/\sigma\). The probit comes from the
memory rather than from the rule. So on **one anchor pair** the two rules are related by a
reparameterisation: they fit any single study identically, with clock rates differing by
exactly \(2\sqrt{SL}/(S+L)\), the same Weber fraction, and the same log likelihood.

## Decision

**The default is the ratio rule.** `BisectionRule.RATIO`, comparison duration
\(\sqrt{SL}\).

**The difference rule is implemented and reachable** as `BisectionRule.DIFFERENCE`, because
the disagreement is real and human bisection points do sometimes land near the arithmetic
mean.

**The rule is part of the model's `signature` and of its `model_name`.** A fit produced
under one rule cannot be read as a fit under the other, and
`temporal-bisection-ratio` and `temporal-bisection-difference` are different models to
`compare_models`, to `nested_select_model` and to a frozen protocol.

The two halves of the argument for the ratio rule:

**Empirical.** Church and Deluty (1977) trained rats on four anchor pairs — 1 v 4, 2 v 8,
3 v 12 and 4 v 16 seconds — and found the bisection point at the geometric mean of each.
The arithmetic rule predicts 2.5, 5, 7.5 and 10 s where the geometric rule predicts 2, 4, 6
and 8. Varying the pair is what makes the two accounts separable; see *Consequences*.

**Theoretical.** The clock's noise is multiplicative, which is the whole content of scalar
expectancy theory. A ratio comparison is the only one that leaves every noise source in the
model multiplicative. This module treats the anchors as remembered exactly, and under that
idealisation the rules differ in one number and nothing else — but any complete account has
to admit that the anchors are remembered with noise too, and the moment it does, the
idealisation breaks in the ratio rule's favour: the ratio comparison's anchor noise is
additive on the log scale and therefore scalar, while the difference comparison's is additive
on the linear scale and therefore is not. The difference rule stops obeying Weber's law at
exactly the point where it stops being a relabelling.

## Consequences

**No single fit of `TemporalBisection` can test the rule.** A model instance carries one
anchor pair, and on one pair the clock rate absorbs the difference between the two comparison
durations exactly. `tests/test_scalar_timing.py::test_one_anchor_pair_cannot_separate_the_two_rules`
asserts this: the same reports fitted under both rules return the same Weber fraction, the
same objective, and clock rates in the ratio \(5/4\) for a 2 s / 8 s pair. Separating the
accounts means fitting several anchor pairs with the clock rate held common across them,
which is Church and Deluty's design and is a comparison *between* models rather than a
parameter inside one. This package does not currently offer a multi-pair model, and adding
one is the natural follow-up.

**A reported bisection point is meaningless without the rule.** That is why the rule is in
the signature rather than a constructor convenience, and why the fit's
`bisection_point` derived quantity carries the rule's name in its description.

**Anchors that are close together cannot tell the accounts apart even in principle.** For
\(L/S < 1.5\) the two comparison durations are within a few per cent of each other, and
`describe()` reports `narrow_anchor_ratio` with the gap between them stated.

**A central-tendency exponent is not offered on this family.** Under this memory the
decision variable is \(\beta \log t / \sigma\), so an exponent and the Weber fraction are
exactly confounded in the slope. `DurationReproduction` estimates one; bisection cannot, and
the honest response is to run both paradigms rather than to report an unidentified number.

## Alternatives considered

**Estimate the comparison duration as a free parameter.** Rejected. It is a
reparameterisation of the clock rate — one number multiplying the other — so the model would
gain a parameter with no likelihood curvature of its own and every fit would sit on a ridge.
The rule is a claim about the *decision*, and a claim is declared, not estimated.

**Offer a third "similarity" rule.** Rejected as a duplicate: Wearden's similarity rule is
the ratio rule under another name, and a third enum member reachable from the same
arithmetic would be a second spelling of one model.

**Default to the arithmetic rule because human data sometimes land there.** Rejected. Human
bisection points scatter between the two means and are pulled by stimulus spacing and by
which anchor was presented more often; the animal literature the rule was formulated in is
unambiguous, and the theoretical argument runs the same way. A default is a claim about the
usual case, and both the strongest data and the theory point at the ratio rule.

**Put the noise on the decision rather than on the memory.** Rejected for this release. A
logistic decision noise on top of an exact memory would make the two rules genuinely
different shapes rather than a relabelling, which would be scientifically interesting, but it
introduces a second noise parameter that a single-pair design cannot separate from the first.
A lapse rate covers the practical case and is already available through
`mix(model, UniformChoiceGuess())`.

## Revisit trigger

Revisit when a multi-anchor-pair bisection model exists — one that fits several \((S, L)\)
pairs with a shared clock rate and Weber fraction — because that model *can* test the rule
against data and would turn this declaration into a comparison. Revisit sooner if noisy
anchor memories are added to the family, since the two rules stop being a reparameterisation
at that point and the default would then be a substantive prediction rather than a
convention.
