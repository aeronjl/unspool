# Scientific decision records

This directory records consequential scientific and API decisions, including
decisions to defer or reject a method. It is an audit trail, not a claim that
every decision is permanent.

## Status vocabulary

- **Draft:** proposed and open to change; not normative.
- **Accepted:** current project policy.
- **Rejected:** considered but not adopted.
- **Superseded:** replaced by a later numbered record, which must link back.

Decision records state their date, evidence available at that date,
alternatives, consequences and an explicit revisit trigger. Later evidence is
appended or linked; the original reasoning is not silently rewritten.

## Inherited numbering

The records below were made in the fiber photometry package
[`fipha`](https://github.com/aeronjl/fipha) and moved here with the code they
govern when the general behaviour surface moved to Behavio. They keep
their original `SDR-NNNN` identifiers so that reports, benchmark protocols and
docstrings that already cite them stay resolvable, and so that `fipha`'s own
register can retire the same numbers rather than reuse them. Records originating
in Behavio will continue that sequence rather than restarting it.

Cross-references that point at records still owned by `fipha` are absolute links
into that repository.

The inherited block ends at **SDR-0059**. **SDR-0060** is the first record originating in
Behavio, and a new record takes the next unused number after the highest in the index below.
A gap in the sequence means a record was moved or lost, not that a number was skipped, so
reserving a number ahead of writing the record is not a way to avoid a collision.

## Index

| Record | Status | Decision |
| --- | --- | --- |
| [SDR-0032](0032-preserve-external-behavior-semantics.md) | Accepted | Preserve external behaviour semantics at separate typed boundaries rather than one generic event table |
| [SDR-0034](0034-fit-only-explicit-matched-pulse-clock-transforms.md) | Accepted | Fit only explicit matched-pulse affine clock transforms against prospective thresholds |
| [SDR-0049](0049-make-interval-policy-order-explicit-and-auditable.md) | Accepted | Make interval-policy order an explicit, ledgered, fingerprinted part of the estimand |
| [SDR-0059](0059-consume-movement-datasets-without-depending-on-movement.md) | Accepted | Consume `movement` poses datasets by duck typing rather than depending on `movement` |
| [SDR-0060](0060-bisect-time-by-the-ratio-rule.md) | Accepted | Bisect time by the ratio rule, declare it in the signature, and say why one anchor pair cannot test it |
| [SDR-0061](0061-fit-patch-leaving-as-a-hazard-not-as-the-marginal-value-theorem.md) | Accepted | Fit patch leaving as a threshold-crossing hazard and read it against the marginal value theorem rather than fitting the theorem |
| [SDR-0062](0062-implement-normative-belief-updating-clean-room.md) | Accepted | Implement normative belief updating clean-room, with every disputed HGF convention declared and validated against a closed form |
| [SDR-0063](0063-defer-the-log-score-only-comparison-and-the-survival-carrying-prediction.md) | Accepted | Record the log-score-only comparison and survival-carrying prediction as shared-layer gaps; both are now resolved by shared contracts |
| [SDR-0064](0064-model-dynamic-transitions-with-multinomial-logits-and-ilr-group-effects.md) | Accepted | Model observed transition non-homogeneity with multinomial logits and complete group effects in an isometric log-ratio coordinate; keep latent session drift separate |
| [SDR-0065](0065-fit-session-dynamic-glm-hmm-paths-by-map-em.md) | Accepted | Fit one-subject session-dynamic GLM-HMM paths by MAP EM with published emission and transition priors, whole-path labels, and an explicit future-session prior-mode forecast |
| [SDR-0066](0066-fit-a-population-session-dynamic-glm-hmm-with-subject-deviation-paths.md) | Accepted | Fit a population emission path and evolving subject deviations for a cross-subject session-dynamic GLM-HMM with explicit unseen-subject prediction |
| [SDR-0067](0067-keep-dynamic-glm-hmm-uncertainty-conditional-on-one-label-mode.md) | Accepted | Estimate dynamic hierarchy hyperparameters and report path uncertainty conditional on one whole-path label mode, with unstable missing-information corrections left visible |
| [SDR-0068](0068-model-laboratories-as-an-exchangeable-level-above-subjects.md) | Accepted | Model laboratories as exchangeable dynamic paths above nested subject paths, with complete-lab prediction and lab-joint scoring |
| [SDR-0069](0069-sample-glm-hmm-states-by-marginalizing-the-discrete-path.md) | Accepted | Sample proper-prior stationary and nested session-dynamic GLM-HMMs by marginalizing discrete paths, jointly propagating hierarchy uncertainty, and retaining draw-wise label ambiguity |

## Record structure

Use four-digit monotonic identifiers and the sections `## Context`,
`## Decision`, `## Consequences`, `## Alternatives considered` and
`## Revisit trigger`. Add every new record to the index above.
