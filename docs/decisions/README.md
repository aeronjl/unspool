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

## Index

| Record | Status | Decision |
| --- | --- | --- |
| [SDR-0032](0032-preserve-external-behavior-semantics.md) | Accepted | Preserve external behaviour semantics at separate typed boundaries rather than one generic event table |
| [SDR-0034](0034-fit-only-explicit-matched-pulse-clock-transforms.md) | Accepted | Fit only explicit matched-pulse affine clock transforms against prospective thresholds |
| [SDR-0049](0049-make-interval-policy-order-explicit-and-auditable.md) | Accepted | Make interval-policy order an explicit, ledgered, fingerprinted part of the estimand |
| [SDR-0059](0059-consume-movement-datasets-without-depending-on-movement.md) | Accepted | Consume `movement` poses datasets by duck typing rather than depending on `movement` |

## Record structure

Use four-digit monotonic identifiers and the sections `## Context`,
`## Decision`, `## Consequences`, `## Alternatives considered` and
`## Revisit trigger`. Add every new record to the index above.
