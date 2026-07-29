# SDR-0049: Make interval-policy order explicit and auditable

- **Status:** Accepted
- **Date:** 2026-07-28
- **Related decisions:** [SDR-0032](0032-preserve-external-behavior-semantics.md), [SDR-0038](https://github.com/aeronjl/fipha/blob/main/docs/decisions/0038-model-variable-duration-behavior-with-physical-intervals-and-progress.md)

!!! note "Moved from fipha"
    This record was made in the photometry package `fipha` and moved here with
    the code it governs when the general behaviour surface moved to Behavio.
    It keeps its original SDR number so that reports which
    cite it stay resolvable. Its module references have been updated; its
    decision has not been changed.

## Context

External behavior tools can produce fragmented, short, overlapping, or
context-dependent bouts. Neural analyses commonly filter duration or confidence,
bridge short gaps, divide long intervals, stratify bouts by experimental context,
or require mutually exclusive states. These transformations change event counts,
durations, exposure, and the meaning of onset, offset, and progress estimands.

The operations are order-dependent. Two individually sub-threshold bouts can pass a
duration rule after merging. Context duplication can create overlaps that did not
exist in the source. Applying undocumented cleanup in an adapter would conceal both
the source-tool semantics and analysis denominator.

## Decision

Represent interval handling as an ordered tuple of typed operations applied only to
externally supplied, already synchronized `BehaviorAnnotations`.

- The package never reorders operations or selects thresholds from the signal
  under analysis.
- Filtering records kept and removed intervals; missing confidence fails a declared
  confidence floor.
- Merging is same-label only and declares its gap and confidence aggregation.
- Splitting uses declared physical timestamps or a maximum physical duration.
- Context labels come from a named annotation source with matching subject, session,
  and clock. Multi-context matches must duplicate, choose by declared priority, or
  reject explicitly.
- Overlap handling is scoped to the same label or all labels. It either rejects or
  subtracts higher-priority spans from lower-priority intervals.
- Every operation records immutable inputs, outputs, action, reason, and original
  source interval IDs, including unchanged and removed intervals.
- A canonical evidence artifact and SHA-256 fingerprint bind policy order, source
  provenance, transformed annotations, ledger, and counts.
- Point events and external source identity pass through unchanged. The result feeds
  the existing physical-edge, duration, and progress encoding contract directly.

## Alternatives considered

- **Clean intervals inside each adapter.** Rejected because the adapter would silently
  change upstream semantics and different source tools would receive different rules.
- **Use one fixed cleanup order.** Rejected because merge-before-filter and
  filter-before-merge answer different questions; order must remain analyst-visible.
- **Automatically merge every overlap.** Rejected because simultaneous labels can be
  scientifically meaningful and cross-label merging invents a new state.
- **Store only the final intervals.** Rejected because removed denominators and the
  parentage of merged, split, or trimmed intervals would be unrecoverable.
- **Put context mappings on `BehaviorInterval` itself.** Deferred because the current
  single-label boundary composes with event kernels and because typed relabelling plus
  lineage is sufficient for the first product contract.

## Consequences

Policies are verbose, and priority resolution can turn one interval into several
physical pieces. This is intentional evidence rather than implementation noise.
Scientists must decide whether co-occurring behaviors are allowed and whether context
overlap warrants duplication, selection, or rejection.

The first contract is deterministic and session-local. It supports broad ingestion
and model composition without claiming to infer behavior or reconcile annotators.

## Revisit trigger

Revisit when at least two downstream consumers need structured multi-axis context
without composite labels, or when probabilistic state sequences require a separate
uncertainty-preserving object model. Add annotator-consensus methods only with a
declared reliability estimand and validation fixtures; do not fold them into this
deterministic policy layer.
