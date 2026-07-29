# Auditable interval and bout policies

External behaviour tools discover or annotate bouts. Behavio should not repeat
that discovery, but scientists still need to state which bouts enter an analysis
and how adjacent, long, contextual, or overlapping bouts are treated. Policy
order changes the estimand, so it is written down rather than improvised.

<figure class="doc-figure doc-figure--wide" data-figure-kind="Conceptual">
  <img src="../assets/interval-policy-pipeline-v0.1.svg" alt="External behaviour bouts pass through an explicitly ordered filter, merge, split, context, and overlap policy. The result retains transformed intervals, a complete lineage ledger, a fingerprint, and direct encoding-model inputs.">
  <figcaption><strong>Policy order is part of the estimand.</strong> Every operation records its inputs, outputs, reason, and source interval lineage, and the whole run carries one fingerprint.</figcaption>
</figure>

## The boundary

DeepLabCut, SLEAP, Keypoint-MoSeq, BORIS, or another upstream system decides what
the behaviour annotation means. `apply_interval_policy()` starts from
synchronised [`BehaviorAnnotations`](ethograms.md); it never examines video,
pose, or a recorded signal to choose a rule. Point events are carried through
unchanged.

| Operation | Declared decision | Important refusal or evidence |
|---|---|---|
| `FilterIntervals` | labels, duration bounds, confidence floor | missing confidence fails a declared floor |
| `MergeIntervals` | same-label maximum gap and confidence aggregation | no cross-label semantic merging |
| `SplitIntervals` | physical cut points and/or maximum segment duration | no data-dependent boundary discovery |
| `ContextualizeIntervals` | overlap source, overlap fraction, multi-match rule, label template | subject/session and clocks must match |
| `ResolveIntervalOverlaps` | scope and reject-or-priority rule | rejection names conflicts; priority retains trimmed lineage |

## Define an ordered policy

```python
from behavio.interval_policy import (
    ContextualizeIntervals,
    FilterIntervals,
    IntervalPolicy,
    MergeIntervals,
    ResolveIntervalOverlaps,
    apply_interval_policy,
)

policy = IntervalPolicy(
    (
        MergeIntervals(
            "merge-short-groom-gaps",
            labels=("groom",),
            maximum_gap_s=0.15,
        ),
        FilterIntervals("minimum-bout", minimum_duration_s=0.5),
        ContextualizeIntervals(
            "task-context",
            context_source="task-epochs",
            minimum_overlap_fraction=0.5,
            multiple_matches="first",
        ),
        ResolveIntervalOverlaps(
            "exclusive-state",
            scope="all",
            strategy="priority",
            label_priority=("groom@test", "locomotion@test"),
        ),
    )
)

result = apply_interval_policy(
    synchronized_bouts,
    policy,
    context_sources={"task-epochs": synchronized_task_epochs},
)
```

The tuple is executed exactly in the order written. Merging two short bouts can
make their combined interval pass a minimum-duration filter; filtering first can
remove both. Neither order is universally correct, so Behavio does not reorder
or optimise the operations.

## Inspect the evidence before fitting

```python
print(result.input_interval_count, result.output_interval_count)
for row in result.ledger:
    print(row.operation_id, row.action, row.reason)

artifact = result.to_dict()  # JSON serializable
print(result.evidence_fingerprint)
```

Each ledger row contains immutable input and output snapshots. Contextual relabelling
rows also contain the precise overlapping context snapshots. Output snapshots retain
`source_interval_ids`, including all contributors to a merge and the common parent
of split or trimmed pieces. Kept and removed decisions are present too, so the
ledger records denominators rather than only successful transformations.

Store the complete dictionary beside the analysis configuration. The SHA-256
fingerprint covers the policy, transformed annotations, complete named context
annotations, ledger, source metadata, clock synchronization IDs, and counts. A
changed operation order, threshold, or context artifact
therefore changes the evidence identity.

## Context labels remain supplied evidence

`ContextualizeIntervals` accepts a named `BehaviorAnnotations` source such as task
epochs, stimulation states, sleep stages, or manual context annotations. Positive
physical overlap is required. `minimum_overlap_fraction` is measured relative to
the target bout, not the context interval.

Choose the multi-match meaning explicitly:

- `duplicate` creates one labelled copy per distinct qualifying context;
- `first` uses `context_labels` as an explicit priority and then lexical order; or
- `reject` stops when a bout matches more than one context.

Use `keep_unmatched=False` only when excluding out-of-context bouts is part of the
prospective analysis. The exclusion will remain in the ledger.

## Overlap policies

`scope="same_label"` addresses repeated annotations of one state while leaving
co-occurring behaviors intact. `scope="all"` declares mutually exclusive states.
The default `strategy="reject"` is safest when overlap has no prespecified meaning.

With `strategy="priority"`, labels earlier in `label_priority` take precedence.
Higher-priority physical spans are subtracted from later intervals. A lower-priority
bout can therefore be trimmed, split into multiple pieces, or removed. Equal and
unlisted priorities resolve by physical start, stop, label, and stable interval ID;
declare all meaningful priorities rather than relying on that deterministic tie-break.

## Continue into a model

The result uses the existing annotation contract, so no conversion layer is
needed:

```python
inputs = result.annotations.interval_encoding_inputs(edge="onset")

inputs.events  # onset (or offset) times per label
inputs.event_values  # {"duration_s": (...)} per label
inputs.intervals  # physical (start_s, stop_s) bounds per label
```

Those three mappings supply physical edges, duration modulation, and
normalised-progress bases without turning policy decisions into behaviour
discovery. `fipha` consumes exactly this bundle in its event-kernel encoding
models. See the complete
[`examples/interval_policy.py`](https://github.com/aeronjl/behavio/blob/main/examples/interval_policy.py)
and the [behaviour-tool interoperability tutorial](tutorials/behavior-tool-interoperability.md).

## Current limits

The policy is session-local and interval-based. It does not infer bouts from
probabilities, reconcile annotators statistically, select thresholds from
analysis outcomes, or fit longitudinal behaviour. Keep probabilistic state
inference and identity QC upstream; the [longitudinal study
contract](data-contract.md) begins only after declared trial-level summaries
exist.
