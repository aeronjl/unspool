# Ethograms: point events and behavioural intervals

An ethogram is a catalogue of what an animal did and when. `behavio.ethograms`
holds that catalogue as two types and the readers that fill them, as one shape
in the wider [observed-behaviour boundary](observed-behaviour.md):

- `BehaviorInterval` — one labelled bout with a physical start and stop, and an
  optional confidence;
- `BehaviorAnnotations` — the intervals of a session alongside its named
  instantaneous `point_events`.

The distinction is load-bearing. A point event and a zero-length interval are
different claims about the data, and tools such as BORIS make that distinction
explicitly. Collapsing it at the boundary would discard information the source
tool took care to record.

!!! note "Why `ethograms`, not `annotations`"
    A module called `annotations` sitting next to `from __future__ import
    annotations` costs every reader a double-take. `ethograms` names the thing.

## Installing

The BORIS readers need only the standard library. The Keypoint-MoSeq file
reader needs HDF5:

```bash
pip install "behavio[readers]"
```

## Projections for modelling

`BehaviorAnnotations.event_times(edge="onset")` provides an explicit projection
from intervals to event-locked predictors. The original intervals are retained,
so an onset analysis does not silently become the only representation of the
data. `normalized_progress()` maps samples inside a declared bout to
zero-to-one progress while the source interval retains its physical duration,
and returns a [`BehaviorCovariate`](covariates.md).

For inference, `interval_encoding_inputs()` returns aligned edge events,
`duration_s` values, and physical bounds as plain mappings. A consuming model
can then evaluate progress only inside each bout while retaining outside-bout
samples as zero-valued design rows. This avoids turning absence of a behaviour
into missing data.

Which bouts enter an analysis at all is a separate, ordered decision; see
[auditable interval and bout policies](interval-policy.md).

## Keypoint-MoSeq

`annotations_from_moseq()` run-length encodes an in-memory `syllable` sequence.
`annotations_from_moseq_results_h5()` reads the documented recording hierarchy
directly. Each interval retains the full duration of one uninterrupted state.

```python
from behavio.ethograms import annotations_from_moseq_results_h5

states = annotations_from_moseq_results_h5(
    "moseq-project/model-a/results.h5",
    recording="recording-01",
    subject="mouse-07",
    session="day-04",
    fps=30.0,
    labels={0: "pause", 1: "rear"},
)
rear_onsets = states.event_times()["rear"]
```

Syllable numbers are model-specific, may be reindexed, and are not universal
behaviour names. A semantic label mapping is therefore optional and provenance
of the fitted MoSeq model remains essential.

## BORIS

`annotations_from_boris()` consumes already-loaded aggregated columns.
`annotations_from_boris_aggregated_file()` reads BORIS aggregated CSV/TSV
directly. `annotations_from_boris_tabular_file()` handles the other common
shape: a metadata preamble followed by START/STOP or point-event rows.

```python
from behavio.ethograms import (
    annotations_from_boris_aggregated_file,
    annotations_from_boris_tabular_file,
)

annotations = annotations_from_boris_tabular_file(
    "mouse-07-day-04-boris.csv",
    subject="mouse-07",
    session="day-04",
    source_subject="mouse-07",
)

aggregated = annotations_from_boris_aggregated_file(
    "mouse-07-day-04-aggregated.tsv",
    subject="mouse-07",
    session="day-04",
    source_subject="mouse-07",
)
```

In tabular files, blank or POINT status rows become point events and START/STOP
rows are paired as intervals. In aggregated tables, POINT and STATE types are
handled directly. Invalid, unmatched or unknown rows are rejected.

## Clock identity

Annotations carry `clock_id` like every other type in this boundary. Moving
them onto a recording clock is
[`ClockSynchronization.synchronize_annotations()`](clock-synchronization.md),
which transforms point and interval times while retaining their point-versus-state
semantics.

Signatures are in the [observed behaviour API](reference/observed-behaviour.md).
