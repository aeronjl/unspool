# Compose pose, behavioural states and a recorded signal

## Scientific question

How can a scientist ask whether a recorded signal relates to movement, an
automatically discovered behavioural bout, or a manually annotated event,
without reimplementing behaviour analysis?

This tutorial composes native-shaped outputs from DeepLabCut, SLEAP,
Keypoint-MoSeq and BORIS into Behavio's typed boundary, puts them on one clock,
and hands them to a consuming analysis. The complete executable synthetic
example is
[`examples/behavior_interoperability.py`](https://github.com/aeronjl/behavio/blob/main/examples/behavior_interoperability.py).

!!! warning "Synchronisation is scientific evidence"
    The source snippets retain the `video` clock. Section 5 estimates and
    validates its mapping to the acquisition clock from explicit paired pulses.
    Renaming both clocks `acquisition` is not a valid mapping.

## 1. Preserve pose confidence

DeepLabCut prediction CSV and pandas HDF5 files contain hierarchical columns.
Select the scorer, individual and body part explicitly when more than one is
present. Install the lightweight file dependencies with
`uv add "behavio[readers]"`.

```python
from behavio.observed.pose import pose_from_deeplabcut_file

nose = pose_from_deeplabcut_file(
    "mouse-07-day-04DLC.h5",
    subject="mouse-07",
    session="day-04",
    keypoint="nose",
    scorer="DLC_resnet50_project",
    fps=30.0,
    clock_id="video",
)
nose_speed = nose.speed(minimum_confidence=0.9)
```

`nose_speed.valid` is false for a displacement whenever either endpoint is missing
or below the declared likelihood threshold. Coordinates are not silently filled.
If a camera calibration is available, pass its declared conversion as
`coordinate_scale` and give the corresponding `output_unit`.

```python
nose_speed_cm = nose.speed(
    minimum_confidence=0.9,
    coordinate_scale=0.042,
    output_unit="cm/s",
)
```

This scalar conversion is appropriate only for a spatially uniform calibration.
Perspective-distorted or three-dimensional arenas require the relevant geometric
calibration upstream.

## 2. Read SLEAP without guessing axes

SLEAP's Analysis HDF5 may use a MATLAB-compatible or Python-native array order.
The file reader uses stored dimension metadata when present. Older files omit it,
so supply `dims` rather than guessing from array size.

```python
from behavio.observed.pose import pose_from_sleap_analysis_h5

nose = pose_from_sleap_analysis_h5(
    "mouse-07-day-04.analysis.h5",
    node="nose",
    # Omit for current files that store dimension metadata.
    dims=("track", "xy", "node", "frame"),
    subject="mouse-07",
    session="day-04",
    track=0,
    fps=30.0,
    clock_id="video",
)
```

Do not assume a SLEAP point score is a probability. The pinned legacy upstream
fixture contains scores above one; the threshold must be appropriate to the
producing model and version.

This tutorial chooses one track. A social experiment should make track identity a
declared part of its design and audit identity switches before neural inference.

## 3. Keep MoSeq bouts as intervals

Keypoint-MoSeq's extracted result contains one syllable value per video frame.
Run-length encoding turns consecutive equal states into bouts without discarding
duration.

```python
from behavio.observed.ethograms import annotations_from_moseq_results_h5

moseq = annotations_from_moseq_results_h5(
    "moseq-project/model-a/results.h5",
    recording="mouse-07-day-04",
    subject="mouse-07",
    session="day-04",
    fps=30.0,
    labels={0: "pause", 1: "rear"},
    clock_id="video",
)
rear_onsets = moseq.event_times(edge="onset")["rear"]
rear_offsets = moseq.event_times(edge="offset")["rear"]
```

The human-readable labels are study metadata, not a claim that syllable 1 has a
universal biological meaning. Preserve the fitted model, any reindexing operation,
and its version beside the analysis.

If pose has already been standardized in NWB, inspect and load the native
`ndx-pose` container instead of returning to a tool-specific export:

```python
from fipha.io.ndx_pose import inspect_ndx_pose_nwb, poses_from_ndx_pose_nwb

inspection = inspect_ndx_pose_nwb("mouse-07-day-04.nwb")
poses = poses_from_ndx_pose_nwb(
    "mouse-07-day-04.nwb",
    subject="mouse-07",
    session="day-04",
    clock_id="video",
    processing_module_name="behavior",
    pose_estimation_name="TopCameraPose",
)
```

That reader lives in `fipha` and returns these same `PoseTrajectory` objects; it
preserves physical conversion, optional z, confidence, reference frame, skeleton
and source-software metadata. See its
[native round-trip guide](https://aeronjl.github.io/fipha/ndx-pose-interoperability/).

For a variable-duration analysis, retain physical bounds and aligned duration
values rather than replacing time:

```python
moseq_inputs = moseq.interval_encoding_inputs(edge="onset")
# moseq_inputs.events["rear"]
# moseq_inputs.event_values["rear"]["duration_s"]
# moseq_inputs.intervals["rear"]
```

`normalized_progress()` remains useful for visualisation. For model fitting,
prefer a progress basis built from these bounds, which keeps outside-bout samples
in the denominator instead of marking them as missing continuous-covariate rows.

## 4. Keep BORIS point and state annotations distinct

For a BORIS tabular CSV, the file reader skips the observation metadata preamble,
selects one source subject, and pairs START/STOP rows.

```python
from behavio.observed.ethograms import annotations_from_boris_tabular_file

boris = annotations_from_boris_tabular_file(
    "mouse-07-day-04-boris.csv",
    subject="mouse-07",
    session="day-04",
    source_subject="mouse-07",
    clock_id="video",
)
```

Blank or POINT rows remain point events. START/STOP pairs become positive-duration
intervals. Aggregated CSV/TSV files can be read directly without first building a
column mapping:

```python
from behavio.observed.ethograms import annotations_from_boris_aggregated_file

aggregated = annotations_from_boris_aggregated_file(
    "mouse-07-day-04-aggregated.tsv",
    subject="mouse-07",
    session="day-04",
    source_subject="mouse-07",
)
```

The analyst then chooses onset, offset, duration or progress according to the
scientific question.

Before projecting intervals into any model, declare any cleanup or contextual
rules as an ordered policy. Do not edit adapter output in place:

```python
from behavio.observed.interval_policy import (
    ContextualizeIntervals,
    FilterIntervals,
    IntervalPolicy,
    MergeIntervals,
    apply_interval_policy,
)

policy = IntervalPolicy(
    (
        MergeIntervals(
            "merge-short-gaps",
            labels=("rear",),
            maximum_gap_s=0.1,
        ),
        FilterIntervals("minimum-duration", minimum_duration_s=0.25),
        ContextualizeIntervals(
            "task-context",
            context_source="task-epochs",
            multiple_matches="reject",
        ),
    )
)
policy_result = apply_interval_policy(
    moseq,
    policy,
    context_sources={"task-epochs": task_epochs},
)
moseq = policy_result.annotations
```

The result retains kept and removed denominators plus the lineage of merged, split,
relabelled, and trimmed intervals. Its fingerprint changes if a threshold or
operation order changes. See the full
[interval-policy method and example](../interval-policy.md).

## 5. Align a pose covariate without bridging missing spans

Fit the clock mapping before interpolation. Pulse correspondence is explicit: the
function never guesses which pulses match.

```python
from behavio.observed.device_clocks import (
    DeviceClockPulses,
    DeviceClockSyncSpec,
    fit_device_clock_sync,
)

synchronization = fit_device_clock_sync(
    DeviceClockPulses.from_arrays(
        source_clock_id="video",
        target_clock_id="acquisition",
        source_time_s=video_sync_pulses,
        target_time_s=acquisition_sync_pulses,
        match_labels=sync_pulse_ids,
    ),
    DeviceClockSyncSpec(
        maximum_absolute_residual_s=0.002,
        maximum_drift_ppm=250,
        minimum_matches=4,
        minimum_source_span_s=recording_duration_s * 0.8,
    ),
)

nose_speed = synchronization.synchronize_covariate(nose_speed)
moseq = synchronization.synchronize_annotations(moseq)
boris = synchronization.synchronize_annotations(boris)
```

The thresholds above are illustrative, not defaults for all acquisition systems.
Choose them before outcome analysis from acquisition precision and the
smallest timing difference the study aims to interpret. By default, transforming
data outside the first and last matched source pulses is refused.

```python
aligned_speed = nose_speed.aligned_to(
    acquisition_time,
    target_clock_id="acquisition",
    max_gap_s=2 / 30,
)
```

The returned `BehaviorCovariate` carries both aligned values and an aligned
validity mask. It is invalid outside pose support, across low-confidence samples
and across gaps larger than `max_gap_s`. `align_to()` remains available when only
the numeric array is needed, but `aligned_to()` is the loss-aware route into a
model. Do not use a global fill that bridges long occlusions.

## 6. Hand the aligned behaviour to a model

At this point every stream carries the same `clock_id`, an explicit validity
mask, and a synchronisation lineage. The three bundles a consuming model needs
are plain mappings:

```python
moseq_inputs = moseq.interval_encoding_inputs(edge="onset")
boris_inputs = boris.interval_encoding_inputs(edge="onset")

events = {
    **moseq.point_events,
    **moseq_inputs.events,
    **boris.point_events,
    **boris_inputs.events,
}
event_values = {**moseq_inputs.event_values, **boris_inputs.event_values}
intervals = {**moseq_inputs.intervals, **boris_inputs.intervals}
```

Event names must not collide when dictionaries are composed; prefix labels such
as `moseq:rear` and `boris:rear` if they represent different operational
definitions.

For a photometry signal, `fipha` consumes exactly these mappings plus the
aligned covariate and its mask:

```python
from fipha.encoding import EncodingSession

session = EncodingSession.from_arrays(
    subject="mouse-07",
    session="day-04",
    time=acquisition_time,
    response=processed_signal,
    events=events,
    event_values=event_values,
    intervals=intervals,
    continuous_covariates={"nose_speed": aligned_speed.values},
    continuous_covariate_validity={"nose_speed": aligned_speed.valid},
)
```

Its fitter combines response and selected-covariate masks using complete cases
without changing the time grid or bridging excluded spans, and reports every
session's retained denominator and exclusion reasons. See the
[event-kernel method contract](https://aeronjl.github.io/fipha/event-kernel-encoding/).

To model the behaviour itself across sessions rather than against a signal, the
trial-level summaries are computed here rather than assembled elsewhere. Declare
the trial timing on the same clock and reduce the covariate and the bouts over
trial windows:

```python
from behavio.observed.trialization import (
    MaximumValue,
    TrialTiming,
    TrialWindow,
    attach_trial_columns,
    reduce_covariate_to_trials,
)

timing = TrialTiming.from_arrays(
    subject="mouse-07",
    session="day-04",
    onset_s=boris.point_events["cue"],
    clock_id="acquisition",
).synchronized_to(clock_synchronization)

peak_speed = reduce_covariate_to_trials(
    aligned_speed,
    timing=timing,
    window=TrialWindow(start_offset_s=0.0, stop_offset_s=1.0),
    reducer=MaximumValue(),
    max_gap_s=2 / camera_fps,
    minimum_coverage=0.8,
)
study = attach_trial_columns(study, [peak_speed])
```

Each column arrives with its coverage and status, so a trial whose window ran
past the end of the video is visibly `NaN` rather than a confident number. See
[From seconds to trials](../observed-behaviour.md#from-seconds-to-trials), then
follow the [prospective validation workflow](../validation.md).

## What this example proves—and does not

The test suite proves structural composition for native-shaped arrays and
tables: identity selection, axis declaration, confidence masks, gap protection,
bout durations, point/state distinctions, and normalised progress.
Checksum-pinned current DeepLabCut, Keypoint-MoSeq, SLEAP and BORIS files pass
semantic parity checks, and those fixtures moved here from `fipha` unchanged.
The validation matrix distinguishes official payloads from writer-contract
artifacts with declared synthetic values. Synthetic tests verify known affine
offset/drift recovery and refusal of bad pulse evidence, but a real
synchronisation record is still missing. None of this proves
acquisition-specific clock accuracy, multi-animal identity stability, or a
biological result. See the
[validation matrix](../evidence/interoperability-validation-v0.1.md).
