# Synchronise behaviour and recording clocks

!!! warning "Experimental"
    This v0.1 boundary validates an affine mapping from already matched hardware
    pulses. It does not discover pulse correspondences, repair missing pulses, or
    establish that the chosen residual and drift thresholds are appropriate for a
    particular behavioural timescale.

!!! note "Not a longitudinal clock"
    This page is about hardware time in seconds. The
    [longitudinal clocks](clocks-and-transforms.md) that place a `Study` in
    learning time are a separate concept.

## Scientific question

Can timestamps produced by a camera or behavioural computer be mapped onto a
recording acquisition clock - photometry, ephys, or another behavioural device -
with enough evidence to support event alignment?

Behavio models the target clock as

\[
t_{target} = a + b t_{source},
\]

where \(a\) is offset, \(b\) is clock scale, and drift is
\((b-1)\times10^6\) parts per million. The inputs are explicit one-to-one pulse
pairs observed on both clocks. Pulse matching remains an acquisition-specific
decision and is never inferred from nearest timestamps.

## Fit an evidence-bearing transform

```python
from behavio.sync import (
    ClockPulseMatches,
    ClockSynchronizationSpec,
    fit_clock_synchronization,
)

matches = ClockPulseMatches.from_arrays(
    source_clock_id="video",
    target_clock_id="photometry",
    source_time_s=[0.0, 20.0, 40.0, 60.0],
    target_time_s=[0.3500, 20.3526, 40.3547, 60.3572],
    match_labels=["start", "pulse-1", "pulse-2", "stop"],
)
spec = ClockSynchronizationSpec(
    maximum_absolute_residual_s=0.002,
    maximum_drift_ppm=250,
    minimum_matches=4,
    minimum_source_span_s=50.0,
)
synchronization = fit_clock_synchronization(matches, spec)
synchronization.to_json()
```

Both acceptance thresholds are prospective inputs. The result retains:

- source and target clock identities;
- all paired, fitted and residual pulse timestamps;
- intercept, scale and signed drift in ppm;
- RMS, median-absolute and maximum-absolute residual error;
- the thresholds, minimum pulse count and minimum source span; and
- a deterministic `synchronization_id` derived from the pulse evidence and
  specification.

Fitting refuses unequal pulse counts, non-monotonic pairs, fewer than three
matches, inadequate temporal span, non-positive mappings, excessive drift, or a
maximum residual above the declared threshold. These are structural safeguards,
not proof that an affine clock model is scientifically sufficient.

## Transform behaviour without discarding provenance

```python
pose_on_photometry_clock = synchronization.synchronize_pose(pose)
speed = pose_on_photometry_clock.speed(minimum_confidence=0.9)

annotations_on_photometry_clock = synchronization.synchronize_annotations(annotations)
```

Equivalent `synchronize_covariate()` support is available for an already derived
`BehaviorCovariate`. Values, confidence, validity masks, point-versus-interval
semantics, units and source artifacts are unchanged. The target `clock_id` is set
and the stable synchronization ID is appended to
`clock_synchronization_ids`. Derived speed and progress covariates retain that
lineage.

Once both data streams genuinely use the target clock, gap-safe interpolation
can proceed:

```python
aligned_speed = speed.aligned_to(
    photometry_time,
    target_clock_id="photometry",
    max_gap_s=2 / camera_fps,
)
```

## Interpolation is not synchronisation

Interpolation answers what value a synchronized covariate has on another sampling
grid. Synchronization answers which physical instant a timestamp denotes. The
operations are therefore ordered:

1. identify pulse correspondences without accessing analysis outcomes;
2. fit and validate the clock transform against prospective thresholds;
3. transform pose, annotations or covariates and retain the evidence ID; then
4. interpolate only within valid source runs.

Renaming `video` to `photometry`, or directly interpolating between them, skips the
second step and is rejected by the typed boundary.

## Trial timing is transformed the same way

Trial onsets are a physical measurement like any other, so
[`TrialTiming`](observed-behaviour.md#from-seconds-to-trials) moves between
clocks through the same accepted mapping rather than around it:

```python
timing = timing.synchronized_to(synchronization, maximum_extrapolation_s=0.0)
```

The returned timing carries the target `clock_id` and appends the
`synchronization_id` to its lineage. `reduce_covariate_to_trials()` and
`reduce_annotations_to_trials()` refuse a covariate on one clock and onsets on
another, so the only route from a camera-clock ethogram to acquisition-clock
trial columns runs through an accepted synchronisation.

## Extrapolation policy

By default, `transform_time()` and the typed synchronization methods refuse every
timestamp outside the first and last matched source pulses. A bounded allowance
must be explicit:

```python
mapped = synchronization.transform_time(
    video_time,
    maximum_extrapolation_s=0.5,
)
```

An accepted in-range fit does not validate unbounded extrapolation. Pulse trains
should bracket the analyzed recording whenever possible.

## Limitations and failure interpretation

An affine mapping captures a stable rate difference, not nonlinear clock warping,
dropped or duplicated frames, variable buffering delay, or a mistaken pulse
correspondence. A low residual can also be misleading when only three closely
spaced pulses are used. Inspect pulse span and residual structure, not only a pass
flag.

Thresholds should be derived from acquisition precision and the shortest event
timing distinction the study intends to interpret. A failed threshold means the
streams are not admissible under that declared analysis—not that a more flexible
mapping should automatically be fitted after seeing analysis outcomes.
