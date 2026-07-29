# Behavioural covariates

`behavio.covariates` holds one type. A `BehaviorCovariate` is a named scalar
observed over time alongside behaviour — confidence-gated speed, pupil area, a
state probability — with the timestamps it was measured on and an explicit
validity mask. It is one shape in the wider
[observed-behaviour boundary](observed-behaviour.md).

The point of the type is that values and validity stay separate. A sample that
failed a confidence threshold keeps its computed value and is marked invalid,
rather than being overwritten with `NaN` or quietly filled. Non-finite values
are forced invalid on construction, so a mask can never claim more than the data
supports.

Covariates are produced rather than read from files: usually by
[`PoseTrajectory.speed()`](pose.md) or
[`BehaviorAnnotations.normalized_progress()`](ethograms.md), both of which pass
through `clock_id`, `source_version`, `source_artifact` and the
synchronisation lineage of whatever produced them.

## Alignment is not synchronisation

`aligned_to()` interpolates a covariate onto a target sampling grid and returns
a new covariate carrying the aligned mask. It refuses to work across a clock
mismatch, does not extrapolate, and never interpolates across an invalid sample
or a source gap wider than the declared maximum:

```python
aligned = speed.aligned_to(
    photometry_time,
    target_clock_id="photometry",
    max_gap_s=2 / camera_fps,
)
```

`align_to()` is the numeric-array convenience form, returning the values alone.

The clock check is the important part. Interpolation answers what value a
covariate has on another sampling grid; it does not establish that two streams
share a physical time base. Giving two unsynchronised clocks the same name is
not synchronisation — fit an explicit transform first, with
[`fit_clock_synchronization()`](clock-synchronization.md).

!!! note "Not the longitudinal clocks"
    `clock_id` here names a hardware time coordinate in seconds. It is
    unrelated to the [longitudinal clocks](clocks-and-transforms.md) that place
    a `Study` in learning time.

## Reduction to trial columns

`reduce_covariate_to_trials()` in `behavio.trialization` turns a covariate into
one value per declared trial, and `attach_trial_columns()` joins those values
onto a `Study` by `subject`/`session`/`trial`. `MeanValue`, `MedianValue`,
`MinimumValue` and `MaximumValue` ship; the `TrialCovariateReducer` protocol is
open.

The mask survives that step too. Reduction takes the same `max_gap_s` rule
`aligned_to()` uses, and reports per trial how much of the window was actually
covered by valid observation. A trial whose window was mostly invalid, mostly a
gap, or entirely past the end of the recording returns `NaN` with a status
naming which of those happened, so a partially observed trial is never mistaken
for a confident measurement. See
[From seconds to trials](observed-behaviour.md#from-seconds-to-trials).

Every shipped reducer reads only one trial's own window, so it is
fold-independent and safe to apply before splitting. A reduction that must
*learn* a threshold or a baseline is not a reducer: produce the raw column here
and fit the learned step inside a training fold with `fit_transform_split()`.

`subject` and `session` accept any hashable identifier, matching `Study`, so an
integer subject id joins without a lossy `str()`.

Signatures are in the [observed behaviour API](reference/observed-behaviour.md).
