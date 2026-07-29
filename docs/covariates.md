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

Signatures are in the [observed behaviour API](reference/observed-behaviour.md).
