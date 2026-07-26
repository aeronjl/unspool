# Clocks and fold-fitted temporal transforms

A longitudinal study rarely has one privileged time axis. Session number, cumulative
exposure, elapsed calendar time, protocol phase, and distance from a learning event answer
different scientific questions. Unspool records that meaning in a `ClockSpec` instead of
silently treating every numeric column as interchangeable.

## Clock metadata

Every clock declares:

- its source column and scientific `kind`;
- whether values are comparable within a subject or globally;
- its unit;
- whether it is numeric and expected to be non-decreasing within a subject; and
- whether missing values are allowed.

`session_order_clock()` selects the canonical session chronology. The first derived clocks
are `with_cumulative_trial_clock()` and `with_elapsed_time_clock()`:

```python
from unspool import with_cumulative_trial_clock, with_elapsed_time_clock

study = with_cumulative_trial_clock(study).study
study = with_elapsed_time_clock(
    study,
    source="timestamp",
    output="elapsed_days",
    unit="days",
).study
```

Both preserve source row order. Cumulative trial counts follow explicit subject/session/
trial chronology and count observed rows; missing trial numbers are not invented. Elapsed
time is measured from each subject's first observation. A numeric time source is assumed to
already use the requested output unit; datetime sources are converted to it.

These design-derived clocks should normally be built on the complete observed `Study`
before creating folds. Computing cumulative exposure independently on a test subset would
reset its origin and change its meaning.

Categorical protocol stages can be described explicitly rather than coerced to numbers:

```python
from unspool import ClockKind, ClockScope, ClockSpec

phase = ClockSpec(
    "task_phase",
    ClockKind.TASK_PHASE,
    scope=ClockScope.GLOBAL,
    numeric=False,
    monotonic_within_subject=False,
)
phase.validate(study)
```

Declaring global scope is a scientific assertion that a phase label is comparable across
subjects. Unspool validates the declaration but cannot establish that assertion from the
table alone.

## Learned clocks belong inside the fold

A behavioural landmark is different from a design clock because its value is estimated
from outcomes. Estimating it once from the complete dataset leaks held-out behaviour into
the training pipeline. `ThresholdLandmarkClock` therefore has separate `fit()` and
`transform()` operations:

```python
from unspool import (
    ClockKind,
    ClockSpec,
    ThresholdLandmarkClock,
    fit_transform_splits,
    forward_session_splits,
)

landmark = ThresholdLandmarkClock(
    clock=ClockSpec(
        "cumulative_trial",
        ClockKind.CUMULATIVE_TRIAL,
        unit="observed_trial",
    ),
    metric="correct",
    output="trials_since_learning",
    threshold=0.8,
    window=20,
    consecutive=3,
)

results = fit_transform_splits(
    landmark,
    study,
    forward_session_splits(study, min_train_sessions=3),
)
```

At each forecasting origin, the helper fits one landmark per subject using only training
rows, freezes those values, and applies them to both sides of the fold. It rejects a
non-prospective split unless `require_prospective=False` is deliberately supplied.

The threshold landmark uses the trailing rolling mean of `metric`. Its landmark is the
source-clock value at the end of the first run of `consecutive` qualifying windows. This is
the time at which the criterion has been confirmed; it is not backdated to a putative onset.
`direction="below"` reverses the threshold comparison.

## Failure and provenance are data

The default `on_missing="error"` raises `LandmarkNotFoundError` if a training fold never
reaches the criterion. This prevents a failed alignment from silently disappearing. With
`on_missing="nan"`, that subject's relative clock is explicitly missing instead.

Every fitted transform exposes immutable provenance:

```python
result = results[0]
result.fitted_transform.provenance.n_fit_trials
result.fitted_transform.provenance.fit_subjects
result.fitted_transform.provenance.learned_values
result.fitted_transform.provenance.transform_signature
```

The retained state makes it possible to audit which observations could have influenced a
coordinate. A subject absent from the fitting data cannot be transformed using a borrowed
landmark.

## Landmark uncertainty remains inside the fold

For a binary metric, `BootstrapThresholdLandmarkClock` makes uncertainty estimation part
of the same generic fold-transform path:

```python
from unspool import BootstrapThresholdLandmarkClock

uncertain_landmark = BootstrapThresholdLandmarkClock(
    landmark,
    n_resamples=500,
    seed=19,
    smoothing_window=10,
    interval_level=0.9,
)
results = fit_transform_splits(uncertain_landmark, study, splits)

fitted = results[0].fitted_transform
estimate = fitted.uncertainty.estimates["mouse-1"]
estimate.point
estimate.resolution_rate
estimate.median
estimate.interval
clock_draws = fitted.transform_samples(test_study)
```

The ordinary transformed studies still use the observed-data point landmark. Uncertainty
draws are separate and immutable: `transform_samples()` returns a draw-by-trial matrix of
alternative landmark-relative coordinates. It only reads subject identity and the source
clock, so changing held-out metric values cannot alter the distribution.

The current procedure causally smooths each subject's binary training metric using a
declared window and Jeffreys regularization, samples plug-in Bernoulli trajectories at the
original chronological positions, and reapplies the unchanged threshold rule. Failed
detections remain `NaN`; intervals condition on resolved draws and must be accompanied by
`resolution_rate`. This is a design-specific parametric bootstrap—not a posterior credible
interval—and it assumes conditional independence around the smoothed trajectory.

## Current boundary

This API makes clock construction and one operational landmark criterion explicit. It does
not claim that the threshold is a universal definition of learning, choose a criterion on
held-out data, model residual serial dependence in its first bootstrap, or automatically
propagate clock draws through every downstream model. Alternative landmark definitions
should implement the same training-only transform contract and be compared as modelling
choices. Within-session rolling origins are available, but automatic composition of
arbitrary fitted transforms with model evaluation remains future work.
