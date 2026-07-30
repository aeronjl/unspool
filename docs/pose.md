# Pose trajectories

`behavio.observed.pose` holds one type and the readers that fill it. A `PoseTrajectory`
is one keypoint of one tracked individual: coordinates, a confidence-like score,
timestamps, and the identity fields that make those numbers interpretable. It is
one shape in the wider [observed-behaviour boundary](observed-behaviour.md).

The readers for DeepLabCut, SLEAP and `movement` live in this module because a
pose is what they produce. They are deliberately thin: they translate a tool's
file or in-memory shape into `PoseTrajectory` and refuse ambiguous selections
rather than guessing.

## Installing

The in-memory readers need only NumPy. The DeepLabCut and SLEAP *file* readers
need HDF5 and pandas:

```bash
pip install "behavio[readers]"
```

## DeepLabCut

`pose_from_deeplabcut()` reads a pandas-like result without taking a hard
dependency on DeepLabCut. `pose_from_deeplabcut_file()` reads prediction CSV or
pandas HDF5 with the optional `readers` dependencies. Both require a scorer or
individual when the MultiIndex would otherwise select more than one
x/y/likelihood triplet.

```python
from importlib.metadata import version

from behavio.observed.pose import pose_from_deeplabcut_file

pose = pose_from_deeplabcut_file(
    "sessionDLC_resnet50.h5",
    subject="mouse-07",
    session="day-04",
    keypoint="nose",
    scorer="DLC_resnet50_project",
    fps=30.0,
    clock_id="camera-0",
    source_version=version("deeplabcut"),
)
speed = pose.speed(minimum_confidence=0.9)
```

The likelihood threshold is an analysis choice and must be declared. The adapter
does not adopt a universal cutoff or interpolate low-confidence coordinates.

## SLEAP

`pose_from_sleap()` consumes dense arrays. `pose_from_sleap_analysis_h5()` reads
the file and uses its dimension attributes when present. Legacy files have no
dimension attributes, so the caller must declare them; Python-native and
MATLAB-compatible presets use different axis orders.

```python
from behavio.observed.pose import pose_from_sleap_analysis_h5

pose = pose_from_sleap_analysis_h5(
    "predictions.analysis.h5",
    node="nose",
    dims=("track", "xy", "node", "frame"),
    subject="mouse-07",
    session="day-04",
    track=0,
    fps=30.0,
)
```

SLEAP point scores are confidence-like values, not guaranteed probabilities. An
official legacy fixture contains scores above one, so only non-negativity is
assumed; thresholds must be calibrated to the producing SLEAP version and model.

SLEAP also exports NWB through `ndx-pose`. `fipha` provides native inspection,
explicit estimator selection, 2D/3D import, schema-valid export, and a tested
file round trip against these `PoseTrajectory` objects; see its
[native ndx-pose contract](https://aeronjl.github.io/fipha/ndx-pose-interoperability/).

## movement

[`movement`](https://github.com/neuroinformatics-unit/movement) is the
maintained community package for pose input and output, built by the
Neuroinformatics Unit on the same xarray substrate. It is BSD-3-Clause and
actively released. Behavio consumes its output rather than competing with it:

```python
from movement.io import load_poses

from behavio.observed.pose import pose_from_movement

dataset = load_poses.from_dlc_file("sessionDLC_resnet50.h5", fps=30.0)
pose = pose_from_movement(
    dataset,
    subject="mouse-07",
    session="day-04",
    keypoint="nose",
    clock_id="camera-0",
)
speed = pose.speed(minimum_confidence=0.9)
```

This gives access to `movement`'s readers that Behavio does not implement,
including Anipose, Lightning Pose and its NWB path. `pose_from_movement()`
duck-types on the xarray interface and never imports `movement`, so it adds no
dependency and no Python version floor.

`movement` is deliberately **not** a dependency. Measured against version 0.17.0
on Python 3.12.13:

- it requires Python `>=3.12.0`, while this package supports `>=3.11`;
- resolving it alone pulls **119 packages and 720 MB**, including Qt, OpenCV,
  `skia-python`, `imageio-ffmpeg` and `numba`, and pins `netCDF4<1.7.3`;
- `tables` is one of its own core dependencies, so depending on it would not
  remove PyTables from the `readers` extra;
- its SLEAP reader applies a hardcoded axis permutation and ignores the `dims`
  attribute, so it cannot read the current sleap-io analysis HDF5 that
  `pose_from_sleap_analysis_h5()` reads.

Three things in this boundary have no counterpart in `movement`, and stay here:

- **Ethogram and annotation types.** `movement` covers poses and bounding boxes;
  point events and behavioural intervals are Behavio types, in
  [`behavio.observed.ethograms`](ethograms.md).
- **Foreign-clock alignment.** `movement` interpolates gaps along a pose's own
  time axis. It has no clock identity, no synchronisation fit and no lineage, so
  [`fit_device_clock_sync()`](clock-synchronization.md) and
  [`BehaviorCovariate.aligned_to()`](covariates.md) remain here.
- **Value/mask separation.** `movement` gates confidence by overwriting
  coordinates with `NaN`, and `compute_speed` uses a central difference. On a
  gated sample that combination reports a speed *at* the gated frame while
  blanking its two well-estimated neighbours. `PoseTrajectory.speed()` instead
  invalidates a step when either endpoint fails the threshold and keeps the
  value beside its mask, as
  [SDR-0033](https://github.com/aeronjl/fipha/blob/main/docs/decisions/0033-retain-validity-masks-without-compressing-time.md)
  requires. Pass the dataset to `pose_from_movement()` *before* running
  `movement.filtering.filter_by_confidence`.

`movement` stores coordinates as `float32`. Values widen to `float64` on import
but carry `float32` precision, so a file read through both paths will not agree
bit-for-bit. See
[SDR-0059](decisions/0059-consume-movement-datasets-without-depending-on-movement.md).

## From pose to a modelled covariate

`PoseTrajectory.speed()` is the one derivation this module performs. It computes
pairwise speed, invalidates a step whenever either endpoint fails the declared
confidence threshold, and returns a
[`BehaviorCovariate`](covariates.md) carrying the pose's clock identity and
synchronisation lineage. From there the composition continues through
[clock synchronisation](clock-synchronization.md) and
[interval policies](interval-policy.md), as described in the
[boundary overview](observed-behaviour.md).

## From pose to a `Study`

A pose is in seconds; a [`Study`](data-contract.md) is in trials and has no time
column. `behavio.observed.trialization` closes that gap. Declare when the trials started
on the pose's own clock, then reduce the derived covariate over trial windows:

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
    onset_s=cue_times,
    clock_id="camera-0",
)
peak_speed = reduce_covariate_to_trials(
    pose.speed(minimum_confidence=0.9),
    timing=timing,
    window=TrialWindow(start_offset_s=0.0, stop_offset_s=1.0),
    reducer=MaximumValue(),
    max_gap_s=2 / 30.0,
    minimum_coverage=0.8,
)
study = attach_trial_columns(study, [peak_speed])
```

`subject` and `session` accept any hashable identifier, matching `Study`, so an
integer subject id joins without a lossy `str()`. Trials whose window ran past
the end of the video, straddled a tracking dropout, or fell below
`minimum_coverage` arrive as `NaN` with an explanatory status column rather than
as a confident number. See
[From seconds to trials](observed-behaviour.md#from-seconds-to-trials).

Signatures are in the [observed behaviour API](reference/observed-behaviour.md).
