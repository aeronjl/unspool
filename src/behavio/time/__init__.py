"""Learning time: the temporal coordinate a longitudinal claim is made against.

:mod:`behavio.time.clocks` names that coordinate explicitly -- session order, cumulative
trial, elapsed time, task phase -- so that "improvement over time" states which time.
:mod:`behavio.time.landmarks` supplies clocks whose *origin* is learned rather than
declared, and :mod:`behavio.time.transforms` is the fold discipline that keeps a learned
origin from leaking into the rows it will be scored on.

Unqualified "clock" means one of these. The physical hardware clocks that observed signals
are stamped on are :mod:`behavio.observed.device_clocks`, and every name there says
``DeviceClock``.
"""

from behavio.time.clocks import (
    ClockedStudy,
    ClockKind,
    ClockScope,
    ClockSpec,
    ClockValidationError,
    session_order_clock,
    with_cumulative_trial_clock,
    with_elapsed_time_clock,
)
from behavio.time.landmarks import (
    BootstrapThresholdLandmarkClock,
    FittedThresholdLandmarkClock,
    LandmarkClockSamples,
    LandmarkNotFoundError,
    LandmarkUncertaintyEstimate,
    ThresholdLandmarkClock,
    ThresholdLandmarkUncertainty,
)
from behavio.time.transforms import (
    FittedStudyTransform,
    FoldTransformResult,
    StudyTransform,
    TransformProvenance,
    fit_transform_split,
    fit_transform_splits,
)

__all__ = [
    "BootstrapThresholdLandmarkClock",
    "ClockKind",
    "ClockScope",
    "ClockSpec",
    "ClockValidationError",
    "ClockedStudy",
    "FittedStudyTransform",
    "FittedThresholdLandmarkClock",
    "FoldTransformResult",
    "LandmarkClockSamples",
    "LandmarkNotFoundError",
    "LandmarkUncertaintyEstimate",
    "StudyTransform",
    "ThresholdLandmarkClock",
    "ThresholdLandmarkUncertainty",
    "TransformProvenance",
    "fit_transform_split",
    "fit_transform_splits",
    "session_order_clock",
    "with_cumulative_trial_clock",
    "with_elapsed_time_clock",
]
