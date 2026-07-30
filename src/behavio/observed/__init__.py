"""Continuously observed behaviour, and the bridge from it to trial-level columns.

Everything in this package is measured against wall-clock seconds rather than against a
trial index: pose keypoints, ethogram intervals, timestamped covariates, and the physical
device clocks those signals are stamped on. :mod:`behavio.observed.trialization` is the
one-way door out -- it reduces a continuous signal over a declared per-trial window and
joins the result onto a :class:`~behavio.trials.Study`.

The package is closed under imports: nothing here reaches into the modelling, validation
or protocol layers, and nothing in those layers is required to read a pose file.
"""

from behavio.observed.covariates import BehaviorCovariate
from behavio.observed.device_clocks import (
    DeviceClockPulses,
    DeviceClockSync,
    DeviceClockSyncSpec,
    fit_device_clock_sync,
)
from behavio.observed.ethograms import (
    BehaviorAnnotations,
    BehaviorInterval,
    IntervalEncodingInputs,
    annotations_from_boris,
    annotations_from_boris_aggregated_file,
    annotations_from_boris_tabular_file,
    annotations_from_moseq,
    annotations_from_moseq_results_h5,
)
from behavio.observed.interval_policy import (
    ContextualizeIntervals,
    FilterIntervals,
    IntervalOperation,
    IntervalPolicy,
    IntervalPolicyContext,
    IntervalPolicyLedgerEntry,
    IntervalPolicyResult,
    IntervalSnapshot,
    MergeIntervals,
    ResolveIntervalOverlaps,
    SplitIntervals,
    apply_interval_policy,
)
from behavio.observed.pose import (
    PoseTrajectory,
    pose_from_deeplabcut,
    pose_from_deeplabcut_file,
    pose_from_movement,
    pose_from_sleap,
    pose_from_sleap_analysis_h5,
)
from behavio.observed.trialization import (
    EventCount,
    FirstOccurrenceLatency,
    FractionOfTimeInState,
    MaximumValue,
    MeanValue,
    MedianValue,
    MinimumValue,
    TrialAnnotationReducer,
    TrialCovariateReducer,
    TrialCoverageStatus,
    TrializationError,
    TrialReduction,
    TrialTiming,
    TrialWindow,
    attach_trial_columns,
    reduce_annotations_to_trials,
    reduce_covariate_to_trials,
    trial_timing_from_events,
)

__all__ = [
    "BehaviorAnnotations",
    "BehaviorCovariate",
    "BehaviorInterval",
    "ContextualizeIntervals",
    "DeviceClockPulses",
    "DeviceClockSync",
    "DeviceClockSyncSpec",
    "EventCount",
    "FilterIntervals",
    "FirstOccurrenceLatency",
    "FractionOfTimeInState",
    "IntervalEncodingInputs",
    "IntervalOperation",
    "IntervalPolicy",
    "IntervalPolicyContext",
    "IntervalPolicyLedgerEntry",
    "IntervalPolicyResult",
    "IntervalSnapshot",
    "MaximumValue",
    "MeanValue",
    "MedianValue",
    "MergeIntervals",
    "MinimumValue",
    "PoseTrajectory",
    "ResolveIntervalOverlaps",
    "SplitIntervals",
    "TrialAnnotationReducer",
    "TrialCovariateReducer",
    "TrialCoverageStatus",
    "TrialReduction",
    "TrialTiming",
    "TrialWindow",
    "TrializationError",
    "annotations_from_boris",
    "annotations_from_boris_aggregated_file",
    "annotations_from_boris_tabular_file",
    "annotations_from_moseq",
    "annotations_from_moseq_results_h5",
    "apply_interval_policy",
    "attach_trial_columns",
    "fit_device_clock_sync",
    "pose_from_deeplabcut",
    "pose_from_deeplabcut_file",
    "pose_from_movement",
    "pose_from_sleap",
    "pose_from_sleap_analysis_h5",
    "reduce_annotations_to_trials",
    "reduce_covariate_to_trials",
    "trial_timing_from_events",
]
