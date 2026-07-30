"""Matched comparison of competing accounts, on scores and on trajectory shape.

:mod:`behavio.compare.models` is the prospective model contest: the same folds, the same
aggregation units, one paired contrast per candidate against a reference, one bootstrap and
one simultaneous family. :mod:`behavio.compare.parameter_trajectories` compares something
else entirely -- the *shape* a fitted parameter traces across sessions, between groups --
and says so in its name, because a trajectory in this package can also mean an animal
moving (:class:`~behavio.observed.pose.PoseTrajectory`).
"""

from behavio.compare.models import (
    DEFAULT_COMPARISON_METRICS,
    BootstrapInterval,
    ComparisonFamily,
    ComparisonMultiplicity,
    NestedProspectiveSelectionReport,
    NestedSelectionFold,
    PairedComparison,
    ProspectiveComparisonReport,
    ProspectiveModelResult,
    ScoreMetric,
    UndeclaredMetric,
    UnscoreableByBrier,
    bootstrap_interval,
    bootstrap_unit_draws,
    compare_models,
    nested_select_model,
    paired_comparisons,
)
from behavio.compare.parameter_trajectories import (
    GroupParameterTrajectorySummary,
    PairwiseTrajectoryShapeComparison,
    ParameterTrajectoryPanel,
    TrajectoryReplicationAudit,
    TrajectoryShapeComparisonReport,
    audit_trajectory_replication,
    compare_trajectory_shapes,
)

__all__ = [
    "DEFAULT_COMPARISON_METRICS",
    "BootstrapInterval",
    "ComparisonFamily",
    "ComparisonMultiplicity",
    "GroupParameterTrajectorySummary",
    "NestedProspectiveSelectionReport",
    "NestedSelectionFold",
    "PairedComparison",
    "PairwiseTrajectoryShapeComparison",
    "ParameterTrajectoryPanel",
    "ProspectiveComparisonReport",
    "ProspectiveModelResult",
    "ScoreMetric",
    "TrajectoryReplicationAudit",
    "TrajectoryShapeComparisonReport",
    "UndeclaredMetric",
    "UnscoreableByBrier",
    "audit_trajectory_replication",
    "bootstrap_interval",
    "bootstrap_unit_draws",
    "compare_models",
    "compare_trajectory_shapes",
    "nested_select_model",
    "paired_comparisons",
]
