"""Cross-group trajectory geometry with explicit replication requirements."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from unspool.comparison import BootstrapInterval
from unspool.models.base import _protected_array


@dataclass(frozen=True, slots=True)
class TrajectoryPanel:
    """One aligned parameter trajectory per subject on a declared common clock.

    Rows of ``values`` are subjects and columns are positions on ``grid``. Alignment is
    deliberately external to this object: constructing a panel never interpolates, shifts,
    or time-warps a trajectory.
    """

    grid: NDArray[np.float64]
    values: NDArray[np.float64]
    subjects: tuple[Any, ...]
    groups: tuple[Any, ...]
    clock_name: str
    parameter_name: str
    group_name: str = "lab"

    def __post_init__(self) -> None:
        grid = _protected_array(self.grid, dtype=np.float64)
        values = _protected_array(self.values, dtype=np.float64)
        subjects = tuple(
            _validated_identifier(value, f"subjects[{index}]")
            for index, value in enumerate(self.subjects)
        )
        groups = tuple(
            _validated_identifier(value, f"groups[{index}]")
            for index, value in enumerate(self.groups)
        )
        if grid.ndim != 1 or grid.size < 2:
            raise ValueError("grid must be one-dimensional with at least two positions")
        if not np.all(np.isfinite(grid)) or np.any(np.diff(grid) <= 0):
            raise ValueError("grid must be finite and strictly increasing")
        if values.ndim != 2 or values.shape != (len(subjects), grid.size):
            raise ValueError("values must contain one aligned trajectory per subject")
        if not np.all(np.isfinite(values)):
            raise ValueError("trajectory values must be finite")
        if not subjects or len(set(subjects)) != len(subjects):
            raise ValueError("subjects must be non-empty and unique")
        if len(groups) != len(subjects):
            raise ValueError("groups must contain one identifier per subject")
        if len(set(groups)) < 2:
            raise ValueError("a cross-group panel requires at least two groups")
        for name, value in (
            ("clock_name", self.clock_name),
            ("parameter_name", self.parameter_name),
            ("group_name", self.group_name),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        object.__setattr__(self, "grid", grid)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "subjects", subjects)
        object.__setattr__(self, "groups", groups)

    @property
    def group_order(self) -> tuple[Any, ...]:
        """Groups in stable first-appearance order."""

        return tuple(dict.fromkeys(self.groups))

    @property
    def group_sizes(self) -> Mapping[Any, int]:
        """Number of independent subjects contributing to each group."""

        return MappingProxyType(
            {group: sum(value == group for value in self.groups) for group in self.group_order}
        )


@dataclass(frozen=True, slots=True)
class TrajectoryReplicationAudit:
    """Whether group comparisons have independent within-group replication."""

    group_name: str
    group_sizes: Mapping[Any, int]
    minimum_subjects_per_group: int
    singleton_groups: tuple[Any, ...]
    under_replicated_groups: tuple[Any, ...]
    inferentially_ready: bool

    def __post_init__(self) -> None:
        if not isinstance(self.group_name, str) or not self.group_name:
            raise ValueError("group_name must be a non-empty string")
        _require_positive_integer(self.minimum_subjects_per_group, "minimum_subjects_per_group")
        sizes = dict(self.group_sizes)
        if len(sizes) < 2 or any(
            isinstance(size, bool) or not isinstance(size, int) or size < 1
            for size in sizes.values()
        ):
            raise ValueError("group_sizes must contain at least two positive counts")
        singletons = tuple(group for group, size in sizes.items() if size == 1)
        under_replicated = tuple(
            group for group, size in sizes.items() if size < self.minimum_subjects_per_group
        )
        if tuple(self.singleton_groups) != singletons:
            raise ValueError("singleton_groups must match group_sizes")
        if tuple(self.under_replicated_groups) != under_replicated:
            raise ValueError("under_replicated_groups must match group_sizes")
        if self.inferentially_ready != (not under_replicated):
            raise ValueError("inferentially_ready must reflect the replication threshold")
        object.__setattr__(self, "group_sizes", MappingProxyType(sizes))
        object.__setattr__(self, "singleton_groups", singletons)
        object.__setattr__(self, "under_replicated_groups", under_replicated)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe replication report."""

        return {
            "group_name": self.group_name,
            "group_sizes": [
                {"group": _json_value(group), "subjects": size}
                for group, size in self.group_sizes.items()
            ],
            "minimum_subjects_per_group": self.minimum_subjects_per_group,
            "singleton_groups": [_json_value(group) for group in self.singleton_groups],
            "under_replicated_groups": [
                _json_value(group) for group in self.under_replicated_groups
            ],
            "inferentially_ready": self.inferentially_ready,
        }


@dataclass(frozen=True, slots=True)
class GroupTrajectorySummary:
    """Equal-subject group mean and its level/amplitude decomposition."""

    group: Any
    subjects: tuple[Any, ...]
    mean_values: NDArray[np.float64]
    level: float
    amplitude: float

    def __post_init__(self) -> None:
        group = _validated_identifier(self.group, "group")
        subjects = tuple(
            _validated_identifier(value, f"subjects[{index}]")
            for index, value in enumerate(self.subjects)
        )
        values = _protected_array(self.mean_values, dtype=np.float64)
        if not subjects or len(set(subjects)) != len(subjects):
            raise ValueError("summary subjects must be non-empty and unique")
        if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
            raise ValueError("mean_values must be a finite one-dimensional trajectory")
        if not np.isfinite(self.level) or not np.isfinite(self.amplitude) or self.amplitude < 0:
            raise ValueError("trajectory level and amplitude must be finite")
        object.__setattr__(self, "group", group)
        object.__setattr__(self, "subjects", subjects)
        object.__setattr__(self, "mean_values", values)

    def to_dict(self) -> dict[str, Any]:
        """Return the group mean and decomposition in JSON-safe form."""

        return {
            "group": _json_value(self.group),
            "subjects": [_json_value(subject) for subject in self.subjects],
            "mean_values": self.mean_values.tolist(),
            "level": self.level,
            "amplitude": self.amplitude,
        }


@dataclass(frozen=True, slots=True)
class PairwiseTrajectoryShapeComparison:
    """Level, amplitude, and shape contrasts between two group mean trajectories.

    Signed differences are always ``left - right``. Raw and centered distances retain the
    parameter's units. Scale-free shape distance is the weighted distance between centered,
    unit-amplitude curves and lies between zero and two. It is unresolved for a flat curve.
    """

    left_group: Any
    right_group: Any
    raw_distance: BootstrapInterval
    level_difference: BootstrapInterval
    centered_distance: BootstrapInterval
    amplitude_difference: BootstrapInterval
    shape_distance: BootstrapInterval | None
    effective_shape_resamples: int

    def __post_init__(self) -> None:
        left = _validated_identifier(self.left_group, "left_group")
        right = _validated_identifier(self.right_group, "right_group")
        if left == right:
            raise ValueError("pairwise comparisons require distinct groups")
        if self.raw_distance.estimate < 0 or self.centered_distance.estimate < 0:
            raise ValueError("trajectory distances must be non-negative")
        if self.shape_distance is not None and not 0 <= self.shape_distance.estimate <= 2 + 1e-12:
            raise ValueError("shape distance must lie between zero and two")
        if (
            isinstance(self.effective_shape_resamples, bool)
            or not isinstance(self.effective_shape_resamples, int)
            or self.effective_shape_resamples < 0
        ):
            raise ValueError("effective_shape_resamples must be a non-negative integer")
        object.__setattr__(self, "left_group", left)
        object.__setattr__(self, "right_group", right)

    def to_dict(self) -> dict[str, Any]:
        """Return estimates, intervals, and comparison direction."""

        return {
            "left_group": _json_value(self.left_group),
            "right_group": _json_value(self.right_group),
            "difference_direction": "left minus right",
            "raw_distance": self.raw_distance.to_dict(),
            "level_difference": self.level_difference.to_dict(),
            "centered_distance": self.centered_distance.to_dict(),
            "amplitude_difference": self.amplitude_difference.to_dict(),
            "shape_distance": (
                None if self.shape_distance is None else self.shape_distance.to_dict()
            ),
            "effective_shape_resamples": self.effective_shape_resamples,
        }


@dataclass(frozen=True, slots=True)
class TrajectoryShapeComparisonReport:
    """Cross-group trajectory comparison on one fixed aligned panel."""

    clock_name: str
    parameter_name: str
    group_name: str
    grid: NDArray[np.float64]
    replication_audit: TrajectoryReplicationAudit
    group_summaries: tuple[GroupTrajectorySummary, ...]
    pairwise_comparisons: tuple[PairwiseTrajectoryShapeComparison, ...]
    bootstrap_resamples: int
    bootstrap_seed: int
    confidence_level: float

    def __post_init__(self) -> None:
        grid = _protected_array(self.grid, dtype=np.float64)
        summaries = tuple(self.group_summaries)
        comparisons = tuple(self.pairwise_comparisons)
        _require_positive_integer(self.bootstrap_resamples, "bootstrap_resamples")
        _require_nonnegative_integer(self.bootstrap_seed, "bootstrap_seed")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must lie strictly between zero and one")
        if grid.ndim != 1 or grid.size < 2 or np.any(np.diff(grid) <= 0):
            raise ValueError("report grid must be one-dimensional and strictly increasing")
        if not self.replication_audit.inferentially_ready:
            raise ValueError("comparison reports require replicated subjects in every group")
        if len(summaries) < 2:
            raise ValueError("comparison reports require at least two group summaries")
        expected_pairs = len(summaries) * (len(summaries) - 1) // 2
        if len(comparisons) != expected_pairs:
            raise ValueError("pairwise comparisons must cover every unordered group pair")
        object.__setattr__(self, "grid", grid)
        object.__setattr__(self, "group_summaries", summaries)
        object.__setattr__(self, "pairwise_comparisons", comparisons)

    def comparison_for(
        self, left_group: Any, right_group: Any
    ) -> PairwiseTrajectoryShapeComparison:
        """Return a named pair, reversing signed intervals when requested in reverse order."""

        for comparison in self.pairwise_comparisons:
            if comparison.left_group == left_group and comparison.right_group == right_group:
                return comparison
            if comparison.left_group == right_group and comparison.right_group == left_group:
                return _reverse_comparison(comparison)
        raise KeyError(f"no trajectory comparison for {left_group!r} and {right_group!r}")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe comparison report with uncertainty provenance."""

        return {
            "clock_name": self.clock_name,
            "parameter_name": self.parameter_name,
            "group_name": self.group_name,
            "grid": self.grid.tolist(),
            "alignment": "fixed common grid; no interpolation or time warping",
            "subject_weighting": "equal within each group",
            "replication_audit": self.replication_audit.to_dict(),
            "group_summaries": [summary.to_dict() for summary in self.group_summaries],
            "pairwise_comparisons": [
                comparison.to_dict() for comparison in self.pairwise_comparisons
            ],
            "bootstrap": {
                "unit": "subject within fixed group",
                "resamples": self.bootstrap_resamples,
                "seed": self.bootstrap_seed,
                "confidence_level": self.confidence_level,
                "scope": "uncertainty in fixed group means, not generalization to new groups",
                "trajectory_estimation": (
                    "input trajectories treated as fixed; trial-level fitting uncertainty "
                    "is not propagated"
                ),
            },
        }


def audit_trajectory_replication(
    panel: TrajectoryPanel, *, minimum_subjects_per_group: int = 2
) -> TrajectoryReplicationAudit:
    """Audit whether each group has enough independent subjects for uncertainty."""

    _require_positive_integer(minimum_subjects_per_group, "minimum_subjects_per_group")
    sizes = dict(panel.group_sizes)
    singletons = tuple(group for group, size in sizes.items() if size == 1)
    under_replicated = tuple(
        group for group, size in sizes.items() if size < minimum_subjects_per_group
    )
    return TrajectoryReplicationAudit(
        group_name=panel.group_name,
        group_sizes=sizes,
        minimum_subjects_per_group=minimum_subjects_per_group,
        singleton_groups=singletons,
        under_replicated_groups=under_replicated,
        inferentially_ready=not under_replicated,
    )


def compare_trajectory_shapes(
    panel: TrajectoryPanel,
    *,
    bootstrap_resamples: int = 2_000,
    bootstrap_seed: int = 0,
    confidence_level: float = 0.95,
    minimum_subjects_per_group: int = 2,
    flat_tolerance: float = 1e-12,
) -> TrajectoryShapeComparisonReport:
    """Compare all group mean trajectories with subject-resampling uncertainty.

    The function conditions on the named groups and resamples independent subjects within
    each group. It therefore quantifies uncertainty in those fixed group means; claims about
    a population of laboratories require a separate lab-level sampling design.
    """

    _require_positive_integer(bootstrap_resamples, "bootstrap_resamples")
    _require_nonnegative_integer(bootstrap_seed, "bootstrap_seed")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie strictly between zero and one")
    if not np.isfinite(flat_tolerance) or flat_tolerance <= 0:
        raise ValueError("flat_tolerance must be finite and positive")
    audit = audit_trajectory_replication(
        panel, minimum_subjects_per_group=minimum_subjects_per_group
    )
    if not audit.inferentially_ready:
        groups = ", ".join(repr(group) for group in audit.under_replicated_groups)
        raise ValueError(
            f"{panel.group_name} trajectory comparison requires at least "
            f"{minimum_subjects_per_group} subjects per group; under-replicated: {groups}"
        )

    weights = _trapezoid_weights(panel.grid)
    arrays: dict[Any, NDArray[np.float64]] = {}
    summaries: list[GroupTrajectorySummary] = []
    for group in panel.group_order:
        indices = np.asarray([value == group for value in panel.groups], dtype=np.bool_)
        group_values = panel.values[indices]
        arrays[group] = group_values
        mean = np.mean(group_values, axis=0)
        level, amplitude, _unit = _curve_components(mean, weights, flat_tolerance)
        summaries.append(
            GroupTrajectorySummary(
                group=group,
                subjects=tuple(
                    subject
                    for subject, subject_group in zip(panel.subjects, panel.groups, strict=True)
                    if subject_group == group
                ),
                mean_values=mean,
                level=level,
                amplitude=amplitude,
            )
        )

    generator = np.random.default_rng(bootstrap_seed)
    comparisons = [
        _compare_pair(
            left,
            right,
            arrays[left],
            arrays[right],
            weights,
            generator,
            bootstrap_resamples,
            confidence_level,
            flat_tolerance,
        )
        for left, right in combinations(panel.group_order, 2)
    ]
    return TrajectoryShapeComparisonReport(
        clock_name=panel.clock_name,
        parameter_name=panel.parameter_name,
        group_name=panel.group_name,
        grid=panel.grid,
        replication_audit=audit,
        group_summaries=tuple(summaries),
        pairwise_comparisons=tuple(comparisons),
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level,
    )


def _compare_pair(
    left_group: Any,
    right_group: Any,
    left_values: NDArray[np.float64],
    right_values: NDArray[np.float64],
    weights: NDArray[np.float64],
    generator: np.random.Generator,
    bootstrap_resamples: int,
    confidence_level: float,
    flat_tolerance: float,
) -> PairwiseTrajectoryShapeComparison:
    observed = _curve_contrasts(
        np.mean(left_values, axis=0),
        np.mean(right_values, axis=0),
        weights,
        flat_tolerance,
    )
    draws: dict[str, list[float]] = {name: [] for name in observed if name != "shape_distance"}
    shape_draws: list[float] = []
    for _ in range(bootstrap_resamples):
        left_mean = np.mean(
            left_values[generator.integers(0, len(left_values), size=len(left_values))], axis=0
        )
        right_mean = np.mean(
            right_values[generator.integers(0, len(right_values), size=len(right_values))], axis=0
        )
        contrast = _curve_contrasts(left_mean, right_mean, weights, flat_tolerance)
        for name in draws:
            draws[name].append(contrast[name])
        if contrast["shape_distance"] is not None:
            shape_draws.append(contrast["shape_distance"])

    shape_interval = None
    observed_shape = observed["shape_distance"]
    if observed_shape is not None and shape_draws:
        shape_interval = _interval(
            observed_shape, np.asarray(shape_draws), confidence_level=confidence_level
        )
    return PairwiseTrajectoryShapeComparison(
        left_group=left_group,
        right_group=right_group,
        raw_distance=_interval(
            observed["raw_distance"],
            np.asarray(draws["raw_distance"]),
            confidence_level=confidence_level,
        ),
        level_difference=_interval(
            observed["level_difference"],
            np.asarray(draws["level_difference"]),
            confidence_level=confidence_level,
        ),
        centered_distance=_interval(
            observed["centered_distance"],
            np.asarray(draws["centered_distance"]),
            confidence_level=confidence_level,
        ),
        amplitude_difference=_interval(
            observed["amplitude_difference"],
            np.asarray(draws["amplitude_difference"]),
            confidence_level=confidence_level,
        ),
        shape_distance=shape_interval,
        effective_shape_resamples=len(shape_draws),
    )


def _curve_contrasts(
    left: NDArray[np.float64],
    right: NDArray[np.float64],
    weights: NDArray[np.float64],
    flat_tolerance: float,
) -> dict[str, float | None]:
    left_level, left_amplitude, left_shape = _curve_components(left, weights, flat_tolerance)
    right_level, right_amplitude, right_shape = _curve_components(right, weights, flat_tolerance)
    left_centered = left - left_level
    right_centered = right - right_level
    shape_distance = None
    if left_shape is not None and right_shape is not None:
        shape_distance = _weighted_rms(left_shape - right_shape, weights)
    return {
        "raw_distance": _weighted_rms(left - right, weights),
        "level_difference": left_level - right_level,
        "centered_distance": _weighted_rms(left_centered - right_centered, weights),
        "amplitude_difference": left_amplitude - right_amplitude,
        "shape_distance": shape_distance,
    }


def _curve_components(
    values: NDArray[np.float64], weights: NDArray[np.float64], flat_tolerance: float
) -> tuple[float, float, NDArray[np.float64] | None]:
    level = float(np.dot(weights, values))
    centered = values - level
    amplitude = _weighted_rms(centered, weights)
    shape = None if amplitude <= flat_tolerance else centered / amplitude
    return level, amplitude, shape


def _weighted_rms(values: NDArray[np.float64], weights: NDArray[np.float64]) -> float:
    return float(np.sqrt(np.dot(weights, np.square(values))))


def _trapezoid_weights(grid: NDArray[np.float64]) -> NDArray[np.float64]:
    intervals = np.diff(grid)
    weights = np.empty_like(grid)
    weights[0] = intervals[0] / 2
    weights[-1] = intervals[-1] / 2
    weights[1:-1] = (intervals[:-1] + intervals[1:]) / 2
    weights /= grid[-1] - grid[0]
    return weights


def _interval(
    estimate: float, draws: NDArray[np.float64], *, confidence_level: float
) -> BootstrapInterval:
    tail = (1 - confidence_level) / 2
    lower, upper = np.quantile(draws, (tail, 1 - tail))
    return BootstrapInterval(
        estimate=float(estimate),
        lower=float(lower),
        upper=float(upper),
        confidence_level=confidence_level,
    )


def _reverse_interval(interval: BootstrapInterval) -> BootstrapInterval:
    return BootstrapInterval(
        estimate=-interval.estimate,
        lower=-interval.upper,
        upper=-interval.lower,
        confidence_level=interval.confidence_level,
    )


def _reverse_comparison(
    comparison: PairwiseTrajectoryShapeComparison,
) -> PairwiseTrajectoryShapeComparison:
    return PairwiseTrajectoryShapeComparison(
        left_group=comparison.right_group,
        right_group=comparison.left_group,
        raw_distance=comparison.raw_distance,
        level_difference=_reverse_interval(comparison.level_difference),
        centered_distance=comparison.centered_distance,
        amplitude_difference=_reverse_interval(comparison.amplitude_difference),
        shape_distance=comparison.shape_distance,
        effective_shape_resamples=comparison.effective_shape_resamples,
    )


def _validated_identifier(value: Any, name: str) -> Any:
    identifier = value.item() if isinstance(value, np.generic) else value
    if identifier is None:
        raise ValueError(f"{name} must not be missing")
    if isinstance(identifier, (float, complex)) and np.isnan(identifier):
        raise ValueError(f"{name} must not be missing")
    try:
        hash(identifier)
    except TypeError:
        raise TypeError(f"{name} must be hashable") from None
    return identifier


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _require_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _require_nonnegative_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
