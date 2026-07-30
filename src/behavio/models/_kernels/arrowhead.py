"""Covariance for a joint fit whose Hessian is an arrowhead, without forming it densely.

A hierarchical maximum-a-posteriori fit estimates one population block and one deviation
block per group, and no two groups share a row. Their cross-derivatives are therefore
exactly zero, and the joint Hessian is an *arrowhead*: a dense population block, a diagonal
of per-group blocks, and one cross block per group. A model with an analytic Hessian never
notices the difference, but a model that differences its objective numerically does: the
dense Hessian of a twenty-subject joint fit costs quadratically many objective evaluations
of which almost all return a structural zero.

Nothing here is drift-diffusion specific. It is here rather than in ``compose`` because it
belongs to a *solver* -- a combinator hands a model a wider problem and does not say how to
invert its curvature -- and because only a family without analytic derivatives needs it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from behavio.models._kernels.curvature import bounded_value_difference_hessian

__all__ = [
    "ArrowheadCovariance",
    "arrowhead_covariance",
    "conditional_block_covariances",
    "numerical_cross_hessian",
]

Objective = Callable[[NDArray[np.float64]], float]


class ArrowheadCovariance:
    """The inverse of an arrowhead Hessian, assembled from its blocks.

    The population block and each group's own block come from the Schur complement, which
    is what makes the inversion cheap; the cross blocks follow from the same factorisation
    and are filled in so that the whole thing is an ordinary covariance matrix a caller can
    slice. That matters because a combinator reads ``covariance[:population, :population]``
    and ``standard_errors[population:]`` off a flat :class:`~behavio.contracts.estimator.\
FitResult` and must not have to know how it was computed.
    """

    __slots__ = ("condition", "covariance", "group_covariances", "population_covariance")

    def __init__(
        self,
        covariance: NDArray[np.float64],
        population_covariance: NDArray[np.float64],
        group_covariances: tuple[NDArray[np.float64], ...],
        condition: float,
    ) -> None:
        self.covariance = covariance
        self.population_covariance = population_covariance
        self.group_covariances = group_covariances
        self.condition = condition


def arrowhead_covariance(
    objective: Objective,
    point: NDArray[np.float64],
    *,
    population_bounds: Sequence[tuple[float, float]],
    group_bounds: Sequence[tuple[float, float]],
    n_groups: int,
) -> ArrowheadCovariance:
    """Invert a numerical arrowhead Hessian without evaluating zero cross-group blocks."""

    n_population = len(population_bounds)
    n_deviation = len(group_bounds)
    joint_bounds = [*population_bounds, *(list(group_bounds) * n_groups)]

    def with_population(values: NDArray[np.float64]) -> float:
        candidate = np.array(point, copy=True)
        candidate[:n_population] = values
        return float(objective(candidate))

    population_hessian = bounded_value_difference_hessian(
        with_population, point[:n_population], population_bounds
    )
    group_hessians: list[NDArray[np.float64]] = []
    cross_hessians: list[NDArray[np.float64]] = []
    group_inverses: list[NDArray[np.float64]] = []
    for group in range(n_groups):
        start = n_population + group * n_deviation
        stop = start + n_deviation

        def with_group(
            values: NDArray[np.float64],
            block_start: int = start,
            block_stop: int = stop,
        ) -> float:
            candidate = np.array(point, copy=True)
            candidate[block_start:block_stop] = values
            return float(objective(candidate))

        group_hessian = bounded_value_difference_hessian(
            with_group, point[start:stop], group_bounds
        )
        cross_hessian = numerical_cross_hessian(
            objective,
            point,
            left_indices=np.arange(n_population),
            right_indices=np.arange(start, stop),
            bounds=joint_bounds,
        )
        group_hessians.append(group_hessian)
        cross_hessians.append(cross_hessian)
        group_inverses.append(np.linalg.pinv(group_hessian, hermitian=True))

    schur = np.array(population_hessian, copy=True)
    for cross, inverse in zip(cross_hessians, group_inverses, strict=True):
        schur -= cross @ inverse @ cross.T
    population_covariance = np.linalg.pinv(schur, hermitian=True)
    group_covariances = tuple(
        inverse + inverse @ cross.T @ population_covariance @ cross @ inverse
        for cross, inverse in zip(cross_hessians, group_inverses, strict=True)
    )
    size = len(point)
    covariance = np.zeros((size, size), dtype=np.float64)
    covariance[:n_population, :n_population] = population_covariance
    weighted = [
        inverse @ cross.T for cross, inverse in zip(cross_hessians, group_inverses, strict=True)
    ]
    for group in range(n_groups):
        start = n_population + group * n_deviation
        stop = start + n_deviation
        covariance[start:stop, start:stop] = group_covariances[group]
        cross_block = -weighted[group] @ population_covariance
        covariance[start:stop, :n_population] = cross_block
        covariance[:n_population, start:stop] = cross_block.T
        for other in range(n_groups):
            if other == group:
                continue
            other_start = n_population + other * n_deviation
            covariance[start:stop, other_start : other_start + n_deviation] = (
                weighted[group] @ population_covariance @ weighted[other].T
            )
    full_hessian = np.zeros((size, size), dtype=np.float64)
    full_hessian[:n_population, :n_population] = population_hessian
    for group, (group_hessian, cross) in enumerate(
        zip(group_hessians, cross_hessians, strict=True)
    ):
        start = n_population + group * n_deviation
        stop = start + n_deviation
        full_hessian[start:stop, start:stop] = group_hessian
        full_hessian[:n_population, start:stop] = cross
        full_hessian[start:stop, :n_population] = cross.T
    return ArrowheadCovariance(
        covariance=covariance,
        population_covariance=population_covariance,
        group_covariances=group_covariances,
        condition=float(np.linalg.cond(full_hessian)),
    )


def conditional_block_covariances(
    objective: Objective,
    point: NDArray[np.float64],
    *,
    n_population: int,
    group_bounds: Sequence[tuple[float, float]],
    n_groups: int,
) -> NDArray[np.float64]:
    """Approximate each group block's covariance *conditional* on the population block.

    This is not the same object as the marginal block of an arrowhead inverse, and the
    difference is the whole point: an EM step over a variance component needs the
    conditional second moment of the deviations given everything else, which is the
    conditional mode outer itself plus this.
    """

    n_deviation = len(group_bounds)
    covariances = np.empty((n_groups, n_deviation, n_deviation), dtype=np.float64)
    for group in range(n_groups):
        start = n_population + group * n_deviation
        stop = start + n_deviation

        def with_group(
            values: NDArray[np.float64],
            block_start: int = start,
            block_stop: int = stop,
        ) -> float:
            candidate = np.array(point, copy=True)
            candidate[block_start:block_stop] = values
            return float(objective(candidate))

        hessian = bounded_value_difference_hessian(with_group, point[start:stop], group_bounds)
        covariance = np.linalg.pinv(hessian, hermitian=True)
        covariances[group] = 0.5 * (covariance + covariance.T)
    return covariances


def numerical_cross_hessian(
    objective: Objective,
    point: NDArray[np.float64],
    *,
    left_indices: NDArray[np.int64],
    right_indices: NDArray[np.int64],
    bounds: Sequence[tuple[float, float]],
) -> NDArray[np.float64]:
    """Return the mixed second differences between two disjoint blocks of coordinates."""

    evaluation_point = np.array(point, copy=True)
    steps = np.maximum(1e-5, 1e-4 * np.maximum(1.0, np.abs(point)))
    for index, (lower, upper) in enumerate(bounds):
        base_step = min(steps[index], (upper - lower) / 4.0)
        evaluation_point[index] = np.clip(
            evaluation_point[index], lower + base_step, upper - base_step
        )
        steps[index] = min(
            base_step,
            (evaluation_point[index] - lower) / 2.0,
            (upper - evaluation_point[index]) / 2.0,
        )
    result = np.empty((len(left_indices), len(right_indices)), dtype=np.float64)
    for left_position, left in enumerate(left_indices):
        for right_position, right in enumerate(right_indices):
            left_step = steps[left]
            right_step = steps[right]
            plus_plus = evaluation_point.copy()
            plus_minus = evaluation_point.copy()
            minus_plus = evaluation_point.copy()
            minus_minus = evaluation_point.copy()
            plus_plus[[left, right]] += [left_step, right_step]
            plus_minus[[left, right]] += [left_step, -right_step]
            minus_plus[[left, right]] += [-left_step, right_step]
            minus_minus[[left, right]] -= [left_step, right_step]
            result[left_position, right_position] = (
                float(objective(plus_plus))
                - float(objective(plus_minus))
                - float(objective(minus_plus))
                + float(objective(minus_minus))
            ) / (4.0 * left_step * right_step)
    return result
