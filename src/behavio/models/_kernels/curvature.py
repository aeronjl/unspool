"""Deterministic finite-difference curvature shared by the optimizer-backed families.

Behavio reports a covariance for every maximum-likelihood fit, and for most families that
covariance comes from a numerical Hessian rather than an analytic one. Six families had
written that Hessian themselves. The copies were not gratuitous -- they differ in what they
are handed (a scalar objective, a gradient, or an objective returning both) and in the
step they take -- but the differences were undocumented, so nobody could tell which
variations were deliberate.

They are separated here into the two things that actually vary:

* **The step rule.** :func:`relative_steps` scales by ``max(1, |x|)`` and
  :func:`offset_steps` by ``1 + |x|``. Both are in use and they are *not*
  interchangeable: they agree only where ``|x| = 0`` or ``|x| = 1``, so swapping one for
  the other moves every fitted standard error in the last digits. Each call site keeps the
  rule it was pinned with.
* **The difference scheme.** :func:`finite_difference_hessian` differentiates a gradient
  once; :func:`value_difference_hessian` differences the objective twice, which is what a
  family without an analytic gradient must do.

A step *policy* is therefore an argument, not a fork in the code, and there is one
implementation of each scheme.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

RELATIVE_STEP = 1e-4
"""Default finite-difference scale: small enough for accuracy, large enough for noise."""


def relative_steps(
    values: NDArray[np.float64], *, scale: float = RELATIVE_STEP
) -> NDArray[np.float64]:
    """Return ``scale * max(1, |x|)`` per coordinate."""

    return scale * np.maximum(1.0, np.abs(np.asarray(values, dtype=np.float64)))


def offset_steps(
    values: NDArray[np.float64], *, scale: float = RELATIVE_STEP
) -> NDArray[np.float64]:
    """Return ``scale * (1 + |x|)`` per coordinate."""

    return scale * (1.0 + np.abs(np.asarray(values, dtype=np.float64)))


def bounded_steps(
    point: NDArray[np.float64], bounds: Sequence[tuple[float, float]]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return an interior evaluation point and steps that stay inside ``bounds``.

    A boundary estimate cannot be differenced symmetrically where it sits, so the point is
    first pulled inside by one step and the step is then shrunk to whichever side is
    tighter. Both are reported, because the returned curvature describes the moved point,
    not the optimizer's answer.
    """

    evaluation_point = np.array(point, copy=True, dtype=np.float64)
    base_steps = np.maximum(1e-5, RELATIVE_STEP * np.maximum(1.0, np.abs(point)))
    steps = np.array(base_steps, copy=True)
    for index, (lower, upper) in enumerate(bounds):
        base_steps[index] = min(base_steps[index], (upper - lower) / 4.0)
        evaluation_point[index] = np.clip(
            evaluation_point[index],
            lower + base_steps[index],
            upper - base_steps[index],
        )
        steps[index] = min(
            base_steps[index],
            (evaluation_point[index] - lower) / 2.0,
            (upper - evaluation_point[index]) / 2.0,
        )
    return evaluation_point, steps


def finite_difference_gradient(
    objective: Callable[[NDArray[np.float64]], float],
    values: NDArray[np.float64],
    *,
    steps: NDArray[np.float64] | None = None,
    relative_step: float = RELATIVE_STEP,
) -> NDArray[np.float64]:
    """Return the central-difference gradient of a scalar objective."""

    step_sizes = relative_steps(values, scale=relative_step) if steps is None else steps
    gradient = np.empty(len(values), dtype=np.float64)
    for index in range(len(values)):
        step = float(step_sizes[index])
        left = np.array(values, dtype=np.float64)
        right = np.array(values, dtype=np.float64)
        left[index] -= step
        right[index] += step
        gradient[index] = (objective(right) - objective(left)) / (2.0 * step)
    return gradient


def finite_difference_hessian(
    gradient: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    values: NDArray[np.float64],
    *,
    steps: NDArray[np.float64] | None = None,
    relative_step: float = RELATIVE_STEP,
) -> NDArray[np.float64]:
    """Return a symmetrized central-difference Hessian built from a gradient callable."""

    step_sizes = relative_steps(values, scale=relative_step) if steps is None else steps
    size = len(values)
    hessian = np.empty((size, size), dtype=np.float64)
    for column in range(size):
        step = float(step_sizes[column])
        left = np.array(values, dtype=np.float64)
        right = np.array(values, dtype=np.float64)
        left[column] -= step
        right[column] += step
        hessian[:, column] = (gradient(right) - gradient(left)) / (2.0 * step)
    return 0.5 * (hessian + hessian.T)


def value_difference_hessian(
    objective: Callable[[NDArray[np.float64]], float],
    point: NDArray[np.float64],
    *,
    steps: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return the second-difference Hessian of a scalar objective with no gradient.

    The mixed partials are evaluated as ``(f(++) - f(+-) - f(-+) + f(--))`` with the outer
    coordinate leading. That ordering is load-bearing: the four evaluations very nearly
    cancel, so re-associating the sum perturbs the result in its last bits and moves every
    standard error that this curvature feeds.
    """

    n_parameters = len(point)
    hessian = np.empty((n_parameters, n_parameters), dtype=np.float64)
    evaluation_point = np.asarray(point, dtype=np.float64)
    center = float(objective(evaluation_point))
    for row in range(n_parameters):
        row_step = steps[row]
        plus = evaluation_point.copy()
        minus = evaluation_point.copy()
        plus[row] += row_step
        minus[row] -= row_step
        hessian[row, row] = (
            float(objective(plus)) - 2.0 * center + float(objective(minus))
        ) / row_step**2
        for column in range(row):
            column_step = steps[column]
            plus_plus = evaluation_point.copy()
            plus_minus = evaluation_point.copy()
            minus_plus = evaluation_point.copy()
            minus_minus = evaluation_point.copy()
            plus_plus[[row, column]] += [row_step, column_step]
            plus_minus[[row, column]] += [row_step, -column_step]
            minus_plus[[row, column]] += [-row_step, column_step]
            minus_minus[[row, column]] -= [row_step, column_step]
            value = (
                float(objective(plus_plus))
                - float(objective(plus_minus))
                - float(objective(minus_plus))
                + float(objective(minus_minus))
            ) / (4.0 * row_step * column_step)
            hessian[row, column] = hessian[column, row] = value
    return hessian


def bounded_value_difference_hessian(
    objective: Callable[[NDArray[np.float64]], float],
    point: NDArray[np.float64],
    bounds: Sequence[tuple[float, float]],
) -> NDArray[np.float64]:
    """Return :func:`value_difference_hessian` at an interior point inside ``bounds``."""

    evaluation_point, steps = bounded_steps(point, bounds)
    return value_difference_hessian(objective, evaluation_point, steps=steps)


def covariance_from_hessian(hessian: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the pseudo-inverse covariance of a symmetric observed-information matrix."""

    return np.linalg.pinv(hessian, hermitian=True)
