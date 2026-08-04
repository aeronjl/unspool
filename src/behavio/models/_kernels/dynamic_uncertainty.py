"""Uncertainty kernels for session-dynamic latent-variable paths.

The path covariance helpers in this module deliberately receive a gradient of the
*observed* (state-marginalized) objective.  Passing the fixed-responsibility gradient from
an EM M-step would instead return complete-data curvature and understate uncertainty.

Gaussian variance components are updated from the conditional second moment of linear
path contrasts.  Their reported covariance applies Louis' missing-information identity in
log-scale coordinates.  The transition concentration is a different object: session rows
have a Dirichlet distribution, so its one-dimensional evidence is Dirichlet-multinomial.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar
from scipy.special import gammaln

from behavio.models._kernels.curvature import finite_difference_hessian, offset_steps


@dataclass(frozen=True, slots=True)
class LocalPathCovariance:
    """A local covariance and the numerical evidence supporting it."""

    covariance: NDArray[np.float64]
    standard_errors: NDArray[np.float64]
    hessian_condition: float
    positive_definite: bool


def observed_path_covariance(
    gradient: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    mode: NDArray[np.float64],
) -> LocalPathCovariance:
    """Differentiate an observed-objective gradient at one canonical path mode."""

    point = np.asarray(mode, dtype=np.float64)
    hessian = finite_difference_hessian(
        gradient,
        point,
        steps=offset_steps(point, scale=1e-5),
    )
    hessian = 0.5 * (hessian + hessian.T)
    eigenvalues = np.linalg.eigvalsh(hessian)
    positive = bool(np.all(np.isfinite(eigenvalues)) and np.min(eigenvalues) > 0.0)
    condition = float(np.linalg.cond(hessian))
    if positive and np.isfinite(condition):
        covariance = np.linalg.pinv(hessian, hermitian=True)
        covariance = 0.5 * (covariance + covariance.T)
        errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    else:
        covariance = np.full((len(point), len(point)), np.nan, dtype=np.float64)
        errors = np.full(len(point), np.nan, dtype=np.float64)
    return LocalPathCovariance(
        covariance=covariance,
        standard_errors=errors,
        hessian_condition=condition,
        positive_definite=positive,
    )


def first_axis_difference_contrast(shape: tuple[int, ...]) -> NDArray[np.float64]:
    """Return ``L`` such that ``L @ values.ravel()`` contains adjacent differences."""

    if len(shape) < 1 or shape[0] < 2 or any(size < 1 for size in shape):
        raise ValueError("a difference contrast needs at least two path positions")
    trailing = int(np.prod(shape[1:], dtype=np.int64))
    rows = (shape[0] - 1) * trailing
    matrix = np.zeros((rows, int(np.prod(shape, dtype=np.int64))), dtype=np.float64)
    row = 0
    for position in range(1, shape[0]):
        previous = (position - 1) * trailing
        current = position * trailing
        for offset in range(trailing):
            matrix[row, previous + offset] = -1.0
            matrix[row, current + offset] = 1.0
            row += 1
    return matrix


def contrast_second_moment(
    mode: NDArray[np.float64],
    covariance: NDArray[np.float64],
    contrast: NDArray[np.float64],
) -> float:
    """Return ``E[||L x||²]`` under a local Gaussian approximation."""

    point = np.asarray(mode, dtype=np.float64)
    values = np.asarray(contrast, dtype=np.float64) @ point
    uncertainty = np.asarray(contrast, dtype=np.float64) @ covariance @ contrast.T
    return float(values @ values + np.trace(uncertainty))


def update_gaussian_scales(
    mode: NDArray[np.float64],
    covariance: NDArray[np.float64],
    contrasts: Sequence[NDArray[np.float64]],
    bounds: tuple[float, float],
) -> NDArray[np.float64]:
    """Run the normalized-Gaussian Laplace-EM M-step for path scales."""

    lower, upper = bounds
    updated = []
    for contrast in contrasts:
        matrix = np.asarray(contrast, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(mode) or not len(matrix):
            raise ValueError("every scale contrast must contain at least one path contrast")
        second = contrast_second_moment(mode, covariance, matrix)
        updated.append(float(np.clip(np.sqrt(second / len(matrix)), lower, upper)))
    return np.asarray(updated, dtype=np.float64)


def gaussian_scale_observed_covariance(
    scales: NDArray[np.float64],
    mode: NDArray[np.float64],
    covariance: NDArray[np.float64],
    contrasts: Sequence[NDArray[np.float64]],
) -> tuple[NDArray[np.float64], NDArray[np.float64], bool]:
    """Return Louis-corrected covariance for Gaussian scales.

    The complete log-scale information is ``2 q`` for ``q`` independent Gaussian
    contrasts.  The covariance of their conditional quadratic scores is the missing
    information.  Cross-scale blocks are retained rather than silently reporting
    independent scale intervals.
    """

    values = np.asarray(scales, dtype=np.float64)
    matrices = tuple(np.asarray(item, dtype=np.float64) for item in contrasts)
    means = tuple(matrix @ np.asarray(mode, dtype=np.float64) for matrix in matrices)
    projected = tuple(tuple(left @ covariance @ right.T for right in matrices) for left in matrices)
    information = np.diag([2.0 * len(matrix) for matrix in matrices])
    precisions = values**-2.0
    for row in range(len(matrices)):
        for column in range(row + 1):
            cross = projected[row][column]
            missing = (
                precisions[row]
                * precisions[column]
                * (2.0 * float(np.sum(cross**2)) + 4.0 * float(means[row] @ cross @ means[column]))
            )
            information[row, column] -= missing
            if row != column:
                information[column, row] -= missing
    information = 0.5 * (information + information.T)
    eigenvalues = np.linalg.eigvalsh(information)
    valid = bool(np.all(np.isfinite(eigenvalues)) and np.min(eigenvalues) > 0.0)
    if not valid:
        shape = (len(values), len(values))
        return np.full(shape, np.nan), np.full(len(values), np.nan), False
    log_covariance = np.linalg.pinv(information, hermitian=True)
    natural = np.diag(values) @ log_covariance @ np.diag(values)
    natural = 0.5 * (natural + natural.T)
    return natural, np.sqrt(np.maximum(np.diag(natural), 0.0)), True


def supplemented_scale_covariance(
    scales: NDArray[np.float64],
    update: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    bounds: tuple[float, float],
    contrast_sizes: Sequence[int],
    tolerance: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float, bool]:
    """Correct complete Gaussian-scale information by differencing the EM map.

    This is the supplemented-EM identity ``I_obs = (I - R) I_com``.  The returned rate
    matrix and spectral radius make instability inspectable; no eigenvalue clipping turns a
    failed correction into an interval.
    """

    values = np.asarray(scales, dtype=np.float64)
    center = np.log(values)
    log_bounds = (float(np.log(bounds[0])), float(np.log(bounds[1])))
    step = max(0.02, tolerance / 2.0)
    rate = np.empty((len(values), len(values)), dtype=np.float64)
    for index in range(len(values)):
        left = center.copy()
        right = center.copy()
        left[index] = max(left[index] - step, log_bounds[0])
        right[index] = min(right[index] + step, log_bounds[1])
        denominator = right[index] - left[index]
        if denominator <= 0:
            shape = (len(values), len(values))
            return (
                np.full(shape, np.nan),
                np.full(len(values), np.nan),
                np.full(shape, np.nan),
                float("inf"),
                False,
            )
        rate[:, index] = (np.log(update(np.exp(right))) - np.log(update(np.exp(left)))) / (
            denominator
        )
    eigenvalues = np.linalg.eigvals(rate)
    spectral_radius = float(np.max(np.abs(eigenvalues)))
    complete = np.diag(2.0 * np.asarray(tuple(contrast_sizes), dtype=np.float64))
    information = (np.eye(len(values)) - rate) @ complete
    information = 0.5 * (information + information.T)
    information_eigenvalues = np.linalg.eigvalsh(information)
    valid = bool(
        np.isfinite(spectral_radius)
        and spectral_radius < 1.0
        and np.all(np.isfinite(information_eigenvalues))
        and np.min(information_eigenvalues) > 0.0
    )
    if not valid:
        shape = (len(values), len(values))
        return (
            np.full(shape, np.nan),
            np.full(len(values), np.nan),
            rate,
            spectral_radius,
            False,
        )
    log_covariance = np.linalg.pinv(information, hermitian=True)
    covariance = np.diag(values) @ log_covariance @ np.diag(values)
    covariance = 0.5 * (covariance + covariance.T)
    return (
        covariance,
        np.sqrt(np.maximum(np.diag(covariance), 0.0)),
        rate,
        spectral_radius,
        True,
    )


def session_transition_counts(
    expectations: NDArray[np.float64],
    sessions: tuple[tuple[int, ...], ...],
) -> NDArray[np.float64]:
    """Aggregate smoothed transition counts by session."""

    return np.stack(
        [np.sum(expectations[np.asarray(indices, dtype=np.intp)], axis=0) for indices in sessions]
    )


def _dirichlet_multinomial_loss(
    log_concentration: float,
    counts: NDArray[np.float64],
    center: NDArray[np.float64],
) -> float:
    concentration = float(np.exp(log_concentration))
    prior = concentration * center + 1.0
    totals = np.sum(counts, axis=-1)
    prior_total = np.sum(prior, axis=-1)
    value = gammaln(prior_total) - gammaln(prior_total + totals)
    value += np.sum(gammaln(prior[None, :, :] + counts) - gammaln(prior[None, :, :]), axis=-1)
    return -float(np.sum(value))


def estimate_transition_concentration(
    counts: NDArray[np.float64],
    center: NDArray[np.float64],
    bounds: tuple[float, float],
) -> tuple[float, float, bool]:
    """Estimate and locally quantify the population Dirichlet concentration."""

    log_bounds = (float(np.log(bounds[0])), float(np.log(bounds[1])))
    result = minimize_scalar(
        _dirichlet_multinomial_loss,
        args=(np.asarray(counts, dtype=np.float64), np.asarray(center, dtype=np.float64)),
        bounds=log_bounds,
        method="bounded",
        options={"xatol": 1e-5},
    )
    log_value = float(result.x)
    concentration = float(np.exp(log_value))
    step = 1e-3
    curvature = (
        _dirichlet_multinomial_loss(log_value + step, counts, center)
        - 2.0 * _dirichlet_multinomial_loss(log_value, counts, center)
        + _dirichlet_multinomial_loss(log_value - step, counts, center)
    ) / step**2
    standard_error = (
        float(concentration / np.sqrt(curvature))
        if result.success and np.isfinite(curvature) and curvature > 0.0
        else float("nan")
    )
    return concentration, standard_error, bool(result.success)


def transition_standard_errors(
    counts: NDArray[np.float64],
    concentration: float,
    center: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return conditional Dirichlet posterior deviations on the probability scale."""

    parameters = np.asarray(counts, dtype=np.float64) + concentration * center[None, :, :] + 1.0
    totals = np.sum(parameters, axis=-1, keepdims=True)
    variance = parameters * (totals - parameters) / (totals**2 * (totals + 1.0))
    return np.sqrt(np.maximum(variance, 0.0))


def at_log_bound(value: float, bounds: tuple[float, float], tolerance: float) -> bool:
    """Return whether a positive estimate is within ``tolerance`` of a log bound."""

    log_value = float(np.log(value))
    return bool(
        abs(log_value - float(np.log(bounds[0]))) <= tolerance
        or abs(log_value - float(np.log(bounds[1]))) <= tolerance
    )
