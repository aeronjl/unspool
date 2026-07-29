"""The two-boundary Wiener likelihood and the simulator's absorption geometry.

Three families -- ``ddm``, ``smooth_ddm`` and ``hierarchical_smooth_ddm`` -- score the same
first-passage density. Before this module existed they shared it by importing four to six
underscore-prefixed names out of ``behavio.models.ddm``, which made the plain DDM the
accidental owner of a likelihood that is not specific to it.

Nothing here knows what a :class:`~behavio.study.Study` is. Every function takes broadcast
arrays in the canonical parameterization -- drift rate, boundary separation, relative
starting bias in ``(0, 1)``, decision time in seconds -- and returns arrays.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp

LOG_DENSITY_FLOOR = float(np.log(np.finfo(np.float64).tiny))
"""Smallest representable log density, used as the floor for impossible observations."""

BRIDGE_EXPONENT_LIMIT = 30.0
"""Above this exponent a Brownian-bridge absorption probability underflows to zero."""


def wiener_log_density(
    decision_time: NDArray[np.float64],
    choice: NDArray[np.float64],
    drift: NDArray[np.float64],
    *,
    boundary: float | NDArray[np.float64],
    starting_bias: float | NDArray[np.float64],
    terms: int,
) -> NDArray[np.float64]:
    """Joint two-boundary Wiener log density using paired convergent series."""

    times, choices, drifts, boundaries, biases = np.broadcast_arrays(
        np.asarray(decision_time, dtype=np.float64),
        np.asarray(choice, dtype=np.float64),
        np.asarray(drift, dtype=np.float64),
        np.asarray(boundary, dtype=np.float64),
        np.asarray(starting_bias, dtype=np.float64),
    )
    result = np.full(times.shape, LOG_DENSITY_FLOOR, dtype=np.float64)
    valid = (
        np.isfinite(times)
        & (times > 0)
        & np.isfinite(drifts)
        & np.isfinite(boundaries)
        & (boundaries > 0)
        & np.isfinite(biases)
        & (biases > 0)
        & (biases < 1)
        & ((choices == 0) | (choices == 1))
    )
    if not np.any(valid):
        return result
    selected_times = times[valid]
    selected_choices = choices[valid]
    selected_drifts = drifts[valid]
    selected_boundaries = boundaries[valid]
    selected_biases = biases[valid]
    effective_drift = np.where(selected_choices == 1, -selected_drifts, selected_drifts)
    effective_bias = np.where(selected_choices == 1, 1.0 - selected_biases, selected_biases)
    scaled_time = selected_times / selected_boundaries**2
    standard = _standard_wiener_log_density(
        scaled_time,
        effective_bias,
        terms=terms,
    )
    log_density = -2.0 * np.log(selected_boundaries)
    log_density += -effective_drift * selected_boundaries * effective_bias
    log_density += -0.5 * effective_drift**2 * selected_times
    log_density += standard
    result[valid] = np.maximum(log_density, LOG_DENSITY_FLOOR)
    return result


def _standard_wiener_log_density(
    scaled_time: NDArray[np.float64],
    starting_bias: float | NDArray[np.float64],
    *,
    terms: int,
) -> NDArray[np.float64]:
    times, biases = np.broadcast_arrays(
        np.asarray(scaled_time, dtype=np.float64),
        np.asarray(starting_bias, dtype=np.float64),
    )
    result = np.empty_like(times)
    small = times < 0.15
    if np.any(small):
        lower = -int(np.ceil((terms - 1) / 2))
        upper = int(np.floor((terms - 1) / 2))
        k = np.arange(lower, upper + 1, dtype=np.float64)
        coefficients = biases[small, None] + 2.0 * k[None, :]
        exponent = -(coefficients**2) / (2.0 * times[small, None])
        log_sum, sign = logsumexp(
            exponent,
            b=coefficients,
            axis=1,
            return_sign=True,
        )
        values = log_sum - 0.5 * np.log(2.0 * np.pi) - 1.5 * np.log(times[small])
        result[small] = np.where(sign > 0, values, LOG_DENSITY_FLOOR)
    if np.any(~small):
        k = np.arange(1, terms + 1, dtype=np.float64)
        coefficients = k[None, :] * np.sin(k[None, :] * np.pi * biases[~small, None])
        exponent = -0.5 * (k[None, :] * np.pi) ** 2 * times[~small, None]
        log_sum, sign = logsumexp(
            exponent,
            b=coefficients,
            axis=1,
            return_sign=True,
        )
        values = np.log(np.pi) + log_sum
        result[~small] = np.where(sign > 0, values, LOG_DENSITY_FLOOR)
    return result


def bridge_crossings(
    previous: NDArray[np.float64],
    current: NDArray[np.float64],
    boundary: float | NDArray[np.float64],
    *,
    time_step: float,
    generator: np.random.Generator,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Return Brownian-bridge absorptions hidden between two interior endpoints.

    A discretised path that starts and ends inside the corridor may still have crossed a
    boundary within the step. The bridge absorption probabilities are exact for a single
    boundary and their overlap is negligible whenever the corridor spans many steps.
    """

    upper_exponent = 2.0 * (boundary - previous) * (boundary - current) / time_step
    lower_exponent = 2.0 * previous * current / time_step
    upper = np.zeros(len(previous), dtype=np.bool_)
    lower = np.zeros(len(previous), dtype=np.bool_)
    candidate = np.flatnonzero(
        (upper_exponent < BRIDGE_EXPONENT_LIMIT) | (lower_exponent < BRIDGE_EXPONENT_LIMIT)
    )
    if not len(candidate):
        return upper, lower
    upper_probability = np.exp(-upper_exponent[candidate])
    lower_probability = np.exp(-lower_exponent[candidate])
    draws = generator.uniform(0.0, 1.0, len(candidate))
    upper[candidate] = draws < upper_probability
    lower[candidate] = (draws >= upper_probability) & (
        draws < upper_probability + lower_probability
    )
    return upper, lower


def crossing_fractions(
    previous: NDArray[np.float64],
    current: NDArray[np.float64],
    target: NDArray[np.float64],
    *,
    endpoint_crossed: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Return within-step crossing positions, midpoints for bridge absorptions."""

    span = current - previous
    interpolated = (target - previous) / np.where(span == 0.0, 1.0, span)
    fraction = np.where(endpoint_crossed & (span != 0.0), interpolated, 0.5)
    return np.clip(fraction, 0.0, 1.0)


def upper_boundary_probability(
    drift: NDArray[np.float64],
    *,
    boundary: float | NDArray[np.float64],
    starting_bias: float | NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return the probability of absorbing at the upper boundary."""

    drifts, boundaries, biases = np.broadcast_arrays(
        np.asarray(drift, dtype=np.float64),
        np.asarray(boundary, dtype=np.float64),
        np.asarray(starting_bias, dtype=np.float64),
    )
    scaled = 2.0 * drifts * boundaries
    probability = np.empty_like(scaled)
    near_zero = np.abs(scaled) < 1e-8
    probability[near_zero] = biases[near_zero]
    positive = (scaled > 0) & ~near_zero
    probability[positive] = np.expm1(-scaled[positive] * biases[positive]) / np.expm1(
        -scaled[positive]
    )
    negative = (scaled < 0) & ~near_zero
    negative_scaled = scaled[negative]
    probability[negative] = (
        np.exp(negative_scaled * (1.0 - biases[negative])) - np.exp(negative_scaled)
    ) / (1.0 - np.exp(negative_scaled))
    return probability


def simulate_trialwise_wiener(
    drift: NDArray[np.float64],
    boundary: NDArray[np.float64],
    starting_bias: NDArray[np.float64],
    nondecision_time: NDArray[np.float64],
    *,
    generator: np.random.Generator,
    time_step: float,
    maximum_time: float,
) -> tuple[NDArray[np.int8], NDArray[np.float64]]:
    """Simulate first-passage choices and response times by Euler-Maruyama steps.

    Discretisation can step over a boundary and back within one interval, so interior
    endpoints are additionally tested against the exact Brownian-bridge absorption
    probability. Crossing times are interpolated within the step that absorbed them.
    """

    n_rows = len(drift)
    positions = boundary * starting_bias
    decision_times = np.zeros(n_rows, dtype=np.float64)
    choices = np.zeros(n_rows, dtype=np.int8)
    active = np.ones(n_rows, dtype=np.bool_)
    noise_scale = np.sqrt(time_step)
    max_steps = int(np.ceil(maximum_time / time_step))
    for step in range(max_steps):
        active_indices = np.flatnonzero(active)
        if not len(active_indices):
            break
        previous = positions[active_indices]
        current = previous + drift[active_indices] * time_step
        current += generator.normal(0.0, noise_scale, len(active_indices))
        positions[active_indices] = current
        active_boundary = boundary[active_indices]
        upper = current >= active_boundary
        lower = current <= 0.0
        endpoint_crossed = upper | lower
        interior = np.flatnonzero(~endpoint_crossed)
        if len(interior):
            bridge_upper, bridge_lower = bridge_crossings(
                previous[interior],
                current[interior],
                active_boundary[interior],
                time_step=time_step,
                generator=generator,
            )
            upper[interior] = bridge_upper
            lower[interior] = bridge_lower
        crossed = upper | lower
        if not np.any(crossed):
            continue
        crossed_indices = active_indices[crossed]
        upper_crossed = upper[crossed]
        fraction = crossing_fractions(
            previous[crossed],
            current[crossed],
            np.where(upper_crossed, boundary[crossed_indices], 0.0),
            endpoint_crossed=endpoint_crossed[crossed],
        )
        decision_times[crossed_indices] = (step + fraction) * time_step
        choices[crossed_indices] = upper_crossed.astype(np.int8)
        active[crossed_indices] = False
    if np.any(active):
        raise RuntimeError(
            f"{int(np.sum(active))} simulated paths did not terminate within "
            f"{maximum_time:g} seconds"
        )
    return choices, decision_times + nondecision_time
