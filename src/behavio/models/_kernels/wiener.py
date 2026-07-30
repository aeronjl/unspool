"""The two-boundary Wiener likelihood and the simulator's absorption geometry.

The drift-diffusion families score the same first-passage density. Before this module
existed they shared it by importing four to six underscore-prefixed names out of
``behavio.models.ddm``, which made the plain DDM the accidental owner of a likelihood that
is not specific to it.

Most of this module knows nothing about what a :class:`~behavio.trials.Study` is. Every
function takes broadcast arrays in the canonical parameterization -- drift rate, boundary
separation, relative starting bias in ``(0, 1)``, decision time in seconds -- and returns
arrays. :class:`WienerLikelihood` is the same density again, exposed as the four operations
:class:`~behavio.contracts.compose.LinearPredictorLikelihood` asks for, so that a combinator
can write an objective over it without importing anything about drift diffusion.

Accuracy of the series
----------------------
:func:`wiener_log_density` sums a fixed ``terms`` of the small-time and large-time
expansions and switches between them at a hardcoded scaled time of ``0.15``. Published
implementations instead choose the number of terms adaptively for a requested error bound
(Navarro & Fuss, 2009), which is strictly better and is a separate change: the switch point
and the term count are the only things about this function anyone would alter, and both are
arguments to a single vectorised expression, so nothing structural depends on them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp

# ``LOG_DENSITY_FLOOR`` is declared in the contract layer, where the prediction type that
# reads a tabulated density off a grid also lives; it used to be defined here and again in
# ``behavio.compose.mixture``, three copies of one number.
from behavio.contracts.estimator import LOG_DENSITY_FLOOR, Prediction, PredictionMode

BRIDGE_EXPONENT_LIMIT = 30.0
"""Above this exponent a Brownian-bridge absorption probability underflows to zero."""

INADMISSIBLE_OBJECTIVE = float(np.finfo(np.float64).max / 1e100)
"""The value a Wiener objective takes where no decision time is left to explain."""

PREDICTOR_CELLS = ("drift", "boundary", "starting_bias", "nondecision_time")
"""The four numbers a Wiener density needs from one row."""

OUTCOME_CHANNELS = ("choice", "response_time")
"""The two numbers a Wiener density scores on one row."""

PREDICTOR_STEP = 1e-6
"""Central-difference step for gradients and curvature in the linear predictor.

The Wiener density has no tractable derivative in its parameters, which is why every
drift-diffusion fit in this package has always used a derivative-free optimizer and a
numerical Hessian. Differentiating in the *predictor* rather than in the coordinate is what
makes it affordable: rows are independent, so one central difference per cell is one extra
vectorised evaluation of the density and not one per parameter.
"""


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


# --------------------------------------------------------------------------------------
# The density again, as the four operations a combinator writes objectives over
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WienerLikelihood:
    """The joint choice/response-time density seen through one cell vector per row.

    A row's linear predictor is ``(drift, boundary, starting_bias, nondecision_time)``, and
    every one of them enters the density on its natural scale: there is no link function,
    and the box that keeps a boundary positive is the estimator's, not this object's. A
    row's observation is ``(choice, response_time)`` in seconds.

    That this is expressible at all is the whole reason the drift-diffusion families
    compose. Nothing about the contract required the *density* to be linear -- only the
    predictor -- and a Wiener density is an arbitrary function of four numbers each of which
    is a linear function of the coefficient vector.
    """

    density_terms: int

    @property
    def cells(self) -> tuple[str, ...]:
        """The predictor cells this likelihood reads."""

        return PREDICTOR_CELLS

    def prediction(
        self, linear_predictor: NDArray[np.float64], *, mode: PredictionMode
    ) -> Prediction:
        """Return marginal upper-boundary choice probabilities."""

        cells = np.asarray(linear_predictor, dtype=np.float64)
        probability = upper_boundary_probability(
            cells[:, 0], boundary=cells[:, 1], starting_bias=cells[:, 2]
        )
        probability = np.clip(probability, np.finfo(float).tiny, 1.0 - np.finfo(float).eps)
        return Prediction(
            probability=probability,
            linear_predictor=np.log(probability) - np.log1p(-probability),
            mode=PredictionMode(mode),
        )

    def pointwise_log_prob(
        self, linear_predictor: NDArray[np.float64], outcomes: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Return the joint log density of each observed choice and response time."""

        cells = np.asarray(linear_predictor, dtype=np.float64)
        observed = np.asarray(outcomes, dtype=np.float64)
        return self.log_density(cells, observed)

    def log_density(
        self, cells: NDArray[np.float64], outcomes: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Return the per-row joint log density."""

        return wiener_log_density(
            outcomes[:, 1] - cells[:, 3],
            outcomes[:, 0],
            cells[:, 0],
            boundary=cells[:, 1],
            starting_bias=cells[:, 2],
            terms=self.density_terms,
        )

    def negative_log_likelihood(
        self, linear_predictor: NDArray[np.float64], outcomes: NDArray[np.float64]
    ) -> float:
        """Return the summed negative log density, or a large finite value if inadmissible.

        A non-decision time longer than a response leaves no decision time to explain, which
        is not a small likelihood but no likelihood at all. Returning a large finite number
        rather than an infinity is what keeps a bounded quasi-Newton search able to walk
        back out of the region.
        """

        cells = np.asarray(linear_predictor, dtype=np.float64)
        observed = np.asarray(outcomes, dtype=np.float64)
        if np.any(observed[:, 1] - cells[:, 3] <= 0):
            return INADMISSIBLE_OBJECTIVE
        return float(-np.sum(self.log_density(cells, observed)))

    def value_and_gradient(
        self, linear_predictor: NDArray[np.float64], outcomes: NDArray[np.float64]
    ) -> tuple[float, NDArray[np.float64]]:
        """Return the negative log likelihood and its central-difference cell gradient."""

        cells = np.asarray(linear_predictor, dtype=np.float64)
        observed = np.asarray(outcomes, dtype=np.float64)
        value = self.negative_log_likelihood(cells, observed)
        gradient = np.zeros_like(cells)
        if value == INADMISSIBLE_OBJECTIVE:
            return value, gradient
        for cell in range(cells.shape[1]):
            forward = cells.copy()
            backward = cells.copy()
            forward[:, cell] += PREDICTOR_STEP
            backward[:, cell] -= PREDICTOR_STEP
            gradient[:, cell] = (
                -self.log_density(forward, observed) + self.log_density(backward, observed)
            ) / (2.0 * PREDICTOR_STEP)
        return value, gradient

    def curvature(
        self, linear_predictor: NDArray[np.float64], outcomes: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Return each row's observed information block in the predictor cells.

        Observed and not expected: a first-passage density has no closed-form expected
        information, so the observation has to be read. Rows are independent, so the whole
        ``(rows, cells, cells)`` stack costs one vectorised density evaluation per entry
        rather than one per row per entry.
        """

        cells = np.asarray(linear_predictor, dtype=np.float64)
        observed = np.asarray(outcomes, dtype=np.float64)
        n_cells = cells.shape[1]
        step = PREDICTOR_STEP
        centre = -self.log_density(cells, observed)
        blocks = np.zeros((cells.shape[0], n_cells, n_cells), dtype=np.float64)
        shifted: dict[tuple[int, int], NDArray[np.float64]] = {}
        for cell in range(n_cells):
            for sign in (1, -1):
                moved = cells.copy()
                moved[:, cell] += sign * step
                shifted[(cell, sign)] = -self.log_density(moved, observed)
            blocks[:, cell, cell] = (
                shifted[(cell, 1)] - 2.0 * centre + shifted[(cell, -1)]
            ) / step**2
        for row_cell in range(n_cells):
            for column_cell in range(row_cell):
                corners = np.zeros(cells.shape[0], dtype=np.float64)
                for row_sign in (1, -1):
                    for column_sign in (1, -1):
                        moved = cells.copy()
                        moved[:, row_cell] += row_sign * step
                        moved[:, column_cell] += column_sign * step
                        corners = corners + row_sign * column_sign * (
                            -self.log_density(moved, observed)
                        )
                value = corners / (4.0 * step**2)
                blocks[:, row_cell, column_cell] = value
                blocks[:, column_cell, row_cell] = value
        return blocks


def wiener_cell_design(features: NDArray[np.float64], *, n_extra_cells: int) -> NDArray[np.float64]:
    """Return the ``(rows, cells, parameters)`` design a Wiener predictor is built from.

    The drift cell holds the predictor features; every other cell is a single intercept
    column, because boundary, starting bias and non-decision time are trial-constant
    parameters of the family rather than functions of the design. Making them cells rather
    than a separate vector of "the other parameters" is what lets one combinator smooth a
    boundary, another let it vary by animal, and a third mix the whole density with a
    simpler process by appending cells of its own.
    """

    n_rows, n_features = features.shape
    n_cells = 1 + n_extra_cells
    design = np.zeros((n_rows, n_cells, n_features + n_extra_cells), dtype=np.float64)
    design[:, 0, :n_features] = features
    for cell in range(n_extra_cells):
        design[:, 1 + cell, n_features + cell] = 1.0
    return design


def wiener_cell_bytes(n_rows: int, n_cells: int, n_parameters: int) -> int:
    """The memory one dense Wiener design occupies, for reporting before allocating it."""

    return int(n_rows) * int(n_cells) * int(n_parameters) * np.dtype(np.float64).itemsize
