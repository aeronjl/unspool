"""One deterministic solver for any quadratically penalised linear-predictor problem.

:func:`fit_penalised_linear` is the arithmetic that used to be written out once per family:
minimise ``likelihood(X theta + offset, y) + 0.5 theta' P theta`` with L-BFGS-B on the
analytic gradient, then read the observed information off the likelihood's own curvature.
Nothing in it is family-specific and nothing in it is shape-specific -- a Bernoulli GLM's
``(rows,)`` predictor and a multinomial logit's ``(rows, categories)`` predictor differ only
in which of the two contractions in :mod:`behavio.contracts.compose` carry a gradient and a
curvature back to the coordinate.

Having one solver is what makes :meth:`~behavio.contracts.compose.PenalisedLinearEstimator.\
fit_penalised` an honest promise: a composed fit runs the model's own arithmetic on a wider
problem, and "the model's own arithmetic" is now a single function rather than a family's
private copy of it.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from behavio.contracts.audit import FitDiagnostics
from behavio.contracts.compose import (
    LinearPredictorLikelihood,
    information_matrix,
    linear_predictor,
    parameter_gradient,
)
from behavio.contracts.estimator import FitResult
from behavio.inference.optimize import OptimizationProblem, ScipyMultistart
from behavio.inference.parameters import ParameterSpace, ParameterSpec


def fit_penalised_linear(
    *,
    model_name: str,
    model_signature: str,
    parameter_names: tuple[str, ...],
    design_matrix: NDArray[np.float64],
    outcomes: NDArray[np.float64],
    penalty_matrix: NDArray[np.float64],
    likelihood: LinearPredictorLikelihood,
    max_iterations: int,
    tolerance: float,
    coefficient_warning_threshold: float,
    offsets: NDArray[np.float64] | None = None,
    box: NDArray[np.float64] | None = None,
    initial_points: tuple[NDArray[np.float64], ...] | None = None,
    derived_estimates: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
    optimizer: str = "L-BFGS-B",
) -> FitResult:
    """Fit a quadratically penalised linear-predictor problem with deterministic L-BFGS-B.

    ``derived_estimates`` names quantities that are functions of the solution but not
    coordinates of it -- a hierarchical fit's population-plus-deviation -- and they are held
    to the same ``coefficient_warning_threshold`` as the coordinate itself, so that the
    boundary a composed fit reports is decided by the same convention as the one it wraps.

    ``box`` and ``initial_points`` are what a *mixed* problem needs and an unmixed one never
    did. A penalised generalized linear objective is convex and has one optimum, reached
    from the origin; mixing a model with a simpler process is not convex in the weight and
    the model's parameters jointly, so the problem becomes a multi-start one and the wrapped
    family's solver has to be able to run it as one. Both default to ``None``, and with both
    absent this is the single search from the origin it has always been, on the same doubles
    in the same order.
    """

    def objective(coefficients: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        predictor = linear_predictor(design_matrix, coefficients, offsets)
        loss, predictor_gradient = likelihood.value_and_gradient(predictor, outcomes)
        loss += 0.5 * float(coefficients @ penalty_matrix @ coefficients)
        gradient = parameter_gradient(design_matrix, predictor_gradient)
        gradient += penalty_matrix @ coefficients
        return float(loss), np.asarray(gradient, dtype=np.float64)

    starts = (
        (np.zeros(len(parameter_names), dtype=np.float64),)
        if initial_points is None
        else tuple(initial_points)
    )
    parameter_space = _coordinate_parameter_space(parameter_names, box)
    problem = OptimizationProblem(
        parameter_space=parameter_space,
        objective=objective,
        starts=starts,
        has_gradient=True,
        objective_name=f"{model_name}_penalised_linear_negative_log_likelihood",
    )
    run = ScipyMultistart(
        max_iterations=max_iterations,
        function_tolerance=tolerance,
        gradient_tolerance=tolerance,
    ).run(problem)
    selected = run.selected
    if selected is None:
        messages = "; ".join(attempt.message for attempt in run.attempts)
        raise ValueError(f"all {model_name} restarts produced non-finite objectives: {messages}")
    estimates = np.asarray(selected.estimate, dtype=np.float64)
    curvature = likelihood.curvature(linear_predictor(design_matrix, estimates, offsets), outcomes)
    hessian = information_matrix(design_matrix, curvature) + penalty_matrix
    condition = float(np.linalg.cond(hessian))
    covariance = np.linalg.pinv(hessian, hermitian=True)
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    _, gradient = objective(estimates)
    reported = (
        estimates
        if derived_estimates is None
        else np.concatenate(
            [estimates, np.ravel(np.asarray(derived_estimates(estimates), dtype=np.float64))]
        )
    )
    diagnostics = FitDiagnostics(
        converged=selected.converged,
        optimizer=optimizer,
        status=selected.status,
        message=selected.message,
        n_iterations=selected.n_iterations,
        objective=selected.objective,
        gradient_norm=float(np.linalg.norm(gradient)),
        hessian_condition=condition,
        boundary_estimate=bool(np.any(np.abs(reported) >= coefficient_warning_threshold)),
    )
    return FitResult(
        model_name=model_name,
        model_signature=model_signature,
        parameter_names=parameter_names,
        estimates=estimates,
        standard_errors=standard_errors,
        covariance=covariance,
        n_observations=len(outcomes),
        diagnostics=diagnostics,
        optimization_run=run,
    )


def _coordinate_parameter_space(
    parameter_names: tuple[str, ...], box: NDArray[np.float64] | None
) -> ParameterSpace:
    """Describe an already-transformed solver coordinate to a common backend."""

    bounds = [None] * len(parameter_names) if box is None else list(np.asarray(box))
    return ParameterSpace(
        tuple(
            ParameterSpec(
                name=name,
                bounds=(None, None)
                if interval is None
                else (float(interval[0]), float(interval[1])),
                optimizer_bounds=None
                if interval is None
                else (float(interval[0]), float(interval[1])),
                description="Model or composed-model solver coordinate.",
            )
            for name, interval in zip(parameter_names, bounds, strict=True)
        )
    )
