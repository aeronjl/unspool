"""One solver for every bounded-coordinate model's row-coefficient problem.

:meth:`~behavio.contracts.bounded.BoundedCoordinateEstimator.fit_rows` is "the model's own
solver on a wider problem", exactly as ``fit_penalised`` is on the penalised-linear side.
For the three bounded-coordinate families the model's own solver is the *same* solver --
deterministic multi-start L-BFGS-B inside a finite box, a numerical Hessian differenced from
the analytic gradient, and a pseudo-inverse covariance -- so writing it once here is what
keeps ``fit_rows`` on each model down to the two things that genuinely are the model's own:
its restart count and tolerance, and its convention for what counts as a boundary estimate.

The conditional group covariances a hierarchical EM step needs are read off the same joint
Hessian rather than recomputed. A group's deviation block, inverted on its own, *is* the
Laplace covariance of that group's deviation conditional on the population -- which is the
second moment the M-step asks for, and the reason
:class:`~behavio.contracts.compose.PenalisedFitResult` exists.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from behavio.contracts.bounded import RowCoefficientDesign
from behavio.contracts.compose import PenalisedFitResult
from behavio.contracts.estimator import FitDiagnostics, ModelDataError
from behavio.inference.optimize import OptimizationProblem, ScipyMultistart
from behavio.inference.parameters import ParameterSpace, ParameterSpec
from behavio.models._kernels.curvature import finite_difference_hessian, offset_steps

__all__ = ["solve_row_coefficients"]


def solve_row_coefficients(
    design: RowCoefficientDesign,
    *,
    model_name: str,
    model_signature: str,
    optimizer: str,
    max_iterations: int,
    tolerance: float,
    boundary: Callable[[NDArray[np.float64], NDArray[np.float64] | None], bool],
) -> PenalisedFitResult:
    """Solve one row-coefficient problem by multi-start L-BFGS-B and report its curvature.

    ``boundary`` is the model's own convention, and it is given two arguments because a
    hierarchical fit has two things that can be at a bound: the joint estimate itself, and
    the population-plus-deviation values that
    :attr:`~behavio.contracts.bounded.RowCoefficientDesign.derived_estimates` names and that
    are not coordinates of the vector the optimizer returned.
    """

    starts = design.initial_points
    if not starts:
        raise ValueError("a row-coefficient fit needs at least one starting vector")
    parameter_space = _coordinate_parameter_space(design.parameter_names, design.box)
    problem = OptimizationProblem(
        parameter_space=parameter_space,
        objective=design.value_and_gradient,
        starts=starts,
        has_gradient=True,
        objective_name=f"{model_name}_row_coefficient_negative_log_likelihood",
    )
    run = ScipyMultistart(
        max_iterations=max_iterations,
        function_tolerance=tolerance,
        gradient_tolerance=tolerance,
    ).run(problem)
    chosen = run.selected
    if chosen is None:
        messages = "; ".join(attempt.message for attempt in run.attempts)
        raise ModelDataError(
            f"all {model_name} restarts produced non-finite objectives: {messages}"
        )
    estimates = np.asarray(chosen.estimate, dtype=np.float64)
    value, gradient = design.value_and_gradient(estimates)
    hessian = finite_difference_hessian(
        lambda vector: design.value_and_gradient(vector)[1],
        estimates,
        steps=offset_steps(estimates, scale=1e-5),
    )
    covariance = np.linalg.pinv(hessian, hermitian=True)
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    derived = None if design.derived_estimates is None else design.derived_estimates(estimates)
    diagnostics = FitDiagnostics(
        converged=chosen.converged,
        optimizer=f"{optimizer} ({len(starts)} deterministic restarts)",
        status=chosen.status,
        message=chosen.message,
        n_iterations=chosen.n_iterations,
        objective=float(value),
        gradient_norm=float(np.linalg.norm(gradient)),
        hessian_condition=float(np.linalg.cond(hessian)),
        boundary_estimate=_at_box(estimates, design.box) or bool(boundary(estimates, derived)),
    )
    return PenalisedFitResult(
        model_name=model_name,
        model_signature=model_signature,
        parameter_names=design.parameter_names,
        estimates=estimates,
        standard_errors=standard_errors,
        covariance=covariance,
        n_observations=design.n_observations,
        diagnostics=diagnostics,
        conditional_group_covariances=_conditional_group_covariances(design, hessian),
        optimization_run=run,
    )


def _coordinate_parameter_space(
    parameter_names: tuple[str, ...], box: NDArray[np.float64] | None
) -> ParameterSpace:
    """Describe an already-transformed row coordinate to a common backend."""

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
                description="Model or composed-model row solver coordinate.",
            )
            for name, interval in zip(parameter_names, bounds, strict=True)
        )
    )


def _at_box(estimates: NDArray[np.float64], box: NDArray[np.float64] | None) -> bool:
    """Whether any coordinate rests on the box the transform was applied to reach.

    Generic rather than a model's own convention, and the one boundary check that is: a
    logit coordinate at ``-12`` is a rate the data cannot distinguish from zero whatever the
    family, and a group deviation resting on the width of its parameter's range is a group
    the population could not be shrunk towards.
    """

    if box is None:
        return False
    tolerances = 1e-4 * np.maximum(1.0, box[:, 1] - box[:, 0])
    return bool(
        np.any(estimates - box[:, 0] <= tolerances) or np.any(box[:, 1] - estimates <= tolerances)
    )


def _conditional_group_covariances(
    design: RowCoefficientDesign, hessian: NDArray[np.float64]
) -> NDArray[np.float64] | None:
    """Invert each group's own Hessian block, which is its conditional covariance."""

    expansion = design.expansion
    if expansion is None:
        return None
    blocks = np.empty((expansion.n_groups, expansion.width, expansion.width), dtype=np.float64)
    for group in range(expansion.n_groups):
        span = expansion.group_slice(group)
        blocks[group] = np.linalg.pinv(hessian[span, span], hermitian=True)
    return blocks
