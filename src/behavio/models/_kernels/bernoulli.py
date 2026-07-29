"""The penalized Bernoulli likelihood and the session ordering every family shares.

``behavio.models.glm`` had become the de-facto utility module for seven sibling families
purely because it happened to be where these two functions were first written. Neither is
about generalized linear models: :func:`fit_bernoulli` fits any design matrix against a
binary outcome under a quadratic penalty, and :func:`ordered_session_indices` is a
statement about Behavio's trial-order contract, not about any one likelihood.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import expit

from behavio.contracts.audit import FitDiagnostics
from behavio.contracts.estimator import FitResult
from behavio.study import Study


def fit_bernoulli(
    *,
    model_name: str,
    model_signature: str,
    parameter_names: tuple[str, ...],
    design_matrix: NDArray[np.float64],
    outcomes: NDArray[np.float64],
    penalty_matrix: NDArray[np.float64],
    max_iterations: int,
    tolerance: float,
    coefficient_warning_threshold: float,
) -> FitResult:
    """Fit a quadratically penalized Bernoulli likelihood with deterministic L-BFGS-B."""

    def objective(coefficients: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        linear_predictor = design_matrix @ coefficients
        loss = np.logaddexp(0.0, linear_predictor).sum() - outcomes @ linear_predictor
        loss += 0.5 * float(coefficients @ penalty_matrix @ coefficients)
        gradient = design_matrix.T @ (expit(linear_predictor) - outcomes)
        gradient += penalty_matrix @ coefficients
        return float(loss), np.asarray(gradient, dtype=np.float64)

    result = minimize(
        objective,
        np.zeros(len(parameter_names), dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": max_iterations, "ftol": tolerance, "gtol": tolerance},
    )
    estimates = np.asarray(result.x, dtype=np.float64)
    probabilities = expit(design_matrix @ estimates)
    weights = probabilities * (1.0 - probabilities)
    hessian = design_matrix.T @ (weights[:, None] * design_matrix) + penalty_matrix
    condition = float(np.linalg.cond(hessian))
    covariance = np.linalg.pinv(hessian, hermitian=True)
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    _, gradient = objective(estimates)
    diagnostics = FitDiagnostics(
        converged=bool(result.success),
        optimizer="L-BFGS-B",
        status=int(result.status),
        message=str(result.message),
        n_iterations=int(result.nit),
        objective=float(result.fun),
        gradient_norm=float(np.linalg.norm(gradient)),
        hessian_condition=condition,
        boundary_estimate=bool(np.any(np.abs(estimates) >= coefficient_warning_threshold)),
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
    )


def ordered_session_indices(study: Study) -> tuple[tuple[int, ...], ...]:
    """Group source row indices by subject and session in chronological order."""

    sessions: dict[tuple[Any, Any], list[int]] = {}
    for raw_index in study.chronological_indices():
        index = int(raw_index)
        subject = _scalar(study["subject"][index])
        session = _scalar(study["session"][index])
        sessions.setdefault((subject, session), []).append(index)
    return tuple(tuple(indices) for indices in sessions.values())


def _scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value
