"""Reference generalized linear models for binary behavioural outcomes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import expit

from unspool.models.base import (
    FitDiagnostics,
    FitResult,
    ModelDataError,
    Prediction,
    PredictionMode,
    UnsupportedPredictionMode,
    _protected_array,
)
from unspool.study import REQUIRED_COLUMNS, Study


@dataclass(frozen=True, slots=True)
class BernoulliHistoryGLM:
    """A static Bernoulli GLM with exogenous covariates and choice history.

    Previous choices are constructed within subject/session boundaries and effect-coded as
    -1 and +1. Missing history at the beginning of each session is encoded as zero. During
    simulation, history is updated recursively from generated choices; during prediction,
    observed past choices provide one-step-ahead filtered history.
    """

    covariates: tuple[str, ...] = ()
    outcome: str = "choice"
    choice_lags: int = 1
    l2: float = 0.0
    max_iterations: int = 1_000
    tolerance: float = 1e-9
    coefficient_warning_threshold: float = 20.0

    def __post_init__(self) -> None:
        covariates = tuple(self.covariates)
        if len(set(covariates)) != len(covariates):
            raise ValueError("covariates must be unique")
        if any(not isinstance(name, str) or not name for name in covariates):
            raise ValueError("covariate names must be non-empty strings")
        if not isinstance(self.outcome, str) or not self.outcome:
            raise ValueError("outcome must be a non-empty column name")
        if self.outcome in REQUIRED_COLUMNS:
            raise ValueError("outcome cannot replace a required Study column")
        if self.outcome in covariates:
            raise ValueError("the outcome cannot also be a covariate")
        if isinstance(self.choice_lags, bool) or not isinstance(self.choice_lags, int):
            raise ValueError("choice_lags must be a non-negative integer")
        if self.choice_lags < 0:
            raise ValueError("choice_lags must be a non-negative integer")
        if not np.isfinite(self.l2) or self.l2 < 0:
            raise ValueError("l2 must be finite and non-negative")
        if (
            isinstance(self.max_iterations, bool)
            or not isinstance(self.max_iterations, int)
            or self.max_iterations < 1
        ):
            raise ValueError("max_iterations must be a positive integer")
        if not np.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("tolerance must be finite and positive")
        if (
            not np.isfinite(self.coefficient_warning_threshold)
            or self.coefficient_warning_threshold <= 0
        ):
            raise ValueError("coefficient_warning_threshold must be finite and positive")
        reserved = {"intercept", *(f"choice_lag_{lag}" for lag in range(1, self.choice_lags + 1))}
        conflict = reserved.intersection(covariates)
        if conflict:
            raise ValueError(f"covariate names conflict with model parameters: {sorted(conflict)}")
        object.__setattr__(self, "covariates", covariates)

    @property
    def model_name(self) -> str:
        return "bernoulli-history-glm"

    @property
    def signature(self) -> str:
        covariates = ",".join(self.covariates)
        return (
            f"{self.model_name}[outcome={self.outcome};covariates={covariates};"
            f"choice_lags={self.choice_lags};l2={self.l2}]"
        )

    @property
    def parameter_names(self) -> tuple[str, ...]:
        history = tuple(f"choice_lag_{lag}" for lag in range(1, self.choice_lags + 1))
        return ("intercept", *self.covariates, *history)

    @property
    def supported_prediction_modes(self) -> tuple[PredictionMode, ...]:
        return (PredictionMode.FILTERED,)

    def simulate(
        self,
        design: Study,
        parameters: Mapping[str, float],
        *,
        seed: int | np.random.Generator,
    ) -> Study:
        """Generate choices in chronological order while preserving source row order."""

        coefficients = self._parameter_vector(parameters)
        covariates = self._covariate_matrix(design)
        generator = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        choices = np.zeros(len(design), dtype=np.int8)

        history_start = 1 + len(self.covariates)
        for session_indices in _ordered_session_indices(design):
            generated_history: list[float] = []
            for index in session_indices:
                linear_predictor = coefficients[0]
                if self.covariates:
                    linear_predictor += float(covariates[index] @ coefficients[1:history_start])
                for lag in range(1, self.choice_lags + 1):
                    history_value = (
                        generated_history[-lag] if len(generated_history) >= lag else 0.0
                    )
                    linear_predictor += coefficients[history_start + lag - 1] * history_value
                choice = int(generator.binomial(1, expit(linear_predictor)))
                choices[index] = choice
                generated_history.append(2.0 * choice - 1.0)

        columns = {name: design[name] for name in design.columns}
        columns[self.outcome] = choices
        return Study(columns)

    def fit(self, study: Study) -> FitResult:
        """Fit the penalized Bernoulli likelihood with deterministic L-BFGS-B."""

        outcomes = self._outcomes(study)
        design_matrix = self._design_matrix(study, outcomes)
        penalty = np.zeros(len(self.parameter_names), dtype=np.float64)
        penalty[1:] = self.l2

        def objective(coefficients: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
            linear_predictor = design_matrix @ coefficients
            loss = np.logaddexp(0.0, linear_predictor).sum() - outcomes @ linear_predictor
            loss += 0.5 * float(np.dot(penalty, coefficients**2))
            gradient = design_matrix.T @ (expit(linear_predictor) - outcomes)
            gradient += penalty * coefficients
            return float(loss), np.asarray(gradient, dtype=np.float64)

        result = minimize(
            objective,
            np.zeros(len(self.parameter_names), dtype=np.float64),
            method="L-BFGS-B",
            jac=True,
            options={
                "maxiter": self.max_iterations,
                "ftol": self.tolerance,
                "gtol": self.tolerance,
            },
        )
        estimates = np.asarray(result.x, dtype=np.float64)
        probabilities = expit(design_matrix @ estimates)
        weights = probabilities * (1.0 - probabilities)
        hessian = design_matrix.T @ (weights[:, None] * design_matrix)
        hessian += np.diag(penalty)
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
            boundary_estimate=bool(np.any(np.abs(estimates) >= self.coefficient_warning_threshold)),
        )
        return FitResult(
            model_name=self.model_name,
            model_signature=self.signature,
            parameter_names=self.parameter_names,
            estimates=estimates,
            standard_errors=standard_errors,
            covariance=covariance,
            n_observations=len(study),
            diagnostics=diagnostics,
        )

    def predict(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> Prediction:
        """Return one-step-ahead probabilities using observed past choices only."""

        prediction_mode = self._prediction_mode(mode)
        self._validate_fit(fit)
        outcomes = self._outcomes(study) if self.choice_lags else None
        design_matrix = self._design_matrix(study, outcomes)
        linear_predictor = design_matrix @ fit.estimates
        return Prediction(
            probability=expit(linear_predictor),
            linear_predictor=linear_predictor,
            mode=prediction_mode,
        )

    def pointwise_log_prob(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> NDArray[np.float64]:
        """Score each observed choice without conditioning on future choices."""

        outcomes = self._outcomes(study)
        prediction = self.predict(study, fit, mode=mode)
        scores = outcomes * -np.logaddexp(0.0, -prediction.linear_predictor)
        scores += (1.0 - outcomes) * -np.logaddexp(0.0, prediction.linear_predictor)
        return _protected_array(scores, dtype=np.float64)

    def _parameter_vector(self, parameters: Mapping[str, float]) -> NDArray[np.float64]:
        expected = set(self.parameter_names)
        observed = set(parameters)
        if observed != expected:
            raise ValueError(
                "parameters must match the model exactly; "
                f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
            )
        values = np.asarray([parameters[name] for name in self.parameter_names], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("parameters must be finite")
        return values

    def _outcomes(self, study: Study) -> NDArray[np.float64]:
        if self.outcome not in study.columns:
            raise ModelDataError(f"study is missing outcome column {self.outcome!r}")
        try:
            outcomes = np.asarray(study[self.outcome], dtype=np.float64)
        except (TypeError, ValueError):
            raise ModelDataError(f"outcome column {self.outcome!r} must be numeric") from None
        if not np.all(np.isfinite(outcomes)) or not np.all((outcomes == 0) | (outcomes == 1)):
            raise ModelDataError(f"outcome column {self.outcome!r} must contain only zero and one")
        return outcomes

    def _covariate_matrix(self, study: Study) -> NDArray[np.float64]:
        if not self.covariates:
            return np.empty((len(study), 0), dtype=np.float64)
        missing = [name for name in self.covariates if name not in study.columns]
        if missing:
            raise ModelDataError(f"study is missing covariate columns: {missing}")
        try:
            matrix = np.column_stack(
                [np.asarray(study[name], dtype=np.float64) for name in self.covariates]
            )
        except (TypeError, ValueError):
            raise ModelDataError("covariate columns must be numeric") from None
        if not np.all(np.isfinite(matrix)):
            raise ModelDataError("covariate columns must be finite")
        return matrix

    def _design_matrix(
        self, study: Study, outcomes: NDArray[np.float64] | None
    ) -> NDArray[np.float64]:
        matrix = np.ones((len(study), len(self.parameter_names)), dtype=np.float64)
        covariate_end = 1 + len(self.covariates)
        matrix[:, 1:covariate_end] = self._covariate_matrix(study)
        if self.choice_lags:
            if outcomes is None:
                raise ModelDataError("observed choices are required to construct filtered history")
            matrix[:, covariate_end:] = 0.0
            for session_indices in _ordered_session_indices(study):
                for position, index in enumerate(session_indices):
                    for lag in range(1, self.choice_lags + 1):
                        if position >= lag:
                            previous_choice = outcomes[session_indices[position - lag]]
                            matrix[index, covariate_end + lag - 1] = 2.0 * previous_choice - 1.0
        return matrix

    def _validate_fit(self, fit: FitResult) -> None:
        if fit.model_signature != self.signature or fit.parameter_names != self.parameter_names:
            raise ValueError("fit result was produced by a different model specification")

    def _prediction_mode(self, mode: PredictionMode) -> PredictionMode:
        prediction_mode = PredictionMode(mode)
        if prediction_mode not in self.supported_prediction_modes:
            raise UnsupportedPredictionMode(
                f"{self.model_name} supports only filtered prediction, "
                f"not {prediction_mode.value!r}"
            )
        return prediction_mode


def _ordered_session_indices(study: Study) -> tuple[tuple[int, ...], ...]:
    sessions: dict[tuple[Any, Any], list[int]] = {}
    for raw_index in study.chronological_indices():
        index = int(raw_index)
        subject = _scalar(study["subject"][index])
        session = _scalar(study["session"][index])
        sessions.setdefault((subject, session), []).append(index)
    return tuple(tuple(indices) for indices in sessions.values())


def _scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value
