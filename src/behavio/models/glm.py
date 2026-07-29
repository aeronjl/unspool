"""Reference generalized linear models for binary behavioural outcomes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import expit

from behavio._internal.arrays import protected_array
from behavio.models.base import (
    FitDiagnostics,
    FitResult,
    ModelDataError,
    Prediction,
    PredictionMode,
    UnsupportedPredictionMode,
)
from behavio.study import REQUIRED_COLUMNS, Study


@dataclass(frozen=True, slots=True)
class CoefficientTrajectory:
    """Named coefficient values evaluated along an explicit temporal clock."""

    clock: str
    times: NDArray[np.float64]
    coefficient_names: tuple[str, ...]
    values: NDArray[np.float64]

    def __post_init__(self) -> None:
        times = protected_array(self.times, dtype=np.float64)
        values = protected_array(self.values, dtype=np.float64)
        names = tuple(self.coefficient_names)
        if times.ndim != 1 or not np.all(np.isfinite(times)):
            raise ValueError("trajectory times must be a finite one-dimensional array")
        if values.shape != (len(times), len(names)):
            raise ValueError("trajectory values must have one column per coefficient")
        if not np.all(np.isfinite(values)):
            raise ValueError("trajectory values must be finite")
        if not names or len(set(names)) != len(names):
            raise ValueError("coefficient names must be non-empty and unique")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "coefficient_names", names)
        object.__setattr__(self, "values", values)


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
    def coefficient_names(self) -> tuple[str, ...]:
        history = tuple(f"choice_lag_{lag}" for lag in range(1, self.choice_lags + 1))
        return ("intercept", *self.covariates, *history)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return self.coefficient_names

    @property
    def scored_columns(self) -> tuple[str, ...]:
        return (self.outcome,)

    @property
    def required_task_columns(self) -> tuple[str, ...]:
        """Covariates that must be declared as task predictors."""

        return self.covariates

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
        return _fit_bernoulli(
            model_name=self.model_name,
            model_signature=self.signature,
            parameter_names=self.parameter_names,
            design_matrix=design_matrix,
            outcomes=outcomes,
            penalty_matrix=np.diag(penalty),
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
            coefficient_warning_threshold=self.coefficient_warning_threshold,
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
        return protected_array(scores, dtype=np.float64)

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
        return self._base_feature_matrix(study, outcomes)

    def _base_feature_matrix(
        self, study: Study, outcomes: NDArray[np.float64] | None
    ) -> NDArray[np.float64]:
        matrix = np.ones((len(study), len(self.coefficient_names)), dtype=np.float64)
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


@dataclass(frozen=True, slots=True)
class SmoothBernoulliHistoryGLM(BernoulliHistoryGLM):
    """A Bernoulli history GLM with smoothly time-varying coefficients.

    Each coefficient is represented by values at fixed temporal knots and linearly
    interpolated between them. A first-difference penalty supplies a Gaussian random-walk
    prior over each path. Knots and their clock are fixed before fitting, so test outcomes
    cannot choose the temporal basis.
    """

    knots: tuple[float, ...] = (0.0, 1.0)
    time: str = "session_order"
    smoothness: float = 1.0
    shared_trajectory: bool = False

    def __post_init__(self) -> None:
        BernoulliHistoryGLM.__post_init__(self)
        try:
            knots = tuple(float(knot) for knot in self.knots)
        except (TypeError, ValueError):
            raise ValueError("knots must contain finite numbers") from None
        if len(knots) < 2 or not np.all(np.isfinite(knots)):
            raise ValueError("knots must contain at least two finite values")
        if any(right <= left for left, right in pairwise(knots)):
            raise ValueError("knots must be strictly increasing")
        if not isinstance(self.time, str) or not self.time:
            raise ValueError("time must be a non-empty Study column name")
        if self.time == self.outcome:
            raise ValueError("time and outcome columns must be distinct")
        if not np.isfinite(self.smoothness) or self.smoothness <= 0:
            raise ValueError("smoothness must be finite and positive")
        if not isinstance(self.shared_trajectory, bool):
            raise ValueError("shared_trajectory must be boolean")
        object.__setattr__(self, "knots", knots)

    @property
    def model_name(self) -> str:
        return "smooth-bernoulli-history-glm"

    @property
    def signature(self) -> str:
        covariates = ",".join(self.covariates)
        knots = ",".join(_format_time(knot) for knot in self.knots)
        return (
            f"{self.model_name}[outcome={self.outcome};covariates={covariates};"
            f"choice_lags={self.choice_lags};time={self.time};knots={knots};"
            f"smoothness={self.smoothness};l2={self.l2};"
            f"shared_trajectory={self.shared_trajectory}]"
        )

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(
            f"{coefficient}[{self.time}={_format_time(knot)}]"
            for coefficient in self.coefficient_names
            for knot in self.knots
        )

    def parameters_from_paths(self, paths: Mapping[str, Sequence[float]]) -> Mapping[str, float]:
        """Pack named coefficient paths into the model's simulation coordinates."""

        expected = set(self.coefficient_names)
        observed = set(paths)
        if observed != expected:
            raise ValueError(
                "paths must match the model coefficients exactly; "
                f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
            )
        parameters: dict[str, float] = {}
        offset = 0
        for coefficient in self.coefficient_names:
            values = np.asarray(paths[coefficient], dtype=np.float64)
            if values.shape != (len(self.knots),) or not np.all(np.isfinite(values)):
                raise ValueError(
                    f"path {coefficient!r} must contain one finite value per temporal knot"
                )
            for value in values:
                parameters[self.parameter_names[offset]] = float(value)
                offset += 1
        return MappingProxyType(parameters)

    def coefficient_trajectory(
        self, fit: FitResult, *, times: Sequence[float] | None = None
    ) -> CoefficientTrajectory:
        """Evaluate fitted coefficient paths at requested clock values."""

        self._validate_fit(fit)
        evaluation_times = self.knots if times is None else times
        time_array = np.asarray(evaluation_times, dtype=np.float64)
        basis = _linear_time_basis(time_array, self.knots)
        knot_values = fit.estimates.reshape(len(self.coefficient_names), len(self.knots))
        values = basis @ knot_values.T
        return CoefficientTrajectory(
            clock=self.time,
            times=time_array,
            coefficient_names=self.coefficient_names,
            values=values,
        )

    def simulate(
        self,
        design: Study,
        parameters: Mapping[str, float],
        *,
        seed: int | np.random.Generator,
    ) -> Study:
        """Generate choices recursively under smooth coefficient paths."""

        self._validate_study_scope(design)
        knot_values = self._parameter_vector(parameters).reshape(
            len(self.coefficient_names), len(self.knots)
        )
        basis = self._time_basis(design)
        coefficients = basis @ knot_values.T
        covariates = self._covariate_matrix(design)
        generator = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        choices = np.zeros(len(design), dtype=np.int8)

        for session_indices in _ordered_session_indices(design):
            generated_history: list[float] = []
            for index in session_indices:
                features = np.empty(len(self.coefficient_names), dtype=np.float64)
                features[0] = 1.0
                covariate_end = 1 + len(self.covariates)
                features[1:covariate_end] = covariates[index]
                for lag in range(1, self.choice_lags + 1):
                    features[covariate_end + lag - 1] = (
                        generated_history[-lag] if len(generated_history) >= lag else 0.0
                    )
                probability = expit(float(features @ coefficients[index]))
                choice = int(generator.binomial(1, probability))
                choices[index] = choice
                generated_history.append(2.0 * choice - 1.0)

        columns = {name: design[name] for name in design.columns}
        columns[self.outcome] = choices
        return Study(columns)

    def fit(self, study: Study) -> FitResult:
        """Fit smooth coefficient paths with a time-scaled random-walk penalty."""

        self._validate_study_scope(study)
        outcomes = self._outcomes(study)
        design_matrix = self._design_matrix(study, outcomes)
        penalty = self._penalty_matrix()
        return _fit_bernoulli(
            model_name=self.model_name,
            model_signature=self.signature,
            parameter_names=self.parameter_names,
            design_matrix=design_matrix,
            outcomes=outcomes,
            penalty_matrix=penalty,
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
            coefficient_warning_threshold=self.coefficient_warning_threshold,
        )

    def predict(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> Prediction:
        self._validate_study_scope(study)
        return BernoulliHistoryGLM.predict(self, study, fit, mode=mode)

    def _design_matrix(
        self, study: Study, outcomes: NDArray[np.float64] | None
    ) -> NDArray[np.float64]:
        features = self._base_feature_matrix(study, outcomes)
        basis = self._time_basis(study)
        return np.einsum("ij,ik->ijk", features, basis).reshape(len(study), -1)

    def _time_basis(self, study: Study) -> NDArray[np.float64]:
        if self.time not in study.columns:
            raise ModelDataError(f"study is missing temporal column {self.time!r}")
        try:
            times = np.asarray(study[self.time], dtype=np.float64)
        except (TypeError, ValueError):
            raise ModelDataError(f"temporal column {self.time!r} must be numeric") from None
        if not np.all(np.isfinite(times)):
            raise ModelDataError(f"temporal column {self.time!r} must be finite")
        try:
            return _linear_time_basis(times, self.knots)
        except ValueError as error:
            raise ModelDataError(str(error)) from None

    def _penalty_matrix(self) -> NDArray[np.float64]:
        n_knots = len(self.knots)
        differences = np.zeros((n_knots - 1, n_knots), dtype=np.float64)
        for row, spacing in enumerate(np.diff(self.knots)):
            scale = 1.0 / np.sqrt(spacing)
            differences[row, row] = -scale
            differences[row, row + 1] = scale
        roughness = self.smoothness * (differences.T @ differences)
        penalty = np.kron(np.eye(len(self.coefficient_names)), roughness)
        if self.l2:
            for coefficient in range(1, len(self.coefficient_names)):
                start = coefficient * n_knots
                penalty[start : start + n_knots, start : start + n_knots] += self.l2 * np.eye(
                    n_knots
                )
        return penalty

    def _validate_study_scope(self, study: Study) -> None:
        if len(study.subjects) > 1 and not self.shared_trajectory:
            raise ModelDataError(
                "smooth coefficient paths are subject-specific by default; fit one subject "
                "at a time or set shared_trajectory=True to align subjects explicitly"
            )


def _fit_bernoulli(
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


def _linear_time_basis(
    times: Sequence[float] | NDArray[np.float64], knots: tuple[float, ...]
) -> NDArray[np.float64]:
    values = np.asarray(times, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("temporal values must be a finite one-dimensional array")
    if np.any(values < knots[0]) or np.any(values > knots[-1]):
        raise ValueError(
            f"temporal values must lie within the fixed knot range [{knots[0]}, {knots[-1]}]"
        )

    basis = np.zeros((len(values), len(knots)), dtype=np.float64)
    knot_array = np.asarray(knots, dtype=np.float64)
    for row, value in enumerate(values):
        if value == knots[-1]:
            basis[row, -1] = 1.0
            continue
        right = int(np.searchsorted(knot_array, value, side="right"))
        left = right - 1
        span = knots[right] - knots[left]
        right_weight = (value - knots[left]) / span
        basis[row, left] = 1.0 - right_weight
        basis[row, right] = right_weight
    return basis


def _format_time(value: float) -> str:
    return np.format_float_positional(value, trim="-")


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
