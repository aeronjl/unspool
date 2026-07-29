"""Reference multinomial and omission-aware choice likelihoods."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import logsumexp

from behavio._internal.arrays import protected_array
from behavio.design import (
    DesignSpec,
    HistoryKernelTerm,
    HistoryTerm,
    InteractionTerm,
)
from behavio.models.base import (
    CategoricalPrediction,
    FitDiagnostics,
    FitResult,
    ModelDataError,
    PredictionMode,
    UnsupportedPredictionMode,
)
from behavio.study import REQUIRED_COLUMNS, Study
from behavio.task import ChoiceData, ChoiceSpec, TaskValidationError


@dataclass(frozen=True, slots=True)
class MultinomialLogit:
    """Treatment-coded softmax regression on a fixed task and design coordinate.

    ``include_omission=True`` pools every omission representation declared by
    :class:`~behavio.task.ChoiceSpec` into one additional modeled category. Trial-specific
    unavailable actions receive exactly zero probability; the omission category remains
    available because it represents failure to emit any valid action.
    """

    choice: ChoiceSpec
    design: DesignSpec = field(default_factory=DesignSpec)
    reference: Any | None = None
    include_omission: bool = False
    omission_label: Any | None = None
    l2: float = 0.0
    max_iterations: int = 1_000
    tolerance: float = 1e-9
    coefficient_warning_threshold: float = 20.0

    def __post_init__(self) -> None:
        if not isinstance(self.choice, ChoiceSpec):
            raise TypeError("choice must be a ChoiceSpec")
        if not isinstance(self.design, DesignSpec):
            raise TypeError("design must be a DesignSpec")
        if not isinstance(self.include_omission, bool):
            raise ValueError("include_omission must be boolean")
        if self.include_omission:
            if not self.choice.omission_values:
                raise ValueError("omission-aware likelihoods require ChoiceSpec.omission_values")
            label = (
                self.choice.omission_values[0]
                if self.omission_label is None
                else _scalar(self.omission_label)
            )
            if _label_key(label) not in {
                _label_key(value) for value in self.choice.omission_values
            }:
                raise ValueError("omission_label must be one of ChoiceSpec.omission_values")
            object.__setattr__(self, "omission_label", label)
        elif self.omission_label is not None:
            raise ValueError("omission_label requires include_omission=True")
        reference = self.choice.options[0] if self.reference is None else _scalar(self.reference)
        if _label_key(reference) not in {_label_key(value) for value in self.categories}:
            raise ValueError("reference must be one of the modeled categories")
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
        object.__setattr__(self, "reference", reference)
        for category in self.categories:
            _portable_category(category)

    @property
    def model_name(self) -> str:
        return "omission-aware-multinomial-logit" if self.include_omission else "multinomial-logit"

    @property
    def categories(self) -> tuple[Any, ...]:
        if self.include_omission:
            return (*self.choice.options, self.omission_label)
        return self.choice.options

    @property
    def signature(self) -> str:
        return (
            f"{self.model_name}[choice={self.choice.column!r};categories={self.categories!r};"
            f"reference={self.reference!r};design={self.design.signature};l2={self.l2}]"
        )

    @property
    def scored_columns(self) -> tuple[str, ...]:
        return (self.choice.column,)

    @property
    def required_task_columns(self) -> tuple[str, ...]:
        columns = self.design.required_columns
        if self.choice.available_options_column is not None:
            columns = (*columns, self.choice.available_options_column)
        return tuple(
            column
            for column in dict.fromkeys(columns)
            if column != self.choice.column and column not in REQUIRED_COLUMNS
        )

    @property
    def supported_prediction_modes(self) -> tuple[PredictionMode, ...]:
        return (PredictionMode.FILTERED,)

    @property
    def reference_index(self) -> int:
        key = _label_key(self.reference)
        return next(
            index for index, value in enumerate(self.categories) if _label_key(value) == key
        )

    @property
    def estimated_category_indices(self) -> tuple[int, ...]:
        return tuple(
            index for index in range(len(self.categories)) if index != self.reference_index
        )

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(
            f"category[{self.categories[category]!r}]::{feature}"
            for category in self.estimated_category_indices
            for feature in self.design.feature_names
        )

    def outcome_codes(self, study: Study) -> NDArray[np.int64]:
        data = self._choice_data(study)
        codes = np.asarray(data.codes, dtype=np.int64).copy()
        if np.any(data.omitted):
            if not self.include_omission:
                raise ModelDataError(
                    "observed omissions require include_omission=True or explicit exclusion"
                )
            codes[data.omitted] = len(self.categories) - 1
        return protected_array(codes, dtype=np.int64)

    def fit(self, study: Study) -> FitResult:
        matrix = self.design.build(study)
        parameter_names = self._parameter_names(matrix.names)
        outcomes = self.outcome_codes(study)
        available = self._availability(study)
        n_categories = len(self.categories)
        n_features = matrix.values.shape[1]
        estimated = self.estimated_category_indices
        penalty = np.asarray([name != "intercept" for name in matrix.names], dtype=np.float64)

        def objective(vector: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
            coefficients = vector.reshape(len(estimated), n_features)
            logits = np.zeros((len(study), n_categories), dtype=np.float64)
            logits[:, estimated] = matrix.values @ coefficients.T
            logits[~available] = -np.inf
            normalizer = logsumexp(logits, axis=1)
            loss = -float(np.sum(logits[np.arange(len(study)), outcomes] - normalizer))
            probabilities = np.exp(logits - normalizer[:, None])
            residual = probabilities
            residual[np.arange(len(study)), outcomes] -= 1.0
            gradient = residual[:, estimated].T @ matrix.values
            if self.l2:
                loss += 0.5 * self.l2 * float(np.sum((coefficients * penalty) ** 2))
                gradient += self.l2 * coefficients * penalty
            return loss, gradient.ravel()

        start = np.zeros(len(parameter_names), dtype=np.float64)
        result = minimize(
            objective,
            start,
            method="L-BFGS-B",
            jac=True,
            options={
                "maxiter": self.max_iterations,
                "ftol": self.tolerance,
                "gtol": self.tolerance,
            },
        )
        estimates = np.asarray(result.x, dtype=np.float64)
        _, gradient = objective(estimates)
        hessian = self._hessian(
            matrix.values,
            available,
            estimates.reshape(len(estimated), n_features),
            penalty,
        )
        condition = float(np.linalg.cond(hessian))
        covariance = np.linalg.pinv(hessian, hermitian=True)
        standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        return FitResult(
            model_name=self.model_name,
            model_signature=self.signature,
            parameter_names=parameter_names,
            estimates=estimates,
            standard_errors=standard_errors,
            covariance=covariance,
            n_observations=len(study),
            diagnostics=FitDiagnostics(
                converged=bool(result.success),
                optimizer="L-BFGS-B (analytic softmax gradient)",
                status=int(result.status),
                message=str(result.message),
                n_iterations=int(result.nit),
                objective=float(result.fun),
                gradient_norm=float(np.linalg.norm(gradient)),
                hessian_condition=condition,
                boundary_estimate=bool(
                    np.any(np.abs(estimates) >= self.coefficient_warning_threshold)
                ),
            ),
        )

    def predict(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> CategoricalPrediction:
        prediction_mode = self._prediction_mode(mode)
        matrix = self.design.build(study)
        self._validate_fit(fit, matrix.names)
        logits = self._logits(matrix.values, fit.estimates, self._availability(study))
        probability = np.exp(logits - logsumexp(logits, axis=1)[:, None])
        return CategoricalPrediction(probability, logits, self.categories, prediction_mode)

    def pointwise_log_prob(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> NDArray[np.float64]:
        prediction = self.predict(study, fit, mode=mode)
        outcomes = self.outcome_codes(study)
        scores = prediction.linear_predictor[np.arange(len(study)), outcomes]
        scores -= logsumexp(prediction.linear_predictor, axis=1)
        return protected_array(scores, dtype=np.float64)

    def simulate(
        self,
        design: Study,
        parameters: Mapping[str, float],
        *,
        seed: int | np.random.Generator,
    ) -> Study:
        if any(_uses_outcome_history(term, self.choice.column) for term in self.design.terms):
            raise ModelDataError(
                "multinomial simulation currently requires outcome-independent design terms"
            )
        matrix = self.design.build(design)
        names = self._parameter_names(matrix.names)
        if set(parameters) != set(names):
            raise ValueError("parameters must match the model and built design exactly")
        try:
            estimates = np.asarray([parameters[name] for name in names], dtype=np.float64)
        except (TypeError, ValueError):
            raise ValueError("parameters must contain finite numeric values") from None
        if not np.all(np.isfinite(estimates)):
            raise ValueError("parameters must contain finite numeric values")
        logits = self._logits(matrix.values, estimates, self._availability(design))
        probability = np.exp(logits - logsumexp(logits, axis=1)[:, None])
        generator = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        codes = np.asarray(
            [generator.choice(len(self.categories), p=row) for row in probability],
            dtype=np.int64,
        )
        choices = np.asarray([self.categories[index] for index in codes], dtype=object)
        columns = {name: design[name] for name in design.columns}
        columns[self.choice.column] = choices
        return Study(columns)

    def _choice_data(self, study: Study) -> ChoiceData:
        try:
            return self.choice.read(study)
        except TaskValidationError as error:
            raise ModelDataError(str(error)) from error

    def _availability(self, study: Study) -> NDArray[np.bool_]:
        try:
            actions = self.choice.availability(study)
        except TaskValidationError as error:
            raise ModelDataError(str(error)) from error
        if self.include_omission:
            return np.column_stack((actions, np.ones(len(study), dtype=np.bool_)))
        return actions

    def _parameter_names(self, feature_names: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(
            f"category[{self.categories[category]!r}]::{feature}"
            for category in self.estimated_category_indices
            for feature in feature_names
        )

    def _logits(
        self,
        matrix: NDArray[np.float64],
        estimates: NDArray[np.float64],
        available: NDArray[np.bool_],
    ) -> NDArray[np.float64]:
        coefficients = estimates.reshape(len(self.estimated_category_indices), matrix.shape[1])
        logits = np.zeros((len(matrix), len(self.categories)), dtype=np.float64)
        logits[:, self.estimated_category_indices] = matrix @ coefficients.T
        logits[~available] = -np.inf
        return logits

    def _hessian(
        self,
        matrix: NDArray[np.float64],
        available: NDArray[np.bool_],
        coefficients: NDArray[np.float64],
        penalty: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        logits = np.zeros((len(matrix), len(self.categories)), dtype=np.float64)
        logits[:, self.estimated_category_indices] = matrix @ coefficients.T
        logits[~available] = -np.inf
        probability = np.exp(logits - logsumexp(logits, axis=1)[:, None])
        selected = probability[:, self.estimated_category_indices]
        size = coefficients.size
        hessian = np.zeros((size, size), dtype=np.float64)
        for row, values in enumerate(selected):
            category_covariance = np.diag(values) - np.outer(values, values)
            feature_outer = np.outer(matrix[row], matrix[row])
            hessian += np.kron(category_covariance, feature_outer)
        if self.l2:
            hessian += np.diag(np.tile(self.l2 * penalty, len(self.estimated_category_indices)))
        return hessian

    def _validate_fit(self, fit: FitResult, feature_names: tuple[str, ...]) -> None:
        if (
            fit.model_name != self.model_name
            or fit.model_signature != self.signature
            or fit.parameter_names != self._parameter_names(feature_names)
        ):
            raise ValueError("fit result was produced by a different model specification")

    def _prediction_mode(self, mode: PredictionMode) -> PredictionMode:
        prediction_mode = PredictionMode(mode)
        if prediction_mode is not PredictionMode.FILTERED:
            raise UnsupportedPredictionMode(
                f"{self.model_name} supports only filtered prediction, "
                f"not {prediction_mode.value!r}"
            )
        return prediction_mode


def _uses_outcome_history(term: Any, outcome: str) -> bool:
    if isinstance(term, (HistoryTerm, HistoryKernelTerm)):
        return term.column == outcome
    if isinstance(term, InteractionTerm):
        return _uses_outcome_history(term.left, outcome) or _uses_outcome_history(
            term.right, outcome
        )
    return False


def _label_key(value: Any) -> tuple[Any, Any]:
    if isinstance(value, bool):
        return bool, value
    if isinstance(value, (int, float)):
        return "number", float(value)
    return type(value), value


def _scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _portable_category(value: Any) -> None:
    scalar = _scalar(value)
    if scalar is None or isinstance(scalar, (str, bool, int)):
        return
    if isinstance(scalar, float) and np.isfinite(scalar):
        return
    raise ValueError(f"modeled categories must be finite JSON scalars: {scalar!r}")
