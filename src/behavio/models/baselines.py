"""Canonical binary-choice baselines with complete generative contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import expit, logit

from behavio._internal.arrays import protected_array
from behavio.contracts import natural_quantities
from behavio.models._kernels.bernoulli import fit_bernoulli, ordered_session_indices
from behavio.models._kernels.curvature import finite_difference_hessian
from behavio.models.base import (
    FitDiagnostics,
    FitResult,
    ModelDataError,
    Prediction,
    PredictionMode,
    UnsupportedPredictionMode,
)
from behavio.models.glm import BernoulliHistoryGLM
from behavio.study import REQUIRED_COLUMNS, Study


class _DelegatedGLMBaseline:
    """Shared identity-preserving adapter around the tested Bernoulli GLM engine."""

    @property
    def model_name(self) -> str:
        raise NotImplementedError

    @property
    def signature(self) -> str:
        raise NotImplementedError

    def _glm(self) -> BernoulliHistoryGLM:
        raise NotImplementedError

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return self._glm().parameter_names

    @property
    def scored_columns(self) -> tuple[str, ...]:
        return self._glm().scored_columns

    @property
    def supported_prediction_modes(self) -> tuple[PredictionMode, ...]:
        return self._glm().supported_prediction_modes

    def simulate(
        self,
        design: Study,
        parameters: Mapping[str, float],
        *,
        seed: int | np.random.Generator,
    ) -> Study:
        return self._glm().simulate(design, parameters, seed=seed)

    def fit(self, study: Study) -> FitResult:
        raw = self._glm().fit(study)
        return replace(raw, model_name=self.model_name, model_signature=self.signature)

    def predict(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> Prediction:
        delegate = self._delegate_fit(fit)
        return self._glm().predict(study, delegate, mode=mode)

    def pointwise_log_prob(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> NDArray[np.float64]:
        delegate = self._delegate_fit(fit)
        return self._glm().pointwise_log_prob(study, delegate, mode=mode)

    def _delegate_fit(self, fit: FitResult) -> FitResult:
        if (
            fit.model_name != self.model_name
            or fit.model_signature != self.signature
            or fit.parameter_names != self.parameter_names
        ):
            raise ValueError("fit result was produced by a different model specification")
        glm = self._glm()
        return replace(fit, model_name=glm.model_name, model_signature=glm.signature)


@dataclass(frozen=True, slots=True)
class BiasOnly(_DelegatedGLMBaseline):
    """Stationary Bernoulli intercept baseline."""

    outcome: str = "choice"
    l2: float = 0.0

    def __post_init__(self) -> None:
        self._glm()

    @property
    def model_name(self) -> str:
        return "bias-only"

    @property
    def signature(self) -> str:
        return f"{self.model_name}[outcome={self.outcome};l2={self.l2}]"

    def _glm(self) -> BernoulliHistoryGLM:
        return BernoulliHistoryGLM(outcome=self.outcome, choice_lags=0, l2=self.l2)


@dataclass(frozen=True, slots=True)
class Psychometric(_DelegatedGLMBaseline):
    """Logistic psychometric curve with intercept and one fixed stimulus column."""

    stimulus: str = "stimulus"
    outcome: str = "choice"
    l2: float = 0.0

    def __post_init__(self) -> None:
        self._glm()

    @property
    def model_name(self) -> str:
        return "psychometric"

    @property
    def signature(self) -> str:
        return f"{self.model_name}[outcome={self.outcome};stimulus={self.stimulus};l2={self.l2}]"

    def _glm(self) -> BernoulliHistoryGLM:
        return BernoulliHistoryGLM(
            covariates=(self.stimulus,), outcome=self.outcome, choice_lags=0, l2=self.l2
        )

    @property
    def required_task_columns(self) -> tuple[str, ...]:
        return (self.stimulus,)


@dataclass(frozen=True, slots=True)
class Perseveration(_DelegatedGLMBaseline):
    """Intercept plus one effect-coded, session-reset previous-choice term."""

    outcome: str = "choice"
    l2: float = 0.0

    def __post_init__(self) -> None:
        self._glm()

    @property
    def model_name(self) -> str:
        return "perseveration"

    @property
    def signature(self) -> str:
        return f"{self.model_name}[outcome={self.outcome};session_reset=True;l2={self.l2}]"

    def _glm(self) -> BernoulliHistoryGLM:
        return BernoulliHistoryGLM(outcome=self.outcome, choice_lags=1, l2=self.l2)


@dataclass(frozen=True, slots=True)
class WinStayLoseShift:
    """Outcome-conditioned choice history with explicit session resets."""

    outcome: str = "choice"
    reward: str = "reward"
    reward_probability_columns: tuple[str, str] = (
        "reward_probability_0",
        "reward_probability_1",
    )
    l2: float = 0.0
    max_iterations: int = 1_000
    tolerance: float = 1e-9
    coefficient_warning_threshold: float = 20.0

    def __post_init__(self) -> None:
        for value, label in ((self.outcome, "outcome"), (self.reward, "reward")):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{label} must be a non-empty column name")
            if value in REQUIRED_COLUMNS:
                raise ValueError(f"{label} cannot replace a required Study column")
        if self.outcome == self.reward:
            raise ValueError("outcome and reward columns must be distinct")
        probability_columns = tuple(self.reward_probability_columns)
        if len(probability_columns) != 2 or len(set(probability_columns)) != 2:
            raise ValueError("reward_probability_columns must contain two distinct names")
        if any(not isinstance(column, str) or not column for column in probability_columns):
            raise ValueError("reward probability columns must be non-empty strings")
        if set(probability_columns) & {self.outcome, self.reward, *REQUIRED_COLUMNS}:
            raise ValueError("reward probability columns must not replace observed columns")
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
        object.__setattr__(self, "reward_probability_columns", probability_columns)

    @property
    def model_name(self) -> str:
        return "win-stay-lose-shift"

    @property
    def signature(self) -> str:
        environment = ",".join(self.reward_probability_columns)
        return (
            f"{self.model_name}[outcome={self.outcome};reward={self.reward};"
            f"environment={environment};session_reset=True;l2={self.l2}]"
        )

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return ("intercept", "win_stay", "lose_shift")

    @property
    def scored_columns(self) -> tuple[str, ...]:
        return (self.outcome,)

    @property
    def required_task_columns(self) -> tuple[str, ...]:
        return (self.reward,)

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
        coefficients = self._parameter_vector(parameters)
        reward_probability = self._reward_probabilities(design)
        generator = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        choices = np.zeros(len(design), dtype=np.int8)
        rewards = np.zeros(len(design), dtype=np.int8)
        for indices in ordered_session_indices(design):
            previous_choice: int | None = None
            previous_reward: int | None = None
            for index in indices:
                win_stay = 0.0
                lose_shift = 0.0
                if previous_choice is not None and previous_reward is not None:
                    effect_choice = 2.0 * previous_choice - 1.0
                    if previous_reward:
                        win_stay = effect_choice
                    else:
                        lose_shift = -effect_choice
                linear = coefficients[0] + coefficients[1] * win_stay + coefficients[2] * lose_shift
                choice = int(generator.binomial(1, expit(linear)))
                reward = int(generator.binomial(1, reward_probability[index, choice]))
                choices[index] = choice
                rewards[index] = reward
                previous_choice = choice
                previous_reward = reward
        columns = {name: design[name] for name in design.columns}
        columns[self.outcome] = choices
        columns[self.reward] = rewards
        return Study(columns)

    def fit(self, study: Study) -> FitResult:
        outcomes = self._binary_column(study, self.outcome, "outcome")
        rewards = self._binary_column(study, self.reward, "reward")
        matrix = self._design_matrix(study, outcomes, rewards)
        penalty = np.diag(np.asarray((0.0, self.l2, self.l2)))
        return fit_bernoulli(
            model_name=self.model_name,
            model_signature=self.signature,
            parameter_names=self.parameter_names,
            design_matrix=matrix,
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
        self._prediction_mode(mode)
        self._validate_fit(fit)
        outcomes = self._binary_column(study, self.outcome, "outcome")
        rewards = self._binary_column(study, self.reward, "reward")
        linear = self._design_matrix(study, outcomes, rewards) @ fit.estimates
        return Prediction(expit(linear), linear, PredictionMode.FILTERED)

    def pointwise_log_prob(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> NDArray[np.float64]:
        outcomes = self._binary_column(study, self.outcome, "outcome")
        linear = self.predict(study, fit, mode=mode).linear_predictor
        scores = outcomes * -np.logaddexp(0.0, -linear)
        scores += (1.0 - outcomes) * -np.logaddexp(0.0, linear)
        return protected_array(scores, dtype=np.float64)

    def _design_matrix(
        self,
        study: Study,
        outcomes: NDArray[np.float64],
        rewards: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        matrix = np.zeros((len(study), 3), dtype=np.float64)
        matrix[:, 0] = 1.0
        for indices in ordered_session_indices(study):
            for position, index in enumerate(indices[1:], start=1):
                previous = indices[position - 1]
                effect_choice = 2.0 * outcomes[previous] - 1.0
                if rewards[previous] == 1.0:
                    matrix[index, 1] = effect_choice
                else:
                    matrix[index, 2] = -effect_choice
        return matrix

    def _reward_probabilities(self, study: Study) -> NDArray[np.float64]:
        missing = [
            column for column in self.reward_probability_columns if column not in study.columns
        ]
        if missing:
            raise ModelDataError(f"study is missing reward probability columns: {missing}")
        try:
            values = np.column_stack(
                [
                    np.asarray(study[column], dtype=np.float64)
                    for column in self.reward_probability_columns
                ]
            )
        except (TypeError, ValueError):
            raise ModelDataError("reward probabilities must be finite values in [0, 1]") from None
        if not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
            raise ModelDataError("reward probabilities must be finite values in [0, 1]")
        return values

    def _binary_column(self, study: Study, column: str, label: str) -> NDArray[np.float64]:
        if column not in study.columns:
            raise ModelDataError(f"study is missing {label} column {column!r}")
        try:
            values = np.asarray(study[column], dtype=np.float64)
        except (TypeError, ValueError):
            raise ModelDataError(f"{label} must contain only zero and one") from None
        if values.ndim != 1 or not np.all((values == 0.0) | (values == 1.0)):
            raise ModelDataError(f"{label} must contain only zero and one")
        return values

    def _parameter_vector(self, parameters: Mapping[str, float]) -> NDArray[np.float64]:
        if set(parameters) != set(self.parameter_names):
            raise ValueError("parameters must match the model exactly")
        try:
            values = np.asarray(
                [parameters[name] for name in self.parameter_names], dtype=np.float64
            )
        except (TypeError, ValueError):
            raise ValueError("parameters must contain finite numeric values") from None
        if not np.all(np.isfinite(values)):
            raise ValueError("parameters must contain finite numeric values")
        return values

    def _validate_fit(self, fit: FitResult) -> None:
        if fit.model_signature != self.signature or fit.parameter_names != self.parameter_names:
            raise ValueError("fit result was produced by a different model specification")

    def _prediction_mode(self, mode: PredictionMode) -> PredictionMode:
        prediction_mode = PredictionMode(mode)
        if prediction_mode is not PredictionMode.FILTERED:
            raise UnsupportedPredictionMode(
                f"{self.model_name} supports only filtered prediction, "
                f"not {prediction_mode.value!r}"
            )
        return prediction_mode


@dataclass(frozen=True, slots=True)
class LapsePsychometricParameters:
    """Natural-scale parameters of a lapse-mixture psychometric curve."""

    intercept: float
    slope: float
    lapse_rate: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.intercept) or not np.isfinite(self.slope):
            raise ValueError("psychometric intercept and slope must be finite")
        if not np.isfinite(self.lapse_rate) or not 0 < self.lapse_rate < 1:
            raise ValueError("lapse_rate must lie strictly between zero and one")


@dataclass(frozen=True, slots=True)
class LapsePsychometricFitResult(FitResult):
    """Lapse-psychometric fit retaining every deterministic restart.

    The four restart fields are the :class:`~behavio.contracts.MultistartFit` contract.
    ``lapse_rate`` is the natural coordinate of ``lapse_logit`` and is read from
    :attr:`~behavio.contracts.FitResult.derived`, where it carries a delta-method standard
    error rather than being a bare number.
    """

    restart_objectives: NDArray[np.float64]
    restart_converged: NDArray[np.bool_]
    restart_messages: tuple[str, ...]
    selected_restart: int

    def __post_init__(self) -> None:
        FitResult.__post_init__(self)
        objectives = protected_array(self.restart_objectives, dtype=np.float64)
        converged = protected_array(self.restart_converged, dtype=np.bool_)
        messages = tuple(self.restart_messages)
        if objectives.ndim != 1 or converged.shape != objectives.shape:
            raise ValueError("restart diagnostics must have one value per restart")
        if len(messages) != len(objectives) or np.any(np.isnan(objectives)):
            raise ValueError("restart messages and non-NaN objectives must align")
        if not 0 <= self.selected_restart < len(objectives):
            raise ValueError("selected_restart must identify one restart")
        if "lapse_rate" not in self.derived_values:
            raise ValueError("a lapse-psychometric fit must declare a derived lapse_rate")
        if not np.isfinite(self.lapse_rate) or not 0 < self.lapse_rate < 1:
            raise ValueError("lapse_rate must lie strictly between zero and one")
        object.__setattr__(self, "restart_objectives", objectives)
        object.__setattr__(self, "restart_converged", converged)
        object.__setattr__(self, "restart_messages", messages)

    @property
    def lapse_rate(self) -> float:
        """Probability of a stimulus-independent random response."""

        return self.derived_value("lapse_rate")


@dataclass(frozen=True, slots=True)
class LapsePsychometric:
    """Logistic psychometric curve mixed with stimulus-independent random responses."""

    stimulus: str = "stimulus"
    outcome: str = "choice"
    maximum_lapse: float = 0.2
    l2: float = 0.0
    n_restarts: int = 3
    max_iterations: int = 1_000
    tolerance: float = 1e-9
    coefficient_warning_threshold: float = 20.0

    def __post_init__(self) -> None:
        for value, label in ((self.stimulus, "stimulus"), (self.outcome, "outcome")):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{label} must be a non-empty column name")
            if value in REQUIRED_COLUMNS:
                raise ValueError(f"{label} cannot replace a required Study column")
        if self.stimulus == self.outcome:
            raise ValueError("stimulus and outcome columns must be distinct")
        if not np.isfinite(self.maximum_lapse) or not 0 < self.maximum_lapse < 1:
            raise ValueError("maximum_lapse must lie strictly between zero and one")
        if not np.isfinite(self.l2) or self.l2 < 0:
            raise ValueError("l2 must be finite and non-negative")
        for value, label in (
            (self.n_restarts, "n_restarts"),
            (self.max_iterations, "max_iterations"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{label} must be a positive integer")
        if not np.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("tolerance must be finite and positive")
        if (
            not np.isfinite(self.coefficient_warning_threshold)
            or self.coefficient_warning_threshold <= 0
        ):
            raise ValueError("coefficient_warning_threshold must be finite and positive")

    @property
    def model_name(self) -> str:
        return "lapse-psychometric"

    @property
    def signature(self) -> str:
        return (
            f"{self.model_name}[outcome={self.outcome};stimulus={self.stimulus};"
            f"maximum_lapse={self.maximum_lapse};l2={self.l2}]"
        )

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return ("intercept", self.stimulus, "lapse_logit")

    @property
    def scored_columns(self) -> tuple[str, ...]:
        return (self.outcome,)

    @property
    def required_task_columns(self) -> tuple[str, ...]:
        return (self.stimulus,)

    @property
    def supported_prediction_modes(self) -> tuple[PredictionMode, ...]:
        return (PredictionMode.FILTERED,)

    @property
    def natural_names(self) -> tuple[str, ...]:
        """The reported coordinate: a slope in stimulus units and a probability."""

        return ("intercept", "slope", "lapse_rate")

    def to_natural(
        self, estimates: Sequence[float] | NDArray[np.floating[Any]]
    ) -> Mapping[str, float]:
        """Decode the optimizer coordinate into an interpretable lapse rate."""

        components = self.parameter_components(
            dict(zip(self.parameter_names, self._natural_input(estimates).tolist(), strict=True))
        )
        return MappingProxyType(
            {
                "intercept": components.intercept,
                "slope": components.slope,
                "lapse_rate": components.lapse_rate,
            }
        )

    def from_natural(self, natural: Mapping[str, float]) -> Mapping[str, float]:
        """Encode a complete natural mapping onto the optimizer coordinate."""

        if not isinstance(natural, Mapping) or set(natural) != set(self.natural_names):
            raise ValueError("natural parameters must match the model exactly")
        return self.parameters_from_components(
            intercept=natural["intercept"],
            slope=natural["slope"],
            lapse_rate=natural["lapse_rate"],
        )

    def natural_jacobian(
        self, estimates: Sequence[float] | NDArray[np.floating[Any]]
    ) -> NDArray[np.float64]:
        """Return the derivative of the natural parameters at one optimizer coordinate."""

        coordinate = self._natural_input(estimates)
        relative = float(expit(coordinate[2]))
        jacobian = np.zeros((3, 3), dtype=np.float64)
        jacobian[0, 0] = 1.0
        jacobian[1, 1] = 1.0
        jacobian[2, 2] = self.maximum_lapse * relative * (1.0 - relative)
        return jacobian

    def _natural_input(
        self, estimates: Sequence[float] | NDArray[np.floating[Any]]
    ) -> NDArray[np.float64]:
        try:
            vector = np.asarray(estimates, dtype=np.float64)
        except (TypeError, ValueError):
            raise ValueError("estimates must contain finite numeric values") from None
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise ValueError("estimates must contain three finite values")
        return vector

    def parameters_from_components(
        self,
        *,
        intercept: float,
        slope: float,
        lapse_rate: float,
    ) -> Mapping[str, float]:
        """Validate and encode a natural lapse rate on the optimizer coordinate.

        Keyword-argument façade over :meth:`from_natural`, which shares this encoding.
        """

        components = LapsePsychometricParameters(intercept, slope, lapse_rate)
        if components.lapse_rate >= self.maximum_lapse:
            raise ValueError("lapse_rate must be smaller than maximum_lapse")
        relative = components.lapse_rate / self.maximum_lapse
        return MappingProxyType(
            {
                "intercept": components.intercept,
                self.stimulus: components.slope,
                "lapse_logit": float(logit(relative)),
            }
        )

    def parameter_components(
        self, parameters: Mapping[str, float] | FitResult
    ) -> LapsePsychometricParameters:
        """Decode optimizer coordinates into an interpretable lapse rate."""

        if isinstance(parameters, FitResult):
            self._validate_fit(parameters)
            values = parameters.estimates
        else:
            if set(parameters) != set(self.parameter_names):
                raise ValueError("parameters must match the model exactly")
            try:
                values = np.asarray(
                    [parameters[name] for name in self.parameter_names], dtype=np.float64
                )
            except (TypeError, ValueError):
                raise ValueError("parameters must contain finite numeric values") from None
            if not np.all(np.isfinite(values)):
                raise ValueError("parameters must contain finite numeric values")
        return LapsePsychometricParameters(
            intercept=float(values[0]),
            slope=float(values[1]),
            lapse_rate=float(self.maximum_lapse * expit(values[2])),
        )

    def simulate(
        self,
        design: Study,
        parameters: Mapping[str, float],
        *,
        seed: int | np.random.Generator,
    ) -> Study:
        components = self.parameter_components(parameters)
        stimulus = self._stimulus(design)
        probability = self._probability(stimulus, components)
        generator = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        columns = {name: design[name] for name in design.columns}
        columns[self.outcome] = generator.binomial(1, probability).astype(np.int8)
        return Study(columns)

    def fit(self, study: Study) -> LapsePsychometricFitResult:
        stimulus = self._stimulus(study)
        outcomes = self._outcomes(study)
        starts = np.zeros((self.n_restarts, 3), dtype=np.float64)
        starts[:, 2] = np.linspace(-5.0, 1.0, self.n_restarts)
        results = [
            minimize(
                lambda values: self._objective(values, stimulus, outcomes),
                start,
                method="L-BFGS-B",
                jac=True,
                bounds=((None, None), (None, None), (-12.0, 12.0)),
                options={
                    "maxiter": self.max_iterations,
                    "ftol": self.tolerance,
                    "gtol": self.tolerance,
                },
            )
            for start in starts
        ]
        objectives = np.asarray([result.fun for result in results], dtype=np.float64)
        selected = int(np.argmin(objectives))
        result = results[selected]
        estimates = np.asarray(result.x, dtype=np.float64)
        _, gradient = self._objective(estimates, stimulus, outcomes)
        hessian = finite_difference_hessian(
            lambda values: self._objective(values, stimulus, outcomes)[1], estimates
        )
        condition = float(np.linalg.cond(hessian))
        covariance = np.linalg.pinv(hessian, hermitian=True)
        standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        diagnostics = FitDiagnostics(
            converged=bool(result.success),
            optimizer="L-BFGS-B multistart",
            status=int(result.status),
            message=str(result.message),
            n_iterations=int(result.nit),
            objective=float(result.fun),
            gradient_norm=float(np.linalg.norm(gradient)),
            hessian_condition=condition,
            boundary_estimate=bool(
                np.any(np.abs(estimates[:2]) >= self.coefficient_warning_threshold)
                or abs(estimates[2]) >= 11.9
            ),
        )
        return LapsePsychometricFitResult(
            model_name=self.model_name,
            model_signature=self.signature,
            parameter_names=self.parameter_names,
            estimates=estimates,
            standard_errors=standard_errors,
            covariance=covariance,
            n_observations=len(study),
            diagnostics=diagnostics,
            derived=natural_quantities(self, estimates, covariance),
            restart_objectives=objectives,
            restart_converged=np.asarray([item.success for item in results]),
            restart_messages=tuple(str(item.message) for item in results),
            selected_restart=selected,
        )

    def predict(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> Prediction:
        self._prediction_mode(mode)
        components = self.parameter_components(fit)
        stimulus = self._stimulus(study)
        linear_predictor = components.intercept + components.slope * stimulus
        return Prediction(
            probability=self._probability(stimulus, components),
            linear_predictor=linear_predictor,
            mode=PredictionMode.FILTERED,
        )

    def pointwise_log_prob(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> NDArray[np.float64]:
        outcomes = self._outcomes(study)
        probability = self.predict(study, fit, mode=mode).probability
        scores = outcomes * np.log(probability) + (1.0 - outcomes) * np.log1p(-probability)
        return protected_array(scores, dtype=np.float64)

    def _objective(
        self,
        values: NDArray[np.float64],
        stimulus: NDArray[np.float64],
        outcomes: NDArray[np.float64],
    ) -> tuple[float, NDArray[np.float64]]:
        intercept, slope, lapse_logit = values
        logistic = expit(intercept + slope * stimulus)
        relative_lapse = expit(lapse_logit)
        lapse = self.maximum_lapse * relative_lapse
        probability = lapse * 0.5 + (1.0 - lapse) * logistic
        loss = -float(outcomes @ np.log(probability) + (1.0 - outcomes) @ np.log1p(-probability))
        loss += 0.5 * self.l2 * float(slope**2)
        score = (probability - outcomes) / (probability * (1.0 - probability))
        probability_eta = (1.0 - lapse) * logistic * (1.0 - logistic)
        lapse_derivative = self.maximum_lapse * relative_lapse * (1.0 - relative_lapse)
        probability_lapse_logit = lapse_derivative * (0.5 - logistic)
        gradient = np.asarray(
            [
                np.sum(score * probability_eta),
                np.sum(score * probability_eta * stimulus) + self.l2 * slope,
                np.sum(score * probability_lapse_logit),
            ],
            dtype=np.float64,
        )
        return loss, gradient

    def _probability(
        self,
        stimulus: NDArray[np.float64],
        components: LapsePsychometricParameters,
    ) -> NDArray[np.float64]:
        logistic = expit(components.intercept + components.slope * stimulus)
        return components.lapse_rate * 0.5 + (1.0 - components.lapse_rate) * logistic

    def _stimulus(self, study: Study) -> NDArray[np.float64]:
        if self.stimulus not in study.columns:
            raise ModelDataError(f"study is missing stimulus column {self.stimulus!r}")
        try:
            values = np.asarray(study[self.stimulus], dtype=np.float64)
        except (TypeError, ValueError):
            raise ModelDataError("stimulus values must be finite numbers") from None
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise ModelDataError("stimulus values must be finite numbers")
        return values

    def _outcomes(self, study: Study) -> NDArray[np.float64]:
        if self.outcome not in study.columns:
            raise ModelDataError(f"study is missing outcome column {self.outcome!r}")
        try:
            outcomes = np.asarray(study[self.outcome], dtype=np.float64)
        except (TypeError, ValueError):
            raise ModelDataError("outcomes must contain only zero and one") from None
        if outcomes.ndim != 1 or not np.all((outcomes == 0.0) | (outcomes == 1.0)):
            raise ModelDataError("outcomes must contain only zero and one")
        return outcomes

    def _validate_fit(self, fit: FitResult) -> None:
        if (
            fit.model_name != self.model_name
            or fit.model_signature != self.signature
            or fit.parameter_names != self.parameter_names
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
