"""Canonical binary-choice baselines with complete generative contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit

from behavio._internal.arrays import protected_array
from behavio.contracts.compose import PenalisedDesign
from behavio.design.matrix import DesignSpec
from behavio.models._kernels.bernoulli import (
    BernoulliLikelihood,
    fit_bernoulli,
    ordered_session_indices,
)
from behavio.models.base import (
    FitResult,
    ModelDataError,
    Prediction,
    PredictionMode,
    UnsupportedPredictionMode,
)
from behavio.models.glm import BernoulliHistoryGLM
from behavio.trials import REQUIRED_COLUMNS, Study


class _DelegatedGLMBaseline:
    """Shared identity-preserving adapter around the tested Bernoulli GLM engine.

    The block of contract members at the end is what makes these baselines *composable*
    rather than merely correct. A canonical baseline is exactly the model somebody wants to
    put a lapse on -- ``mix(Psychometric(...), UniformChoiceGuess())`` is the four-parameter
    psychophysical comparator a whole separate class used to exist for -- and every one of
    those members is the wrapped GLM's own, so a composed baseline runs the same arithmetic
    under the baseline's own name and signature.
    """

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
    def coefficient_names(self) -> tuple[str, ...]:
        return self._glm().coefficient_names

    @property
    def design_spec(self) -> DesignSpec:
        return self._glm().design_spec

    @property
    def declared_priors(self) -> tuple[str, ...]:
        return self._glm().declared_priors

    @property
    def likelihood(self) -> BernoulliLikelihood:
        return self._glm().likelihood

    @property
    def predictor_cells(self) -> tuple[str, ...]:
        return self._glm().predictor_cells

    @property
    def outcome_channels(self) -> tuple[str, ...]:
        return self._glm().outcome_channels

    def outcomes(self, study: Study) -> NDArray[np.float64]:
        return self._glm().outcomes(study)

    def design_matrix(self, study: Study) -> NDArray[np.float64]:
        return self._glm().design_matrix(study)

    def predictor_offsets(self, study: Study) -> NDArray[np.float64] | None:
        return self._glm().predictor_offsets(study)

    def penalty_matrix(self) -> NDArray[np.float64]:
        return self._glm().penalty_matrix()

    def coordinate_box(self, study: Study) -> NDArray[np.float64] | None:
        return self._glm().coordinate_box(study)

    def initial_points(self, study: Study) -> tuple[NDArray[np.float64], ...]:
        return self._glm().initial_points(study)

    def group_parameter_expansion(self, name: str) -> tuple[str, ...]:
        return self._glm().group_parameter_expansion(name)

    def fit_penalised(
        self,
        design: PenalisedDesign,
        *,
        model_name: str,
        model_signature: str,
    ) -> FitResult:
        return self._glm().fit_penalised(
            design, model_name=model_name, model_signature=model_signature
        )

    def simulate_rows(
        self,
        design: Study,
        coefficients: NDArray[np.float64],
        *,
        seed: int | np.random.Generator,
    ) -> Study:
        return self._glm().simulate_rows(design, coefficients, seed=seed)

    def group_penalty(
        self, columns: NDArray[np.intp], scales: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return self._glm().group_penalty(columns, scales)

    def draw_group_deviations(
        self,
        columns: NDArray[np.intp],
        scales: NDArray[np.float64],
        *,
        groups: int,
        generator: np.random.Generator,
    ) -> NDArray[np.float64]:
        return self._glm().draw_group_deviations(
            columns, scales, groups=groups, generator=generator
        )

    @property
    def scored_columns(self) -> tuple[str, ...]:
        return self._glm().scored_columns

    @property
    def required_task_columns(self) -> tuple[str, ...]:
        """No predictive context by default: these baselines read only their own history.

        ``BehaviourEstimator`` now requires this declaration, and the empty tuple is a
        real answer rather than a missing one -- ``BiasOnly`` is a stationary intercept and
        ``Perseveration`` reads only the outcome column it also scores. ``Psychometric``
        overrides it with its stimulus.
        """

        return ()

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
            predictors=(self.stimulus,), outcome=self.outcome, choice_lags=0, l2=self.l2
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
