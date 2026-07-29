"""Compact session-reset binary Q-learning reference model."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit

from behavio._internal.arrays import protected_array
from behavio.inference import OptimizationProblem, OptimizationRun, ScipyMultistart
from behavio.models.base import (
    FitDiagnostics,
    FitResult,
    ModelDataError,
    Prediction,
    PredictionMode,
    UnsupportedPredictionMode,
)
from behavio.models.glm import _ordered_session_indices
from behavio.parameters import ParameterSpace, ParameterSpec, ParameterTransform
from behavio.study import REQUIRED_COLUMNS, Study

_Q_LEARNING_PARAMETER_SPACE = ParameterSpace(
    (
        ParameterSpec(
            name="learning_rate",
            optimizer_name="learning_rate_logit",
            transform=ParameterTransform.BOUNDED_LOGIT,
            bounds=(0.0, 1.0),
            plausible_bounds=(0.05, 0.95),
            optimizer_bounds=(-12.0, 12.0),
            description="Fraction of the reward prediction error applied per update.",
        ),
        ParameterSpec(
            name="inverse_temperature",
            optimizer_name="inverse_temperature_log",
            transform=ParameterTransform.LOG,
            bounds=(0.0, None),
            plausible_bounds=(0.1, 10.0),
            optimizer_bounds=(-5.0, 5.0),
            description="Choice sensitivity to the learned action-value difference.",
        ),
        ParameterSpec(
            name="choice_bias",
            plausible_bounds=(-5.0, 5.0),
            optimizer_bounds=(-30.0, 30.0),
            description="Value-independent log-odds bias towards action one.",
        ),
        ParameterSpec(
            name="perseveration",
            plausible_bounds=(-5.0, 5.0),
            optimizer_bounds=(-30.0, 30.0),
            description="Effect-coded influence of the preceding choice.",
        ),
    )
)


@dataclass(frozen=True, slots=True)
class QLearningParameters:
    """Natural-scale parameters of the binary Q-learning agent."""

    learning_rate: float
    inverse_temperature: float
    choice_bias: float
    perseveration: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.learning_rate) or not 0 < self.learning_rate < 1:
            raise ValueError("learning_rate must lie strictly between zero and one")
        if not np.isfinite(self.inverse_temperature) or self.inverse_temperature <= 0:
            raise ValueError("inverse_temperature must be finite and positive")
        if not np.isfinite(self.choice_bias):
            raise ValueError("choice_bias must be finite")
        if not np.isfinite(self.perseveration):
            raise ValueError("perseveration must be finite")


@dataclass(frozen=True, slots=True)
class QLearningFitResult(FitResult):
    """A fitted Q-learning agent with all restart outcomes retained."""

    restart_objectives: NDArray[np.float64]
    restart_converged: NDArray[np.bool_]
    restart_messages: tuple[str, ...]
    selected_restart: int
    learning_rate: float
    inverse_temperature: float
    optimization_run: OptimizationRun

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
        if not isinstance(self.optimization_run, OptimizationRun):
            raise ValueError("optimization_run must be an OptimizationRun")
        attempts = self.optimization_run.attempts
        if (
            len(attempts) != len(objectives)
            or self.optimization_run.selected_index != self.selected_restart
            or not np.allclose(
                [attempt.objective for attempt in attempts], objectives, equal_nan=True
            )
            or not np.array_equal([attempt.converged for attempt in attempts], converged)
            or tuple(attempt.message for attempt in attempts) != messages
        ):
            raise ValueError("optimization_run must match retained restart summaries")
        decoded = _decode_parameters(self.estimates)
        if not np.isclose(self.learning_rate, decoded.learning_rate) or not np.isclose(
            self.inverse_temperature, decoded.inverse_temperature
        ):
            raise ValueError("natural-scale fit parameters must match estimates")
        object.__setattr__(self, "restart_objectives", objectives)
        object.__setattr__(self, "restart_converged", converged)
        object.__setattr__(self, "restart_messages", messages)


@dataclass(frozen=True, slots=True)
class ValueTrajectory:
    """Pre-choice and post-update values in the Study's source row order."""

    pre_choice: NDArray[np.float64]
    post_update: NDArray[np.float64]
    prediction_error: NDArray[np.float64]
    linear_predictor: NDArray[np.float64]

    def __post_init__(self) -> None:
        pre_choice = protected_array(self.pre_choice, dtype=np.float64)
        post_update = protected_array(self.post_update, dtype=np.float64)
        prediction_error = protected_array(self.prediction_error, dtype=np.float64)
        linear_predictor = protected_array(self.linear_predictor, dtype=np.float64)
        if pre_choice.ndim != 2 or pre_choice.shape[1] != 2:
            raise ValueError("pre_choice must have one row and two action values per trial")
        if post_update.shape != pre_choice.shape:
            raise ValueError("post_update must match pre_choice")
        if prediction_error.shape != (len(pre_choice),):
            raise ValueError("prediction_error must contain one value per trial")
        if linear_predictor.shape != (len(pre_choice),):
            raise ValueError("linear_predictor must contain one value per trial")
        if not all(
            np.all(np.isfinite(values))
            for values in (pre_choice, post_update, prediction_error, linear_predictor)
        ):
            raise ValueError("value trajectories must contain only finite values")
        object.__setattr__(self, "pre_choice", pre_choice)
        object.__setattr__(self, "post_update", post_update)
        object.__setattr__(self, "prediction_error", prediction_error)
        object.__setattr__(self, "linear_predictor", linear_predictor)


@dataclass(frozen=True, slots=True)
class BinaryQLearning:
    """A binary session-reset Q-learning agent with filtered choice prediction.

    Values initialize to ``initial_value`` at every subject/session boundary. The chosen
    value is updated after its choice and reward are observed; unchosen values remain fixed.
    Simulation samples reward from the chosen action's explicit environment column.
    """

    outcome: str = "choice"
    reward: str = "reward"
    reward_probability_columns: tuple[str, str] = (
        "reward_probability_0",
        "reward_probability_1",
    )
    initial_value: float = 0.5
    n_restarts: int = 5
    random_seed: int = 0
    max_iterations: int = 1_000
    tolerance: float = 1e-9
    coefficient_warning_threshold: float = 20.0
    learning_rate_warning_threshold: float = 1e-4
    inverse_temperature_warning_threshold: float = 50.0

    def __post_init__(self) -> None:
        for name, value in (("outcome", self.outcome), ("reward", self.reward)):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty column name")
            if value in REQUIRED_COLUMNS:
                raise ValueError(f"{name} cannot replace a required Study column")
        if self.outcome == self.reward:
            raise ValueError("outcome and reward columns must be distinct")
        probability_columns = tuple(self.reward_probability_columns)
        if len(probability_columns) != 2 or len(set(probability_columns)) != 2:
            raise ValueError("reward_probability_columns must contain two distinct names")
        if any(not isinstance(name, str) or not name for name in probability_columns):
            raise ValueError("reward probability column names must be non-empty strings")
        if set(probability_columns) & {self.outcome, self.reward, *REQUIRED_COLUMNS}:
            raise ValueError("reward probability columns must not replace observed Study columns")
        if not np.isfinite(self.initial_value) or not 0 <= self.initial_value <= 1:
            raise ValueError("initial_value must be finite and lie between zero and one")
        _require_positive_integer(self.n_restarts, "n_restarts")
        _require_nonnegative_integer(self.random_seed, "random_seed")
        _require_positive_integer(self.max_iterations, "max_iterations")
        if not np.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("tolerance must be finite and positive")
        if (
            not np.isfinite(self.coefficient_warning_threshold)
            or self.coefficient_warning_threshold <= 0
        ):
            raise ValueError("coefficient_warning_threshold must be finite and positive")
        if not 0 < self.learning_rate_warning_threshold < 0.5:
            raise ValueError(
                "learning_rate_warning_threshold must lie strictly between zero and 0.5"
            )
        if (
            not np.isfinite(self.inverse_temperature_warning_threshold)
            or self.inverse_temperature_warning_threshold <= 1
        ):
            raise ValueError("inverse_temperature_warning_threshold must be finite and above one")
        object.__setattr__(self, "reward_probability_columns", probability_columns)

    @property
    def model_name(self) -> str:
        return "binary-q-learning"

    @property
    def signature(self) -> str:
        environment = ",".join(self.reward_probability_columns)
        return (
            f"{self.model_name}[outcome={self.outcome};reward={self.reward};"
            f"environment={environment};initial_value={self.initial_value};session_reset=True]"
        )

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Legacy optimizer-coordinate names retained for fit compatibility."""

        return self.parameter_space.optimizer_names

    @property
    def parameter_space(self) -> ParameterSpace:
        """Stable natural and optimizer coordinates shared by inference backends."""

        return _Q_LEARNING_PARAMETER_SPACE

    @property
    def scored_columns(self) -> tuple[str, ...]:
        return (self.outcome,)

    @property
    def required_task_columns(self) -> tuple[str, ...]:
        return (self.reward,)

    @property
    def supported_prediction_modes(self) -> tuple[PredictionMode, ...]:
        return (PredictionMode.FILTERED,)

    def parameters_from_components(
        self,
        *,
        learning_rate: float,
        inverse_temperature: float,
        choice_bias: float = 0.0,
        perseveration: float = 0.0,
    ) -> Mapping[str, float]:
        """Validate and encode natural-scale agent parameters."""

        parameters = QLearningParameters(
            learning_rate=learning_rate,
            inverse_temperature=inverse_temperature,
            choice_bias=choice_bias,
            perseveration=perseveration,
        )
        return self.parameter_space.encode_mapping(
            {
                "learning_rate": parameters.learning_rate,
                "inverse_temperature": parameters.inverse_temperature,
                "choice_bias": parameters.choice_bias,
                "perseveration": parameters.perseveration,
            }
        )

    def parameter_components(
        self,
        parameters: Mapping[str, float] | FitResult,
    ) -> QLearningParameters:
        """Decode optimizer coordinates into natural-scale agent parameters."""

        if isinstance(parameters, FitResult):
            self._validate_fit(parameters)
            vector = parameters.estimates
        else:
            expected = set(self.parameter_names)
            observed = set(parameters)
            if observed != expected:
                raise ValueError(
                    "parameters must match the model exactly; "
                    f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
                )
            try:
                vector = np.asarray(
                    [parameters[name] for name in self.parameter_names], dtype=np.float64
                )
            except (TypeError, ValueError):
                raise ValueError("parameters must contain finite numeric values") from None
            if not np.all(np.isfinite(vector)):
                raise ValueError("parameters must contain finite numeric values")
        decoded = self.parameter_space.decode(vector)
        return QLearningParameters(**decoded)

    def simulate(
        self,
        design: Study,
        parameters: Mapping[str, float],
        *,
        seed: int | np.random.Generator,
    ) -> Study:
        """Generate choices and action-contingent binary rewards recursively."""

        components = self.parameter_components(parameters)
        environment = self._reward_probabilities(design)
        generator = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        choices = np.zeros(len(design), dtype=np.int8)
        rewards = np.zeros(len(design), dtype=np.int8)

        for session_indices in _ordered_session_indices(design):
            values = np.full(2, self.initial_value, dtype=np.float64)
            previous_choice = 0.0
            for index in session_indices:
                linear = (
                    components.inverse_temperature * (values[1] - values[0])
                    + components.choice_bias
                    + components.perseveration * previous_choice
                )
                choice = int(generator.binomial(1, expit(linear)))
                reward = int(generator.binomial(1, environment[index, choice]))
                choices[index] = choice
                rewards[index] = reward
                values[choice] += components.learning_rate * (reward - values[choice])
                previous_choice = 2.0 * choice - 1.0

        columns = {name: design[name] for name in design.columns}
        columns[self.outcome] = choices
        columns[self.reward] = rewards
        return Study(columns)

    def fit(self, study: Study) -> QLearningFitResult:
        """Fit choice likelihood with deterministic multi-start L-BFGS-B."""

        choices = self._choices(study)
        rewards = self._rewards(study)
        sessions = _ordered_session_indices(study)

        def objective(vector: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
            return self._objective_gradient(vector, choices, rewards, sessions)

        problem = OptimizationProblem(
            parameter_space=self.parameter_space,
            objective=objective,
            starts=self._initial_points(),
            has_gradient=True,
            objective_name="binary_q_learning_negative_log_likelihood",
        )
        run = ScipyMultistart(
            max_iterations=self.max_iterations,
            function_tolerance=self.tolerance,
            gradient_tolerance=self.tolerance,
        ).run(problem)
        restart_objectives = np.asarray(
            [attempt.objective for attempt in run.attempts], dtype=np.float64
        )
        chosen = run.selected
        if chosen is None:
            messages = "; ".join(attempt.message for attempt in run.attempts)
            raise ModelDataError(
                f"all Q-learning restarts produced non-finite objectives: {messages}"
            )
        selected = chosen.index
        estimates = np.asarray(chosen.estimate, dtype=np.float64)
        value, gradient = objective(estimates)
        hessian = _numerical_hessian(objective, estimates)
        condition = float(np.linalg.cond(hessian))
        covariance = np.linalg.pinv(hessian, hermitian=True)
        standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        components = _decode_parameters(estimates)
        boundary = bool(
            components.learning_rate <= self.learning_rate_warning_threshold
            or components.learning_rate >= 1.0 - self.learning_rate_warning_threshold
            or components.inverse_temperature <= 1.0 / self.inverse_temperature_warning_threshold
            or components.inverse_temperature >= self.inverse_temperature_warning_threshold
            or abs(components.choice_bias) >= self.coefficient_warning_threshold
            or abs(components.perseveration) >= self.coefficient_warning_threshold
        )
        diagnostics = FitDiagnostics(
            converged=chosen.converged,
            optimizer=f"{run.backend} ({self.n_restarts} deterministic restarts)",
            status=chosen.status,
            message=chosen.message,
            n_iterations=chosen.n_iterations,
            objective=float(value),
            gradient_norm=float(np.linalg.norm(gradient)),
            hessian_condition=condition,
            boundary_estimate=boundary,
        )
        return QLearningFitResult(
            model_name=self.model_name,
            model_signature=self.signature,
            parameter_names=self.parameter_names,
            estimates=estimates,
            standard_errors=standard_errors,
            covariance=covariance,
            n_observations=len(study),
            diagnostics=diagnostics,
            restart_objectives=restart_objectives,
            restart_converged=np.asarray([attempt.converged for attempt in run.attempts]),
            restart_messages=tuple(attempt.message for attempt in run.attempts),
            selected_restart=selected,
            learning_rate=components.learning_rate,
            inverse_temperature=components.inverse_temperature,
            optimization_run=run,
        )

    def predict(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> Prediction:
        """Return choice probabilities using only earlier choices and rewards."""

        prediction_mode = self._prediction_mode(mode)
        self._validate_fit(fit)
        trajectory = self.value_trajectory(study, fit)
        return Prediction(
            probability=expit(trajectory.linear_predictor),
            linear_predictor=trajectory.linear_predictor,
            mode=prediction_mode,
        )

    def pointwise_log_prob(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> NDArray[np.float64]:
        """Score each observed choice before its reward updates agent values."""

        choices = self._choices(study)
        prediction = self.predict(study, fit, mode=mode)
        scores = choices * -np.logaddexp(0.0, -prediction.linear_predictor)
        scores += (1.0 - choices) * -np.logaddexp(0.0, prediction.linear_predictor)
        return protected_array(scores, dtype=np.float64)

    def value_trajectory(self, study: Study, fit: FitResult) -> ValueTrajectory:
        """Reconstruct pre-choice values, updates, and reward-prediction errors."""

        self._validate_fit(fit)
        choices = self._choices(study)
        rewards = self._rewards(study)
        components = self.parameter_components(fit)
        pre_choice = np.empty((len(study), 2), dtype=np.float64)
        post_update = np.empty_like(pre_choice)
        prediction_error = np.empty(len(study), dtype=np.float64)
        linear_predictor = np.empty(len(study), dtype=np.float64)
        for session_indices in _ordered_session_indices(study):
            values = np.full(2, self.initial_value, dtype=np.float64)
            previous_choice = 0.0
            for index in session_indices:
                pre_choice[index] = values
                linear_predictor[index] = (
                    components.inverse_temperature * (values[1] - values[0])
                    + components.choice_bias
                    + components.perseveration * previous_choice
                )
                choice = int(choices[index])
                error = rewards[index] - values[choice]
                prediction_error[index] = error
                values[choice] += components.learning_rate * error
                post_update[index] = values
                previous_choice = 2.0 * choice - 1.0
        return ValueTrajectory(
            pre_choice=pre_choice,
            post_update=post_update,
            prediction_error=prediction_error,
            linear_predictor=linear_predictor,
        )

    def _objective_gradient(
        self,
        vector: NDArray[np.float64],
        choices: NDArray[np.float64],
        rewards: NDArray[np.float64],
        sessions: tuple[tuple[int, ...], ...],
    ) -> tuple[float, NDArray[np.float64]]:
        parameters = _decode_parameters(vector)
        natural_gradient = np.zeros(4, dtype=np.float64)
        loss = 0.0
        for session_indices in sessions:
            values = np.full(2, self.initial_value, dtype=np.float64)
            value_derivative = np.zeros(2, dtype=np.float64)
            previous_choice = 0.0
            for index in session_indices:
                value_difference = values[1] - values[0]
                derivative_difference = value_derivative[1] - value_derivative[0]
                linear = (
                    parameters.inverse_temperature * value_difference
                    + parameters.choice_bias
                    + parameters.perseveration * previous_choice
                )
                choice = choices[index]
                residual = expit(linear) - choice
                loss += float(np.logaddexp(0.0, linear) - choice * linear)
                natural_gradient[0] += (
                    residual * parameters.inverse_temperature * derivative_difference
                )
                natural_gradient[1] += residual * value_difference
                natural_gradient[2] += residual
                natural_gradient[3] += residual * previous_choice

                action = int(choice)
                old_value = values[action]
                old_derivative = value_derivative[action]
                value_derivative[action] = (
                    (1.0 - parameters.learning_rate) * old_derivative + rewards[index] - old_value
                )
                values[action] = old_value + parameters.learning_rate * (rewards[index] - old_value)
                previous_choice = 2.0 * choice - 1.0

        gradient = np.asarray(
            [
                natural_gradient[0] * parameters.learning_rate * (1.0 - parameters.learning_rate),
                natural_gradient[1] * parameters.inverse_temperature,
                natural_gradient[2],
                natural_gradient[3],
            ],
            dtype=np.float64,
        )
        return loss, gradient

    def _initial_points(self) -> tuple[NDArray[np.float64], ...]:
        generator = np.random.default_rng(self.random_seed)
        learning_rates = np.linspace(0.15, 0.85, self.n_restarts)
        starts: list[NDArray[np.float64]] = []
        for restart, learning_rate in enumerate(learning_rates):
            scale = 0.0 if restart == 0 else 0.25
            starts.append(
                np.asarray(
                    [
                        np.log(learning_rate) - np.log1p(-learning_rate),
                        np.log(2.0) + generator.normal(0.0, scale),
                        generator.normal(0.0, scale),
                        generator.normal(0.0, scale),
                    ]
                )
            )
        return tuple(starts)

    def _choices(self, study: Study) -> NDArray[np.float64]:
        if self.outcome not in study.columns:
            raise ModelDataError(f"study is missing outcome column {self.outcome!r}")
        try:
            choices = np.asarray(study[self.outcome], dtype=np.float64)
        except (TypeError, ValueError):
            raise ModelDataError(f"outcome column {self.outcome!r} must be numeric") from None
        if not np.all(np.isfinite(choices)) or not np.all((choices == 0) | (choices == 1)):
            raise ModelDataError(f"outcome column {self.outcome!r} must contain only zero and one")
        return choices

    def _rewards(self, study: Study) -> NDArray[np.float64]:
        if self.reward not in study.columns:
            raise ModelDataError(f"study is missing reward column {self.reward!r}")
        try:
            rewards = np.asarray(study[self.reward], dtype=np.float64)
        except (TypeError, ValueError):
            raise ModelDataError(f"reward column {self.reward!r} must be numeric") from None
        if not np.all(np.isfinite(rewards)) or np.any((rewards < 0) | (rewards > 1)):
            raise ModelDataError(f"reward column {self.reward!r} must lie between zero and one")
        return rewards

    def _reward_probabilities(self, study: Study) -> NDArray[np.float64]:
        missing = [name for name in self.reward_probability_columns if name not in study.columns]
        if missing:
            raise ModelDataError(f"study is missing reward probability columns: {missing}")
        try:
            probabilities = np.column_stack(
                [
                    np.asarray(study[name], dtype=np.float64)
                    for name in self.reward_probability_columns
                ]
            )
        except (TypeError, ValueError):
            raise ModelDataError("reward probability columns must be numeric") from None
        if not np.all(np.isfinite(probabilities)) or np.any(
            (probabilities < 0) | (probabilities > 1)
        ):
            raise ModelDataError("reward probability columns must lie between zero and one")
        return probabilities

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


def _decode_parameters(vector: Sequence[float]) -> QLearningParameters:
    try:
        return QLearningParameters(**_Q_LEARNING_PARAMETER_SPACE.decode(vector))
    except ValueError as error:
        raise ValueError(
            "parameter vector must contain four finite optimizer coordinates"
        ) from error


def _numerical_hessian(
    objective: Callable[
        [NDArray[np.float64]],
        tuple[float, NDArray[np.float64]],
    ],
    estimates: NDArray[np.float64],
) -> NDArray[np.float64]:
    hessian = np.empty((len(estimates), len(estimates)), dtype=np.float64)
    for column in range(len(estimates)):
        step = 1e-5 * (1.0 + abs(float(estimates[column])))
        positive = estimates.copy()
        negative = estimates.copy()
        positive[column] += step
        negative[column] -= step
        _, positive_gradient = objective(positive)
        _, negative_gradient = objective(negative)
        hessian[:, column] = (positive_gradient - negative_gradient) / (2.0 * step)
    return 0.5 * (hessian + hessian.T)


def _require_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _require_nonnegative_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
