"""Compact session-reset binary Q-learning reference model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit

from behavio._internal.arrays import protected_array
from behavio.contracts.bounded import (
    RowCoefficientDesign,
    block_constant_coordinates,
    validated_row_coefficients,
)
from behavio.contracts.compose import ridge_group_draw, ridge_group_penalty
from behavio.inference.optimize import OptimizationProblem, OptimizationRun, ScipyMultistart
from behavio.inference.parameters import ParameterSpace, ParameterSpec, ParameterTransform
from behavio.models._kernels.bernoulli import ordered_session_indices
from behavio.models._kernels.curvature import finite_difference_hessian, offset_steps
from behavio.models._kernels.rowfit import solve_row_coefficients
from behavio.models.base import (
    FitDiagnostics,
    FitResult,
    ModelDataError,
    Prediction,
    PredictionMode,
    UnsupportedPredictionMode,
)
from behavio.trials import REQUIRED_COLUMNS, Study

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

        for session_indices in ordered_session_indices(design):
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
        sessions = ordered_session_indices(study)

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
        hessian = finite_difference_hessian(
            lambda vector: objective(vector)[1],
            estimates,
            steps=offset_steps(estimates, scale=1e-5),
        )
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
        for session_indices in ordered_session_indices(study):
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

    # -- the bounded-coordinate composition contract --------------------------------------
    #
    # ``parameter_names`` is already the unconstrained coordinate -- a logit learning rate
    # and a log inverse temperature -- so hierarchy and smoothness need nothing added to it.
    # What they need is the likelihood as a function of one such coordinate *per row*, which
    # is the mirror image of ``simulate_rows``. See ``behavio.contracts.bounded``.

    def row_objective(self, study: Study) -> _QLearningRowObjective:
        """Return this study's negative log likelihood in one coordinate per row."""

        sessions = ordered_session_indices(study)
        row_blocks = np.empty(len(study), dtype=np.intp)
        for block, indices in enumerate(sessions):
            row_blocks[np.asarray(indices, dtype=np.intp)] = block
        return _QLearningRowObjective(
            model=self,
            choices=self._choices(study),
            rewards=self._rewards(study),
            sessions=sessions,
            row_blocks=row_blocks,
            n_rows=len(study),
        )

    def penalty_matrix(self) -> NDArray[np.float64]:
        """Return the quadratic penalty on the coordinate, which is none.

        A Q-learning fit is a maximum-likelihood fit inside a box; whatever regularisation
        it has comes from the box and from the transforms, not from a ridge. Composition
        adds a group or roughness prior on top of this, which is the only place a penalty
        on this model has ever come from.
        """

        return np.zeros((4, 4), dtype=np.float64)

    def coordinate_box(self, study: Study) -> NDArray[np.float64]:
        """Return the finite optimizer box the parameter space declares."""

        del study
        return np.asarray(
            [bounds for bounds in self.parameter_space.optimizer_bounds], dtype=np.float64
        )

    def initial_points(self, study: Study) -> tuple[NDArray[np.float64], ...]:
        """Return the deterministic restarts this model's own solver would use."""

        del study
        return self._initial_points()

    def group_parameter_expansion(self, name: str) -> tuple[str, ...]:
        """Return the reported parameters one declared varying name stands for."""

        return (name,)

    def group_penalty(
        self, columns: NDArray[np.intp], scales: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Return the isotropic Gaussian precision on one group's deviation."""

        del columns
        return ridge_group_penalty(scales)

    def draw_group_deviations(
        self,
        columns: NDArray[np.intp],
        scales: NDArray[np.float64],
        *,
        groups: int,
        generator: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Draw Gaussian deviations on the *transformed* coordinate, never the natural one."""

        del columns
        return ridge_group_draw(scales, groups=groups, generator=generator)

    def fit_rows(
        self,
        design: RowCoefficientDesign,
        *,
        model_name: str,
        model_signature: str,
    ) -> FitResult:
        """Solve a row-coefficient problem on this model's own optimizer settings."""

        return solve_row_coefficients(
            design,
            model_name=model_name,
            model_signature=model_signature,
            optimizer="L-BFGS-B",
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
            boundary=self._row_boundary,
        )

    def simulate_rows(
        self,
        design: Study,
        coefficients: NDArray[np.float64],
        *,
        seed: int | np.random.Generator,
    ) -> Study:
        """Generate choices and rewards with one parameter vector per row."""

        rows = validated_row_coefficients(
            coefficients, n_rows=len(design), n_parameters=4, what="simulate_rows"
        )
        environment = self._reward_probabilities(design)
        generator = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        choices = np.zeros(len(design), dtype=np.int8)
        rewards = np.zeros(len(design), dtype=np.int8)
        for session_indices in ordered_session_indices(design):
            components = _session_parameters(rows, session_indices, "simulate_rows coefficients")
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

    def predict_rows(
        self,
        study: Study,
        coefficients: NDArray[np.float64],
        *,
        mode: PredictionMode,
    ) -> Prediction:
        """Return filtered choice probabilities under one parameter vector per row."""

        prediction_mode = self._prediction_mode(mode)
        linear = self._row_linear_predictor(study, coefficients)
        return Prediction(probability=expit(linear), linear_predictor=linear, mode=prediction_mode)

    def pointwise_log_prob_rows(
        self, study: Study, coefficients: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Score each observed choice under one parameter vector per row."""

        choices = self._choices(study)
        linear = self._row_linear_predictor(study, coefficients)
        scores = choices * -np.logaddexp(0.0, -linear)
        scores += (1.0 - choices) * -np.logaddexp(0.0, linear)
        return protected_array(scores, dtype=np.float64)

    def _row_linear_predictor(
        self, study: Study, coefficients: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        rows = validated_row_coefficients(
            coefficients, n_rows=len(study), n_parameters=4, what="row coefficients"
        )
        choices = self._choices(study)
        rewards = self._rewards(study)
        linear = np.empty(len(study), dtype=np.float64)
        for session_indices in ordered_session_indices(study):
            components = _session_parameters(rows, session_indices, "row coefficients")
            values = np.full(2, self.initial_value, dtype=np.float64)
            previous_choice = 0.0
            for index in session_indices:
                linear[index] = (
                    components.inverse_temperature * (values[1] - values[0])
                    + components.choice_bias
                    + components.perseveration * previous_choice
                )
                choice = int(choices[index])
                values[choice] += components.learning_rate * (rewards[index] - values[choice])
                previous_choice = 2.0 * choice - 1.0
        return linear

    def _row_boundary(
        self,
        estimates: NDArray[np.float64],
        derived: NDArray[np.float64] | None,
    ) -> bool:
        """Apply this model's own boundary convention to a composed estimate.

        The convention is the natural-scale one ``fit`` already uses. It reads the population
        block, and each group's *population plus deviation* whenever that sum is a complete
        coordinate -- because an animal's behaviour is generated by the sum and not by either
        half of it. Where only some parameters vary the sum is a fragment that cannot be
        decoded, and the generic box check in
        :func:`~behavio.models._kernels.rowfit.solve_row_coefficients` is what covers it: a
        deviation large enough to saturate a rate reaches the box first.
        """

        candidates = [np.asarray(estimates, dtype=np.float64)[:4]]
        if derived is not None:
            candidates.extend(np.atleast_2d(np.asarray(derived, dtype=np.float64)))
        for vector in candidates:
            if len(vector) != 4:
                continue
            components = _decode_parameters(vector)
            if (
                components.learning_rate <= self.learning_rate_warning_threshold
                or components.learning_rate >= 1.0 - self.learning_rate_warning_threshold
                or components.inverse_temperature
                <= 1.0 / self.inverse_temperature_warning_threshold
                or components.inverse_temperature >= self.inverse_temperature_warning_threshold
                or abs(components.choice_bias) >= self.coefficient_warning_threshold
                or abs(components.perseveration) >= self.coefficient_warning_threshold
            ):
                return True
        return False

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


@dataclass(frozen=True, slots=True)
class _QLearningRowObjective:
    """The agent's negative log likelihood as a function of one coordinate per trial.

    Session-blocked rather than trial-separable, and the block structure is the model rather
    than an implementation convenience: values reset at every subject/session boundary and
    are carried forward inside one, so a parameter that changed mid-session would leave the
    value trace unable to say which of its values produced which part of it. Within a block
    the coordinate is one vector, and the session's own analytic gradient -- the same forward
    recursion :meth:`BinaryQLearning.fit` differentiates -- is exact for it.
    """

    model: BinaryQLearning
    choices: NDArray[np.float64]
    rewards: NDArray[np.float64]
    sessions: tuple[tuple[int, ...], ...]
    row_blocks: NDArray[np.intp]
    n_rows: int

    @property
    def n_parameters(self) -> int:
        """The width of one row's coordinate."""

        return 4

    def value_and_gradient(self, rows: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        """Return the negative log likelihood and its gradient in the row coordinates."""

        values = validated_row_coefficients(
            rows, n_rows=self.n_rows, n_parameters=4, what="row coordinates"
        )
        block_constant_coordinates(values, self.row_blocks, what="a Q-learning agent's parameters")
        total = 0.0
        gradient = np.zeros_like(values)
        for indices in self.sessions:
            index = np.asarray(indices, dtype=np.intp)
            value, block_gradient = self.model._objective_gradient(
                values[index[0]], self.choices, self.rewards, (indices,)
            )
            total += value
            gradient[index] = block_gradient / len(index)
        return float(total), gradient


def _session_parameters(
    rows: NDArray[np.float64], indices: tuple[int, ...], what: str
) -> QLearningParameters:
    """Decode the one coordinate a session's rows share, refusing rows that disagree."""

    index = np.asarray(indices, dtype=np.intp)
    block = rows[index]
    if not np.array_equal(block, np.tile(block[0], (len(index), 1))):
        raise ValueError(
            f"{what} must be constant within a session: a value trace cannot say which of "
            "two parameter values produced it"
        )
    return _decode_parameters(block[0])


def _decode_parameters(vector: Sequence[float]) -> QLearningParameters:
    try:
        return QLearningParameters(**_Q_LEARNING_PARAMETER_SPACE.decode(vector))
    except ValueError as error:
        raise ValueError(
            "parameter vector must contain four finite optimizer coordinates"
        ) from error


def _require_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _require_nonnegative_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
