"""Composable binary reinforcement-learning agents and inspectable trajectories."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import expit, logit

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


class LearningRule(Protocol):
    """A named value-update component with optimizer and natural coordinates."""

    @property
    def parameter_names(self) -> tuple[str, ...]: ...

    @property
    def natural_names(self) -> tuple[str, ...]: ...

    @property
    def signature(self) -> str: ...

    def encode(self, natural: Mapping[str, float]) -> tuple[float, ...]: ...

    def decode(self, optimizer: Mapping[str, float]) -> Mapping[str, float]: ...

    def rate(self, prediction_error: float, natural: Mapping[str, float]) -> float: ...


@dataclass(frozen=True, slots=True)
class SymmetricLearning:
    """One fitted delta-rule learning rate for positive and negative errors."""

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return ("learning_rate_logit",)

    @property
    def natural_names(self) -> tuple[str, ...]:
        return ("learning_rate",)

    @property
    def signature(self) -> str:
        return "symmetric_delta"

    def encode(self, natural: Mapping[str, float]) -> tuple[float, ...]:
        rate = _unit_interval(natural, "learning_rate")
        return (float(logit(rate)),)

    def decode(self, optimizer: Mapping[str, float]) -> Mapping[str, float]:
        return MappingProxyType(
            {"learning_rate": float(expit(_finite(optimizer, "learning_rate_logit")))}
        )

    def rate(self, prediction_error: float, natural: Mapping[str, float]) -> float:
        return float(natural["learning_rate"])


@dataclass(frozen=True, slots=True)
class AsymmetricLearning:
    """Separate fitted delta-rule rates for non-negative and negative errors."""

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return ("positive_learning_rate_logit", "negative_learning_rate_logit")

    @property
    def natural_names(self) -> tuple[str, ...]:
        return ("positive_learning_rate", "negative_learning_rate")

    @property
    def signature(self) -> str:
        return "asymmetric_delta"

    def encode(self, natural: Mapping[str, float]) -> tuple[float, ...]:
        positive = _unit_interval(natural, "positive_learning_rate")
        negative = _unit_interval(natural, "negative_learning_rate")
        return float(logit(positive)), float(logit(negative))

    def decode(self, optimizer: Mapping[str, float]) -> Mapping[str, float]:
        return MappingProxyType(
            {
                "positive_learning_rate": float(
                    expit(_finite(optimizer, "positive_learning_rate_logit"))
                ),
                "negative_learning_rate": float(
                    expit(_finite(optimizer, "negative_learning_rate_logit"))
                ),
            }
        )

    def rate(self, prediction_error: float, natural: Mapping[str, float]) -> float:
        name = "positive_learning_rate" if prediction_error >= 0 else "negative_learning_rate"
        return float(natural[name])


@dataclass(frozen=True, slots=True)
class UnchosenForgetting:
    """Fitted decay of unchosen values toward the declared initial value."""

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return ("forgetting_rate_logit",)

    @property
    def natural_names(self) -> tuple[str, ...]:
        return ("forgetting_rate",)

    @property
    def signature(self) -> str:
        return "unchosen_toward_initial"

    def encode(self, natural: Mapping[str, float]) -> tuple[float, ...]:
        return (float(logit(_unit_interval(natural, "forgetting_rate"))),)

    def decode(self, optimizer: Mapping[str, float]) -> Mapping[str, float]:
        return MappingProxyType(
            {"forgetting_rate": float(expit(_finite(optimizer, "forgetting_rate_logit")))}
        )

    def apply(
        self,
        values: NDArray[np.float64],
        chosen: int,
        initial_value: float,
        natural: Mapping[str, float],
    ) -> None:
        unchosen = 1 - chosen
        rate = float(natural["forgetting_rate"])
        values[unchosen] += rate * (initial_value - values[unchosen])


@dataclass(frozen=True, slots=True)
class ChoiceKernel:
    """Exponentially updated action-choice trace with a fitted policy weight."""

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return ("choice_kernel_rate_logit", "choice_kernel_weight")

    @property
    def natural_names(self) -> tuple[str, ...]:
        return ("choice_kernel_rate", "choice_kernel_weight")

    @property
    def signature(self) -> str:
        return "exponential_choice_kernel"

    def encode(self, natural: Mapping[str, float]) -> tuple[float, ...]:
        rate = _unit_interval(natural, "choice_kernel_rate")
        weight = _finite(natural, "choice_kernel_weight")
        return float(logit(rate)), weight

    def decode(self, optimizer: Mapping[str, float]) -> Mapping[str, float]:
        return MappingProxyType(
            {
                "choice_kernel_rate": float(expit(_finite(optimizer, "choice_kernel_rate_logit"))),
                "choice_kernel_weight": _finite(optimizer, "choice_kernel_weight"),
            }
        )

    def contribution(self, kernel: NDArray[np.float64], natural: Mapping[str, float]) -> float:
        return float(natural["choice_kernel_weight"] * (kernel[1] - kernel[0]))

    def update(
        self,
        kernel: NDArray[np.float64],
        chosen: int,
        natural: Mapping[str, float],
    ) -> None:
        rate = float(natural["choice_kernel_rate"])
        target = np.zeros(2, dtype=np.float64)
        target[chosen] = 1.0
        kernel += rate * (target - kernel)


@dataclass(frozen=True, slots=True)
class SoftmaxPolicy:
    """Binary softmax policy with optional bias and random-response lapse."""

    include_bias: bool = True
    maximum_lapse: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.include_bias, bool):
            raise ValueError("include_bias must be boolean")
        if self.maximum_lapse is not None and (
            not np.isfinite(self.maximum_lapse) or not 0 < self.maximum_lapse < 1
        ):
            raise ValueError("maximum_lapse must be null or lie strictly between zero and one")

    @property
    def parameter_names(self) -> tuple[str, ...]:
        names = ["inverse_temperature_log"]
        if self.include_bias:
            names.append("choice_bias")
        if self.maximum_lapse is not None:
            names.append("lapse_logit")
        return tuple(names)

    @property
    def natural_names(self) -> tuple[str, ...]:
        names = ["inverse_temperature"]
        if self.include_bias:
            names.append("choice_bias")
        if self.maximum_lapse is not None:
            names.append("lapse_rate")
        return tuple(names)

    @property
    def signature(self) -> str:
        return (
            f"binary_softmax(include_bias={self.include_bias!r},"
            f"maximum_lapse={self.maximum_lapse!r})"
        )

    def encode(self, natural: Mapping[str, float]) -> tuple[float, ...]:
        inverse_temperature = _positive(natural, "inverse_temperature")
        values = [float(np.log(inverse_temperature))]
        if self.include_bias:
            values.append(_finite(natural, "choice_bias"))
        if self.maximum_lapse is not None:
            lapse = _unit_interval(natural, "lapse_rate")
            if lapse >= self.maximum_lapse:
                raise ValueError("lapse_rate must be smaller than maximum_lapse")
            values.append(float(logit(lapse / self.maximum_lapse)))
        return tuple(values)

    def decode(self, optimizer: Mapping[str, float]) -> Mapping[str, float]:
        natural: dict[str, float] = {
            "inverse_temperature": float(np.exp(_finite(optimizer, "inverse_temperature_log")))
        }
        if self.include_bias:
            natural["choice_bias"] = _finite(optimizer, "choice_bias")
        if self.maximum_lapse is not None:
            natural["lapse_rate"] = float(
                self.maximum_lapse * expit(_finite(optimizer, "lapse_logit"))
            )
        return MappingProxyType(natural)

    def probability(self, linear: float, natural: Mapping[str, float]) -> float:
        probability = float(expit(linear))
        if self.maximum_lapse is not None:
            lapse = float(natural["lapse_rate"])
            probability = lapse * 0.5 + (1.0 - lapse) * probability
        return probability


@dataclass(frozen=True, slots=True)
class ResetRule:
    """Columns whose joint identity starts a fresh value and kernel state."""

    columns: tuple[str, ...] = ("subject", "session")

    def __post_init__(self) -> None:
        columns = tuple(self.columns)
        if not columns or len(set(columns)) != len(columns):
            raise ValueError("reset columns must be non-empty and unique")
        if any(not isinstance(column, str) or not column for column in columns):
            raise ValueError("reset columns must contain non-empty strings")
        if "subject" not in columns:
            raise ValueError("reset columns must include subject")
        object.__setattr__(self, "columns", columns)

    @property
    def signature(self) -> str:
        return f"reset_by{self.columns!r}"


@dataclass(frozen=True, slots=True)
class RLFitResult(FitResult):
    """Composable-agent fit with natural parameters and every restart retained."""

    restart_objectives: NDArray[np.float64]
    restart_converged: NDArray[np.bool_]
    restart_messages: tuple[str, ...]
    selected_restart: int
    natural_parameter_names: tuple[str, ...]
    natural_parameter_values: NDArray[np.float64]

    def __post_init__(self) -> None:
        FitResult.__post_init__(self)
        objectives = _protected_array(self.restart_objectives, dtype=np.float64)
        converged = _protected_array(self.restart_converged, dtype=np.bool_)
        messages = tuple(self.restart_messages)
        names = tuple(self.natural_parameter_names)
        values = _protected_array(self.natural_parameter_values, dtype=np.float64)
        if objectives.ndim != 1 or converged.shape != objectives.shape:
            raise ValueError("restart diagnostics must have one value per restart")
        if len(messages) != len(objectives) or np.any(np.isnan(objectives)):
            raise ValueError("restart messages and non-NaN objectives must align")
        if not 0 <= self.selected_restart < len(objectives):
            raise ValueError("selected_restart must identify one restart")
        if not names or len(set(names)) != len(names) or values.shape != (len(names),):
            raise ValueError("natural parameter names and values must be aligned and unique")
        if not np.all(np.isfinite(values)):
            raise ValueError("natural parameter values must be finite")
        object.__setattr__(self, "restart_objectives", objectives)
        object.__setattr__(self, "restart_converged", converged)
        object.__setattr__(self, "restart_messages", messages)
        object.__setattr__(self, "natural_parameter_names", names)
        object.__setattr__(self, "natural_parameter_values", values)

    @property
    def natural_parameters(self) -> Mapping[str, float]:
        return MappingProxyType(
            dict(
                zip(
                    self.natural_parameter_names,
                    self.natural_parameter_values.tolist(),
                    strict=True,
                )
            )
        )


@dataclass(frozen=True, slots=True)
class RLTrajectory:
    """Pre-choice states, post-outcome values, and filtered policy evidence."""

    pre_choice_values: NDArray[np.float64]
    post_update_values: NDArray[np.float64]
    pre_choice_kernel: NDArray[np.float64]
    post_update_kernel: NDArray[np.float64]
    prediction_error: NDArray[np.float64]
    probability: NDArray[np.float64]
    linear_predictor: NDArray[np.float64]

    def __post_init__(self) -> None:
        arrays = tuple(
            _protected_array(values, dtype=np.float64)
            for values in (
                self.pre_choice_values,
                self.post_update_values,
                self.pre_choice_kernel,
                self.post_update_kernel,
                self.prediction_error,
                self.probability,
                self.linear_predictor,
            )
        )
        pre, post, kernel_pre, kernel_post, error, probability, linear = arrays
        if pre.ndim != 2 or pre.shape[1] != 2 or post.shape != pre.shape:
            raise ValueError("value trajectories must have two aligned action columns")
        if kernel_pre.shape != pre.shape or kernel_post.shape != pre.shape:
            raise ValueError("choice-kernel trajectories must align with values")
        if any(vector.shape != (len(pre),) for vector in (error, probability, linear)):
            raise ValueError("trajectory vectors must contain one value per trial")
        if not all(np.all(np.isfinite(values)) for values in arrays):
            raise ValueError("trajectory arrays must be finite")
        if np.any((probability <= 0) | (probability >= 1)):
            raise ValueError("trajectory probabilities must lie strictly between zero and one")
        for name, value in zip(
            (
                "pre_choice_values",
                "post_update_values",
                "pre_choice_kernel",
                "post_update_kernel",
                "prediction_error",
                "probability",
                "linear_predictor",
            ),
            arrays,
            strict=True,
        ):
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class BinaryRLAgent:
    """A two-action agent assembled from explicit learning, memory, and policy parts."""

    learning: LearningRule = SymmetricLearning()
    forgetting: UnchosenForgetting | None = None
    choice_kernel: ChoiceKernel | None = None
    policy: SoftmaxPolicy = SoftmaxPolicy()
    reset: ResetRule = ResetRule()
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
    parameter_warning_threshold: float = 20.0

    def __post_init__(self) -> None:
        _validate_component(self.learning, "learning", ("parameter_names", "natural_names"))
        if self.forgetting is not None and not isinstance(self.forgetting, UnchosenForgetting):
            raise TypeError("forgetting must be UnchosenForgetting or null")
        if self.choice_kernel is not None and not isinstance(self.choice_kernel, ChoiceKernel):
            raise TypeError("choice_kernel must be ChoiceKernel or null")
        if not isinstance(self.policy, SoftmaxPolicy):
            raise TypeError("policy must be a SoftmaxPolicy")
        if not isinstance(self.reset, ResetRule):
            raise TypeError("reset must be a ResetRule")
        for value, label in ((self.outcome, "outcome"), (self.reward, "reward")):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{label} must be a non-empty column name")
            if value in REQUIRED_COLUMNS:
                raise ValueError(f"{label} cannot replace a required Study column")
        if self.outcome == self.reward:
            raise ValueError("outcome and reward columns must be distinct")
        environment = tuple(self.reward_probability_columns)
        if len(environment) != 2 or len(set(environment)) != 2:
            raise ValueError("reward_probability_columns must contain two distinct names")
        if any(not isinstance(column, str) or not column for column in environment):
            raise ValueError("reward probability columns must be non-empty strings")
        if set(environment) & {self.outcome, self.reward, *REQUIRED_COLUMNS}:
            raise ValueError("reward probability columns must not replace observed columns")
        if not np.isfinite(self.initial_value) or not 0 <= self.initial_value <= 1:
            raise ValueError("initial_value must be finite and lie between zero and one")
        _positive_integer(self.n_restarts, "n_restarts")
        _nonnegative_integer(self.random_seed, "random_seed")
        _positive_integer(self.max_iterations, "max_iterations")
        if not np.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("tolerance must be finite and positive")
        if (
            not np.isfinite(self.parameter_warning_threshold)
            or self.parameter_warning_threshold <= 0
        ):
            raise ValueError("parameter_warning_threshold must be finite and positive")
        if len(set(self.parameter_names)) != len(self.parameter_names):
            raise ValueError("RL components produce colliding optimizer parameter names")
        if len(set(self.natural_parameter_names)) != len(self.natural_parameter_names):
            raise ValueError("RL components produce colliding natural parameter names")
        object.__setattr__(self, "reward_probability_columns", environment)

    @property
    def model_name(self) -> str:
        return "binary-rl-agent"

    @property
    def signature(self) -> str:
        forgetting = "none" if self.forgetting is None else self.forgetting.signature
        kernel = "none" if self.choice_kernel is None else self.choice_kernel.signature
        environment = ",".join(self.reward_probability_columns)
        return (
            f"{self.model_name}[learning={self.learning.signature};forgetting={forgetting};"
            f"choice_kernel={kernel};policy={self.policy.signature};reset={self.reset.signature};"
            f"outcome={self.outcome};reward={self.reward};environment={environment};"
            f"initial_value={self.initial_value}]"
        )

    @property
    def parameter_names(self) -> tuple[str, ...]:
        names = [*self.learning.parameter_names]
        if self.forgetting is not None:
            names.extend(self.forgetting.parameter_names)
        if self.choice_kernel is not None:
            names.extend(self.choice_kernel.parameter_names)
        names.extend(self.policy.parameter_names)
        return tuple(names)

    @property
    def natural_parameter_names(self) -> tuple[str, ...]:
        names = [*self.learning.natural_names]
        if self.forgetting is not None:
            names.extend(self.forgetting.natural_names)
        if self.choice_kernel is not None:
            names.extend(self.choice_kernel.natural_names)
        names.extend(self.policy.natural_names)
        return tuple(names)

    @property
    def scored_columns(self) -> tuple[str, ...]:
        return (self.outcome,)

    @property
    def required_task_columns(self) -> tuple[str, ...]:
        return (self.reward,)

    @property
    def supported_prediction_modes(self) -> tuple[PredictionMode, ...]:
        return (PredictionMode.FILTERED,)

    def parameters_from_components(self, **natural: float) -> Mapping[str, float]:
        """Encode one exact natural-parameter mapping for this component assembly."""

        if set(natural) != set(self.natural_parameter_names):
            raise ValueError(
                "natural parameters must match the assembled agent exactly; "
                f"missing={sorted(set(self.natural_parameter_names) - set(natural))}, "
                f"extra={sorted(set(natural) - set(self.natural_parameter_names))}"
            )
        encoded: list[float] = [*self.learning.encode(natural)]
        if self.forgetting is not None:
            encoded.extend(self.forgetting.encode(natural))
        if self.choice_kernel is not None:
            encoded.extend(self.choice_kernel.encode(natural))
        encoded.extend(self.policy.encode(natural))
        return MappingProxyType(dict(zip(self.parameter_names, encoded, strict=True)))

    def parameter_components(
        self, parameters: Mapping[str, float] | FitResult
    ) -> Mapping[str, float]:
        """Decode optimizer coordinates into all named natural components."""

        if isinstance(parameters, FitResult):
            self._validate_fit(parameters)
            optimizer = dict(zip(self.parameter_names, parameters.estimates, strict=True))
        else:
            if set(parameters) != set(self.parameter_names):
                raise ValueError("parameters must match the assembled agent exactly")
            optimizer = {name: _finite(parameters, name) for name in self.parameter_names}
        natural: dict[str, float] = dict(self.learning.decode(optimizer))
        if self.forgetting is not None:
            natural.update(self.forgetting.decode(optimizer))
        if self.choice_kernel is not None:
            natural.update(self.choice_kernel.decode(optimizer))
        natural.update(self.policy.decode(optimizer))
        return MappingProxyType(natural)

    def simulate(
        self,
        design: Study,
        parameters: Mapping[str, float],
        *,
        seed: int | np.random.Generator,
    ) -> Study:
        natural = self.parameter_components(parameters)
        environment = self._reward_probabilities(design)
        generator = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        choices = np.zeros(len(design), dtype=np.int8)
        rewards = np.zeros(len(design), dtype=np.int8)
        for indices in self._groups(design):
            values = np.full(2, self.initial_value, dtype=np.float64)
            kernel = np.zeros(2, dtype=np.float64)
            for index in indices:
                linear = self._linear(values, kernel, natural)
                probability = self.policy.probability(linear, natural)
                choice = int(generator.binomial(1, probability))
                reward = int(generator.binomial(1, environment[index, choice]))
                choices[index] = choice
                rewards[index] = reward
                self._update(values, kernel, choice, reward, natural)
        columns = {name: design[name] for name in design.columns}
        columns[self.outcome] = choices
        columns[self.reward] = rewards
        return Study(columns)

    def fit(self, study: Study) -> RLFitResult:
        choices = self._choices(study)
        rewards = self._rewards(study)
        groups = self._groups(study)

        def objective(vector: NDArray[np.float64]) -> float:
            natural = self.parameter_components(
                dict(zip(self.parameter_names, vector, strict=True))
            )
            loss = 0.0
            for indices in groups:
                values = np.full(2, self.initial_value, dtype=np.float64)
                kernel = np.zeros(2, dtype=np.float64)
                for index in indices:
                    linear = self._linear(values, kernel, natural)
                    probability = self.policy.probability(linear, natural)
                    choice = int(choices[index])
                    loss -= float(
                        choice * np.log(probability) + (1 - choice) * np.log1p(-probability)
                    )
                    self._update(values, kernel, choice, rewards[index], natural)
            return loss

        starts = self._initial_points()
        bounds = self._bounds()
        results = [
            minimize(
                objective,
                start,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": self.max_iterations, "ftol": self.tolerance},
            )
            for start in starts
        ]
        objectives = np.asarray(
            [float(result.fun) if np.isfinite(result.fun) else np.inf for result in results]
        )
        finite = np.flatnonzero(np.isfinite(objectives)).tolist()
        if not finite:
            raise ModelDataError("all composable RL restarts produced non-finite objectives")
        converged = [index for index in finite if results[index].success]
        eligible = converged if converged else finite
        selected = min(eligible, key=lambda index: float(objectives[index]))
        chosen = results[selected]
        estimates = np.asarray(chosen.x, dtype=np.float64)
        gradient = _finite_gradient(objective, estimates)
        hessian = _finite_hessian(objective, estimates)
        condition = float(np.linalg.cond(hessian))
        covariance = np.linalg.pinv(hessian, hermitian=True)
        standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        natural = self.parameter_components(dict(zip(self.parameter_names, estimates, strict=True)))
        diagnostics = FitDiagnostics(
            converged=bool(chosen.success),
            optimizer=f"L-BFGS-B ({self.n_restarts} deterministic restarts; numeric gradient)",
            status=int(chosen.status),
            message=str(chosen.message),
            n_iterations=int(chosen.nit),
            objective=float(chosen.fun),
            gradient_norm=float(np.linalg.norm(gradient)),
            hessian_condition=condition,
            boundary_estimate=self._boundary(estimates, natural),
        )
        return RLFitResult(
            model_name=self.model_name,
            model_signature=self.signature,
            parameter_names=self.parameter_names,
            estimates=estimates,
            standard_errors=standard_errors,
            covariance=covariance,
            n_observations=len(study),
            diagnostics=diagnostics,
            restart_objectives=objectives,
            restart_converged=np.asarray([result.success for result in results]),
            restart_messages=tuple(str(result.message) for result in results),
            selected_restart=selected,
            natural_parameter_names=self.natural_parameter_names,
            natural_parameter_values=np.asarray(
                [natural[name] for name in self.natural_parameter_names]
            ),
        )

    def predict(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> Prediction:
        prediction_mode = self._prediction_mode(mode)
        trajectory = self.trajectory(study, fit)
        return Prediction(
            probability=trajectory.probability,
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
        choices = self._choices(study)
        probability = self.predict(study, fit, mode=mode).probability
        scores = choices * np.log(probability) + (1.0 - choices) * np.log1p(-probability)
        return _protected_array(scores, dtype=np.float64)

    def trajectory(self, study: Study, fit: FitResult) -> RLTrajectory:
        """Replay the declared components using earlier choices and rewards only."""

        self._validate_fit(fit)
        choices = self._choices(study)
        rewards = self._rewards(study)
        natural = self.parameter_components(fit)
        pre = np.empty((len(study), 2), dtype=np.float64)
        post = np.empty_like(pre)
        kernel_pre = np.empty_like(pre)
        kernel_post = np.empty_like(pre)
        errors = np.empty(len(study), dtype=np.float64)
        probability = np.empty(len(study), dtype=np.float64)
        linear_predictor = np.empty(len(study), dtype=np.float64)
        for indices in self._groups(study):
            values = np.full(2, self.initial_value, dtype=np.float64)
            kernel = np.zeros(2, dtype=np.float64)
            for index in indices:
                pre[index] = values
                kernel_pre[index] = kernel
                linear = self._linear(values, kernel, natural)
                trial_probability = self.policy.probability(linear, natural)
                probability[index] = trial_probability
                linear_predictor[index] = float(logit(trial_probability))
                choice = int(choices[index])
                errors[index] = rewards[index] - values[choice]
                self._update(values, kernel, choice, rewards[index], natural)
                post[index] = values
                kernel_post[index] = kernel
        return RLTrajectory(
            pre_choice_values=pre,
            post_update_values=post,
            pre_choice_kernel=kernel_pre,
            post_update_kernel=kernel_post,
            prediction_error=errors,
            probability=probability,
            linear_predictor=linear_predictor,
        )

    def _linear(
        self,
        values: NDArray[np.float64],
        kernel: NDArray[np.float64],
        natural: Mapping[str, float],
    ) -> float:
        linear = float(natural["inverse_temperature"] * (values[1] - values[0]))
        if self.policy.include_bias:
            linear += float(natural["choice_bias"])
        if self.choice_kernel is not None:
            linear += self.choice_kernel.contribution(kernel, natural)
        return linear

    def _update(
        self,
        values: NDArray[np.float64],
        kernel: NDArray[np.float64],
        choice: int,
        reward: float,
        natural: Mapping[str, float],
    ) -> None:
        error = float(reward - values[choice])
        values[choice] += self.learning.rate(error, natural) * error
        if self.forgetting is not None:
            self.forgetting.apply(values, choice, self.initial_value, natural)
        if self.choice_kernel is not None:
            self.choice_kernel.update(kernel, choice, natural)

    def _groups(self, study: Study) -> tuple[tuple[int, ...], ...]:
        missing = [column for column in self.reset.columns if column not in study.columns]
        if missing:
            raise ModelDataError(f"study is missing reset columns: {missing}")
        groups: dict[tuple[Any, ...], list[int]] = {}
        for raw_index in study.chronological_indices():
            index = int(raw_index)
            key = tuple(_label(study[column][index]) for column in self.reset.columns)
            groups.setdefault(key, []).append(index)
        return tuple(tuple(indices) for indices in groups.values())

    def _choices(self, study: Study) -> NDArray[np.float64]:
        return self._numeric_observation(study, self.outcome, binary=True)

    def _rewards(self, study: Study) -> NDArray[np.float64]:
        return self._numeric_observation(study, self.reward, binary=False)

    def _numeric_observation(
        self, study: Study, column: str, *, binary: bool
    ) -> NDArray[np.float64]:
        if column not in study.columns:
            raise ModelDataError(f"study is missing observation column {column!r}")
        try:
            values = np.asarray(study[column], dtype=np.float64)
        except (TypeError, ValueError):
            raise ModelDataError(f"observation column {column!r} must be numeric") from None
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise ModelDataError(f"observation column {column!r} must be finite")
        if binary and not np.all((values == 0) | (values == 1)):
            raise ModelDataError(f"observation column {column!r} must contain zero and one")
        if not binary and np.any((values < 0) | (values > 1)):
            raise ModelDataError(f"observation column {column!r} must lie between zero and one")
        return values

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
            raise ModelDataError("reward probabilities must be numeric") from None
        if not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
            raise ModelDataError("reward probabilities must lie between zero and one")
        return values

    def _initial_points(self) -> tuple[NDArray[np.float64], ...]:
        generator = np.random.default_rng(self.random_seed)
        starts: list[NDArray[np.float64]] = []
        base_natural = self._default_natural()
        base = np.asarray(
            list(self.parameters_from_components(**base_natural).values()), dtype=np.float64
        )
        for restart in range(self.n_restarts):
            scale = 0.0 if restart == 0 else 0.35
            starts.append(base + generator.normal(0.0, scale, size=len(base)))
        return tuple(starts)

    def _default_natural(self) -> dict[str, float]:
        natural: dict[str, float] = {}
        for name in self.natural_parameter_names:
            if "learning_rate" in name:
                natural[name] = 0.25
            elif name == "forgetting_rate":
                natural[name] = 0.1
            elif name == "choice_kernel_rate":
                natural[name] = 0.25
            elif name == "choice_kernel_weight":
                natural[name] = 0.0
            elif name == "inverse_temperature":
                natural[name] = 2.0
            elif name == "choice_bias":
                natural[name] = 0.0
            elif name == "lapse_rate":
                natural[name] = min(0.02, float(self.policy.maximum_lapse) / 2.0)
            else:
                raise RuntimeError(f"no default natural parameter for {name!r}")
        return natural

    def _bounds(self) -> tuple[tuple[float, float], ...]:
        bounds = []
        for name in self.parameter_names:
            if name.endswith("_logit"):
                bounds.append((-12.0, 12.0))
            elif name == "inverse_temperature_log":
                bounds.append((-5.0, 5.0))
            else:
                bounds.append((-30.0, 30.0))
        return tuple(bounds)

    def _boundary(self, estimates: NDArray[np.float64], natural: Mapping[str, float]) -> bool:
        if any(
            value <= 1e-4 or value >= 1.0 - 1e-4
            for name, value in natural.items()
            if name.endswith("_rate") and name != "lapse_rate"
        ):
            return True
        if natural["inverse_temperature"] <= 0.02 or natural["inverse_temperature"] >= 50:
            return True
        return bool(np.any(np.abs(estimates) >= self.parameter_warning_threshold))

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


def _finite_gradient(function: Any, values: NDArray[np.float64]) -> NDArray[np.float64]:
    gradient = np.empty(len(values), dtype=np.float64)
    for index in range(len(values)):
        step = 1e-5 * (1.0 + abs(float(values[index])))
        positive = values.copy()
        negative = values.copy()
        positive[index] += step
        negative[index] -= step
        gradient[index] = (function(positive) - function(negative)) / (2.0 * step)
    return gradient


def _finite_hessian(function: Any, values: NDArray[np.float64]) -> NDArray[np.float64]:
    hessian = np.empty((len(values), len(values)), dtype=np.float64)
    for column in range(len(values)):
        step = 1e-4 * (1.0 + abs(float(values[column])))
        positive = values.copy()
        negative = values.copy()
        positive[column] += step
        negative[column] -= step
        hessian[:, column] = (
            _finite_gradient(function, positive) - _finite_gradient(function, negative)
        ) / (2.0 * step)
    return 0.5 * (hessian + hessian.T)


def _validate_component(component: Any, label: str, attributes: tuple[str, ...]) -> None:
    if any(not hasattr(component, attribute) for attribute in attributes):
        raise TypeError(f"{label} does not satisfy the component contract")


def _finite(values: Mapping[str, float], name: str) -> float:
    try:
        value = float(values[name])
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"{name} must be a finite numeric value") from None
    if not np.isfinite(value):
        raise ValueError(f"{name} must be a finite numeric value")
    return value


def _unit_interval(values: Mapping[str, float], name: str) -> float:
    value = _finite(values, name)
    if not 0 < value < 1:
        raise ValueError(f"{name} must lie strictly between zero and one")
    return value


def _positive(values: Mapping[str, float], name: str) -> float:
    value = _finite(values, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _label(value: Any) -> tuple[type[Any], Any]:
    scalar = value.item() if isinstance(value, np.generic) else value
    return type(scalar), scalar


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _nonnegative_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
