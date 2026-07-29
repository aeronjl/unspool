"""Backend-neutral deterministic optimization.

``ObjectiveTarget``, ``PriorMeasure`` and the ``OptimizationBackend`` protocol now live in
:mod:`behavio.contracts.backend` and are re-exported here, so
``from behavio.inference import OptimizationBackend`` keeps working. ``OptimizationProblem``,
``OptimizationAttempt`` and ``OptimizationRun`` stay here: they carry substantial validation
logic and need a concrete :class:`~behavio.parameters.ParameterSpace`, which
``behavio.contracts`` must not import at runtime.
"""

from __future__ import annotations

import importlib
import importlib.metadata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from behavio.contracts.backend import (  # noqa: F401  (OptimizationBackend is re-exported)
    ObjectiveTarget,
    OptimizationBackend,
    PriorMeasure,
)
from behavio.parameters import ParameterSpace, ParameterSpaceError

ObjectiveOutput = float | tuple[float, Sequence[float] | NDArray[np.floating[Any]]]
ObjectiveFunction = Callable[[NDArray[np.float64]], ObjectiveOutput]


class InferenceError(ValueError):
    """Raised when an inference problem or backend result violates its contract."""


class PyBADSUnavailableError(ImportError):
    """Raised when the optional PyBADS backend is requested but not installed."""


@dataclass(frozen=True, slots=True)
class OptimizationProblem:
    """A fully specified objective and deterministic optimizer starts.

    ``objective`` is a negative log-likelihood on ordered optimizer coordinates. For MAP,
    priors declared by ``parameter_space`` are added automatically. Natural-measure MAP
    omits transform Jacobians; optimizer-measure MAP includes them explicitly.
    """

    parameter_space: ParameterSpace
    objective: ObjectiveFunction
    starts: tuple[Sequence[float] | NDArray[np.floating[Any]], ...]
    has_gradient: bool
    target: ObjectiveTarget = ObjectiveTarget.MAXIMUM_LIKELIHOOD
    prior_measure: PriorMeasure = PriorMeasure.NATURAL
    objective_name: str = "negative_log_likelihood"

    def __post_init__(self) -> None:
        if not isinstance(self.parameter_space, ParameterSpace):
            raise TypeError("parameter_space must be a ParameterSpace")
        if not callable(self.objective):
            raise TypeError("objective must be callable")
        if not isinstance(self.has_gradient, bool):
            raise InferenceError("has_gradient must be boolean")
        if not isinstance(self.objective_name, str) or not self.objective_name:
            raise InferenceError("objective_name must be a non-empty string")
        try:
            target = ObjectiveTarget(self.target)
            prior_measure = PriorMeasure(self.prior_measure)
        except ValueError as error:
            raise InferenceError("invalid objective target or prior measure") from error
        if target == ObjectiveTarget.MAXIMUM_LIKELIHOOD and prior_measure != PriorMeasure.NATURAL:
            raise InferenceError("prior_measure is only meaningful for MAP objectives")
        starts = tuple(
            self._validate_vector(start, f"start {index}")
            for index, start in enumerate(self.starts)
        )
        if not starts:
            raise InferenceError("starts must contain at least one optimizer vector")
        for index, start in enumerate(starts):
            for name, value, bounds in zip(
                self.parameter_space.optimizer_names,
                start,
                self.parameter_space.optimizer_bounds,
                strict=True,
            ):
                lower, upper = bounds
                if (lower is not None and value < lower) or (upper is not None and value > upper):
                    raise InferenceError(
                        f"start {index} coordinate {name!r} lies outside optimizer bounds"
                    )
        if target == ObjectiveTarget.MAXIMUM_A_POSTERIORI:
            natural = self.parameter_space.decode(starts[0])
            try:
                self.parameter_space.log_prior(natural, require_all=True)
            except ParameterSpaceError as error:
                raise InferenceError(str(error)) from error
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "prior_measure", prior_measure)
        object.__setattr__(self, "starts", starts)

    def evaluate(self, optimizer: Sequence[float] | NDArray[np.floating[Any]]) -> ObjectiveOutput:
        """Evaluate the complete MLE or MAP objective on optimizer coordinates."""

        vector = self._validate_vector(optimizer, "optimizer")
        raw = self.objective(vector)
        if self.has_gradient:
            if not isinstance(raw, tuple) or len(raw) != 2:
                raise InferenceError("gradient objectives must return (value, gradient)")
            value = _objective_value(raw[0])
            gradient = self._validate_vector(raw[1], "objective gradient")
        else:
            if isinstance(raw, tuple):
                raise InferenceError("value-only objectives must return one scalar")
            value = _objective_value(raw)
            gradient = None

        if self.target == ObjectiveTarget.MAXIMUM_A_POSTERIORI:
            natural = self.parameter_space.decode(vector)
            log_prior = self.parameter_space.log_prior(natural, require_all=True)
            if not np.isfinite(log_prior):
                if gradient is None:
                    return np.inf
                return np.inf, _protected_vector(np.zeros_like(vector))
            value -= log_prior
            if gradient is not None:
                gradient = gradient - self.parameter_space.grad_log_prior_optimizer(
                    vector, require_all=True
                )
            if self.prior_measure == PriorMeasure.OPTIMIZER:
                value -= self.parameter_space.log_abs_det_inverse_jacobian(vector)
                if gradient is not None:
                    gradient = gradient - self.parameter_space.grad_log_abs_det_inverse_jacobian(
                        vector
                    )
        if gradient is None:
            return float(value)
        return float(value), _protected_vector(gradient)

    def to_dict(self) -> dict[str, Any]:
        """Return non-executable problem provenance."""

        return {
            "objective_name": self.objective_name,
            "target": self.target.value,
            "prior_measure": self.prior_measure.value,
            "has_gradient": self.has_gradient,
            "parameter_space_fingerprint": self.parameter_space.fingerprint,
            "optimizer_names": list(self.parameter_space.optimizer_names),
            "starts": [start.tolist() for start in self.starts],
        }

    def _validate_vector(
        self,
        values: Sequence[float] | NDArray[np.floating[Any]],
        label: str,
    ) -> NDArray[np.float64]:
        try:
            vector = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError):
            raise InferenceError(f"{label} must contain finite numeric values") from None
        expected = len(self.parameter_space.optimizer_names)
        if vector.shape != (expected,) or not np.all(np.isfinite(vector)):
            raise InferenceError(f"{label} must contain {expected} finite values")
        return _protected_vector(vector)


@dataclass(frozen=True, slots=True)
class OptimizationAttempt:
    """One complete attempted optimum, including unsuccessful evidence."""

    index: int
    start: NDArray[np.float64]
    estimate: NDArray[np.float64]
    objective: float
    converged: bool
    status: int
    message: str
    n_iterations: int
    n_evaluations: int
    n_gradient_evaluations: int
    gradient_norm: float

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise InferenceError("attempt index must be a non-negative integer")
        start = _protected_vector(self.start)
        estimate = _protected_vector(self.estimate, finite=False)
        if start.ndim != 1 or estimate.shape != start.shape or not np.all(np.isfinite(start)):
            raise InferenceError("attempt start and estimate vectors must align")
        objective = _non_nan_float(self.objective, "attempt objective")
        gradient_norm = _non_nan_float(self.gradient_norm, "attempt gradient_norm")
        if not isinstance(self.converged, bool):
            raise InferenceError("attempt converged must be boolean")
        if not isinstance(self.status, int) or isinstance(self.status, bool):
            raise InferenceError("attempt status must be an integer")
        if not isinstance(self.message, str):
            raise InferenceError("attempt message must be a string")
        for value, label in (
            (self.n_iterations, "n_iterations"),
            (self.n_evaluations, "n_evaluations"),
            (self.n_gradient_evaluations, "n_gradient_evaluations"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise InferenceError(f"attempt {label} must be a non-negative integer")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "estimate", estimate)
        object.__setattr__(self, "objective", objective)
        object.__setattr__(self, "gradient_norm", gradient_norm)

    @property
    def finite(self) -> bool:
        """Whether objective and estimated coordinates are all finite."""

        return bool(np.isfinite(self.objective) and np.all(np.isfinite(self.estimate)))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe attempt record without suppressing failures."""

        return {
            "index": self.index,
            "start": [_finite_or_none(value) for value in self.start],
            "estimate": [_finite_or_none(value) for value in self.estimate],
            "objective": _finite_or_none(self.objective),
            "converged": self.converged,
            "status": self.status,
            "message": self.message,
            "n_iterations": self.n_iterations,
            "n_evaluations": self.n_evaluations,
            "n_gradient_evaluations": self.n_gradient_evaluations,
            "gradient_norm": _finite_or_none(self.gradient_norm),
        }


@dataclass(frozen=True, slots=True)
class OptimizationRun:
    """All attempts and the deterministic selected optimum from one backend run."""

    backend: str
    backend_config: Mapping[str, Any]
    problem: Mapping[str, Any]
    attempts: tuple[OptimizationAttempt, ...]
    selected_index: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.backend, str) or not self.backend:
            raise InferenceError("backend must be a non-empty string")
        if not isinstance(self.backend_config, Mapping):
            raise InferenceError("backend_config must be a mapping")
        if not isinstance(self.problem, Mapping):
            raise InferenceError("problem must be a mapping")
        attempts = tuple(self.attempts)
        if not attempts or any(not isinstance(item, OptimizationAttempt) for item in attempts):
            raise InferenceError("attempts must contain OptimizationAttempt records")
        if tuple(item.index for item in attempts) != tuple(range(len(attempts))):
            raise InferenceError("attempt indices must be contiguous and ordered")
        if self.selected_index is not None:
            if (
                isinstance(self.selected_index, bool)
                or not isinstance(self.selected_index, int)
                or not 0 <= self.selected_index < len(attempts)
            ):
                raise InferenceError("selected_index must identify an attempted optimum")
            if not attempts[self.selected_index].finite:
                raise InferenceError("selected attempt must have finite estimates and objective")
        object.__setattr__(self, "backend_config", _freeze_record(self.backend_config))
        object.__setattr__(self, "problem", _freeze_record(self.problem))
        object.__setattr__(self, "attempts", attempts)

    @property
    def selected(self) -> OptimizationAttempt | None:
        """Selected attempt, or null when every attempt is numerically unusable."""

        return None if self.selected_index is None else self.attempts[self.selected_index]

    @property
    def any_converged(self) -> bool:
        """Whether at least one finite attempted optimum reported convergence."""

        return any(item.converged and item.finite for item in self.attempts)

    def to_dict(self) -> dict[str, Any]:
        """Return complete JSON-safe optimization evidence."""

        return {
            "backend": self.backend,
            "backend_config": _thaw_record(self.backend_config),
            "problem": _thaw_record(self.problem),
            "selected_index": self.selected_index,
            "attempts": [item.to_dict() for item in self.attempts],
        }


@dataclass(frozen=True, slots=True)
class ScipyMultistart:
    """Deterministic SciPy minimize backend retaining every attempted optimum."""

    method: str = "L-BFGS-B"
    max_iterations: int = 1_000
    function_tolerance: float = 1e-9
    gradient_tolerance: float = 1e-9

    def __post_init__(self) -> None:
        if self.method != "L-BFGS-B":
            raise InferenceError("ScipyMultistart currently supports only L-BFGS-B")
        if (
            isinstance(self.max_iterations, bool)
            or not isinstance(self.max_iterations, int)
            or self.max_iterations < 1
        ):
            raise InferenceError("max_iterations must be a positive integer")
        for value, label in (
            (self.function_tolerance, "function_tolerance"),
            (self.gradient_tolerance, "gradient_tolerance"),
        ):
            if not np.isfinite(value) or value <= 0:
                raise InferenceError(f"{label} must be finite and positive")

    @property
    def backend_name(self) -> str:
        """Stable backend and method identifier."""

        return f"scipy.optimize.minimize/{self.method}"

    def run(self, problem: OptimizationProblem) -> OptimizationRun:
        """Optimize every declared start and retain the complete outcomes."""

        if not isinstance(problem, OptimizationProblem):
            raise TypeError("problem must be an OptimizationProblem")
        attempts: list[OptimizationAttempt] = []
        bounds = list(problem.parameter_space.optimizer_bounds)
        for index, start in enumerate(problem.starts):
            result = minimize(
                problem.evaluate,
                start,
                method=self.method,
                jac=problem.has_gradient,
                bounds=bounds,
                options={
                    "maxiter": self.max_iterations,
                    "ftol": self.function_tolerance,
                    "gtol": self.gradient_tolerance,
                },
            )
            estimate = np.asarray(result.x, dtype=np.float64)
            objective = float(result.fun) if not np.isnan(result.fun) else np.inf
            gradient_norm = _result_gradient_norm(result.jac)
            converged = bool(
                result.success and np.isfinite(objective) and np.all(np.isfinite(estimate))
            )
            attempts.append(
                OptimizationAttempt(
                    index=index,
                    start=start,
                    estimate=estimate,
                    objective=objective,
                    converged=converged,
                    status=int(result.status),
                    message=str(result.message),
                    n_iterations=int(getattr(result, "nit", 0)),
                    n_evaluations=int(getattr(result, "nfev", 0)),
                    n_gradient_evaluations=int(getattr(result, "njev", 0)),
                    gradient_norm=gradient_norm,
                )
            )
        selected_index = _selected_attempt_index(attempts)
        return OptimizationRun(
            backend=self.backend_name,
            backend_config={
                "method": self.method,
                "max_iterations": self.max_iterations,
                "function_tolerance": self.function_tolerance,
                "gradient_tolerance": self.gradient_tolerance,
            },
            problem=problem.to_dict(),
            attempts=tuple(attempts),
            selected_index=selected_index,
        )


@dataclass(frozen=True, slots=True)
class PyBADSMultistart:
    """Optional deterministic-seed PyBADS backend using the common run contract.

    ``random_seed`` is entropy for :class:`numpy.random.SeedSequence`, not the 32-bit word
    BADS itself consumes: it is any non-negative integer, and each attempt receives its own
    derived unsigned 32-bit seed. That is what lets a seed emitted by :mod:`behavio.sbc`,
    :mod:`behavio.recovery`, or :mod:`behavio.model_recovery` be handed straight to this
    backend, and it removes the former ``random_seed + attempt`` overflow ceiling.
    """

    random_seed: int = 0
    max_iterations: int | None = None
    max_function_evaluations: int | None = None
    function_tolerance: float | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.random_seed, bool)
            or not isinstance(self.random_seed, int)
            or self.random_seed < 0
        ):
            raise InferenceError("random_seed must be a non-negative integer")
        for value, label in (
            (self.max_iterations, "max_iterations"),
            (self.max_function_evaluations, "max_function_evaluations"),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise InferenceError(f"{label} must be a positive integer or null")
        if self.function_tolerance is not None and (
            not np.isfinite(self.function_tolerance) or self.function_tolerance <= 0
        ):
            raise InferenceError("function_tolerance must be finite and positive or null")

    @property
    def backend_name(self) -> str:
        """Stable backend and algorithm identifier."""

        return "pybads/BADS"

    def run(self, problem: OptimizationProblem) -> OptimizationRun:
        """Run one independently seeded BADS search for every declared start."""

        if not isinstance(problem, OptimizationProblem):
            raise TypeError("problem must be an OptimizationProblem")
        attempt_seeds = np.random.SeedSequence(self.random_seed).generate_state(
            len(problem.starts), dtype=np.uint32
        )
        bads_class, version = _load_pybads()
        lower, upper, plausible_lower, plausible_upper = _pybads_bounds(problem)
        attempts: list[OptimizationAttempt] = []
        for index, start in enumerate(problem.starts):
            seed = int(attempt_seeds[index])
            options: dict[str, Any] = {
                "display": "off",
                "random_seed": seed,
                "uncertainty_handling": False,
            }
            if self.max_iterations is not None:
                options["max_iter"] = self.max_iterations
            if self.max_function_evaluations is not None:
                options["max_fun_evals"] = self.max_function_evaluations
            if self.function_tolerance is not None:
                options["tol_fun"] = self.function_tolerance

            problem.evaluate(start)
            bads = None
            random_state = np.random.get_state()
            try:
                bads = bads_class(
                    lambda vector: _value_only(problem.evaluate(vector)),
                    x0=np.asarray(start),
                    lower_bounds=lower,
                    upper_bounds=upper,
                    plausible_lower_bounds=plausible_lower,
                    plausible_upper_bounds=plausible_upper,
                    options=options,
                )
                result = bads.optimize()
                estimate = np.asarray(result["x"], dtype=np.float64).reshape(-1)
                objective = float(result["fval"])
                message = str(result.get("message", ""))
                limit_reached = "reached maximum number" in message.lower()
                finite = np.isfinite(objective) and np.all(np.isfinite(estimate))
                converged = bool(result.get("success", False) and finite and not limit_reached)
                attempts.append(
                    OptimizationAttempt(
                        index=index,
                        start=start,
                        estimate=estimate,
                        objective=objective if not np.isnan(objective) else np.inf,
                        converged=converged,
                        status=0 if converged else 1,
                        message=message,
                        n_iterations=max(0, int(result.get("iterations", 0))),
                        n_evaluations=max(0, int(result.get("func_count", 0))),
                        n_gradient_evaluations=0,
                        gradient_norm=np.inf,
                    )
                )
            except Exception as error:
                function_logger = getattr(bads, "function_logger", None)
                attempts.append(
                    OptimizationAttempt(
                        index=index,
                        start=start,
                        estimate=start,
                        objective=np.inf,
                        converged=False,
                        status=-1,
                        message=f"PyBADS failed: {type(error).__name__}: {error}",
                        n_iterations=0,
                        n_evaluations=max(0, int(getattr(function_logger, "func_count", 0))),
                        n_gradient_evaluations=0,
                        gradient_norm=np.inf,
                    )
                )
            finally:
                np.random.set_state(random_state)
        return OptimizationRun(
            backend=self.backend_name,
            backend_config={
                "version": version,
                "random_seed": self.random_seed,
                "max_iterations": self.max_iterations,
                "max_function_evaluations": self.max_function_evaluations,
                "function_tolerance": self.function_tolerance,
                "uncertainty_handling": False,
            },
            problem=problem.to_dict(),
            attempts=tuple(attempts),
            selected_index=_selected_attempt_index(attempts),
        )


def _load_pybads() -> tuple[type[Any], str]:
    try:
        module = importlib.import_module("pybads")
        bads_class = module.BADS
        version = importlib.metadata.version("pybads")
    except (ImportError, AttributeError, importlib.metadata.PackageNotFoundError) as error:
        raise PyBADSUnavailableError(
            "PyBADS optimization requires optional dependencies; install `behavio[optimization]`"
        ) from error
    return bads_class, version


def _pybads_bounds(
    problem: OptimizationProblem,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    optimizer_bounds = problem.parameter_space.optimizer_bounds
    lower = np.asarray([-np.inf if bounds[0] is None else bounds[0] for bounds in optimizer_bounds])
    upper = np.asarray([np.inf if bounds[1] is None else bounds[1] for bounds in optimizer_bounds])
    plausible = problem.parameter_space.optimizer_plausible_bounds
    missing = [
        name
        for name, bounds in zip(problem.parameter_space.optimizer_names, plausible, strict=True)
        if bounds is None or bounds[0] is None or bounds[1] is None
    ]
    if missing:
        raise InferenceError(
            "PyBADS requires two finite plausible bounds for every free parameter; "
            f"missing={missing}"
        )
    plausible_lower = np.asarray([bounds[0] for bounds in plausible], dtype=np.float64)
    plausible_upper = np.asarray([bounds[1] for bounds in plausible], dtype=np.float64)
    for index, start in enumerate(problem.starts):
        if np.any(start < plausible_lower) or np.any(start > plausible_upper):
            raise InferenceError(f"PyBADS start {index} must lie inside plausible bounds")
    return lower, upper, plausible_lower, plausible_upper


def _value_only(output: ObjectiveOutput) -> float:
    return float(output[0] if isinstance(output, tuple) else output)


def _selected_attempt_index(attempts: Sequence[OptimizationAttempt]) -> int | None:
    successful = [item for item in attempts if item.converged and item.finite]
    eligible = successful or [item for item in attempts if item.finite]
    return None if not eligible else min(eligible, key=lambda item: item.objective).index


def _objective_value(value: Any) -> float:
    if isinstance(value, bool):
        raise InferenceError("objective value must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise InferenceError("objective value must be numeric") from None
    if np.isnan(result):
        raise InferenceError("objective value must not be NaN")
    return result


def _non_nan_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise InferenceError(f"{label} must be numeric") from None
    if np.isnan(result):
        raise InferenceError(f"{label} must not be NaN")
    return result


def _result_gradient_norm(value: Any) -> float:
    if value is None:
        return np.inf
    try:
        gradient = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return np.inf
    if gradient.ndim != 1 or not np.all(np.isfinite(gradient)):
        return np.inf
    return float(np.linalg.norm(gradient))


def _protected_vector(
    values: Sequence[float] | NDArray[np.floating[Any]], *, finite: bool = True
) -> NDArray[np.float64]:
    result = np.array(values, dtype=np.float64, copy=True)
    if finite and not np.all(np.isfinite(result)):
        raise InferenceError("vector must contain finite values")
    result.setflags(write=False)
    return result


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _freeze_record(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_record(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_record(item) for item in value)
    if isinstance(value, float) and not np.isfinite(value):
        raise InferenceError("optimization provenance cannot contain non-finite numbers")
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise InferenceError(f"optimization provenance is not portable: {type(value).__name__}")


def _thaw_record(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_record(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_record(item) for item in value]
    return value
