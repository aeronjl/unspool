"""Smooth longitudinal trajectories for Wiener drift-diffusion parameters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from unspool.models.base import (
    FitDiagnostics,
    FitResult,
    ModelDataError,
    Prediction,
    PredictionMode,
    UnsupportedPredictionMode,
    _protected_array,
)
from unspool.models.ddm import (
    _LOG_DENSITY_FLOOR,
    DriftDiffusionFitResult,
    DriftDiffusionParameters,
    DriftDiffusionSimulation,
    WienerDriftDiffusion,
    _numerical_hessian,
    _upper_boundary_probability,
    _wiener_log_density,
)
from unspool.models.glm import _format_time, _linear_time_basis
from unspool.study import Study


@dataclass(frozen=True, slots=True)
class DriftDiffusionTrajectory:
    """Natural-scale Wiener parameter values along one explicit clock."""

    clock: str
    times: NDArray[np.float64]
    parameter_names: tuple[str, ...]
    values: NDArray[np.float64]

    def __post_init__(self) -> None:
        times = _protected_array(self.times, dtype=np.float64)
        names = tuple(self.parameter_names)
        values = _protected_array(self.values, dtype=np.float64)
        if not isinstance(self.clock, str) or not self.clock:
            raise ValueError("trajectory clock must be a non-empty string")
        if times.ndim != 1 or not np.all(np.isfinite(times)):
            raise ValueError("trajectory times must be a finite one-dimensional array")
        if not names or len(set(names)) != len(names):
            raise ValueError("trajectory parameter names must be non-empty and unique")
        if values.shape != (len(times), len(names)) or not np.all(np.isfinite(values)):
            raise ValueError("trajectory values must align with times and parameter names")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "parameter_names", names)
        object.__setattr__(self, "values", values)

    def path(self, parameter: str) -> NDArray[np.float64]:
        """Return one protected parameter path by name."""

        try:
            column = self.parameter_names.index(parameter)
        except ValueError:
            raise KeyError(parameter) from None
        return _protected_array(self.values[:, column], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class SmoothDriftDiffusionFitResult(DriftDiffusionFitResult):
    """Smooth Wiener fit with retained trajectory coordinates."""

    clock: str
    knots: tuple[float, ...]
    varying_parameters: tuple[str, ...]
    smoothness: float

    def __post_init__(self) -> None:
        DriftDiffusionFitResult.__post_init__(self)
        knots = tuple(float(knot) for knot in self.knots)
        varying = tuple(self.varying_parameters)
        if not isinstance(self.clock, str) or not self.clock:
            raise ValueError("fit clock must be a non-empty string")
        if len(knots) < 2 or not np.all(np.isfinite(knots)):
            raise ValueError("fit knots must contain at least two finite values")
        if any(right <= left for left, right in pairwise(knots)):
            raise ValueError("fit knots must be strictly increasing")
        if not varying or len(set(varying)) != len(varying):
            raise ValueError("varying_parameters must be non-empty and unique")
        if not np.isfinite(self.smoothness) or self.smoothness <= 0:
            raise ValueError("smoothness must be finite and positive")
        object.__setattr__(self, "knots", knots)
        object.__setattr__(self, "varying_parameters", varying)


@dataclass(frozen=True, slots=True)
class SmoothWienerDriftDiffusion(WienerDriftDiffusion):
    """A Wiener model with smooth parameter paths over an explicit clock.

    Drift coefficients, boundary, and starting bias may vary between fixed knots.
    Non-decision time remains stationary. Paths are single-subject by default; a shared
    multi-subject path requires an explicit opt-in.
    """

    knots: tuple[float, ...] = (0.0, 1.0)
    time: str = "session_order"
    smoothness: float = 10.0
    varying_parameters: tuple[str, ...] | None = None
    shared_trajectory: bool = False

    def __post_init__(self) -> None:
        WienerDriftDiffusion.__post_init__(self)
        if self.contaminant is not None:
            raise ValueError("SmoothWienerDriftDiffusion does not yet support contaminants")
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
        if self.time in self.scored_columns or self.time in self.covariates:
            raise ValueError("time must be distinct from scored and covariate columns")
        if not np.isfinite(self.smoothness) or self.smoothness <= 0:
            raise ValueError("smoothness must be finite and positive")
        if not isinstance(self.shared_trajectory, bool):
            raise ValueError("shared_trajectory must be boolean")
        if self.varying_parameters is None:
            varying = (*self.coefficient_names, "boundary", "starting_bias")
        else:
            varying = tuple(self.varying_parameters)
        allowed = set(self.base_parameter_names) - {"nondecision_time"}
        if not varying or len(set(varying)) != len(varying):
            raise ValueError("varying_parameters must be non-empty and unique")
        if set(varying) - allowed:
            raise ValueError(
                "varying_parameters may contain drift coefficients, boundary, and starting_bias"
            )
        object.__setattr__(self, "knots", knots)
        object.__setattr__(self, "varying_parameters", varying)

    @property
    def model_name(self) -> str:
        return "smooth-wiener-drift-diffusion"

    @property
    def signature(self) -> str:
        covariates = ",".join(self.covariates)
        knots = ",".join(_format_time(knot) for knot in self.knots)
        varying = ",".join(self.varying_parameters or ())
        nondecision = (
            "data-constrained"
            if self.nondecision_time_bounds is None
            else str(self.nondecision_time_bounds)
        )
        return (
            f"{self.model_name}[outcome={self.outcome};response_time="
            f"{self.response_time.column}:{self.response_time.unit.value};"
            f"covariates={covariates};diffusion_scale=1;density_terms={self.density_terms};"
            f"simulation_dt={self.simulation_time_step};time={self.time};knots={knots};"
            f"varying={varying};smoothness={self.smoothness};"
            f"nondecision_bounds={nondecision};shared_trajectory={self.shared_trajectory}]"
        )

    @property
    def base_parameter_names(self) -> tuple[str, ...]:
        return (
            *self.coefficient_names,
            "boundary",
            "starting_bias",
            "nondecision_time",
        )

    @property
    def parameter_names(self) -> tuple[str, ...]:
        names: list[str] = []
        varying = set(self.varying_parameters or ())
        for parameter in self.base_parameter_names:
            if parameter in varying:
                names.extend(
                    f"{parameter}[{self.time}={_format_time(knot)}]" for knot in self.knots
                )
            else:
                names.append(parameter)
        return tuple(names)

    def parameters_from_paths(
        self,
        paths: Mapping[str, float | Sequence[float]],
    ) -> Mapping[str, float]:
        """Pack natural-scale varying paths and stationary values."""

        expected = set(self.base_parameter_names)
        observed = set(paths)
        if observed != expected:
            raise ValueError(
                "paths must match base_parameter_names exactly; "
                f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
            )
        packed: dict[str, float] = {}
        varying = set(self.varying_parameters or ())
        parameter_name_offset = 0
        for parameter in self.base_parameter_names:
            if parameter in varying:
                try:
                    values = np.asarray(paths[parameter], dtype=np.float64)
                except (TypeError, ValueError):
                    raise ValueError(
                        f"path {parameter!r} must contain one finite value per knot"
                    ) from None
                if values.shape != (len(self.knots),) or not np.all(np.isfinite(values)):
                    raise ValueError(f"path {parameter!r} must contain one finite value per knot")
            else:
                try:
                    values = np.asarray([paths[parameter]], dtype=np.float64)
                except (TypeError, ValueError):
                    raise ValueError(f"stationary value {parameter!r} must be finite") from None
                if values.shape != (1,) or not np.all(np.isfinite(values)):
                    raise ValueError(f"stationary value {parameter!r} must be finite")
            self._validate_natural_values(parameter, values)
            for value in values:
                packed[self.parameter_names[parameter_name_offset]] = float(value)
                parameter_name_offset += 1
        return MappingProxyType(packed)

    def parameters_from_components(
        self,
        *,
        drift: Mapping[str, float],
        boundary: float,
        starting_bias: float = 0.5,
        nondecision_time: float = 0.0,
        contaminant_probability: float = 0.0,
    ) -> Mapping[str, float]:
        """Construct a stationary path in the smooth model's coordinates."""

        if contaminant_probability != 0:
            raise ValueError("SmoothWienerDriftDiffusion does not support contaminants")
        if set(drift) != set(self.coefficient_names):
            raise ValueError("drift must match coefficient_names exactly")
        values: dict[str, float | Sequence[float]] = {
            parameter: (
                [float(drift[parameter])] * len(self.knots)
                if parameter in set(self.varying_parameters or ())
                else float(drift[parameter])
            )
            for parameter in self.coefficient_names
        }
        values["boundary"] = self._path_or_scalar("boundary", boundary)
        values["starting_bias"] = self._path_or_scalar("starting_bias", starting_bias)
        values["nondecision_time"] = float(nondecision_time)
        return self.parameters_from_paths(values)

    def parameter_components(
        self,
        parameters: Mapping[str, float] | FitResult,
    ) -> DriftDiffusionParameters:
        """Reject collapse of a trajectory to one undocumented parameter vector."""

        raise TypeError("use parameter_trajectory() for a SmoothWienerDriftDiffusion fit")

    def parameter_trajectory(
        self,
        fit: FitResult,
        *,
        times: Sequence[float] | None = None,
    ) -> DriftDiffusionTrajectory:
        """Evaluate every natural-scale parameter at requested clock values."""

        self._validate_fit(fit)
        evaluation_times = self.knots if times is None else times
        time_array = np.asarray(evaluation_times, dtype=np.float64)
        basis = _linear_time_basis(time_array, self.knots)
        knot_values = self._unpack_knot_values(fit.estimates)
        return DriftDiffusionTrajectory(
            clock=self.time,
            times=time_array,
            parameter_names=self.base_parameter_names,
            values=basis @ knot_values.T,
        )

    def simulate(
        self,
        design: Study,
        parameters: Mapping[str, float],
        *,
        seed: int | np.random.Generator,
    ) -> Study:
        """Simulate trialwise Wiener paths under longitudinal parameter trajectories."""

        self._validate_study_scope(design)
        vector = self._parameter_vector(parameters)
        trial_values = self._trial_parameter_values(design, vector)
        features = self._feature_matrix(design)
        n_coefficients = len(self.coefficient_names)
        drifts = np.sum(features * trial_values[:, :n_coefficients], axis=1)
        boundary = trial_values[:, n_coefficients]
        starting_bias = trial_values[:, n_coefficients + 1]
        nondecision_time = trial_values[:, n_coefficients + 2]
        generator = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        choices, response_seconds = _simulate_trialwise_wiener(
            drifts,
            boundary,
            starting_bias,
            nondecision_time,
            generator=generator,
            time_step=self.simulation_time_step,
            maximum_time=self.simulation_max_time,
        )
        columns = {name: design[name] for name in design.columns}
        columns[self.outcome] = choices
        columns[self.response_time.column] = (
            response_seconds / self.response_time.unit.seconds_per_unit
        )
        return Study(columns)

    def simulate_with_contaminants(
        self,
        design: Study,
        parameters: Mapping[str, float],
        *,
        seed: int | np.random.Generator,
    ) -> DriftDiffusionSimulation:
        """Return an all-false contaminant mask for API symmetry with the static model."""

        study = self.simulate(design, parameters, seed=seed)
        return DriftDiffusionSimulation(
            study=study,
            contaminants=np.zeros(len(study), dtype=np.bool_),
        )

    def fit(self, study: Study) -> SmoothDriftDiffusionFitResult:
        """Fit smooth natural-scale paths with deterministic bounded restarts."""

        self._validate_study_scope(study)
        outcomes = self._outcomes(study)
        response_times = self.response_time.read(study).seconds
        features = self._feature_matrix(study)
        basis = self._time_basis(study)
        minimum_response_time = float(np.min(response_times))
        nondecision_bounds = self._fit_nondecision_time_bounds(minimum_response_time)
        bounds = self._optimization_bounds(nondecision_bounds)
        penalty = self._penalty_matrix()

        def objective(values: NDArray[np.float64]) -> float:
            trial_values = self._trial_parameter_values_from_basis(basis, values)
            n_coefficients = len(self.coefficient_names)
            decision_times = response_times - trial_values[:, n_coefficients + 2]
            if np.any(decision_times <= 0):
                return float(np.finfo(np.float64).max / 1e100)
            drifts = np.sum(features * trial_values[:, :n_coefficients], axis=1)
            log_density = _wiener_log_density(
                decision_times,
                outcomes,
                drifts,
                boundary=trial_values[:, n_coefficients],
                starting_bias=trial_values[:, n_coefficients + 1],
                terms=self.density_terms,
            )
            roughness = 0.5 * float(values @ penalty @ values)
            return float(-np.sum(log_density) + roughness)

        static_starts = self._initial_points(outcomes, response_times, nondecision_bounds)
        starts = tuple(self._expand_static_vector(start) for start in static_starts)
        results = [
            minimize(
                objective,
                start,
                method="L-BFGS-B",
                bounds=bounds,
                options={
                    "maxiter": self.max_iterations,
                    "ftol": self.tolerance,
                    "gtol": self.tolerance,
                    "maxls": 50,
                },
            )
            for start in starts
        ]
        restart_objectives = np.asarray(
            [float(result.fun) if np.isfinite(result.fun) else np.inf for result in results]
        )
        finite = np.flatnonzero(np.isfinite(restart_objectives)).tolist()
        if not finite:
            messages = "; ".join(str(result.message) for result in results)
            raise ModelDataError(
                f"all smooth DDM restarts produced non-finite objectives: {messages}"
            )
        successful = [index for index in finite if results[index].success]
        eligible = successful if successful else finite
        selected = min(eligible, key=lambda index: float(restart_objectives[index]))
        chosen = results[selected]
        estimates = np.asarray(chosen.x, dtype=np.float64)
        value = objective(estimates)
        hessian = _numerical_hessian(objective, estimates, bounds)
        condition = float(np.linalg.cond(hessian))
        covariance = np.linalg.pinv(hessian, hermitian=True)
        standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        trial_values = self._trial_parameter_values_from_basis(basis, estimates)
        n_coefficients = len(self.coefficient_names)
        decision_times = response_times - trial_values[:, n_coefficients + 2]
        drifts = np.sum(features * trial_values[:, :n_coefficients], axis=1)
        log_density = _wiener_log_density(
            decision_times,
            outcomes,
            drifts,
            boundary=trial_values[:, n_coefficients],
            starting_bias=trial_values[:, n_coefficients + 1],
            terms=self.density_terms,
        )
        floor_count = int(np.sum(log_density <= _LOG_DENSITY_FLOOR))
        gradient = np.asarray(chosen.jac, dtype=np.float64)
        diagnostics = FitDiagnostics(
            converged=bool(chosen.success),
            optimizer=f"L-BFGS-B ({self.n_restarts} deterministic restarts)",
            status=int(chosen.status),
            message=str(chosen.message),
            n_iterations=int(chosen.nit),
            objective=float(value),
            gradient_norm=float(np.linalg.norm(gradient)),
            hessian_condition=condition,
            boundary_estimate=self._trajectory_boundary_warning(
                estimates,
                bounds,
                minimum_response_time=minimum_response_time,
                floor_count=floor_count,
            ),
        )
        return SmoothDriftDiffusionFitResult(
            model_name=self.model_name,
            model_signature=self.signature,
            parameter_names=self.parameter_names,
            estimates=estimates,
            standard_errors=standard_errors,
            covariance=covariance,
            n_observations=len(study),
            diagnostics=diagnostics,
            restart_objectives=restart_objectives,
            restart_converged=np.asarray([result.success for result in results]),
            restart_messages=tuple(str(result.message) for result in results),
            selected_restart=selected,
            minimum_observed_response_time=minimum_response_time,
            likelihood_floor_count=floor_count,
            posterior_contaminant_probability=np.zeros(len(study), dtype=np.float64),
            clock=self.time,
            knots=self.knots,
            varying_parameters=tuple(self.varying_parameters or ()),
            smoothness=self.smoothness,
        )

    def predict(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> Prediction:
        """Return marginal choices under the fitted longitudinal paths."""

        prediction_mode = PredictionMode(mode)
        if prediction_mode not in self.supported_prediction_modes:
            raise UnsupportedPredictionMode(
                "SmoothWienerDriftDiffusion supports only filtered prediction"
            )
        self._validate_study_scope(study)
        self._validate_fit(fit)
        trial_values = self._trial_parameter_values(study, fit.estimates)
        n_coefficients = len(self.coefficient_names)
        drifts = np.sum(
            self._feature_matrix(study) * trial_values[:, :n_coefficients],
            axis=1,
        )
        probability = _upper_boundary_probability(
            drifts,
            boundary=trial_values[:, n_coefficients],
            starting_bias=trial_values[:, n_coefficients + 1],
        )
        probability = np.clip(probability, np.finfo(float).tiny, 1.0 - np.finfo(float).eps)
        return Prediction(
            probability=probability,
            linear_predictor=np.log(probability) - np.log1p(-probability),
            mode=prediction_mode,
        )

    def pointwise_log_prob(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> NDArray[np.float64]:
        """Return joint choice/RT log densities under fitted paths."""

        prediction_mode = PredictionMode(mode)
        if prediction_mode not in self.supported_prediction_modes:
            raise UnsupportedPredictionMode(
                "SmoothWienerDriftDiffusion supports only filtered prediction"
            )
        self._validate_study_scope(study)
        self._validate_fit(fit)
        outcomes = self._outcomes(study)
        response_times = self.response_time.read(study).seconds
        trial_values = self._trial_parameter_values(study, fit.estimates)
        n_coefficients = len(self.coefficient_names)
        drifts = np.sum(
            self._feature_matrix(study) * trial_values[:, :n_coefficients],
            axis=1,
        )
        scores = _wiener_log_density(
            response_times - trial_values[:, n_coefficients + 2],
            outcomes,
            drifts,
            boundary=trial_values[:, n_coefficients],
            starting_bias=trial_values[:, n_coefficients + 1],
            terms=self.density_terms,
        )
        return _protected_array(scores, dtype=np.float64)

    def contaminant_responsibility(
        self,
        study: Study,
        fit: FitResult,
    ) -> NDArray[np.float64]:
        """Return zeros because this first smooth family has no contaminant component."""

        self._validate_study_scope(study)
        self._validate_fit(fit)
        return _protected_array(np.zeros(len(study)), dtype=np.float64)

    def _parameter_vector(self, parameters: Mapping[str, float]) -> NDArray[np.float64]:
        expected = set(self.parameter_names)
        observed = set(parameters)
        if observed != expected:
            raise ValueError(
                "parameters must match the model exactly; "
                f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
            )
        try:
            values = np.asarray(
                [parameters[name] for name in self.parameter_names],
                dtype=np.float64,
            )
        except (TypeError, ValueError):
            raise ValueError("parameters must contain finite numeric values") from None
        if not np.all(np.isfinite(values)):
            raise ValueError("parameters must contain finite numeric values")
        self._validate_vector_natural_values(values)
        return values

    def _unpack_knot_values(self, vector: NDArray[np.float64]) -> NDArray[np.float64]:
        knot_values = np.empty(
            (len(self.base_parameter_names), len(self.knots)),
            dtype=np.float64,
        )
        varying = set(self.varying_parameters or ())
        offset = 0
        for row, parameter in enumerate(self.base_parameter_names):
            if parameter in varying:
                knot_values[row] = vector[offset : offset + len(self.knots)]
                offset += len(self.knots)
            else:
                knot_values[row] = vector[offset]
                offset += 1
        if offset != len(vector):
            raise ValueError("parameter vector does not match the smooth model")
        return knot_values

    def _trial_parameter_values(
        self,
        study: Study,
        vector: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return self._trial_parameter_values_from_basis(self._time_basis(study), vector)

    def _trial_parameter_values_from_basis(
        self,
        basis: NDArray[np.float64],
        vector: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return basis @ self._unpack_knot_values(vector).T

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

    def _optimization_bounds(
        self,
        nondecision_bounds: tuple[float, float],
    ) -> list[tuple[float, float]]:
        bounds: list[tuple[float, float]] = []
        varying = set(self.varying_parameters or ())
        for parameter in self.base_parameter_names:
            parameter_bounds = self._natural_bounds(parameter, nondecision_bounds)
            repeats = len(self.knots) if parameter in varying else 1
            bounds.extend([parameter_bounds] * repeats)
        return bounds

    def _penalty_matrix(self) -> NDArray[np.float64]:
        penalty = np.zeros((len(self.parameter_names), len(self.parameter_names)))
        varying = set(self.varying_parameters or ())
        roughness = self.smoothness * _roughness_matrix(self.knots)
        offset = 0
        for parameter in self.base_parameter_names:
            if parameter in varying:
                stop = offset + len(self.knots)
                penalty[offset:stop, offset:stop] = roughness
                offset = stop
            else:
                offset += 1
        return penalty

    def _expand_static_vector(self, static: NDArray[np.float64]) -> NDArray[np.float64]:
        expanded: list[float] = []
        varying = set(self.varying_parameters or ())
        for parameter, value in zip(self.base_parameter_names, static, strict=True):
            repeats = len(self.knots) if parameter in varying else 1
            expanded.extend([float(value)] * repeats)
        return np.asarray(expanded, dtype=np.float64)

    def _validate_vector_natural_values(self, vector: NDArray[np.float64]) -> None:
        knot_values = self._unpack_knot_values(vector)
        for parameter, values in zip(self.base_parameter_names, knot_values, strict=True):
            self._validate_natural_values(parameter, values)

    def _validate_natural_values(
        self,
        parameter: str,
        values: NDArray[np.float64],
    ) -> None:
        if parameter.startswith("drift."):
            valid = np.all(np.abs(values) <= self.drift_bound)
        elif parameter == "boundary":
            valid = np.all(
                (values >= self.boundary_bounds[0]) & (values <= self.boundary_bounds[1])
            )
        elif parameter == "starting_bias":
            valid = np.all(
                (values >= self.starting_bias_bounds[0]) & (values <= self.starting_bias_bounds[1])
            )
        elif parameter == "nondecision_time":
            valid = np.all(values >= 0)
            if self.nondecision_time_bounds is not None:
                valid &= np.all(
                    (values >= self.nondecision_time_bounds[0])
                    & (values <= self.nondecision_time_bounds[1])
                )
        else:
            raise ValueError(f"unknown base parameter {parameter!r}")
        if not valid:
            raise ValueError(f"values for {parameter!r} lie outside the model bounds")

    def _natural_bounds(
        self,
        parameter: str,
        nondecision_bounds: tuple[float, float],
    ) -> tuple[float, float]:
        if parameter.startswith("drift."):
            return -self.drift_bound, self.drift_bound
        if parameter == "boundary":
            return self.boundary_bounds
        if parameter == "starting_bias":
            return self.starting_bias_bounds
        if parameter == "nondecision_time":
            return nondecision_bounds
        raise ValueError(f"unknown base parameter {parameter!r}")

    def _path_or_scalar(self, parameter: str, value: float) -> float | Sequence[float]:
        if parameter in set(self.varying_parameters or ()):
            return [float(value)] * len(self.knots)
        return float(value)

    def _trajectory_boundary_warning(
        self,
        estimates: NDArray[np.float64],
        bounds: Sequence[tuple[float, float]],
        *,
        minimum_response_time: float,
        floor_count: int,
    ) -> bool:
        near_bound = False
        for value, (lower, upper) in zip(estimates, bounds, strict=True):
            tolerance = 1e-4 * max(1.0, upper - lower)
            near_bound |= value - lower <= tolerance or upper - value <= tolerance
        knot_values = self._unpack_knot_values(estimates)
        nondecision = knot_values[self.base_parameter_names.index("nondecision_time"), 0]
        near_rt = minimum_response_time - nondecision <= 5 * self.minimum_decision_time
        return bool(near_bound or near_rt or floor_count > 0)

    def _validate_study_scope(self, study: Study) -> None:
        if len(study.subjects) > 1 and not self.shared_trajectory:
            raise ModelDataError(
                "smooth DDM paths are subject-specific by default; fit one subject at a "
                "time or set shared_trajectory=True to align subjects explicitly"
            )


def _roughness_matrix(knots: tuple[float, ...]) -> NDArray[np.float64]:
    differences = np.zeros((len(knots) - 1, len(knots)), dtype=np.float64)
    for row, spacing in enumerate(np.diff(knots)):
        scale = 1.0 / np.sqrt(spacing)
        differences[row, row] = -scale
        differences[row, row + 1] = scale
    return differences.T @ differences


def _simulate_trialwise_wiener(
    drift: NDArray[np.float64],
    boundary: NDArray[np.float64],
    starting_bias: NDArray[np.float64],
    nondecision_time: NDArray[np.float64],
    *,
    generator: np.random.Generator,
    time_step: float,
    maximum_time: float,
) -> tuple[NDArray[np.int8], NDArray[np.float64]]:
    n_rows = len(drift)
    positions = boundary * starting_bias
    decision_times = np.zeros(n_rows, dtype=np.float64)
    choices = np.zeros(n_rows, dtype=np.int8)
    active = np.ones(n_rows, dtype=np.bool_)
    noise_scale = np.sqrt(time_step)
    max_steps = int(np.ceil(maximum_time / time_step))
    for step in range(max_steps):
        active_indices = np.flatnonzero(active)
        if not len(active_indices):
            break
        previous = positions[active_indices]
        current = previous + drift[active_indices] * time_step
        current += generator.normal(0.0, noise_scale, len(active_indices))
        positions[active_indices] = current
        upper = current >= boundary[active_indices]
        lower = current <= 0.0
        crossed = upper | lower
        if not np.any(crossed):
            continue
        crossed_indices = active_indices[crossed]
        previous_crossed = previous[crossed]
        current_crossed = current[crossed]
        upper_crossed = upper[crossed]
        target = np.where(upper_crossed, boundary[crossed_indices], 0.0)
        fraction = (target - previous_crossed) / (current_crossed - previous_crossed)
        fraction = np.clip(fraction, 0.0, 1.0)
        decision_times[crossed_indices] = (step + fraction) * time_step
        choices[crossed_indices] = upper_crossed.astype(np.int8)
        active[crossed_indices] = False
    if np.any(active):
        raise RuntimeError(
            f"{int(np.sum(active))} simulated paths did not terminate within "
            f"{maximum_time:g} seconds"
        )
    return choices, decision_times + nondecision_time
