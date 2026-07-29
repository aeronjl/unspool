"""A simple Wiener drift-diffusion model for binary choice and response time."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import logsumexp

from unspool.models.base import (
    FitDiagnostics,
    FitResult,
    ModelDataError,
    Prediction,
    PredictionMode,
    UnsupportedPredictionMode,
    _protected_array,
)
from unspool.response_times import ResponseTimeSpec
from unspool.study import REQUIRED_COLUMNS, Study

_LOG_DENSITY_FLOOR = float(np.log(np.finfo(np.float64).tiny))
_BRIDGE_EXPONENT_LIMIT = 30.0


@dataclass(frozen=True, slots=True)
class UniformResponseTimeContaminant:
    """A fixed-support joint choice/RT contaminant distribution.

    Response time is uniform over ``time_bounds`` in canonical seconds and choice is an
    independent Bernoulli draw with the fixed upper-boundary probability
    ``choice_probability``. Only the mixture probability is fitted.
    """

    time_bounds: tuple[float, float]
    probability_bounds: tuple[float, float] = (0.0, 0.25)
    choice_probability: float = 0.5

    def __post_init__(self) -> None:
        time_bounds = _ordered_bounds(self.time_bounds, "contaminant time_bounds")
        probability_bounds = _probability_bounds(
            self.probability_bounds,
            "contaminant probability_bounds",
        )
        if not np.isfinite(self.choice_probability) or not 0 < self.choice_probability < 1:
            raise ValueError(
                "contaminant choice_probability must lie strictly between zero and one"
            )
        object.__setattr__(self, "time_bounds", time_bounds)
        object.__setattr__(self, "probability_bounds", probability_bounds)

    def pointwise_log_density(
        self,
        response_time_seconds: NDArray[np.float64],
        choice: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return the normalized joint contaminant log density."""

        response_times, choices = np.broadcast_arrays(
            np.asarray(response_time_seconds, dtype=np.float64),
            np.asarray(choice, dtype=np.float64),
        )
        lower, upper = self.time_bounds
        result = np.full(response_times.shape, -np.inf, dtype=np.float64)
        valid = (
            np.isfinite(response_times)
            & (response_times >= lower)
            & (response_times <= upper)
            & ((choices == 0) | (choices == 1))
        )
        if np.any(valid):
            log_choice = np.where(
                choices[valid] == 1,
                np.log(self.choice_probability),
                np.log1p(-self.choice_probability),
            )
            result[valid] = log_choice - np.log(upper - lower)
        return result


@dataclass(frozen=True, slots=True)
class DriftDiffusionParameters:
    """Natural-scale parameters of the fixed-parameter Wiener model."""

    drift_coefficients: NDArray[np.float64]
    coefficient_names: tuple[str, ...]
    boundary: float
    starting_bias: float
    nondecision_time: float
    contaminant_probability: float = 0.0

    def __post_init__(self) -> None:
        coefficients = _protected_array(self.drift_coefficients, dtype=np.float64)
        names = tuple(self.coefficient_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("coefficient_names must be non-empty and unique")
        if coefficients.shape != (len(names),) or not np.all(np.isfinite(coefficients)):
            raise ValueError("drift coefficients must be finite and match coefficient_names")
        if not np.isfinite(self.boundary) or self.boundary <= 0:
            raise ValueError("boundary must be finite and positive")
        if not np.isfinite(self.starting_bias) or not 0 < self.starting_bias < 1:
            raise ValueError("starting_bias must lie strictly between zero and one")
        if not np.isfinite(self.nondecision_time) or self.nondecision_time < 0:
            raise ValueError("nondecision_time must be finite and non-negative")
        if (
            not np.isfinite(self.contaminant_probability)
            or not 0 <= self.contaminant_probability < 1
        ):
            raise ValueError("contaminant_probability must lie between zero inclusive and one")
        object.__setattr__(self, "drift_coefficients", coefficients)
        object.__setattr__(self, "coefficient_names", names)

    @property
    def drift_map(self) -> Mapping[str, float]:
        return MappingProxyType(
            dict(zip(self.coefficient_names, self.drift_coefficients.tolist(), strict=True))
        )


@dataclass(frozen=True, slots=True)
class DriftDiffusionSimulation:
    """Observed simulated study paired with unexposed contaminant truth."""

    study: Study
    contaminants: NDArray[np.bool_]

    def __post_init__(self) -> None:
        if not isinstance(self.study, Study):
            raise TypeError("study must be a Study")
        contaminants = _protected_array(self.contaminants, dtype=np.bool_)
        if contaminants.shape != (len(self.study),):
            raise ValueError("contaminants must contain one indicator per study row")
        object.__setattr__(self, "contaminants", contaminants)


@dataclass(frozen=True, slots=True)
class DriftDiffusionFitResult(FitResult):
    """Wiener fit with retained deterministic restart evidence."""

    restart_objectives: NDArray[np.float64]
    restart_converged: NDArray[np.bool_]
    restart_messages: tuple[str, ...]
    selected_restart: int
    minimum_observed_response_time: float
    likelihood_floor_count: int
    posterior_contaminant_probability: NDArray[np.float64]

    def __post_init__(self) -> None:
        FitResult.__post_init__(self)
        objectives = _protected_array(self.restart_objectives, dtype=np.float64)
        converged = _protected_array(self.restart_converged, dtype=np.bool_)
        posterior = _protected_array(
            self.posterior_contaminant_probability,
            dtype=np.float64,
        )
        messages = tuple(self.restart_messages)
        if objectives.ndim != 1 or converged.shape != objectives.shape:
            raise ValueError("restart arrays must be one-dimensional and aligned")
        if len(messages) != len(objectives) or not len(messages):
            raise ValueError("every restart must retain one optimizer message")
        if not 0 <= self.selected_restart < len(messages):
            raise ValueError("selected_restart must identify one retained restart")
        if not np.isfinite(self.minimum_observed_response_time) or (
            self.minimum_observed_response_time <= 0
        ):
            raise ValueError("minimum_observed_response_time must be finite and positive")
        if (
            isinstance(self.likelihood_floor_count, bool)
            or not isinstance(self.likelihood_floor_count, int)
            or self.likelihood_floor_count < 0
        ):
            raise ValueError("likelihood_floor_count must be a non-negative integer")
        if posterior.shape != (self.n_observations,) or not np.all(np.isfinite(posterior)):
            raise ValueError(
                "posterior_contaminant_probability must contain one finite value per observation"
            )
        if np.any((posterior < 0) | (posterior > 1)):
            raise ValueError("posterior contaminant probabilities must lie between zero and one")
        object.__setattr__(self, "restart_objectives", objectives)
        object.__setattr__(self, "restart_converged", converged)
        object.__setattr__(self, "restart_messages", messages)
        object.__setattr__(self, "posterior_contaminant_probability", posterior)

    @property
    def expected_contaminant_count(self) -> float:
        """Return the sum of fitted per-trial contaminant responsibilities."""

        return float(np.sum(self.posterior_contaminant_probability))


@dataclass(frozen=True, slots=True)
class WienerDriftDiffusion:
    """Fixed-parameter two-boundary Wiener diffusion with covariate-dependent drift.

    Diffusion variance is fixed to one. Boundary separation, relative starting bias, and
    non-decision time are shared across trials; drift is a linear function of named numeric
    covariates. The pointwise likelihood is joint over binary choice and response time.
    """

    covariates: tuple[str, ...] = ()
    outcome: str = "choice"
    response_time: ResponseTimeSpec = field(default_factory=ResponseTimeSpec)
    n_restarts: int = 4
    max_iterations: int = 500
    tolerance: float = 1e-8
    density_terms: int = 12
    simulation_time_step: float = 0.0005
    simulation_max_time: float = 20.0
    drift_bound: float = 12.0
    boundary_bounds: tuple[float, float] = (0.1, 5.0)
    starting_bias_bounds: tuple[float, float] = (0.02, 0.98)
    nondecision_time_bounds: tuple[float, float] | None = None
    contaminant: UniformResponseTimeContaminant | None = None
    minimum_decision_time: float = 1e-4

    def __post_init__(self) -> None:
        covariates = tuple(self.covariates)
        if len(set(covariates)) != len(covariates):
            raise ValueError("covariates must be unique")
        if any(not isinstance(name, str) or not name for name in covariates):
            raise ValueError("covariate names must be non-empty strings")
        if not isinstance(self.outcome, str) or not self.outcome:
            raise ValueError("outcome must be a non-empty column name")
        if self.outcome in REQUIRED_COLUMNS or self.outcome in covariates:
            raise ValueError("outcome must be distinct from required and covariate columns")
        if not isinstance(self.response_time, ResponseTimeSpec):
            raise TypeError("response_time must be a ResponseTimeSpec")
        if self.response_time.column == self.outcome or self.response_time.column in covariates:
            raise ValueError("response-time, outcome, and covariate columns must be distinct")
        _positive_integer(self.n_restarts, "n_restarts")
        _positive_integer(self.max_iterations, "max_iterations")
        _positive_integer(self.density_terms, "density_terms")
        if self.density_terms < 8:
            raise ValueError("density_terms must be at least eight")
        for value, name in (
            (self.tolerance, "tolerance"),
            (self.simulation_time_step, "simulation_time_step"),
            (self.simulation_max_time, "simulation_max_time"),
            (self.drift_bound, "drift_bound"),
            (self.minimum_decision_time, "minimum_decision_time"),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        boundary_bounds = _ordered_bounds(self.boundary_bounds, "boundary_bounds")
        starting_bounds = _ordered_bounds(self.starting_bias_bounds, "starting_bias_bounds")
        if not 0 < starting_bounds[0] < starting_bounds[1] < 1:
            raise ValueError("starting_bias_bounds must lie strictly between zero and one")
        nondecision_bounds = self.nondecision_time_bounds
        if nondecision_bounds is not None:
            nondecision_bounds = _nonnegative_bounds(
                nondecision_bounds,
                "nondecision_time_bounds",
            )
        if self.contaminant is not None and not isinstance(
            self.contaminant,
            UniformResponseTimeContaminant,
        ):
            raise TypeError("contaminant must be a UniformResponseTimeContaminant")
        if self.contaminant is not None and nondecision_bounds is None:
            raise ValueError("contaminant-aware fitting requires fixed nondecision_time_bounds")
        if self.simulation_time_step >= self.simulation_max_time:
            raise ValueError("simulation_time_step must be smaller than simulation_max_time")
        object.__setattr__(self, "covariates", covariates)
        object.__setattr__(self, "boundary_bounds", boundary_bounds)
        object.__setattr__(self, "starting_bias_bounds", starting_bounds)
        object.__setattr__(self, "nondecision_time_bounds", nondecision_bounds)

    @property
    def model_name(self) -> str:
        if self.contaminant is not None:
            return "uniform-contaminant-wiener-drift-diffusion"
        return "wiener-drift-diffusion"

    @property
    def signature(self) -> str:
        covariates = ",".join(self.covariates)
        signature = (
            f"{self.model_name}[outcome={self.outcome};response_time="
            f"{self.response_time.column}:{self.response_time.unit.value};"
            f"covariates={covariates};diffusion_scale=1;density_terms={self.density_terms};"
            f"simulation_dt={self.simulation_time_step}"
        )
        if self.nondecision_time_bounds is not None:
            signature += f";nondecision_bounds={self.nondecision_time_bounds}"
        if self.contaminant is not None:
            signature += (
                f";contaminant_time_bounds={self.contaminant.time_bounds};"
                f"contaminant_probability_bounds={self.contaminant.probability_bounds};"
                f"contaminant_choice_probability={self.contaminant.choice_probability}"
            )
        return signature + "]"

    @property
    def coefficient_names(self) -> tuple[str, ...]:
        return ("drift.intercept", *(f"drift.{name}" for name in self.covariates))

    @property
    def parameter_names(self) -> tuple[str, ...]:
        names = (
            *self.coefficient_names,
            "boundary",
            "starting_bias",
            "nondecision_time",
        )
        if self.contaminant is not None:
            return (*names, "contaminant_probability")
        return names

    @property
    def scored_columns(self) -> tuple[str, ...]:
        return (self.outcome, self.response_time.column)

    @property
    def supported_prediction_modes(self) -> tuple[PredictionMode, ...]:
        return (PredictionMode.FILTERED,)

    def parameters_from_components(
        self,
        *,
        drift: Mapping[str, float],
        boundary: float,
        starting_bias: float = 0.5,
        nondecision_time: float = 0.0,
        contaminant_probability: float = 0.0,
    ) -> Mapping[str, float]:
        """Validate and pack natural-scale drift and timing parameters."""

        if set(drift) != set(self.coefficient_names):
            expected = set(self.coefficient_names)
            observed = set(drift)
            raise ValueError(
                "drift must match coefficient_names exactly; "
                f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
            )
        components = DriftDiffusionParameters(
            drift_coefficients=np.asarray(
                [drift[name] for name in self.coefficient_names], dtype=np.float64
            ),
            coefficient_names=self.coefficient_names,
            boundary=float(boundary),
            starting_bias=float(starting_bias),
            nondecision_time=float(nondecision_time),
            contaminant_probability=float(contaminant_probability),
        )
        self._validate_contaminant_probability(components.contaminant_probability)
        values = (
            *components.drift_coefficients.tolist(),
            components.boundary,
            components.starting_bias,
            components.nondecision_time,
        )
        if self.contaminant is not None:
            values = (*values, components.contaminant_probability)
        return MappingProxyType(dict(zip(self.parameter_names, values, strict=True)))

    def parameter_components(
        self,
        parameters: Mapping[str, float] | FitResult,
    ) -> DriftDiffusionParameters:
        """Validate and unpack natural-scale model parameters."""

        if isinstance(parameters, FitResult):
            self._validate_fit(parameters)
            values = parameters.estimates
        else:
            expected = set(self.parameter_names)
            observed = set(parameters)
            if observed != expected:
                raise ValueError(
                    "parameters must match the model exactly; "
                    f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
                )
            try:
                values = np.asarray(
                    [parameters[name] for name in self.parameter_names], dtype=np.float64
                )
            except (TypeError, ValueError):
                raise ValueError("parameters must contain finite numeric values") from None
            if not np.all(np.isfinite(values)):
                raise ValueError("parameters must contain finite numeric values")
        n_coefficients = len(self.coefficient_names)
        components = DriftDiffusionParameters(
            drift_coefficients=values[:n_coefficients],
            coefficient_names=self.coefficient_names,
            boundary=float(values[n_coefficients]),
            starting_bias=float(values[n_coefficients + 1]),
            nondecision_time=float(values[n_coefficients + 2]),
            contaminant_probability=(
                float(values[n_coefficients + 3]) if self.contaminant is not None else 0.0
            ),
        )
        self._validate_contaminant_probability(components.contaminant_probability)
        return components

    def simulate(
        self,
        design: Study,
        parameters: Mapping[str, float],
        *,
        seed: int | np.random.Generator,
    ) -> Study:
        """Simulate observed choices and response times without exposing mixture truth."""

        return self.simulate_with_contaminants(design, parameters, seed=seed).study

    def simulate_with_contaminants(
        self,
        design: Study,
        parameters: Mapping[str, float],
        *,
        seed: int | np.random.Generator,
    ) -> DriftDiffusionSimulation:
        """Simulate observations and retain contaminant indicators separately."""

        components = self.parameter_components(parameters)
        drifts = self._drifts(design, components.drift_coefficients)
        generator = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        n_rows = len(design)
        if components.contaminant_probability > 0:
            contaminants = generator.binomial(
                1,
                components.contaminant_probability,
                n_rows,
            ).astype(np.bool_)
        else:
            contaminants = np.zeros(n_rows, dtype=np.bool_)
        positions = np.full(n_rows, components.boundary * components.starting_bias)
        decision_times = np.zeros(n_rows, dtype=np.float64)
        choices = np.zeros(n_rows, dtype=np.int8)
        active = ~contaminants
        time_step = self.simulation_time_step
        noise_scale = np.sqrt(time_step)
        max_steps = int(np.ceil(self.simulation_max_time / time_step))

        for step in range(max_steps):
            active_indices = np.flatnonzero(active)
            if not len(active_indices):
                break
            previous = positions[active_indices]
            current = previous + drifts[active_indices] * time_step
            current += generator.normal(0.0, noise_scale, len(active_indices))
            positions[active_indices] = current
            upper = current >= components.boundary
            lower = current <= 0.0
            endpoint_crossed = upper | lower
            interior = np.flatnonzero(~endpoint_crossed)
            if len(interior):
                bridge_upper, bridge_lower = _bridge_crossings(
                    previous[interior],
                    current[interior],
                    components.boundary,
                    time_step=time_step,
                    generator=generator,
                )
                upper[interior] = bridge_upper
                lower[interior] = bridge_lower
            crossed = upper | lower
            if not np.any(crossed):
                continue
            crossed_indices = active_indices[crossed]
            upper_crossed = upper[crossed]
            fraction = _crossing_fractions(
                previous[crossed],
                current[crossed],
                np.where(upper_crossed, components.boundary, 0.0),
                endpoint_crossed=endpoint_crossed[crossed],
            )
            decision_times[crossed_indices] = (step + fraction) * time_step
            choices[crossed_indices] = upper_crossed.astype(np.int8)
            active[crossed_indices] = False
        if np.any(active):
            raise RuntimeError(
                f"{int(np.sum(active))} simulated paths did not terminate within "
                f"{self.simulation_max_time:g} seconds"
            )

        response_seconds = decision_times + components.nondecision_time
        if np.any(contaminants):
            if self.contaminant is None:
                raise RuntimeError("non-zero contaminant probability requires a contaminant model")
            n_contaminants = int(np.sum(contaminants))
            choices[contaminants] = generator.binomial(
                1,
                self.contaminant.choice_probability,
                n_contaminants,
            )
            lower, upper = self.contaminant.time_bounds
            response_seconds[contaminants] = generator.uniform(
                lower,
                upper,
                n_contaminants,
            )
        response_values = response_seconds / self.response_time.unit.seconds_per_unit
        columns = {name: design[name] for name in design.columns}
        columns[self.outcome] = choices
        columns[self.response_time.column] = response_values
        return DriftDiffusionSimulation(
            study=Study(columns),
            contaminants=contaminants,
        )

    def fit(self, study: Study) -> DriftDiffusionFitResult:
        """Fit the joint choice/RT likelihood with deterministic bounded restarts."""

        outcomes = self._outcomes(study)
        response_times = self.response_time.read(study).seconds
        features = self._feature_matrix(study)
        minimum_response_time = float(np.min(response_times))
        nondecision_bounds = self._fit_nondecision_time_bounds(minimum_response_time)
        n_coefficients = len(self.coefficient_names)
        bounds = [(-self.drift_bound, self.drift_bound)] * n_coefficients
        bounds.extend(
            [
                self.boundary_bounds,
                self.starting_bias_bounds,
                nondecision_bounds,
            ]
        )
        if self.contaminant is not None:
            bounds.append(self.contaminant.probability_bounds)

        def objective(values: NDArray[np.float64]) -> float:
            components = self._components_from_vector(values)
            decision_times = response_times - components.nondecision_time
            if self.contaminant is None and np.any(decision_times <= 0):
                return float(np.finfo(np.float64).max / 1e100)
            drifts = features @ components.drift_coefficients
            log_density = self._joint_log_density(
                response_times,
                outcomes,
                drifts,
                components,
            )
            return float(-np.sum(log_density))

        starts = self._initial_points(outcomes, response_times, nondecision_bounds)
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
            raise ModelDataError(f"all DDM restarts produced non-finite objectives: {messages}")
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
        components = self._components_from_vector(estimates)
        drifts = features @ components.drift_coefficients
        log_density = self._joint_log_density(
            response_times,
            outcomes,
            drifts,
            components,
        )
        posterior_contaminant_probability = self._posterior_contaminant_probability(
            response_times,
            outcomes,
            drifts,
            components,
        )
        floor_count = int(np.sum(log_density <= _LOG_DENSITY_FLOOR))
        gradient = np.asarray(chosen.jac, dtype=np.float64)
        boundary_warning = self._boundary_warning(
            estimates,
            bounds,
            minimum_response_time=minimum_response_time,
            floor_count=floor_count,
        )
        diagnostics = FitDiagnostics(
            converged=bool(chosen.success),
            optimizer=f"L-BFGS-B ({self.n_restarts} deterministic restarts)",
            status=int(chosen.status),
            message=str(chosen.message),
            n_iterations=int(chosen.nit),
            objective=float(value),
            gradient_norm=float(np.linalg.norm(gradient)),
            hessian_condition=condition,
            boundary_estimate=boundary_warning,
        )
        return DriftDiffusionFitResult(
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
            posterior_contaminant_probability=posterior_contaminant_probability,
        )

    def predict(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> Prediction:
        """Return marginal upper-boundary choice probabilities."""

        prediction_mode = PredictionMode(mode)
        if prediction_mode not in self.supported_prediction_modes:
            raise UnsupportedPredictionMode(
                "WienerDriftDiffusion supports only filtered prediction"
            )
        components = self.parameter_components(fit)
        drifts = self._drifts(study, components.drift_coefficients)
        probability = _upper_boundary_probability(
            drifts,
            boundary=components.boundary,
            starting_bias=components.starting_bias,
        )
        if self.contaminant is not None:
            probability = (1.0 - components.contaminant_probability) * probability
            probability += components.contaminant_probability * self.contaminant.choice_probability
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
        """Return the joint log density of each observed choice and response time."""

        prediction_mode = PredictionMode(mode)
        if prediction_mode not in self.supported_prediction_modes:
            raise UnsupportedPredictionMode(
                "WienerDriftDiffusion supports only filtered prediction"
            )
        components = self.parameter_components(fit)
        outcomes = self._outcomes(study)
        response_times = self.response_time.read(study).seconds
        drifts = self._drifts(study, components.drift_coefficients)
        scores = self._joint_log_density(
            response_times,
            outcomes,
            drifts,
            components,
        )
        return _protected_array(scores, dtype=np.float64)

    def contaminant_responsibility(
        self,
        study: Study,
        fit: FitResult,
    ) -> NDArray[np.float64]:
        """Return fitted posterior contaminant probability for each requested trial."""

        components = self.parameter_components(fit)
        outcomes = self._outcomes(study)
        response_times = self.response_time.read(study).seconds
        drifts = self._drifts(study, components.drift_coefficients)
        posterior = self._posterior_contaminant_probability(
            response_times,
            outcomes,
            drifts,
            components,
        )
        return _protected_array(posterior, dtype=np.float64)

    def _joint_log_density(
        self,
        response_times: NDArray[np.float64],
        outcomes: NDArray[np.float64],
        drifts: NDArray[np.float64],
        components: DriftDiffusionParameters,
    ) -> NDArray[np.float64]:
        wiener_log_density = _wiener_log_density(
            response_times - components.nondecision_time,
            outcomes,
            drifts,
            boundary=components.boundary,
            starting_bias=components.starting_bias,
            terms=self.density_terms,
        )
        if self.contaminant is None:
            return wiener_log_density
        probability = components.contaminant_probability
        regular_component = np.log1p(-probability) + wiener_log_density
        if probability == 0:
            return regular_component
        contaminant_component = np.log(probability)
        contaminant_component += self.contaminant.pointwise_log_density(
            response_times,
            outcomes,
        )
        mixture = np.logaddexp(regular_component, contaminant_component)
        return np.maximum(mixture, _LOG_DENSITY_FLOOR)

    def _posterior_contaminant_probability(
        self,
        response_times: NDArray[np.float64],
        outcomes: NDArray[np.float64],
        drifts: NDArray[np.float64],
        components: DriftDiffusionParameters,
    ) -> NDArray[np.float64]:
        if self.contaminant is None or components.contaminant_probability == 0:
            return np.zeros(len(response_times), dtype=np.float64)
        contaminant_component = np.log(components.contaminant_probability)
        contaminant_component += self.contaminant.pointwise_log_density(
            response_times,
            outcomes,
        )
        denominator = self._joint_log_density(
            response_times,
            outcomes,
            drifts,
            components,
        )
        posterior = np.exp(contaminant_component - denominator)
        return np.clip(posterior, 0.0, 1.0)

    def _feature_matrix(self, study: Study) -> NDArray[np.float64]:
        columns = [np.ones(len(study), dtype=np.float64)]
        for name in self.covariates:
            if name not in study.columns:
                raise ModelDataError(f"study is missing covariate {name!r}")
            try:
                values = np.asarray(study[name], dtype=np.float64)
            except (TypeError, ValueError):
                raise ModelDataError(f"covariate {name!r} must be numeric") from None
            if not np.all(np.isfinite(values)):
                raise ModelDataError(f"covariate {name!r} must be finite")
            columns.append(values)
        return np.column_stack(columns)

    def _drifts(
        self,
        study: Study,
        coefficients: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return self._feature_matrix(study) @ coefficients

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

    def _components_from_vector(self, values: NDArray[np.float64]) -> DriftDiffusionParameters:
        n_coefficients = len(self.coefficient_names)
        return DriftDiffusionParameters(
            drift_coefficients=values[:n_coefficients],
            coefficient_names=self.coefficient_names,
            boundary=float(values[n_coefficients]),
            starting_bias=float(values[n_coefficients + 1]),
            nondecision_time=float(values[n_coefficients + 2]),
            contaminant_probability=(
                float(values[n_coefficients + 3]) if self.contaminant is not None else 0.0
            ),
        )

    def _initial_points(
        self,
        outcomes: NDArray[np.float64],
        response_times: NDArray[np.float64],
        nondecision_bounds: tuple[float, float],
    ) -> tuple[NDArray[np.float64], ...]:
        mean_choice = float(np.clip(np.mean(outcomes), 0.05, 0.95))
        base_drift = float(np.log(mean_choice / (1.0 - mean_choice)))
        nondecision_lower, nondecision_upper = nondecision_bounds
        if self.contaminant is None:
            nondecision_reference = float(np.min(response_times))
        else:
            nondecision_reference = float(np.quantile(response_times, 0.1))
        nondecision_reference = np.clip(
            nondecision_reference,
            nondecision_lower,
            nondecision_upper,
        )
        configurations = (
            (1.0, 0.5, 0.50, 0.10),
            (0.7, 0.4, 0.25, 0.25),
            (1.4, 0.6, 0.75, 0.50),
            (2.0, mean_choice, 0.10, 0.75),
        )
        starts: list[NDArray[np.float64]] = []
        for boundary, bias, nondecision_fraction, contaminant_fraction in configurations[
            : self.n_restarts
        ]:
            drift = np.zeros(len(self.coefficient_names), dtype=np.float64)
            drift[0] = np.clip(base_drift / boundary, -self.drift_bound, self.drift_bound)
            nondecision = nondecision_lower + nondecision_fraction * (
                nondecision_reference - nondecision_lower
            )
            natural_parameters = [
                np.clip(boundary, *self.boundary_bounds),
                np.clip(bias, *self.starting_bias_bounds),
                np.clip(nondecision, nondecision_lower, nondecision_upper),
            ]
            if self.contaminant is not None:
                probability_lower, probability_upper = self.contaminant.probability_bounds
                natural_parameters.append(
                    probability_lower
                    + contaminant_fraction * (probability_upper - probability_lower)
                )
            values = np.concatenate(
                (
                    drift,
                    np.asarray(natural_parameters),
                )
            )
            starts.append(values)
        while len(starts) < self.n_restarts:
            index = len(starts)
            source = np.array(starts[index % len(configurations)], copy=True)
            source[0] = np.clip(
                source[0] + (-1.0 if index % 2 else 1.0) * 0.25,
                -self.drift_bound,
                self.drift_bound,
            )
            starts.append(source)
        return tuple(starts)

    def _fit_nondecision_time_bounds(
        self,
        minimum_response_time: float,
    ) -> tuple[float, float]:
        maximum_from_data = minimum_response_time - self.minimum_decision_time
        if self.nondecision_time_bounds is None:
            if maximum_from_data <= 0:
                raise ModelDataError(
                    "minimum response time is too short for the configured minimum_decision_time"
                )
            return 0.0, maximum_from_data
        lower, configured_upper = self.nondecision_time_bounds
        if self.contaminant is not None:
            return lower, configured_upper
        upper = min(configured_upper, maximum_from_data)
        if upper <= lower:
            raise ModelDataError(
                "observed response times leave no admissible nondecision-time interval"
            )
        return lower, upper

    def _validate_contaminant_probability(self, probability: float) -> None:
        if self.contaminant is None:
            if probability != 0:
                raise ValueError("non-zero contaminant_probability requires a contaminant model")
            return
        lower, upper = self.contaminant.probability_bounds
        if not lower <= probability <= upper:
            raise ValueError(
                "contaminant_probability must lie within contaminant probability_bounds"
            )

    def _boundary_warning(
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
        nondecision_index = len(self.coefficient_names) + 2
        nondecision = float(estimates[nondecision_index])
        near_rt = self.contaminant is None and (
            minimum_response_time - nondecision <= 5 * self.minimum_decision_time
        )
        return bool(near_bound or near_rt or floor_count > 0)

    def _validate_fit(self, fit: FitResult) -> None:
        if fit.model_signature != self.signature or fit.parameter_names != self.parameter_names:
            raise ValueError("fit result belongs to a different model specification")


def _wiener_log_density(
    decision_time: NDArray[np.float64],
    choice: NDArray[np.float64],
    drift: NDArray[np.float64],
    *,
    boundary: float | NDArray[np.float64],
    starting_bias: float | NDArray[np.float64],
    terms: int,
) -> NDArray[np.float64]:
    """Joint two-boundary Wiener log density using paired convergent series."""

    times, choices, drifts, boundaries, biases = np.broadcast_arrays(
        np.asarray(decision_time, dtype=np.float64),
        np.asarray(choice, dtype=np.float64),
        np.asarray(drift, dtype=np.float64),
        np.asarray(boundary, dtype=np.float64),
        np.asarray(starting_bias, dtype=np.float64),
    )
    result = np.full(times.shape, _LOG_DENSITY_FLOOR, dtype=np.float64)
    valid = (
        np.isfinite(times)
        & (times > 0)
        & np.isfinite(drifts)
        & np.isfinite(boundaries)
        & (boundaries > 0)
        & np.isfinite(biases)
        & (biases > 0)
        & (biases < 1)
        & ((choices == 0) | (choices == 1))
    )
    if not np.any(valid):
        return result
    selected_times = times[valid]
    selected_choices = choices[valid]
    selected_drifts = drifts[valid]
    selected_boundaries = boundaries[valid]
    selected_biases = biases[valid]
    effective_drift = np.where(selected_choices == 1, -selected_drifts, selected_drifts)
    effective_bias = np.where(selected_choices == 1, 1.0 - selected_biases, selected_biases)
    scaled_time = selected_times / selected_boundaries**2
    standard = _standard_wiener_log_density(
        scaled_time,
        effective_bias,
        terms=terms,
    )
    log_density = -2.0 * np.log(selected_boundaries)
    log_density += -effective_drift * selected_boundaries * effective_bias
    log_density += -0.5 * effective_drift**2 * selected_times
    log_density += standard
    result[valid] = np.maximum(log_density, _LOG_DENSITY_FLOOR)
    return result


def _standard_wiener_log_density(
    scaled_time: NDArray[np.float64],
    starting_bias: float | NDArray[np.float64],
    *,
    terms: int,
) -> NDArray[np.float64]:
    times, biases = np.broadcast_arrays(
        np.asarray(scaled_time, dtype=np.float64),
        np.asarray(starting_bias, dtype=np.float64),
    )
    result = np.empty_like(times)
    small = times < 0.15
    if np.any(small):
        lower = -int(np.ceil((terms - 1) / 2))
        upper = int(np.floor((terms - 1) / 2))
        k = np.arange(lower, upper + 1, dtype=np.float64)
        coefficients = biases[small, None] + 2.0 * k[None, :]
        exponent = -(coefficients**2) / (2.0 * times[small, None])
        log_sum, sign = logsumexp(
            exponent,
            b=coefficients,
            axis=1,
            return_sign=True,
        )
        values = log_sum - 0.5 * np.log(2.0 * np.pi) - 1.5 * np.log(times[small])
        result[small] = np.where(sign > 0, values, _LOG_DENSITY_FLOOR)
    if np.any(~small):
        k = np.arange(1, terms + 1, dtype=np.float64)
        coefficients = k[None, :] * np.sin(k[None, :] * np.pi * biases[~small, None])
        exponent = -0.5 * (k[None, :] * np.pi) ** 2 * times[~small, None]
        log_sum, sign = logsumexp(
            exponent,
            b=coefficients,
            axis=1,
            return_sign=True,
        )
        values = np.log(np.pi) + log_sum
        result[~small] = np.where(sign > 0, values, _LOG_DENSITY_FLOOR)
    return result


def _bridge_crossings(
    previous: NDArray[np.float64],
    current: NDArray[np.float64],
    boundary: float | NDArray[np.float64],
    *,
    time_step: float,
    generator: np.random.Generator,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Return Brownian-bridge absorptions hidden between two interior endpoints.

    A discretised path that starts and ends inside the corridor may still have crossed a
    boundary within the step. The bridge absorption probabilities are exact for a single
    boundary and their overlap is negligible whenever the corridor spans many steps.
    """

    upper_exponent = 2.0 * (boundary - previous) * (boundary - current) / time_step
    lower_exponent = 2.0 * previous * current / time_step
    upper = np.zeros(len(previous), dtype=np.bool_)
    lower = np.zeros(len(previous), dtype=np.bool_)
    candidate = np.flatnonzero(
        (upper_exponent < _BRIDGE_EXPONENT_LIMIT) | (lower_exponent < _BRIDGE_EXPONENT_LIMIT)
    )
    if not len(candidate):
        return upper, lower
    upper_probability = np.exp(-upper_exponent[candidate])
    lower_probability = np.exp(-lower_exponent[candidate])
    draws = generator.uniform(0.0, 1.0, len(candidate))
    upper[candidate] = draws < upper_probability
    lower[candidate] = (draws >= upper_probability) & (
        draws < upper_probability + lower_probability
    )
    return upper, lower


def _crossing_fractions(
    previous: NDArray[np.float64],
    current: NDArray[np.float64],
    target: NDArray[np.float64],
    *,
    endpoint_crossed: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Return within-step crossing positions, midpoints for bridge absorptions."""

    span = current - previous
    interpolated = (target - previous) / np.where(span == 0.0, 1.0, span)
    fraction = np.where(endpoint_crossed & (span != 0.0), interpolated, 0.5)
    return np.clip(fraction, 0.0, 1.0)


def _upper_boundary_probability(
    drift: NDArray[np.float64],
    *,
    boundary: float | NDArray[np.float64],
    starting_bias: float | NDArray[np.float64],
) -> NDArray[np.float64]:
    drifts, boundaries, biases = np.broadcast_arrays(
        np.asarray(drift, dtype=np.float64),
        np.asarray(boundary, dtype=np.float64),
        np.asarray(starting_bias, dtype=np.float64),
    )
    scaled = 2.0 * drifts * boundaries
    probability = np.empty_like(scaled)
    near_zero = np.abs(scaled) < 1e-8
    probability[near_zero] = biases[near_zero]
    positive = (scaled > 0) & ~near_zero
    probability[positive] = np.expm1(-scaled[positive] * biases[positive]) / np.expm1(
        -scaled[positive]
    )
    negative = (scaled < 0) & ~near_zero
    negative_scaled = scaled[negative]
    probability[negative] = (
        np.exp(negative_scaled * (1.0 - biases[negative])) - np.exp(negative_scaled)
    ) / (1.0 - np.exp(negative_scaled))
    return probability


def _numerical_hessian(
    objective: Any,
    point: NDArray[np.float64],
    bounds: Sequence[tuple[float, float]],
) -> NDArray[np.float64]:
    n_parameters = len(point)
    hessian = np.empty((n_parameters, n_parameters), dtype=np.float64)
    evaluation_point = np.array(point, copy=True)
    base_steps = np.maximum(1e-5, 1e-4 * np.maximum(1.0, np.abs(point)))
    steps = np.array(base_steps, copy=True)
    for index, (lower, upper) in enumerate(bounds):
        base_steps[index] = min(base_steps[index], (upper - lower) / 4.0)
        evaluation_point[index] = np.clip(
            evaluation_point[index],
            lower + base_steps[index],
            upper - base_steps[index],
        )
        steps[index] = min(
            base_steps[index],
            (evaluation_point[index] - lower) / 2.0,
            (upper - evaluation_point[index]) / 2.0,
        )
    center = float(objective(evaluation_point))
    for row in range(n_parameters):
        row_step = steps[row]
        plus = evaluation_point.copy()
        minus = evaluation_point.copy()
        plus[row] += row_step
        minus[row] -= row_step
        hessian[row, row] = (
            float(objective(plus)) - 2.0 * center + float(objective(minus))
        ) / row_step**2
        for column in range(row):
            column_step = steps[column]
            plus_plus = evaluation_point.copy()
            plus_minus = evaluation_point.copy()
            minus_plus = evaluation_point.copy()
            minus_minus = evaluation_point.copy()
            plus_plus[[row, column]] += [row_step, column_step]
            plus_minus[[row, column]] += [row_step, -column_step]
            minus_plus[[row, column]] += [-row_step, column_step]
            minus_minus[[row, column]] -= [row_step, column_step]
            value = (
                float(objective(plus_plus))
                - float(objective(plus_minus))
                - float(objective(minus_plus))
                + float(objective(minus_minus))
            ) / (4.0 * row_step * column_step)
            hessian[row, column] = hessian[column, row] = value
    return hessian


def _ordered_bounds(values: Sequence[float], name: str) -> tuple[float, float]:
    if len(values) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    lower, upper = (float(value) for value in values)
    if not np.isfinite(lower) or not np.isfinite(upper) or not 0 < lower < upper:
        raise ValueError(f"{name} must contain finite positive ordered values")
    return lower, upper


def _nonnegative_bounds(values: Sequence[float], name: str) -> tuple[float, float]:
    if len(values) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    lower, upper = (float(value) for value in values)
    if not np.isfinite(lower) or not np.isfinite(upper) or not 0 <= lower < upper:
        raise ValueError(f"{name} must contain finite non-negative ordered values")
    return lower, upper


def _probability_bounds(values: Sequence[float], name: str) -> tuple[float, float]:
    lower, upper = _nonnegative_bounds(values, name)
    if upper >= 1:
        raise ValueError(f"{name} must have an upper bound below one")
    return lower, upper


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
