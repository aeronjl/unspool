"""Partially pooled longitudinal Wiener parameter trajectories."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

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
    _numerical_hessian,
    _upper_boundary_probability,
    _wiener_log_density,
)
from unspool.models.glm import _format_time, _linear_time_basis
from unspool.models.smooth_ddm import (
    DriftDiffusionTrajectory,
    SmoothDriftDiffusionFitResult,
    SmoothWienerDriftDiffusion,
    _roughness_matrix,
    _simulate_trialwise_wiener,
)
from unspool.study import Study

_INVALID_OBJECTIVE = float(np.finfo(np.float64).max / 1e100)
_CONSTRAINT_PENALTY = 1e6


@dataclass(frozen=True, slots=True)
class HierarchicalSmoothDriftDiffusionSimulation:
    """Observed trials paired with unexposed population and subject path truth."""

    study: Study
    subjects: tuple[Any, ...]
    clock: str
    knots: tuple[float, ...]
    parameter_names: tuple[str, ...]
    subject_parameters: tuple[str, ...]
    population_knot_values: NDArray[np.float64]
    subject_deviation_knot_values: NDArray[np.float64]

    def __post_init__(self) -> None:
        subjects = tuple(_scalar(subject) for subject in self.subjects)
        knots = tuple(float(knot) for knot in self.knots)
        names = tuple(self.parameter_names)
        subject_parameters = tuple(self.subject_parameters)
        population = _protected_array(self.population_knot_values, dtype=np.float64)
        deviations = _protected_array(self.subject_deviation_knot_values, dtype=np.float64)
        if subjects != tuple(_scalar(subject) for subject in self.study.subjects):
            raise ValueError("simulation subjects must match the study's subject order")
        _validate_coordinates(self.clock, knots, names, subject_parameters)
        if population.shape != (len(names), len(knots)):
            raise ValueError("population paths must align with parameters and knots")
        if deviations.shape != (len(subjects), len(subject_parameters), len(knots)):
            raise ValueError("subject deviations must align with subjects, parameters, and knots")
        if not np.all(np.isfinite(population)) or not np.all(np.isfinite(deviations)):
            raise ValueError("simulation paths must be finite")
        object.__setattr__(self, "subjects", subjects)
        object.__setattr__(self, "knots", knots)
        object.__setattr__(self, "parameter_names", names)
        object.__setattr__(self, "subject_parameters", subject_parameters)
        object.__setattr__(self, "population_knot_values", population)
        object.__setattr__(self, "subject_deviation_knot_values", deviations)

    @property
    def subject_knot_values(self) -> NDArray[np.float64]:
        """Return realized full parameter paths for every subject."""

        values = np.broadcast_to(
            self.population_knot_values,
            (len(self.subjects), *self.population_knot_values.shape),
        ).copy()
        for deviation_index, parameter in enumerate(self.subject_parameters):
            parameter_index = self.parameter_names.index(parameter)
            values[:, parameter_index, :] += self.subject_deviation_knot_values[
                :, deviation_index, :
            ]
        return _protected_array(values, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class HierarchicalSmoothDriftDiffusionFitResult(SmoothDriftDiffusionFitResult):
    """Population paths and shrunken subject deviations from one joint MAP fit."""

    subjects: tuple[Any, ...]
    base_parameter_names: tuple[str, ...]
    subject_parameters: tuple[str, ...]
    subject_deviations: NDArray[np.float64]
    subject_standard_errors: NDArray[np.float64]
    subject_scale: float
    subject_smoothness: float
    unseen_subject_policy: str = "population-trajectory-plugin"
    uncertainty_policy: str = "arrowhead-local-hessian"

    def __post_init__(self) -> None:
        SmoothDriftDiffusionFitResult.__post_init__(self)
        subjects = tuple(_scalar(subject) for subject in self.subjects)
        names = tuple(self.base_parameter_names)
        subject_parameters = tuple(self.subject_parameters)
        deviations = _protected_array(self.subject_deviations, dtype=np.float64)
        standard_errors = _protected_array(self.subject_standard_errors, dtype=np.float64)
        if not subjects or len(set(subjects)) != len(subjects):
            raise ValueError("fit subjects must be non-empty and unique")
        _validate_coordinates(self.clock, self.knots, names, subject_parameters)
        expected = (len(subjects), len(subject_parameters), len(self.knots))
        if deviations.shape != expected or standard_errors.shape != expected:
            raise ValueError("subject estimates must align with subjects, parameters, and knots")
        if not np.all(np.isfinite(deviations)) or not np.all(np.isfinite(standard_errors)):
            raise ValueError("subject estimates and standard errors must be finite")
        if np.any(standard_errors < 0):
            raise ValueError("subject standard errors must be non-negative")
        if set(subject_parameters) - set(self.varying_parameters):
            raise ValueError("subject parameters must be varying population parameters")
        if not np.isfinite(self.subject_scale) or self.subject_scale <= 0:
            raise ValueError("subject_scale must be finite and positive")
        if not np.isfinite(self.subject_smoothness) or self.subject_smoothness <= 0:
            raise ValueError("subject_smoothness must be finite and positive")
        if self.unseen_subject_policy != "population-trajectory-plugin":
            raise ValueError("unseen_subject_policy must be 'population-trajectory-plugin'")
        if self.uncertainty_policy != "arrowhead-local-hessian":
            raise ValueError("uncertainty_policy must be 'arrowhead-local-hessian'")
        object.__setattr__(self, "subjects", subjects)
        object.__setattr__(self, "base_parameter_names", names)
        object.__setattr__(self, "subject_parameters", subject_parameters)
        object.__setattr__(self, "subject_deviations", deviations)
        object.__setattr__(self, "subject_standard_errors", standard_errors)

    @property
    def population_knot_values(self) -> NDArray[np.float64]:
        """Return every population parameter on the common knot grid."""

        values = _unpack_population_values(
            self.estimates,
            base_parameter_names=self.base_parameter_names,
            varying_parameters=self.varying_parameters,
            n_knots=len(self.knots),
        )
        return _protected_array(values, dtype=np.float64)

    @property
    def subject_knot_values(self) -> NDArray[np.float64]:
        """Return population-plus-deviation paths for fitted subjects."""

        values = np.broadcast_to(
            self.population_knot_values,
            (len(self.subjects), *self.population_knot_values.shape),
        ).copy()
        for deviation_index, parameter in enumerate(self.subject_parameters):
            parameter_index = self.base_parameter_names.index(parameter)
            values[:, parameter_index, :] += self.subject_deviations[:, deviation_index, :]
        return _protected_array(values, dtype=np.float64)

    def subject_was_fitted(self, subject: Any) -> bool:
        """Report whether prediction can use an estimated subject path."""

        return _scalar(subject) in self.subjects


@dataclass(frozen=True, slots=True)
class HierarchicalSmoothWienerDriftDiffusion(SmoothWienerDriftDiffusion):
    """Smooth population Wiener paths with shrunken subject-deviation paths.

    Population parameters retain the smooth Wiener's natural-scale coordinates. Selected
    varying parameters receive additive subject deviations with a fixed Gaussian scale
    and a time-scaled first-difference penalty. Unseen subjects use the population path.
    """

    subject_parameters: tuple[str, ...] | None = None
    subject_scale: float = 0.25
    subject_smoothness: float = 10.0

    def __post_init__(self) -> None:
        SmoothWienerDriftDiffusion.__post_init__(self)
        if self.shared_trajectory:
            raise ValueError("shared_trajectory is not used by the hierarchical model")
        if not np.isfinite(self.subject_scale) or self.subject_scale <= 0:
            raise ValueError("subject_scale must be finite and positive")
        if not np.isfinite(self.subject_smoothness) or self.subject_smoothness <= 0:
            raise ValueError("subject_smoothness must be finite and positive")
        if self.subject_parameters is None:
            subject_parameters = tuple(self.varying_parameters or ())
        else:
            subject_parameters = tuple(self.subject_parameters)
        if not subject_parameters or len(set(subject_parameters)) != len(subject_parameters):
            raise ValueError("subject_parameters must be non-empty and unique")
        if set(subject_parameters) - set(self.varying_parameters or ()):
            raise ValueError("subject_parameters must be selected varying_parameters")
        object.__setattr__(self, "subject_parameters", subject_parameters)

    @property
    def model_name(self) -> str:
        return "hierarchical-smooth-wiener-drift-diffusion"

    @property
    def signature(self) -> str:
        covariates = ",".join(self.covariates)
        knots = ",".join(_format_time(knot) for knot in self.knots)
        varying = ",".join(self.varying_parameters or ())
        subject_parameters = ",".join(self.subject_parameters or ())
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
            f"subject_parameters={subject_parameters};subject_scale={self.subject_scale};"
            f"subject_smoothness={self.subject_smoothness};"
            f"nondecision_bounds={nondecision}]"
        )

    def simulate(
        self,
        design: Study,
        parameters: Mapping[str, float],
        *,
        seed: int | np.random.Generator,
    ) -> Study:
        """Generate trials without exposing realized subject trajectories."""

        return self.simulate_with_effects(design, parameters, seed=seed).study

    def simulate_with_effects(
        self,
        design: Study,
        parameters: Mapping[str, float],
        *,
        seed: int | np.random.Generator,
        subject_deviation_paths: Mapping[Any, Mapping[str, Sequence[float]]] | None = None,
    ) -> HierarchicalSmoothDriftDiffusionSimulation:
        """Generate trials and retain the random-effect truth outside observed data."""

        population_vector = self._parameter_vector(parameters)
        population_knots = self._unpack_knot_values(population_vector)
        basis = self._time_basis(design)
        generator = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        subjects = tuple(_scalar(subject) for subject in design.subjects)
        deviations = (
            self._draw_subject_deviations(subjects, population_knots, generator)
            if subject_deviation_paths is None
            else self._subject_deviation_array(
                subjects,
                population_knots,
                subject_deviation_paths,
            )
        )
        row_subjects = _row_subject_indices(design, subjects)
        trial_values = self._subject_trial_values_from_basis(
            basis,
            population_vector,
            deviations,
            row_subjects,
        )
        features = self._feature_matrix(design)
        n_coefficients = len(self.coefficient_names)
        drifts = np.sum(features * trial_values[:, :n_coefficients], axis=1)
        choices, response_seconds = _simulate_trialwise_wiener(
            drifts,
            trial_values[:, n_coefficients],
            trial_values[:, n_coefficients + 1],
            trial_values[:, n_coefficients + 2],
            generator=generator,
            time_step=self.simulation_time_step,
            maximum_time=self.simulation_max_time,
        )
        columns = {name: design[name] for name in design.columns}
        columns[self.outcome] = choices
        columns[self.response_time.column] = (
            response_seconds / self.response_time.unit.seconds_per_unit
        )
        return HierarchicalSmoothDriftDiffusionSimulation(
            study=Study(columns),
            subjects=subjects,
            clock=self.time,
            knots=self.knots,
            parameter_names=self.base_parameter_names,
            subject_parameters=tuple(self.subject_parameters or ()),
            population_knot_values=population_knots,
            subject_deviation_knot_values=deviations,
        )

    def fit(self, study: Study) -> HierarchicalSmoothDriftDiffusionFitResult:
        """Jointly fit population paths and smooth shrunken subject deviations."""

        subjects = tuple(_scalar(subject) for subject in study.subjects)
        if len(subjects) < 2:
            raise ModelDataError("hierarchical smooth DDM fitting requires at least two subjects")
        outcomes = self._outcomes(study)
        response_times = self.response_time.read(study).seconds
        features = self._feature_matrix(study)
        basis = self._time_basis(study)
        row_subjects = _row_subject_indices(study, subjects)
        minimum_response_time = float(np.min(response_times))
        nondecision_bounds = self._fit_nondecision_time_bounds(minimum_response_time)
        population_bounds = self._optimization_bounds(nondecision_bounds)
        deviation_bounds = self._deviation_bounds()
        n_population = len(self.parameter_names)
        n_deviation = len(deviation_bounds)
        joint_bounds = [
            *population_bounds,
            *(deviation_bounds * len(subjects)),
        ]
        population_penalty = self._penalty_matrix()
        subject_penalty = self._subject_penalty_matrix()

        def objective(values: NDArray[np.float64]) -> float:
            population = values[:n_population]
            deviations = values[n_population:].reshape(
                len(subjects),
                len(self.subject_parameters or ()),
                len(self.knots),
            )
            trial_values, constraint_penalty = self._constrained_subject_trial_values(
                basis,
                population,
                deviations,
                row_subjects,
            )
            n_coefficients = len(self.coefficient_names)
            decision_times = response_times - trial_values[:, n_coefficients + 2]
            if np.any(decision_times <= 0):
                return _INVALID_OBJECTIVE
            drifts = np.sum(features * trial_values[:, :n_coefficients], axis=1)
            log_density = _wiener_log_density(
                decision_times,
                outcomes,
                drifts,
                boundary=trial_values[:, n_coefficients],
                starting_bias=trial_values[:, n_coefficients + 1],
                terms=self.density_terms,
            )
            population_roughness = 0.5 * float(population @ population_penalty @ population)
            subject_roughness = 0.5 * float(
                np.einsum("sdk,kl,sdl->", deviations, subject_penalty, deviations)
            )
            return float(
                -np.sum(log_density) + population_roughness + subject_roughness + constraint_penalty
            )

        static_starts = self._initial_points(outcomes, response_times, nondecision_bounds)
        starts = tuple(
            np.concatenate(
                (
                    self._expand_static_vector(start),
                    np.zeros(len(subjects) * n_deviation, dtype=np.float64),
                )
            )
            for start in static_starts
        )
        results = [
            minimize(
                objective,
                start,
                method="L-BFGS-B",
                bounds=joint_bounds,
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
                f"all hierarchical smooth DDM restarts were non-finite: {messages}"
            )
        successful = [index for index in finite if results[index].success]
        eligible = successful if successful else finite
        selected = min(eligible, key=lambda index: float(restart_objectives[index]))
        chosen = results[selected]
        joint_estimates = np.asarray(chosen.x, dtype=np.float64)
        population = joint_estimates[:n_population]
        deviations = joint_estimates[n_population:].reshape(
            len(subjects),
            len(self.subject_parameters or ()),
            len(self.knots),
        )
        if not self._effective_values_valid(population, deviations, tolerance=1e-5):
            raise ModelDataError(
                "hierarchical smooth DDM optimum violates effective natural-scale bounds"
            )
        population_covariance, subject_covariances, condition = _arrowhead_covariance(
            objective,
            joint_estimates,
            population_bounds=population_bounds,
            deviation_bounds=deviation_bounds,
            n_subjects=len(subjects),
        )
        population_standard_errors = np.sqrt(np.maximum(np.diag(population_covariance), 0.0))
        subject_standard_errors = np.sqrt(
            np.maximum(
                np.stack([np.diag(covariance) for covariance in subject_covariances]),
                0.0,
            )
        ).reshape(deviations.shape)
        trial_values = self._subject_trial_values_from_basis(
            basis,
            population,
            deviations,
            row_subjects,
        )
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
        diagnostics = FitDiagnostics(
            converged=bool(chosen.success),
            optimizer=f"joint L-BFGS-B ({self.n_restarts} deterministic restarts)",
            status=int(chosen.status),
            message=str(chosen.message),
            n_iterations=int(chosen.nit),
            objective=float(objective(joint_estimates)),
            gradient_norm=float(np.linalg.norm(np.asarray(chosen.jac, dtype=np.float64))),
            hessian_condition=condition,
            boundary_estimate=self._hierarchical_boundary_warning(
                population,
                deviations,
                population_bounds,
                minimum_response_time=minimum_response_time,
                floor_count=floor_count,
            ),
        )
        return HierarchicalSmoothDriftDiffusionFitResult(
            model_name=self.model_name,
            model_signature=self.signature,
            parameter_names=self.parameter_names,
            estimates=population,
            standard_errors=population_standard_errors,
            covariance=population_covariance,
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
            subjects=subjects,
            base_parameter_names=self.base_parameter_names,
            subject_parameters=tuple(self.subject_parameters or ()),
            subject_deviations=deviations,
            subject_standard_errors=subject_standard_errors,
            subject_scale=self.subject_scale,
            subject_smoothness=self.subject_smoothness,
        )

    def predict(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> Prediction:
        """Predict with fitted subject paths or the population path for unseen subjects."""

        prediction_mode = PredictionMode(mode)
        if prediction_mode not in self.supported_prediction_modes:
            raise UnsupportedPredictionMode(
                "HierarchicalSmoothWienerDriftDiffusion supports only filtered prediction"
            )
        hierarchical_fit = self._validated_hierarchical_fit(fit)
        trial_values = self._fitted_trial_values(study, hierarchical_fit)
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
        """Score joint choice/RT observations under fitted or population paths."""

        prediction_mode = PredictionMode(mode)
        if prediction_mode not in self.supported_prediction_modes:
            raise UnsupportedPredictionMode(
                "HierarchicalSmoothWienerDriftDiffusion supports only filtered prediction"
            )
        hierarchical_fit = self._validated_hierarchical_fit(fit)
        outcomes = self._outcomes(study)
        response_times = self.response_time.read(study).seconds
        trial_values = self._fitted_trial_values(study, hierarchical_fit)
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

    def parameter_trajectory(
        self,
        fit: FitResult,
        *,
        times: Sequence[float] | None = None,
    ) -> DriftDiffusionTrajectory:
        """Evaluate the fitted population trajectory."""

        self._validated_hierarchical_fit(fit)
        return SmoothWienerDriftDiffusion.parameter_trajectory(self, fit, times=times)

    def population_trajectory(
        self,
        fit: FitResult,
        *,
        times: Sequence[float] | None = None,
    ) -> DriftDiffusionTrajectory:
        """Evaluate the fitted population trajectory."""

        return self.parameter_trajectory(fit, times=times)

    def subject_trajectory(
        self,
        fit: FitResult,
        subject: Any,
        *,
        times: Sequence[float] | None = None,
    ) -> DriftDiffusionTrajectory:
        """Evaluate a fitted subject path or the unseen-subject population plug-in."""

        hierarchical_fit = self._validated_hierarchical_fit(fit)
        subject_key = _scalar(subject)
        try:
            subject_position = hierarchical_fit.subjects.index(subject_key)
        except ValueError:
            knot_values = hierarchical_fit.population_knot_values
        else:
            knot_values = hierarchical_fit.subject_knot_values[subject_position]
        evaluation_times = self.knots if times is None else times
        time_array = np.asarray(evaluation_times, dtype=np.float64)
        basis = _linear_time_basis(time_array, self.knots)
        return DriftDiffusionTrajectory(
            clock=self.time,
            times=time_array,
            parameter_names=self.base_parameter_names,
            values=basis @ knot_values.T,
        )

    def _subject_penalty_matrix(self) -> NDArray[np.float64]:
        penalty = np.eye(len(self.knots), dtype=np.float64) / self.subject_scale**2
        penalty += self.subject_smoothness * _roughness_matrix(self.knots)
        return penalty

    def _deviation_bounds(self) -> list[tuple[float, float]]:
        bounds: list[tuple[float, float]] = []
        for parameter in self.subject_parameters or ():
            if parameter.startswith("drift."):
                limit = 2.0 * self.drift_bound
            elif parameter == "boundary":
                limit = self.boundary_bounds[1] - self.boundary_bounds[0]
            elif parameter == "starting_bias":
                limit = self.starting_bias_bounds[1] - self.starting_bias_bounds[0]
            else:
                raise ValueError(f"unsupported subject parameter {parameter!r}")
            bounds.extend([(-limit, limit)] * len(self.knots))
        return bounds

    def _subject_trial_values_from_basis(
        self,
        basis: NDArray[np.float64],
        population: NDArray[np.float64],
        deviations: NDArray[np.float64],
        row_subjects: NDArray[np.intp],
    ) -> NDArray[np.float64]:
        values = self._trial_parameter_values_from_basis(basis, population)
        for deviation_index, parameter in enumerate(self.subject_parameters or ()):
            parameter_index = self.base_parameter_names.index(parameter)
            values[:, parameter_index] += np.einsum(
                "ij,ij->i",
                basis,
                deviations[row_subjects, deviation_index, :],
            )
        return values

    def _effective_values_valid(
        self,
        population: NDArray[np.float64],
        deviations: NDArray[np.float64],
        *,
        tolerance: float = 0.0,
    ) -> bool:
        population_knots = self._unpack_knot_values(population)
        for subject in range(len(deviations)):
            for deviation_index, parameter in enumerate(self.subject_parameters or ()):
                parameter_index = self.base_parameter_names.index(parameter)
                effective = population_knots[parameter_index] + deviations[subject, deviation_index]
                lower, upper = self._parameter_path_bounds(parameter)
                if np.any(effective < lower - tolerance) or np.any(effective > upper + tolerance):
                    return False
        return True

    def _constrained_subject_trial_values(
        self,
        basis: NDArray[np.float64],
        population: NDArray[np.float64],
        deviations: NDArray[np.float64],
        row_subjects: NDArray[np.intp],
    ) -> tuple[NDArray[np.float64], float]:
        effective = np.broadcast_to(
            self._unpack_knot_values(population),
            (len(deviations), len(self.base_parameter_names), len(self.knots)),
        ).copy()
        penalty = 0.0
        for deviation_index, parameter in enumerate(self.subject_parameters or ()):
            parameter_index = self.base_parameter_names.index(parameter)
            values = effective[:, parameter_index, :] + deviations[:, deviation_index, :]
            lower, upper = self._parameter_path_bounds(parameter)
            violations = np.minimum(values - lower, 0.0) + np.maximum(values - upper, 0.0)
            penalty += _CONSTRAINT_PENALTY * float(np.sum(violations**2))
            effective[:, parameter_index, :] = np.clip(values, lower, upper)
        trial_values = np.einsum("ik,ijk->ij", basis, effective[row_subjects])
        return trial_values, penalty

    def _parameter_path_bounds(self, parameter: str) -> tuple[float, float]:
        if parameter.startswith("drift."):
            return -self.drift_bound, self.drift_bound
        if parameter == "boundary":
            return self.boundary_bounds
        if parameter == "starting_bias":
            return self.starting_bias_bounds
        raise ValueError(f"unsupported subject parameter {parameter!r}")

    def _draw_subject_deviations(
        self,
        subjects: tuple[Any, ...],
        population_knots: NDArray[np.float64],
        generator: np.random.Generator,
    ) -> NDArray[np.float64]:
        covariance = np.linalg.pinv(self._subject_penalty_matrix(), hermitian=True)
        deviations = np.empty(
            (len(subjects), len(self.subject_parameters or ()), len(self.knots)),
            dtype=np.float64,
        )
        for subject in range(len(subjects)):
            for deviation_index, parameter in enumerate(self.subject_parameters or ()):
                parameter_index = self.base_parameter_names.index(parameter)
                for _attempt in range(1_000):
                    candidate = generator.multivariate_normal(
                        np.zeros(len(self.knots)),
                        covariance,
                    )
                    try:
                        self._validate_natural_values(
                            parameter,
                            population_knots[parameter_index] + candidate,
                        )
                    except ValueError:
                        continue
                    deviations[subject, deviation_index] = candidate
                    break
                else:
                    raise RuntimeError(f"could not draw a valid deviation path for {parameter!r}")
        return deviations

    def _subject_deviation_array(
        self,
        subjects: tuple[Any, ...],
        population_knots: NDArray[np.float64],
        paths: Mapping[Any, Mapping[str, Sequence[float]]],
    ) -> NDArray[np.float64]:
        normalized = {_scalar(subject): values for subject, values in paths.items()}
        if set(normalized) != set(subjects):
            raise ValueError("subject_deviation_paths must contain every design subject exactly")
        deviations = np.empty(
            (len(subjects), len(self.subject_parameters or ()), len(self.knots)),
            dtype=np.float64,
        )
        for subject_position, subject in enumerate(subjects):
            subject_paths = normalized[subject]
            if set(subject_paths) != set(self.subject_parameters or ()):
                raise ValueError("each subject deviation must contain every subject parameter")
            for deviation_index, parameter in enumerate(self.subject_parameters or ()):
                values = np.asarray(subject_paths[parameter], dtype=np.float64)
                if values.shape != (len(self.knots),) or not np.all(np.isfinite(values)):
                    raise ValueError("subject deviation paths require one finite value per knot")
                parameter_index = self.base_parameter_names.index(parameter)
                self._validate_natural_values(
                    parameter,
                    population_knots[parameter_index] + values,
                )
                deviations[subject_position, deviation_index] = values
        return deviations

    def _fitted_trial_values(
        self,
        study: Study,
        fit: HierarchicalSmoothDriftDiffusionFitResult,
    ) -> NDArray[np.float64]:
        basis = self._time_basis(study)
        population = self._trial_parameter_values_from_basis(basis, fit.estimates)
        subject_lookup = {subject: index for index, subject in enumerate(fit.subjects)}
        for row, subject in enumerate(study["subject"]):
            subject_position = subject_lookup.get(_scalar(subject))
            if subject_position is None:
                continue
            for deviation_index, parameter in enumerate(fit.subject_parameters):
                parameter_index = self.base_parameter_names.index(parameter)
                population[row, parameter_index] += float(
                    basis[row] @ fit.subject_deviations[subject_position, deviation_index]
                )
        return population

    def _hierarchical_boundary_warning(
        self,
        population: NDArray[np.float64],
        deviations: NDArray[np.float64],
        population_bounds: Sequence[tuple[float, float]],
        *,
        minimum_response_time: float,
        floor_count: int,
    ) -> bool:
        if self._trajectory_boundary_warning(
            population,
            population_bounds,
            minimum_response_time=minimum_response_time,
            floor_count=floor_count,
        ):
            return True
        population_knots = self._unpack_knot_values(population)
        for subject in range(len(deviations)):
            for deviation_index, parameter in enumerate(self.subject_parameters or ()):
                parameter_index = self.base_parameter_names.index(parameter)
                values = population_knots[parameter_index] + deviations[subject, deviation_index]
                if parameter.startswith("drift."):
                    lower, upper = -self.drift_bound, self.drift_bound
                elif parameter == "boundary":
                    lower, upper = self.boundary_bounds
                else:
                    lower, upper = self.starting_bias_bounds
                tolerance = 1e-4 * max(1.0, upper - lower)
                if np.any(values - lower <= tolerance) or np.any(upper - values <= tolerance):
                    return True
        return False

    def _validated_hierarchical_fit(
        self,
        fit: FitResult,
    ) -> HierarchicalSmoothDriftDiffusionFitResult:
        if not isinstance(fit, HierarchicalSmoothDriftDiffusionFitResult):
            raise ValueError("fit result does not retain hierarchical Wiener trajectories")
        self._validate_fit(fit)
        return fit

    def _validate_study_scope(self, study: Study) -> None:
        return None


def _arrowhead_covariance(
    objective: Any,
    point: NDArray[np.float64],
    *,
    population_bounds: Sequence[tuple[float, float]],
    deviation_bounds: Sequence[tuple[float, float]],
    n_subjects: int,
) -> tuple[NDArray[np.float64], tuple[NDArray[np.float64], ...], float]:
    """Invert a numerical arrowhead Hessian without evaluating zero cross-subject blocks."""

    n_population = len(population_bounds)
    n_deviation = len(deviation_bounds)
    population_point = point[:n_population]

    def with_population(values: NDArray[np.float64]) -> float:
        candidate = np.array(point, copy=True)
        candidate[:n_population] = values
        return float(objective(candidate))

    population_hessian = _numerical_hessian(
        with_population,
        population_point,
        population_bounds,
    )
    deviation_hessians: list[NDArray[np.float64]] = []
    cross_hessians: list[NDArray[np.float64]] = []
    deviation_inverses: list[NDArray[np.float64]] = []
    for subject in range(n_subjects):
        start = n_population + subject * n_deviation
        stop = start + n_deviation

        def with_deviation(
            values: NDArray[np.float64],
            block_start: int = start,
            block_stop: int = stop,
        ) -> float:
            candidate = np.array(point, copy=True)
            candidate[block_start:block_stop] = values
            return float(objective(candidate))

        deviation_hessian = _numerical_hessian(
            with_deviation,
            point[start:stop],
            deviation_bounds,
        )
        cross_hessian = _numerical_cross_hessian(
            objective,
            point,
            left_indices=np.arange(n_population),
            right_indices=np.arange(start, stop),
            bounds=[*population_bounds, *(deviation_bounds * n_subjects)],
        )
        deviation_hessians.append(deviation_hessian)
        cross_hessians.append(cross_hessian)
        deviation_inverses.append(np.linalg.pinv(deviation_hessian, hermitian=True))

    schur = np.array(population_hessian, copy=True)
    for cross, deviation_inverse in zip(
        cross_hessians,
        deviation_inverses,
        strict=True,
    ):
        schur -= cross @ deviation_inverse @ cross.T
    population_covariance = np.linalg.pinv(schur, hermitian=True)
    subject_covariances = tuple(
        deviation_inverse
        + deviation_inverse @ cross.T @ population_covariance @ cross @ deviation_inverse
        for cross, deviation_inverse in zip(
            cross_hessians,
            deviation_inverses,
            strict=True,
        )
    )
    full_hessian = np.zeros((len(point), len(point)), dtype=np.float64)
    full_hessian[:n_population, :n_population] = population_hessian
    for subject, (deviation_hessian, cross) in enumerate(
        zip(deviation_hessians, cross_hessians, strict=True)
    ):
        start = n_population + subject * n_deviation
        stop = start + n_deviation
        full_hessian[start:stop, start:stop] = deviation_hessian
        full_hessian[:n_population, start:stop] = cross
        full_hessian[start:stop, :n_population] = cross.T
    return population_covariance, subject_covariances, float(np.linalg.cond(full_hessian))


def _numerical_cross_hessian(
    objective: Any,
    point: NDArray[np.float64],
    *,
    left_indices: NDArray[np.int64],
    right_indices: NDArray[np.int64],
    bounds: Sequence[tuple[float, float]],
) -> NDArray[np.float64]:
    evaluation_point = np.array(point, copy=True)
    steps = np.maximum(1e-5, 1e-4 * np.maximum(1.0, np.abs(point)))
    for index, (lower, upper) in enumerate(bounds):
        base_step = min(steps[index], (upper - lower) / 4.0)
        evaluation_point[index] = np.clip(
            evaluation_point[index],
            lower + base_step,
            upper - base_step,
        )
        steps[index] = min(
            base_step,
            (evaluation_point[index] - lower) / 2.0,
            (upper - evaluation_point[index]) / 2.0,
        )
    result = np.empty((len(left_indices), len(right_indices)), dtype=np.float64)
    for left_position, left in enumerate(left_indices):
        for right_position, right in enumerate(right_indices):
            left_step = steps[left]
            right_step = steps[right]
            plus_plus = evaluation_point.copy()
            plus_minus = evaluation_point.copy()
            minus_plus = evaluation_point.copy()
            minus_minus = evaluation_point.copy()
            plus_plus[[left, right]] += [left_step, right_step]
            plus_minus[[left, right]] += [left_step, -right_step]
            minus_plus[[left, right]] += [-left_step, right_step]
            minus_minus[[left, right]] -= [left_step, right_step]
            result[left_position, right_position] = (
                float(objective(plus_plus))
                - float(objective(plus_minus))
                - float(objective(minus_plus))
                + float(objective(minus_minus))
            ) / (4.0 * left_step * right_step)
    return result


def _unpack_population_values(
    values: NDArray[np.float64],
    *,
    base_parameter_names: tuple[str, ...],
    varying_parameters: tuple[str, ...],
    n_knots: int,
) -> NDArray[np.float64]:
    knot_values = np.empty((len(base_parameter_names), n_knots), dtype=np.float64)
    varying = set(varying_parameters)
    offset = 0
    for row, parameter in enumerate(base_parameter_names):
        if parameter in varying:
            knot_values[row] = values[offset : offset + n_knots]
            offset += n_knots
        else:
            knot_values[row] = values[offset]
            offset += 1
    if offset != len(values):
        raise ValueError("population estimates do not match trajectory coordinates")
    return knot_values


def _row_subject_indices(study: Study, subjects: tuple[Any, ...]) -> NDArray[np.intp]:
    subject_index = {subject: index for index, subject in enumerate(subjects)}
    return np.asarray(
        [subject_index[_scalar(subject)] for subject in study["subject"]],
        dtype=np.intp,
    )


def _validate_coordinates(
    clock: str,
    knots: tuple[float, ...],
    parameter_names: tuple[str, ...],
    subject_parameters: tuple[str, ...],
) -> None:
    if not isinstance(clock, str) or not clock:
        raise ValueError("trajectory clock must be a non-empty string")
    if len(knots) < 2 or not np.all(np.isfinite(knots)):
        raise ValueError("trajectory knots must contain at least two finite values")
    if any(right <= left for left, right in pairwise(knots)):
        raise ValueError("trajectory knots must be strictly increasing")
    if not parameter_names or len(set(parameter_names)) != len(parameter_names):
        raise ValueError("parameter names must be non-empty and unique")
    if not subject_parameters or len(set(subject_parameters)) != len(subject_parameters):
        raise ValueError("subject parameters must be non-empty and unique")
    if set(subject_parameters) - set(parameter_names):
        raise ValueError("subject parameters must be named model parameters")


def _scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value
