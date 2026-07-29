"""Partially pooled longitudinal Wiener parameter trajectories."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp

from behavio.models.base import (
    FitDiagnostics,
    FitResult,
    ModelDataError,
    Prediction,
    PredictionMode,
    UnsupportedPredictionMode,
    _protected_array,
)
from behavio.models.ddm import (
    _LOG_DENSITY_FLOOR,
    _numerical_hessian,
    _upper_boundary_probability,
    _wiener_log_density,
)
from behavio.models.glm import _format_time, _linear_time_basis
from behavio.models.smooth_ddm import (
    DriftDiffusionTrajectory,
    SmoothDriftDiffusionFitResult,
    SmoothWienerDriftDiffusion,
    _roughness_matrix,
    _simulate_trialwise_wiener,
)
from behavio.study import Study

_INVALID_OBJECTIVE = float(np.finfo(np.float64).max / 1e100)
_CONSTRAINT_PENALTY = 1e6
_SCALE_UNCERTAINTY_POLICIES = MappingProxyType(
    {
        "local": "local-expected-prior-curvature",
        "observed": "louis-observed-information",
        "supplemented": "supplemented-laplace-em",
    }
)


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
class UnseenSubjectPosteriorPrediction:
    """Empirical-Bayes random-effect prediction for entirely unseen subjects."""

    subjects: tuple[Any, ...]
    row_subject_indices: NDArray[np.intp]
    probability: NDArray[np.float64]
    draw_probabilities: NDArray[np.float64]
    draw_pointwise_log_probability: NDArray[np.float64]
    pointwise_marginal_log_probability: NDArray[np.float64]
    subject_joint_log_probability: NDArray[np.float64]
    subject_effective_draws: NDArray[np.float64]
    subject_log_probability_mcse: NDArray[np.float64]
    random_effect_draws: NDArray[np.float64]
    subject_parameters: tuple[str, ...]
    subject_parameter_scales: NDArray[np.float64]
    seed: int
    policy: str = "empirical-bayes-random-effect-monte-carlo"

    def __post_init__(self) -> None:
        subjects = tuple(_scalar(subject) for subject in self.subjects)
        row_subjects = _protected_array(self.row_subject_indices, dtype=np.intp)
        probability = _protected_array(self.probability, dtype=np.float64)
        draw_probabilities = _protected_array(self.draw_probabilities, dtype=np.float64)
        draw_pointwise = _protected_array(
            self.draw_pointwise_log_probability,
            dtype=np.float64,
        )
        pointwise = _protected_array(
            self.pointwise_marginal_log_probability,
            dtype=np.float64,
        )
        subject_joint = _protected_array(
            self.subject_joint_log_probability,
            dtype=np.float64,
        )
        effective_draws = _protected_array(self.subject_effective_draws, dtype=np.float64)
        monte_carlo_error = _protected_array(
            self.subject_log_probability_mcse,
            dtype=np.float64,
        )
        draws = _protected_array(self.random_effect_draws, dtype=np.float64)
        parameters = tuple(self.subject_parameters)
        scales = _protected_array(self.subject_parameter_scales, dtype=np.float64)
        if not subjects or len(set(subjects)) != len(subjects):
            raise ValueError("predictive subjects must be non-empty and unique")
        if row_subjects.ndim != 1 or np.any((row_subjects < 0) | (row_subjects >= len(subjects))):
            raise ValueError("row subject indices must reference predictive subjects")
        n_draws = draw_probabilities.shape[0] if draw_probabilities.ndim == 2 else -1
        if n_draws < 1 or draw_probabilities.shape[1:] != probability.shape:
            raise ValueError("draw probabilities must align with predicted rows")
        if draw_pointwise.shape != draw_probabilities.shape:
            raise ValueError("draw log probabilities must align with predicted rows")
        if probability.shape != row_subjects.shape or pointwise.shape != probability.shape:
            raise ValueError("predictive row summaries must have one value per row")
        if subject_joint.shape != (len(subjects),):
            raise ValueError("subject joint scores must have one value per subject")
        if (
            effective_draws.shape != subject_joint.shape
            or monte_carlo_error.shape != subject_joint.shape
        ):
            raise ValueError("Monte Carlo diagnostics must have one value per subject")
        expected_draws = (n_draws, len(subjects), len(parameters))
        if draws.ndim != 4 or draws.shape[:3] != expected_draws:
            raise ValueError("random-effect draws must align with draws, subjects, and parameters")
        if scales.shape != (len(parameters),) or np.any(scales <= 0):
            raise ValueError("predictive scales must align with subject parameters")
        if not np.all(np.isfinite(draw_probabilities)) or np.any(
            (draw_probabilities < 0) | (draw_probabilities > 1)
        ):
            raise ValueError("draw probabilities must be finite values between zero and one")
        if (
            not np.all(np.isfinite(probability))
            or not np.all(np.isfinite(draw_pointwise))
            or not np.all(np.isfinite(pointwise))
        ):
            raise ValueError("predictive row summaries must be finite")
        if not np.all(np.isfinite(subject_joint)) or not np.all(np.isfinite(draws)):
            raise ValueError("predictive scores and random effects must be finite")
        if (
            not np.all(np.isfinite(effective_draws))
            or np.any((effective_draws < 1.0) | (effective_draws > n_draws))
            or not np.all(np.isfinite(monte_carlo_error))
            or np.any(monte_carlo_error < 0)
        ):
            raise ValueError("Monte Carlo diagnostics must be finite and admissible")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("predictive seed must be a non-negative integer")
        if self.policy != "empirical-bayes-random-effect-monte-carlo":
            raise ValueError("unsupported unseen-subject prediction policy")
        object.__setattr__(self, "subjects", subjects)
        object.__setattr__(self, "row_subject_indices", row_subjects)
        object.__setattr__(self, "probability", probability)
        object.__setattr__(self, "draw_probabilities", draw_probabilities)
        object.__setattr__(self, "draw_pointwise_log_probability", draw_pointwise)
        object.__setattr__(self, "pointwise_marginal_log_probability", pointwise)
        object.__setattr__(self, "subject_joint_log_probability", subject_joint)
        object.__setattr__(self, "subject_effective_draws", effective_draws)
        object.__setattr__(self, "subject_log_probability_mcse", monte_carlo_error)
        object.__setattr__(self, "random_effect_draws", draws)
        object.__setattr__(self, "subject_parameters", parameters)
        object.__setattr__(self, "subject_parameter_scales", scales)

    @property
    def n_draws(self) -> int:
        return len(self.draw_probabilities)

    @property
    def prediction(self) -> Prediction:
        probability = np.clip(
            self.probability,
            np.finfo(float).tiny,
            1.0 - np.finfo(float).eps,
        )
        return Prediction(
            probability=probability,
            linear_predictor=np.log(probability) - np.log1p(-probability),
            mode=PredictionMode.FILTERED,
        )

    @property
    def subject_joint_log_probability_map(self) -> Mapping[Any, float]:
        return MappingProxyType(
            dict(
                zip(
                    self.subjects,
                    self.subject_joint_log_probability.tolist(),
                    strict=True,
                )
            )
        )


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
    subject_parameter_scales: NDArray[np.float64] | None = None
    subject_scale_standard_errors: NDArray[np.float64] | None = None
    subject_scale_local_standard_errors: NDArray[np.float64] | None = None
    subject_scale_covariance: NDArray[np.float64] | None = None
    subject_scale_em_rate_matrix: NDArray[np.float64] | None = None
    subject_scale_bounds: tuple[float, float] | None = None
    subject_scales_estimated: bool = False
    subject_scales_at_boundary: NDArray[np.bool_] | None = None
    scale_estimation_iterations: int = 0
    scale_estimation_converged: bool = True
    scale_estimation_policy: str = "fixed"
    subject_scale_uncertainty_policy: str = "fixed"
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
        scales = (
            np.full(len(subject_parameters), self.subject_scale, dtype=np.float64)
            if self.subject_parameter_scales is None
            else _protected_array(self.subject_parameter_scales, dtype=np.float64)
        )
        if scales.shape != (len(subject_parameters),) or not np.all(np.isfinite(scales)):
            raise ValueError("subject parameter scales must align with subject_parameters")
        if np.any(scales <= 0):
            raise ValueError("subject parameter scales must be positive")
        scale_standard_errors = (
            None
            if self.subject_scale_standard_errors is None
            else _protected_array(self.subject_scale_standard_errors, dtype=np.float64)
        )
        local_standard_errors = (
            None
            if self.subject_scale_local_standard_errors is None
            else _protected_array(
                self.subject_scale_local_standard_errors,
                dtype=np.float64,
            )
        )
        scale_covariance = (
            None
            if self.subject_scale_covariance is None
            else _protected_array(self.subject_scale_covariance, dtype=np.float64)
        )
        rate_matrix = (
            None
            if self.subject_scale_em_rate_matrix is None
            else _protected_array(self.subject_scale_em_rate_matrix, dtype=np.float64)
        )
        at_boundary = (
            None
            if self.subject_scales_at_boundary is None
            else _protected_array(self.subject_scales_at_boundary, dtype=np.bool_)
        )
        if not isinstance(self.subject_scales_estimated, bool):
            raise ValueError("subject_scales_estimated must be boolean")
        if not isinstance(self.scale_estimation_converged, bool):
            raise ValueError("scale_estimation_converged must be boolean")
        if (
            isinstance(self.scale_estimation_iterations, bool)
            or not isinstance(self.scale_estimation_iterations, int)
            or self.scale_estimation_iterations < 0
        ):
            raise ValueError("scale_estimation_iterations must be a non-negative integer")
        if self.subject_scales_estimated:
            bounds = _validate_scale_bounds(self.subject_scale_bounds)
            if np.any(scales < bounds[0]) or np.any(scales > bounds[1]):
                raise ValueError("estimated subject scales must lie within subject_scale_bounds")
            if (
                scale_standard_errors is None
                or scale_standard_errors.shape != scales.shape
                or not np.all(np.isfinite(scale_standard_errors))
                or np.any(scale_standard_errors < 0)
            ):
                raise ValueError("estimated subject scales require finite standard errors")
            if (
                local_standard_errors is None
                or local_standard_errors.shape != scales.shape
                or not np.all(np.isfinite(local_standard_errors))
                or np.any(local_standard_errors <= 0)
            ):
                raise ValueError("estimated subject scales require local standard errors")
            if (
                scale_covariance is None
                or scale_covariance.shape != (len(scales), len(scales))
                or not np.all(np.isfinite(scale_covariance))
            ):
                raise ValueError("estimated subject scales require a finite covariance")
            if (
                not np.allclose(scale_covariance, scale_covariance.T)
                or np.min(np.linalg.eigvalsh(scale_covariance)) < -1e-10
            ):
                raise ValueError("subject scale covariance must be positive semidefinite")
            if at_boundary is None or at_boundary.shape != scales.shape:
                raise ValueError("estimated subject scales require boundary indicators")
            if self.scale_estimation_policy != "laplace-em":
                raise ValueError("estimated subject scale policy must be 'laplace-em'")
            if self.subject_scale_uncertainty_policy in {
                "local-expected-prior-curvature",
                "louis-observed-information",
            }:
                if rate_matrix is not None:
                    raise ValueError(
                        "closed-form scale uncertainty cannot retain an EM rate matrix"
                    )
            elif self.subject_scale_uncertainty_policy == "supplemented-laplace-em":
                if (
                    rate_matrix is None
                    or rate_matrix.shape != scale_covariance.shape
                    or not np.all(np.isfinite(rate_matrix))
                ):
                    raise ValueError("supplemented scale uncertainty requires an EM rate matrix")
            else:
                raise ValueError("unsupported subject scale uncertainty policy")
            object.__setattr__(self, "subject_scale_bounds", bounds)
        elif (
            scale_standard_errors is not None
            or local_standard_errors is not None
            or scale_covariance is not None
            or rate_matrix is not None
            or self.subject_scale_bounds is not None
            or at_boundary is not None
            or self.scale_estimation_iterations != 0
            or not self.scale_estimation_converged
            or self.scale_estimation_policy != "fixed"
            or self.subject_scale_uncertainty_policy != "fixed"
        ):
            raise ValueError("fixed subject scales cannot retain estimation diagnostics")
        if self.unseen_subject_policy != "population-trajectory-plugin":
            raise ValueError("unseen_subject_policy must be 'population-trajectory-plugin'")
        if self.uncertainty_policy != "arrowhead-local-hessian":
            raise ValueError("uncertainty_policy must be 'arrowhead-local-hessian'")
        object.__setattr__(self, "subjects", subjects)
        object.__setattr__(self, "base_parameter_names", names)
        object.__setattr__(self, "subject_parameters", subject_parameters)
        object.__setattr__(self, "subject_deviations", deviations)
        object.__setattr__(self, "subject_standard_errors", standard_errors)
        object.__setattr__(
            self,
            "subject_parameter_scales",
            _protected_array(scales, dtype=np.float64),
        )
        object.__setattr__(self, "subject_scale_standard_errors", scale_standard_errors)
        object.__setattr__(
            self,
            "subject_scale_local_standard_errors",
            local_standard_errors,
        )
        object.__setattr__(self, "subject_scale_covariance", scale_covariance)
        object.__setattr__(self, "subject_scale_em_rate_matrix", rate_matrix)
        object.__setattr__(self, "subject_scales_at_boundary", at_boundary)

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

    @property
    def subject_scale_map(self) -> Mapping[str, float]:
        """Return the fitted or fixed scale for each subject-varying parameter."""

        return MappingProxyType(
            dict(
                zip(
                    self.subject_parameters,
                    self.subject_parameter_scales.tolist(),
                    strict=True,
                )
            )
        )

    @property
    def subject_scale_standard_error_map(self) -> Mapping[str, float] | None:
        """Return approximate scale uncertainty when scales were estimated."""

        if self.subject_scale_standard_errors is None:
            return None
        return MappingProxyType(
            dict(
                zip(
                    self.subject_parameters,
                    self.subject_scale_standard_errors.tolist(),
                    strict=True,
                )
            )
        )

    @property
    def subject_scale_at_boundary_map(self) -> Mapping[str, bool] | None:
        """Return whether each estimated scale met a declared bound."""

        if self.subject_scales_at_boundary is None:
            return None
        return MappingProxyType(
            dict(
                zip(
                    self.subject_parameters,
                    self.subject_scales_at_boundary.tolist(),
                    strict=True,
                )
            )
        )

    @property
    def subject_scale_confidence_intervals_95(
        self,
    ) -> Mapping[str, tuple[float, float]] | None:
        """Return approximate delta-method intervals in log-scale coordinates."""

        if self.subject_scale_standard_errors is None:
            return None
        return self._subject_scale_intervals(self.subject_scale_standard_errors)

    @property
    def subject_scale_local_confidence_intervals_95(
        self,
    ) -> Mapping[str, tuple[float, float]] | None:
        """Return uncorrected complete-data curvature ranges, which are not calibrated.

        These ranges use expected-prior curvature alone and are systematically narrower
        than the observed information allows, so they undercover their nominal rate. They
        are retained as an optimization diagnostic and as the comparison baseline for the
        corrections reported by ``subject_scale_confidence_intervals_95``.
        """

        if self.subject_scale_local_standard_errors is None:
            return None
        return self._subject_scale_intervals(self.subject_scale_local_standard_errors)

    @property
    def subject_scale_em_spectral_radius(self) -> float | None:
        """Return the largest absolute EM-rate eigenvalue when supplemented."""

        if self.subject_scale_em_rate_matrix is None:
            return None
        return float(np.max(np.abs(np.linalg.eigvals(self.subject_scale_em_rate_matrix))))

    def _subject_scale_intervals(
        self,
        standard_errors: NDArray[np.float64],
    ) -> Mapping[str, tuple[float, float]]:
        lower_bound, upper_bound = (
            self.subject_scale_bounds if self.subject_scale_bounds is not None else (0.0, np.inf)
        )
        intervals = {}
        for parameter, scale, standard_error in zip(
            self.subject_parameters,
            self.subject_parameter_scales,
            standard_errors,
            strict=True,
        ):
            log_standard_error = float(standard_error / scale)
            log_scale = float(np.log(scale))
            intervals[parameter] = (
                float(
                    np.exp(
                        max(
                            np.log(lower_bound) if lower_bound > 0 else -np.inf,
                            log_scale - 1.96 * log_standard_error,
                        )
                    )
                ),
                float(
                    np.exp(
                        min(
                            np.log(upper_bound),
                            log_scale + 1.96 * log_standard_error,
                        )
                    )
                ),
            )
        return MappingProxyType(intervals)


@dataclass(frozen=True, slots=True)
class HierarchicalSmoothWienerDriftDiffusion(SmoothWienerDriftDiffusion):
    """Smooth population Wiener paths with shrunken subject-deviation paths.

    Population parameters retain the smooth Wiener's natural-scale coordinates. Selected
    varying parameters receive additive subject deviations with named fixed or estimated
    Gaussian scales and a time-scaled first-difference penalty. Unseen subjects use the
    population path.
    """

    subject_parameters: tuple[str, ...] | None = None
    subject_scale: float = 0.25
    subject_parameter_scales: Mapping[str, float] | None = None
    estimate_subject_scales: bool = False
    subject_scale_bounds: tuple[float, float] = (0.05, 2.0)
    scale_max_iterations: int = 12
    scale_tolerance: float = 0.01
    subject_scale_uncertainty: str = "observed"
    subject_smoothness: float = 10.0

    def __post_init__(self) -> None:
        SmoothWienerDriftDiffusion.__post_init__(self)
        if self.shared_trajectory:
            raise ValueError("shared_trajectory is not used by the hierarchical model")
        if not np.isfinite(self.subject_scale) or self.subject_scale <= 0:
            raise ValueError("subject_scale must be finite and positive")
        if not isinstance(self.estimate_subject_scales, bool):
            raise ValueError("estimate_subject_scales must be boolean")
        scale_bounds = _validate_scale_bounds(self.subject_scale_bounds)
        if (
            isinstance(self.scale_max_iterations, bool)
            or not isinstance(self.scale_max_iterations, int)
            or self.scale_max_iterations < 1
        ):
            raise ValueError("scale_max_iterations must be a positive integer")
        if not np.isfinite(self.scale_tolerance) or self.scale_tolerance <= 0:
            raise ValueError("scale_tolerance must be finite and positive")
        if self.subject_scale_uncertainty not in {"local", "observed", "supplemented"}:
            raise ValueError(
                "subject_scale_uncertainty must be 'local', 'observed', or 'supplemented'"
            )
        if self.subject_scale_uncertainty == "supplemented" and not self.estimate_subject_scales:
            raise ValueError(
                "supplemented subject-scale uncertainty requires estimate_subject_scales=True"
            )
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
        if self.subject_parameter_scales is None:
            parameter_scales = None
            configured_scales = np.full(
                len(subject_parameters), self.subject_scale, dtype=np.float64
            )
        else:
            parameter_scales = {
                str(parameter): float(scale)
                for parameter, scale in self.subject_parameter_scales.items()
            }
            if set(parameter_scales) != set(subject_parameters):
                raise ValueError(
                    "subject_parameter_scales must contain every subject parameter exactly"
                )
            configured_scales = np.asarray(
                [parameter_scales[parameter] for parameter in subject_parameters]
            )
            if not np.all(np.isfinite(configured_scales)) or np.any(configured_scales <= 0):
                raise ValueError("subject parameter scales must be finite and positive")
            parameter_scales = MappingProxyType(parameter_scales)
        if self.estimate_subject_scales and (
            np.any(configured_scales < scale_bounds[0])
            or np.any(configured_scales > scale_bounds[1])
        ):
            raise ValueError("initial subject scales must lie within subject_scale_bounds")
        object.__setattr__(self, "subject_parameters", subject_parameters)
        object.__setattr__(self, "subject_parameter_scales", parameter_scales)
        object.__setattr__(self, "subject_scale_bounds", scale_bounds)

    @property
    def model_name(self) -> str:
        return "hierarchical-smooth-wiener-drift-diffusion"

    @property
    def signature(self) -> str:
        covariates = ",".join(self.covariates)
        knots = ",".join(_format_time(knot) for knot in self.knots)
        varying = ",".join(self.varying_parameters or ())
        subject_parameters = ",".join(self.subject_parameters or ())
        subject_scales = ",".join(
            f"{parameter}:{_format_time(scale)}"
            for parameter, scale in zip(
                self.subject_parameters or (),
                self._configured_subject_scales(),
                strict=True,
            )
        )
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
            f"subject_parameter_scales={subject_scales};"
            f"estimate_subject_scales={self.estimate_subject_scales};"
            f"subject_scale_bounds={self.subject_scale_bounds[0]},{self.subject_scale_bounds[1]};"
            f"scale_max_iterations={self.scale_max_iterations};"
            f"scale_tolerance={self.scale_tolerance};"
            f"subject_scale_uncertainty={self.subject_scale_uncertainty};"
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

        def objective_for(subject_scales: NDArray[np.float64]) -> Any:
            subject_penalties = self._subject_penalty_matrices(subject_scales)

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
                    np.einsum(
                        "sdk,dkl,sdl->",
                        deviations,
                        subject_penalties,
                        deviations,
                    )
                )
                return float(
                    -np.sum(log_density)
                    + population_roughness
                    + subject_roughness
                    + constraint_penalty
                )

            return objective

        def optimize_joint(objective: Any, initial_points: Sequence[NDArray[np.float64]]):
            fitted = [
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
                for start in initial_points
            ]
            objectives = np.asarray(
                [float(result.fun) if np.isfinite(result.fun) else np.inf for result in fitted]
            )
            finite_indices = np.flatnonzero(np.isfinite(objectives)).tolist()
            if not finite_indices:
                messages = "; ".join(str(result.message) for result in fitted)
                raise ModelDataError(
                    f"all hierarchical smooth DDM restarts were non-finite: {messages}"
                )
            successful_indices = [index for index in finite_indices if fitted[index].success]
            eligible_indices = successful_indices if successful_indices else finite_indices
            selected_index = min(
                eligible_indices,
                key=lambda index: float(objectives[index]),
            )
            return fitted, objectives, selected_index

        log_scale_bounds = tuple(np.log(self.subject_scale_bounds))

        def em_scale_update(
            input_scales: NDArray[np.float64],
            initial_points: Sequence[NDArray[np.float64]],
        ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
            scale_objective = objective_for(input_scales)
            scale_results, _, scale_selected = optimize_joint(
                scale_objective,
                initial_points,
            )
            scale_point = np.asarray(scale_results[scale_selected].x, dtype=np.float64)
            scale_deviations = scale_point[n_population:].reshape(
                len(subjects),
                len(self.subject_parameters or ()),
                len(self.knots),
            )
            conditional_covariances = _conditional_deviation_covariances(
                scale_objective,
                scale_point,
                n_population=n_population,
                deviation_bounds=deviation_bounds,
                n_subjects=len(subjects),
            )
            updated_scales = np.empty_like(input_scales)
            for parameter_index in range(len(self.subject_parameters or ())):
                block_start = parameter_index * len(self.knots)
                block_stop = block_start + len(self.knots)
                covariance_blocks = tuple(
                    covariance[block_start:block_stop, block_start:block_stop]
                    for covariance in conditional_covariances
                )
                parameter_deviations = scale_deviations[:, parameter_index, :]
                scale_fit = minimize_scalar(
                    _expected_subject_prior_objective,
                    args=(
                        parameter_deviations,
                        covariance_blocks,
                        self.subject_smoothness,
                        self.knots,
                    ),
                    bounds=log_scale_bounds,
                    method="bounded",
                    options={"xatol": self.scale_tolerance / 10.0},
                )
                updated_scales[parameter_index] = float(np.exp(scale_fit.x))
            return updated_scales, scale_point

        subject_scales = self._configured_subject_scales()
        scale_standard_errors: NDArray[np.float64] | None = None
        scale_local_standard_errors: NDArray[np.float64] | None = None
        scale_covariance: NDArray[np.float64] | None = None
        scale_rate_matrix: NDArray[np.float64] | None = None
        scale_at_boundary: NDArray[np.bool_] | None = None
        scale_iterations = 0
        scale_converged = True
        if self.estimate_subject_scales:
            scale_converged = False
            warm_start: NDArray[np.float64] | None = None
            for iteration in range(1, self.scale_max_iterations + 1):
                scale_iterations = iteration
                initial_points = starts if warm_start is None else (warm_start,)
                updated_scales, scale_point = em_scale_update(
                    subject_scales,
                    initial_points,
                )
                maximum_change = float(
                    np.max(np.abs(np.log(updated_scales) - np.log(subject_scales)))
                )
                subject_scales = updated_scales
                warm_start = scale_point
                if maximum_change <= self.scale_tolerance:
                    scale_converged = True
                    break
            scale_at_boundary = _scales_at_bounds(
                subject_scales,
                self.subject_scale_bounds,
                tolerance=max(self.scale_tolerance, 1e-4),
            )

        objective = objective_for(subject_scales)
        results, restart_objectives, selected = optimize_joint(objective, starts)
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
        if self.estimate_subject_scales:
            final_conditional_covariances = _conditional_deviation_covariances(
                objective,
                joint_estimates,
                n_population=n_population,
                deviation_bounds=deviation_bounds,
                n_subjects=len(subjects),
            )
            scale_local_standard_errors = np.empty_like(subject_scales)
            for parameter_index, scale in enumerate(subject_scales):
                block_start = parameter_index * len(self.knots)
                block_stop = block_start + len(self.knots)
                covariance_blocks = tuple(
                    covariance[block_start:block_stop, block_start:block_stop]
                    for covariance in final_conditional_covariances
                )
                scale_local_standard_errors[parameter_index] = _scale_standard_error(
                    scale,
                    deviations[:, parameter_index, :],
                    covariance_blocks,
                    subject_smoothness=self.subject_smoothness,
                    knots=self.knots,
                )
            local_log_covariance = np.diag((scale_local_standard_errors / subject_scales) ** 2)
            if self.subject_scale_uncertainty == "supplemented":
                if not scale_converged:
                    raise ModelDataError("supplemented scale uncertainty requires EM convergence")
                scale_rate_matrix = np.empty(
                    (len(subject_scales), len(subject_scales)),
                    dtype=np.float64,
                )
                center = np.log(subject_scales)
                step = max(0.02, self.scale_tolerance / 2.0)
                for input_parameter in range(len(subject_scales)):
                    lower_point = center.copy()
                    upper_point = center.copy()
                    lower_point[input_parameter] = max(
                        lower_point[input_parameter] - step,
                        log_scale_bounds[0],
                    )
                    upper_point[input_parameter] = min(
                        upper_point[input_parameter] + step,
                        log_scale_bounds[1],
                    )
                    lower_update, _ = em_scale_update(
                        np.exp(lower_point),
                        (joint_estimates,),
                    )
                    upper_update, _ = em_scale_update(
                        np.exp(upper_point),
                        (joint_estimates,),
                    )
                    scale_rate_matrix[:, input_parameter] = (
                        np.log(upper_update) - np.log(lower_update)
                    ) / (upper_point[input_parameter] - lower_point[input_parameter])
                if np.max(np.abs(np.linalg.eigvals(scale_rate_matrix))) >= 1.0:
                    raise ModelDataError(
                        "supplemented scale uncertainty requires a stable EM rate matrix"
                    )
                complete_information = np.linalg.pinv(
                    local_log_covariance,
                    hermitian=True,
                )
                observed_information = (
                    np.eye(len(subject_scales)) - scale_rate_matrix
                ) @ complete_information
                observed_information = 0.5 * (observed_information + observed_information.T)
                if np.min(np.linalg.eigvalsh(observed_information)) <= 0:
                    raise ModelDataError("supplemented scale information is not positive definite")
                log_covariance = np.linalg.pinv(
                    observed_information,
                    hermitian=True,
                )
            elif self.subject_scale_uncertainty == "observed":
                observed_information = _louis_scale_information(
                    subject_scales,
                    deviations,
                    final_conditional_covariances,
                    subject_smoothness=self.subject_smoothness,
                    knots=self.knots,
                )
                if np.min(np.linalg.eigvalsh(observed_information)) <= 0:
                    raise ModelDataError("observed scale information is not positive definite")
                log_covariance = np.linalg.pinv(observed_information, hermitian=True)
            else:
                log_covariance = local_log_covariance
            scale_covariance = np.diag(subject_scales) @ log_covariance @ np.diag(subject_scales)
            scale_standard_errors = np.sqrt(np.maximum(np.diag(scale_covariance), 0.0))
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
            subject_parameter_scales=subject_scales,
            subject_scale_standard_errors=scale_standard_errors,
            subject_scale_local_standard_errors=scale_local_standard_errors,
            subject_scale_covariance=scale_covariance,
            subject_scale_em_rate_matrix=scale_rate_matrix,
            subject_scale_bounds=self.subject_scale_bounds
            if self.estimate_subject_scales
            else None,
            subject_scales_estimated=self.estimate_subject_scales,
            subject_scales_at_boundary=scale_at_boundary,
            scale_estimation_iterations=scale_iterations,
            scale_estimation_converged=scale_converged,
            scale_estimation_policy="laplace-em" if self.estimate_subject_scales else "fixed",
            subject_scale_uncertainty_policy=(
                _SCALE_UNCERTAINTY_POLICIES[self.subject_scale_uncertainty]
                if self.estimate_subject_scales
                else "fixed"
            ),
        )

    def predict_new_subjects(
        self,
        study: Study,
        fit: FitResult,
        *,
        n_draws: int,
        seed: int,
    ) -> UnseenSubjectPosteriorPrediction:
        """Integrate fitted random-effect paths for entirely unseen subjects."""

        hierarchical_fit = self._validated_hierarchical_fit(fit)
        if isinstance(n_draws, bool) or not isinstance(n_draws, int) or n_draws < 2:
            raise ValueError("n_draws must be an integer of at least two")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("predictive seed must be a non-negative integer")
        subjects = tuple(_scalar(subject) for subject in study.subjects)
        overlap = set(subjects) & set(hierarchical_fit.subjects)
        if overlap:
            raise ValueError("predict_new_subjects requires entirely unseen subjects")

        outcomes = self._outcomes(study)
        response_times = self.response_time.read(study).seconds
        features = self._feature_matrix(study)
        basis = self._time_basis(study)
        row_subjects = _row_subject_indices(study, subjects)
        generator = np.random.default_rng(seed)
        n_rows = len(study)
        n_coefficients = len(self.coefficient_names)
        probabilities = np.empty((n_draws, n_rows), dtype=np.float64)
        log_probabilities = np.empty_like(probabilities)
        deviations = np.empty(
            (
                n_draws,
                len(subjects),
                len(hierarchical_fit.subject_parameters),
                len(self.knots),
            ),
            dtype=np.float64,
        )
        for draw in range(n_draws):
            deviations[draw] = self._draw_subject_deviations(
                subjects,
                hierarchical_fit.population_knot_values,
                generator,
                scales=hierarchical_fit.subject_parameter_scales,
            )
            trial_values = self._subject_trial_values_from_basis(
                basis,
                hierarchical_fit.estimates,
                deviations[draw],
                row_subjects,
            )
            drifts = np.sum(features * trial_values[:, :n_coefficients], axis=1)
            probabilities[draw] = _upper_boundary_probability(
                drifts,
                boundary=trial_values[:, n_coefficients],
                starting_bias=trial_values[:, n_coefficients + 1],
            )
            log_probabilities[draw] = _wiener_log_density(
                response_times - trial_values[:, n_coefficients + 2],
                outcomes,
                drifts,
                boundary=trial_values[:, n_coefficients],
                starting_bias=trial_values[:, n_coefficients + 1],
                terms=self.density_terms,
            )

        log_draws = np.log(float(n_draws))
        subject_joint = np.empty(len(subjects), dtype=np.float64)
        effective_draws = np.empty(len(subjects), dtype=np.float64)
        monte_carlo_error = np.empty(len(subjects), dtype=np.float64)
        for subject in range(len(subjects)):
            log_weights = np.sum(
                log_probabilities[:, row_subjects == subject],
                axis=1,
            )
            log_weight_sum = float(logsumexp(log_weights))
            subject_joint[subject] = log_weight_sum - log_draws
            normalized_weights = np.exp(log_weights - log_weight_sum)
            effective_draws[subject] = 1.0 / float(np.sum(normalized_weights**2))
            shifted_weights = np.exp(log_weights - float(np.max(log_weights)))
            monte_carlo_error[subject] = float(
                np.std(shifted_weights, ddof=1) / np.sqrt(n_draws) / np.mean(shifted_weights)
            )
        return UnseenSubjectPosteriorPrediction(
            subjects=subjects,
            row_subject_indices=row_subjects,
            probability=np.mean(probabilities, axis=0),
            draw_probabilities=probabilities,
            draw_pointwise_log_probability=log_probabilities,
            pointwise_marginal_log_probability=logsumexp(
                log_probabilities,
                axis=0,
            )
            - log_draws,
            subject_joint_log_probability=subject_joint,
            subject_effective_draws=effective_draws,
            subject_log_probability_mcse=monte_carlo_error,
            random_effect_draws=deviations,
            subject_parameters=hierarchical_fit.subject_parameters,
            subject_parameter_scales=hierarchical_fit.subject_parameter_scales,
            seed=seed,
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

    def _configured_subject_scales(self) -> NDArray[np.float64]:
        if self.subject_parameter_scales is None:
            return np.full(
                len(self.subject_parameters or ()),
                self.subject_scale,
                dtype=np.float64,
            )
        return np.asarray(
            [
                self.subject_parameter_scales[parameter]
                for parameter in self.subject_parameters or ()
            ],
            dtype=np.float64,
        )

    def _subject_penalty_matrices(
        self,
        scales: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        subject_scales = self._configured_subject_scales() if scales is None else scales
        identity = np.eye(len(self.knots), dtype=np.float64)
        roughness = self.subject_smoothness * _roughness_matrix(self.knots)
        return np.stack([identity / scale**2 + roughness for scale in subject_scales])

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
        *,
        scales: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        covariances = tuple(
            np.linalg.pinv(penalty, hermitian=True)
            for penalty in self._subject_penalty_matrices(scales)
        )
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
                        covariances[deviation_index],
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


def _conditional_deviation_covariances(
    objective: Any,
    point: NDArray[np.float64],
    *,
    n_population: int,
    deviation_bounds: Sequence[tuple[float, float]],
    n_subjects: int,
) -> tuple[NDArray[np.float64], ...]:
    """Approximate each deviation block conditional on the population coordinates."""

    n_deviation = len(deviation_bounds)
    covariances = []
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

        hessian = _numerical_hessian(
            with_deviation,
            point[start:stop],
            deviation_bounds,
        )
        covariance = np.linalg.pinv(hessian, hermitian=True)
        covariances.append(0.5 * (covariance + covariance.T))
    return tuple(covariances)


def _expected_subject_prior_objective(
    log_scale: float,
    deviations: NDArray[np.float64],
    covariance_blocks: tuple[NDArray[np.float64], ...],
    subject_smoothness: float,
    knots: tuple[float, ...],
) -> float:
    """Expected normalized Gaussian-prior loss for one parameter's path scale."""

    scale = float(np.exp(log_scale))
    precision = np.eye(len(knots), dtype=np.float64) / scale**2
    precision += subject_smoothness * _roughness_matrix(knots)
    sign, log_determinant = np.linalg.slogdet(precision)
    if sign <= 0 or not np.isfinite(log_determinant):
        return _INVALID_OBJECTIVE
    quadratic = 0.0
    for deviation, covariance in zip(deviations, covariance_blocks, strict=True):
        second_moment = covariance + np.outer(deviation, deviation)
        quadratic += float(np.trace(precision @ second_moment))
    return 0.5 * (quadratic - len(deviations) * float(log_determinant))


def _louis_scale_information(
    scales: NDArray[np.float64],
    deviations: NDArray[np.float64],
    covariances: tuple[NDArray[np.float64], ...],
    *,
    subject_smoothness: float,
    knots: tuple[float, ...],
) -> NDArray[np.float64]:
    """Return observed log-scale information as complete minus missing information.

    The expected-prior curvature the M-step minimizes is complete-data information, which
    is never smaller than the observed information the marginal likelihood carries. Louis'
    identity subtracts the conditional variance of the complete-data score, which for a
    Gaussian deviation prior is a closed form in the conditional means and covariances.
    """

    n_knots = len(knots)
    n_parameters = len(scales)
    n_subjects = len(covariances)
    roughness = _roughness_matrix(knots)
    precisions = np.asarray(scales, dtype=np.float64) ** -2.0
    blocks = tuple(slice(index * n_knots, (index + 1) * n_knots) for index in range(n_parameters))
    complete = np.zeros((n_parameters, n_parameters), dtype=np.float64)
    for index, precision in enumerate(precisions):
        prior_precision = precision * np.eye(n_knots) + subject_smoothness * roughness
        prior_covariance = np.linalg.inv(prior_precision)
        block = blocks[index]
        second_moment = float(np.sum(deviations[:, index, :] ** 2)) + sum(
            float(np.trace(covariance[block, block])) for covariance in covariances
        )
        complete[index, index] = 2.0 * precision * (
            second_moment - n_subjects * float(np.trace(prior_covariance))
        ) + 2.0 * n_subjects * precision**2 * float(np.sum(prior_covariance * prior_covariance))
    missing = np.zeros((n_parameters, n_parameters), dtype=np.float64)
    for row in range(n_parameters):
        for column in range(row + 1):
            total = 0.0
            for subject, covariance in enumerate(covariances):
                cross = covariance[blocks[row], blocks[column]]
                left = deviations[subject, row, :]
                right = deviations[subject, column, :]
                total += 2.0 * float(np.sum(cross * cross)) + 4.0 * float(left @ cross @ right)
            value = precisions[row] * precisions[column] * total
            missing[row, column] = missing[column, row] = value
    information = complete - missing
    return 0.5 * (information + information.T)


def _scale_standard_error(
    scale: float,
    deviations: NDArray[np.float64],
    covariance_blocks: tuple[NDArray[np.float64], ...],
    *,
    subject_smoothness: float,
    knots: tuple[float, ...],
) -> float:
    """Return local expected-prior curvature uncertainty on the natural scale."""

    center = float(np.log(scale))
    step = 1e-3
    objective = lambda value: _expected_subject_prior_objective(  # noqa: E731
        value,
        deviations,
        covariance_blocks,
        subject_smoothness,
        knots,
    )
    curvature = (
        objective(center + step) - 2.0 * objective(center) + objective(center - step)
    ) / step**2
    safe_curvature = max(float(curvature), np.finfo(np.float64).eps)
    return float(scale / np.sqrt(safe_curvature))


def _validate_scale_bounds(
    bounds: tuple[float, float] | None,
) -> tuple[float, float]:
    if bounds is None or len(bounds) != 2:
        raise ValueError("subject_scale_bounds must contain lower and upper bounds")
    lower, upper = (float(value) for value in bounds)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower <= 0 or upper <= lower:
        raise ValueError("subject_scale_bounds must be finite, positive, and increasing")
    return lower, upper


def _scales_at_bounds(
    scales: NDArray[np.float64],
    bounds: tuple[float, float],
    *,
    tolerance: float,
) -> NDArray[np.bool_]:
    """Mark variance components within a log-scale tolerance of either bound."""

    lower, upper = _validate_scale_bounds(bounds)
    log_scales = np.log(scales)
    return _protected_array(
        (np.abs(log_scales - np.log(lower)) <= tolerance)
        | (np.abs(log_scales - np.log(upper)) <= tolerance),
        dtype=np.bool_,
    )


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
