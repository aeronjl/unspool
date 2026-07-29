"""A bounded partial-pooling Bernoulli history GLM."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import expit

from behavio._internal.arrays import protected_array
from behavio.models._kernels.bernoulli import fit_bernoulli, ordered_session_indices
from behavio.models._kernels.curvature import relative_steps, value_difference_hessian
from behavio.models.base import (
    FitDiagnostics,
    FitResult,
    ModelDataError,
    Prediction,
    PredictionMode,
)
from behavio.models.glm import BernoulliHistoryGLM
from behavio.study import Study


@dataclass(frozen=True, slots=True)
class HierarchicalGLMSimulation:
    """Observed choices paired with their unexposed population and subject truth."""

    study: Study
    subjects: tuple[Any, ...]
    coefficient_names: tuple[str, ...]
    population_coefficients: NDArray[np.float64]
    subject_deviations: NDArray[np.float64]

    def __post_init__(self) -> None:
        subjects = tuple(_scalar(subject) for subject in self.subjects)
        names = tuple(self.coefficient_names)
        population = protected_array(self.population_coefficients, dtype=np.float64)
        deviations = protected_array(self.subject_deviations, dtype=np.float64)
        if subjects != tuple(_scalar(subject) for subject in self.study.subjects):
            raise ValueError("simulation subjects must match the study's subject order")
        if not names or len(set(names)) != len(names):
            raise ValueError("coefficient names must be non-empty and unique")
        if population.shape != (len(names),) or deviations.shape != (len(subjects), len(names)):
            raise ValueError(
                "population and subject truth must align with subjects and coefficients"
            )
        if not np.all(np.isfinite(population)) or not np.all(np.isfinite(deviations)):
            raise ValueError("simulation coefficients must be finite")
        object.__setattr__(self, "subjects", subjects)
        object.__setattr__(self, "coefficient_names", names)
        object.__setattr__(self, "population_coefficients", population)
        object.__setattr__(self, "subject_deviations", deviations)

    @property
    def subject_coefficients(self) -> NDArray[np.float64]:
        """Return realized population-plus-deviation coefficients by subject."""

        return protected_array(
            self.population_coefficients[None, :] + self.subject_deviations,
            dtype=np.float64,
        )


@dataclass(frozen=True, slots=True)
class HierarchicalGLMFitResult(FitResult):
    """Population coefficients and shrunken deviations from one joint MAP fit."""

    subjects: tuple[Any, ...]
    subject_deviations: NDArray[np.float64]
    subject_standard_errors: NDArray[np.float64]
    subject_scale: float
    subject_scale_standard_error: float | None = None
    subject_scale_bounds: tuple[float, float] | None = None
    subject_scale_estimated: bool = False
    subject_scale_at_boundary: bool = False
    unseen_subject_policy: str = "population-mean-plugin"

    def __post_init__(self) -> None:
        FitResult.__post_init__(self)
        subjects = tuple(_scalar(subject) for subject in self.subjects)
        deviations = protected_array(self.subject_deviations, dtype=np.float64)
        standard_errors = protected_array(self.subject_standard_errors, dtype=np.float64)
        expected_shape = (len(subjects), len(self.parameter_names))
        if not subjects or len(set(subjects)) != len(subjects):
            raise ValueError("fit subjects must be non-empty and unique")
        if deviations.shape != expected_shape or standard_errors.shape != expected_shape:
            raise ValueError("subject estimates must have one row per fitted subject")
        if not np.all(np.isfinite(deviations)) or not np.all(np.isfinite(standard_errors)):
            raise ValueError("subject estimates and standard errors must be finite")
        if np.any(standard_errors < 0):
            raise ValueError("subject standard errors must be non-negative")
        if not np.isfinite(self.subject_scale) or self.subject_scale <= 0:
            raise ValueError("subject_scale must be finite and positive")
        if not isinstance(self.subject_scale_estimated, bool):
            raise ValueError("subject_scale_estimated must be boolean")
        if not isinstance(self.subject_scale_at_boundary, bool):
            raise ValueError("subject_scale_at_boundary must be boolean")
        if self.subject_scale_estimated:
            if (
                self.subject_scale_standard_error is None
                or not np.isfinite(self.subject_scale_standard_error)
                or self.subject_scale_standard_error < 0
            ):
                raise ValueError(
                    "estimated subject scales require a finite non-negative standard error"
                )
            bounds = _validate_scale_bounds(self.subject_scale_bounds)
            if not bounds[0] <= self.subject_scale <= bounds[1]:
                raise ValueError("estimated subject_scale must lie within subject_scale_bounds")
            object.__setattr__(self, "subject_scale_bounds", bounds)
        elif (
            self.subject_scale_standard_error is not None
            or self.subject_scale_bounds is not None
            or self.subject_scale_at_boundary
        ):
            raise ValueError("fixed subject scales cannot retain estimation uncertainty")
        if self.unseen_subject_policy != "population-mean-plugin":
            raise ValueError("unseen_subject_policy must be 'population-mean-plugin'")
        object.__setattr__(self, "subjects", subjects)
        object.__setattr__(self, "subject_deviations", deviations)
        object.__setattr__(self, "subject_standard_errors", standard_errors)

    @property
    def subject_coefficients(self) -> NDArray[np.float64]:
        """Return population-plus-deviation coefficient estimates by fitted subject."""

        return protected_array(self.estimates[None, :] + self.subject_deviations, dtype=np.float64)

    def coefficients_for(self, subject: Any) -> Mapping[str, float]:
        """Return fitted-subject coefficients or the declared unseen-subject plug-in."""

        subject_key = _scalar(subject)
        try:
            index = self.subjects.index(subject_key)
        except ValueError:
            values = self.estimates
        else:
            values = self.estimates + self.subject_deviations[index]
        return MappingProxyType(dict(zip(self.parameter_names, values.tolist(), strict=True)))

    def subject_was_fitted(self, subject: Any) -> bool:
        """Report whether prediction can use an estimated subject deviation."""

        return _scalar(subject) in self.subjects

    @property
    def subject_scale_confidence_interval_95(self) -> tuple[float, float] | None:
        """Return a delta-method log-scale interval when the scale was estimated."""

        if self.subject_scale_standard_error is None:
            return None
        log_standard_error = self.subject_scale_standard_error / self.subject_scale
        return (
            float(self.subject_scale * np.exp(-1.96 * log_standard_error)),
            float(self.subject_scale * np.exp(1.96 * log_standard_error)),
        )


@dataclass(frozen=True, slots=True)
class HierarchicalBernoulliHistoryGLM(BernoulliHistoryGLM):
    """Static Bernoulli GLM with Gaussian subject deviations.

    The joint maximum-a-posteriori fit estimates one population coefficient vector and one
    shrunken deviation vector per training subject. By default, ``subject_scale`` is fixed
    before fitting. When ``estimate_subject_scale=True``, it is an initial value for a
    bounded Laplace marginal-likelihood estimate. Predictions for unseen subjects use the
    population coefficient plug-in explicitly.
    """

    subject_scale: float = 0.5
    estimate_subject_scale: bool = False
    subject_scale_bounds: tuple[float, float] = (0.05, 2.0)

    def __post_init__(self) -> None:
        BernoulliHistoryGLM.__post_init__(self)
        if not np.isfinite(self.subject_scale) or self.subject_scale <= 0:
            raise ValueError("subject_scale must be finite and positive")
        if not isinstance(self.estimate_subject_scale, bool):
            raise ValueError("estimate_subject_scale must be boolean")
        bounds = _validate_scale_bounds(self.subject_scale_bounds)
        if self.estimate_subject_scale and not bounds[0] <= self.subject_scale <= bounds[1]:
            raise ValueError("the initial subject_scale must lie within subject_scale_bounds")
        object.__setattr__(self, "subject_scale_bounds", bounds)

    @property
    def model_name(self) -> str:
        return "hierarchical-bernoulli-history-glm"

    @property
    def signature(self) -> str:
        covariates = ",".join(self.covariates)
        return (
            f"{self.model_name}[outcome={self.outcome};covariates={covariates};"
            f"choice_lags={self.choice_lags};l2={self.l2};subject_scale={self.subject_scale};"
            f"estimate_subject_scale={self.estimate_subject_scale};"
            f"subject_scale_bounds={self.subject_scale_bounds[0]},"
            f"{self.subject_scale_bounds[1]}{self._design_signature}]"
        )

    def simulate(
        self,
        design: Study,
        parameters: Mapping[str, float],
        *,
        seed: int | np.random.Generator,
    ) -> Study:
        """Generate choices without exposing realized subject effects as observed data."""

        return self.simulate_with_effects(design, parameters, seed=seed).study

    def simulate_with_effects(
        self,
        design: Study,
        parameters: Mapping[str, float],
        *,
        seed: int | np.random.Generator,
    ) -> HierarchicalGLMSimulation:
        """Generate choices and retain the realized random-effect truth separately."""

        population = self._parameter_vector(parameters)
        covariates = self._covariate_matrix(design)
        generator = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        subjects = tuple(_scalar(subject) for subject in design.subjects)
        deviations = generator.normal(
            loc=0.0,
            scale=self.subject_scale,
            size=(len(subjects), len(self.coefficient_names)),
        )
        subject_index = {subject: index for index, subject in enumerate(subjects)}
        choices = np.zeros(len(design), dtype=np.int8)
        history_start = 1 + len(self.covariates)

        for session_indices in ordered_session_indices(design):
            subject = _scalar(design["subject"][session_indices[0]])
            coefficients = population + deviations[subject_index[subject]]
            generated_history: list[float] = []
            for index in session_indices:
                linear_predictor = coefficients[0]
                if self.covariates:
                    linear_predictor += float(covariates[index] @ coefficients[1:history_start])
                for lag in range(1, self.choice_lags + 1):
                    history_value = (
                        generated_history[-lag] if len(generated_history) >= lag else 0.0
                    )
                    linear_predictor += coefficients[history_start + lag - 1] * history_value
                choice = int(generator.binomial(1, expit(linear_predictor)))
                choices[index] = choice
                generated_history.append(2.0 * choice - 1.0)

        columns = {name: design[name] for name in design.columns}
        columns[self.outcome] = choices
        return HierarchicalGLMSimulation(
            study=Study(columns),
            subjects=subjects,
            coefficient_names=self.coefficient_names,
            population_coefficients=population,
            subject_deviations=deviations,
        )

    def fit(self, study: Study) -> HierarchicalGLMFitResult:
        """Jointly fit population coefficients and shrunken subject deviations."""

        subjects = tuple(_scalar(subject) for subject in study.subjects)
        if len(subjects) < 2:
            raise ModelDataError("hierarchical fitting requires at least two subjects")
        outcomes = self._outcomes(study)
        features = self._base_feature_matrix(study, outcomes)
        if self.estimate_subject_scale:
            return self._fit_estimated_subject_scale(study, subjects, outcomes, features)
        return self._fit_fixed_subject_scale(study, subjects, outcomes, features)

    def _fit_fixed_subject_scale(
        self,
        study: Study,
        subjects: tuple[Any, ...],
        outcomes: NDArray[np.float64],
        features: NDArray[np.float64],
    ) -> HierarchicalGLMFitResult:
        n_coefficients = len(self.coefficient_names)
        subject_index = {subject: index for index, subject in enumerate(subjects)}
        row_subjects = np.asarray(
            [subject_index[_scalar(subject)] for subject in study["subject"]],
            dtype=np.intp,
        )
        joint_design = np.zeros(
            (len(study), n_coefficients * (1 + len(subjects))),
            dtype=np.float64,
        )
        joint_design[:, :n_coefficients] = features
        for row, subject_position in enumerate(row_subjects):
            start = n_coefficients * (1 + int(subject_position))
            joint_design[row, start : start + n_coefficients] = features[row]

        penalty = np.zeros(joint_design.shape[1], dtype=np.float64)
        penalty[1:n_coefficients] = self.l2
        penalty[n_coefficients:] = 1.0 / self.subject_scale**2
        joint_names = tuple(
            [f"population.{name}" for name in self.coefficient_names]
            + [
                f"subject[{position}].{name}"
                for position in range(len(subjects))
                for name in self.coefficient_names
            ]
        )
        joint_fit = fit_bernoulli(
            model_name=self.model_name,
            model_signature=self.signature,
            parameter_names=joint_names,
            design_matrix=joint_design,
            outcomes=outcomes,
            penalty_matrix=np.diag(penalty),
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
            coefficient_warning_threshold=self.coefficient_warning_threshold,
        )
        population = joint_fit.estimates[:n_coefficients]
        deviations = joint_fit.estimates[n_coefficients:].reshape(len(subjects), n_coefficients)
        subject_standard_errors = joint_fit.standard_errors[n_coefficients:].reshape(
            len(subjects), n_coefficients
        )
        effective_coefficients = population[None, :] + deviations
        diagnostics = replace(
            joint_fit.diagnostics,
            boundary_estimate=joint_fit.diagnostics.boundary_estimate
            or bool(np.any(np.abs(effective_coefficients) >= self.coefficient_warning_threshold)),
        )
        return HierarchicalGLMFitResult(
            model_name=self.model_name,
            model_signature=self.signature,
            parameter_names=self.parameter_names,
            estimates=population,
            standard_errors=joint_fit.standard_errors[:n_coefficients],
            covariance=joint_fit.covariance[:n_coefficients, :n_coefficients],
            n_observations=len(study),
            diagnostics=diagnostics,
            subjects=subjects,
            subject_deviations=deviations,
            subject_standard_errors=subject_standard_errors,
            subject_scale=self.subject_scale,
        )

    def _fit_estimated_subject_scale(
        self,
        study: Study,
        subjects: tuple[Any, ...],
        outcomes: NDArray[np.float64],
        features: NDArray[np.float64],
    ) -> HierarchicalGLMFitResult:
        n_coefficients = len(self.coefficient_names)
        groups = _subject_groups(study, subjects, features, outcomes)
        population_penalty = np.zeros(n_coefficients, dtype=np.float64)
        population_penalty[1:] = self.l2
        population_fit = fit_bernoulli(
            model_name=self.model_name,
            model_signature=self.signature,
            parameter_names=self.parameter_names,
            design_matrix=features,
            outcomes=outcomes,
            penalty_matrix=np.diag(population_penalty),
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
            coefficient_warning_threshold=self.coefficient_warning_threshold,
        )

        def profile_objective(parameters: NDArray[np.float64]) -> float:
            population = parameters[:n_coefficients]
            subject_scale = float(np.exp(parameters[-1]))
            return _laplace_profile(
                groups,
                population,
                subject_scale,
                l2=self.l2,
                max_iterations=self.max_iterations,
                tolerance=self.tolerance,
            ).objective

        initial = np.concatenate(
            [population_fit.estimates, np.asarray([np.log(self.subject_scale)])]
        )
        lower, upper = self.subject_scale_bounds
        outer_bounds = [(None, None)] * n_coefficients + [(np.log(lower), np.log(upper))]
        outer_fit = minimize(
            profile_objective,
            initial,
            method="L-BFGS-B",
            bounds=outer_bounds,
            options={
                "maxiter": self.max_iterations,
                "ftol": self.tolerance,
                "gtol": self.tolerance,
            },
        )
        optimizer = "L-BFGS-B with Laplace marginal likelihood"
        if not outer_fit.success:
            outer_fit = minimize(
                profile_objective,
                np.asarray(outer_fit.x, dtype=np.float64),
                method="Powell",
                bounds=outer_bounds,
                options={
                    "maxiter": self.max_iterations,
                    "ftol": self.tolerance,
                    "xtol": self.tolerance,
                },
            )
            optimizer = "Powell fallback with Laplace marginal likelihood"
        population = np.asarray(outer_fit.x[:n_coefficients], dtype=np.float64)
        log_subject_scale = float(outer_fit.x[-1])
        subject_scale = float(np.exp(log_subject_scale))
        profile = _laplace_profile(
            groups,
            population,
            subject_scale,
            l2=self.l2,
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
        )
        outer_point = np.asarray(outer_fit.x, dtype=np.float64)
        hessian = value_difference_hessian(
            profile_objective, outer_point, steps=relative_steps(outer_point, scale=1e-3)
        )
        gradient = _numerical_gradient(profile_objective, np.asarray(outer_fit.x, dtype=np.float64))
        covariance = np.linalg.pinv(hessian, hermitian=True)
        standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        scale_standard_error = subject_scale * standard_errors[-1]
        boundary_tolerance = 1e-5 * max(1.0, upper - lower)
        scale_at_boundary = bool(
            subject_scale - lower <= boundary_tolerance
            or upper - subject_scale <= boundary_tolerance
        )
        if (subject_scale - lower <= boundary_tolerance and gradient[-1] > 0) or (
            upper - subject_scale <= boundary_tolerance and gradient[-1] < 0
        ):
            gradient[-1] = 0.0
        effective_coefficients = population[None, :] + profile.deviations
        converged = bool(outer_fit.success and profile.all_subject_modes_converged)
        message = str(outer_fit.message)
        if not profile.all_subject_modes_converged:
            message = f"{message}; at least one conditional subject mode did not converge"
        diagnostics = FitDiagnostics(
            converged=converged,
            optimizer=optimizer,
            status=int(outer_fit.status),
            message=message,
            n_iterations=int(outer_fit.nit),
            objective=float(profile.objective),
            gradient_norm=float(np.linalg.norm(gradient)),
            hessian_condition=float(np.linalg.cond(hessian)),
            boundary_estimate=scale_at_boundary
            or bool(np.any(np.abs(population) >= self.coefficient_warning_threshold))
            or bool(np.any(np.abs(effective_coefficients) >= self.coefficient_warning_threshold)),
        )
        subject_standard_errors = np.sqrt(
            np.maximum(
                np.diagonal(profile.conditional_covariances, axis1=1, axis2=2),
                0.0,
            )
        )
        return HierarchicalGLMFitResult(
            model_name=self.model_name,
            model_signature=self.signature,
            parameter_names=self.parameter_names,
            estimates=population,
            standard_errors=standard_errors[:n_coefficients],
            covariance=covariance[:n_coefficients, :n_coefficients],
            n_observations=len(study),
            diagnostics=diagnostics,
            subjects=subjects,
            subject_deviations=profile.deviations,
            subject_standard_errors=subject_standard_errors,
            subject_scale=subject_scale,
            subject_scale_standard_error=float(scale_standard_error),
            subject_scale_bounds=self.subject_scale_bounds,
            subject_scale_estimated=True,
            subject_scale_at_boundary=scale_at_boundary,
        )

    def predict(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> Prediction:
        """Predict with fitted deviations or the population plug-in for unseen subjects."""

        prediction_mode = self._prediction_mode(mode)
        if not isinstance(fit, HierarchicalGLMFitResult):
            raise ValueError("fit result does not retain hierarchical subject effects")
        self._validate_fit(fit)
        outcomes = self._outcomes(study) if self.choice_lags else None
        features = self._base_feature_matrix(study, outcomes)
        subject_coefficients = {
            subject: fit.estimates + fit.subject_deviations[index]
            for index, subject in enumerate(fit.subjects)
        }
        coefficients = np.vstack(
            [
                subject_coefficients.get(_scalar(subject), fit.estimates)
                for subject in study["subject"]
            ]
        )
        linear_predictor = np.einsum("ij,ij->i", features, coefficients)
        return Prediction(
            probability=expit(linear_predictor),
            linear_predictor=linear_predictor,
            mode=prediction_mode,
        )

    def pointwise_log_prob(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> NDArray[np.float64]:
        """Score observed choices under fitted or unseen-subject plug-in effects."""

        outcomes = self._outcomes(study)
        prediction = self.predict(study, fit, mode=mode)
        scores = outcomes * -np.logaddexp(0.0, -prediction.linear_predictor)
        scores += (1.0 - outcomes) * -np.logaddexp(0.0, prediction.linear_predictor)
        return protected_array(scores, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class _LaplaceProfile:
    objective: float
    deviations: NDArray[np.float64]
    conditional_covariances: NDArray[np.float64]
    all_subject_modes_converged: bool


def _subject_groups(
    study: Study,
    subjects: tuple[Any, ...],
    features: NDArray[np.float64],
    outcomes: NDArray[np.float64],
) -> tuple[tuple[NDArray[np.float64], NDArray[np.float64]], ...]:
    row_subjects = np.asarray([_scalar(subject) for subject in study["subject"]], dtype=object)
    return tuple(
        (features[row_subjects == subject], outcomes[row_subjects == subject])
        for subject in subjects
    )


def _laplace_profile(
    groups: tuple[tuple[NDArray[np.float64], NDArray[np.float64]], ...],
    population: NDArray[np.float64],
    subject_scale: float,
    *,
    l2: float,
    max_iterations: int,
    tolerance: float,
) -> _LaplaceProfile:
    n_coefficients = len(population)
    precision = 1.0 / subject_scale**2
    deviations = np.zeros((len(groups), n_coefficients), dtype=np.float64)
    conditional_covariances = np.zeros(
        (len(groups), n_coefficients, n_coefficients), dtype=np.float64
    )
    objective = 0.5 * l2 * float(population[1:] @ population[1:])
    all_converged = True

    for subject_index, (subject_features, subject_outcomes) in enumerate(groups):

        def conditional_objective(
            deviation: NDArray[np.float64],
            feature_matrix: NDArray[np.float64] = subject_features,
            observed: NDArray[np.float64] = subject_outcomes,
        ) -> tuple[float, NDArray[np.float64]]:
            linear_predictor = feature_matrix @ (population + deviation)
            loss = np.logaddexp(0.0, linear_predictor).sum()
            loss -= observed @ linear_predictor
            loss += 0.5 * precision * float(deviation @ deviation)
            gradient = feature_matrix.T @ (expit(linear_predictor) - observed)
            gradient += precision * deviation
            return float(loss), np.asarray(gradient, dtype=np.float64)

        mode = minimize(
            conditional_objective,
            np.zeros(n_coefficients, dtype=np.float64),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": max_iterations, "ftol": tolerance, "gtol": tolerance},
        )
        deviation = np.asarray(mode.x, dtype=np.float64)
        linear_predictor = subject_features @ (population + deviation)
        probabilities = expit(linear_predictor)
        weights = probabilities * (1.0 - probabilities)
        conditional_hessian = subject_features.T @ (
            weights[:, None] * subject_features
        ) + precision * np.eye(n_coefficients)
        sign, log_determinant = np.linalg.slogdet(conditional_hessian)
        if sign <= 0:
            return _LaplaceProfile(
                objective=float("inf"),
                deviations=deviations,
                conditional_covariances=conditional_covariances,
                all_subject_modes_converged=False,
            )
        deviations[subject_index] = deviation
        conditional_covariances[subject_index] = np.linalg.pinv(conditional_hessian, hermitian=True)
        conditional_value, _ = conditional_objective(deviation)
        objective += conditional_value
        objective += n_coefficients * np.log(subject_scale) + 0.5 * log_determinant
        all_converged = all_converged and bool(mode.success)

    return _LaplaceProfile(
        objective=float(objective),
        deviations=deviations,
        conditional_covariances=conditional_covariances,
        all_subject_modes_converged=all_converged,
    )


def _numerical_gradient(
    objective: Callable[[NDArray[np.float64]], float],
    optimum: NDArray[np.float64],
) -> NDArray[np.float64]:
    steps = 1e-4 * np.maximum(1.0, np.abs(optimum))
    gradient = np.zeros(len(optimum), dtype=np.float64)
    for index, step in enumerate(steps):
        displacement = np.zeros(len(optimum), dtype=np.float64)
        displacement[index] = step
        gradient[index] = (
            objective(optimum + displacement) - objective(optimum - displacement)
        ) / (2.0 * step)
    return gradient


def _validate_scale_bounds(
    bounds: tuple[float, float] | None,
) -> tuple[float, float]:
    if bounds is None or not isinstance(bounds, tuple) or len(bounds) != 2:
        raise ValueError("subject_scale_bounds must contain a lower and upper value")
    try:
        lower, upper = (float(value) for value in bounds)
    except (TypeError, ValueError):
        raise ValueError("subject_scale_bounds must contain finite numbers") from None
    if not np.isfinite(lower) or not np.isfinite(upper) or lower <= 0 or upper <= lower:
        raise ValueError("subject_scale_bounds must be finite, positive, and increasing")
    return lower, upper


def _scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value
