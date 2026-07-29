"""Partially pooled smooth coefficient trajectories for longitudinal behaviour."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import pairwise
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit

from behavio._internal.arrays import protected_array
from behavio.models._kernels.basis import (
    format_time,
    linear_time_basis,
    roughness_matrix,
    validated_knots,
)
from behavio.models._kernels.bernoulli import fit_bernoulli, ordered_session_indices
from behavio.models.base import (
    FitResult,
    ModelDataError,
    Prediction,
    PredictionMode,
)
from behavio.models.glm import BernoulliHistoryGLM, CoefficientTrajectory
from behavio.study import Study


@dataclass(frozen=True, slots=True)
class HierarchicalSmoothGLMSimulation:
    """Observed choices paired with unexposed population and subject paths."""

    study: Study
    subjects: tuple[Any, ...]
    clock: str
    knots: tuple[float, ...]
    coefficient_names: tuple[str, ...]
    population_knot_values: NDArray[np.float64]
    subject_deviation_knot_values: NDArray[np.float64]

    def __post_init__(self) -> None:
        subjects = tuple(_scalar(subject) for subject in self.subjects)
        knots = tuple(float(knot) for knot in self.knots)
        names = tuple(self.coefficient_names)
        population = protected_array(self.population_knot_values, dtype=np.float64)
        deviations = protected_array(self.subject_deviation_knot_values, dtype=np.float64)
        if subjects != tuple(_scalar(subject) for subject in self.study.subjects):
            raise ValueError("simulation subjects must match the study's subject order")
        _validate_path_coordinates(self.clock, knots, names)
        expected_population = (len(names), len(knots))
        expected_deviations = (len(subjects), len(names), len(knots))
        if population.shape != expected_population or deviations.shape != expected_deviations:
            raise ValueError("simulation paths must align with subjects, coefficients, and knots")
        if not np.all(np.isfinite(population)) or not np.all(np.isfinite(deviations)):
            raise ValueError("simulation paths must be finite")
        object.__setattr__(self, "subjects", subjects)
        object.__setattr__(self, "knots", knots)
        object.__setattr__(self, "coefficient_names", names)
        object.__setattr__(self, "population_knot_values", population)
        object.__setattr__(self, "subject_deviation_knot_values", deviations)

    @property
    def subject_knot_values(self) -> NDArray[np.float64]:
        """Return realized population-plus-deviation paths by subject."""

        return protected_array(
            self.population_knot_values[None, :, :] + self.subject_deviation_knot_values,
            dtype=np.float64,
        )


@dataclass(frozen=True, slots=True)
class HierarchicalSmoothGLMFitResult(FitResult):
    """Population paths and shrunken smooth deviations from one joint MAP fit."""

    subjects: tuple[Any, ...]
    clock: str
    knots: tuple[float, ...]
    coefficient_names: tuple[str, ...]
    subject_deviations: NDArray[np.float64]
    subject_standard_errors: NDArray[np.float64]
    subject_scale: float
    subject_smoothness: float
    unseen_subject_policy: str = "population-trajectory-plugin"

    def __post_init__(self) -> None:
        FitResult.__post_init__(self)
        subjects = tuple(_scalar(subject) for subject in self.subjects)
        knots = tuple(float(knot) for knot in self.knots)
        names = tuple(self.coefficient_names)
        deviations = protected_array(self.subject_deviations, dtype=np.float64)
        standard_errors = protected_array(self.subject_standard_errors, dtype=np.float64)
        expected = (len(subjects), len(names), len(knots))
        if not subjects or len(set(subjects)) != len(subjects):
            raise ValueError("fit subjects must be non-empty and unique")
        _validate_path_coordinates(self.clock, knots, names)
        if deviations.shape != expected or standard_errors.shape != expected:
            raise ValueError("subject path estimates must align with subjects and knots")
        if not np.all(np.isfinite(deviations)) or not np.all(np.isfinite(standard_errors)):
            raise ValueError("subject path estimates and standard errors must be finite")
        if np.any(standard_errors < 0):
            raise ValueError("subject path standard errors must be non-negative")
        if not np.isfinite(self.subject_scale) or self.subject_scale <= 0:
            raise ValueError("subject_scale must be finite and positive")
        if not np.isfinite(self.subject_smoothness) or self.subject_smoothness <= 0:
            raise ValueError("subject_smoothness must be finite and positive")
        if self.unseen_subject_policy != "population-trajectory-plugin":
            raise ValueError("unseen_subject_policy must be 'population-trajectory-plugin'")
        if self.estimates.shape != (len(names) * len(knots),):
            raise ValueError("population estimates must contain every coefficient knot")
        object.__setattr__(self, "subjects", subjects)
        object.__setattr__(self, "knots", knots)
        object.__setattr__(self, "coefficient_names", names)
        object.__setattr__(self, "subject_deviations", deviations)
        object.__setattr__(self, "subject_standard_errors", standard_errors)

    @property
    def population_knot_values(self) -> NDArray[np.float64]:
        """Return fitted population paths in coefficient-by-knot order."""

        return protected_array(
            self.estimates.reshape(len(self.coefficient_names), len(self.knots)),
            dtype=np.float64,
        )

    @property
    def subject_knot_values(self) -> NDArray[np.float64]:
        """Return population-plus-deviation paths for fitted subjects."""

        return protected_array(
            self.population_knot_values[None, :, :] + self.subject_deviations,
            dtype=np.float64,
        )

    def subject_was_fitted(self, subject: Any) -> bool:
        """Report whether a subject has an estimated deviation path."""

        return _scalar(subject) in self.subjects


@dataclass(frozen=True, slots=True)
class HierarchicalSmoothBernoulliHistoryGLM(BernoulliHistoryGLM):
    """Smooth population paths with shrunken smooth subject-deviation paths."""

    knots: tuple[float, ...] = (0.0, 1.0)
    time: str = "session_order"
    smoothness: float = 1.0
    subject_scale: float = 0.5
    subject_smoothness: float = 1.0

    def __post_init__(self) -> None:
        BernoulliHistoryGLM.__post_init__(self)
        knots = validated_knots(self.knots)
        if not isinstance(self.time, str) or not self.time:
            raise ValueError("time must be a non-empty Study column name")
        if self.time == self.outcome:
            raise ValueError("time and outcome columns must be distinct")
        if not np.isfinite(self.smoothness) or self.smoothness <= 0:
            raise ValueError("smoothness must be finite and positive")
        if not np.isfinite(self.subject_scale) or self.subject_scale <= 0:
            raise ValueError("subject_scale must be finite and positive")
        if not np.isfinite(self.subject_smoothness) or self.subject_smoothness <= 0:
            raise ValueError("subject_smoothness must be finite and positive")
        object.__setattr__(self, "knots", knots)

    @property
    def model_name(self) -> str:
        return "hierarchical-smooth-bernoulli-history-glm"

    @property
    def signature(self) -> str:
        covariates = ",".join(self.covariates)
        knots = ",".join(format_time(knot) for knot in self.knots)
        return (
            f"{self.model_name}[outcome={self.outcome};covariates={covariates};"
            f"choice_lags={self.choice_lags};time={self.time};knots={knots};"
            f"smoothness={self.smoothness};l2={self.l2};subject_scale={self.subject_scale};"
            f"subject_smoothness={self.subject_smoothness}{self._design_signature}]"
        )

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(
            f"{coefficient}[{self.time}={format_time(knot)}]"
            for coefficient in self.coefficient_names
            for knot in self.knots
        )

    def parameters_from_paths(self, paths: Mapping[str, Sequence[float]]) -> Mapping[str, float]:
        """Pack population coefficient paths into simulation coordinates."""

        expected = set(self.coefficient_names)
        observed = set(paths)
        if observed != expected:
            raise ValueError(
                "paths must match the model coefficients exactly; "
                f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
            )
        values: dict[str, float] = {}
        offset = 0
        for coefficient in self.coefficient_names:
            path = np.asarray(paths[coefficient], dtype=np.float64)
            if path.shape != (len(self.knots),) or not np.all(np.isfinite(path)):
                raise ValueError(
                    f"path {coefficient!r} must contain one finite value per temporal knot"
                )
            for value in path:
                values[self.parameter_names[offset]] = float(value)
                offset += 1
        return MappingProxyType(values)

    def simulate(
        self,
        design: Study,
        parameters: Mapping[str, float],
        *,
        seed: int | np.random.Generator,
    ) -> Study:
        """Generate choices without exposing realized subject trajectories."""

        return self.simulate_with_effects(design, parameters, seed=seed).study

    def simulate_with_effects(
        self,
        design: Study,
        parameters: Mapping[str, float],
        *,
        seed: int | np.random.Generator,
        subject_deviation_paths: Mapping[Any, Mapping[str, Sequence[float]]] | None = None,
    ) -> HierarchicalSmoothGLMSimulation:
        """Generate choices and retain realized subject paths outside observed data."""

        population = self._parameter_vector(parameters).reshape(
            len(self.coefficient_names), len(self.knots)
        )
        basis = self._time_basis(design)
        generator = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        subjects = tuple(_scalar(subject) for subject in design.subjects)
        deviations = (
            self._draw_subject_deviations(subjects, generator)
            if subject_deviation_paths is None
            else self._subject_deviation_array(subjects, subject_deviation_paths)
        )
        subject_index = {subject: index for index, subject in enumerate(subjects)}
        row_subjects = np.asarray(
            [subject_index[_scalar(subject)] for subject in design["subject"]], dtype=np.intp
        )
        population_coefficients = basis @ population.T
        deviation_coefficients = np.einsum("ik,ijk->ij", basis, deviations[row_subjects])
        coefficients = population_coefficients + deviation_coefficients
        covariates = self._covariate_matrix(design)
        choices = np.zeros(len(design), dtype=np.int8)
        covariate_end = 1 + len(self.covariates)

        for session_indices in ordered_session_indices(design):
            generated_history: list[float] = []
            for index in session_indices:
                features = np.empty(len(self.coefficient_names), dtype=np.float64)
                features[0] = 1.0
                features[1:covariate_end] = covariates[index]
                for lag in range(1, self.choice_lags + 1):
                    features[covariate_end + lag - 1] = (
                        generated_history[-lag] if len(generated_history) >= lag else 0.0
                    )
                probability = expit(float(features @ coefficients[index]))
                choice = int(generator.binomial(1, probability))
                choices[index] = choice
                generated_history.append(2.0 * choice - 1.0)

        columns = {name: design[name] for name in design.columns}
        columns[self.outcome] = choices
        return HierarchicalSmoothGLMSimulation(
            study=Study(columns),
            subjects=subjects,
            clock=self.time,
            knots=self.knots,
            coefficient_names=self.coefficient_names,
            population_knot_values=population,
            subject_deviation_knot_values=deviations,
        )

    def fit(self, study: Study) -> HierarchicalSmoothGLMFitResult:
        """Jointly fit population and smooth subject-deviation paths."""

        subjects = tuple(_scalar(subject) for subject in study.subjects)
        if len(subjects) < 2:
            raise ModelDataError("hierarchical smooth fitting requires at least two subjects")
        outcomes = self._outcomes(study)
        dynamic_design = self._dynamic_design_matrix(study, outcomes)
        n_path_parameters = len(self.parameter_names)
        subject_index = {subject: index for index, subject in enumerate(subjects)}
        row_subjects = np.asarray(
            [subject_index[_scalar(subject)] for subject in study["subject"]], dtype=np.intp
        )
        joint_design = np.zeros(
            (len(study), n_path_parameters * (1 + len(subjects))), dtype=np.float64
        )
        joint_design[:, :n_path_parameters] = dynamic_design
        for row, subject_position in enumerate(row_subjects):
            start = n_path_parameters * (1 + int(subject_position))
            joint_design[row, start : start + n_path_parameters] = dynamic_design[row]

        population_penalty = self._population_penalty()
        subject_penalty = self._subject_penalty()
        joint_penalty = np.zeros((joint_design.shape[1], joint_design.shape[1]), dtype=np.float64)
        joint_penalty[:n_path_parameters, :n_path_parameters] = population_penalty
        for position in range(len(subjects)):
            start = n_path_parameters * (1 + position)
            joint_penalty[
                start : start + n_path_parameters,
                start : start + n_path_parameters,
            ] = subject_penalty
        joint_names = tuple(
            [f"population.{name}" for name in self.parameter_names]
            + [
                f"subject[{position}].{name}"
                for position in range(len(subjects))
                for name in self.parameter_names
            ]
        )
        joint_fit = fit_bernoulli(
            model_name=self.model_name,
            model_signature=self.signature,
            parameter_names=joint_names,
            design_matrix=joint_design,
            outcomes=outcomes,
            penalty_matrix=joint_penalty,
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
            coefficient_warning_threshold=self.coefficient_warning_threshold,
        )
        population = joint_fit.estimates[:n_path_parameters]
        deviations = joint_fit.estimates[n_path_parameters:].reshape(
            len(subjects), len(self.coefficient_names), len(self.knots)
        )
        subject_standard_errors = joint_fit.standard_errors[n_path_parameters:].reshape(
            len(subjects), len(self.coefficient_names), len(self.knots)
        )
        effective_paths = (
            population.reshape(len(self.coefficient_names), len(self.knots))[None, :, :]
            + deviations
        )
        diagnostics = replace(
            joint_fit.diagnostics,
            boundary_estimate=joint_fit.diagnostics.boundary_estimate
            or bool(np.any(np.abs(effective_paths) >= self.coefficient_warning_threshold)),
        )
        return HierarchicalSmoothGLMFitResult(
            model_name=self.model_name,
            model_signature=self.signature,
            parameter_names=self.parameter_names,
            estimates=population,
            standard_errors=joint_fit.standard_errors[:n_path_parameters],
            covariance=joint_fit.covariance[:n_path_parameters, :n_path_parameters],
            n_observations=len(study),
            diagnostics=diagnostics,
            subjects=subjects,
            clock=self.time,
            knots=self.knots,
            coefficient_names=self.coefficient_names,
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

        prediction_mode = self._prediction_mode(mode)
        hierarchical_fit = self._validated_hierarchical_fit(fit)
        outcomes = self._outcomes(study) if self.choice_lags else None
        dynamic_design = self._dynamic_design_matrix(study, outcomes)
        subject_paths = {
            subject: hierarchical_fit.estimates
            + hierarchical_fit.subject_deviations[index].reshape(-1)
            for index, subject in enumerate(hierarchical_fit.subjects)
        }
        coefficients = np.vstack(
            [
                subject_paths.get(_scalar(subject), hierarchical_fit.estimates)
                for subject in study["subject"]
            ]
        )
        linear_predictor = np.einsum("ij,ij->i", dynamic_design, coefficients)
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
        """Score observed choices under fitted or population plug-in trajectories."""

        outcomes = self._outcomes(study)
        prediction = self.predict(study, fit, mode=mode)
        scores = outcomes * -np.logaddexp(0.0, -prediction.linear_predictor)
        scores += (1.0 - outcomes) * -np.logaddexp(0.0, prediction.linear_predictor)
        return protected_array(scores, dtype=np.float64)

    def population_trajectory(
        self,
        fit: FitResult,
        *,
        times: Sequence[float] | None = None,
    ) -> CoefficientTrajectory:
        """Evaluate the fitted population path at requested clock values."""

        hierarchical_fit = self._validated_hierarchical_fit(fit)
        evaluation_times = self.knots if times is None else times
        time_array = np.asarray(evaluation_times, dtype=np.float64)
        basis = linear_time_basis(time_array, self.knots)
        values = basis @ hierarchical_fit.population_knot_values.T
        return CoefficientTrajectory(
            clock=self.time,
            times=time_array,
            coefficient_names=self.coefficient_names,
            values=values,
        )

    def subject_trajectory(
        self,
        fit: FitResult,
        subject: Any,
        *,
        times: Sequence[float] | None = None,
    ) -> CoefficientTrajectory:
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
        basis = linear_time_basis(time_array, self.knots)
        return CoefficientTrajectory(
            clock=self.time,
            times=time_array,
            coefficient_names=self.coefficient_names,
            values=basis @ knot_values.T,
        )

    def _dynamic_design_matrix(
        self, study: Study, outcomes: NDArray[np.float64] | None
    ) -> NDArray[np.float64]:
        features = self._base_feature_matrix(study, outcomes)
        basis = self._time_basis(study)
        return np.einsum("ij,ik->ijk", features, basis).reshape(len(study), -1)

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
            return linear_time_basis(times, self.knots)
        except ValueError as error:
            raise ModelDataError(str(error)) from None

    def _population_penalty(self) -> NDArray[np.float64]:
        roughness = self.smoothness * roughness_matrix(self.knots)
        penalty = np.kron(np.eye(len(self.coefficient_names)), roughness)
        if self.l2:
            n_knots = len(self.knots)
            for coefficient in range(1, len(self.coefficient_names)):
                start = coefficient * n_knots
                penalty[start : start + n_knots, start : start + n_knots] += self.l2 * np.eye(
                    n_knots
                )
        return penalty

    def _subject_penalty(self) -> NDArray[np.float64]:
        knot_penalty = np.eye(len(self.knots)) / self.subject_scale**2
        knot_penalty += self.subject_smoothness * roughness_matrix(self.knots)
        return np.kron(np.eye(len(self.coefficient_names)), knot_penalty)

    def _draw_subject_deviations(
        self,
        subjects: tuple[Any, ...],
        generator: np.random.Generator,
    ) -> NDArray[np.float64]:
        covariance = np.linalg.pinv(
            np.eye(len(self.knots)) / self.subject_scale**2
            + self.subject_smoothness * roughness_matrix(self.knots),
            hermitian=True,
        )
        deviations = np.empty(
            (len(subjects), len(self.coefficient_names), len(self.knots)), dtype=np.float64
        )
        for subject in range(len(subjects)):
            for coefficient in range(len(self.coefficient_names)):
                deviations[subject, coefficient] = generator.multivariate_normal(
                    np.zeros(len(self.knots)), covariance
                )
        return deviations

    def _subject_deviation_array(
        self,
        subjects: tuple[Any, ...],
        paths: Mapping[Any, Mapping[str, Sequence[float]]],
    ) -> NDArray[np.float64]:
        normalized = {_scalar(subject): values for subject, values in paths.items()}
        if set(normalized) != set(subjects):
            raise ValueError("subject_deviation_paths must contain every design subject exactly")
        deviations = np.empty(
            (len(subjects), len(self.coefficient_names), len(self.knots)), dtype=np.float64
        )
        for subject_position, subject in enumerate(subjects):
            subject_paths = normalized[subject]
            if set(subject_paths) != set(self.coefficient_names):
                raise ValueError("each subject deviation must contain every coefficient exactly")
            for coefficient_position, coefficient in enumerate(self.coefficient_names):
                values = np.asarray(subject_paths[coefficient], dtype=np.float64)
                if values.shape != (len(self.knots),) or not np.all(np.isfinite(values)):
                    raise ValueError("subject deviation paths require one finite value per knot")
                deviations[subject_position, coefficient_position] = values
        return deviations

    def _validated_hierarchical_fit(self, fit: FitResult) -> HierarchicalSmoothGLMFitResult:
        if not isinstance(fit, HierarchicalSmoothGLMFitResult):
            raise ValueError("fit result does not retain hierarchical smooth trajectories")
        self._validate_fit(fit)
        return fit


def _validate_path_coordinates(
    clock: str,
    knots: tuple[float, ...],
    coefficient_names: tuple[str, ...],
) -> None:
    if not isinstance(clock, str) or not clock:
        raise ValueError("trajectory clock must be a non-empty string")
    if (
        len(knots) < 2
        or not np.all(np.isfinite(knots))
        or any(right <= left for left, right in pairwise(knots))
    ):
        raise ValueError("trajectory knots must be finite and strictly increasing")
    if (
        not coefficient_names
        or len(set(coefficient_names)) != len(coefficient_names)
        or any(not isinstance(name, str) or not name for name in coefficient_names)
    ):
        raise ValueError("coefficient names must be non-empty and unique")


def _scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value
