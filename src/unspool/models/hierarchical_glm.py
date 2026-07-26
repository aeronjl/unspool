"""A bounded partial-pooling Bernoulli history GLM."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit

from unspool.models.base import (
    FitResult,
    ModelDataError,
    Prediction,
    PredictionMode,
    _protected_array,
)
from unspool.models.glm import BernoulliHistoryGLM, _fit_bernoulli, _ordered_session_indices
from unspool.study import Study


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
        population = _protected_array(self.population_coefficients, dtype=np.float64)
        deviations = _protected_array(self.subject_deviations, dtype=np.float64)
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

        return _protected_array(
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
    unseen_subject_policy: str = "population-mean-plugin"

    def __post_init__(self) -> None:
        FitResult.__post_init__(self)
        subjects = tuple(_scalar(subject) for subject in self.subjects)
        deviations = _protected_array(self.subject_deviations, dtype=np.float64)
        standard_errors = _protected_array(self.subject_standard_errors, dtype=np.float64)
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
        if self.unseen_subject_policy != "population-mean-plugin":
            raise ValueError("unseen_subject_policy must be 'population-mean-plugin'")
        object.__setattr__(self, "subjects", subjects)
        object.__setattr__(self, "subject_deviations", deviations)
        object.__setattr__(self, "subject_standard_errors", standard_errors)

    @property
    def subject_coefficients(self) -> NDArray[np.float64]:
        """Return population-plus-deviation coefficient estimates by fitted subject."""

        return _protected_array(self.estimates[None, :] + self.subject_deviations, dtype=np.float64)

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


@dataclass(frozen=True, slots=True)
class HierarchicalBernoulliHistoryGLM(BernoulliHistoryGLM):
    """Static Bernoulli GLM with fixed-scale Gaussian subject deviations.

    The joint maximum-a-posteriori fit estimates one population coefficient vector and one
    shrunken deviation vector per training subject. ``subject_scale`` is a fixed prior
    standard deviation selected before fitting; it is not estimated from the same data.
    Predictions for unseen subjects use the population coefficient plug-in explicitly.
    """

    subject_scale: float = 0.5

    def __post_init__(self) -> None:
        BernoulliHistoryGLM.__post_init__(self)
        if not np.isfinite(self.subject_scale) or self.subject_scale <= 0:
            raise ValueError("subject_scale must be finite and positive")

    @property
    def model_name(self) -> str:
        return "hierarchical-bernoulli-history-glm"

    @property
    def signature(self) -> str:
        covariates = ",".join(self.covariates)
        return (
            f"{self.model_name}[outcome={self.outcome};covariates={covariates};"
            f"choice_lags={self.choice_lags};l2={self.l2};subject_scale={self.subject_scale}]"
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

        for session_indices in _ordered_session_indices(design):
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
        joint_fit = _fit_bernoulli(
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
        return _protected_array(scores, dtype=np.float64)


def _scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value
