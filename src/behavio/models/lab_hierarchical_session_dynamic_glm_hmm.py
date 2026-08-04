"""Population-to-laboratory-to-subject session-dynamic Bernoulli GLM-HMMs.

This module adds laboratory as an exchangeable sampling level rather than treating it as
another fixed label.  Population emission weights evolve over observed session order,
laboratory deviations evolve around that population path, and subject deviations evolve
around their laboratory path.  Subjects must be nested in exactly one laboratory and model
fitting requires independent subject replication within every laboratory.

The transition model deliberately remains the direct population-shrunk session model used
by :mod:`behavio.models.hierarchical_session_dynamic_glm_hmm`.  Persistent laboratory or
subject transition styles would be a different model and are not implied here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import pairwise
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

from behavio._internal.arrays import protected_array
from behavio.models._kernels.bernoulli import ordered_session_indices
from behavio.models._kernels.dynamic_uncertainty import (
    at_log_bound,
    estimate_transition_concentration,
    gaussian_scale_observed_covariance,
    observed_path_covariance,
    session_transition_counts,
    supplemented_scale_covariance,
    transition_standard_errors,
    update_gaussian_scales,
)
from behavio.models.base import (
    FitDiagnostics,
    FitResult,
    ModelDataError,
    Prediction,
    PredictionMode,
)
from behavio.models.glm_hmm import (
    BernoulliGLMHMM,
    FilteredStateProbabilities,
    GLMHMMParameters,
    _minimum_pairwise_distance,
)
from behavio.models.hierarchical_session_dynamic_glm_hmm import (
    HierarchicalSessionDynamicBernoulliGLMHMM,
    _normal_path_intervals,
    _normalized_positive,
    _scalar,
    _subject_blocks,
)
from behavio.models.state_alignment import LatentStateAlignment, align_latent_states
from behavio.trials import Study


@dataclass(frozen=True, slots=True)
class _LabStructure:
    labs: tuple[Any, ...]
    subjects: tuple[Any, ...]
    subject_labs: tuple[Any, ...]
    path_subjects: tuple[Any, ...]
    path_labs: tuple[Any, ...]
    keys: tuple[Any, ...]
    orders: NDArray[np.int64]
    sessions: tuple[tuple[int, ...], ...]
    population_orders: NDArray[np.int64]
    population_index: NDArray[np.intp]
    subject_blocks: tuple[tuple[int, ...], ...]
    lab_path_labs: tuple[Any, ...]
    lab_path_orders: NDArray[np.int64]
    lab_population_index: NDArray[np.intp]
    session_lab_index: NDArray[np.intp]
    lab_blocks: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class LabHierarchicalSessionDynamicGLMHMMSimulation:
    """Observed choices with retained population, lab, subject, and state truth."""

    study: Study
    states: NDArray[np.int64]
    n_states: int
    lab_column: str
    labs: tuple[Any, ...]
    subjects: tuple[Any, ...]
    subject_labs: tuple[Any, ...]
    path_subjects: tuple[Any, ...]
    path_labs: tuple[Any, ...]
    session_keys: tuple[Any, ...]
    session_orders: NDArray[np.int64]
    population_session_orders: NDArray[np.int64]
    lab_path_labs: tuple[Any, ...]
    lab_path_orders: NDArray[np.int64]
    population_emission_coefficients: NDArray[np.float64]
    lab_deviation_coefficients: NDArray[np.float64]
    session_emission_coefficients: NDArray[np.float64]
    session_transition_matrices: NDArray[np.float64]
    global_transition_matrix: NDArray[np.float64]

    def __post_init__(self) -> None:
        states = protected_array(self.states, dtype=np.int64)
        if states.shape != (len(self.study),) or np.any((states < 0) | (states >= self.n_states)):
            raise ValueError("states must contain one valid label per trial")
        if self.lab_column not in self.study.columns:
            raise ValueError("simulation study must retain its laboratory column")
        _validate_lab_paths(
            self,
            n_states=self.n_states,
            require_fit_evidence=False,
        )
        object.__setattr__(self, "states", states)

    @property
    def lab_emission_coefficients(self) -> NDArray[np.float64]:
        """Return the realized population-plus-laboratory path."""

        positions = {
            int(order): index for index, order in enumerate(self.population_session_orders)
        }
        values = np.stack(
            [
                self.population_emission_coefficients[positions[int(order)]] + deviation
                for order, deviation in zip(
                    self.lab_path_orders,
                    self.lab_deviation_coefficients,
                    strict=True,
                )
            ]
        )
        return protected_array(values, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class LabHierarchicalSessionDynamicTrajectoryRecovery:
    """Truth-aligned recovery at every nested emission level."""

    alignment: LatentStateAlignment
    population_emission_rmse: float
    lab_deviation_rmse: float
    subject_emission_rmse: float
    transition_rmse: float
    lab_deviation_rmse_by_lab: NDArray[np.float64]
    subject_emission_rmse_by_subject: NDArray[np.float64]

    def __post_init__(self) -> None:
        lab_values = protected_array(self.lab_deviation_rmse_by_lab, dtype=np.float64)
        subject_values = protected_array(self.subject_emission_rmse_by_subject, dtype=np.float64)
        scalars = (
            self.population_emission_rmse,
            self.lab_deviation_rmse,
            self.subject_emission_rmse,
            self.transition_rmse,
        )
        if lab_values.ndim != 1 or not len(lab_values):
            raise ValueError("recovery needs one laboratory RMSE per laboratory")
        if subject_values.ndim != 1 or not len(subject_values):
            raise ValueError("recovery needs one subject RMSE per subject")
        if not all(np.isfinite(value) and value >= 0 for value in scalars):
            raise ValueError("trajectory recovery RMSE values must be finite and non-negative")
        if any(
            not np.all(np.isfinite(values)) or np.any(values < 0)
            for values in (lab_values, subject_values)
        ):
            raise ValueError("group-specific recovery RMSE values must be finite and non-negative")
        object.__setattr__(self, "lab_deviation_rmse_by_lab", lab_values)
        object.__setattr__(self, "subject_emission_rmse_by_subject", subject_values)


@dataclass(frozen=True, slots=True)
class UnseenSubjectInLabDynamicPrediction:
    """Integrated prediction for new subjects nested in fitted laboratories."""

    labs: tuple[Any, ...]
    subjects: tuple[Any, ...]
    subject_labs: tuple[Any, ...]
    row_subject_indices: NDArray[np.intp]
    probability: NDArray[np.float64]
    draw_probabilities: NDArray[np.float64]
    draw_pointwise_log_probability: NDArray[np.float64]
    pointwise_marginal_log_probability: NDArray[np.float64]
    subject_joint_log_probability: NDArray[np.float64]
    subject_effective_draws: NDArray[np.float64]
    subject_log_probability_mcse: NDArray[np.float64]
    draw_session_emission_coefficients: NDArray[np.float64]
    draw_session_transition_matrices: NDArray[np.float64]
    n_draws: int
    seed: int
    includes_fitted_path_uncertainty: bool
    label_path_ambiguous: bool
    label_policy: str = "conditional-on-one-whole-path-canonical-mode"

    def __post_init__(self) -> None:
        _validate_integrated_prediction(
            self,
            groups=self.subjects,
            row_group_indices=self.row_subject_indices,
            joint=self.subject_joint_log_probability,
            effective=self.subject_effective_draws,
            mcse=self.subject_log_probability_mcse,
        )
        labs = tuple(_scalar(value) for value in self.labs)
        subjects = tuple(_scalar(value) for value in self.subjects)
        subject_labs = tuple(_scalar(value) for value in self.subject_labs)
        if len(subject_labs) != len(subjects) or not set(subject_labs).issubset(set(labs)):
            raise ValueError("each prediction subject must name one prediction laboratory")
        object.__setattr__(self, "labs", labs)
        object.__setattr__(self, "subjects", subjects)
        object.__setattr__(self, "subject_labs", subject_labs)


@dataclass(frozen=True, slots=True)
class UnseenLabDynamicPrediction:
    """Integrated prediction whose coherent likelihood unit is an unseen laboratory."""

    labs: tuple[Any, ...]
    subjects: tuple[Any, ...]
    subject_labs: tuple[Any, ...]
    row_lab_indices: NDArray[np.intp]
    probability: NDArray[np.float64]
    draw_probabilities: NDArray[np.float64]
    draw_pointwise_log_probability: NDArray[np.float64]
    pointwise_marginal_log_probability: NDArray[np.float64]
    lab_joint_log_probability: NDArray[np.float64]
    lab_effective_draws: NDArray[np.float64]
    lab_log_probability_mcse: NDArray[np.float64]
    draw_lab_deviation_coefficients: NDArray[np.float64]
    draw_session_emission_coefficients: NDArray[np.float64]
    draw_session_transition_matrices: NDArray[np.float64]
    n_draws: int
    seed: int
    includes_population_path_uncertainty: bool
    label_path_ambiguous: bool
    label_policy: str = "conditional-on-one-whole-path-canonical-mode"

    def __post_init__(self) -> None:
        _validate_integrated_prediction(
            self,
            groups=self.labs,
            row_group_indices=self.row_lab_indices,
            joint=self.lab_joint_log_probability,
            effective=self.lab_effective_draws,
            mcse=self.lab_log_probability_mcse,
        )
        labs = tuple(_scalar(value) for value in self.labs)
        subjects = tuple(_scalar(value) for value in self.subjects)
        subject_labs = tuple(_scalar(value) for value in self.subject_labs)
        lab_draws = protected_array(self.draw_lab_deviation_coefficients, dtype=np.float64)
        if len(subject_labs) != len(subjects) or not set(subject_labs).issubset(set(labs)):
            raise ValueError("each prediction subject must name one prediction laboratory")
        if lab_draws.ndim != 4 or lab_draws.shape[0] != self.n_draws:
            raise ValueError("laboratory draws must cover draw, lab-order, state, coefficient")
        if not np.all(np.isfinite(lab_draws)):
            raise ValueError("laboratory draws must be finite")
        object.__setattr__(self, "labs", labs)
        object.__setattr__(self, "subjects", subjects)
        object.__setattr__(self, "subject_labs", subject_labs)
        object.__setattr__(self, "draw_lab_deviation_coefficients", lab_draws)


@dataclass(frozen=True, slots=True)
class LabHierarchicalSessionDynamicGLMHMMFitResult(FitResult):
    """Nested population, laboratory, and subject paths with numerical evidence."""

    lab_column: str
    labs: tuple[Any, ...]
    subjects: tuple[Any, ...]
    subject_labs: tuple[Any, ...]
    path_subjects: tuple[Any, ...]
    path_labs: tuple[Any, ...]
    session_keys: tuple[Any, ...]
    session_orders: NDArray[np.int64]
    population_session_orders: NDArray[np.int64]
    lab_path_labs: tuple[Any, ...]
    lab_path_orders: NDArray[np.int64]
    population_emission_coefficients: NDArray[np.float64]
    lab_deviation_coefficients: NDArray[np.float64]
    session_emission_coefficients: NDArray[np.float64]
    session_transition_matrices: NDArray[np.float64]
    global_transition_matrix: NDArray[np.float64]
    partial_objective_history: NDArray[np.float64]
    partial_converged: bool
    partial_emission_optimizer_converged: bool
    partial_emission_optimizer_message: str
    full_converged: bool
    objective_history: NDArray[np.float64]
    canonical_permutation: tuple[int, ...]
    state_occupancy: NDArray[np.float64]
    minimum_state_separation: float
    minimum_population_label_path_gap: float
    minimum_lab_label_path_gap: float
    minimum_subject_label_path_gap: float
    population_label_crossings: NDArray[np.bool_]
    lab_label_crossings: NDArray[np.bool_]
    lab_label_crossing_labs: tuple[Any, ...]
    subject_label_crossings: NDArray[np.bool_]
    subject_label_crossing_subjects: tuple[Any, ...]
    label_path_ambiguous: bool
    low_occupancy: bool
    emission_optimizer_converged: bool
    emission_optimizer_message: str
    initialization_restart_objectives: NDArray[np.float64]
    initialization_restart_converged: NDArray[np.bool_]
    initialization_restart_messages: tuple[str, ...]
    initialization_selected_restart: int
    population_emission_step_scale: float
    lab_emission_scale: float
    lab_emission_step_scale: float
    subject_emission_scale: float
    emission_step_scale: float
    transition_concentration: float
    population_emission_standard_errors: NDArray[np.float64]
    population_emission_covariance: NDArray[np.float64]
    lab_deviation_standard_errors: NDArray[np.float64]
    lab_emission_standard_errors: NDArray[np.float64]
    session_emission_standard_errors: NDArray[np.float64]
    session_emission_covariance: NDArray[np.float64]
    subject_deviation_standard_errors: NDArray[np.float64]
    joint_emission_covariance: NDArray[np.float64]
    session_transition_standard_errors: NDArray[np.float64]
    path_hessian_condition: float
    path_covariance_positive_definite: bool
    hyperparameter_names: tuple[str, ...]
    hyperparameter_estimates: NDArray[np.float64]
    hyperparameter_standard_errors: NDArray[np.float64]
    hyperparameter_covariance: NDArray[np.float64]
    hyperparameters_estimated: bool
    hyperparameter_estimation_converged: bool
    hyperparameter_estimation_iterations: int
    hyperparameters_at_boundary: NDArray[np.bool_]
    gaussian_scale_em_rate_matrix: NDArray[np.float64]
    gaussian_scale_em_spectral_radius: float
    hyperparameter_uncertainty_policy: str
    uncertainty_policy: str = "observed-laplace-conditional-on-canonical-path"
    uncertainty_label_policy: str = "conditional-on-one-whole-path-canonical-mode"
    seen_subject_future_policy: str = (
        "population-plus-lab-path-plus-carried-subject-deviation/use-global-transitions"
    )
    unseen_subject_seen_lab_policy: str = (
        "population-plus-lab-path-zero-subject-deviation/use-global-transitions"
    )
    unseen_lab_policy: str = "population-path-zero-lab-and-subject-deviation/use-global-transitions"

    def __post_init__(self) -> None:
        FitResult.__post_init__(self)
        _validate_lab_paths(self, n_states=self.n_states, require_fit_evidence=True)
        partial = protected_array(self.partial_objective_history, dtype=np.float64)
        full = protected_array(self.objective_history, dtype=np.float64)
        occupancy = protected_array(self.state_occupancy, dtype=np.float64)
        population_crossings = protected_array(self.population_label_crossings, dtype=np.bool_)
        lab_crossings = protected_array(self.lab_label_crossings, dtype=np.bool_)
        subject_crossings = protected_array(self.subject_label_crossings, dtype=np.bool_)
        restart_objectives = protected_array(
            self.initialization_restart_objectives, dtype=np.float64
        )
        restart_converged = protected_array(self.initialization_restart_converged, dtype=np.bool_)
        population_errors = protected_array(
            self.population_emission_standard_errors, dtype=np.float64
        )
        population_covariance = protected_array(
            self.population_emission_covariance, dtype=np.float64
        )
        lab_deviation_errors = protected_array(self.lab_deviation_standard_errors, dtype=np.float64)
        lab_errors = protected_array(self.lab_emission_standard_errors, dtype=np.float64)
        session_errors = protected_array(self.session_emission_standard_errors, dtype=np.float64)
        session_covariance = protected_array(self.session_emission_covariance, dtype=np.float64)
        subject_errors = protected_array(self.subject_deviation_standard_errors, dtype=np.float64)
        joint_covariance = protected_array(self.joint_emission_covariance, dtype=np.float64)
        transition_errors = protected_array(
            self.session_transition_standard_errors, dtype=np.float64
        )
        hyper_estimates = protected_array(self.hyperparameter_estimates, dtype=np.float64)
        hyper_errors = protected_array(self.hyperparameter_standard_errors, dtype=np.float64)
        hyper_covariance = protected_array(self.hyperparameter_covariance, dtype=np.float64)
        at_boundary = protected_array(self.hyperparameters_at_boundary, dtype=np.bool_)
        rate = protected_array(self.gaussian_scale_em_rate_matrix, dtype=np.float64)

        histories = (("partial", partial), ("full", full))
        if any(
            values.ndim != 1 or not len(values) or not np.all(np.isfinite(values))
            for _, values in histories
        ):
            raise ValueError("both hierarchy EM histories must contain finite objectives")
        if (
            occupancy.shape != (self.n_states,)
            or np.any(occupancy < 0)
            or not np.isclose(occupancy.sum(), 1.0)
        ):
            raise ValueError("state occupancy must contain one probability per state")
        permutation = tuple(self.canonical_permutation)
        if sorted(permutation) != list(range(self.n_states)):
            raise ValueError("canonical_permutation must permute every latent state")
        n_pairs = self.n_states * (self.n_states - 1) // 2
        if population_crossings.shape != (max(0, len(self.population_session_orders) - 1), n_pairs):
            raise ValueError("population crossings must cover adjacent population orders")
        structure = _structure_from_paths(self)
        expected_lab = sum(max(0, len(blocks) - 1) for blocks in structure.lab_blocks)
        expected_subject = sum(max(0, len(blocks) - 1) for blocks in structure.subject_blocks)
        if (
            lab_crossings.shape != (expected_lab, n_pairs)
            or len(self.lab_label_crossing_labs) != expected_lab
        ):
            raise ValueError("laboratory crossings must cover every within-lab adjacency")
        if (
            subject_crossings.shape != (expected_subject, n_pairs)
            or len(self.subject_label_crossing_subjects) != expected_subject
        ):
            raise ValueError("subject crossings must cover every within-subject adjacency")
        for name, value in (
            ("minimum_state_separation", self.minimum_state_separation),
            ("minimum_population_label_path_gap", self.minimum_population_label_path_gap),
            ("minimum_lab_label_path_gap", self.minimum_lab_label_path_gap),
            ("minimum_subject_label_path_gap", self.minimum_subject_label_path_gap),
        ):
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if (
            restart_objectives.ndim != 1
            or not len(restart_objectives)
            or restart_converged.shape != restart_objectives.shape
            or len(self.initialization_restart_messages) != len(restart_objectives)
            or not 0 <= self.initialization_selected_restart < len(restart_objectives)
        ):
            raise ValueError("initialization restart evidence must align")
        for name, value in zip(self.hyperparameter_names, hyper_estimates, strict=True):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        expected_names = (
            "population_emission_step_scale",
            "lab_emission_scale",
            "lab_emission_step_scale",
            "subject_emission_scale",
            "emission_step_scale",
            "transition_concentration",
        )
        if tuple(self.hyperparameter_names) != expected_names:
            raise ValueError("hyperparameters must identify the complete laboratory hierarchy")
        if (
            hyper_estimates.shape != (6,)
            or hyper_errors.shape != (6,)
            or hyper_covariance.shape != (6, 6)
            or at_boundary.shape != (6,)
        ):
            raise ValueError("hierarchy hyperparameter arrays must align")
        if self.hyperparameters_estimated:
            finite = np.isfinite(hyper_errors)
            if np.any(hyper_errors[finite] < 0) or not np.array_equal(
                np.isfinite(np.diag(hyper_covariance)), finite
            ):
                raise ValueError("available hierarchy hyperparameter uncertainty must align")
        elif not np.all(np.isnan(hyper_errors)) or not np.all(np.isnan(hyper_covariance)):
            raise ValueError("fixed hierarchy hyperparameters must not acquire uncertainty")
        if rate.shape != (5, 5):
            raise ValueError("the Gaussian hierarchy scale rate matrix must be five by five")

        p_width = self.population_emission_coefficients.size
        l_width = self.lab_deviation_coefficients.size
        s_width = self.session_emission_coefficients.size
        if (
            population_errors.shape != self.population_emission_coefficients.shape
            or population_covariance.shape != (p_width, p_width)
        ):
            raise ValueError("population uncertainty must cover the population path")
        if (
            lab_deviation_errors.shape != self.lab_deviation_coefficients.shape
            or lab_errors.shape != self.lab_deviation_coefficients.shape
        ):
            raise ValueError("laboratory uncertainty must cover every laboratory path point")
        if (
            session_errors.shape != self.session_emission_coefficients.shape
            or subject_errors.shape != self.session_emission_coefficients.shape
        ):
            raise ValueError("subject uncertainty must cover every subject-session path point")
        if (
            session_covariance.shape != (s_width, s_width)
            or joint_covariance.shape != (p_width + l_width + s_width,) * 2
        ):
            raise ValueError("path covariance dimensions must cover the complete hierarchy")
        if transition_errors.shape != self.session_transition_matrices.shape or not np.all(
            np.isfinite(transition_errors)
        ):
            raise ValueError("transition errors must align with session transition matrices")
        path_arrays = (
            population_errors,
            population_covariance,
            lab_deviation_errors,
            lab_errors,
            session_errors,
            subject_errors,
            session_covariance,
            joint_covariance,
        )
        if self.path_covariance_positive_definite:
            if not all(np.all(np.isfinite(values)) for values in path_arrays):
                raise ValueError("a valid nested path covariance must be finite")
        elif not all(np.all(np.isnan(values)) for values in path_arrays):
            raise ValueError("an invalid nested path covariance must remain unavailable")
        if np.isnan(self.path_hessian_condition) or self.path_hessian_condition <= 0:
            raise ValueError("path_hessian_condition must be positive and not NaN")
        if self.uncertainty_label_policy != "conditional-on-one-whole-path-canonical-mode":
            raise ValueError("uncertainty must retain one canonical path label mode")
        if not self.hyperparameter_uncertainty_policy or not self.uncertainty_policy:
            raise ValueError("uncertainty policies must remain explicit")

        object.__setattr__(self, "partial_objective_history", partial)
        object.__setattr__(self, "objective_history", full)
        object.__setattr__(self, "canonical_permutation", permutation)
        object.__setattr__(self, "state_occupancy", occupancy)
        object.__setattr__(self, "population_label_crossings", population_crossings)
        object.__setattr__(self, "lab_label_crossings", lab_crossings)
        object.__setattr__(self, "subject_label_crossings", subject_crossings)
        object.__setattr__(self, "initialization_restart_objectives", restart_objectives)
        object.__setattr__(self, "initialization_restart_converged", restart_converged)
        object.__setattr__(self, "population_emission_standard_errors", population_errors)
        object.__setattr__(self, "population_emission_covariance", population_covariance)
        object.__setattr__(self, "lab_deviation_standard_errors", lab_deviation_errors)
        object.__setattr__(self, "lab_emission_standard_errors", lab_errors)
        object.__setattr__(self, "session_emission_standard_errors", session_errors)
        object.__setattr__(self, "session_emission_covariance", session_covariance)
        object.__setattr__(self, "subject_deviation_standard_errors", subject_errors)
        object.__setattr__(self, "joint_emission_covariance", joint_covariance)
        object.__setattr__(self, "session_transition_standard_errors", transition_errors)
        object.__setattr__(self, "hyperparameter_names", expected_names)
        object.__setattr__(self, "hyperparameter_estimates", hyper_estimates)
        object.__setattr__(self, "hyperparameter_standard_errors", hyper_errors)
        object.__setattr__(self, "hyperparameter_covariance", hyper_covariance)
        object.__setattr__(self, "hyperparameters_at_boundary", at_boundary)
        object.__setattr__(self, "gaussian_scale_em_rate_matrix", rate)

    @property
    def n_states(self) -> int:
        return int(self.population_emission_coefficients.shape[1])

    @property
    def grouping(self) -> str:
        return self.lab_column

    @property
    def groups(self) -> tuple[Any, ...]:
        return self.labs

    @property
    def state_separation(self) -> float:
        return self.minimum_state_separation

    @property
    def label_order_gap(self) -> float:
        return min(
            self.minimum_population_label_path_gap,
            self.minimum_lab_label_path_gap,
            self.minimum_subject_label_path_gap,
        )

    @property
    def label_ambiguous(self) -> bool:
        return self.label_path_ambiguous

    @property
    def restart_objectives(self) -> NDArray[np.float64]:
        return self.initialization_restart_objectives

    @property
    def restart_converged(self) -> NDArray[np.bool_]:
        return self.initialization_restart_converged

    @property
    def restart_messages(self) -> tuple[str, ...]:
        return self.initialization_restart_messages

    @property
    def selected_restart(self) -> int:
        return self.initialization_selected_restart

    @property
    def lab_emission_coefficients(self) -> NDArray[np.float64]:
        positions = {
            int(order): index for index, order in enumerate(self.population_session_orders)
        }
        values = np.stack(
            [
                self.population_emission_coefficients[positions[int(order)]] + deviation
                for order, deviation in zip(
                    self.lab_path_orders,
                    self.lab_deviation_coefficients,
                    strict=True,
                )
            ]
        )
        return protected_array(values, dtype=np.float64)

    @property
    def subject_deviations(self) -> NDArray[np.float64]:
        structure = _structure_from_paths(self)
        centers = (
            self.population_emission_coefficients[structure.population_index]
            + self.lab_deviation_coefficients[structure.session_lab_index]
        )
        return protected_array(self.session_emission_coefficients - centers, dtype=np.float64)

    def lab_was_fitted(self, lab: Any) -> bool:
        return _scalar(lab) in self.labs

    def subject_was_fitted(self, subject: Any) -> bool:
        return _scalar(subject) in self.subjects

    def population_emission_intervals(self, *, level: float = 0.95) -> NDArray[np.float64]:
        return _normal_path_intervals(
            self.population_emission_coefficients,
            self.population_emission_standard_errors,
            level=level,
            available=self.path_covariance_positive_definite,
        )

    def lab_emission_intervals(self, *, level: float = 0.95) -> NDArray[np.float64]:
        return _normal_path_intervals(
            self.lab_emission_coefficients,
            self.lab_emission_standard_errors,
            level=level,
            available=self.path_covariance_positive_definite,
        )

    def lab_deviation_intervals(self, *, level: float = 0.95) -> NDArray[np.float64]:
        return _normal_path_intervals(
            self.lab_deviation_coefficients,
            self.lab_deviation_standard_errors,
            level=level,
            available=self.path_covariance_positive_definite,
        )

    def subject_deviation_intervals(self, *, level: float = 0.95) -> NDArray[np.float64]:
        return _normal_path_intervals(
            self.subject_deviations,
            self.subject_deviation_standard_errors,
            level=level,
            available=self.path_covariance_positive_definite,
        )


@dataclass(frozen=True, slots=True)
class LabHierarchicalSessionDynamicBernoulliGLMHMM(HierarchicalSessionDynamicBernoulliGLMHMM):
    """An exchangeable population-to-lab-to-subject dynamic GLM-HMM.

    For population order ``r``, laboratory ``l``, and subject ``m`` nested in that
    laboratory, the emission path is

    ``M[r] ~ Normal(M[r-1], population_emission_step_scale)``

    ``L[l, 0] ~ Normal(0, lab_emission_scale)``

    ``L[l, s] ~ Normal(L[l, s-1], lab_emission_step_scale)``

    ``D[m, 0] ~ Normal(0, subject_emission_scale)``

    ``D[m, s] ~ Normal(D[m, s-1], emission_step_scale)``

    with realized subject emission ``W[m, s] = M[r] + L[l, s] + D[m, s]``.
    """

    lab_column: str = "lab"
    lab_emission_scale: float = 0.35
    lab_emission_step_scale: float = 0.15

    def __post_init__(self) -> None:
        HierarchicalSessionDynamicBernoulliGLMHMM.__post_init__(self)
        if not isinstance(self.lab_column, str) or not self.lab_column:
            raise ValueError("lab_column must be a non-empty string")
        if self.lab_column in {"subject", "session", "trial", "session_order"}:
            raise ValueError("lab_column must differ from canonical study columns")
        for name, value in (
            ("lab_emission_scale", self.lab_emission_scale),
            ("lab_emission_step_scale", self.lab_emission_step_scale),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")

    @property
    def model_name(self) -> str:
        return "lab-hierarchical-session-dynamic-bernoulli-glm-hmm"

    @property
    def is_lab_hierarchical(self) -> bool:
        return True

    @property
    def signature(self) -> str:
        predictors = ",".join(self.predictors)
        return (
            f"{self.model_name}[states={self.n_states};outcome={self.outcome};"
            f"predictors={predictors};choice_lags={self.choice_lags};label_by={self.label_by};"
            f"lab_column={self.lab_column};l2={self.l2};population_emission_step_scale="
            f"{self.population_emission_step_scale};lab_emission_scale="
            f"{self.lab_emission_scale};lab_emission_step_scale="
            f"{self.lab_emission_step_scale};subject_emission_scale="
            f"{self.subject_emission_scale};emission_step_scale={self.emission_step_scale};"
            f"transition_concentration={self.transition_concentration};"
            f"estimate_hyperparameters={self.estimate_hyperparameters};"
            f"gaussian_scale_bounds={self.gaussian_scale_bounds};"
            f"transition_concentration_bounds={self.transition_concentration_bounds}"
            f"{self._design_signature}]"
        )

    @property
    def declared_priors(self) -> tuple[str, ...]:
        declared = [
            "population emission random walk: Normal(previous population session, "
            f"{self.population_emission_step_scale:.4g})",
            "first observed laboratory deviation from the population path: "
            f"Normal(0, {self.lab_emission_scale:.4g})",
            "laboratory deviation random walk: Normal(previous laboratory deviation, "
            f"{self.lab_emission_step_scale:.4g})",
            "first observed subject deviation from its laboratory path: "
            f"Normal(0, {self.subject_emission_scale:.4g})",
            "subject deviation random walk: Normal(previous subject deviation, "
            f"{self.emission_step_scale:.4g})",
            "independent subject-session transition rows around the population row: "
            "Dirichlet(transition_concentration * global_transition + 1), "
            f"transition_concentration={self.transition_concentration:.4g}",
        ]
        if self.l2:
            declared.append(
                "ridge on every non-intercept population-path coefficient: "
                f"Normal(0, {1.0 / self.l2**0.5:.4g}) (l2={self.l2})"
            )
        if self.estimate_hyperparameters:
            declared.extend(
                (
                    "training-only bounded Laplace-EM estimation of all five Gaussian "
                    "hierarchy scales",
                    "training-only conditional Dirichlet-multinomial estimation of "
                    "transition concentration",
                )
            )
        return tuple(declared)

    @property
    def bounded_coordinate_refusal(self) -> str:
        return (
            "a laboratory-hierarchical session-dynamic GLM-HMM already contains nested "
            "population, laboratory, and subject paths with data-dependent dimensions; "
            "another generic hierarchy or smoother would not define a valid model"
        )

    def varying_parameter_refusal(
        self, parameters: Sequence[str] | None, *, combinator: str
    ) -> str:
        del parameters, combinator
        return self.bounded_coordinate_refusal

    def simulate(
        self,
        design: Study,
        parameters: Mapping[str, float],
        *,
        seed: int | np.random.Generator,
    ) -> Study:
        return self.simulate_with_trajectories(design, parameters, seed=seed).study

    def simulate_with_states(
        self,
        design: Study,
        parameters: Mapping[str, float],
        *,
        seed: int | np.random.Generator,
    ) -> LabHierarchicalSessionDynamicGLMHMMSimulation:
        return self.simulate_with_trajectories(design, parameters, seed=seed)

    def simulate_with_trajectories(
        self,
        design: Study,
        parameters: Mapping[str, float],
        *,
        seed: int | np.random.Generator,
    ) -> LabHierarchicalSessionDynamicGLMHMMSimulation:
        """Draw all three emission levels, session transitions, states, and choices."""

        structure = _lab_structure(
            design,
            lab_column=self.lab_column,
            require_multiple_labs=True,
            require_replicated_subjects=True,
        )
        base = self.parameter_components(parameters)
        generator = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        shape = (self.n_states, len(self.coefficient_names))
        population = np.empty((len(structure.population_orders), *shape), dtype=np.float64)
        population[0] = base.emission_coefficients
        for position in range(1, len(population)):
            population[position] = population[position - 1] + generator.normal(
                0.0, self.population_emission_step_scale, shape
            )
        lab_deviations = np.empty((len(structure.lab_path_labs), *shape), dtype=np.float64)
        for blocks in structure.lab_blocks:
            deviation = generator.normal(0.0, self.lab_emission_scale, shape)
            for within_lab, block in enumerate(blocks):
                if within_lab:
                    deviation = deviation + generator.normal(
                        0.0, self.lab_emission_step_scale, shape
                    )
                lab_deviations[block] = deviation
        emissions = np.empty((len(structure.sessions), *shape), dtype=np.float64)
        centers = (
            population[structure.population_index] + lab_deviations[structure.session_lab_index]
        )
        for blocks in structure.subject_blocks:
            deviation = generator.normal(0.0, self.subject_emission_scale, shape)
            for within_subject, block in enumerate(blocks):
                if within_subject:
                    deviation = deviation + generator.normal(0.0, self.emission_step_scale, shape)
                emissions[block] = centers[block] + deviation
        transitions = np.empty(
            (len(structure.sessions), self.n_states, self.n_states), dtype=np.float64
        )
        for block in range(len(structure.sessions)):
            for state in range(self.n_states):
                transitions[block, state] = generator.dirichlet(
                    self.transition_concentration * base.transition_matrix[state] + 1.0
                )
        components = tuple(
            GLMHMMParameters(
                initial_probabilities=base.initial_probabilities,
                transition_matrix=transitions[block],
                emission_coefficients=emissions[block],
                coefficient_names=self.coefficient_names,
            )
            for block in range(len(structure.sessions))
        )
        by_opening_row = {
            int(indices[0]): components[block] for block, indices in enumerate(structure.sessions)
        }
        choices, states = self._generate(
            design, lambda indices: by_opening_row[int(indices[0])], generator
        )
        columns = {name: design[name] for name in design.columns}
        columns[self.outcome] = choices
        return LabHierarchicalSessionDynamicGLMHMMSimulation(
            study=Study(columns),
            states=states,
            n_states=self.n_states,
            lab_column=self.lab_column,
            labs=structure.labs,
            subjects=structure.subjects,
            subject_labs=structure.subject_labs,
            path_subjects=structure.path_subjects,
            path_labs=structure.path_labs,
            session_keys=structure.keys,
            session_orders=structure.orders,
            population_session_orders=structure.population_orders,
            lab_path_labs=structure.lab_path_labs,
            lab_path_orders=structure.lab_path_orders,
            population_emission_coefficients=population,
            lab_deviation_coefficients=lab_deviations,
            session_emission_coefficients=emissions,
            session_transition_matrices=transitions,
            global_transition_matrix=base.transition_matrix,
        )

    def fit(self, study: Study) -> LabHierarchicalSessionDynamicGLMHMMFitResult:
        if self.estimate_hyperparameters:
            return self._fit_estimated_lab_hyperparameters(study)
        return self._fit_fixed_lab_hyperparameters(study)

    def _fit_fixed_lab_hyperparameters(
        self, study: Study
    ) -> LabHierarchicalSessionDynamicGLMHMMFitResult:
        """Fit stationary, emission-dynamic, and fully dynamic nested MAP-EM stages."""

        structure = _lab_structure(
            study,
            lab_column=self.lab_column,
            require_multiple_labs=True,
            require_replicated_subjects=True,
        )
        outcomes = self.outcomes(study)
        features = self.design_matrix(study)
        initialization = BernoulliGLMHMM.fit(self, study)
        base = self.parameter_components(initialization)
        initial = np.asarray(base.initial_probabilities).copy()
        global_transition = np.asarray(base.transition_matrix).copy()
        population = np.tile(
            np.asarray(base.emission_coefficients),
            (len(structure.population_orders), 1, 1),
        ).copy()
        lab_deviations = np.zeros(
            (len(structure.lab_path_labs), self.n_states, len(self.coefficient_names)),
            dtype=np.float64,
        )
        emissions = np.tile(
            np.asarray(base.emission_coefficients),
            (len(structure.sessions), 1, 1),
        ).copy()
        transitions = np.tile(global_transition, (len(structure.sessions), 1, 1)).copy()

        partial_history: list[float] = []
        partial_message = "partial nested emission M-step was not run"
        partial_emission_converged = False
        partial_converged = False
        for _ in range(self.dynamic_max_iterations):
            posterior = self._dynamic_posterior(
                features,
                outcomes,
                structure.sessions,
                initial,
                emissions,
                transitions,
            )
            initial = _normalized_positive(posterior.initial_counts)
            global_transition = self._stationary_transition_m_step(
                posterior.transition_expectations,
                structure.sessions,
                global_transition,
            )
            transitions = np.tile(global_transition, (len(structure.sessions), 1, 1)).copy()
            result = self._optimize_lab_emissions(
                population,
                lab_deviations,
                emissions,
                features,
                outcomes,
                posterior.state_probabilities,
                structure,
            )
            population, lab_deviations, emissions = self._unpack_lab_coordinate(result.x, structure)
            partial_emission_converged = bool(result.success)
            partial_message = str(result.message)
            partial_history.append(
                self._lab_map_objective(
                    features,
                    outcomes,
                    initial,
                    population,
                    lab_deviations,
                    emissions,
                    transitions,
                    structure,
                    include_transition_prior=False,
                    global_transition=global_transition,
                )
            )
            if self._objective_converged(partial_history):
                partial_converged = partial_emission_converged
                break

        objective_history: list[float] = []
        emission_message = "full nested emission M-step was not run"
        emission_converged = False
        full_converged = False
        for _ in range(self.dynamic_max_iterations):
            posterior = self._dynamic_posterior(
                features,
                outcomes,
                structure.sessions,
                initial,
                emissions,
                transitions,
            )
            initial = _normalized_positive(posterior.initial_counts)
            transitions = self._transition_m_step(
                posterior.transition_expectations,
                structure.sessions,
                global_transition,
            )
            result = self._optimize_lab_emissions(
                population,
                lab_deviations,
                emissions,
                features,
                outcomes,
                posterior.state_probabilities,
                structure,
            )
            population, lab_deviations, emissions = self._unpack_lab_coordinate(result.x, structure)
            emission_converged = bool(result.success)
            emission_message = str(result.message)
            objective_history.append(
                self._lab_map_objective(
                    features,
                    outcomes,
                    initial,
                    population,
                    lab_deviations,
                    emissions,
                    transitions,
                    structure,
                    include_transition_prior=True,
                    global_transition=global_transition,
                )
            )
            if self._objective_converged(objective_history):
                full_converged = emission_converged
                break

        (
            population,
            lab_deviations,
            emissions,
            transitions,
            initial,
            global_transition,
            permutation,
        ) = self._canonicalize_lab_trajectory(
            population,
            lab_deviations,
            emissions,
            transitions,
            initial,
            global_transition,
        )
        posterior = self._dynamic_posterior(
            features,
            outcomes,
            structure.sessions,
            initial,
            emissions,
            transitions,
        )
        uncertainty = self._observed_lab_path_covariance(
            features,
            outcomes,
            initial,
            population,
            lab_deviations,
            emissions,
            transitions,
            structure,
        )
        p_width = population.size
        l_width = lab_deviations.size
        s_width = emissions.size
        if uncertainty.positive_definite:
            population_covariance = uncertainty.covariance[:p_width, :p_width]
            session_covariance = uncertainty.covariance[p_width + l_width :, p_width + l_width :]
            lab_deviation_map, lab_emission_map, subject_deviation_map = _lab_path_contrasts(
                structure,
                population.shape,
                lab_deviations.shape,
                emissions.shape,
            )
            lab_deviation_errors = _contrast_errors(
                lab_deviation_map, uncertainty.covariance, lab_deviations.shape
            )
            lab_emission_errors = _contrast_errors(
                lab_emission_map, uncertainty.covariance, lab_deviations.shape
            )
            subject_deviation_errors = _contrast_errors(
                subject_deviation_map, uncertainty.covariance, emissions.shape
            )
        else:
            population_covariance = np.full((p_width, p_width), np.nan)
            session_covariance = np.full((s_width, s_width), np.nan)
            lab_deviation_errors = np.full(lab_deviations.shape, np.nan)
            lab_emission_errors = np.full(lab_deviations.shape, np.nan)
            subject_deviation_errors = np.full(emissions.shape, np.nan)

        transition_counts = session_transition_counts(
            posterior.transition_expectations, structure.sessions
        )
        transition_errors = transition_standard_errors(
            transition_counts, self.transition_concentration, global_transition
        )
        occupancy = np.mean(posterior.state_probabilities, axis=0)
        population_crossings, population_gap = self._label_path_diagnostics(population)
        lab_paths = population[structure.lab_population_index] + lab_deviations
        lab_crossings, crossing_labs, lab_gap = self._nested_label_diagnostics(
            lab_paths, structure.labs, structure.lab_blocks
        )
        subject_crossings, crossing_subjects, subject_gap = self._nested_label_diagnostics(
            emissions, structure.subjects, structure.subject_blocks
        )
        minimum_separation = min(
            _minimum_pairwise_distance(values)
            for values in (*tuple(population), *tuple(lab_paths), *tuple(emissions))
        )
        label_ambiguous = bool(
            np.any(population_crossings)
            or np.any(lab_crossings)
            or np.any(subject_crossings)
            or min(population_gap, lab_gap, subject_gap) <= self.label_tolerance
        )
        final_components = GLMHMMParameters(
            initial_probabilities=initial,
            transition_matrix=global_transition,
            emission_coefficients=population[0],
            coefficient_names=self.coefficient_names,
        )
        estimates = self._pack_components(final_components)
        _, final_gradient = self._lab_emission_m_step_objective(
            self._pack_lab_coordinate(population, lab_deviations, emissions),
            features,
            outcomes,
            posterior.state_probabilities,
            structure,
        )
        probability_values = np.concatenate((initial, transitions.ravel()))
        boundary = bool(
            np.any(np.abs(population) >= self.coefficient_warning_threshold)
            or np.any(np.abs(lab_paths) >= self.coefficient_warning_threshold)
            or np.any(np.abs(emissions) >= self.coefficient_warning_threshold)
            or np.any(probability_values <= self.probability_warning_threshold)
            or np.any(probability_values >= 1.0 - self.probability_warning_threshold)
        )
        converged = partial_converged and full_converged
        diagnostics = FitDiagnostics(
            converged=converged,
            optimizer=(
                "nested three-stage MAP EM (pooled stationary multistart; "
                "population/lab/subject emission paths with static transitions; fully "
                "dynamic; joint L-BFGS-B emission M-steps)"
            ),
            status=0 if converged else 1,
            message=(
                "nested three-stage EM met the declared tolerance; "
                if converged
                else "nested three-stage EM did not meet the declared tolerance; "
            )
            + f"partial stage {'converged' if partial_converged else 'did not converge'}; "
            + f"full stage {'converged' if full_converged else 'did not converge'}; "
            + emission_message,
            n_iterations=len(partial_history) + len(objective_history),
            objective=float(objective_history[-1]),
            gradient_norm=float(np.linalg.norm(final_gradient)),
            hessian_condition=uncertainty.hessian_condition,
            boundary_estimate=boundary,
        )
        n_parameters = len(estimates)
        return LabHierarchicalSessionDynamicGLMHMMFitResult(
            model_name=self.model_name,
            model_signature=self.signature,
            parameter_names=self.parameter_names,
            estimates=estimates,
            standard_errors=np.full(n_parameters, np.nan),
            covariance=np.full((n_parameters, n_parameters), np.nan),
            n_observations=len(study),
            diagnostics=diagnostics,
            lab_column=self.lab_column,
            labs=structure.labs,
            subjects=structure.subjects,
            subject_labs=structure.subject_labs,
            path_subjects=structure.path_subjects,
            path_labs=structure.path_labs,
            session_keys=structure.keys,
            session_orders=structure.orders,
            population_session_orders=structure.population_orders,
            lab_path_labs=structure.lab_path_labs,
            lab_path_orders=structure.lab_path_orders,
            population_emission_coefficients=population,
            lab_deviation_coefficients=lab_deviations,
            session_emission_coefficients=emissions,
            session_transition_matrices=transitions,
            global_transition_matrix=global_transition,
            partial_objective_history=np.asarray(partial_history),
            partial_converged=partial_converged,
            partial_emission_optimizer_converged=partial_emission_converged,
            partial_emission_optimizer_message=partial_message,
            full_converged=full_converged,
            objective_history=np.asarray(objective_history),
            canonical_permutation=permutation,
            state_occupancy=occupancy,
            minimum_state_separation=minimum_separation,
            minimum_population_label_path_gap=population_gap,
            minimum_lab_label_path_gap=lab_gap,
            minimum_subject_label_path_gap=subject_gap,
            population_label_crossings=population_crossings,
            lab_label_crossings=lab_crossings,
            lab_label_crossing_labs=crossing_labs,
            subject_label_crossings=subject_crossings,
            subject_label_crossing_subjects=crossing_subjects,
            label_path_ambiguous=label_ambiguous,
            low_occupancy=bool(np.any(occupancy < self.state_occupancy_warning)),
            emission_optimizer_converged=emission_converged,
            emission_optimizer_message=emission_message,
            initialization_restart_objectives=initialization.restart_objectives,
            initialization_restart_converged=initialization.restart_converged,
            initialization_restart_messages=initialization.restart_messages,
            initialization_selected_restart=initialization.selected_restart,
            population_emission_step_scale=self.population_emission_step_scale,
            lab_emission_scale=self.lab_emission_scale,
            lab_emission_step_scale=self.lab_emission_step_scale,
            subject_emission_scale=self.subject_emission_scale,
            emission_step_scale=self.emission_step_scale,
            transition_concentration=self.transition_concentration,
            population_emission_standard_errors=uncertainty.standard_errors[:p_width].reshape(
                population.shape
            ),
            population_emission_covariance=population_covariance,
            lab_deviation_standard_errors=lab_deviation_errors,
            lab_emission_standard_errors=lab_emission_errors,
            session_emission_standard_errors=uncertainty.standard_errors[
                p_width + l_width :
            ].reshape(emissions.shape),
            session_emission_covariance=session_covariance,
            subject_deviation_standard_errors=subject_deviation_errors,
            joint_emission_covariance=uncertainty.covariance,
            session_transition_standard_errors=transition_errors,
            path_hessian_condition=uncertainty.hessian_condition,
            path_covariance_positive_definite=uncertainty.positive_definite,
            hyperparameter_names=_LAB_HYPERPARAMETER_NAMES,
            hyperparameter_estimates=np.asarray(
                [
                    self.population_emission_step_scale,
                    self.lab_emission_scale,
                    self.lab_emission_step_scale,
                    self.subject_emission_scale,
                    self.emission_step_scale,
                    self.transition_concentration,
                ]
            ),
            hyperparameter_standard_errors=np.full(6, np.nan),
            hyperparameter_covariance=np.full((6, 6), np.nan),
            hyperparameters_estimated=False,
            hyperparameter_estimation_converged=False,
            hyperparameter_estimation_iterations=0,
            hyperparameters_at_boundary=np.zeros(6, dtype=np.bool_),
            gaussian_scale_em_rate_matrix=np.full((5, 5), np.nan),
            gaussian_scale_em_spectral_radius=float("nan"),
            hyperparameter_uncertainty_policy="fixed-hyperparameters",
            uncertainty_policy=(
                "observed-laplace-conditional-on-canonical-path"
                if uncertainty.positive_definite
                else "unavailable-nonpositive-observed-path-information"
            ),
        )

    def _fit_estimated_lab_hyperparameters(
        self, study: Study
    ) -> LabHierarchicalSessionDynamicGLMHMMFitResult:
        """Estimate five Gaussian scales and the transition concentration in training."""

        structure = _lab_structure(
            study,
            lab_column=self.lab_column,
            require_multiple_labs=True,
            require_replicated_subjects=True,
        )
        if len(structure.population_orders) < 2:
            raise ModelDataError(
                "estimating the population random-walk scale requires two population orders"
            )
        if not all(len(blocks) > 1 for blocks in structure.lab_blocks):
            raise ModelDataError(
                "estimating the laboratory random-walk scale requires repeated orders in "
                "every laboratory"
            )
        if not any(len(blocks) > 1 for blocks in structure.subject_blocks):
            raise ModelDataError(
                "estimating the subject random-walk scale requires a repeated-session subject"
            )
        scales = np.clip(
            np.asarray(
                [
                    self.population_emission_step_scale,
                    self.lab_emission_scale,
                    self.lab_emission_step_scale,
                    self.subject_emission_scale,
                    self.emission_step_scale,
                ],
                dtype=np.float64,
            ),
            *self.gaussian_scale_bounds,
        )
        concentration = float(
            np.clip(self.transition_concentration, *self.transition_concentration_bounds)
        )
        converged = False
        iterations = 0
        for iteration in range(1, self.hyperparameter_max_iterations + 1):
            iterations = iteration
            fixed = self._with_lab_hyperparameters(scales, concentration)
            fitted = fixed._fit_fixed_lab_hyperparameters(study)
            if not fitted.path_covariance_positive_definite:
                raise ModelDataError(
                    "laboratory hierarchy hyperparameter estimation requires a "
                    "positive-definite observed path covariance"
                )
            coordinate = fixed._pack_lab_coordinate(
                fitted.population_emission_coefficients,
                fitted.lab_deviation_coefficients,
                fitted.session_emission_coefficients,
            )
            contrasts = _lab_hierarchy_scale_contrasts(
                structure,
                fitted.population_emission_coefficients.shape,
                fitted.lab_deviation_coefficients.shape,
                fitted.session_emission_coefficients.shape,
            )
            updated_scales = update_gaussian_scales(
                coordinate,
                fitted.joint_emission_covariance,
                contrasts,
                self.gaussian_scale_bounds,
            )
            posterior = fixed._dynamic_posterior(
                fixed.design_matrix(study),
                fixed.outcomes(study),
                structure.sessions,
                fixed.parameter_components(fitted).initial_probabilities,
                fitted.session_emission_coefficients,
                fitted.session_transition_matrices,
            )
            counts = session_transition_counts(
                posterior.transition_expectations, structure.sessions
            )
            updated_concentration, _, concentration_converged = estimate_transition_concentration(
                counts,
                fitted.global_transition_matrix,
                self.transition_concentration_bounds,
            )
            change = max(
                float(np.max(np.abs(np.log(updated_scales / scales)))),
                abs(float(np.log(updated_concentration / concentration))),
            )
            scales = updated_scales
            concentration = updated_concentration
            if change <= self.hyperparameter_tolerance and concentration_converged:
                converged = True
                break

        final_model = self._with_lab_hyperparameters(scales, concentration)
        fitted = final_model._fit_fixed_lab_hyperparameters(study)
        if not fitted.path_covariance_positive_definite:
            raise ModelDataError(
                "estimated laboratory hierarchy requires a positive-definite final path covariance"
            )
        coordinate = final_model._pack_lab_coordinate(
            fitted.population_emission_coefficients,
            fitted.lab_deviation_coefficients,
            fitted.session_emission_coefficients,
        )
        contrasts = _lab_hierarchy_scale_contrasts(
            structure,
            fitted.population_emission_coefficients.shape,
            fitted.lab_deviation_coefficients.shape,
            fitted.session_emission_coefficients.shape,
        )
        scale_covariance, scale_errors, scale_valid = gaussian_scale_observed_covariance(
            scales,
            coordinate,
            fitted.joint_emission_covariance,
            contrasts,
        )
        rate = np.full((5, 5), np.nan, dtype=np.float64)
        spectral_radius = float("nan")
        scale_policy = "louis-observed-information"
        if not scale_valid:

            def scale_update(candidate: NDArray[np.float64]) -> NDArray[np.float64]:
                probe = self._with_lab_hyperparameters(candidate, concentration)
                probe_fit = probe._fit_fixed_lab_hyperparameters(study)
                if not probe_fit.path_covariance_positive_definite:
                    return np.full(5, np.nan)
                probe_coordinate = probe._pack_lab_coordinate(
                    probe_fit.population_emission_coefficients,
                    probe_fit.lab_deviation_coefficients,
                    probe_fit.session_emission_coefficients,
                )
                probe_contrasts = _lab_hierarchy_scale_contrasts(
                    structure,
                    probe_fit.population_emission_coefficients.shape,
                    probe_fit.lab_deviation_coefficients.shape,
                    probe_fit.session_emission_coefficients.shape,
                )
                return update_gaussian_scales(
                    probe_coordinate,
                    probe_fit.joint_emission_covariance,
                    probe_contrasts,
                    self.gaussian_scale_bounds,
                )

            (
                scale_covariance,
                scale_errors,
                rate,
                spectral_radius,
                scale_valid,
            ) = supplemented_scale_covariance(
                scales,
                scale_update,
                self.gaussian_scale_bounds,
                tuple(len(contrast) for contrast in contrasts),
                self.hyperparameter_tolerance,
            )
            scale_policy = "supplemented-em"
        posterior = final_model._dynamic_posterior(
            final_model.design_matrix(study),
            final_model.outcomes(study),
            structure.sessions,
            final_model.parameter_components(fitted).initial_probabilities,
            fitted.session_emission_coefficients,
            fitted.session_transition_matrices,
        )
        counts = session_transition_counts(posterior.transition_expectations, structure.sessions)
        _, concentration_error, concentration_valid = estimate_transition_concentration(
            counts,
            fitted.global_transition_matrix,
            self.transition_concentration_bounds,
        )
        hyper_covariance = np.full((6, 6), np.nan, dtype=np.float64)
        hyper_errors = np.full(6, np.nan, dtype=np.float64)
        if scale_valid:
            hyper_covariance[:5, :5] = scale_covariance
            hyper_errors[:5] = scale_errors
        if concentration_valid and np.isfinite(concentration_error):
            hyper_covariance[5, 5] = concentration_error**2
            hyper_errors[5] = concentration_error
        if np.all(np.isfinite(hyper_errors)):
            hyper_covariance[:5, 5] = hyper_covariance[5, :5] = 0.0
        values = np.concatenate((scales, [concentration]))
        boundary_tolerance = max(self.hyperparameter_tolerance, 1e-4)
        at_boundary = np.asarray(
            [
                *(
                    at_log_bound(float(scale), self.gaussian_scale_bounds, boundary_tolerance)
                    for scale in scales
                ),
                at_log_bound(
                    concentration,
                    self.transition_concentration_bounds,
                    boundary_tolerance,
                ),
            ],
            dtype=np.bool_,
        )
        fixed_converged = bool(fitted.diagnostics.converged)
        diagnostics = replace(
            fitted.diagnostics,
            converged=fixed_converged and converged,
            optimizer=f"{fitted.diagnostics.optimizer}; outer nested Laplace-EM/Dirichlet evidence",
            status=0 if fixed_converged and converged else 1,
            message=(
                f"{fitted.diagnostics.message}; laboratory hierarchy hyperparameter "
                f"estimation {'converged' if converged else 'did not converge'} after "
                f"{iterations} iterations"
            ),
            n_iterations=(fitted.diagnostics.n_iterations or 0) + iterations,
            boundary_estimate=bool(fitted.diagnostics.boundary_estimate or np.any(at_boundary)),
        )
        reported_scale_policy = (
            scale_policy if scale_valid else "scale-unavailable-unstable-information"
        )
        return replace(
            fitted,
            model_signature=self.signature,
            diagnostics=diagnostics,
            population_emission_step_scale=float(scales[0]),
            lab_emission_scale=float(scales[1]),
            lab_emission_step_scale=float(scales[2]),
            subject_emission_scale=float(scales[3]),
            emission_step_scale=float(scales[4]),
            transition_concentration=concentration,
            hyperparameter_estimates=values,
            hyperparameter_standard_errors=hyper_errors,
            hyperparameter_covariance=hyper_covariance,
            hyperparameters_estimated=True,
            hyperparameter_estimation_converged=converged,
            hyperparameter_estimation_iterations=iterations,
            hyperparameters_at_boundary=at_boundary,
            gaussian_scale_em_rate_matrix=rate,
            gaussian_scale_em_spectral_radius=spectral_radius,
            hyperparameter_uncertainty_policy=(
                f"{reported_scale_policy};conditional-dirichlet-multinomial;"
                "block-diagonal-cross-family-covariance"
            ),
        )

    def _with_lab_hyperparameters(
        self, scales: NDArray[np.float64], concentration: float
    ) -> LabHierarchicalSessionDynamicBernoulliGLMHMM:
        return replace(
            self,
            estimate_hyperparameters=False,
            population_emission_step_scale=float(scales[0]),
            lab_emission_scale=float(scales[1]),
            lab_emission_step_scale=float(scales[2]),
            subject_emission_scale=float(scales[3]),
            emission_step_scale=float(scales[4]),
            transition_concentration=float(concentration),
        )

    def _pack_lab_coordinate(
        self,
        population: NDArray[np.float64],
        lab_deviations: NDArray[np.float64],
        emissions: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return np.concatenate((population.ravel(), lab_deviations.ravel(), emissions.ravel()))

    def _unpack_lab_coordinate(
        self,
        vector: NDArray[np.float64],
        structure: _LabStructure,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        width = self.n_states * len(self.coefficient_names)
        p_size = len(structure.population_orders) * width
        l_size = len(structure.lab_path_labs) * width
        values = np.asarray(vector, dtype=np.float64)
        population = values[:p_size].reshape(
            len(structure.population_orders), self.n_states, len(self.coefficient_names)
        )
        lab_deviations = values[p_size : p_size + l_size].reshape(
            len(structure.lab_path_labs), self.n_states, len(self.coefficient_names)
        )
        emissions = values[p_size + l_size :].reshape(
            len(structure.sessions), self.n_states, len(self.coefficient_names)
        )
        return population, lab_deviations, emissions

    def _optimize_lab_emissions(
        self,
        population: NDArray[np.float64],
        lab_deviations: NDArray[np.float64],
        emissions: NDArray[np.float64],
        features: NDArray[np.float64],
        outcomes: NDArray[np.float64],
        state_probabilities: NDArray[np.float64],
        structure: _LabStructure,
    ) -> Any:
        initial = self._pack_lab_coordinate(population, lab_deviations, emissions)
        return minimize(
            lambda vector: self._lab_emission_m_step_objective(
                vector,
                features,
                outcomes,
                state_probabilities,
                structure,
            ),
            initial,
            method="L-BFGS-B",
            jac=True,
            bounds=[(-30.0, 30.0)] * len(initial),
            options={
                "maxiter": self.max_iterations,
                "ftol": self.tolerance,
                "gtol": self.tolerance,
            },
        )

    def _lab_emission_m_step_objective(
        self,
        vector: NDArray[np.float64],
        features: NDArray[np.float64],
        outcomes: NDArray[np.float64],
        state_probabilities: NDArray[np.float64],
        structure: _LabStructure,
    ) -> tuple[float, NDArray[np.float64]]:
        population, lab_deviations, emissions = self._unpack_lab_coordinate(vector, structure)
        population_gradient = np.zeros_like(population)
        lab_gradient = np.zeros_like(lab_deviations)
        emission_gradient = np.zeros_like(emissions)
        loss = 0.0
        for block, session_indices in enumerate(structure.sessions):
            index = np.asarray(session_indices, dtype=np.intp)
            linear = features[index] @ emissions[block].T
            weights = state_probabilities[index]
            loss += float(
                np.sum(weights * (np.logaddexp(0.0, linear) - outcomes[index, None] * linear))
            )
            residual = weights * (expit(linear) - outcomes[index, None])
            emission_gradient[block] = residual.T @ features[index]
        loss += self._lab_emission_prior(
            population,
            lab_deviations,
            emissions,
            population_gradient,
            lab_gradient,
            emission_gradient,
            structure,
        )
        return float(loss), self._pack_lab_coordinate(
            population_gradient, lab_gradient, emission_gradient
        )

    def _lab_emission_prior(
        self,
        population: NDArray[np.float64],
        lab_deviations: NDArray[np.float64],
        emissions: NDArray[np.float64],
        population_gradient: NDArray[np.float64] | None,
        lab_gradient: NDArray[np.float64] | None,
        emission_gradient: NDArray[np.float64] | None,
        structure: _LabStructure,
    ) -> float:
        loss = 0.0
        population_differences = np.diff(population, axis=0)
        population_precision = 1.0 / self.population_emission_step_scale**2
        loss += 0.5 * population_precision * float(np.sum(population_differences**2))
        if population_gradient is not None and len(population) > 1:
            population_gradient[:-1] -= population_precision * population_differences
            population_gradient[1:] += population_precision * population_differences

        lab_initial_precision = 1.0 / self.lab_emission_scale**2
        lab_step_precision = 1.0 / self.lab_emission_step_scale**2
        for blocks in structure.lab_blocks:
            first = blocks[0]
            loss += 0.5 * lab_initial_precision * float(np.sum(lab_deviations[first] ** 2))
            if lab_gradient is not None:
                lab_gradient[first] += lab_initial_precision * lab_deviations[first]
            for previous, current in pairwise(blocks):
                difference = lab_deviations[current] - lab_deviations[previous]
                loss += 0.5 * lab_step_precision * float(np.sum(difference**2))
                if lab_gradient is not None:
                    value = lab_step_precision * difference
                    lab_gradient[previous] -= value
                    lab_gradient[current] += value

        centers = (
            population[structure.population_index] + lab_deviations[structure.session_lab_index]
        )
        subject_deviations = emissions - centers
        subject_initial_precision = 1.0 / self.subject_emission_scale**2
        subject_step_precision = 1.0 / self.emission_step_scale**2
        for blocks in structure.subject_blocks:
            first = blocks[0]
            loss += 0.5 * subject_initial_precision * float(np.sum(subject_deviations[first] ** 2))
            if (
                population_gradient is not None
                and lab_gradient is not None
                and emission_gradient is not None
            ):
                value = subject_initial_precision * subject_deviations[first]
                emission_gradient[first] += value
                population_gradient[structure.population_index[first]] -= value
                lab_gradient[structure.session_lab_index[first]] -= value
            for previous, current in pairwise(blocks):
                difference = subject_deviations[current] - subject_deviations[previous]
                loss += 0.5 * subject_step_precision * float(np.sum(difference**2))
                if (
                    population_gradient is not None
                    and lab_gradient is not None
                    and emission_gradient is not None
                ):
                    value = subject_step_precision * difference
                    emission_gradient[previous] -= value
                    population_gradient[structure.population_index[previous]] += value
                    lab_gradient[structure.session_lab_index[previous]] += value
                    emission_gradient[current] += value
                    population_gradient[structure.population_index[current]] -= value
                    lab_gradient[structure.session_lab_index[current]] -= value
        if self.l2:
            penalized = np.asarray(
                [name != "intercept" for name in self.coefficient_names], dtype=bool
            )
            loss += 0.5 * self.l2 * float(np.sum(population[:, :, penalized] ** 2))
            if population_gradient is not None:
                population_gradient[:, :, penalized] += self.l2 * population[:, :, penalized]
        return float(loss)

    def _lab_map_objective(
        self,
        features: NDArray[np.float64],
        outcomes: NDArray[np.float64],
        initial: NDArray[np.float64],
        population: NDArray[np.float64],
        lab_deviations: NDArray[np.float64],
        emissions: NDArray[np.float64],
        transitions: NDArray[np.float64],
        structure: _LabStructure,
        *,
        include_transition_prior: bool,
        global_transition: NDArray[np.float64],
    ) -> float:
        posterior = self._dynamic_posterior(
            features,
            outcomes,
            structure.sessions,
            initial,
            emissions,
            transitions,
        )
        loss = -posterior.log_likelihood
        loss += self._lab_emission_prior(
            population,
            lab_deviations,
            emissions,
            None,
            None,
            None,
            structure,
        )
        if include_transition_prior:
            loss -= self.transition_concentration * float(
                np.sum(global_transition[None, :, :] * np.log(transitions))
            )
        return float(loss)

    def _observed_lab_path_covariance(
        self,
        features: NDArray[np.float64],
        outcomes: NDArray[np.float64],
        initial: NDArray[np.float64],
        population: NDArray[np.float64],
        lab_deviations: NDArray[np.float64],
        emissions: NDArray[np.float64],
        transitions: NDArray[np.float64],
        structure: _LabStructure,
    ) -> Any:
        mode = self._pack_lab_coordinate(population, lab_deviations, emissions)

        def gradient(vector: NDArray[np.float64]) -> NDArray[np.float64]:
            _, _, candidate_emissions = self._unpack_lab_coordinate(vector, structure)
            posterior = self._dynamic_posterior(
                features,
                outcomes,
                structure.sessions,
                initial,
                candidate_emissions,
                transitions,
            )
            _, values = self._lab_emission_m_step_objective(
                vector,
                features,
                outcomes,
                posterior.state_probabilities,
                structure,
            )
            return values

        return observed_path_covariance(gradient, mode)

    def _canonicalize_lab_trajectory(
        self,
        population: NDArray[np.float64],
        lab_deviations: NDArray[np.float64],
        emissions: NDArray[np.float64],
        transitions: NDArray[np.float64],
        initial: NDArray[np.float64],
        global_transition: NDArray[np.float64],
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        tuple[int, ...],
    ]:
        label_index = self.coefficient_names.index(self.label_by)
        means = np.mean(population, axis=0)
        other = tuple(index for index in range(means.shape[1]) if index != label_index)
        permutation = tuple(
            sorted(
                range(self.n_states),
                key=lambda state: (
                    float(means[state, label_index]),
                    *(float(means[state, index]) for index in other),
                ),
            )
        )
        indices = np.asarray(permutation, dtype=np.intp)
        return (
            population[:, indices, :],
            lab_deviations[:, indices, :],
            emissions[:, indices, :],
            transitions[:, indices][:, :, indices],
            initial[indices],
            global_transition[np.ix_(indices, indices)],
            permutation,
        )

    def _nested_label_diagnostics(
        self,
        paths: NDArray[np.float64],
        groups: tuple[Any, ...],
        blocks_by_group: tuple[tuple[int, ...], ...],
    ) -> tuple[NDArray[np.bool_], tuple[Any, ...], float]:
        crossing_rows: list[NDArray[np.bool_]] = []
        crossing_groups: list[Any] = []
        gaps: list[float] = []
        n_pairs = self.n_states * (self.n_states - 1) // 2
        for group, blocks in zip(groups, blocks_by_group, strict=True):
            crossings, gap = self._label_path_diagnostics(paths[np.asarray(blocks)])
            gaps.append(gap)
            crossing_rows.extend(crossings)
            crossing_groups.extend([group] * len(crossings))
        values = (
            np.stack(crossing_rows) if crossing_rows else np.zeros((0, n_pairs), dtype=np.bool_)
        )
        return np.asarray(values, dtype=np.bool_), tuple(crossing_groups), float(min(gaps))

    def transition_probabilities(
        self,
        study: Study,
        parameters: Mapping[str, float] | FitResult,
    ) -> NDArray[np.float64]:
        if not isinstance(parameters, FitResult):
            return BernoulliGLMHMM.transition_probabilities(self, study, parameters)
        fit = self._validate_lab_fit(parameters)
        components = self._lab_prediction_components(study, fit)
        values = np.empty((len(study), self.n_states, self.n_states), dtype=np.float64)
        for block, indices in enumerate(ordered_session_indices(study)):
            values[np.asarray(indices, dtype=np.intp)] = components[block].transition_matrix
        return protected_array(values, dtype=np.float64)

    def predict(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> Prediction:
        prediction_mode = self._prediction_mode(mode)
        lab_fit = self._validate_lab_fit(fit)
        outcomes = self.outcomes(study)
        features = self.design_matrix(study)
        components = self._lab_prediction_components(study, lab_fit)
        probabilities = self._dynamic_filtered(features, outcomes, study, components)
        emission_probability = np.empty((len(study), self.n_states), dtype=np.float64)
        for block, indices in enumerate(ordered_session_indices(study)):
            index = np.asarray(indices, dtype=np.intp)
            emission_probability[index] = expit(
                features[index] @ components[block].emission_coefficients.T
            )
        probability = np.sum(probabilities.predictive * emission_probability, axis=1)
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
        outcomes = self.outcomes(study)
        probability = self.predict(study, fit, mode=mode).probability
        scores = outcomes * np.log(probability) + (1.0 - outcomes) * np.log1p(-probability)
        return protected_array(scores, dtype=np.float64)

    def state_probabilities(self, study: Study, fit: FitResult) -> FilteredStateProbabilities:
        lab_fit = self._validate_lab_fit(fit)
        return self._dynamic_filtered(
            self.design_matrix(study),
            self.outcomes(study),
            study,
            self._lab_prediction_components(study, lab_fit),
        )

    def state_recovery(
        self,
        simulation: LabHierarchicalSessionDynamicGLMHMMSimulation,
        fit: FitResult,
        *,
        ambiguity_tolerance: float = 0.05,
    ) -> LatentStateAlignment:
        if not isinstance(simulation, LabHierarchicalSessionDynamicGLMHMMSimulation):
            raise TypeError("simulation must be a LabHierarchicalSessionDynamicGLMHMMSimulation")
        if simulation.n_states != self.n_states:
            raise ValueError("simulation and model must contain the same number of states")
        return align_latent_states(
            simulation.states,
            self.state_probabilities(simulation.study, fit).filtered,
            ambiguity_tolerance=ambiguity_tolerance,
        )

    def trajectory_recovery(
        self,
        simulation: LabHierarchicalSessionDynamicGLMHMMSimulation,
        fit: FitResult,
        *,
        ambiguity_tolerance: float = 0.05,
    ) -> LabHierarchicalSessionDynamicTrajectoryRecovery:
        lab_fit = self._validate_lab_fit(fit)
        if not isinstance(simulation, LabHierarchicalSessionDynamicGLMHMMSimulation):
            raise TypeError("simulation must be a LabHierarchicalSessionDynamicGLMHMMSimulation")
        identities_match = (
            simulation.lab_column == lab_fit.lab_column
            and simulation.labs == lab_fit.labs
            and simulation.subjects == lab_fit.subjects
            and simulation.subject_labs == lab_fit.subject_labs
            and simulation.path_subjects == lab_fit.path_subjects
            and simulation.path_labs == lab_fit.path_labs
            and simulation.session_keys == lab_fit.session_keys
            and simulation.lab_path_labs == lab_fit.lab_path_labs
            and np.array_equal(simulation.session_orders, lab_fit.session_orders)
            and np.array_equal(simulation.lab_path_orders, lab_fit.lab_path_orders)
            and np.array_equal(
                simulation.population_session_orders,
                lab_fit.population_session_orders,
            )
        )
        if not identities_match:
            raise ValueError("simulation and fit must describe the same nested paths")
        alignment = self.state_recovery(
            simulation, lab_fit, ambiguity_tolerance=ambiguity_tolerance
        )
        mapping = np.asarray(alignment.reference_to_inferred, dtype=np.intp)
        population_error = (
            lab_fit.population_emission_coefficients[:, mapping, :]
            - simulation.population_emission_coefficients
        )
        lab_error = (
            lab_fit.lab_deviation_coefficients[:, mapping, :]
            - simulation.lab_deviation_coefficients
        )
        subject_error = (
            lab_fit.session_emission_coefficients[:, mapping, :]
            - simulation.session_emission_coefficients
        )
        transition_error = (
            lab_fit.session_transition_matrices[:, mapping][:, :, mapping]
            - simulation.session_transition_matrices
        )
        lab_by_group = [
            float(np.sqrt(np.mean(lab_error[np.asarray(blocks)] ** 2)))
            for blocks in _structure_from_paths(lab_fit).lab_blocks
        ]
        subject_by_group = [
            float(np.sqrt(np.mean(subject_error[np.asarray(blocks)] ** 2)))
            for blocks in _structure_from_paths(lab_fit).subject_blocks
        ]
        return LabHierarchicalSessionDynamicTrajectoryRecovery(
            alignment=alignment,
            population_emission_rmse=float(np.sqrt(np.mean(population_error**2))),
            lab_deviation_rmse=float(np.sqrt(np.mean(lab_error**2))),
            subject_emission_rmse=float(np.sqrt(np.mean(subject_error**2))),
            transition_rmse=float(np.sqrt(np.mean(transition_error**2))),
            lab_deviation_rmse_by_lab=np.asarray(lab_by_group),
            subject_emission_rmse_by_subject=np.asarray(subject_by_group),
        )

    def predict_new_subjects(
        self,
        study: Study,
        fit: FitResult,
        *,
        n_draws: int,
        seed: int,
        include_fitted_path_uncertainty: bool = True,
    ) -> UnseenSubjectInLabDynamicPrediction:
        """Integrate new subject paths conditional on laboratories represented in training."""

        lab_fit = self._validate_lab_fit(fit)
        _validate_draw_request(
            n_draws,
            seed,
            include_fitted_path_uncertainty,
            uncertainty_name="include_fitted_path_uncertainty",
        )
        if include_fitted_path_uncertainty and not lab_fit.path_covariance_positive_definite:
            raise ModelDataError(
                "fitted population/laboratory path integration requires a positive-definite "
                "path covariance"
            )
        structure = _lab_structure(
            study,
            lab_column=self.lab_column,
            require_multiple_labs=False,
            require_replicated_subjects=False,
        )
        subject_overlap = set(structure.subjects) & set(lab_fit.subjects)
        if subject_overlap:
            raise ValueError(
                "predict_new_subjects requires entirely unseen subjects; fitted labels "
                f"include {sorted(subject_overlap, key=repr)!r}"
            )
        unseen_labs = set(structure.labs) - set(lab_fit.labs)
        if unseen_labs:
            raise ValueError(
                "predict_new_subjects conditions on fitted laboratories; use "
                "predict_new_labs for "
                f"{sorted(unseen_labs, key=repr)!r}"
            )
        self._validate_integrated_orders(structure, lab_fit, require_seen_labs=True)

        generator = np.random.default_rng(seed)
        features = self.design_matrix(study)
        outcomes = self.outcomes(study)
        base = self.parameter_components(lab_fit)
        probabilities, log_probabilities, emission_draws, transition_draws = (
            self._allocate_prediction_draws(n_draws, len(study), len(structure.sessions))
        )
        p_width = lab_fit.population_emission_coefficients.size
        l_width = lab_fit.lab_deviation_coefficients.size
        path_mean = np.concatenate(
            (
                lab_fit.population_emission_coefficients.ravel(),
                lab_fit.lab_deviation_coefficients.ravel(),
            )
        )
        path_covariance = lab_fit.joint_emission_covariance[
            : p_width + l_width, : p_width + l_width
        ]
        for draw in range(n_draws):
            if include_fitted_path_uncertainty:
                sampled = generator.multivariate_normal(
                    path_mean, path_covariance, check_valid="raise"
                )
            else:
                sampled = path_mean
            population = sampled[:p_width].reshape(lab_fit.population_emission_coefficients.shape)
            lab_deviations = sampled[p_width:].reshape(lab_fit.lab_deviation_coefficients.shape)
            by_order = self._sample_population_orders(
                structure.orders,
                lab_fit,
                population,
                generator,
            )
            by_lab_order = self._sample_seen_lab_orders(
                structure,
                lab_fit,
                lab_deviations,
                generator,
            )
            shape = (self.n_states, len(self.coefficient_names))
            for blocks in structure.subject_blocks:
                deviation = generator.normal(0.0, lab_fit.subject_emission_scale, shape)
                for within_subject, block in enumerate(blocks):
                    if within_subject:
                        deviation = deviation + generator.normal(
                            0.0, lab_fit.emission_step_scale, shape
                        )
                    identity = (
                        structure.path_labs[block],
                        int(structure.orders[block]),
                    )
                    emission_draws[draw, block] = (
                        by_order[int(structure.orders[block])] + by_lab_order[identity] + deviation
                    )
            self._draw_transitions(transition_draws[draw], lab_fit, generator)
            self._score_prediction_draw(
                draw,
                study,
                structure,
                features,
                outcomes,
                base.initial_probabilities,
                emission_draws,
                transition_draws,
                probabilities,
                log_probabilities,
            )
        subject_position = {subject: index for index, subject in enumerate(structure.subjects)}
        row_subjects = np.asarray(
            [subject_position[_scalar(value)] for value in study["subject"]], dtype=np.intp
        )
        summary = _joint_prediction_summary(
            log_probabilities, row_subjects, len(structure.subjects)
        )
        return UnseenSubjectInLabDynamicPrediction(
            labs=structure.labs,
            subjects=structure.subjects,
            subject_labs=structure.subject_labs,
            row_subject_indices=row_subjects,
            probability=np.mean(probabilities, axis=0),
            draw_probabilities=probabilities,
            draw_pointwise_log_probability=log_probabilities,
            pointwise_marginal_log_probability=(
                logsumexp(log_probabilities, axis=0) - np.log(float(n_draws))
            ),
            subject_joint_log_probability=summary[0],
            subject_effective_draws=summary[1],
            subject_log_probability_mcse=summary[2],
            draw_session_emission_coefficients=emission_draws,
            draw_session_transition_matrices=transition_draws,
            n_draws=n_draws,
            seed=seed,
            includes_fitted_path_uncertainty=include_fitted_path_uncertainty,
            label_path_ambiguous=lab_fit.label_path_ambiguous,
        )

    def predict_new_labs(
        self,
        study: Study,
        fit: FitResult,
        *,
        n_draws: int,
        seed: int,
        include_population_path_uncertainty: bool = True,
    ) -> UnseenLabDynamicPrediction:
        """Integrate coherent laboratory and subject paths for entirely unseen labs."""

        lab_fit = self._validate_lab_fit(fit)
        _validate_draw_request(
            n_draws,
            seed,
            include_population_path_uncertainty,
            uncertainty_name="include_population_path_uncertainty",
        )
        if include_population_path_uncertainty and not lab_fit.path_covariance_positive_definite:
            raise ModelDataError(
                "population path integration requires a positive-definite fitted covariance"
            )
        structure = _lab_structure(
            study,
            lab_column=self.lab_column,
            require_multiple_labs=False,
            require_replicated_subjects=False,
        )
        lab_overlap = set(structure.labs) & set(lab_fit.labs)
        if lab_overlap:
            raise ValueError(
                "predict_new_labs requires entirely unseen laboratories; fitted labels "
                f"include {sorted(lab_overlap, key=repr)!r}"
            )
        subject_overlap = set(structure.subjects) & set(lab_fit.subjects)
        if subject_overlap:
            raise ValueError(
                "subjects cannot move to unseen laboratories; fitted labels include "
                f"{sorted(subject_overlap, key=repr)!r}"
            )
        self._validate_integrated_orders(structure, lab_fit, require_seen_labs=False)

        generator = np.random.default_rng(seed)
        features = self.design_matrix(study)
        outcomes = self.outcomes(study)
        base = self.parameter_components(lab_fit)
        probabilities, log_probabilities, emission_draws, transition_draws = (
            self._allocate_prediction_draws(n_draws, len(study), len(structure.sessions))
        )
        shape = (self.n_states, len(self.coefficient_names))
        lab_draws = np.empty((n_draws, len(structure.lab_path_labs), *shape), dtype=np.float64)
        population_mean = lab_fit.population_emission_coefficients.ravel()
        for draw in range(n_draws):
            if include_population_path_uncertainty:
                population = generator.multivariate_normal(
                    population_mean,
                    lab_fit.population_emission_covariance,
                    check_valid="raise",
                ).reshape(lab_fit.population_emission_coefficients.shape)
            else:
                population = np.asarray(lab_fit.population_emission_coefficients)
            by_order = self._sample_population_orders(
                structure.orders, lab_fit, population, generator
            )
            for blocks in structure.lab_blocks:
                deviation = generator.normal(0.0, lab_fit.lab_emission_scale, shape)
                for within_lab, lab_block in enumerate(blocks):
                    if within_lab:
                        deviation = deviation + generator.normal(
                            0.0, lab_fit.lab_emission_step_scale, shape
                        )
                    lab_draws[draw, lab_block] = deviation
            for blocks in structure.subject_blocks:
                deviation = generator.normal(0.0, lab_fit.subject_emission_scale, shape)
                for within_subject, block in enumerate(blocks):
                    if within_subject:
                        deviation = deviation + generator.normal(
                            0.0, lab_fit.emission_step_scale, shape
                        )
                    emission_draws[draw, block] = (
                        by_order[int(structure.orders[block])]
                        + lab_draws[draw, structure.session_lab_index[block]]
                        + deviation
                    )
            self._draw_transitions(transition_draws[draw], lab_fit, generator)
            self._score_prediction_draw(
                draw,
                study,
                structure,
                features,
                outcomes,
                base.initial_probabilities,
                emission_draws,
                transition_draws,
                probabilities,
                log_probabilities,
            )
        lab_position = {lab: index for index, lab in enumerate(structure.labs)}
        row_labs = np.asarray(
            [lab_position[_scalar(value)] for value in study[self.lab_column]],
            dtype=np.intp,
        )
        summary = _joint_prediction_summary(log_probabilities, row_labs, len(structure.labs))
        return UnseenLabDynamicPrediction(
            labs=structure.labs,
            subjects=structure.subjects,
            subject_labs=structure.subject_labs,
            row_lab_indices=row_labs,
            probability=np.mean(probabilities, axis=0),
            draw_probabilities=probabilities,
            draw_pointwise_log_probability=log_probabilities,
            pointwise_marginal_log_probability=(
                logsumexp(log_probabilities, axis=0) - np.log(float(n_draws))
            ),
            lab_joint_log_probability=summary[0],
            lab_effective_draws=summary[1],
            lab_log_probability_mcse=summary[2],
            draw_lab_deviation_coefficients=lab_draws,
            draw_session_emission_coefficients=emission_draws,
            draw_session_transition_matrices=transition_draws,
            n_draws=n_draws,
            seed=seed,
            includes_population_path_uncertainty=include_population_path_uncertainty,
            label_path_ambiguous=lab_fit.label_path_ambiguous,
        )

    def _validate_lab_fit(self, fit: FitResult) -> LabHierarchicalSessionDynamicGLMHMMFitResult:
        self._validate_fit(fit)
        if not isinstance(fit, LabHierarchicalSessionDynamicGLMHMMFitResult):
            raise TypeError(
                "laboratory-hierarchical prediction requires a "
                "LabHierarchicalSessionDynamicGLMHMMFitResult"
            )
        return fit

    def _lab_prediction_components(
        self,
        study: Study,
        fit: LabHierarchicalSessionDynamicGLMHMMFitResult,
    ) -> tuple[GLMHMMParameters, ...]:
        structure = _lab_structure(
            study,
            lab_column=self.lab_column,
            require_multiple_labs=False,
            require_replicated_subjects=False,
        )
        fitted_blocks = {
            (subject, key): (lab, int(order), block)
            for block, (subject, lab, key, order) in enumerate(
                zip(
                    fit.path_subjects,
                    fit.path_labs,
                    fit.session_keys,
                    fit.session_orders,
                    strict=True,
                )
            )
        }
        fitted_subject_labs = dict(zip(fit.subjects, fit.subject_labs, strict=True))
        subject_blocks = _subject_blocks(fit.subjects, fit.path_subjects)
        last_by_subject = {
            subject: blocks[-1]
            for subject, blocks in zip(fit.subjects, subject_blocks, strict=True)
        }
        population_positions = {
            int(order): position for position, order in enumerate(fit.population_session_orders)
        }
        lab_positions = {
            (lab, int(order)): position
            for position, (lab, order) in enumerate(
                zip(fit.lab_path_labs, fit.lab_path_orders, strict=True)
            )
        }
        lab_blocks = _blocks_by_group(fit.labs, fit.lab_path_labs)
        last_by_lab = {lab: blocks[-1] for lab, blocks in zip(fit.labs, lab_blocks, strict=True)}
        base = self.parameter_components(fit)
        components: list[GLMHMMParameters] = []
        for subject, lab, key, order_value in zip(
            structure.path_subjects,
            structure.path_labs,
            structure.keys,
            structure.orders,
            strict=True,
        ):
            order = int(order_value)
            identity = (subject, key)
            if subject in fitted_subject_labs and fitted_subject_labs[subject] != lab:
                raise ModelDataError(
                    f"subject {subject!r} was fitted in laboratory "
                    f"{fitted_subject_labs[subject]!r}, not {lab!r}"
                )
            if identity in fitted_blocks:
                fitted_lab, fitted_order, block = fitted_blocks[identity]
                if lab != fitted_lab or order != fitted_order:
                    raise ModelDataError(
                        f"subject {subject!r} session {key!r} was fitted in lab/order "
                        f"({fitted_lab!r}, {fitted_order}), not ({lab!r}, {order})"
                    )
                emissions = fit.session_emission_coefficients[block]
                transitions = fit.session_transition_matrices[block]
            else:
                population = self._population_for_order(order, fit, population_positions)
                lab_deviation = self._lab_deviation_for_order(
                    lab,
                    order,
                    fit,
                    lab_positions,
                    last_by_lab,
                )
                if subject in last_by_subject:
                    last_block = last_by_subject[subject]
                    last_order = int(fit.session_orders[last_block])
                    if order <= last_order:
                        raise ModelDataError(
                            f"subject {subject!r} session {key!r} at order {order} was not "
                            f"fitted and is not later than that subject's final training "
                            f"session ({last_order})"
                        )
                    last_lab = fit.path_labs[last_block]
                    last_lab_deviation = fit.lab_deviation_coefficients[
                        lab_positions[(last_lab, last_order)]
                    ]
                    last_population = fit.population_emission_coefficients[
                        population_positions[last_order]
                    ]
                    subject_deviation = (
                        fit.session_emission_coefficients[last_block]
                        - last_population
                        - last_lab_deviation
                    )
                else:
                    subject_deviation = 0.0
                emissions = population + lab_deviation + subject_deviation
                transitions = fit.global_transition_matrix
            components.append(
                GLMHMMParameters(
                    initial_probabilities=base.initial_probabilities,
                    transition_matrix=transitions,
                    emission_coefficients=emissions,
                    coefficient_names=self.coefficient_names,
                )
            )
        return tuple(components)

    def _population_for_order(
        self,
        order: int,
        fit: LabHierarchicalSessionDynamicGLMHMMFitResult,
        positions: Mapping[int, int],
    ) -> NDArray[np.float64]:
        if order in positions:
            return fit.population_emission_coefficients[positions[order]]
        last_order = int(fit.population_session_orders[-1])
        if order > last_order:
            return fit.population_emission_coefficients[-1]
        raise ModelDataError(
            f"population session order {order} was not fitted and is not prospectively "
            f"later than the final population order ({last_order})"
        )

    def _lab_deviation_for_order(
        self,
        lab: Any,
        order: int,
        fit: LabHierarchicalSessionDynamicGLMHMMFitResult,
        positions: Mapping[tuple[Any, int], int],
        last_by_lab: Mapping[Any, int],
    ) -> NDArray[np.float64] | float:
        identity = (lab, order)
        if identity in positions:
            return fit.lab_deviation_coefficients[positions[identity]]
        if lab not in last_by_lab:
            return 0.0
        last_block = last_by_lab[lab]
        last_order = int(fit.lab_path_orders[last_block])
        if order > last_order:
            return fit.lab_deviation_coefficients[last_block]
        raise ModelDataError(
            f"laboratory {lab!r} order {order} was not fitted and is not prospectively "
            f"later than that laboratory's final order ({last_order})"
        )

    def _validate_integrated_orders(
        self,
        structure: _LabStructure,
        fit: LabHierarchicalSessionDynamicGLMHMMFitResult,
        *,
        require_seen_labs: bool,
    ) -> None:
        fitted_population_orders = set(int(value) for value in fit.population_session_orders)
        final_population_order = int(fit.population_session_orders[-1])
        invalid_population = sorted(
            {
                int(order)
                for order in structure.orders
                if int(order) not in fitted_population_orders
                and int(order) <= final_population_order
            }
        )
        if invalid_population:
            raise ModelDataError(
                "integrated prediction cannot invent earlier or interleaved population "
                f"orders {invalid_population!r}"
            )
        if not require_seen_labs:
            return
        fitted_by_lab = {
            lab: sorted(
                int(order)
                for candidate, order in zip(fit.lab_path_labs, fit.lab_path_orders, strict=True)
                if candidate == lab
            )
            for lab in fit.labs
        }
        invalid: list[tuple[Any, int]] = []
        for lab, order in zip(structure.lab_path_labs, structure.lab_path_orders, strict=True):
            fitted = fitted_by_lab[lab]
            if int(order) not in fitted and int(order) <= fitted[-1]:
                invalid.append((lab, int(order)))
        if invalid:
            raise ModelDataError(
                "integrated prediction cannot invent earlier or interleaved laboratory "
                f"orders {invalid!r}"
            )

    def _sample_population_orders(
        self,
        requested: NDArray[np.int64],
        fit: LabHierarchicalSessionDynamicGLMHMMFitResult,
        population: NDArray[np.float64],
        generator: np.random.Generator,
    ) -> dict[int, NDArray[np.float64]]:
        fitted = {
            int(order): population[position]
            for position, order in enumerate(fit.population_session_orders)
        }
        final_order = int(fit.population_session_orders[-1])
        previous = population[-1]
        shape = (self.n_states, len(self.coefficient_names))
        for order in sorted(set(int(value) for value in requested)):
            if order > final_order:
                previous = previous + generator.normal(
                    0.0, fit.population_emission_step_scale, shape
                )
                fitted[order] = previous
        return fitted

    def _sample_seen_lab_orders(
        self,
        structure: _LabStructure,
        fit: LabHierarchicalSessionDynamicGLMHMMFitResult,
        lab_deviations: NDArray[np.float64],
        generator: np.random.Generator,
    ) -> dict[tuple[Any, int], NDArray[np.float64]]:
        values = {
            (lab, int(order)): lab_deviations[position]
            for position, (lab, order) in enumerate(
                zip(fit.lab_path_labs, fit.lab_path_orders, strict=True)
            )
        }
        shape = (self.n_states, len(self.coefficient_names))
        for lab in structure.labs:
            fitted_orders = sorted(order for candidate, order in values if candidate == lab)
            previous = values[(lab, fitted_orders[-1])]
            for order in sorted(
                int(value)
                for candidate, value in zip(
                    structure.lab_path_labs, structure.lab_path_orders, strict=True
                )
                if candidate == lab and int(value) > fitted_orders[-1]
            ):
                previous = previous + generator.normal(0.0, fit.lab_emission_step_scale, shape)
                values[(lab, order)] = previous
        return values

    def _allocate_prediction_draws(
        self, n_draws: int, n_rows: int, n_blocks: int
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        probabilities = np.empty((n_draws, n_rows), dtype=np.float64)
        log_probabilities = np.empty_like(probabilities)
        emissions = np.empty(
            (
                n_draws,
                n_blocks,
                self.n_states,
                len(self.coefficient_names),
            ),
            dtype=np.float64,
        )
        transitions = np.empty((n_draws, n_blocks, self.n_states, self.n_states), dtype=np.float64)
        return probabilities, log_probabilities, emissions, transitions

    def _draw_transitions(
        self,
        target: NDArray[np.float64],
        fit: LabHierarchicalSessionDynamicGLMHMMFitResult,
        generator: np.random.Generator,
    ) -> None:
        for block in range(len(target)):
            for state in range(self.n_states):
                target[block, state] = generator.dirichlet(
                    fit.transition_concentration * fit.global_transition_matrix[state] + 1.0
                )

    def _score_prediction_draw(
        self,
        draw: int,
        study: Study,
        structure: _LabStructure,
        features: NDArray[np.float64],
        outcomes: NDArray[np.float64],
        initial: NDArray[np.float64],
        emission_draws: NDArray[np.float64],
        transition_draws: NDArray[np.float64],
        probabilities: NDArray[np.float64],
        log_probabilities: NDArray[np.float64],
    ) -> None:
        components = tuple(
            GLMHMMParameters(
                initial_probabilities=initial,
                transition_matrix=transition_draws[draw, block],
                emission_coefficients=emission_draws[draw, block],
                coefficient_names=self.coefficient_names,
            )
            for block in range(len(structure.sessions))
        )
        filtered = self._dynamic_filtered(features, outcomes, study, components)
        emission_probability = np.empty((len(study), self.n_states), dtype=np.float64)
        for block, indices in enumerate(structure.sessions):
            index = np.asarray(indices, dtype=np.intp)
            emission_probability[index] = expit(features[index] @ emission_draws[draw, block].T)
        probability = np.sum(filtered.predictive * emission_probability, axis=1)
        probability = np.clip(probability, np.finfo(float).tiny, 1.0 - np.finfo(float).eps)
        probabilities[draw] = probability
        log_probabilities[draw] = outcomes * np.log(probability) + (1.0 - outcomes) * np.log1p(
            -probability
        )


_LAB_HYPERPARAMETER_NAMES = (
    "population_emission_step_scale",
    "lab_emission_scale",
    "lab_emission_step_scale",
    "subject_emission_scale",
    "emission_step_scale",
    "transition_concentration",
)


def _lab_structure(
    study: Study,
    *,
    lab_column: str,
    require_multiple_labs: bool,
    require_replicated_subjects: bool,
) -> _LabStructure:
    if lab_column not in study.columns:
        raise ModelDataError(f"study does not contain laboratory column {lab_column!r}")
    subjects = tuple(_scalar(value) for value in study.subjects)
    lab_values = tuple(_scalar(value) for value in study[lab_column])
    if any(_missing(value) for value in lab_values):
        raise ModelDataError("laboratory labels must not be missing")
    subject_labs: list[Any] = []
    for subject in subjects:
        assigned = _unique_equal(
            lab
            for candidate, lab in zip(study["subject"], lab_values, strict=True)
            if _scalar(candidate) == subject
        )
        if len(assigned) != 1:
            raise ModelDataError(
                f"subject {subject!r} must be nested in exactly one laboratory; found {assigned!r}"
            )
        subject_labs.append(assigned[0])
    labs = _unique_equal(subject_labs)
    if require_multiple_labs and len(labs) < 2:
        raise ModelDataError(
            "LabHierarchicalSessionDynamicBernoulliGLMHMM requires at least two laboratories"
        )
    if require_replicated_subjects:
        counts = {lab: sum(_equal(value, lab) for value in subject_labs) for lab in labs}
        unreplicated = {lab: count for lab, count in counts.items() if count < 2}
        if unreplicated:
            raise ModelDataError(
                "laboratory effects require at least two independent subjects in every "
                f"laboratory; found {unreplicated!r}"
            )

    sessions = ordered_session_indices(study)
    path_subjects: list[Any] = []
    path_labs: list[Any] = []
    keys: list[Any] = []
    orders: list[int] = []
    for indices in sessions:
        opening = indices[0]
        path_subjects.append(_scalar(study["subject"][opening]))
        path_labs.append(_scalar(study[lab_column][opening]))
        keys.append(_scalar(study["session"][opening]))
        orders.append(int(study["session_order"][opening]))
    population_orders = np.asarray(sorted(set(orders)), dtype=np.int64)
    population_position = {int(order): index for index, order in enumerate(population_orders)}
    population_index = np.asarray([population_position[order] for order in orders], dtype=np.intp)
    subject_blocks = _subject_blocks(subjects, tuple(path_subjects))

    lab_path_labs: list[Any] = []
    lab_path_orders: list[int] = []
    lab_blocks: list[tuple[int, ...]] = []
    lab_identity_position: dict[tuple[Any, int], int] = {}
    for lab in labs:
        positions: list[int] = []
        for order in sorted(
            {
                order
                for candidate, order in zip(path_labs, orders, strict=True)
                if _equal(candidate, lab)
            }
        ):
            position = len(lab_path_labs)
            lab_path_labs.append(lab)
            lab_path_orders.append(order)
            lab_identity_position[(lab, order)] = position
            positions.append(position)
        lab_blocks.append(tuple(positions))
    session_lab_index = np.asarray(
        [lab_identity_position[(lab, order)] for lab, order in zip(path_labs, orders, strict=True)],
        dtype=np.intp,
    )
    lab_population_index = np.asarray(
        [population_position[order] for order in lab_path_orders], dtype=np.intp
    )
    return _LabStructure(
        labs=tuple(labs),
        subjects=subjects,
        subject_labs=tuple(subject_labs),
        path_subjects=tuple(path_subjects),
        path_labs=tuple(path_labs),
        keys=tuple(keys),
        orders=np.asarray(orders, dtype=np.int64),
        sessions=sessions,
        population_orders=population_orders,
        population_index=population_index,
        subject_blocks=subject_blocks,
        lab_path_labs=tuple(lab_path_labs),
        lab_path_orders=np.asarray(lab_path_orders, dtype=np.int64),
        lab_population_index=lab_population_index,
        session_lab_index=session_lab_index,
        lab_blocks=tuple(lab_blocks),
    )


def _structure_from_paths(target: Any) -> _LabStructure:
    """Reconstruct the index-only structure retained by a simulation or fit."""

    population_positions = {
        int(order): index for index, order in enumerate(target.population_session_orders)
    }
    lab_positions = {
        (lab, int(order)): index
        for index, (lab, order) in enumerate(
            zip(target.lab_path_labs, target.lab_path_orders, strict=True)
        )
    }
    population_index = np.asarray(
        [population_positions[int(order)] for order in target.session_orders], dtype=np.intp
    )
    session_lab_index = np.asarray(
        [
            lab_positions[(lab, int(order))]
            for lab, order in zip(target.path_labs, target.session_orders, strict=True)
        ],
        dtype=np.intp,
    )
    lab_population_index = np.asarray(
        [population_positions[int(order)] for order in target.lab_path_orders], dtype=np.intp
    )
    return _LabStructure(
        labs=tuple(target.labs),
        subjects=tuple(target.subjects),
        subject_labs=tuple(target.subject_labs),
        path_subjects=tuple(target.path_subjects),
        path_labs=tuple(target.path_labs),
        keys=tuple(target.session_keys),
        orders=np.asarray(target.session_orders, dtype=np.int64),
        sessions=(),
        population_orders=np.asarray(target.population_session_orders, dtype=np.int64),
        population_index=population_index,
        subject_blocks=_subject_blocks(tuple(target.subjects), tuple(target.path_subjects)),
        lab_path_labs=tuple(target.lab_path_labs),
        lab_path_orders=np.asarray(target.lab_path_orders, dtype=np.int64),
        lab_population_index=lab_population_index,
        session_lab_index=session_lab_index,
        lab_blocks=_blocks_by_group(tuple(target.labs), tuple(target.lab_path_labs)),
    )


def _validate_lab_paths(target: Any, *, n_states: int, require_fit_evidence: bool) -> None:
    labs = tuple(_scalar(value) for value in target.labs)
    subjects = tuple(_scalar(value) for value in target.subjects)
    subject_labs = tuple(_scalar(value) for value in target.subject_labs)
    path_subjects = tuple(_scalar(value) for value in target.path_subjects)
    path_labs = tuple(_scalar(value) for value in target.path_labs)
    keys = tuple(_scalar(value) for value in target.session_keys)
    orders = protected_array(target.session_orders, dtype=np.int64)
    population_orders = protected_array(target.population_session_orders, dtype=np.int64)
    lab_path_labs = tuple(_scalar(value) for value in target.lab_path_labs)
    lab_path_orders = protected_array(target.lab_path_orders, dtype=np.int64)
    population = protected_array(target.population_emission_coefficients, dtype=np.float64)
    lab_deviations = protected_array(target.lab_deviation_coefficients, dtype=np.float64)
    emissions = protected_array(target.session_emission_coefficients, dtype=np.float64)
    transitions = protected_array(target.session_transition_matrices, dtype=np.float64)
    global_transition = protected_array(target.global_transition_matrix, dtype=np.float64)
    if not isinstance(target.lab_column, str) or not target.lab_column:
        raise ValueError("lab_column must be a non-empty string")
    if n_states < 2 or len(labs) < 2 or len(set(labs)) != len(labs):
        raise ValueError("nested paths require at least two unique laboratories and states")
    if len(subjects) < 4 or len(set(subjects)) != len(subjects):
        raise ValueError("nested paths require unique replicated subjects")
    if len(subject_labs) != len(subjects) or not set(subject_labs).issubset(set(labs)):
        raise ValueError("each subject must name one fitted laboratory")
    if require_fit_evidence:
        counts = {lab: sum(_equal(value, lab) for value in subject_labs) for lab in labs}
        if any(count < 2 for count in counts.values()):
            raise ValueError("each fitted laboratory must retain at least two subjects")
    n_sessions = len(path_subjects)
    if (
        not n_sessions
        or len(path_labs) != n_sessions
        or len(keys) != n_sessions
        or orders.shape != (n_sessions,)
    ):
        raise ValueError("subject-session path labels must align")
    assignment = dict(zip(subjects, subject_labs, strict=True))
    if any(
        subject not in assignment or assignment[subject] != lab
        for subject, lab in zip(path_subjects, path_labs, strict=True)
    ):
        raise ValueError("subject-session paths must respect the retained laboratory nesting")
    if len(set(zip(path_subjects, keys, strict=True))) != n_sessions:
        raise ValueError("session keys must be unique within subject")
    for blocks in _subject_blocks(subjects, path_subjects):
        if not blocks or np.any(np.diff(orders[np.asarray(blocks)]) <= 0):
            raise ValueError("session orders must increase strictly within every subject")
    if (
        population_orders.ndim != 1
        or not len(population_orders)
        or np.any(np.diff(population_orders) <= 0)
        or set(orders.tolist()) - set(population_orders.tolist())
    ):
        raise ValueError("population orders must increase and cover every session path")
    n_lab_paths = len(lab_path_labs)
    if lab_path_orders.shape != (n_lab_paths,) or not set(lab_path_labs).issubset(set(labs)):
        raise ValueError("laboratory path labels and orders must align")
    if len(set(zip(lab_path_labs, lab_path_orders.tolist(), strict=True))) != n_lab_paths:
        raise ValueError("laboratory/order path identities must be unique")
    for blocks in _blocks_by_group(labs, lab_path_labs):
        if not blocks or np.any(np.diff(lab_path_orders[np.asarray(blocks)]) <= 0):
            raise ValueError("laboratory path orders must increase within laboratory")
    if population.ndim != 3 or population.shape[:2] != (len(population_orders), n_states):
        raise ValueError("population emissions must cover order and state")
    expected_trailing = (n_states, population.shape[2])
    if lab_deviations.shape != (n_lab_paths, *expected_trailing):
        raise ValueError("laboratory deviations must cover every laboratory/order path")
    if emissions.shape != (n_sessions, *expected_trailing):
        raise ValueError("session emissions must cover every subject-session path")
    if not all(np.all(np.isfinite(values)) for values in (population, lab_deviations, emissions)):
        raise ValueError("every emission hierarchy path must be finite")
    if transitions.shape != (n_sessions, n_states, n_states) or global_transition.shape != (
        n_states,
        n_states,
    ):
        raise ValueError("transition arrays must align with sessions and states")
    for name, values in (
        ("session transitions", transitions),
        ("global transition", global_transition),
    ):
        if (
            not np.all(np.isfinite(values))
            or np.any(values <= 0)
            or not np.allclose(values.sum(axis=-1), 1.0)
        ):
            raise ValueError(f"{name} must contain strictly positive probability rows")
    for name, values in (
        ("labs", labs),
        ("subjects", subjects),
        ("subject_labs", subject_labs),
        ("path_subjects", path_subjects),
        ("path_labs", path_labs),
        ("session_keys", keys),
        ("lab_path_labs", lab_path_labs),
    ):
        object.__setattr__(target, name, values)
    object.__setattr__(target, "session_orders", orders)
    object.__setattr__(target, "population_session_orders", population_orders)
    object.__setattr__(target, "lab_path_orders", lab_path_orders)
    object.__setattr__(target, "population_emission_coefficients", population)
    object.__setattr__(target, "lab_deviation_coefficients", lab_deviations)
    object.__setattr__(target, "session_emission_coefficients", emissions)
    object.__setattr__(target, "session_transition_matrices", transitions)
    object.__setattr__(target, "global_transition_matrix", global_transition)


def _blocks_by_group(
    groups: tuple[Any, ...], values: tuple[Any, ...]
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(index for index, value in enumerate(values) if _equal(value, group))
        for group in groups
    )


def _lab_path_contrasts(
    structure: _LabStructure,
    population_shape: tuple[int, ...],
    lab_shape: tuple[int, ...],
    emission_shape: tuple[int, ...],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Map joint ``(population, lab deviation, subject path)`` to reported paths."""

    trailing = int(np.prod(population_shape[1:], dtype=np.int64))
    p_width = int(np.prod(population_shape, dtype=np.int64))
    l_width = int(np.prod(lab_shape, dtype=np.int64))
    s_width = int(np.prod(emission_shape, dtype=np.int64))
    width = p_width + l_width + s_width
    lab_deviation = np.zeros((l_width, width), dtype=np.float64)
    lab_deviation[:, p_width : p_width + l_width] = np.eye(l_width)
    lab_emission = lab_deviation.copy()
    for block, population_index in enumerate(structure.lab_population_index):
        for offset in range(trailing):
            row = block * trailing + offset
            lab_emission[row, int(population_index) * trailing + offset] = 1.0
    subject_deviation = np.zeros((s_width, width), dtype=np.float64)
    for block, (population_index, lab_index) in enumerate(
        zip(structure.population_index, structure.session_lab_index, strict=True)
    ):
        for offset in range(trailing):
            row = block * trailing + offset
            subject_deviation[row, int(population_index) * trailing + offset] = -1.0
            subject_deviation[row, p_width + int(lab_index) * trailing + offset] = -1.0
            subject_deviation[row, p_width + l_width + row] = 1.0
    return lab_deviation, lab_emission, subject_deviation


def _lab_hierarchy_scale_contrasts(
    structure: _LabStructure,
    population_shape: tuple[int, ...],
    lab_shape: tuple[int, ...],
    emission_shape: tuple[int, ...],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Contrasts for population steps, lab initial/steps, and subject initial/steps."""

    trailing = int(np.prod(population_shape[1:], dtype=np.int64))
    p_width = int(np.prod(population_shape, dtype=np.int64))
    l_width = int(np.prod(lab_shape, dtype=np.int64))
    s_width = int(np.prod(emission_shape, dtype=np.int64))
    joint_width = p_width + l_width + s_width
    population_rows: list[NDArray[np.float64]] = []
    for position in range(1, population_shape[0]):
        for offset in range(trailing):
            row = np.zeros(joint_width)
            row[(position - 1) * trailing + offset] = -1.0
            row[position * trailing + offset] = 1.0
            population_rows.append(row)
    lab_deviation, _, subject_deviation = _lab_path_contrasts(
        structure, population_shape, lab_shape, emission_shape
    )
    lab_initial_rows = np.asarray(
        [
            block * trailing + offset
            for blocks in structure.lab_blocks
            for block in blocks[:1]
            for offset in range(trailing)
        ],
        dtype=np.intp,
    )
    lab_step_rows: list[NDArray[np.float64]] = []
    for blocks in structure.lab_blocks:
        for previous, current in pairwise(blocks):
            for offset in range(trailing):
                lab_step_rows.append(
                    lab_deviation[current * trailing + offset]
                    - lab_deviation[previous * trailing + offset]
                )
    subject_initial_rows = np.asarray(
        [
            block * trailing + offset
            for blocks in structure.subject_blocks
            for block in blocks[:1]
            for offset in range(trailing)
        ],
        dtype=np.intp,
    )
    subject_step_rows: list[NDArray[np.float64]] = []
    for blocks in structure.subject_blocks:
        for previous, current in pairwise(blocks):
            for offset in range(trailing):
                subject_step_rows.append(
                    subject_deviation[current * trailing + offset]
                    - subject_deviation[previous * trailing + offset]
                )
    if not population_rows or not lab_step_rows or not subject_step_rows:
        raise ValueError(
            "nested scale contrasts require repeated population, lab, and subject paths"
        )
    return (
        np.stack(population_rows),
        lab_deviation[lab_initial_rows],
        np.stack(lab_step_rows),
        subject_deviation[subject_initial_rows],
        np.stack(subject_step_rows),
    )


def _contrast_errors(
    contrast: NDArray[np.float64],
    covariance: NDArray[np.float64],
    shape: tuple[int, ...],
) -> NDArray[np.float64]:
    variance = np.einsum("ij,jk,ik->i", contrast, covariance, contrast, optimize=True)
    return np.sqrt(np.maximum(variance, 0.0)).reshape(shape)


def _validate_draw_request(
    n_draws: int, seed: int, include_uncertainty: bool, *, uncertainty_name: str
) -> None:
    if isinstance(n_draws, bool) or not isinstance(n_draws, int) or n_draws < 2:
        raise ValueError("n_draws must be an integer of at least two")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not isinstance(include_uncertainty, bool):
        raise ValueError(f"{uncertainty_name} must be boolean")


def _joint_prediction_summary(
    log_probabilities: NDArray[np.float64],
    row_groups: NDArray[np.intp],
    n_groups: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    n_draws = len(log_probabilities)
    joint = np.empty(n_groups, dtype=np.float64)
    effective = np.empty(n_groups, dtype=np.float64)
    mcse = np.empty(n_groups, dtype=np.float64)
    for group in range(n_groups):
        weights = np.sum(log_probabilities[:, row_groups == group], axis=1)
        log_weight_sum = float(logsumexp(weights))
        joint[group] = log_weight_sum - np.log(float(n_draws))
        normalized = np.exp(weights - log_weight_sum)
        effective[group] = 1.0 / float(np.sum(normalized**2))
        shifted = np.exp(weights - float(np.max(weights)))
        mcse[group] = float(np.std(shifted, ddof=1) / np.sqrt(n_draws) / np.mean(shifted))
    return joint, effective, mcse


def _validate_integrated_prediction(
    target: Any,
    *,
    groups: tuple[Any, ...],
    row_group_indices: NDArray[np.intp],
    joint: NDArray[np.float64],
    effective: NDArray[np.float64],
    mcse: NDArray[np.float64],
) -> None:
    group_values = tuple(_scalar(value) for value in groups)
    rows = protected_array(row_group_indices, dtype=np.intp)
    probability = protected_array(target.probability, dtype=np.float64)
    draws = protected_array(target.draw_probabilities, dtype=np.float64)
    draw_log = protected_array(target.draw_pointwise_log_probability, dtype=np.float64)
    marginal = protected_array(target.pointwise_marginal_log_probability, dtype=np.float64)
    joint_values = protected_array(joint, dtype=np.float64)
    effective_values = protected_array(effective, dtype=np.float64)
    mcse_values = protected_array(mcse, dtype=np.float64)
    emission_draws = protected_array(target.draw_session_emission_coefficients, dtype=np.float64)
    transition_draws = protected_array(target.draw_session_transition_matrices, dtype=np.float64)
    if not group_values or len(set(group_values)) != len(group_values):
        raise ValueError("prediction groups must be unique and non-empty")
    if probability.ndim != 1 or rows.shape != probability.shape:
        raise ValueError("prediction rows and group indices must align")
    if draws.shape != (target.n_draws, len(probability)) or draw_log.shape != draws.shape:
        raise ValueError("predictive draws must contain one value per draw and row")
    if marginal.shape != probability.shape:
        raise ValueError("pointwise marginal scores must align with prediction rows")
    if (
        joint_values.shape != (len(group_values),)
        or effective_values.shape != joint_values.shape
        or mcse_values.shape != joint_values.shape
    ):
        raise ValueError("joint Monte Carlo diagnostics must align with prediction groups")
    if emission_draws.ndim != 4 or emission_draws.shape[0] != target.n_draws:
        raise ValueError("emission draws must cover draw, session, state, coefficient")
    if transition_draws.ndim != 4 or transition_draws.shape[:2] != emission_draws.shape[:2]:
        raise ValueError("transition draws must align with draw and session")
    if np.any((rows < 0) | (rows >= len(group_values))):
        raise ValueError("row group indices must identify prediction groups")
    arrays = (
        probability,
        draws,
        draw_log,
        marginal,
        joint_values,
        effective_values,
        mcse_values,
        emission_draws,
        transition_draws,
    )
    if not all(np.all(np.isfinite(values)) for values in arrays):
        raise ValueError("integrated prediction arrays must be finite")
    if not np.all((probability > 0) & (probability < 1)):
        raise ValueError("marginal probabilities must lie strictly inside zero and one")
    if not np.all((effective_values >= 1.0) & (effective_values <= target.n_draws + 1e-8)):
        raise ValueError("effective draws must lie between one and n_draws")
    if np.any(mcse_values < 0):
        raise ValueError("Monte Carlo errors must be non-negative")
    if target.label_policy != "conditional-on-one-whole-path-canonical-mode":
        raise ValueError("label_policy must retain the fitted canonical path mode")
    object.__setattr__(
        target,
        "row_subject_indices" if hasattr(target, "row_subject_indices") else "row_lab_indices",
        rows,
    )
    object.__setattr__(target, "probability", probability)
    object.__setattr__(target, "draw_probabilities", draws)
    object.__setattr__(target, "draw_pointwise_log_probability", draw_log)
    object.__setattr__(target, "pointwise_marginal_log_probability", marginal)
    joint_name = (
        "subject_joint_log_probability"
        if hasattr(target, "subject_joint_log_probability")
        else "lab_joint_log_probability"
    )
    effective_name = (
        "subject_effective_draws"
        if hasattr(target, "subject_effective_draws")
        else "lab_effective_draws"
    )
    mcse_name = (
        "subject_log_probability_mcse"
        if hasattr(target, "subject_log_probability_mcse")
        else "lab_log_probability_mcse"
    )
    object.__setattr__(target, joint_name, joint_values)
    object.__setattr__(target, effective_name, effective_values)
    object.__setattr__(target, mcse_name, mcse_values)
    object.__setattr__(target, "draw_session_emission_coefficients", emission_draws)
    object.__setattr__(target, "draw_session_transition_matrices", transition_draws)


def _missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        result = value != value
    except (TypeError, ValueError):
        return False
    return bool(result) if isinstance(result, (bool, np.bool_)) else False


def _equal(left: Any, right: Any) -> bool:
    try:
        value = left == right
    except (TypeError, ValueError):
        return False
    return bool(value) if isinstance(value, (bool, np.bool_)) else False


def _unique_equal(values: Sequence[Any] | Any) -> tuple[Any, ...]:
    unique: list[Any] = []
    for value in values:
        scalar = _scalar(value)
        if not any(_equal(scalar, existing) for existing in unique):
            unique.append(scalar)
    return tuple(unique)
