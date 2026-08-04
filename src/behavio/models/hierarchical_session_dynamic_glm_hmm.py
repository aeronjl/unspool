"""Cross-subject session-dynamic Bernoulli GLM-HMMs.

The population extension in this module is deliberately explicit.  Population emission
weights follow a Gaussian random walk over the observed session-order coordinate.  Each
subject has a zero-centred deviation at its first observed session and that deviation follows
its own Gaussian random walk thereafter.  Subject-session transition matrices are
conditionally independent Dirichlet draws around one population matrix; they are not a
second temporal random walk.

This is a mixed-effects extension of the session-dynamic model, not a claim that the source
paper fitted a population hierarchy.  The extra Gaussian distributions are named, simulated,
fitted, and used for unseen-subject prediction rather than being implied by pooled data.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import pairwise
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import expit, logsumexp, ndtri

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
from behavio.models.session_dynamic_glm_hmm import SessionDynamicBernoulliGLMHMM
from behavio.models.state_alignment import LatentStateAlignment, align_latent_states
from behavio.trials import Study


@dataclass(frozen=True, slots=True)
class HierarchicalSessionDynamicGLMHMMSimulation:
    """Observed choices with retained population, subject-path, and state truth."""

    study: Study
    states: NDArray[np.int64]
    n_states: int
    subjects: tuple[Any, ...]
    path_subjects: tuple[Any, ...]
    session_keys: tuple[Any, ...]
    session_orders: NDArray[np.int64]
    population_session_orders: NDArray[np.int64]
    population_emission_coefficients: NDArray[np.float64]
    session_emission_coefficients: NDArray[np.float64]
    session_transition_matrices: NDArray[np.float64]
    global_transition_matrix: NDArray[np.float64]

    def __post_init__(self) -> None:
        states = protected_array(self.states, dtype=np.int64)
        if states.shape != (len(self.study),) or np.any((states < 0) | (states >= self.n_states)):
            raise ValueError("states must contain one valid label per trial")
        validated = _validated_population_trajectory(
            self.subjects,
            self.path_subjects,
            self.session_keys,
            self.session_orders,
            self.population_session_orders,
            self.population_emission_coefficients,
            self.session_emission_coefficients,
            self.session_transition_matrices,
            self.global_transition_matrix,
            n_states=self.n_states,
        )
        object.__setattr__(self, "states", states)
        _set_validated_trajectory(self, validated)


@dataclass(frozen=True, slots=True)
class HierarchicalSessionDynamicTrajectoryRecovery:
    """Truth-aligned population, subject, and transition-path recovery."""

    alignment: LatentStateAlignment
    population_emission_rmse: float
    subject_emission_rmse: float
    transition_rmse: float
    subject_emission_rmse_by_subject: NDArray[np.float64]

    def __post_init__(self) -> None:
        values = protected_array(self.subject_emission_rmse_by_subject, dtype=np.float64)
        scalars = (
            self.population_emission_rmse,
            self.subject_emission_rmse,
            self.transition_rmse,
        )
        if values.ndim != 1 or not len(values):
            raise ValueError("recovery needs one subject emission RMSE per subject")
        if not all(np.isfinite(value) and value >= 0 for value in scalars):
            raise ValueError("trajectory recovery RMSE values must be finite and non-negative")
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("subject emission RMSE values must be finite and non-negative")
        object.__setattr__(self, "subject_emission_rmse_by_subject", values)


@dataclass(frozen=True, slots=True)
class UnseenSubjectDynamicPrediction:
    """Monte Carlo integration over coherent unseen-subject paths and transitions."""

    subjects: tuple[Any, ...]
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
    includes_population_path_uncertainty: bool
    label_path_ambiguous: bool
    label_policy: str = "conditional-on-one-whole-path-canonical-mode"

    def __post_init__(self) -> None:
        subjects = tuple(_scalar(value) for value in self.subjects)
        row_subjects = protected_array(self.row_subject_indices, dtype=np.intp)
        probability = protected_array(self.probability, dtype=np.float64)
        draws = protected_array(self.draw_probabilities, dtype=np.float64)
        draw_log = protected_array(self.draw_pointwise_log_probability, dtype=np.float64)
        marginal = protected_array(self.pointwise_marginal_log_probability, dtype=np.float64)
        joint = protected_array(self.subject_joint_log_probability, dtype=np.float64)
        effective = protected_array(self.subject_effective_draws, dtype=np.float64)
        mcse = protected_array(self.subject_log_probability_mcse, dtype=np.float64)
        emission_draws = protected_array(self.draw_session_emission_coefficients, dtype=np.float64)
        transition_draws = protected_array(self.draw_session_transition_matrices, dtype=np.float64)
        if not subjects or len(set(subjects)) != len(subjects):
            raise ValueError("prediction subjects must be unique and non-empty")
        if probability.ndim != 1 or row_subjects.shape != probability.shape:
            raise ValueError("prediction rows and subject indices must align")
        if draws.shape != (self.n_draws, len(probability)) or draw_log.shape != draws.shape:
            raise ValueError("predictive draws must contain one value per draw and row")
        if marginal.shape != probability.shape:
            raise ValueError("pointwise marginal scores must align with prediction rows")
        if (
            joint.shape != (len(subjects),)
            or effective.shape != joint.shape
            or mcse.shape != joint.shape
        ):
            raise ValueError("subject-joint Monte Carlo diagnostics must align")
        if emission_draws.ndim != 4 or emission_draws.shape[0] != self.n_draws:
            raise ValueError("emission draws must cover draw, session, state, and coefficient")
        if transition_draws.shape[:2] != emission_draws.shape[:2] or transition_draws.ndim != 4:
            raise ValueError("transition draws must align with draw and session")
        if np.any((row_subjects < 0) | (row_subjects >= len(subjects))):
            raise ValueError("row subject indices must identify prediction subjects")
        for name, values in (
            ("probability", probability),
            ("draw probabilities", draws),
            ("draw log probability", draw_log),
            ("pointwise marginal log probability", marginal),
            ("subject joint log probability", joint),
            ("subject effective draws", effective),
            ("subject log probability MCSE", mcse),
            ("emission draws", emission_draws),
            ("transition draws", transition_draws),
        ):
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must be finite")
        if not np.all((probability > 0) & (probability < 1)):
            raise ValueError("marginal probabilities must lie strictly inside zero and one")
        if not np.all((effective >= 1.0) & (effective <= self.n_draws + 1e-8)):
            raise ValueError("effective draws must lie between one and n_draws")
        if np.any(mcse < 0):
            raise ValueError("Monte Carlo errors must be non-negative")
        if self.label_policy != "conditional-on-one-whole-path-canonical-mode":
            raise ValueError("label_policy must retain the fitted canonical path mode")
        object.__setattr__(self, "subjects", subjects)
        object.__setattr__(self, "row_subject_indices", row_subjects)
        object.__setattr__(self, "probability", probability)
        object.__setattr__(self, "draw_probabilities", draws)
        object.__setattr__(self, "draw_pointwise_log_probability", draw_log)
        object.__setattr__(self, "pointwise_marginal_log_probability", marginal)
        object.__setattr__(self, "subject_joint_log_probability", joint)
        object.__setattr__(self, "subject_effective_draws", effective)
        object.__setattr__(self, "subject_log_probability_mcse", mcse)
        object.__setattr__(self, "draw_session_emission_coefficients", emission_draws)
        object.__setattr__(self, "draw_session_transition_matrices", transition_draws)


@dataclass(frozen=True, slots=True)
class HierarchicalSessionDynamicGLMHMMFitResult(FitResult):
    """Population and subject paths with retained three-stage fitting evidence."""

    subjects: tuple[Any, ...]
    path_subjects: tuple[Any, ...]
    session_keys: tuple[Any, ...]
    session_orders: NDArray[np.int64]
    population_session_orders: NDArray[np.int64]
    population_emission_coefficients: NDArray[np.float64]
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
    minimum_subject_label_path_gap: float
    population_label_crossings: NDArray[np.bool_]
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
    subject_emission_scale: float
    emission_step_scale: float
    transition_concentration: float
    population_emission_standard_errors: NDArray[np.float64]
    population_emission_covariance: NDArray[np.float64]
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
    seen_future_session_policy: str = (
        "population-path-plus-carried-subject-deviation/use-global-transitions"
    )
    unseen_subject_policy: str = "population-path-plugin/use-global-transitions"

    def __post_init__(self) -> None:
        FitResult.__post_init__(self)
        n_states = (
            self.population_emission_coefficients.shape[1]
            if self.population_emission_coefficients.ndim == 3
            else 0
        )
        validated = _validated_population_trajectory(
            self.subjects,
            self.path_subjects,
            self.session_keys,
            self.session_orders,
            self.population_session_orders,
            self.population_emission_coefficients,
            self.session_emission_coefficients,
            self.session_transition_matrices,
            self.global_transition_matrix,
            n_states=n_states,
        )
        partial = protected_array(self.partial_objective_history, dtype=np.float64)
        full = protected_array(self.objective_history, dtype=np.float64)
        occupancy = protected_array(self.state_occupancy, dtype=np.float64)
        population_crossings = protected_array(self.population_label_crossings, dtype=np.bool_)
        subject_crossings = protected_array(self.subject_label_crossings, dtype=np.bool_)
        restart_objectives = protected_array(
            self.initialization_restart_objectives, dtype=np.float64
        )
        restart_converged = protected_array(self.initialization_restart_converged, dtype=np.bool_)
        permutation = tuple(self.canonical_permutation)
        crossing_subjects = tuple(_scalar(value) for value in self.subject_label_crossing_subjects)
        population_errors = protected_array(
            self.population_emission_standard_errors, dtype=np.float64
        )
        population_covariance = protected_array(
            self.population_emission_covariance, dtype=np.float64
        )
        session_errors = protected_array(self.session_emission_standard_errors, dtype=np.float64)
        session_covariance = protected_array(self.session_emission_covariance, dtype=np.float64)
        deviation_errors = protected_array(self.subject_deviation_standard_errors, dtype=np.float64)
        joint_covariance = protected_array(self.joint_emission_covariance, dtype=np.float64)
        transition_errors = protected_array(
            self.session_transition_standard_errors, dtype=np.float64
        )
        hyperparameter_names = tuple(self.hyperparameter_names)
        hyperparameter_estimates = protected_array(self.hyperparameter_estimates, dtype=np.float64)
        hyperparameter_errors = protected_array(
            self.hyperparameter_standard_errors, dtype=np.float64
        )
        hyperparameter_covariance = protected_array(
            self.hyperparameter_covariance, dtype=np.float64
        )
        hyperparameters_at_boundary = protected_array(
            self.hyperparameters_at_boundary, dtype=np.bool_
        )
        rate_matrix = protected_array(self.gaussian_scale_em_rate_matrix, dtype=np.float64)
        if sorted(permutation) != list(range(n_states)) or n_states < 2:
            raise ValueError("canonical_permutation must permute every latent state")
        for name, values in (
            ("partial_objective_history", partial),
            ("objective_history", full),
        ):
            if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain finite EM objectives")
        if occupancy.shape != (n_states,) or np.any(occupancy < 0):
            raise ValueError("state_occupancy must contain one non-negative value per state")
        if not np.isclose(occupancy.sum(), 1.0, atol=1e-8):
            raise ValueError("state_occupancy must sum to one")
        n_pairs = n_states * (n_states - 1) // 2
        if population_crossings.shape != (
            max(0, len(validated.population_orders) - 1),
            n_pairs,
        ):
            raise ValueError("population label crossings must cover adjacent population orders")
        expected_subject_adjacencies = sum(
            max(0, len(blocks) - 1)
            for blocks in _subject_blocks(validated.subjects, validated.path_subjects)
        )
        if subject_crossings.shape != (expected_subject_adjacencies, n_pairs):
            raise ValueError("subject label crossings must cover every within-subject adjacency")
        if len(crossing_subjects) != expected_subject_adjacencies:
            raise ValueError("subject crossing labels must align with subject path adjacencies")
        if not set(crossing_subjects).issubset(set(validated.subjects)):
            raise ValueError("subject crossing labels must name fitted subjects")
        for name, value in (
            ("minimum_state_separation", self.minimum_state_separation),
            ("minimum_population_label_path_gap", self.minimum_population_label_path_gap),
            ("minimum_subject_label_path_gap", self.minimum_subject_label_path_gap),
        ):
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        restart_messages = tuple(self.initialization_restart_messages)
        if (
            restart_objectives.ndim != 1
            or not len(restart_objectives)
            or restart_converged.shape != restart_objectives.shape
            or len(restart_messages) != len(restart_objectives)
        ):
            raise ValueError("initialization restart evidence must align")
        if not 0 <= self.initialization_selected_restart < len(restart_objectives):
            raise ValueError("initialization_selected_restart must identify one restart")
        if not self.partial_emission_optimizer_message or not self.emission_optimizer_message:
            raise ValueError("both emission optimizer messages must be retained")
        for name, value in (
            ("population_emission_step_scale", self.population_emission_step_scale),
            ("subject_emission_scale", self.subject_emission_scale),
            ("emission_step_scale", self.emission_step_scale),
            ("transition_concentration", self.transition_concentration),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not self.uncertainty_policy:
            raise ValueError("uncertainty_policy must be non-empty")
        if self.uncertainty_label_policy != "conditional-on-one-whole-path-canonical-mode":
            raise ValueError("uncertainty_label_policy must retain the canonical label mode")
        population_width = validated.population_emissions.size
        session_width = validated.emissions.size
        joint_width = population_width + session_width
        if population_errors.shape != validated.population_emissions.shape:
            raise ValueError("population path standard errors must align")
        if session_errors.shape != validated.emissions.shape:
            raise ValueError("subject-session path standard errors must align")
        if deviation_errors.shape != validated.emissions.shape:
            raise ValueError("subject-deviation standard errors must align")
        if population_covariance.shape != (population_width, population_width):
            raise ValueError("population covariance must cover the complete population path")
        if session_covariance.shape != (session_width, session_width):
            raise ValueError("session covariance must cover every realized subject path")
        if joint_covariance.shape != (joint_width, joint_width):
            raise ValueError("joint covariance must cover population and subject paths")
        if transition_errors.shape != validated.transitions.shape or not np.all(
            np.isfinite(transition_errors)
        ):
            raise ValueError("session transition standard errors must align and be finite")
        path_arrays = (
            population_errors,
            session_errors,
            deviation_errors,
            population_covariance,
            session_covariance,
            joint_covariance,
        )
        if self.path_covariance_positive_definite:
            if not all(np.all(np.isfinite(values)) for values in path_arrays):
                raise ValueError("a valid hierarchical path covariance must be finite")
        elif not all(np.all(np.isnan(values)) for values in path_arrays):
            raise ValueError("an invalid hierarchical path covariance must remain unavailable")
        if np.isnan(self.path_hessian_condition) or self.path_hessian_condition <= 0:
            raise ValueError("path_hessian_condition must be positive and not NaN")
        expected_hyperparameters = (
            "population_emission_step_scale",
            "subject_emission_scale",
            "emission_step_scale",
            "transition_concentration",
        )
        if hyperparameter_names != expected_hyperparameters:
            raise ValueError("hyperparameter_names must identify the complete hierarchy")
        n_hyperparameters = len(hyperparameter_names)
        if (
            hyperparameter_estimates.shape != (n_hyperparameters,)
            or hyperparameter_errors.shape != (n_hyperparameters,)
            or hyperparameter_covariance.shape != (n_hyperparameters, n_hyperparameters)
            or hyperparameters_at_boundary.shape != (n_hyperparameters,)
        ):
            raise ValueError("hierarchy hyperparameter uncertainty arrays must align")
        if not np.all(np.isfinite(hyperparameter_estimates)) or np.any(
            hyperparameter_estimates <= 0
        ):
            raise ValueError("hierarchy hyperparameter estimates must be finite and positive")
        if self.hyperparameters_estimated:
            finite_errors = np.isfinite(hyperparameter_errors)
            if np.any(hyperparameter_errors[finite_errors] < 0):
                raise ValueError("available hierarchy hyperparameter errors must be non-negative")
            if not np.array_equal(np.isfinite(np.diag(hyperparameter_covariance)), finite_errors):
                raise ValueError("hierarchy covariance diagonals must match reported errors")
        elif not np.all(np.isnan(hyperparameter_errors)) or not np.all(
            np.isnan(hyperparameter_covariance)
        ):
            raise ValueError("fixed hierarchy hyperparameters must not acquire uncertainty")
        if (
            isinstance(self.hyperparameter_estimation_iterations, bool)
            or not isinstance(self.hyperparameter_estimation_iterations, int)
            or self.hyperparameter_estimation_iterations < 0
        ):
            raise ValueError("hyperparameter_estimation_iterations must be non-negative")
        if rate_matrix.shape != (3, 3):
            raise ValueError("the hierarchy scale EM rate matrix must be three by three")
        if not self.hyperparameter_uncertainty_policy:
            raise ValueError("hyperparameter_uncertainty_policy must be non-empty")
        if self.seen_future_session_policy != (
            "population-path-plus-carried-subject-deviation/use-global-transitions"
        ):
            raise ValueError("seen_future_session_policy must name the declared forecast rule")
        if self.unseen_subject_policy != "population-path-plugin/use-global-transitions":
            raise ValueError("unseen_subject_policy must name the declared population plug-in")
        _set_validated_trajectory(self, validated)
        object.__setattr__(self, "partial_objective_history", partial)
        object.__setattr__(self, "objective_history", full)
        object.__setattr__(self, "canonical_permutation", permutation)
        object.__setattr__(self, "state_occupancy", occupancy)
        object.__setattr__(self, "population_label_crossings", population_crossings)
        object.__setattr__(self, "subject_label_crossings", subject_crossings)
        object.__setattr__(self, "subject_label_crossing_subjects", crossing_subjects)
        object.__setattr__(self, "initialization_restart_objectives", restart_objectives)
        object.__setattr__(self, "initialization_restart_converged", restart_converged)
        object.__setattr__(self, "initialization_restart_messages", restart_messages)
        object.__setattr__(self, "population_emission_standard_errors", population_errors)
        object.__setattr__(self, "population_emission_covariance", population_covariance)
        object.__setattr__(self, "session_emission_standard_errors", session_errors)
        object.__setattr__(self, "session_emission_covariance", session_covariance)
        object.__setattr__(self, "subject_deviation_standard_errors", deviation_errors)
        object.__setattr__(self, "joint_emission_covariance", joint_covariance)
        object.__setattr__(self, "session_transition_standard_errors", transition_errors)
        object.__setattr__(self, "hyperparameter_names", hyperparameter_names)
        object.__setattr__(self, "hyperparameter_estimates", hyperparameter_estimates)
        object.__setattr__(self, "hyperparameter_standard_errors", hyperparameter_errors)
        object.__setattr__(self, "hyperparameter_covariance", hyperparameter_covariance)
        object.__setattr__(self, "hyperparameters_at_boundary", hyperparameters_at_boundary)
        object.__setattr__(self, "gaussian_scale_em_rate_matrix", rate_matrix)

    @property
    def grouping(self) -> str:
        return "subject"

    @property
    def groups(self) -> tuple[Any, ...]:
        return self.subjects

    @property
    def state_separation(self) -> float:
        return self.minimum_state_separation

    @property
    def label_order_gap(self) -> float:
        return min(
            self.minimum_population_label_path_gap,
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
    def subject_deviations(self) -> NDArray[np.float64]:
        """Subject-session deviations from the fitted population path."""

        order_position = {
            int(order): position for position, order in enumerate(self.population_session_orders)
        }
        population = np.stack(
            [
                self.population_emission_coefficients[order_position[int(order)]]
                for order in self.session_orders
            ]
        )
        return protected_array(self.session_emission_coefficients - population, dtype=np.float64)

    def subject_was_fitted(self, subject: Any) -> bool:
        return _scalar(subject) in self.subjects

    def population_emission_intervals(self, *, level: float = 0.95) -> NDArray[np.float64]:
        """Return canonical-mode Gaussian intervals for the population path."""

        return _normal_path_intervals(
            self.population_emission_coefficients,
            self.population_emission_standard_errors,
            level=level,
            available=self.path_covariance_positive_definite,
        )

    def subject_deviation_intervals(self, *, level: float = 0.95) -> NDArray[np.float64]:
        """Return canonical-mode Gaussian intervals for subject deviations."""

        return _normal_path_intervals(
            self.subject_deviations,
            self.subject_deviation_standard_errors,
            level=level,
            available=self.path_covariance_positive_definite,
        )


@dataclass(frozen=True, slots=True)
class HierarchicalSessionDynamicBernoulliGLMHMM(SessionDynamicBernoulliGLMHMM):
    """A population emission path with temporally evolving subject deviations.

    For population order ``r`` and subject-session ``(m, s)`` at that order,

    ``M[r] ~ Normal(M[r-1], population_emission_step_scale)``

    ``D[m, 0] ~ Normal(0, subject_emission_scale)``

    ``D[m, s] ~ Normal(D[m, s-1], emission_step_scale)``

    and ``W[m, s] = M[r] + D[m, s]``.  Session transition rows retain the published
    ``Dirichlet(transition_concentration * global_transition + 1)`` distribution directly
    across every subject-session block.
    """

    population_emission_step_scale: float = 0.25
    subject_emission_scale: float = 0.5

    def __post_init__(self) -> None:
        SessionDynamicBernoulliGLMHMM.__post_init__(self)
        for name, value in (
            ("population_emission_step_scale", self.population_emission_step_scale),
            ("subject_emission_scale", self.subject_emission_scale),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")

    @property
    def model_name(self) -> str:
        return "hierarchical-session-dynamic-bernoulli-glm-hmm"

    @property
    def is_population_dynamic(self) -> bool:
        return True

    @property
    def signature(self) -> str:
        predictors = ",".join(self.predictors)
        return (
            f"{self.model_name}[states={self.n_states};outcome={self.outcome};"
            f"predictors={predictors};choice_lags={self.choice_lags};label_by={self.label_by};"
            f"l2={self.l2};population_emission_step_scale="
            f"{self.population_emission_step_scale};subject_emission_scale="
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
            "first observed subject deviation from population path: "
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
                    "training-only bounded Laplace-EM estimation of all three Gaussian "
                    "hierarchy scales",
                    "training-only conditional Dirichlet-multinomial estimation of "
                    "transition concentration",
                )
            )
        return tuple(declared)

    @property
    def bounded_coordinate_refusal(self) -> str:
        return (
            "a hierarchical session-dynamic GLM-HMM already contains a population path and "
            "subject deviation paths with data-dependent dimensions; another generic "
            "hierarchical or smooth wrapper would not declare the required population model"
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
    ) -> HierarchicalSessionDynamicGLMHMMSimulation:
        return self.simulate_with_trajectories(design, parameters, seed=seed)

    def simulate_with_trajectories(
        self,
        design: Study,
        parameters: Mapping[str, float],
        *,
        seed: int | np.random.Generator,
    ) -> HierarchicalSessionDynamicGLMHMMSimulation:
        """Draw a population path, subject deviations, session matrices, and choices."""

        structure = _population_structure(design, require_multiple_subjects=True)
        base = self.parameter_components(parameters)
        generator = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        n_population = len(structure.population_orders)
        shape = (self.n_states, len(self.coefficient_names))
        population = np.empty((n_population, *shape), dtype=np.float64)
        population[0] = base.emission_coefficients
        for position in range(1, n_population):
            population[position] = population[position - 1] + generator.normal(
                0.0,
                self.population_emission_step_scale,
                shape,
            )
        emissions = np.empty((len(structure.sessions), *shape), dtype=np.float64)
        for blocks in structure.subject_blocks:
            deviation = generator.normal(0.0, self.subject_emission_scale, shape)
            for within_subject, block in enumerate(blocks):
                if within_subject:
                    deviation = deviation + generator.normal(
                        0.0,
                        self.emission_step_scale,
                        shape,
                    )
                emissions[block] = population[structure.population_index[block]] + deviation
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
            design,
            lambda indices: by_opening_row[int(indices[0])],
            generator,
        )
        columns = {name: design[name] for name in design.columns}
        columns[self.outcome] = choices
        return HierarchicalSessionDynamicGLMHMMSimulation(
            study=Study(columns),
            states=states,
            n_states=self.n_states,
            subjects=structure.subjects,
            path_subjects=structure.path_subjects,
            session_keys=structure.keys,
            session_orders=structure.orders,
            population_session_orders=structure.population_orders,
            population_emission_coefficients=population,
            session_emission_coefficients=emissions,
            session_transition_matrices=transitions,
            global_transition_matrix=base.transition_matrix,
        )

    def fit(self, study: Study) -> HierarchicalSessionDynamicGLMHMMFitResult:
        """Fit fixed hierarchy scales or estimate them from the training study."""

        if self.estimate_hyperparameters:
            return self._fit_estimated_hyperparameters(study)
        return self._fit_fixed_hyperparameters(study)

    def _fit_fixed_hyperparameters(self, study: Study) -> HierarchicalSessionDynamicGLMHMMFitResult:
        """Fit population and subject paths with stationary, partial, and full MAP EM."""

        structure = _population_structure(study, require_multiple_subjects=True)
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
        emissions = np.tile(
            np.asarray(base.emission_coefficients),
            (len(structure.sessions), 1, 1),
        ).copy()
        transitions = np.tile(
            global_transition,
            (len(structure.sessions), 1, 1),
        ).copy()

        partial_history: list[float] = []
        partial_message = "partial population/subject emission M-step was not run"
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
            transitions = np.tile(
                global_transition,
                (len(structure.sessions), 1, 1),
            ).copy()
            result = self._optimize_population_emissions(
                population,
                emissions,
                features,
                outcomes,
                posterior.state_probabilities,
                structure,
            )
            population, emissions = self._unpack_emission_coordinate(result.x, structure)
            partial_emission_converged = bool(result.success)
            partial_message = str(result.message)
            partial_history.append(
                self._population_map_objective(
                    features,
                    outcomes,
                    initial,
                    population,
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
        emission_message = "full population/subject emission M-step was not run"
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
            result = self._optimize_population_emissions(
                population,
                emissions,
                features,
                outcomes,
                posterior.state_probabilities,
                structure,
            )
            population, emissions = self._unpack_emission_coordinate(result.x, structure)
            emission_converged = bool(result.success)
            emission_message = str(result.message)
            objective_history.append(
                self._population_map_objective(
                    features,
                    outcomes,
                    initial,
                    population,
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

        population, emissions, transitions, initial, global_transition, permutation = (
            self._canonicalize_population_trajectory(
                population,
                emissions,
                transitions,
                initial,
                global_transition,
            )
        )
        posterior = self._dynamic_posterior(
            features,
            outcomes,
            structure.sessions,
            initial,
            emissions,
            transitions,
        )
        path_uncertainty = self._observed_population_path_covariance(
            features,
            outcomes,
            initial,
            population,
            emissions,
            transitions,
            structure,
        )
        population_width = population.size
        if path_uncertainty.positive_definite:
            population_covariance = path_uncertainty.covariance[
                :population_width, :population_width
            ]
            session_covariance = path_uncertainty.covariance[population_width:, population_width:]
            deviation_map = _subject_deviation_contrast(
                structure,
                population.shape,
                emissions.shape,
            )
            deviation_variance = np.einsum(
                "ij,jk,ik->i",
                deviation_map,
                path_uncertainty.covariance,
                deviation_map,
                optimize=True,
            )
            deviation_errors = np.sqrt(np.maximum(deviation_variance, 0.0)).reshape(emissions.shape)
        else:
            population_covariance = np.full((population_width, population_width), np.nan)
            session_covariance = np.full((emissions.size, emissions.size), np.nan)
            deviation_errors = np.full(emissions.shape, np.nan)
        transition_counts = session_transition_counts(
            posterior.transition_expectations,
            structure.sessions,
        )
        transition_errors = transition_standard_errors(
            transition_counts,
            self.transition_concentration,
            global_transition,
        )
        occupancy = np.mean(posterior.state_probabilities, axis=0)
        population_crossings, population_gap = self._label_path_diagnostics(population)
        subject_crossings, crossing_subjects, subject_gap = self._subject_label_diagnostics(
            emissions,
            structure,
        )
        minimum_separation = min(
            _minimum_pairwise_distance(values) for values in (*tuple(population), *tuple(emissions))
        )
        label_ambiguous = bool(
            np.any(population_crossings)
            or np.any(subject_crossings)
            or min(population_gap, subject_gap) <= self.label_tolerance
        )
        final_components = GLMHMMParameters(
            initial_probabilities=initial,
            transition_matrix=global_transition,
            emission_coefficients=population[0],
            coefficient_names=self.coefficient_names,
        )
        estimates = self._pack_components(final_components)
        _, final_gradient = self._population_emission_m_step_objective(
            self._pack_emission_coordinate(population, emissions),
            features,
            outcomes,
            posterior.state_probabilities,
            structure,
        )
        probability_values = np.concatenate((initial, transitions.ravel()))
        boundary = bool(
            np.any(np.abs(population) >= self.coefficient_warning_threshold)
            or np.any(np.abs(emissions) >= self.coefficient_warning_threshold)
            or np.any(probability_values <= self.probability_warning_threshold)
            or np.any(probability_values >= 1.0 - self.probability_warning_threshold)
        )
        converged = partial_converged and full_converged
        diagnostics = FitDiagnostics(
            converged=converged,
            optimizer=(
                "hierarchical three-stage MAP EM (pooled stationary multistart; "
                "population/subject emission paths with static transitions; fully dynamic; "
                "joint L-BFGS-B emission M-steps)"
            ),
            status=0 if converged else 1,
            message=(
                "hierarchical three-stage EM met the declared tolerance; "
                if converged
                else "hierarchical three-stage EM did not meet the declared tolerance; "
            )
            + f"partial stage {'converged' if partial_converged else 'did not converge'}; "
            + f"full stage {'converged' if full_converged else 'did not converge'}; "
            + emission_message,
            n_iterations=len(partial_history) + len(objective_history),
            objective=float(objective_history[-1]),
            gradient_norm=float(np.linalg.norm(final_gradient)),
            hessian_condition=path_uncertainty.hessian_condition,
            boundary_estimate=boundary,
        )
        n_parameters = len(estimates)
        return HierarchicalSessionDynamicGLMHMMFitResult(
            model_name=self.model_name,
            model_signature=self.signature,
            parameter_names=self.parameter_names,
            estimates=estimates,
            standard_errors=np.full(n_parameters, np.nan),
            covariance=np.full((n_parameters, n_parameters), np.nan),
            n_observations=len(study),
            diagnostics=diagnostics,
            subjects=structure.subjects,
            path_subjects=structure.path_subjects,
            session_keys=structure.keys,
            session_orders=structure.orders,
            population_session_orders=structure.population_orders,
            population_emission_coefficients=population,
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
            minimum_subject_label_path_gap=subject_gap,
            population_label_crossings=population_crossings,
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
            subject_emission_scale=self.subject_emission_scale,
            emission_step_scale=self.emission_step_scale,
            transition_concentration=self.transition_concentration,
            population_emission_standard_errors=path_uncertainty.standard_errors[
                :population_width
            ].reshape(population.shape),
            population_emission_covariance=population_covariance,
            session_emission_standard_errors=path_uncertainty.standard_errors[
                population_width:
            ].reshape(emissions.shape),
            session_emission_covariance=session_covariance,
            subject_deviation_standard_errors=deviation_errors,
            joint_emission_covariance=path_uncertainty.covariance,
            session_transition_standard_errors=transition_errors,
            path_hessian_condition=path_uncertainty.hessian_condition,
            path_covariance_positive_definite=path_uncertainty.positive_definite,
            hyperparameter_names=(
                "population_emission_step_scale",
                "subject_emission_scale",
                "emission_step_scale",
                "transition_concentration",
            ),
            hyperparameter_estimates=np.asarray(
                [
                    self.population_emission_step_scale,
                    self.subject_emission_scale,
                    self.emission_step_scale,
                    self.transition_concentration,
                ]
            ),
            hyperparameter_standard_errors=np.full(4, np.nan),
            hyperparameter_covariance=np.full((4, 4), np.nan),
            hyperparameters_estimated=False,
            hyperparameter_estimation_converged=False,
            hyperparameter_estimation_iterations=0,
            hyperparameters_at_boundary=np.zeros(4, dtype=np.bool_),
            gaussian_scale_em_rate_matrix=np.full((3, 3), np.nan),
            gaussian_scale_em_spectral_radius=float("nan"),
            hyperparameter_uncertainty_policy="fixed-hyperparameters",
            uncertainty_policy=(
                "observed-laplace-conditional-on-canonical-path"
                if path_uncertainty.positive_definite
                else "unavailable-nonpositive-observed-path-information"
            ),
        )

    def _fit_estimated_hyperparameters(
        self, study: Study
    ) -> HierarchicalSessionDynamicGLMHMMFitResult:
        """Estimate all Gaussian hierarchy scales and the transition concentration."""

        structure = _population_structure(study, require_multiple_subjects=True)
        if len(structure.population_orders) < 2:
            raise ModelDataError(
                "estimating the population random-walk scale requires two population orders"
            )
        if not any(len(blocks) > 1 for blocks in structure.subject_blocks):
            raise ModelDataError(
                "estimating the subject random-walk scale requires a repeated-session subject"
            )
        scales = np.clip(
            np.asarray(
                [
                    self.population_emission_step_scale,
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
            fixed = replace(
                self,
                estimate_hyperparameters=False,
                population_emission_step_scale=float(scales[0]),
                subject_emission_scale=float(scales[1]),
                emission_step_scale=float(scales[2]),
                transition_concentration=concentration,
            )
            fitted = fixed._fit_fixed_hyperparameters(study)
            if not fitted.path_covariance_positive_definite:
                raise ModelDataError(
                    "hierarchy hyperparameter estimation requires a positive-definite "
                    "observed path covariance"
                )
            coordinate = fixed._pack_emission_coordinate(
                fitted.population_emission_coefficients,
                fitted.session_emission_coefficients,
            )
            contrasts = _hierarchy_scale_contrasts(
                structure,
                fitted.population_emission_coefficients.shape,
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
                posterior.transition_expectations,
                structure.sessions,
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

        final_model = replace(
            self,
            estimate_hyperparameters=False,
            population_emission_step_scale=float(scales[0]),
            subject_emission_scale=float(scales[1]),
            emission_step_scale=float(scales[2]),
            transition_concentration=concentration,
        )
        fitted = final_model._fit_fixed_hyperparameters(study)
        if not fitted.path_covariance_positive_definite:
            raise ModelDataError(
                "estimated hierarchy requires a positive-definite final path covariance"
            )
        coordinate = final_model._pack_emission_coordinate(
            fitted.population_emission_coefficients,
            fitted.session_emission_coefficients,
        )
        contrasts = _hierarchy_scale_contrasts(
            structure,
            fitted.population_emission_coefficients.shape,
            fitted.session_emission_coefficients.shape,
        )
        scale_covariance, scale_errors, scale_uncertainty_valid = (
            gaussian_scale_observed_covariance(
                scales,
                coordinate,
                fitted.joint_emission_covariance,
                contrasts,
            )
        )
        rate_matrix = np.full((3, 3), np.nan, dtype=np.float64)
        spectral_radius = float("nan")
        scale_policy = "louis-observed-information"
        if not scale_uncertainty_valid:

            def scale_update(candidate: NDArray[np.float64]) -> NDArray[np.float64]:
                probe = replace(
                    self,
                    estimate_hyperparameters=False,
                    population_emission_step_scale=float(candidate[0]),
                    subject_emission_scale=float(candidate[1]),
                    emission_step_scale=float(candidate[2]),
                    transition_concentration=concentration,
                )
                probe_fit = probe._fit_fixed_hyperparameters(study)
                if not probe_fit.path_covariance_positive_definite:
                    return np.full(3, np.nan)
                probe_coordinate = probe._pack_emission_coordinate(
                    probe_fit.population_emission_coefficients,
                    probe_fit.session_emission_coefficients,
                )
                probe_contrasts = _hierarchy_scale_contrasts(
                    structure,
                    probe_fit.population_emission_coefficients.shape,
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
                rate_matrix,
                spectral_radius,
                scale_uncertainty_valid,
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
        counts = session_transition_counts(
            posterior.transition_expectations,
            structure.sessions,
        )
        _, concentration_error, concentration_uncertainty_valid = estimate_transition_concentration(
            counts,
            fitted.global_transition_matrix,
            self.transition_concentration_bounds,
        )
        hyper_covariance = np.full((4, 4), np.nan, dtype=np.float64)
        hyper_errors = np.full(4, np.nan, dtype=np.float64)
        if scale_uncertainty_valid:
            hyper_covariance[:3, :3] = scale_covariance
            hyper_errors[:3] = scale_errors
        if concentration_uncertainty_valid and np.isfinite(concentration_error):
            hyper_covariance[3, 3] = concentration_error**2
            hyper_errors[3] = concentration_error
        if np.all(np.isfinite(hyper_errors)):
            hyper_covariance[:3, 3] = hyper_covariance[3, :3] = 0.0
        estimates = np.concatenate((scales, [concentration]))
        boundary_tolerance = max(self.hyperparameter_tolerance, 1e-4)
        reported_scale_policy = (
            scale_policy if scale_uncertainty_valid else "scale-unavailable-unstable-information"
        )
        at_boundary = np.asarray(
            [
                *(
                    at_log_bound(
                        float(scale),
                        self.gaussian_scale_bounds,
                        boundary_tolerance,
                    )
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
        hyper_optimizer = f"{fitted.diagnostics.optimizer}; outer hierarchy "
        hyper_optimizer += "Laplace-EM/Dirichlet evidence"
        diagnostics = replace(
            fitted.diagnostics,
            converged=fixed_converged and converged,
            optimizer=hyper_optimizer,
            status=0 if fixed_converged and converged else 1,
            message=(
                f"{fitted.diagnostics.message}; hierarchy hyperparameter estimation "
                f"{'converged' if converged else 'did not converge'} after {iterations} iterations"
            ),
            n_iterations=(fitted.diagnostics.n_iterations or 0) + iterations,
            boundary_estimate=bool(fitted.diagnostics.boundary_estimate or np.any(at_boundary)),
        )
        return replace(
            fitted,
            model_signature=self.signature,
            diagnostics=diagnostics,
            population_emission_step_scale=float(scales[0]),
            subject_emission_scale=float(scales[1]),
            emission_step_scale=float(scales[2]),
            transition_concentration=concentration,
            hyperparameter_estimates=estimates,
            hyperparameter_standard_errors=hyper_errors,
            hyperparameter_covariance=hyper_covariance,
            hyperparameters_estimated=True,
            hyperparameter_estimation_converged=converged,
            hyperparameter_estimation_iterations=iterations,
            hyperparameters_at_boundary=at_boundary,
            gaussian_scale_em_rate_matrix=rate_matrix,
            gaussian_scale_em_spectral_radius=spectral_radius,
            hyperparameter_uncertainty_policy=(
                f"{reported_scale_policy};"
                "conditional-dirichlet-multinomial;block-diagonal-cross-family-covariance"
            ),
        )

    def _observed_population_path_covariance(
        self,
        features: NDArray[np.float64],
        outcomes: NDArray[np.float64],
        initial: NDArray[np.float64],
        population: NDArray[np.float64],
        emissions: NDArray[np.float64],
        transitions: NDArray[np.float64],
        structure: _PopulationStructure,
    ) -> Any:
        """Differentiate the state-marginalized population/subject path objective."""

        mode = self._pack_emission_coordinate(population, emissions)

        def gradient(vector: NDArray[np.float64]) -> NDArray[np.float64]:
            _, candidate_emissions = self._unpack_emission_coordinate(vector, structure)
            posterior = self._dynamic_posterior(
                features,
                outcomes,
                structure.sessions,
                initial,
                candidate_emissions,
                transitions,
            )
            _, values = self._population_emission_m_step_objective(
                vector,
                features,
                outcomes,
                posterior.state_probabilities,
                structure,
            )
            return values

        return observed_path_covariance(gradient, mode)

    def transition_probabilities(
        self,
        study: Study,
        parameters: Mapping[str, float] | FitResult,
    ) -> NDArray[np.float64]:
        if not isinstance(parameters, FitResult):
            return BernoulliGLMHMM.transition_probabilities(self, study, parameters)
        fit = self._validate_population_fit(parameters)
        components = self._prediction_components(study, fit)
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
        population_fit = self._validate_population_fit(fit)
        outcomes = self.outcomes(study)
        features = self.design_matrix(study)
        components = self._prediction_components(study, population_fit)
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

    def predict_new_subjects(
        self,
        study: Study,
        fit: FitResult,
        *,
        n_draws: int,
        seed: int,
        include_population_path_uncertainty: bool = True,
    ) -> UnseenSubjectDynamicPrediction:
        """Integrate coherent random paths for subjects absent from the training fit.

        One Monte Carlo draw contains one population path shared by all prediction subjects,
        one evolving deviation path per subject, and one Dirichlet session-transition matrix
        per subject-session block.  Pointwise marginal scores are retained for diagnostics,
        but the coherent predictive likelihood is the subject-joint score.
        """

        population_fit = self._validate_population_fit(fit)
        if isinstance(n_draws, bool) or not isinstance(n_draws, int) or n_draws < 2:
            raise ValueError("n_draws must be an integer of at least two")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not isinstance(include_population_path_uncertainty, bool):
            raise ValueError("include_population_path_uncertainty must be boolean")
        if (
            include_population_path_uncertainty
            and not population_fit.path_covariance_positive_definite
        ):
            raise ModelDataError(
                "population path integration requires a positive-definite fitted covariance"
            )
        structure = _population_structure(study, require_multiple_subjects=False)
        overlap = set(structure.subjects) & set(population_fit.subjects)
        if overlap:
            raise ValueError(
                "predict_new_subjects requires entirely unseen subjects; fitted labels include "
                f"{sorted(overlap, key=repr)!r}"
            )
        fitted_positions = {
            int(order): position
            for position, order in enumerate(population_fit.population_session_orders)
        }
        last_fitted_order = int(population_fit.population_session_orders[-1])
        requested_orders = tuple(sorted(set(int(value) for value in structure.orders)))
        invalid = [
            order
            for order in requested_orders
            if order not in fitted_positions and order <= last_fitted_order
        ]
        if invalid:
            raise ModelDataError(
                "unseen-subject integration cannot invent an earlier or interleaved population "
                f"path; missing fitted orders are {invalid!r}"
            )

        generator = np.random.default_rng(seed)
        features = self.design_matrix(study)
        outcomes = self.outcomes(study)
        base = self.parameter_components(population_fit)
        n_blocks = len(structure.sessions)
        shape = (self.n_states, len(self.coefficient_names))
        probabilities = np.empty((n_draws, len(study)), dtype=np.float64)
        log_probabilities = np.empty_like(probabilities)
        emission_draws = np.empty((n_draws, n_blocks, *shape), dtype=np.float64)
        transition_draws = np.empty(
            (n_draws, n_blocks, self.n_states, self.n_states), dtype=np.float64
        )
        population_mean = population_fit.population_emission_coefficients.ravel()
        population_covariance = population_fit.population_emission_covariance

        for draw in range(n_draws):
            if include_population_path_uncertainty:
                fitted_population = generator.multivariate_normal(
                    population_mean,
                    population_covariance,
                    check_valid="raise",
                ).reshape(population_fit.population_emission_coefficients.shape)
            else:
                fitted_population = np.asarray(
                    population_fit.population_emission_coefficients,
                    dtype=np.float64,
                )
            by_order = {
                order: fitted_population[position] for order, position in fitted_positions.items()
            }
            previous = fitted_population[-1]
            for order in requested_orders:
                if order > last_fitted_order:
                    previous = previous + generator.normal(
                        0.0,
                        population_fit.population_emission_step_scale,
                        shape,
                    )
                    by_order[order] = previous

            for blocks in structure.subject_blocks:
                deviation = generator.normal(
                    0.0,
                    population_fit.subject_emission_scale,
                    shape,
                )
                for within_subject, block in enumerate(blocks):
                    if within_subject:
                        deviation = deviation + generator.normal(
                            0.0,
                            population_fit.emission_step_scale,
                            shape,
                        )
                    emission_draws[draw, block] = by_order[int(structure.orders[block])] + deviation
            for block in range(n_blocks):
                for state in range(self.n_states):
                    transition_draws[draw, block, state] = generator.dirichlet(
                        population_fit.transition_concentration
                        * population_fit.global_transition_matrix[state]
                        + 1.0
                    )
            components = tuple(
                GLMHMMParameters(
                    initial_probabilities=base.initial_probabilities,
                    transition_matrix=transition_draws[draw, block],
                    emission_coefficients=emission_draws[draw, block],
                    coefficient_names=self.coefficient_names,
                )
                for block in range(n_blocks)
            )
            filtered = self._dynamic_filtered(features, outcomes, study, components)
            emission_probability = np.empty((len(study), self.n_states), dtype=np.float64)
            for block, indices in enumerate(structure.sessions):
                index = np.asarray(indices, dtype=np.intp)
                emission_probability[index] = expit(features[index] @ emission_draws[draw, block].T)
            probability = np.sum(filtered.predictive * emission_probability, axis=1)
            probability = np.clip(
                probability,
                np.finfo(float).tiny,
                1.0 - np.finfo(float).eps,
            )
            probabilities[draw] = probability
            log_probabilities[draw] = outcomes * np.log(probability) + (1.0 - outcomes) * np.log1p(
                -probability
            )

        subject_position = {subject: index for index, subject in enumerate(structure.subjects)}
        row_subjects = np.asarray(
            [subject_position[_scalar(value)] for value in study["subject"]],
            dtype=np.intp,
        )
        log_draws = np.log(float(n_draws))
        joint = np.empty(len(structure.subjects), dtype=np.float64)
        effective = np.empty_like(joint)
        mcse = np.empty_like(joint)
        for subject in range(len(structure.subjects)):
            weights = np.sum(log_probabilities[:, row_subjects == subject], axis=1)
            log_weight_sum = float(logsumexp(weights))
            joint[subject] = log_weight_sum - log_draws
            normalized = np.exp(weights - log_weight_sum)
            effective[subject] = 1.0 / float(np.sum(normalized**2))
            shifted = np.exp(weights - float(np.max(weights)))
            mcse[subject] = float(np.std(shifted, ddof=1) / np.sqrt(n_draws) / np.mean(shifted))
        return UnseenSubjectDynamicPrediction(
            subjects=structure.subjects,
            row_subject_indices=row_subjects,
            probability=np.mean(probabilities, axis=0),
            draw_probabilities=probabilities,
            draw_pointwise_log_probability=log_probabilities,
            pointwise_marginal_log_probability=(logsumexp(log_probabilities, axis=0) - log_draws),
            subject_joint_log_probability=joint,
            subject_effective_draws=effective,
            subject_log_probability_mcse=mcse,
            draw_session_emission_coefficients=emission_draws,
            draw_session_transition_matrices=transition_draws,
            n_draws=n_draws,
            seed=seed,
            includes_population_path_uncertainty=include_population_path_uncertainty,
            label_path_ambiguous=population_fit.label_path_ambiguous,
        )

    def state_probabilities(
        self,
        study: Study,
        fit: FitResult,
    ) -> FilteredStateProbabilities:
        population_fit = self._validate_population_fit(fit)
        return self._dynamic_filtered(
            self.design_matrix(study),
            self.outcomes(study),
            study,
            self._prediction_components(study, population_fit),
        )

    def state_recovery(
        self,
        simulation: HierarchicalSessionDynamicGLMHMMSimulation,
        fit: FitResult,
        *,
        ambiguity_tolerance: float = 0.05,
    ) -> LatentStateAlignment:
        if not isinstance(simulation, HierarchicalSessionDynamicGLMHMMSimulation):
            raise TypeError("simulation must be a HierarchicalSessionDynamicGLMHMMSimulation")
        if simulation.n_states != self.n_states:
            raise ValueError("simulation and model must contain the same number of states")
        return align_latent_states(
            simulation.states,
            self.state_probabilities(simulation.study, fit).filtered,
            ambiguity_tolerance=ambiguity_tolerance,
        )

    def trajectory_recovery(
        self,
        simulation: HierarchicalSessionDynamicGLMHMMSimulation,
        fit: FitResult,
        *,
        ambiguity_tolerance: float = 0.05,
    ) -> HierarchicalSessionDynamicTrajectoryRecovery:
        population_fit = self._validate_population_fit(fit)
        if not isinstance(simulation, HierarchicalSessionDynamicGLMHMMSimulation):
            raise TypeError("simulation must be a HierarchicalSessionDynamicGLMHMMSimulation")
        if (
            simulation.subjects != population_fit.subjects
            or simulation.path_subjects != population_fit.path_subjects
            or simulation.session_keys != population_fit.session_keys
            or not np.array_equal(simulation.session_orders, population_fit.session_orders)
            or not np.array_equal(
                simulation.population_session_orders,
                population_fit.population_session_orders,
            )
        ):
            raise ValueError("simulation and fit must describe the same population paths")
        alignment = self.state_recovery(
            simulation,
            population_fit,
            ambiguity_tolerance=ambiguity_tolerance,
        )
        mapping = np.asarray(alignment.reference_to_inferred, dtype=np.intp)
        population_error = (
            population_fit.population_emission_coefficients[:, mapping, :]
            - simulation.population_emission_coefficients
        )
        subject_error = (
            population_fit.session_emission_coefficients[:, mapping, :]
            - simulation.session_emission_coefficients
        )
        transition_error = (
            population_fit.session_transition_matrices[:, mapping][:, :, mapping]
            - simulation.session_transition_matrices
        )
        by_subject = []
        for subject in population_fit.subjects:
            blocks = np.asarray(
                [
                    block
                    for block, path_subject in enumerate(population_fit.path_subjects)
                    if path_subject == subject
                ],
                dtype=np.intp,
            )
            by_subject.append(float(np.sqrt(np.mean(subject_error[blocks] ** 2))))
        return HierarchicalSessionDynamicTrajectoryRecovery(
            alignment=alignment,
            population_emission_rmse=float(np.sqrt(np.mean(population_error**2))),
            subject_emission_rmse=float(np.sqrt(np.mean(subject_error**2))),
            transition_rmse=float(np.sqrt(np.mean(transition_error**2))),
            subject_emission_rmse_by_subject=np.asarray(by_subject),
        )

    def _validate_population_fit(self, fit: FitResult) -> HierarchicalSessionDynamicGLMHMMFitResult:
        self._validate_fit(fit)
        if not isinstance(fit, HierarchicalSessionDynamicGLMHMMFitResult):
            raise TypeError(
                "hierarchical session-dynamic prediction requires a "
                "HierarchicalSessionDynamicGLMHMMFitResult"
            )
        return fit

    def _prediction_components(
        self,
        study: Study,
        fit: HierarchicalSessionDynamicGLMHMMFitResult,
    ) -> tuple[GLMHMMParameters, ...]:
        structure = _population_structure(study, require_multiple_subjects=False)
        fitted_blocks = {
            (subject, key): (int(order), block)
            for block, (subject, key, order) in enumerate(
                zip(
                    fit.path_subjects,
                    fit.session_keys,
                    fit.session_orders,
                    strict=True,
                )
            )
        }
        population_positions = {
            int(order): position for position, order in enumerate(fit.population_session_orders)
        }
        subject_blocks = _subject_blocks(fit.subjects, fit.path_subjects)
        last_by_subject = {
            subject: blocks[-1]
            for subject, blocks in zip(fit.subjects, subject_blocks, strict=True)
        }
        base = self.parameter_components(fit)
        components: list[GLMHMMParameters] = []
        for subject, key, order_value in zip(
            structure.path_subjects,
            structure.keys,
            structure.orders,
            strict=True,
        ):
            order = int(order_value)
            identity = (subject, key)
            if identity in fitted_blocks:
                fitted_order, block = fitted_blocks[identity]
                if order != fitted_order:
                    raise ModelDataError(
                        f"subject {subject!r} session {key!r} was fitted at order "
                        f"{fitted_order}, not {order}"
                    )
                emissions = fit.session_emission_coefficients[block]
                transitions = fit.session_transition_matrices[block]
            elif subject in last_by_subject:
                last_block = last_by_subject[subject]
                last_order = int(fit.session_orders[last_block])
                if order <= last_order:
                    raise ModelDataError(
                        f"subject {subject!r} session {key!r} at order {order} was not fitted "
                        f"and is not later than that subject's last training session "
                        f"({last_order})"
                    )
                population = self._population_emissions_for_order(
                    order,
                    fit,
                    population_positions,
                )
                last_population = fit.population_emission_coefficients[
                    population_positions[last_order]
                ]
                deviation = fit.session_emission_coefficients[last_block] - last_population
                emissions = population + deviation
                transitions = fit.global_transition_matrix
            else:
                emissions = self._population_emissions_for_order(
                    order,
                    fit,
                    population_positions,
                )
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

    def _population_emissions_for_order(
        self,
        order: int,
        fit: HierarchicalSessionDynamicGLMHMMFitResult,
        positions: Mapping[int, int],
    ) -> NDArray[np.float64]:
        if order in positions:
            return fit.population_emission_coefficients[positions[order]]
        last_order = int(fit.population_session_orders[-1])
        if order > last_order:
            return fit.population_emission_coefficients[-1]
        raise ModelDataError(
            f"population session order {order} was not fitted and is not prospectively "
            f"later than the last population order ({last_order})"
        )

    def _pack_emission_coordinate(
        self,
        population: NDArray[np.float64],
        emissions: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return np.concatenate((population.ravel(), emissions.ravel()))

    def _unpack_emission_coordinate(
        self,
        vector: NDArray[np.float64],
        structure: _PopulationStructure,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        width = self.n_states * len(self.coefficient_names)
        population_size = len(structure.population_orders) * width
        values = np.asarray(vector, dtype=np.float64)
        population = values[:population_size].reshape(
            len(structure.population_orders),
            self.n_states,
            len(self.coefficient_names),
        )
        emissions = values[population_size:].reshape(
            len(structure.sessions),
            self.n_states,
            len(self.coefficient_names),
        )
        return population, emissions

    def _optimize_population_emissions(
        self,
        population: NDArray[np.float64],
        emissions: NDArray[np.float64],
        features: NDArray[np.float64],
        outcomes: NDArray[np.float64],
        state_probabilities: NDArray[np.float64],
        structure: _PopulationStructure,
    ) -> Any:
        initial = self._pack_emission_coordinate(population, emissions)
        return minimize(
            lambda vector: self._population_emission_m_step_objective(
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

    def _population_emission_m_step_objective(
        self,
        vector: NDArray[np.float64],
        features: NDArray[np.float64],
        outcomes: NDArray[np.float64],
        state_probabilities: NDArray[np.float64],
        structure: _PopulationStructure,
    ) -> tuple[float, NDArray[np.float64]]:
        population, emissions = self._unpack_emission_coordinate(vector, structure)
        population_gradient = np.zeros_like(population)
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
        prior_loss = self._population_emission_prior(
            population,
            emissions,
            population_gradient,
            emission_gradient,
            structure,
        )
        return (
            float(loss + prior_loss),
            self._pack_emission_coordinate(population_gradient, emission_gradient),
        )

    def _population_emission_prior(
        self,
        population: NDArray[np.float64],
        emissions: NDArray[np.float64],
        population_gradient: NDArray[np.float64] | None,
        emission_gradient: NDArray[np.float64] | None,
        structure: _PopulationStructure,
    ) -> float:
        loss = 0.0
        population_differences = np.diff(population, axis=0)
        population_precision = 1.0 / self.population_emission_step_scale**2
        loss += 0.5 * population_precision * float(np.sum(population_differences**2))
        if population_gradient is not None and len(population) > 1:
            population_gradient[:-1] -= population_precision * population_differences
            population_gradient[1:] += population_precision * population_differences

        deviations = emissions - population[structure.population_index]
        initial_precision = 1.0 / self.subject_emission_scale**2
        step_precision = 1.0 / self.emission_step_scale**2
        for blocks in structure.subject_blocks:
            first = blocks[0]
            loss += 0.5 * initial_precision * float(np.sum(deviations[first] ** 2))
            if population_gradient is not None and emission_gradient is not None:
                value = initial_precision * deviations[first]
                emission_gradient[first] += value
                population_gradient[structure.population_index[first]] -= value
            for previous, current in pairwise(blocks):
                difference = deviations[current] - deviations[previous]
                loss += 0.5 * step_precision * float(np.sum(difference**2))
                if population_gradient is not None and emission_gradient is not None:
                    value = step_precision * difference
                    emission_gradient[previous] -= value
                    population_gradient[structure.population_index[previous]] += value
                    emission_gradient[current] += value
                    population_gradient[structure.population_index[current]] -= value
        if self.l2:
            penalized = np.asarray(
                [name != "intercept" for name in self.coefficient_names],
                dtype=bool,
            )
            loss += 0.5 * self.l2 * float(np.sum(population[:, :, penalized] ** 2))
            if population_gradient is not None:
                population_gradient[:, :, penalized] += self.l2 * population[:, :, penalized]
        return float(loss)

    def _population_map_objective(
        self,
        features: NDArray[np.float64],
        outcomes: NDArray[np.float64],
        initial: NDArray[np.float64],
        population: NDArray[np.float64],
        emissions: NDArray[np.float64],
        transitions: NDArray[np.float64],
        structure: _PopulationStructure,
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
        loss += self._population_emission_prior(
            population,
            emissions,
            None,
            None,
            structure,
        )
        if include_transition_prior:
            loss -= self.transition_concentration * float(
                np.sum(global_transition[None, :, :] * np.log(transitions))
            )
        return float(loss)

    def _canonicalize_population_trajectory(
        self,
        population: NDArray[np.float64],
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
            emissions[:, indices, :],
            transitions[:, indices][:, :, indices],
            initial[indices],
            global_transition[np.ix_(indices, indices)],
            permutation,
        )

    def _subject_label_diagnostics(
        self,
        emissions: NDArray[np.float64],
        structure: _PopulationStructure,
    ) -> tuple[NDArray[np.bool_], tuple[Any, ...], float]:
        crossing_rows: list[NDArray[np.bool_]] = []
        crossing_subjects: list[Any] = []
        gaps: list[float] = []
        n_pairs = self.n_states * (self.n_states - 1) // 2
        for subject, blocks in zip(
            structure.subjects,
            structure.subject_blocks,
            strict=True,
        ):
            crossings, gap = self._label_path_diagnostics(emissions[np.asarray(blocks)])
            gaps.append(gap)
            crossing_rows.extend(crossings)
            crossing_subjects.extend([subject] * len(crossings))
        values = (
            np.stack(crossing_rows) if crossing_rows else np.zeros((0, n_pairs), dtype=np.bool_)
        )
        return np.asarray(values, dtype=np.bool_), tuple(crossing_subjects), float(min(gaps))


@dataclass(frozen=True, slots=True)
class _PopulationStructure:
    subjects: tuple[Any, ...]
    path_subjects: tuple[Any, ...]
    keys: tuple[Any, ...]
    orders: NDArray[np.int64]
    sessions: tuple[tuple[int, ...], ...]
    population_orders: NDArray[np.int64]
    population_index: NDArray[np.intp]
    subject_blocks: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class _ValidatedPopulationTrajectory:
    subjects: tuple[Any, ...]
    path_subjects: tuple[Any, ...]
    keys: tuple[Any, ...]
    orders: NDArray[np.int64]
    population_orders: NDArray[np.int64]
    population_emissions: NDArray[np.float64]
    emissions: NDArray[np.float64]
    transitions: NDArray[np.float64]
    global_transition: NDArray[np.float64]


def _population_structure(
    study: Study,
    *,
    require_multiple_subjects: bool,
) -> _PopulationStructure:
    subjects = tuple(_scalar(subject) for subject in study.subjects)
    if require_multiple_subjects and len(subjects) < 2:
        raise ModelDataError(
            "HierarchicalSessionDynamicBernoulliGLMHMM requires at least two subjects"
        )
    sessions = ordered_session_indices(study)
    path_subjects: list[Any] = []
    keys: list[Any] = []
    orders: list[int] = []
    for indices in sessions:
        opening = indices[0]
        path_subjects.append(_scalar(study["subject"][opening]))
        keys.append(_scalar(study["session"][opening]))
        orders.append(int(study["session_order"][opening]))
    population_orders = np.asarray(sorted(set(orders)), dtype=np.int64)
    position = {int(order): index for index, order in enumerate(population_orders)}
    population_index = np.asarray([position[order] for order in orders], dtype=np.intp)
    blocks = _subject_blocks(subjects, tuple(path_subjects))
    return _PopulationStructure(
        subjects=subjects,
        path_subjects=tuple(path_subjects),
        keys=tuple(keys),
        orders=np.asarray(orders, dtype=np.int64),
        sessions=sessions,
        population_orders=population_orders,
        population_index=population_index,
        subject_blocks=blocks,
    )


def _subject_blocks(
    subjects: tuple[Any, ...], path_subjects: tuple[Any, ...]
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(index for index, value in enumerate(path_subjects) if value == subject)
        for subject in subjects
    )


def _validated_population_trajectory(
    subjects: tuple[Any, ...],
    path_subjects: tuple[Any, ...],
    keys: tuple[Any, ...],
    orders: NDArray[np.int64],
    population_orders: NDArray[np.int64],
    population_emissions: NDArray[np.float64],
    emissions: NDArray[np.float64],
    transitions: NDArray[np.float64],
    global_transition: NDArray[np.float64],
    *,
    n_states: int,
) -> _ValidatedPopulationTrajectory:
    subject_values = tuple(_scalar(value) for value in subjects)
    path_values = tuple(_scalar(value) for value in path_subjects)
    key_values = tuple(_scalar(value) for value in keys)
    order_values = protected_array(orders, dtype=np.int64)
    population_order_values = protected_array(population_orders, dtype=np.int64)
    population = protected_array(population_emissions, dtype=np.float64)
    subject_emissions = protected_array(emissions, dtype=np.float64)
    session_transitions = protected_array(transitions, dtype=np.float64)
    global_values = protected_array(global_transition, dtype=np.float64)
    if n_states < 2:
        raise ValueError("population trajectory must contain at least two states")
    if len(subject_values) < 2 or len(set(subject_values)) != len(subject_values):
        raise ValueError("population trajectories require at least two unique subjects")
    n_blocks = len(path_values)
    if not n_blocks or len(key_values) != n_blocks or order_values.shape != (n_blocks,):
        raise ValueError("path subjects, session keys, orders, and trajectories must align")
    if not set(path_values).issubset(set(subject_values)):
        raise ValueError("every path subject must name a fitted subject")
    if len(set(zip(path_values, key_values, strict=True))) != n_blocks:
        raise ValueError("session keys must be unique within subject")
    blocks = _subject_blocks(subject_values, path_values)
    if any(not block for block in blocks):
        raise ValueError("every fitted subject must have at least one session")
    for block in blocks:
        if np.any(np.diff(order_values[np.asarray(block, dtype=np.intp)]) <= 0):
            raise ValueError("session orders must increase strictly within subject")
    if (
        population_order_values.ndim != 1
        or not len(population_order_values)
        or np.any(np.diff(population_order_values) <= 0)
        or set(order_values.tolist()) - set(population_order_values.tolist())
    ):
        raise ValueError("population orders must be unique, increasing, and cover every path")
    if population.ndim != 3:
        raise ValueError("population emissions must be a three-dimensional path")
    expected_population = (len(population_order_values), n_states, population.shape[2])
    if population.shape != expected_population or population.shape[2] < 1:
        raise ValueError("population emissions must cover every order, state, and coefficient")
    if subject_emissions.shape != (n_blocks, n_states, population.shape[2]):
        raise ValueError("subject emissions must contain one path point per subject-session")
    if not np.all(np.isfinite(population)) or not np.all(np.isfinite(subject_emissions)):
        raise ValueError("population and subject emission paths must be finite")
    if session_transitions.shape != (n_blocks, n_states, n_states):
        raise ValueError("session transitions must contain one square matrix per path block")
    if global_values.shape != (n_states, n_states):
        raise ValueError("global transition matrix must contain one row per state")
    for name, values in (
        ("session transitions", session_transitions),
        ("global transition", global_values),
    ):
        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError(f"{name} must contain finite strictly positive probabilities")
        if not np.allclose(values.sum(axis=-1), 1.0, atol=1e-8):
            raise ValueError(f"{name} rows must sum to one")
    return _ValidatedPopulationTrajectory(
        subjects=subject_values,
        path_subjects=path_values,
        keys=key_values,
        orders=order_values,
        population_orders=population_order_values,
        population_emissions=population,
        emissions=subject_emissions,
        transitions=session_transitions,
        global_transition=global_values,
    )


def _set_validated_trajectory(
    target: Any,
    values: _ValidatedPopulationTrajectory,
) -> None:
    object.__setattr__(target, "subjects", values.subjects)
    object.__setattr__(target, "path_subjects", values.path_subjects)
    object.__setattr__(target, "session_keys", values.keys)
    object.__setattr__(target, "session_orders", values.orders)
    object.__setattr__(target, "population_session_orders", values.population_orders)
    object.__setattr__(
        target,
        "population_emission_coefficients",
        values.population_emissions,
    )
    object.__setattr__(target, "session_emission_coefficients", values.emissions)
    object.__setattr__(target, "session_transition_matrices", values.transitions)
    object.__setattr__(target, "global_transition_matrix", values.global_transition)


def _subject_deviation_contrast(
    structure: _PopulationStructure,
    population_shape: tuple[int, ...],
    emission_shape: tuple[int, ...],
) -> NDArray[np.float64]:
    """Map the joint ``(population, realized path)`` coordinate onto deviations."""

    trailing = int(np.prod(population_shape[1:], dtype=np.int64))
    if emission_shape[1:] != population_shape[1:]:
        raise ValueError("population and session path coordinates must share trailing axes")
    population_width = int(np.prod(population_shape, dtype=np.int64))
    session_width = int(np.prod(emission_shape, dtype=np.int64))
    matrix = np.zeros((session_width, population_width + session_width), dtype=np.float64)
    for block, population_index in enumerate(structure.population_index):
        for offset in range(trailing):
            row = block * trailing + offset
            matrix[row, int(population_index) * trailing + offset] = -1.0
            matrix[row, population_width + row] = 1.0
    return matrix


def _hierarchy_scale_contrasts(
    structure: _PopulationStructure,
    population_shape: tuple[int, ...],
    emission_shape: tuple[int, ...],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return contrasts for population steps, initial deviations, and subject steps."""

    trailing = int(np.prod(population_shape[1:], dtype=np.int64))
    population_width = int(np.prod(population_shape, dtype=np.int64))
    joint_width = population_width + int(np.prod(emission_shape, dtype=np.int64))
    population = np.zeros(((population_shape[0] - 1) * trailing, joint_width))
    row = 0
    for position in range(1, population_shape[0]):
        for offset in range(trailing):
            population[row, (position - 1) * trailing + offset] = -1.0
            population[row, position * trailing + offset] = 1.0
            row += 1

    deviations = _subject_deviation_contrast(structure, population_shape, emission_shape)
    initial_rows = np.asarray(
        [
            block * trailing + offset
            for blocks in structure.subject_blocks
            for block in blocks[:1]
            for offset in range(trailing)
        ],
        dtype=np.intp,
    )
    initial = deviations[initial_rows]
    step_rows: list[NDArray[np.float64]] = []
    for blocks in structure.subject_blocks:
        for previous, current in pairwise(blocks):
            for offset in range(trailing):
                step_rows.append(
                    deviations[current * trailing + offset]
                    - deviations[previous * trailing + offset]
                )
    if not step_rows:
        raise ValueError("subject-step contrasts require a repeated-session subject")
    return population, initial, np.stack(step_rows)


def _normal_path_intervals(
    estimates: NDArray[np.float64],
    standard_errors: NDArray[np.float64],
    *,
    level: float,
    available: bool,
) -> NDArray[np.float64]:
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie strictly between zero and one")
    if not available:
        raise ValueError("the local path covariance is unavailable")
    critical = float(ndtri(0.5 + level / 2.0))
    radius = critical * np.asarray(standard_errors, dtype=np.float64)
    return protected_array(
        np.stack((np.asarray(estimates) - radius, np.asarray(estimates) + radius), axis=-1),
        dtype=np.float64,
    )


def _normalized_positive(values: NDArray[np.float64]) -> NDArray[np.float64]:
    positive = np.maximum(np.asarray(values, dtype=np.float64), np.finfo(float).tiny)
    return positive / positive.sum()


def _scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value
