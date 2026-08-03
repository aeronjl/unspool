"""Session-dynamic Bernoulli GLM-HMMs with explicit trajectory diagnostics.

This module implements the temporal model of Lenc et al.: state-specific emission
coefficients follow Gaussian random walks across sessions, while each session's transition
rows are independent Dirichlet deviations around one global transition matrix.  The two
mechanisms are deliberately not described by the same word: only the emissions have a
temporally continuous prior.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

from behavio._internal.arrays import protected_array
from behavio.models._kernels.bernoulli import ordered_session_indices
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
    _forward_backward,
    _minimum_pairwise_distance,
)
from behavio.models.state_alignment import LatentStateAlignment, align_latent_states
from behavio.trials import Study


@dataclass(frozen=True, slots=True)
class SessionDynamicGLMHMMSimulation:
    """A simulated study paired with latent states and session-parameter truth."""

    study: Study
    states: NDArray[np.int64]
    n_states: int
    subject: Any
    session_keys: tuple[Any, ...]
    session_orders: NDArray[np.int64]
    emission_coefficients: NDArray[np.float64]
    transition_matrices: NDArray[np.float64]
    global_transition_matrix: NDArray[np.float64]

    def __post_init__(self) -> None:
        states = protected_array(self.states, dtype=np.int64)
        orders = protected_array(self.session_orders, dtype=np.int64)
        emissions = protected_array(self.emission_coefficients, dtype=np.float64)
        transitions = protected_array(self.transition_matrices, dtype=np.float64)
        global_transition = protected_array(self.global_transition_matrix, dtype=np.float64)
        keys = tuple(self.session_keys)
        if (
            isinstance(self.n_states, bool)
            or not isinstance(self.n_states, int)
            or self.n_states < 2
        ):
            raise ValueError("n_states must be an integer of at least two")
        if states.shape != (len(self.study),) or np.any((states < 0) | (states >= self.n_states)):
            raise ValueError("states must contain one valid label per trial")
        _validate_session_trajectory(
            keys,
            orders,
            emissions,
            transitions,
            global_transition,
            n_states=self.n_states,
        )
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "session_keys", keys)
        object.__setattr__(self, "session_orders", orders)
        object.__setattr__(self, "emission_coefficients", emissions)
        object.__setattr__(self, "transition_matrices", transitions)
        object.__setattr__(self, "global_transition_matrix", global_transition)


@dataclass(frozen=True, slots=True)
class SessionDynamicTrajectoryRecovery:
    """Truth-aligned recovery of state paths and their session parameters."""

    alignment: LatentStateAlignment
    emission_rmse: float
    transition_rmse: float
    session_emission_rmse: NDArray[np.float64]
    session_transition_rmse: NDArray[np.float64]

    def __post_init__(self) -> None:
        emission = protected_array(self.session_emission_rmse, dtype=np.float64)
        transition = protected_array(self.session_transition_rmse, dtype=np.float64)
        if emission.ndim != 1 or transition.shape != emission.shape or not len(emission):
            raise ValueError(
                "trajectory recovery needs one emission and transition RMSE per session"
            )
        if not np.all(np.isfinite(emission)) or np.any(emission < 0):
            raise ValueError("session emission RMSE values must be finite and non-negative")
        if not np.all(np.isfinite(transition)) or np.any(transition < 0):
            raise ValueError("session transition RMSE values must be finite and non-negative")
        if not np.isclose(self.emission_rmse, np.sqrt(np.mean(emission**2)), atol=1e-12):
            raise ValueError("emission_rmse must aggregate the session emission errors")
        if not np.isclose(self.transition_rmse, np.sqrt(np.mean(transition**2)), atol=1e-12):
            raise ValueError("transition_rmse must aggregate the session transition errors")
        object.__setattr__(self, "session_emission_rmse", emission)
        object.__setattr__(self, "session_transition_rmse", transition)


@dataclass(frozen=True, slots=True)
class SessionDynamicGLMHMMFitResult(FitResult):
    """A fitted session path with retained EM, label, and forecast evidence."""

    subject: Any
    session_keys: tuple[Any, ...]
    session_orders: NDArray[np.int64]
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
    minimum_label_path_gap: float
    label_crossings: NDArray[np.bool_]
    label_path_ambiguous: bool
    low_occupancy: bool
    emission_optimizer_converged: bool
    emission_optimizer_message: str
    initialization_restart_objectives: NDArray[np.float64]
    initialization_restart_converged: NDArray[np.bool_]
    initialization_restart_messages: tuple[str, ...]
    initialization_selected_restart: int
    emission_step_scale: float
    transition_concentration: float
    uncertainty_policy: str = "not-estimated"
    future_session_policy: str = "carry-last-emissions/use-global-transitions"

    def __post_init__(self) -> None:
        FitResult.__post_init__(self)
        orders = protected_array(self.session_orders, dtype=np.int64)
        emissions = protected_array(self.session_emission_coefficients, dtype=np.float64)
        transitions = protected_array(self.session_transition_matrices, dtype=np.float64)
        global_transition = protected_array(self.global_transition_matrix, dtype=np.float64)
        partial_history = protected_array(self.partial_objective_history, dtype=np.float64)
        history = protected_array(self.objective_history, dtype=np.float64)
        occupancy = protected_array(self.state_occupancy, dtype=np.float64)
        crossings = protected_array(self.label_crossings, dtype=np.bool_)
        restart_objectives = protected_array(
            self.initialization_restart_objectives, dtype=np.float64
        )
        restart_converged = protected_array(self.initialization_restart_converged, dtype=np.bool_)
        restart_messages = tuple(self.initialization_restart_messages)
        keys = tuple(self.session_keys)
        permutation = tuple(self.canonical_permutation)
        n_states = emissions.shape[1] if emissions.ndim == 3 else 0
        _validate_session_trajectory(
            keys,
            orders,
            emissions,
            transitions,
            global_transition,
            n_states=n_states,
        )
        if sorted(permutation) != list(range(n_states)) or n_states < 2:
            raise ValueError("canonical_permutation must permute every latent state")
        for name, values in (
            ("partial_objective_history", partial_history),
            ("objective_history", history),
        ):
            if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain finite EM objectives")
        if not self.partial_emission_optimizer_message:
            raise ValueError("partial_emission_optimizer_message must be non-empty")
        if occupancy.shape != (n_states,) or np.any(occupancy < 0):
            raise ValueError("state_occupancy must contain one non-negative value per state")
        if not np.isclose(occupancy.sum(), 1.0, atol=1e-8):
            raise ValueError("state_occupancy must sum to one")
        n_pairs = n_states * (n_states - 1) // 2
        if crossings.shape != (max(0, len(keys) - 1), n_pairs):
            raise ValueError("label_crossings must contain every adjacent-session state pair")
        for name, value in (
            ("minimum_state_separation", self.minimum_state_separation),
            ("minimum_label_path_gap", self.minimum_label_path_gap),
        ):
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if (
            restart_objectives.ndim != 1
            or not len(restart_objectives)
            or restart_converged.shape != restart_objectives.shape
            or len(restart_messages) != len(restart_objectives)
        ):
            raise ValueError("initialization restart evidence must align")
        if not 0 <= self.initialization_selected_restart < len(restart_objectives):
            raise ValueError("initialization_selected_restart must identify one restart")
        if not self.uncertainty_policy:
            raise ValueError("uncertainty_policy must be non-empty")
        if self.future_session_policy != "carry-last-emissions/use-global-transitions":
            raise ValueError("future_session_policy must name the model's declared forecast rule")
        object.__setattr__(self, "session_keys", keys)
        object.__setattr__(self, "session_orders", orders)
        object.__setattr__(self, "session_emission_coefficients", emissions)
        object.__setattr__(self, "session_transition_matrices", transitions)
        object.__setattr__(self, "global_transition_matrix", global_transition)
        object.__setattr__(self, "partial_objective_history", partial_history)
        object.__setattr__(self, "objective_history", history)
        object.__setattr__(self, "canonical_permutation", permutation)
        object.__setattr__(self, "state_occupancy", occupancy)
        object.__setattr__(self, "label_crossings", crossings)
        object.__setattr__(self, "initialization_restart_objectives", restart_objectives)
        object.__setattr__(self, "initialization_restart_converged", restart_converged)
        object.__setattr__(self, "initialization_restart_messages", restart_messages)

    @property
    def state_separation(self) -> float:
        """Alias the worst session separation onto the common latent-state audit contract."""

        return self.minimum_state_separation

    @property
    def label_order_gap(self) -> float:
        """Alias the smallest whole-path gap onto the common latent-state audit contract."""

        return self.minimum_label_path_gap

    @property
    def label_ambiguous(self) -> bool:
        """Alias whole-path ambiguity onto the common latent-state audit contract."""

        return self.label_path_ambiguous

    @property
    def restart_objectives(self) -> NDArray[np.float64]:
        """Expose stationary initialization objectives to the common restart audit."""

        return self.initialization_restart_objectives

    @property
    def restart_converged(self) -> NDArray[np.bool_]:
        """Expose stationary initialization convergence to the common restart audit."""

        return self.initialization_restart_converged

    @property
    def restart_messages(self) -> tuple[str, ...]:
        """Expose stationary initialization messages to the common restart audit."""

        return self.initialization_restart_messages

    @property
    def selected_restart(self) -> int:
        """Expose the selected stationary initializer to the common restart audit."""

        return self.initialization_selected_restart


@dataclass(frozen=True, slots=True)
class SessionDynamicBernoulliGLMHMM(BernoulliGLMHMM):
    """A per-subject GLM-HMM whose emission weights evolve between sessions.

    Given global transition row ``A[k]``, session row ``P[s, k]`` has prior
    ``Dirichlet(transition_concentration * A[k] + 1)``.  Emission coefficients have the
    first-order Gaussian prior ``W[s] ~ Normal(W[s-1], emission_step_scale)``.  Fitting uses
    MAP EM initialized by the corresponding stationary GLM-HMM.

    The estimator intentionally fits one subject at a time.  A joint cross-subject dynamic
    hierarchy would require another declared population distribution and is not smuggled in
    as pooled sessions.
    """

    emission_step_scale: float = 0.5
    transition_concentration: float = 10.0
    dynamic_max_iterations: int = 100
    dynamic_tolerance: float = 1e-6

    def __post_init__(self) -> None:
        BernoulliGLMHMM.__post_init__(self)
        if self.transition_predictors or self.transition_design is not None:
            raise ValueError(
                "session-dynamic transitions and trial-covariate transition regression are "
                "different mechanisms and cannot be combined in this estimator"
            )
        if self.stickiness:
            raise ValueError(
                "stickiness applies to one stationary matrix; session transitions already "
                "have an explicit Dirichlet prior around their global matrix"
            )
        if not np.isfinite(self.emission_step_scale) or self.emission_step_scale <= 0:
            raise ValueError("emission_step_scale must be finite and positive")
        if not np.isfinite(self.transition_concentration) or self.transition_concentration <= 0:
            raise ValueError("transition_concentration must be finite and positive")
        if (
            isinstance(self.dynamic_max_iterations, bool)
            or not isinstance(self.dynamic_max_iterations, int)
            or self.dynamic_max_iterations < 1
        ):
            raise ValueError("dynamic_max_iterations must be a positive integer")
        if not np.isfinite(self.dynamic_tolerance) or self.dynamic_tolerance <= 0:
            raise ValueError("dynamic_tolerance must be finite and positive")

    @property
    def model_name(self) -> str:
        return "session-dynamic-bernoulli-glm-hmm"

    @property
    def is_session_dynamic(self) -> bool:
        """Return true for the latent session-parameter model.

        ``is_dynamic`` on the parent class continues to mean trial-level transition
        regression. Keeping the two predicates distinct prevents a session prior from
        silently changing the stationary initializer's transition coordinate.
        """

        return True

    @property
    def signature(self) -> str:
        predictors = ",".join(self.predictors)
        return (
            f"{self.model_name}[states={self.n_states};outcome={self.outcome};"
            f"predictors={predictors};choice_lags={self.choice_lags};label_by={self.label_by};"
            f"l2={self.l2};emission_step_scale={self.emission_step_scale};"
            f"transition_concentration={self.transition_concentration}{self._design_signature}]"
        )

    @property
    def declared_priors(self) -> tuple[str, ...]:
        declared = list(BernoulliGLMHMM.declared_priors.fget(self))
        declared.extend(
            (
                "Gaussian random walk on every state-specific emission coefficient: "
                f"Normal(previous session, {self.emission_step_scale:.4g})",
                "independent session transition rows around the global row: "
                "Dirichlet(transition_concentration * global_transition + 1), "
                f"transition_concentration={self.transition_concentration:.4g}",
            )
        )
        return tuple(declared)

    @property
    def bounded_coordinate_refusal(self) -> str:
        return (
            "a session-dynamic GLM-HMM's fitted coordinate contains one emission vector and "
            "transition matrix per observed session, so its dimension is data-dependent and "
            "its Gaussian random-walk and Dirichlet path priors are neither a row score nor "
            "one fixed quadratic penalty. Fit it per subject; a cross-subject dynamic "
            "hierarchy needs its own declared population model"
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
    ) -> SessionDynamicGLMHMMSimulation:
        return self.simulate_with_trajectories(design, parameters, seed=seed)

    def simulate_with_trajectories(
        self,
        design: Study,
        parameters: Mapping[str, float],
        *,
        seed: int | np.random.Generator,
    ) -> SessionDynamicGLMHMMSimulation:
        """Draw session parameters, choices, and separately retained latent truth."""

        subject, keys, orders, sessions = _session_structure(design)
        base = self.parameter_components(parameters)
        generator = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        n_sessions = len(sessions)
        emissions = np.empty(
            (n_sessions, self.n_states, len(self.coefficient_names)), dtype=np.float64
        )
        emissions[0] = base.emission_coefficients
        for session in range(1, n_sessions):
            emissions[session] = emissions[session - 1] + generator.normal(
                0.0, self.emission_step_scale, emissions[session].shape
            )
        transitions = np.empty((n_sessions, self.n_states, self.n_states), dtype=np.float64)
        for session in range(n_sessions):
            for state in range(self.n_states):
                transitions[session, state] = generator.dirichlet(
                    self.transition_concentration * base.transition_matrix[state] + 1.0
                )
        components = tuple(
            GLMHMMParameters(
                initial_probabilities=base.initial_probabilities,
                transition_matrix=transitions[session],
                emission_coefficients=emissions[session],
                coefficient_names=self.coefficient_names,
            )
            for session in range(n_sessions)
        )
        by_opening_row = {
            int(indices[0]): components[session] for session, indices in enumerate(sessions)
        }
        choices, states = self._generate(
            design,
            lambda indices: by_opening_row[int(indices[0])],
            generator,
        )
        columns = {name: design[name] for name in design.columns}
        columns[self.outcome] = choices
        return SessionDynamicGLMHMMSimulation(
            study=Study(columns),
            states=states,
            n_states=self.n_states,
            subject=subject,
            session_keys=keys,
            session_orders=np.asarray(orders, dtype=np.int64),
            emission_coefficients=emissions,
            transition_matrices=transitions,
            global_transition_matrix=base.transition_matrix,
        )

    def fit(self, study: Study) -> SessionDynamicGLMHMMFitResult:
        """Fit the session path with the reference stationary, partial, and full stages."""

        subject, keys, orders, sessions = _session_structure(study)
        outcomes = self.outcomes(study)
        features = self.design_matrix(study)
        initialization = BernoulliGLMHMM.fit(self, study)
        base = self.parameter_components(initialization)
        global_transition = np.asarray(base.transition_matrix).copy()
        initial = np.asarray(base.initial_probabilities).copy()
        emissions = np.tile(np.asarray(base.emission_coefficients), (len(sessions), 1, 1)).copy()
        transitions = np.tile(global_transition, (len(sessions), 1, 1)).copy()
        partial_history: list[float] = []
        partial_message = "partial emission M-step was not run"
        partial_emission_converged = False
        partial_converged = False

        # The published fitting pipeline first permits emission paths to vary while one
        # transition matrix remains shared.  Besides being faithful to that sequence, this
        # prevents session transition deviations from absorbing structure before the
        # random-walk emission path has moved away from its stationary initializer.
        for _ in range(self.dynamic_max_iterations):
            posterior = self._dynamic_posterior(
                features, outcomes, sessions, initial, emissions, transitions
            )
            initial = _normalized_positive(posterior.initial_counts)
            global_transition = self._stationary_transition_m_step(
                posterior.transition_expectations,
                sessions,
                global_transition,
            )
            transitions = np.tile(global_transition, (len(sessions), 1, 1)).copy()
            result = self._optimize_emissions(
                emissions,
                features,
                outcomes,
                sessions,
                posterior.state_probabilities,
            )
            emissions = np.asarray(result.x, dtype=np.float64).reshape(emissions.shape)
            partial_emission_converged = bool(result.success)
            partial_message = str(result.message)
            objective = self._partial_map_objective(
                features,
                outcomes,
                sessions,
                initial,
                emissions,
                transitions,
            )
            partial_history.append(objective)
            if self._objective_converged(partial_history):
                partial_converged = partial_emission_converged
                break

        objective_history: list[float] = []
        emission_message = "emission M-step was not run"
        emission_converged = False
        converged = False

        for _ in range(self.dynamic_max_iterations):
            posterior = self._dynamic_posterior(
                features, outcomes, sessions, initial, emissions, transitions
            )
            initial = _normalized_positive(posterior.initial_counts)
            transitions = self._transition_m_step(
                posterior.transition_expectations,
                sessions,
                global_transition,
            )
            result = self._optimize_emissions(
                emissions,
                features,
                outcomes,
                sessions,
                posterior.state_probabilities,
            )
            emissions = np.asarray(result.x, dtype=np.float64).reshape(emissions.shape)
            emission_converged = bool(result.success)
            emission_message = str(result.message)
            objective = self._dynamic_map_objective(
                features,
                outcomes,
                sessions,
                initial,
                emissions,
                transitions,
                global_transition,
            )
            objective_history.append(objective)
            if self._objective_converged(objective_history):
                converged = emission_converged
                break

        emissions, transitions, initial, global_transition, permutation = (
            self._canonicalize_trajectory(
                emissions,
                transitions,
                initial,
                global_transition,
            )
        )
        posterior = self._dynamic_posterior(
            features, outcomes, sessions, initial, emissions, transitions
        )
        occupancy = np.mean(posterior.state_probabilities, axis=0)
        crossings, minimum_label_gap = self._label_path_diagnostics(emissions)
        minimum_separation = min(
            _minimum_pairwise_distance(session_emissions) for session_emissions in emissions
        )
        final_components = GLMHMMParameters(
            initial_probabilities=initial,
            transition_matrix=global_transition,
            emission_coefficients=emissions[0],
            coefficient_names=self.coefficient_names,
        )
        estimates = self._pack_components(final_components)
        final_value, final_gradient = self._emission_m_step_objective(
            emissions.ravel(),
            features,
            outcomes,
            sessions,
            posterior.state_probabilities,
        )
        del final_value
        probability_values = np.concatenate((initial, transitions.ravel()))
        boundary = bool(
            np.any(np.abs(emissions) >= self.coefficient_warning_threshold)
            or np.any(probability_values <= self.probability_warning_threshold)
            or np.any(probability_values >= 1.0 - self.probability_warning_threshold)
        )
        label_ambiguous = bool(np.any(crossings) or minimum_label_gap <= self.label_tolerance)
        overall_converged = partial_converged and converged
        diagnostics = FitDiagnostics(
            converged=overall_converged,
            optimizer=(
                "three-stage MAP EM (stationary multistart; emission-dynamic/static-"
                "transition; fully dynamic; L-BFGS-B emission M-steps)"
            ),
            status=0 if overall_converged else 1,
            message=(
                "three-stage EM met the declared tolerance; "
                if overall_converged
                else "three-stage EM did not meet the declared tolerance; "
            )
            + f"partial stage {'converged' if partial_converged else 'did not converge'}; "
            + f"full stage {'converged' if converged else 'did not converge'}; "
            + emission_message,
            n_iterations=len(partial_history) + len(objective_history),
            objective=float(objective_history[-1]),
            gradient_norm=float(np.linalg.norm(final_gradient)),
            hessian_condition=None,
            boundary_estimate=boundary,
        )
        n_parameters = len(estimates)
        return SessionDynamicGLMHMMFitResult(
            model_name=self.model_name,
            model_signature=self.signature,
            parameter_names=self.parameter_names,
            estimates=estimates,
            standard_errors=np.full(n_parameters, np.nan),
            covariance=np.full((n_parameters, n_parameters), np.nan),
            n_observations=len(study),
            diagnostics=diagnostics,
            subject=subject,
            session_keys=keys,
            session_orders=np.asarray(orders, dtype=np.int64),
            session_emission_coefficients=emissions,
            session_transition_matrices=transitions,
            global_transition_matrix=global_transition,
            partial_objective_history=np.asarray(partial_history),
            partial_converged=partial_converged,
            partial_emission_optimizer_converged=partial_emission_converged,
            partial_emission_optimizer_message=partial_message,
            full_converged=converged,
            objective_history=np.asarray(objective_history),
            canonical_permutation=permutation,
            state_occupancy=occupancy,
            minimum_state_separation=minimum_separation,
            minimum_label_path_gap=minimum_label_gap,
            label_crossings=crossings,
            label_path_ambiguous=label_ambiguous,
            low_occupancy=bool(np.any(occupancy < self.state_occupancy_warning)),
            emission_optimizer_converged=emission_converged,
            emission_optimizer_message=emission_message,
            initialization_restart_objectives=initialization.restart_objectives,
            initialization_restart_converged=initialization.restart_converged,
            initialization_restart_messages=initialization.restart_messages,
            initialization_selected_restart=initialization.selected_restart,
            emission_step_scale=self.emission_step_scale,
            transition_concentration=self.transition_concentration,
        )

    def transition_probabilities(
        self,
        study: Study,
        parameters: Mapping[str, float] | FitResult,
    ) -> NDArray[np.float64]:
        if not isinstance(parameters, FitResult):
            return BernoulliGLMHMM.transition_probabilities(self, study, parameters)
        fit = self._validate_dynamic_fit(parameters)
        components = self._prediction_components(study, fit)
        values = np.empty((len(study), self.n_states, self.n_states), dtype=np.float64)
        for session, indices in enumerate(ordered_session_indices(study)):
            values[np.asarray(indices, dtype=np.intp)] = components[session].transition_matrix
        return protected_array(values, dtype=np.float64)

    def predict(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> Prediction:
        prediction_mode = self._prediction_mode(mode)
        dynamic_fit = self._validate_dynamic_fit(fit)
        outcomes = self.outcomes(study)
        features = self.design_matrix(study)
        components = self._prediction_components(study, dynamic_fit)
        probabilities = self._dynamic_filtered(features, outcomes, study, components)
        emission_probability = np.empty((len(study), self.n_states), dtype=np.float64)
        for session, indices in enumerate(ordered_session_indices(study)):
            index = np.asarray(indices, dtype=np.intp)
            emission_probability[index] = expit(
                features[index] @ components[session].emission_coefficients.T
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
        prediction = self.predict(study, fit, mode=mode)
        scores = outcomes * np.log(prediction.probability)
        scores += (1.0 - outcomes) * np.log1p(-prediction.probability)
        return protected_array(scores, dtype=np.float64)

    def state_probabilities(
        self,
        study: Study,
        fit: FitResult,
    ) -> FilteredStateProbabilities:
        dynamic_fit = self._validate_dynamic_fit(fit)
        outcomes = self.outcomes(study)
        features = self.design_matrix(study)
        components = self._prediction_components(study, dynamic_fit)
        return self._dynamic_filtered(features, outcomes, study, components)

    def state_recovery(
        self,
        simulation: SessionDynamicGLMHMMSimulation,
        fit: FitResult,
        *,
        ambiguity_tolerance: float = 0.05,
    ) -> LatentStateAlignment:
        if not isinstance(simulation, SessionDynamicGLMHMMSimulation):
            raise TypeError("simulation must be a SessionDynamicGLMHMMSimulation")
        if simulation.n_states != self.n_states:
            raise ValueError("simulation and model must contain the same number of states")
        return align_latent_states(
            simulation.states,
            self.state_probabilities(simulation.study, fit).filtered,
            ambiguity_tolerance=ambiguity_tolerance,
        )

    def trajectory_recovery(
        self,
        simulation: SessionDynamicGLMHMMSimulation,
        fit: FitResult,
        *,
        ambiguity_tolerance: float = 0.05,
    ) -> SessionDynamicTrajectoryRecovery:
        """Compare fitted session parameters with truth after one whole-path alignment."""

        dynamic_fit = self._validate_dynamic_fit(fit)
        if not isinstance(simulation, SessionDynamicGLMHMMSimulation):
            raise TypeError("simulation must be a SessionDynamicGLMHMMSimulation")
        if simulation.session_keys != dynamic_fit.session_keys or not np.array_equal(
            simulation.session_orders, dynamic_fit.session_orders
        ):
            raise ValueError("simulation and fit must describe the same ordered sessions")
        alignment = self.state_recovery(
            simulation,
            dynamic_fit,
            ambiguity_tolerance=ambiguity_tolerance,
        )
        mapping = np.asarray(alignment.reference_to_inferred, dtype=np.intp)
        fitted_emissions = dynamic_fit.session_emission_coefficients[:, mapping, :]
        fitted_transitions = dynamic_fit.session_transition_matrices[:, mapping][:, :, mapping]
        emission_error = fitted_emissions - simulation.emission_coefficients
        transition_error = fitted_transitions - simulation.transition_matrices
        session_emission = np.sqrt(np.mean(emission_error**2, axis=(1, 2)))
        session_transition = np.sqrt(np.mean(transition_error**2, axis=(1, 2)))
        return SessionDynamicTrajectoryRecovery(
            alignment=alignment,
            emission_rmse=float(np.sqrt(np.mean(emission_error**2))),
            transition_rmse=float(np.sqrt(np.mean(transition_error**2))),
            session_emission_rmse=session_emission,
            session_transition_rmse=session_transition,
        )

    def _validate_dynamic_fit(self, fit: FitResult) -> SessionDynamicGLMHMMFitResult:
        self._validate_fit(fit)
        if not isinstance(fit, SessionDynamicGLMHMMFitResult):
            raise TypeError("session-dynamic prediction requires a SessionDynamicGLMHMMFitResult")
        return fit

    def _prediction_components(
        self,
        study: Study,
        fit: SessionDynamicGLMHMMFitResult,
    ) -> tuple[GLMHMMParameters, ...]:
        subject, keys, orders, _ = _session_structure(study)
        if subject != fit.subject:
            raise ModelDataError(
                "session-dynamic prediction is conditional on the fitted subject; an unseen "
                "subject needs a declared population hierarchy"
            )
        known = {
            key: (int(order), session)
            for session, (key, order) in enumerate(
                zip(fit.session_keys, fit.session_orders, strict=True)
            )
        }
        components: list[GLMHMMParameters] = []
        last_order = int(fit.session_orders[-1])
        base = self.parameter_components(fit)
        for key, order_value in zip(keys, orders, strict=True):
            order = int(order_value)
            if key in known:
                fitted_order, session = known[key]
                if order != fitted_order:
                    raise ModelDataError(
                        f"session {key!r} was fitted at order {fitted_order}, not {order}"
                    )
                emissions = fit.session_emission_coefficients[session]
                transitions = fit.session_transition_matrices[session]
            elif order > last_order:
                emissions = fit.session_emission_coefficients[-1]
                transitions = fit.global_transition_matrix
            else:
                raise ModelDataError(
                    f"session {key!r} at order {order} was not fitted and is not prospectively "
                    f"later than the last training session ({last_order})"
                )
            components.append(
                GLMHMMParameters(
                    initial_probabilities=base.initial_probabilities,
                    transition_matrix=transitions,
                    emission_coefficients=emissions,
                    coefficient_names=self.coefficient_names,
                )
            )
        return tuple(components)

    def _dynamic_filtered(
        self,
        features: NDArray[np.float64],
        outcomes: NDArray[np.float64],
        study: Study,
        components: tuple[GLMHMMParameters, ...],
    ) -> FilteredStateProbabilities:
        predictive = np.empty((len(study), self.n_states), dtype=np.float64)
        filtered = np.empty_like(predictive)
        for session, session_indices in enumerate(ordered_session_indices(study)):
            index = np.asarray(session_indices, dtype=np.intp)
            block = components[session]
            linear = features[index] @ block.emission_coefficients.T
            emission_log = outcomes[index, None] * -np.logaddexp(0.0, -linear) + (
                1.0 - outcomes[index, None]
            ) * -np.logaddexp(0.0, linear)
            prior = np.asarray(block.initial_probabilities)
            for position, row in enumerate(index):
                predictive[row] = prior
                log_weight = np.log(prior) + emission_log[position]
                posterior = np.exp(log_weight - logsumexp(log_weight))
                filtered[row] = posterior
                if position + 1 < len(index):
                    prior = posterior @ block.transition_matrix
        return FilteredStateProbabilities(predictive=predictive, filtered=filtered)

    def _dynamic_posterior(
        self,
        features: NDArray[np.float64],
        outcomes: NDArray[np.float64],
        sessions: tuple[tuple[int, ...], ...],
        initial: NDArray[np.float64],
        emissions: NDArray[np.float64],
        transitions: NDArray[np.float64],
    ) -> Any:
        emission_log = np.empty((len(outcomes), self.n_states), dtype=np.float64)
        transition_rows = np.empty((len(outcomes), self.n_states, self.n_states), dtype=np.float64)
        for session, session_indices in enumerate(sessions):
            index = np.asarray(session_indices, dtype=np.intp)
            linear = features[index] @ emissions[session].T
            emission_log[index] = outcomes[index, None] * -np.logaddexp(0.0, -linear) + (
                1.0 - outcomes[index, None]
            ) * -np.logaddexp(0.0, linear)
            transition_rows[index] = transitions[session]
        return _forward_backward(
            np.log(initial),
            np.log(transition_rows),
            emission_log,
            sessions,
        )

    def _transition_m_step(
        self,
        transition_expectations: NDArray[np.float64],
        sessions: tuple[tuple[int, ...], ...],
        global_transition: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return the closed-form Dirichlet-regularized session transition update."""

        transitions = np.empty((len(sessions), self.n_states, self.n_states), dtype=np.float64)
        for session, session_indices in enumerate(sessions):
            counts = np.sum(
                transition_expectations[np.asarray(session_indices, dtype=np.intp)], axis=0
            )
            numerator = counts + self.transition_concentration * global_transition
            transitions[session] = numerator / numerator.sum(axis=1, keepdims=True)
        return transitions

    def _stationary_transition_m_step(
        self,
        transition_expectations: NDArray[np.float64],
        sessions: tuple[tuple[int, ...], ...],
        fallback: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Fit one shared transition matrix for the intermediate reference stage."""

        counts = np.sum(
            np.concatenate(
                [
                    transition_expectations[np.asarray(indices, dtype=np.intp)]
                    for indices in sessions
                ],
                axis=0,
            ),
            axis=0,
        )
        row_totals = counts.sum(axis=1, keepdims=True)
        shared = np.divide(
            counts,
            row_totals,
            out=np.asarray(fallback, dtype=np.float64).copy(),
            where=row_totals > 0,
        )
        return np.asarray(shared, dtype=np.float64)

    def _optimize_emissions(
        self,
        emissions: NDArray[np.float64],
        features: NDArray[np.float64],
        outcomes: NDArray[np.float64],
        sessions: tuple[tuple[int, ...], ...],
        state_probabilities: NDArray[np.float64],
    ) -> Any:
        """Run the common random-walk emission M-step used by both dynamic stages."""

        return minimize(
            lambda vector: self._emission_m_step_objective(
                vector,
                features,
                outcomes,
                sessions,
                state_probabilities,
            ),
            emissions.ravel(),
            method="L-BFGS-B",
            jac=True,
            bounds=[(-30.0, 30.0)] * emissions.size,
            options={
                "maxiter": self.max_iterations,
                "ftol": self.tolerance,
                "gtol": self.tolerance,
            },
        )

    def _objective_converged(self, history: Sequence[float]) -> bool:
        """Apply the declared relative stopping rule to a stage objective history."""

        if len(history) < 2:
            return False
        change = abs(history[-1] - history[-2])
        scale = 1.0 + abs(history[-2])
        return bool(change <= self.dynamic_tolerance * scale)

    def _emission_m_step_objective(
        self,
        vector: NDArray[np.float64],
        features: NDArray[np.float64],
        outcomes: NDArray[np.float64],
        sessions: tuple[tuple[int, ...], ...],
        state_probabilities: NDArray[np.float64],
    ) -> tuple[float, NDArray[np.float64]]:
        """Expected complete emission loss and analytic gradient for one EM M-step."""

        shape = (len(sessions), self.n_states, features.shape[1])
        emissions = np.asarray(vector, dtype=np.float64).reshape(shape)
        gradient = np.zeros_like(emissions)
        loss = 0.0
        for session, session_indices in enumerate(sessions):
            index = np.asarray(session_indices, dtype=np.intp)
            linear = features[index] @ emissions[session].T
            weights = state_probabilities[index]
            loss += float(
                np.sum(weights * (np.logaddexp(0.0, linear) - outcomes[index, None] * linear))
            )
            residual = weights * (expit(linear) - outcomes[index, None])
            gradient[session] = residual.T @ features[index]
        inverse_variance = 1.0 / self.emission_step_scale**2
        differences = np.diff(emissions, axis=0)
        loss += 0.5 * inverse_variance * float(np.sum(differences**2))
        if len(sessions) > 1:
            gradient[:-1] -= inverse_variance * differences
            gradient[1:] += inverse_variance * differences
        if self.l2:
            penalized = np.asarray(
                [name != "intercept" for name in self.coefficient_names], dtype=bool
            )
            loss += 0.5 * self.l2 * float(np.sum(emissions[:, :, penalized] ** 2))
            gradient[:, :, penalized] += self.l2 * emissions[:, :, penalized]
        return float(loss), gradient.ravel()

    def _dynamic_map_objective(
        self,
        features: NDArray[np.float64],
        outcomes: NDArray[np.float64],
        sessions: tuple[tuple[int, ...], ...],
        initial: NDArray[np.float64],
        emissions: NDArray[np.float64],
        transitions: NDArray[np.float64],
        global_transition: NDArray[np.float64],
    ) -> float:
        posterior = self._dynamic_posterior(
            features, outcomes, sessions, initial, emissions, transitions
        )
        differences = np.diff(emissions, axis=0)
        loss = -posterior.log_likelihood
        loss += 0.5 / self.emission_step_scale**2 * float(np.sum(differences**2))
        if self.l2:
            penalized = np.asarray(
                [name != "intercept" for name in self.coefficient_names], dtype=bool
            )
            loss += 0.5 * self.l2 * float(np.sum(emissions[:, :, penalized] ** 2))
        loss -= self.transition_concentration * float(
            np.sum(global_transition[None, :, :] * np.log(transitions))
        )
        return float(loss)

    def _partial_map_objective(
        self,
        features: NDArray[np.float64],
        outcomes: NDArray[np.float64],
        sessions: tuple[tuple[int, ...], ...],
        initial: NDArray[np.float64],
        emissions: NDArray[np.float64],
        transitions: NDArray[np.float64],
    ) -> float:
        """MAP objective for emission paths with one shared transition matrix."""

        posterior = self._dynamic_posterior(
            features,
            outcomes,
            sessions,
            initial,
            emissions,
            transitions,
        )
        differences = np.diff(emissions, axis=0)
        loss = -posterior.log_likelihood
        loss += 0.5 / self.emission_step_scale**2 * float(np.sum(differences**2))
        if self.l2:
            penalized = np.asarray(
                [name != "intercept" for name in self.coefficient_names], dtype=bool
            )
            loss += 0.5 * self.l2 * float(np.sum(emissions[:, :, penalized] ** 2))
        return float(loss)

    def _canonicalize_trajectory(
        self,
        emissions: NDArray[np.float64],
        transitions: NDArray[np.float64],
        initial: NDArray[np.float64],
        global_transition: NDArray[np.float64],
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        tuple[int, ...],
    ]:
        label_index = self.coefficient_names.index(self.label_by)
        means = np.mean(emissions, axis=0)
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
            emissions[:, indices, :],
            transitions[:, indices][:, :, indices],
            initial[indices],
            global_transition[np.ix_(indices, indices)],
            permutation,
        )

    def _label_path_diagnostics(
        self,
        emissions: NDArray[np.float64],
    ) -> tuple[NDArray[np.bool_], float]:
        label_index = self.coefficient_names.index(self.label_by)
        values = emissions[:, :, label_index]
        pairs = [
            (left, right)
            for left in range(self.n_states)
            for right in range(left + 1, self.n_states)
        ]
        gaps = np.column_stack([values[:, right] - values[:, left] for left, right in pairs])
        minimum_gap = float(np.min(np.abs(gaps)))
        if len(emissions) == 1:
            crossings = np.zeros((0, len(pairs)), dtype=np.bool_)
        else:
            crossings = (gaps[:-1] * gaps[1:] <= 0) | (
                np.minimum(np.abs(gaps[:-1]), np.abs(gaps[1:])) <= self.label_tolerance
            )
        return np.asarray(crossings, dtype=np.bool_), minimum_gap


def _session_structure(
    study: Study,
) -> tuple[Any, tuple[Any, ...], tuple[int, ...], tuple[tuple[int, ...], ...]]:
    if len(study.subjects) != 1:
        raise ModelDataError(
            "SessionDynamicBernoulliGLMHMM fits one subject at a time; use an explicit "
            "cross-subject hierarchy rather than pooling subject-specific paths"
        )
    sessions = ordered_session_indices(study)
    keys: list[Any] = []
    orders: list[int] = []
    for indices in sessions:
        opening = indices[0]
        key = study["session"][opening]
        keys.append(key.item() if isinstance(key, np.generic) else key)
        orders.append(int(study["session_order"][opening]))
    return study.subjects[0], tuple(keys), tuple(orders), sessions


def _normalized_positive(values: NDArray[np.float64]) -> NDArray[np.float64]:
    positive = np.maximum(np.asarray(values, dtype=np.float64), np.finfo(float).tiny)
    return positive / positive.sum()


def _validate_session_trajectory(
    keys: tuple[Any, ...],
    orders: NDArray[np.int64],
    emissions: NDArray[np.float64],
    transitions: NDArray[np.float64],
    global_transition: NDArray[np.float64],
    *,
    n_states: int,
) -> None:
    if n_states < 2 or emissions.ndim != 3:
        raise ValueError("emission trajectory must contain at least two states")
    n_sessions, observed_states, n_coefficients = emissions.shape
    if not keys or len(keys) != n_sessions or orders.shape != (n_sessions,):
        raise ValueError("session keys, orders, and trajectories must align")
    if len(set(keys)) != len(keys) or np.any(np.diff(orders) <= 0):
        raise ValueError("session keys must be unique and session orders strictly increasing")
    if observed_states != n_states or n_coefficients < 1 or not np.all(np.isfinite(emissions)):
        raise ValueError("emission trajectory must contain finite coefficients for every state")
    if transitions.shape != (n_sessions, n_states, n_states):
        raise ValueError("transition trajectory must contain one square matrix per session")
    if global_transition.shape != (n_states, n_states):
        raise ValueError("global transition matrix must contain one row per state")
    for name, values in (
        ("session transitions", transitions),
        ("global transition", global_transition),
    ):
        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError(f"{name} must contain finite strictly positive probabilities")
        if not np.allclose(values.sum(axis=-1), 1.0, atol=1e-8):
            raise ValueError(f"{name} rows must sum to one")
