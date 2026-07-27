"""Fixed-transition Bernoulli GLM-HMM with explicit label diagnostics."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

from unspool.models.base import (
    FitDiagnostics,
    FitResult,
    ModelDataError,
    Prediction,
    PredictionMode,
    _protected_array,
)
from unspool.models.glm import BernoulliHistoryGLM, _ordered_session_indices
from unspool.state_alignment import LatentStateAlignment, align_latent_states
from unspool.study import Study


@dataclass(frozen=True, slots=True)
class GLMHMMParameters:
    """Natural-scale components of a Bernoulli GLM-HMM parameter vector."""

    initial_probabilities: NDArray[np.float64]
    transition_matrix: NDArray[np.float64]
    emission_coefficients: NDArray[np.float64]
    coefficient_names: tuple[str, ...]

    def __post_init__(self) -> None:
        initial = _protected_array(self.initial_probabilities, dtype=np.float64)
        transition = _protected_array(self.transition_matrix, dtype=np.float64)
        emissions = _protected_array(self.emission_coefficients, dtype=np.float64)
        names = tuple(self.coefficient_names)
        if initial.ndim != 1 or len(initial) < 2:
            raise ValueError("initial_probabilities must contain at least two states")
        n_states = len(initial)
        if transition.shape != (n_states, n_states):
            raise ValueError("transition_matrix must have one row and column per state")
        if emissions.shape != (n_states, len(names)) or not names:
            raise ValueError("emission_coefficients must have one named row per state")
        if len(set(names)) != len(names):
            raise ValueError("coefficient_names must be unique")
        if not np.all(np.isfinite(initial)) or np.any(initial <= 0):
            raise ValueError("initial probabilities must be finite and strictly positive")
        if not np.isclose(initial.sum(), 1.0, atol=1e-8):
            raise ValueError("initial probabilities must sum to one")
        if not np.all(np.isfinite(transition)) or np.any(transition <= 0):
            raise ValueError("transition probabilities must be finite and strictly positive")
        if not np.allclose(transition.sum(axis=1), 1.0, atol=1e-8):
            raise ValueError("every transition row must sum to one")
        if not np.all(np.isfinite(emissions)):
            raise ValueError("emission coefficients must be finite")
        object.__setattr__(self, "initial_probabilities", initial)
        object.__setattr__(self, "transition_matrix", transition)
        object.__setattr__(self, "emission_coefficients", emissions)
        object.__setattr__(self, "coefficient_names", names)

    @property
    def n_states(self) -> int:
        return len(self.initial_probabilities)


@dataclass(frozen=True, slots=True)
class GLMHMMSimulation:
    """A simulated observed study paired with its unexposed latent-state truth."""

    study: Study
    states: NDArray[np.int64]
    n_states: int

    def __post_init__(self) -> None:
        states = _protected_array(self.states, dtype=np.int64)
        if (
            isinstance(self.n_states, bool)
            or not isinstance(self.n_states, int)
            or self.n_states < 2
        ):
            raise ValueError("n_states must be an integer of at least two")
        if states.shape != (len(self.study),) or np.any((states < 0) | (states >= self.n_states)):
            raise ValueError("states must contain one valid label per trial")
        object.__setattr__(self, "states", states)


@dataclass(frozen=True, slots=True)
class FilteredStateProbabilities:
    """Predictive and outcome-updated state probabilities in source row order."""

    predictive: NDArray[np.float64]
    filtered: NDArray[np.float64]

    def __post_init__(self) -> None:
        predictive = _protected_array(self.predictive, dtype=np.float64)
        filtered = _protected_array(self.filtered, dtype=np.float64)
        if predictive.ndim != 2 or filtered.shape != predictive.shape:
            raise ValueError("state-probability arrays must be equally sized matrices")
        if predictive.shape[1] < 2:
            raise ValueError("state probabilities must contain at least two states")
        for name, values in (("predictive", predictive), ("filtered", filtered)):
            if not np.all(np.isfinite(values)) or np.any(values < 0):
                raise ValueError(f"{name} state probabilities must be finite and non-negative")
            if not np.allclose(values.sum(axis=1), 1.0, atol=1e-8):
                raise ValueError(f"{name} state probabilities must sum to one")
        object.__setattr__(self, "predictive", predictive)
        object.__setattr__(self, "filtered", filtered)


@dataclass(frozen=True, slots=True)
class _PosteriorStatistics:
    log_likelihood: float
    state_probabilities: NDArray[np.float64]
    initial_counts: NDArray[np.float64]
    transition_counts: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class GLMHMMFitResult(FitResult):
    """A fitted GLM-HMM with restart, occupancy, and label-order diagnostics."""

    restart_objectives: NDArray[np.float64]
    restart_converged: NDArray[np.bool_]
    restart_messages: tuple[str, ...]
    selected_restart: int
    canonical_permutation: tuple[int, ...]
    state_occupancy: NDArray[np.float64]
    state_separation: float
    label_order_gap: float
    label_ambiguous: bool
    low_occupancy: bool

    def __post_init__(self) -> None:
        FitResult.__post_init__(self)
        objectives = _protected_array(self.restart_objectives, dtype=np.float64)
        converged = _protected_array(self.restart_converged, dtype=np.bool_)
        messages = tuple(self.restart_messages)
        permutation = tuple(self.canonical_permutation)
        occupancy = _protected_array(self.state_occupancy, dtype=np.float64)
        if objectives.ndim != 1 or converged.shape != objectives.shape:
            raise ValueError("restart diagnostics must have one value per restart")
        if len(messages) != len(objectives) or np.any(np.isnan(objectives)):
            raise ValueError("restart messages and non-NaN objectives must align")
        if not 0 <= self.selected_restart < len(objectives):
            raise ValueError("selected_restart must identify one restart")
        if sorted(permutation) != list(range(len(permutation))) or len(permutation) < 2:
            raise ValueError("canonical_permutation must permute every latent state")
        if occupancy.shape != (len(permutation),) or np.any(occupancy < 0):
            raise ValueError("state_occupancy must contain one non-negative value per state")
        if not np.isclose(occupancy.sum(), 1.0, atol=1e-8):
            raise ValueError("state occupancy must sum to one")
        if not np.isfinite(self.state_separation) or self.state_separation < 0:
            raise ValueError("state_separation must be finite and non-negative")
        if not np.isfinite(self.label_order_gap) or self.label_order_gap < 0:
            raise ValueError("label_order_gap must be finite and non-negative")
        object.__setattr__(self, "restart_objectives", objectives)
        object.__setattr__(self, "restart_converged", converged)
        object.__setattr__(self, "restart_messages", messages)
        object.__setattr__(self, "canonical_permutation", permutation)
        object.__setattr__(self, "state_occupancy", occupancy)


@dataclass(frozen=True, slots=True)
class BernoulliGLMHMM(BernoulliHistoryGLM):
    """A stationary-transition HMM with state-specific Bernoulli GLM emissions.

    The initial distribution resets at each subject/session boundary. Transition
    probabilities remain constant across observed trials and sessions. Latent labels are
    canonicalized by increasing values of ``label_by``, with the complete emission vector
    used only as a deterministic tie-breaker.
    """

    n_states: int = 2
    n_restarts: int = 5
    random_seed: int = 0
    label_by: str = "intercept"
    label_tolerance: float = 1e-3
    state_occupancy_warning: float = 0.01
    probability_warning_threshold: float = 1e-4
    stickiness: float = 0.0

    def __post_init__(self) -> None:
        BernoulliHistoryGLM.__post_init__(self)
        if (
            isinstance(self.n_states, bool)
            or not isinstance(self.n_states, int)
            or self.n_states < 2
        ):
            raise ValueError("n_states must be an integer of at least two")
        if (
            isinstance(self.n_restarts, bool)
            or not isinstance(self.n_restarts, int)
            or self.n_restarts < 1
        ):
            raise ValueError("n_restarts must be a positive integer")
        if (
            isinstance(self.random_seed, bool)
            or not isinstance(self.random_seed, int)
            or self.random_seed < 0
        ):
            raise ValueError("random_seed must be a non-negative integer")
        if self.label_by not in self.coefficient_names:
            raise ValueError(
                f"label_by must name an emission coefficient: {self.coefficient_names}"
            )
        if not np.isfinite(self.label_tolerance) or self.label_tolerance < 0:
            raise ValueError("label_tolerance must be finite and non-negative")
        if not 0 < self.state_occupancy_warning < 1:
            raise ValueError("state_occupancy_warning must lie strictly between zero and one")
        if not 0 < self.probability_warning_threshold < 0.5:
            raise ValueError("probability_warning_threshold must lie strictly between zero and 0.5")
        if not np.isfinite(self.stickiness) or self.stickiness < 0:
            raise ValueError("stickiness must be finite and non-negative")

    @property
    def model_name(self) -> str:
        return "bernoulli-glm-hmm"

    @property
    def signature(self) -> str:
        covariates = ",".join(self.covariates)
        return (
            f"{self.model_name}[states={self.n_states};outcome={self.outcome};"
            f"covariates={covariates};choice_lags={self.choice_lags};"
            f"label_by={self.label_by};l2={self.l2};stickiness={self.stickiness}]"
        )

    @property
    def parameter_names(self) -> tuple[str, ...]:
        emissions = tuple(
            f"state[{state}].{coefficient}"
            for state in range(self.n_states)
            for coefficient in self.coefficient_names
        )
        initial = tuple(
            f"initial_logit[state={state}|reference={self.n_states - 1}]"
            for state in range(self.n_states - 1)
        )
        transitions = tuple(
            f"transition_logit[from={source},to={destination}|reference={self.n_states - 1}]"
            for source in range(self.n_states)
            for destination in range(self.n_states - 1)
        )
        return (*emissions, *initial, *transitions)

    @property
    def scored_columns(self) -> tuple[str, ...]:
        return (self.outcome,)

    @property
    def supported_prediction_modes(self) -> tuple[PredictionMode, ...]:
        return (PredictionMode.FILTERED,)

    def parameters_from_components(
        self,
        *,
        initial_probabilities: Sequence[float],
        transition_matrix: Sequence[Sequence[float]],
        emissions: Mapping[str, Sequence[float]],
    ) -> Mapping[str, float]:
        """Validate, canonicalize, and pack natural-scale model components."""

        if set(emissions) != set(self.coefficient_names):
            expected = set(self.coefficient_names)
            observed = set(emissions)
            raise ValueError(
                "emissions must match model coefficients exactly; "
                f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
            )
        emission_matrix = np.column_stack(
            [np.asarray(emissions[name], dtype=np.float64) for name in self.coefficient_names]
        )
        components = GLMHMMParameters(
            initial_probabilities=np.asarray(initial_probabilities, dtype=np.float64),
            transition_matrix=np.asarray(transition_matrix, dtype=np.float64),
            emission_coefficients=emission_matrix,
            coefficient_names=self.coefficient_names,
        )
        if components.n_states != self.n_states:
            raise ValueError(f"components must contain exactly {self.n_states} states")
        canonical, _ = self._canonicalize_components(components)
        values = self._pack_components(canonical)
        return MappingProxyType(dict(zip(self.parameter_names, values.tolist(), strict=True)))

    def parameter_components(
        self,
        parameters: Mapping[str, float] | FitResult,
    ) -> GLMHMMParameters:
        """Decode optimizer coordinates into probabilities and emission coefficients."""

        if isinstance(parameters, FitResult):
            self._validate_fit(parameters)
            vector = parameters.estimates
        else:
            expected = set(self.parameter_names)
            observed = set(parameters)
            if observed != expected:
                raise ValueError(
                    "parameters must match the model exactly; "
                    f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
                )
            try:
                vector = np.asarray(
                    [parameters[name] for name in self.parameter_names], dtype=np.float64
                )
            except (TypeError, ValueError):
                raise ValueError("parameters must contain finite numeric values") from None
            if not np.all(np.isfinite(vector)):
                raise ValueError("parameters must contain finite numeric values")
        return self._unpack_components(vector)

    def simulate(
        self,
        design: Study,
        parameters: Mapping[str, float],
        *,
        seed: int | np.random.Generator,
    ) -> Study:
        """Generate choices without exposing latent-state truth as an observed column."""

        return self.simulate_with_states(design, parameters, seed=seed).study

    def simulate_with_states(
        self,
        design: Study,
        parameters: Mapping[str, float],
        *,
        seed: int | np.random.Generator,
    ) -> GLMHMMSimulation:
        """Generate choices and return latent-state truth in a separate result."""

        components = self.parameter_components(parameters)
        covariates = self._covariate_matrix(design)
        generator = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        choices = np.zeros(len(design), dtype=np.int8)
        states = np.zeros(len(design), dtype=np.int64)
        covariate_end = 1 + len(self.covariates)

        for session_indices in _ordered_session_indices(design):
            generated_history: list[float] = []
            state = int(generator.choice(self.n_states, p=components.initial_probabilities))
            for position, index in enumerate(session_indices):
                if position:
                    state = int(
                        generator.choice(self.n_states, p=components.transition_matrix[state])
                    )
                features = np.empty(len(self.coefficient_names), dtype=np.float64)
                features[0] = 1.0
                features[1:covariate_end] = covariates[index]
                for lag in range(1, self.choice_lags + 1):
                    features[covariate_end + lag - 1] = (
                        generated_history[-lag] if len(generated_history) >= lag else 0.0
                    )
                probability = expit(float(features @ components.emission_coefficients[state]))
                choice = int(generator.binomial(1, probability))
                choices[index] = choice
                states[index] = state
                generated_history.append(2.0 * choice - 1.0)

        columns = {name: design[name] for name in design.columns}
        columns[self.outcome] = choices
        return GLMHMMSimulation(study=Study(columns), states=states, n_states=self.n_states)

    def fit(self, study: Study) -> GLMHMMFitResult:
        """Fit by deterministic multi-start maximum penalized likelihood."""

        outcomes = self._outcomes(study)
        features = self._base_feature_matrix(study, outcomes)
        sessions = _ordered_session_indices(study)

        def objective(vector: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
            return self._objective_gradient(vector, features, outcomes, sessions)

        starts = self._initial_points(features, outcomes)
        bounds = [(-30.0, 30.0)] * len(self.parameter_names)
        results = [
            minimize(
                objective,
                start,
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                options={
                    "maxiter": self.max_iterations,
                    "ftol": self.tolerance,
                    "gtol": self.tolerance,
                },
            )
            for start in starts
        ]
        restart_objectives = np.asarray(
            [float(result.fun) if np.isfinite(result.fun) else np.inf for result in results]
        )
        finite = [index for index, value in enumerate(restart_objectives) if np.isfinite(value)]
        if not finite:
            messages = "; ".join(str(result.message) for result in results)
            raise ModelDataError(f"all GLM-HMM restarts produced non-finite objectives: {messages}")
        successful = [index for index in finite if results[index].success]
        eligible = successful if successful else finite
        selected = min(eligible, key=lambda index: float(restart_objectives[index]))
        raw = np.asarray(results[selected].x, dtype=np.float64)
        canonical, permutation = self._canonicalize_components(self._unpack_components(raw))
        estimates = self._pack_components(canonical)
        value, gradient = objective(estimates)
        hessian = _numerical_hessian(objective, estimates)
        condition = float(np.linalg.cond(hessian))
        covariance = np.linalg.pinv(hessian, hermitian=True)
        standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        gamma = self._posterior_state_probabilities(
            estimates,
            features,
            outcomes,
            sessions,
        )
        occupancy = np.mean(gamma, axis=0)
        separation = _minimum_pairwise_distance(canonical.emission_coefficients)
        label_values = canonical.emission_coefficients[
            :, self.coefficient_names.index(self.label_by)
        ]
        label_gap = float(np.min(np.diff(label_values)))
        probability_values = np.concatenate(
            (canonical.initial_probabilities, canonical.transition_matrix.ravel())
        )
        boundary = bool(
            np.any(np.abs(canonical.emission_coefficients) >= self.coefficient_warning_threshold)
            or np.any(probability_values <= self.probability_warning_threshold)
            or np.any(probability_values >= 1.0 - self.probability_warning_threshold)
        )
        chosen = results[selected]
        diagnostics = FitDiagnostics(
            converged=bool(chosen.success),
            optimizer=f"L-BFGS-B ({self.n_restarts} deterministic restarts)",
            status=int(chosen.status),
            message=str(chosen.message),
            n_iterations=int(chosen.nit),
            objective=float(value),
            gradient_norm=float(np.linalg.norm(gradient)),
            hessian_condition=condition,
            boundary_estimate=boundary,
        )
        return GLMHMMFitResult(
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
            canonical_permutation=permutation,
            state_occupancy=occupancy,
            state_separation=separation,
            label_order_gap=label_gap,
            label_ambiguous=bool(label_gap <= self.label_tolerance),
            low_occupancy=bool(np.any(occupancy < self.state_occupancy_warning)),
        )

    def predict(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> Prediction:
        """Return one-step-ahead choice probabilities under filtered latent state."""

        prediction_mode = self._prediction_mode(mode)
        self._validate_fit(fit)
        outcomes = self._outcomes(study)
        features = self._base_feature_matrix(study, outcomes)
        components = self.parameter_components(fit)
        state_probabilities = self._filtered_state_probabilities(
            features,
            outcomes,
            _ordered_session_indices(study),
            components,
        )
        emission_probabilities = expit(features @ components.emission_coefficients.T)
        probability = np.sum(state_probabilities.predictive * emission_probabilities, axis=1)
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
        """Score observed choices under sequential filtered predictions."""

        outcomes = self._outcomes(study)
        prediction = self.predict(study, fit, mode=mode)
        scores = outcomes * np.log(prediction.probability)
        scores += (1.0 - outcomes) * np.log1p(-prediction.probability)
        return _protected_array(scores, dtype=np.float64)

    def state_probabilities(
        self,
        study: Study,
        fit: FitResult,
    ) -> FilteredStateProbabilities:
        """Return predictive and outcome-updated state probabilities."""

        self._validate_fit(fit)
        outcomes = self._outcomes(study)
        features = self._base_feature_matrix(study, outcomes)
        return self._filtered_state_probabilities(
            features,
            outcomes,
            _ordered_session_indices(study),
            self.parameter_components(fit),
        )

    def state_recovery(
        self,
        simulation: GLMHMMSimulation,
        fit: FitResult,
        *,
        ambiguity_tolerance: float = 0.05,
    ) -> LatentStateAlignment:
        """Align outcome-filtered state probabilities to separately retained truth."""

        if not isinstance(simulation, GLMHMMSimulation):
            raise TypeError("simulation must be a GLMHMMSimulation")
        if simulation.n_states != self.n_states:
            raise ValueError("simulation and model must contain the same number of states")
        filtered = self.state_probabilities(simulation.study, fit).filtered
        return align_latent_states(
            simulation.states,
            filtered,
            ambiguity_tolerance=ambiguity_tolerance,
        )

    def _initial_points(
        self,
        features: NDArray[np.float64],
        outcomes: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], ...]:
        penalty = np.zeros(features.shape[1], dtype=np.float64)
        penalty[1:] = self.l2

        def glm_objective(coefficients: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
            linear = features @ coefficients
            loss = np.logaddexp(0.0, linear).sum() - outcomes @ linear
            loss += 0.5 * float(np.sum(penalty * coefficients**2))
            gradient = features.T @ (expit(linear) - outcomes) + penalty * coefficients
            return float(loss), gradient

        base_result = minimize(
            glm_objective,
            np.zeros(features.shape[1]),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": self.max_iterations, "ftol": self.tolerance},
        )
        base = np.asarray(base_result.x, dtype=np.float64)
        persistent = np.full((self.n_states, self.n_states), 0.1 / (self.n_states - 1))
        np.fill_diagonal(persistent, 0.9)
        transition_logits = _encode_probability_rows(persistent).ravel()
        generator = np.random.default_rng(self.random_seed)
        label_index = self.coefficient_names.index(self.label_by)
        starts: list[NDArray[np.float64]] = []
        for restart in range(self.n_restarts):
            emissions = np.tile(base, (self.n_states, 1))
            emissions[:, label_index] += np.linspace(-0.75, 0.75, self.n_states)
            scale = 0.05 if restart == 0 else 0.35
            emissions += generator.normal(0.0, scale, emissions.shape)
            initial_logits = generator.normal(0.0, scale, self.n_states - 1)
            transitions = transition_logits + generator.normal(
                0.0, scale, self.n_states * (self.n_states - 1)
            )
            starts.append(np.concatenate((emissions.ravel(), initial_logits, transitions)))
        return tuple(starts)

    def _objective_gradient(
        self,
        vector: NDArray[np.float64],
        features: NDArray[np.float64],
        outcomes: NDArray[np.float64],
        sessions: tuple[tuple[int, ...], ...],
    ) -> tuple[float, NDArray[np.float64]]:
        components = self._unpack_components(vector)
        emissions = components.emission_coefficients
        initial = components.initial_probabilities
        transition = components.transition_matrix
        linear = features @ emissions.T
        emission_log_probability = outcomes[:, None] * -np.logaddexp(0.0, -linear) + (
            1.0 - outcomes[:, None]
        ) * -np.logaddexp(0.0, linear)
        log_initial = np.log(initial)
        log_transition = np.log(transition)
        posterior = _forward_backward(
            log_initial,
            log_transition,
            emission_log_probability,
            sessions,
        )
        gamma = posterior.state_probabilities

        emission_gradient = np.empty_like(emissions)
        probabilities = expit(linear)
        for state in range(self.n_states):
            emission_gradient[state] = features.T @ (
                gamma[:, state] * (probabilities[:, state] - outcomes)
            )
        penalty = np.zeros_like(emissions)
        penalty[:, 1:] = self.l2 * emissions[:, 1:]
        loss = -posterior.log_likelihood + 0.5 * self.l2 * float(np.sum(emissions[:, 1:] ** 2))
        emission_gradient += penalty
        initial_gradient = len(sessions) * initial[:-1] - posterior.initial_counts[:-1]
        departures = posterior.transition_counts.sum(axis=1)
        transition_gradient = (
            departures[:, None] * transition[:, :-1] - posterior.transition_counts[:, :-1]
        )
        if self.stickiness:
            # A sticky Dirichlet prior adds kappa pseudo-counts to self-transitions.
            # Constants independent of the parameters are omitted from the MAP objective.
            loss -= self.stickiness * float(np.sum(np.log(np.diag(transition))))
            sticky_gradient = self.stickiness * transition[:, :-1]
            for state in range(self.n_states - 1):
                sticky_gradient[state, state] -= self.stickiness
            transition_gradient += sticky_gradient
        gradient = np.concatenate(
            (emission_gradient.ravel(), initial_gradient, transition_gradient.ravel())
        )
        return float(loss), gradient

    def _posterior_state_probabilities(
        self,
        vector: NDArray[np.float64],
        features: NDArray[np.float64],
        outcomes: NDArray[np.float64],
        sessions: tuple[tuple[int, ...], ...],
    ) -> NDArray[np.float64]:
        components = self._unpack_components(vector)
        linear = features @ components.emission_coefficients.T
        emission = outcomes[:, None] * -np.logaddexp(0.0, -linear) + (
            1.0 - outcomes[:, None]
        ) * -np.logaddexp(0.0, linear)
        log_initial = np.log(components.initial_probabilities)
        log_transition = np.log(components.transition_matrix)
        return _forward_backward(
            log_initial,
            log_transition,
            emission,
            sessions,
        ).state_probabilities

    def _filtered_state_probabilities(
        self,
        features: NDArray[np.float64],
        outcomes: NDArray[np.float64],
        sessions: tuple[tuple[int, ...], ...],
        components: GLMHMMParameters,
    ) -> FilteredStateProbabilities:
        linear = features @ components.emission_coefficients.T
        emission_log = outcomes[:, None] * -np.logaddexp(0.0, -linear) + (
            1.0 - outcomes[:, None]
        ) * -np.logaddexp(0.0, linear)
        predictive = np.empty((len(outcomes), self.n_states), dtype=np.float64)
        filtered = np.empty_like(predictive)
        for session_indices in sessions:
            prior = np.asarray(components.initial_probabilities)
            for index in session_indices:
                predictive[index] = prior
                log_weight = np.log(prior) + emission_log[index]
                posterior = np.exp(log_weight - logsumexp(log_weight))
                filtered[index] = posterior
                prior = posterior @ components.transition_matrix
        return FilteredStateProbabilities(predictive=predictive, filtered=filtered)

    def _canonicalize_components(
        self,
        components: GLMHMMParameters,
    ) -> tuple[GLMHMMParameters, tuple[int, ...]]:
        label_index = self.coefficient_names.index(self.label_by)
        other = tuple(index for index in range(len(self.coefficient_names)) if index != label_index)
        permutation = tuple(
            sorted(
                range(self.n_states),
                key=lambda state: (
                    float(components.emission_coefficients[state, label_index]),
                    *(float(components.emission_coefficients[state, index]) for index in other),
                ),
            )
        )
        indices = np.asarray(permutation, dtype=np.intp)
        canonical = GLMHMMParameters(
            initial_probabilities=components.initial_probabilities[indices],
            transition_matrix=components.transition_matrix[np.ix_(indices, indices)],
            emission_coefficients=components.emission_coefficients[indices],
            coefficient_names=self.coefficient_names,
        )
        return canonical, permutation

    def _pack_components(self, components: GLMHMMParameters) -> NDArray[np.float64]:
        return np.concatenate(
            (
                components.emission_coefficients.ravel(),
                _encode_probability_vector(components.initial_probabilities),
                _encode_probability_rows(components.transition_matrix).ravel(),
            )
        )

    def _unpack_components(self, vector: Sequence[float]) -> GLMHMMParameters:
        values = np.asarray(vector, dtype=np.float64)
        if values.shape != (len(self.parameter_names),) or not np.all(np.isfinite(values)):
            raise ValueError("parameter vector has the wrong shape or contains non-finite values")
        n_coefficients = len(self.coefficient_names)
        emission_end = self.n_states * n_coefficients
        initial_end = emission_end + self.n_states - 1
        emissions = values[:emission_end].reshape(self.n_states, n_coefficients)
        initial = _decode_reference_logits(values[emission_end:initial_end])
        transition_logits = values[initial_end:].reshape(self.n_states, self.n_states - 1)
        transition = np.vstack([_decode_reference_logits(row) for row in transition_logits])
        return GLMHMMParameters(
            initial_probabilities=initial,
            transition_matrix=transition,
            emission_coefficients=emissions,
            coefficient_names=self.coefficient_names,
        )


def _decode_reference_logits(logits: Sequence[float]) -> NDArray[np.float64]:
    coordinates = np.append(np.asarray(logits, dtype=np.float64), 0.0)
    return np.exp(coordinates - logsumexp(coordinates))


def _encode_probability_vector(probabilities: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.log(probabilities[:-1]) - np.log(probabilities[-1])


def _encode_probability_rows(probabilities: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.log(probabilities[:, :-1]) - np.log(probabilities[:, -1:])


def _minimum_pairwise_distance(values: NDArray[np.float64]) -> float:
    return float(
        min(
            np.linalg.norm(values[left] - values[right])
            for left, right in combinations(range(len(values)), 2)
        )
    )


def _forward_backward(
    log_initial: NDArray[np.float64],
    log_transition: NDArray[np.float64],
    emission_log_probability: NDArray[np.float64],
    sessions: tuple[tuple[int, ...], ...],
) -> _PosteriorStatistics:
    n_states = len(log_initial)
    gamma = np.empty_like(emission_log_probability)
    initial_counts = np.zeros(n_states, dtype=np.float64)
    transition_counts = np.zeros((n_states, n_states), dtype=np.float64)
    log_likelihood = 0.0
    for session_indices in sessions:
        indices = np.asarray(session_indices, dtype=np.intp)
        emission = emission_log_probability[indices]
        n_trials = len(indices)
        alpha = np.empty((n_trials, n_states), dtype=np.float64)
        beta = np.zeros_like(alpha)
        alpha[0] = log_initial + emission[0]
        for trial in range(1, n_trials):
            alpha[trial] = emission[trial] + logsumexp(
                alpha[trial - 1, :, None] + log_transition,
                axis=0,
            )
        session_likelihood = float(logsumexp(alpha[-1]))
        log_likelihood += session_likelihood
        for trial in range(n_trials - 2, -1, -1):
            beta[trial] = logsumexp(
                log_transition + emission[trial + 1][None, :] + beta[trial + 1][None, :],
                axis=1,
            )
        session_gamma = np.exp(alpha + beta - session_likelihood)
        gamma[indices] = session_gamma
        initial_counts += session_gamma[0]
        for trial in range(1, n_trials):
            transition_counts += np.exp(
                alpha[trial - 1, :, None]
                + log_transition
                + emission[trial][None, :]
                + beta[trial][None, :]
                - session_likelihood
            )
    return _PosteriorStatistics(
        log_likelihood=log_likelihood,
        state_probabilities=gamma,
        initial_counts=initial_counts,
        transition_counts=transition_counts,
    )


def _numerical_hessian(
    objective: Callable[
        [NDArray[np.float64]],
        tuple[float, NDArray[np.float64]],
    ],
    estimates: NDArray[np.float64],
) -> NDArray[np.float64]:
    hessian = np.empty((len(estimates), len(estimates)), dtype=np.float64)
    for column in range(len(estimates)):
        step = 1e-5 * (1.0 + abs(float(estimates[column])))
        positive = estimates.copy()
        negative = estimates.copy()
        positive[column] += step
        negative[column] -= step
        _, positive_gradient = objective(positive)
        _, negative_gradient = objective(negative)
        hessian[:, column] = (positive_gradient - negative_gradient) / (2.0 * step)
    return 0.5 * (hessian + hessian.T)
