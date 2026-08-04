"""Full-posterior PyMC inference for stationary and session-dynamic GLM-HMMs.

The discrete state sequence is marginalized with the HMM forward recursion.  NUTS therefore
samples only continuous emission, transition, hierarchy, and variance-component parameters.
Symmetric state priors deliberately leave the posterior permutation-invariant; complete draws
are relabelled after sampling by the model's declared emission coordinate, and path crossings
remain an explicit ambiguity diagnostic.
"""

from __future__ import annotations

import importlib
import importlib.metadata
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit

from behavio.contracts.estimator import FitResult, Prediction, PredictionMode
from behavio.contracts.posterior import (
    PosteriorCentre,
    posterior_log_predictive_density,
    posterior_point_summary,
)
from behavio.models._kernels.bernoulli import ordered_session_indices
from behavio.models.glm_hmm import BernoulliGLMHMM
from behavio.models.hierarchical_session_dynamic_glm_hmm import (
    HierarchicalSessionDynamicBernoulliGLMHMM,
    _population_structure,
)
from behavio.models.lab_hierarchical_session_dynamic_glm_hmm import (
    LabHierarchicalSessionDynamicBernoulliGLMHMM,
    _lab_structure,
)
from behavio.models.session_dynamic_glm_hmm import SessionDynamicBernoulliGLMHMM
from behavio.posterior.result import (
    POSTERIOR_GROUPS,
    PosteriorGroup,
    PosteriorResult,
    PosteriorVariable,
    posterior_result_from_arviz,
)
from behavio.posterior.simulation_based_calibration import SBCSimulation
from behavio.pymc_backend import PyMCBackendError, _import_pymc
from behavio.trials import Study

DynamicGLMHMM = (
    SessionDynamicBernoulliGLMHMM
    | HierarchicalSessionDynamicBernoulliGLMHMM
    | LabHierarchicalSessionDynamicBernoulliGLMHMM
)


@dataclass(frozen=True, slots=True)
class _PathStructure:
    """Backend-neutral indexing for the three session-dynamic hierarchy depths."""

    sessions: tuple[tuple[int, ...], ...]
    labels: tuple[str, ...]
    population_orders: NDArray[np.int64]
    population_index: NDArray[np.intp]
    subjects: tuple[Any, ...]
    subject_blocks: tuple[tuple[int, ...], ...]
    labs: tuple[Any, ...] = ()
    lab_blocks: tuple[tuple[int, ...], ...] = ()
    session_lab_index: NDArray[np.intp] | None = None


@dataclass(frozen=True, slots=True)
class PyMCBernoulliGLMHMM:
    """Proper-prior Bayesian inference for the complete Bernoulli GLM-HMM family.

    ``model`` supplies the observation, design, session-boundary, and state-label semantics.
    This wrapper supplies a separate normalized Bayesian model; it does not reinterpret the
    base estimator's ridge penalties or bounded empirical-Bayes scales as posterior priors.

    The stationary model samples emission coefficients, the initial simplex, transition rows,
    and (when declared) centred transition-regression effects.  Session-dynamic models also
    sample every session transition matrix, every Gaussian path, their scale parameters, and
    the transition concentration.  Population and laboratory variants use non-centred
    population, lab, and subject innovations, so no MAP layer is held fixed.

    Prediction and pointwise scoring integrate the filtered one-step density over posterior
    draws.  Dynamic posteriors currently score the fitted subject-session blocks: propagating
    full posterior paths into unseen sessions, subjects, or laboratories is a separate
    posterior-predictive design problem and is refused instead of substituting a plug-in path.
    """

    model: BernoulliGLMHMM
    emission_prior_scale: float = 2.5
    initial_prior_concentration: float = 1.0
    transition_prior_concentration: float = 1.0
    transition_effect_prior_scale: float = 1.0
    population_step_prior_scale: float = 0.5
    lab_initial_prior_scale: float = 0.5
    lab_step_prior_scale: float = 0.25
    subject_initial_prior_scale: float = 0.75
    subject_step_prior_scale: float = 0.5
    session_step_prior_scale: float = 0.5
    session_transition_concentration_prior_mean: float = 10.0
    draws: int = 1_000
    tune: int = 1_000
    chains: int = 4
    cores: int = 1
    target_accept: float = 0.9
    seed: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.model, BernoulliGLMHMM):
            raise TypeError("model must be a BernoulliGLMHMM")
        for value, name in (
            (self.emission_prior_scale, "emission_prior_scale"),
            (self.initial_prior_concentration, "initial_prior_concentration"),
            (self.transition_prior_concentration, "transition_prior_concentration"),
            (self.transition_effect_prior_scale, "transition_effect_prior_scale"),
            (self.population_step_prior_scale, "population_step_prior_scale"),
            (self.lab_initial_prior_scale, "lab_initial_prior_scale"),
            (self.lab_step_prior_scale, "lab_step_prior_scale"),
            (self.subject_initial_prior_scale, "subject_initial_prior_scale"),
            (self.subject_step_prior_scale, "subject_step_prior_scale"),
            (self.session_step_prior_scale, "session_step_prior_scale"),
            (
                self.session_transition_concentration_prior_mean,
                "session_transition_concentration_prior_mean",
            ),
        ):
            if not np.isfinite(value) or value <= 0:
                raise PyMCBackendError(f"{name} must be finite and positive")
        for value, name in (
            (self.draws, "draws"),
            (self.tune, "tune"),
            (self.chains, "chains"),
            (self.cores, "cores"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise PyMCBackendError(f"{name} must be a positive integer")
        if self.chains < 2:
            raise PyMCBackendError("chains must be at least two for cross-chain diagnostics")
        if self.cores > self.chains:
            raise PyMCBackendError("cores cannot exceed chains")
        if not np.isfinite(self.target_accept) or not 0.5 <= self.target_accept < 1.0:
            raise PyMCBackendError("target_accept must be finite and in [0.5, 1)")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise PyMCBackendError("seed must be a non-negative integer")

    @property
    def model_name(self) -> str:
        return f"bayesian-{self.model.model_name}"

    @property
    def signature(self) -> str:
        priors = ",".join(f"{key}={value}" for key, value in self.prior_specification.items())
        return f"{self.model_name}[base={self.model.signature};priors={priors}]"

    @property
    def scored_columns(self) -> tuple[str, ...]:
        return self.model.scored_columns

    @property
    def required_task_columns(self) -> tuple[str, ...]:
        return self.model.required_task_columns

    @property
    def supported_prediction_modes(self) -> tuple[PredictionMode, ...]:
        return (PredictionMode.FILTERED,)

    @property
    def backend_name(self) -> str:
        return "pymc.NUTS/marginalized-HMM"

    @property
    def backend_config(self) -> MappingProxyType[str, Any]:
        return MappingProxyType(
            {
                "draws": self.draws,
                "tune": self.tune,
                "chains": self.chains,
                "cores": self.cores,
                "target_accept": self.target_accept,
                "seed": self.seed,
                "nuts_sampler": "pymc",
                "init": "jitter+adapt_diag",
            }
        )

    @property
    def prior_specification(self) -> MappingProxyType[str, float]:
        values = {
            "emission_normal_scale": self.emission_prior_scale,
            "initial_dirichlet_concentration": self.initial_prior_concentration,
            "transition_dirichlet_concentration": self.transition_prior_concentration,
        }
        if self.model.is_dynamic:
            values["transition_effect_normal_scale"] = self.transition_effect_prior_scale
        if isinstance(self.model, SessionDynamicBernoulliGLMHMM):
            values.update(
                {
                    "session_step_halfnormal_scale": self.session_step_prior_scale,
                    "session_transition_concentration_exponential_mean": (
                        self.session_transition_concentration_prior_mean
                    ),
                }
            )
        if isinstance(self.model, HierarchicalSessionDynamicBernoulliGLMHMM):
            values.update(
                {
                    "population_step_halfnormal_scale": self.population_step_prior_scale,
                    "subject_initial_halfnormal_scale": self.subject_initial_prior_scale,
                    "subject_step_halfnormal_scale": self.subject_step_prior_scale,
                }
            )
        if isinstance(self.model, LabHierarchicalSessionDynamicBernoulliGLMHMM):
            values.update(
                {
                    "lab_initial_halfnormal_scale": self.lab_initial_prior_scale,
                    "lab_step_halfnormal_scale": self.lab_step_prior_scale,
                }
            )
        return MappingProxyType(values)

    def sample(self, study: Study) -> PosteriorResult:
        """Sample the state-marginalized posterior and attach filtered evidence."""

        if not isinstance(study, Study):
            raise TypeError("study must be a Study")
        outcomes = np.asarray(self.model.outcomes(study), dtype=np.int8)
        features = np.asarray(self.model.design_matrix(study), dtype=np.float64)
        structure = self._path_structure(study)
        pymc = _import_pymc()
        tensor = importlib.import_module("pytensor.tensor")
        inference_version = importlib.metadata.version("pymc")
        sampling_seed, predictive_seed = (
            int(value) for value in np.random.SeedSequence(self.seed).generate_state(2)
        )
        coords = self._coords(study, structure)

        with pymc.Model(coords=coords):
            initial = pymc.Dirichlet(
                "initial_probabilities",
                a=np.full(self.model.n_states, self.initial_prior_concentration),
                dims="state",
            )
            transition_alpha = np.full(
                (self.model.n_states, self.model.n_states),
                self.transition_prior_concentration,
            )
            if not isinstance(self.model, SessionDynamicBernoulliGLMHMM):
                transition_alpha += self.model.stickiness * np.eye(self.model.n_states)
            global_transition = pymc.Dirichlet(
                "global_transition_matrix",
                a=transition_alpha,
                dims=("source_state", "destination_state"),
            )

            if structure is None:
                emissions = pymc.Normal(
                    "emission_coefficients",
                    mu=0.0,
                    sigma=self.emission_prior_scale,
                    dims=("state", "coefficient"),
                )
                parameter_names = [
                    "initial_probabilities",
                    "global_transition_matrix",
                    "emission_coefficients",
                ]
                transitions: Any = global_transition
                if self.model.is_dynamic:
                    raw_effect = pymc.Normal(
                        "transition_effect_raw",
                        mu=0.0,
                        sigma=self.transition_effect_prior_scale,
                        dims=("source_state", "destination_state", "transition_coefficient"),
                    )
                    effects = pymc.Deterministic(
                        "transition_effects",
                        raw_effect - tensor.mean(raw_effect, axis=1, keepdims=True),
                        dims=("source_state", "destination_state", "transition_coefficient"),
                    )
                    transition_features = self.model.transition_design_matrix(study)
                    offsets = tensor.tensordot(transition_features, effects, axes=((1,), (2,)))
                    transitions = tensor.special.softmax(
                        tensor.log(global_transition)[None, :, :] + offsets,
                        axis=2,
                    )
                    parameter_names.append("transition_effects")
                emission_by_session = None
            else:
                (
                    emission_by_session,
                    parameter_names,
                ) = self._dynamic_emissions(pymc, tensor, structure)
                session_concentration = pymc.Exponential(
                    "session_transition_concentration",
                    lam=1.0 / self.session_transition_concentration_prior_mean,
                )
                transitions = pymc.Dirichlet(
                    "session_transition_matrices",
                    a=session_concentration * global_transition[None, :, :] + 1.0,
                    dims=("path_session", "source_state", "destination_state"),
                )
                parameter_names = [
                    "initial_probabilities",
                    "global_transition_matrix",
                    "session_transition_matrices",
                    *parameter_names,
                    "session_transition_concentration",
                ]
                emissions = None

            graph = _forward_graph(
                tensor,
                features,
                outcomes,
                ordered_session_indices(study),
                initial,
                transitions,
                stationary_emissions=emissions,
                session_emissions=emission_by_session,
            )
            pymc.Deterministic("choice_probability", graph[0], dims="trial")
            pymc.Deterministic("choice_log_likelihood", graph[1], dims="trial")
            pymc.Deterministic(
                "predictive_state_probability",
                graph[2],
                dims=("trial", "state"),
            )
            pymc.Deterministic(
                "filtered_state_probability",
                graph[3],
                dims=("trial", "state"),
            )
            pymc.Potential("marginalized_choice_likelihood", tensor.sum(graph[1]))
            inference_data = pymc.sample(
                draws=self.draws,
                tune=self.tune,
                chains=self.chains,
                cores=self.cores,
                target_accept=self.target_accept,
                random_seed=sampling_seed,
                nuts_sampler="pymc",
                init="jitter+adapt_diag",
                progressbar=False,
                return_inferencedata=True,
            )

        _canonicalize_draws(
            inference_data,
            label_coefficient=self.model.coefficient_names.index(self.model.label_by),
            tolerance=self.model.label_tolerance,
            dynamic=structure is not None,
        )
        result = posterior_result_from_arviz(
            inference_data,
            model_name=self.model_name,
            model_signature=self.signature,
            inference_library="PyMC",
            inference_library_version=inference_version,
            parameter_names=tuple(parameter_names),
        )
        probability = result["posterior"]["choice_probability"].values
        log_likelihood = result["posterior"]["choice_log_likelihood"].values
        predictive = np.random.default_rng(predictive_seed).binomial(1, probability).astype(np.int8)
        result = _attach_evidence(
            result,
            study,
            outcome=self.model.outcome,
            log_likelihood=log_likelihood,
            posterior_predictive=predictive,
            features=features,
            structure=structure,
        )
        ambiguity = result["posterior"]["label_ambiguous"].values
        attrs = {
            **dict(result.attrs),
            "backend": self.backend_name,
            "backend_config": dict(self.backend_config),
            "base_model_signature": self.model.signature,
            "state_sequence": "analytically marginalized by session-blocked forward recursion",
            "prediction_mode": "filtered",
            "posterior_predictive_conditioning": (
                "one-step choices conditional on observed choices earlier in each session"
            ),
            "prior_specification": dict(self.prior_specification),
            "all_declared_priors_normalized": True,
            "hierarchy_parameterization": "non-centred Gaussian innovations",
            "label_policy": (
                f"post-hoc whole-draw ordering by increasing {self.model.label_by!r}; "
                "transition rows and columns and every state-indexed path are permuted together"
            ),
            "label_ambiguous_fraction": float(np.mean(ambiguity)),
            "label_ambiguity_is_retained": True,
            "dynamic_prediction_scope": (
                "fitted subject-session blocks only"
                if structure is not None
                else "any task-compatible study"
            ),
            "scored_columns": list(self.scored_columns),
            "trial_coordinate": "study source-row position",
        }
        return PosteriorResult(
            model_name=result.model_name,
            model_signature=result.model_signature,
            inference_library=result.inference_library,
            inference_library_version=result.inference_library_version,
            parameter_names=result.parameter_names,
            groups=result.groups,
            attrs=attrs,
        )

    def sample_with_seed(self, study: Study, seed: int) -> PosteriorResult:
        """SBC-compatible sampling with replicate-specific entropy."""

        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise PyMCBackendError("seed must be a non-negative integer")
        return replace(self, seed=seed).sample(study)

    def predict(
        self,
        study: Study,
        posterior: PosteriorResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> Prediction:
        """Integrate source-ordered filtered probabilities over posterior draws."""

        self._prediction_mode(mode)
        probabilities, _ = self._draw_scores(study, posterior)
        probability = np.mean(probabilities, axis=0)
        epsilon = np.finfo(np.float64).eps
        probability = np.clip(probability, epsilon, 1.0 - epsilon)
        return Prediction(
            probability=probability,
            linear_predictor=np.log(probability) - np.log1p(-probability),
            mode=PredictionMode.FILTERED,
        )

    def pointwise_log_prob(
        self,
        study: Study,
        posterior: PosteriorResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> NDArray[np.float64]:
        """Return posterior-integrated one-step log predictive densities."""

        self._prediction_mode(mode)
        _, scores = self._draw_scores(study, posterior)
        return posterior_log_predictive_density(scores)

    def point_summary(
        self,
        posterior: PosteriorResult,
        *,
        converged: bool,
        centre: PosteriorCentre = PosteriorCentre.MEAN,
    ) -> FitResult:
        self._validate_posterior(posterior)
        return posterior_point_summary(posterior, converged=converged, centre=centre)

    def prior_predictive_simulation(self, design: Study, seed: int) -> SBCSimulation:
        """Draw from the normalized prior joint and retain labelled SBC truth."""

        if not isinstance(design, Study):
            raise TypeError("design must be a Study")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise PyMCBackendError("seed must be a non-negative integer")
        generator = np.random.default_rng(seed)
        states = self.model.n_states
        coefficients = len(self.model.coefficient_names)
        initial = generator.dirichlet(np.full(states, self.initial_prior_concentration))
        alpha = np.full((states, states), self.transition_prior_concentration)
        if not isinstance(self.model, SessionDynamicBernoulliGLMHMM):
            alpha += self.model.stickiness * np.eye(states)
        transition = np.stack([generator.dirichlet(row) for row in alpha])
        emissions = generator.normal(0.0, self.emission_prior_scale, (states, coefficients))
        transition_effects = None
        if self.model.is_dynamic:
            raw = generator.normal(
                0.0,
                self.transition_effect_prior_scale,
                (states, states, len(self.model.transition_coefficient_names)),
            )
            transition_effects = raw - raw.mean(axis=1, keepdims=True)
        base_parameters = self.model.parameters_from_components(
            initial_probabilities=initial,
            transition_matrix=transition,
            emissions={
                name: emissions[:, index] for index, name in enumerate(self.model.coefficient_names)
            },
            transition_coefficients=(
                None
                if transition_effects is None
                else {
                    name: transition_effects[:, :, index]
                    for index, name in enumerate(self.model.transition_coefficient_names)
                }
            ),
        )
        canonical = self.model.parameter_components(base_parameters)
        truth: dict[str, Any] = {
            "initial_probabilities": canonical.initial_probabilities,
            "global_transition_matrix": canonical.transition_matrix,
        }
        if not isinstance(self.model, SessionDynamicBernoulliGLMHMM):
            truth["emission_coefficients"] = canonical.emission_coefficients
            if transition_effects is not None:
                truth["transition_effects"] = canonical.transition_effects
            study = self.model.simulate(design, base_parameters, seed=generator)
            return SBCSimulation(study=study, truth=truth)

        dynamic_model, scales = self._draw_dynamic_hyperparameters(generator)
        simulation = dynamic_model.simulate_with_trajectories(
            design,
            base_parameters,
            seed=generator,
        )
        truth.update(scales)
        truth["session_transition_concentration"] = dynamic_model.transition_concentration
        truth["session_transition_matrices"] = (
            simulation.transition_matrices
            if hasattr(simulation, "transition_matrices")
            else simulation.session_transition_matrices
        )
        truth["session_emission_coefficients"] = (
            simulation.emission_coefficients
            if hasattr(simulation, "emission_coefficients")
            else simulation.session_emission_coefficients
        )
        if hasattr(simulation, "population_emission_coefficients"):
            truth["population_emission_coefficients"] = simulation.population_emission_coefficients
        if hasattr(simulation, "lab_deviation_coefficients"):
            truth["lab_deviation_coefficients"] = simulation.lab_deviation_coefficients
        structure = self._path_structure(simulation.study)
        if isinstance(self.model, HierarchicalSessionDynamicBernoulliGLMHMM):
            assert structure is not None
            center = np.asarray(truth["population_emission_coefficients"])[
                structure.population_index
            ]
            if "lab_deviation_coefficients" in truth:
                assert structure.session_lab_index is not None
                center = (
                    center
                    + np.asarray(truth["lab_deviation_coefficients"])[structure.session_lab_index]
                )
            truth["subject_deviation_coefficients"] = (
                np.asarray(truth["session_emission_coefficients"]) - center
            )
        permutation = _path_permutation(
            np.asarray(truth["session_emission_coefficients"]),
            self.model.coefficient_names.index(self.model.label_by),
        )
        for name in (
            "initial_probabilities",
            "global_transition_matrix",
            "session_transition_matrices",
            "session_emission_coefficients",
            "population_emission_coefficients",
            "lab_deviation_coefficients",
            "subject_deviation_coefficients",
        ):
            if name in truth:
                truth[name] = _permute_truth(name, np.asarray(truth[name]), permutation)
        return SBCSimulation(study=simulation.study, truth=truth)

    def _coords(self, study: Study, structure: _PathStructure | None) -> dict[str, Any]:
        state = np.arange(self.model.n_states, dtype=np.int64)
        coords: dict[str, Any] = {
            "trial": np.arange(len(study), dtype=np.int64),
            "state": state,
            "source_state": state,
            "destination_state": state,
            "coefficient": np.asarray(self.model.coefficient_names),
        }
        if self.model.is_dynamic:
            coords["transition_coefficient"] = np.asarray(self.model.transition_coefficient_names)
        if structure is not None:
            coords["path_session"] = np.asarray(structure.labels)
            if len(structure.population_orders):
                coords["population_order"] = structure.population_orders
            if structure.subjects:
                coords["subject"] = _object_coordinate(structure.subjects)
                steps = len(structure.sessions) - len(structure.subjects)
                if steps:
                    coords["subject_step"] = np.arange(steps, dtype=np.int64)
            if structure.labs:
                coords["lab"] = _object_coordinate(structure.labs)
                lab_points = sum(len(blocks) for blocks in structure.lab_blocks)
                lab_steps = lab_points - len(structure.labs)
                coords["lab_path"] = np.arange(lab_points, dtype=np.int64)
                if lab_steps:
                    coords["lab_step"] = np.arange(lab_steps, dtype=np.int64)
            if len(structure.population_orders) > 1:
                coords["population_step"] = structure.population_orders[1:]
            if (
                not isinstance(self.model, HierarchicalSessionDynamicBernoulliGLMHMM)
                and len(structure.sessions) > 1
            ):
                coords["session_step"] = np.arange(len(structure.sessions) - 1)
        return coords

    def _dynamic_emissions(
        self,
        pymc: Any,
        tensor: Any,
        structure: _PathStructure,
    ) -> tuple[Any, list[str]]:
        shape_dims = ("state", "coefficient")
        if not isinstance(self.model, HierarchicalSessionDynamicBernoulliGLMHMM):
            initial = pymc.Normal(
                "emission_initial",
                mu=0.0,
                sigma=self.emission_prior_scale,
                dims=shape_dims,
            )
            step_scale = pymc.HalfNormal(
                "emission_step_scale",
                sigma=self.session_step_prior_scale,
            )
            if len(structure.sessions) > 1:
                raw = pymc.Normal(
                    "emission_step_raw",
                    mu=0.0,
                    sigma=1.0,
                    dims=("session_step", *shape_dims),
                )
                path = tensor.concatenate(
                    (
                        initial[None, :, :],
                        initial[None, :, :] + tensor.cumsum(raw * step_scale, axis=0),
                    ),
                    axis=0,
                )
            else:
                path = initial[None, :, :]
            emissions = pymc.Deterministic(
                "session_emission_coefficients",
                path,
                dims=("path_session", *shape_dims),
            )
            return emissions, ["session_emission_coefficients", "emission_step_scale"]

        population_initial = pymc.Normal(
            "population_emission_initial",
            mu=0.0,
            sigma=self.emission_prior_scale,
            dims=shape_dims,
        )
        population_scale = pymc.HalfNormal(
            "population_emission_step_scale",
            sigma=self.population_step_prior_scale,
        )
        if len(structure.population_orders) > 1:
            population_raw = pymc.Normal(
                "population_emission_step_raw",
                mu=0.0,
                sigma=1.0,
                dims=("population_step", *shape_dims),
            )
            population_path = tensor.concatenate(
                (
                    population_initial[None, :, :],
                    population_initial[None, :, :]
                    + tensor.cumsum(population_raw * population_scale, axis=0),
                ),
                axis=0,
            )
        else:
            population_path = population_initial[None, :, :]
        population = pymc.Deterministic(
            "population_emission_coefficients",
            population_path,
            dims=("population_order", *shape_dims),
        )

        subject_initial_scale = pymc.HalfNormal(
            "subject_emission_scale",
            sigma=self.subject_initial_prior_scale,
        )
        subject_step_scale = pymc.HalfNormal(
            "emission_step_scale",
            sigma=self.subject_step_prior_scale,
        )
        subject_initial_raw = pymc.Normal(
            "subject_emission_initial_raw",
            mu=0.0,
            sigma=1.0,
            dims=("subject", *shape_dims),
        )
        n_subject_steps = len(structure.sessions) - len(structure.subjects)
        subject_step_raw = (
            pymc.Normal(
                "subject_emission_step_raw",
                mu=0.0,
                sigma=1.0,
                dims=("subject_step", *shape_dims),
            )
            if n_subject_steps
            else None
        )
        subject_values: list[Any] = [None] * len(structure.sessions)
        cursor = 0
        for subject, blocks in enumerate(structure.subject_blocks):
            deviation = subject_initial_raw[subject] * subject_initial_scale
            for within, block in enumerate(blocks):
                if within:
                    assert subject_step_raw is not None
                    deviation = deviation + subject_step_raw[cursor] * subject_step_scale
                    cursor += 1
                subject_values[block] = deviation
        subject_deviation = pymc.Deterministic(
            "subject_deviation_coefficients",
            tensor.stack(subject_values),
            dims=("path_session", *shape_dims),
        )
        parameter_names = [
            "population_emission_coefficients",
            "subject_deviation_coefficients",
            "population_emission_step_scale",
            "subject_emission_scale",
            "emission_step_scale",
        ]
        center = population[np.asarray(structure.population_index)]

        if isinstance(self.model, LabHierarchicalSessionDynamicBernoulliGLMHMM):
            lab_initial_scale = pymc.HalfNormal(
                "lab_emission_scale",
                sigma=self.lab_initial_prior_scale,
            )
            lab_step_scale = pymc.HalfNormal(
                "lab_emission_step_scale",
                sigma=self.lab_step_prior_scale,
            )
            lab_initial_raw = pymc.Normal(
                "lab_emission_initial_raw",
                mu=0.0,
                sigma=1.0,
                dims=("lab", *shape_dims),
            )
            lab_step_count = sum(len(blocks) for blocks in structure.lab_blocks) - len(
                structure.labs
            )
            lab_step_raw = (
                pymc.Normal(
                    "lab_emission_step_raw",
                    mu=0.0,
                    sigma=1.0,
                    dims=("lab_step", *shape_dims),
                )
                if lab_step_count
                else None
            )
            lab_points = sum(len(blocks) for blocks in structure.lab_blocks)
            lab_values: list[Any] = [None] * lab_points
            cursor = 0
            for lab, blocks in enumerate(structure.lab_blocks):
                deviation = lab_initial_raw[lab] * lab_initial_scale
                for within, block in enumerate(blocks):
                    if within:
                        assert lab_step_raw is not None
                        deviation = deviation + lab_step_raw[cursor] * lab_step_scale
                        cursor += 1
                    lab_values[block] = deviation
            lab_deviation = pymc.Deterministic(
                "lab_deviation_coefficients",
                tensor.stack(lab_values),
                dims=("lab_path", *shape_dims),
            )
            assert structure.session_lab_index is not None
            center = center + lab_deviation[np.asarray(structure.session_lab_index)]
            parameter_names.extend(
                [
                    "lab_deviation_coefficients",
                    "lab_emission_scale",
                    "lab_emission_step_scale",
                ]
            )

        emissions = pymc.Deterministic(
            "session_emission_coefficients",
            center + subject_deviation,
            dims=("path_session", *shape_dims),
        )
        return emissions, ["session_emission_coefficients", *parameter_names]

    def _path_structure(self, study: Study) -> _PathStructure | None:
        if not isinstance(self.model, SessionDynamicBernoulliGLMHMM):
            return None
        if isinstance(self.model, LabHierarchicalSessionDynamicBernoulliGLMHMM):
            values = _lab_structure(
                study,
                lab_column=self.model.lab_column,
                require_multiple_labs=True,
                require_replicated_subjects=True,
            )
            labels = tuple(
                f"{subject!r}/{session!r}"
                for subject, session in zip(values.path_subjects, values.keys, strict=True)
            )
            return _PathStructure(
                sessions=values.sessions,
                labels=labels,
                population_orders=values.population_orders,
                population_index=values.population_index,
                subjects=values.subjects,
                subject_blocks=values.subject_blocks,
                labs=values.labs,
                lab_blocks=values.lab_blocks,
                session_lab_index=values.session_lab_index,
            )
        if isinstance(self.model, HierarchicalSessionDynamicBernoulliGLMHMM):
            values = _population_structure(study, require_multiple_subjects=True)
            labels = tuple(
                f"{subject!r}/{session!r}"
                for subject, session in zip(values.path_subjects, values.keys, strict=True)
            )
            return _PathStructure(
                sessions=values.sessions,
                labels=labels,
                population_orders=values.population_orders,
                population_index=values.population_index,
                subjects=values.subjects,
                subject_blocks=values.subject_blocks,
            )
        sessions = ordered_session_indices(study)
        subjects = {_scalar(study["subject"][indices[0]]) for indices in sessions}
        if len(subjects) != 1:
            raise PyMCBackendError(
                "SessionDynamicBernoulliGLMHMM sampling requires exactly one subject"
            )
        labels = tuple(
            f"{_scalar(study['subject'][indices[0]])!r}/{_scalar(study['session'][indices[0]])!r}"
            for indices in sessions
        )
        orders = np.asarray(
            [int(study["session_order"][indices[0]]) for indices in sessions],
            dtype=np.int64,
        )
        return _PathStructure(
            sessions=sessions,
            labels=labels,
            population_orders=orders,
            population_index=np.arange(len(sessions), dtype=np.intp),
            subjects=(),
            subject_blocks=(),
        )

    def _draw_scores(
        self,
        study: Study,
        posterior: PosteriorResult,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        self._validate_posterior(posterior)
        outcomes = np.asarray(self.model.outcomes(study), dtype=np.float64)
        features = np.asarray(self.model.design_matrix(study), dtype=np.float64)
        group = posterior["posterior"]
        initial = _flatten_samples(group["initial_probabilities"].values)
        global_transition = _flatten_samples(group["global_transition_matrix"].values)
        structure = self._path_structure(study)
        if structure is None:
            emission = _flatten_samples(group["emission_coefficients"].values)
            if self.model.is_dynamic:
                effects = _flatten_samples(group["transition_effects"].values)
                offsets = np.einsum(
                    "tp,dsjp->dtsj",
                    self.model.transition_design_matrix(study),
                    effects,
                    optimize=True,
                )
                logits = np.log(global_transition)[:, None, :, :] + offsets
                logits -= np.logaddexp.reduce(logits, axis=3, keepdims=True)
                transitions = np.exp(logits)
            else:
                transitions = global_transition
            session_emission = None
        else:
            fitted_labels = tuple(
                str(value)
                for value in group["session_emission_coefficients"].coords["path_session"]
            )
            if fitted_labels != structure.labels:
                raise PyMCBackendError(
                    "dynamic posterior prediction currently requires the fitted "
                    "subject-session blocks in the same canonical order; unseen-session, "
                    "unseen-subject, and unseen-lab prediction needs explicit path propagation"
                )
            emission = None
            session_emission = _flatten_samples(group["session_emission_coefficients"].values)
            transitions = _flatten_samples(group["session_transition_matrices"].values)
        return _numpy_filtered_scores(
            features,
            outcomes,
            ordered_session_indices(study),
            initial,
            transitions,
            stationary_emissions=emission,
            session_emissions=session_emission,
        )

    def _draw_dynamic_hyperparameters(
        self, generator: np.random.Generator
    ) -> tuple[DynamicGLMHMM, dict[str, float]]:
        assert isinstance(self.model, SessionDynamicBernoulliGLMHMM)
        concentration = generator.exponential(self.session_transition_concentration_prior_mean)
        values: dict[str, float] = {}
        replacements: dict[str, float] = {"transition_concentration": concentration}
        if isinstance(self.model, HierarchicalSessionDynamicBernoulliGLMHMM):
            population = abs(generator.normal(0.0, self.population_step_prior_scale))
            subject_initial = abs(generator.normal(0.0, self.subject_initial_prior_scale))
            subject_step = abs(generator.normal(0.0, self.subject_step_prior_scale))
            replacements.update(
                population_emission_step_scale=population,
                subject_emission_scale=subject_initial,
                emission_step_scale=subject_step,
            )
            values.update(
                population_emission_step_scale=population,
                subject_emission_scale=subject_initial,
                emission_step_scale=subject_step,
            )
        else:
            session_step = abs(generator.normal(0.0, self.session_step_prior_scale))
            replacements["emission_step_scale"] = session_step
            values["emission_step_scale"] = session_step
        if isinstance(self.model, LabHierarchicalSessionDynamicBernoulliGLMHMM):
            lab_initial = abs(generator.normal(0.0, self.lab_initial_prior_scale))
            lab_step = abs(generator.normal(0.0, self.lab_step_prior_scale))
            replacements.update(
                lab_emission_scale=lab_initial,
                lab_emission_step_scale=lab_step,
            )
            values.update(
                lab_emission_scale=lab_initial,
                lab_emission_step_scale=lab_step,
            )
        # A continuous half-Normal/Exponential draw is positive almost surely. Guard the
        # representational zero that a finite generator can theoretically return.
        replacements = {
            name: max(float(value), np.finfo(np.float64).tiny)
            for name, value in replacements.items()
        }
        values = {name: replacements[name] for name in values}
        return replace(self.model, **replacements), values

    def _validate_posterior(self, posterior: PosteriorResult) -> None:
        if not isinstance(posterior, PosteriorResult):
            raise TypeError("posterior must be a PosteriorResult")
        if posterior.model_signature != self.signature:
            raise ValueError("posterior was produced by a different Bayesian GLM-HMM")

    @staticmethod
    def _prediction_mode(mode: PredictionMode) -> PredictionMode:
        selected = PredictionMode(mode)
        if selected is not PredictionMode.FILTERED:
            raise ValueError("Bayesian GLM-HMMs support filtered prediction only")
        return selected


def _forward_graph(
    tensor: Any,
    features: NDArray[np.float64],
    outcomes: NDArray[np.int8],
    sessions: tuple[tuple[int, ...], ...],
    initial: Any,
    transitions: Any,
    *,
    stationary_emissions: Any | None,
    session_emissions: Any | None,
) -> tuple[Any, Any, Any, Any]:
    """Build a differentiable, scaled forward recursion in source-row order."""

    probability: list[Any] = [None] * len(outcomes)
    log_likelihood: list[Any] = [None] * len(outcomes)
    predictive_state: list[Any] = [None] * len(outcomes)
    filtered_state: list[Any] = [None] * len(outcomes)
    log_initial = tensor.log(initial)
    for block, indices in enumerate(sessions):
        emissions = stationary_emissions if session_emissions is None else session_emissions[block]
        linear = tensor.dot(features[np.asarray(indices)], emissions.T)
        log_predictive = log_initial
        for within, index in enumerate(indices):
            choice_probability = tensor.sigmoid(linear[within])
            predictive = tensor.exp(log_predictive)
            probability[index] = tensor.sum(predictive * choice_probability)
            log_emission = tensor.switch(
                outcomes[index],
                -tensor.softplus(-linear[within]),
                -tensor.softplus(linear[within]),
            )
            evidence = tensor.logsumexp(log_predictive + log_emission)
            log_filtered = log_predictive + log_emission - evidence
            log_likelihood[index] = evidence
            predictive_state[index] = predictive
            filtered_state[index] = tensor.exp(log_filtered)
            if within + 1 < len(indices):
                transition = transitions if session_emissions is None else transitions[block]
                if session_emissions is None and getattr(transitions, "ndim", 0) == 3:
                    transition = transitions[indices[within + 1]]
                log_predictive = tensor.logsumexp(
                    log_filtered[:, None] + tensor.log(transition),
                    axis=0,
                )
    return (
        tensor.stack(probability),
        tensor.stack(log_likelihood),
        tensor.stack(predictive_state),
        tensor.stack(filtered_state),
    )


def _canonicalize_draws(
    inference_data: Any,
    *,
    label_coefficient: int,
    tolerance: float,
    dynamic: bool,
) -> None:
    """Relabel every state-indexed posterior variable with one permutation per draw."""

    posterior = inference_data.posterior
    xarray = importlib.import_module("xarray")
    emission_name = "session_emission_coefficients" if dynamic else "emission_coefficients"
    values = np.asarray(posterior[emission_name].values)
    score = values[..., label_coefficient]
    if dynamic:
        score = np.mean(score, axis=2)
    permutations = np.argsort(score, axis=-1, kind="stable")
    sorted_score = np.take_along_axis(score, permutations, axis=-1)
    gaps = np.min(np.diff(sorted_score, axis=-1), axis=-1)

    state_dims = {"state", "source_state", "destination_state"}
    for variable in posterior.data_vars:
        array = posterior[variable]
        axes = [axis for axis, dim in enumerate(array.dims) if dim in state_dims]
        if not axes or array.dims[:2] != ("chain", "draw"):
            continue
        reordered = np.asarray(array.values).copy()
        for chain in range(reordered.shape[0]):
            for draw in range(reordered.shape[1]):
                sample = reordered[chain, draw]
                for axis in axes:
                    sample = np.take(sample, permutations[chain, draw], axis=axis - 2)
                reordered[chain, draw] = sample
        posterior[variable].values = reordered

    crossing = np.zeros(gaps.shape, dtype=np.bool_)
    if dynamic:
        relabelled = np.asarray(posterior[emission_name].values)[..., label_coefficient]
        crossing = np.any(np.diff(relabelled, axis=-1) < 0.0, axis=(2, 3))
    sample_coords = {
        "chain": np.asarray(posterior.coords["chain"]),
        "draw": np.asarray(posterior.coords["draw"]),
    }
    posterior["label_permutation"] = xarray.DataArray(
        permutations,
        dims=("chain", "draw", "state"),
        coords={**sample_coords, "state": np.asarray(posterior.coords["state"])},
    )
    posterior["label_minimum_gap"] = xarray.DataArray(
        gaps,
        dims=("chain", "draw"),
        coords=sample_coords,
    )
    posterior["label_path_crossing"] = xarray.DataArray(
        crossing,
        dims=("chain", "draw"),
        coords=sample_coords,
    )
    posterior["label_ambiguous"] = xarray.DataArray(
        (gaps <= tolerance) | crossing,
        dims=("chain", "draw"),
        coords=sample_coords,
    )


def _attach_evidence(
    result: PosteriorResult,
    study: Study,
    *,
    outcome: str,
    log_likelihood: NDArray[np.float64],
    posterior_predictive: NDArray[np.int8],
    features: NDArray[np.float64],
    structure: _PathStructure | None,
) -> PosteriorResult:
    posterior_reference = result["posterior"].variables[0]
    sample_coords = {
        "chain": posterior_reference.coords["chain"],
        "draw": posterior_reference.coords["draw"],
    }
    trial = np.arange(len(study), dtype=np.int64)
    coordinates = {**sample_coords, "trial": trial}
    observed = np.asarray(study[outcome], dtype=np.int8)
    groups = {group.name: group for group in result.groups}
    groups["log_likelihood"] = PosteriorGroup(
        "log_likelihood",
        (PosteriorVariable(outcome, log_likelihood, ("chain", "draw", "trial"), coordinates),),
    )
    groups["posterior_predictive"] = PosteriorGroup(
        "posterior_predictive",
        (
            PosteriorVariable(
                outcome,
                posterior_predictive,
                ("chain", "draw", "trial"),
                coordinates,
            ),
        ),
    )
    groups["observed_data"] = PosteriorGroup(
        "observed_data",
        (PosteriorVariable(outcome, observed, ("trial",), {"trial": trial}),),
    )
    constant = [
        PosteriorVariable("trial_subject", study["subject"], ("trial",), {"trial": trial}),
        PosteriorVariable("trial_session", study["session"], ("trial",), {"trial": trial}),
        PosteriorVariable("trial_in_session", study["trial"], ("trial",), {"trial": trial}),
        PosteriorVariable(
            "trial_session_order", study["session_order"], ("trial",), {"trial": trial}
        ),
        PosteriorVariable(
            "design_matrix",
            features,
            ("trial", "coefficient"),
            {
                "trial": trial,
                "coefficient": np.asarray(
                    result["posterior"][
                        "session_emission_coefficients"
                        if structure is not None
                        else "emission_coefficients"
                    ].coords["coefficient"]
                ),
            },
        ),
    ]
    if structure is not None:
        row_block = np.empty(len(study), dtype=np.int64)
        for block, indices in enumerate(structure.sessions):
            row_block[np.asarray(indices)] = block
        constant.append(
            PosteriorVariable("trial_path_session_index", row_block, ("trial",), {"trial": trial})
        )
    groups["constant_data"] = PosteriorGroup("constant_data", tuple(constant))
    ordered = tuple(groups[name] for name in POSTERIOR_GROUPS if name in groups)
    return PosteriorResult(
        model_name=result.model_name,
        model_signature=result.model_signature,
        inference_library=result.inference_library,
        inference_library_version=result.inference_library_version,
        parameter_names=result.parameter_names,
        groups=ordered,
        attrs=result.attrs,
    )


def _numpy_filtered_scores(
    features: NDArray[np.float64],
    outcomes: NDArray[np.float64],
    sessions: tuple[tuple[int, ...], ...],
    initial: NDArray[np.float64],
    transitions: NDArray[np.float64],
    *,
    stationary_emissions: NDArray[np.float64] | None,
    session_emissions: NDArray[np.float64] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    draws = len(initial)
    probability = np.empty((draws, len(outcomes)), dtype=np.float64)
    scores = np.empty_like(probability)
    for block, indices in enumerate(sessions):
        emissions = (
            stationary_emissions if session_emissions is None else session_emissions[:, block]
        )
        assert emissions is not None
        linear = np.einsum("tc,dsc->dts", features[np.asarray(indices)], emissions)
        predictive = initial.copy()
        for within, index in enumerate(indices):
            emission_probability = expit(linear[:, within])
            probability[:, index] = np.sum(predictive * emission_probability, axis=1)
            observed_probability = np.where(
                outcomes[index] != 0,
                emission_probability,
                1.0 - emission_probability,
            )
            evidence = np.sum(predictive * observed_probability, axis=1)
            scores[:, index] = np.log(evidence)
            filtered = predictive * observed_probability / evidence[:, None]
            if within + 1 < len(indices):
                transition = transitions
                if session_emissions is not None:
                    transition = transitions[:, block]
                elif transitions.ndim == 4:
                    transition = transitions[:, indices[within + 1]]
                predictive = np.einsum("ds,dsj->dj", filtered, transition)
    return probability, scores


def _flatten_samples(values: NDArray[Any]) -> NDArray[Any]:
    array = np.asarray(values)
    return array.reshape((array.shape[0] * array.shape[1], *array.shape[2:]))


def _path_permutation(emissions: NDArray[np.float64], coefficient: int) -> NDArray[np.intp]:
    return np.argsort(np.mean(emissions[..., coefficient], axis=0), kind="stable")


def _permute_truth(name: str, values: NDArray[Any], permutation: NDArray[np.intp]) -> NDArray[Any]:
    if name == "initial_probabilities":
        return values[permutation]
    if name == "global_transition_matrix":
        return values[np.ix_(permutation, permutation)]
    if name == "session_transition_matrices":
        return values[:, permutation][:, :, permutation]
    return values[:, permutation]


def _object_coordinate(values: tuple[Any, ...]) -> NDArray[Any]:
    coordinate = np.empty(len(values), dtype=object)
    coordinate[:] = values
    return coordinate


def _scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value
