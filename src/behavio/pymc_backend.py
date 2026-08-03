"""Optional PyMC inference for a hierarchical penalised-linear behavioural model.

This adapter used to name one class, ``HierarchicalBernoulliHistoryGLM``, and reach into two
of its private methods for the outcome vector and the filtered feature matrix. Both are now
members of :class:`behavio.contracts.compose.PenalisedLinearEstimator`, so the adapter
dispatches on the contract: anything that :func:`behavio.compose.hierarchical` produced over
a penalised linear model with a diagonal penalty can be sampled, and the two priors are read
off the model's own penalty matrices rather than re-derived from its ``l2`` field.
"""

from __future__ import annotations

import importlib
import importlib.metadata
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit

from behavio.compose.hierarchy import HierarchicalModel
from behavio.contracts.compose import group_blocks
from behavio.contracts.estimator import FitResult, Prediction, PredictionMode
from behavio.contracts.posterior import (
    PosteriorCentre,
    posterior_log_predictive_density,
    posterior_point_summary,
)
from behavio.inference.parameters import ParameterSpace, PriorFamily
from behavio.models._kernels.bernoulli import BernoulliLikelihood, ordered_session_indices
from behavio.models.q_learning import BinaryQLearning
from behavio.posterior.result import (
    POSTERIOR_GROUPS,
    PosteriorGroup,
    PosteriorResult,
    PosteriorVariable,
    posterior_result_from_arviz,
)
from behavio.posterior.simulation_based_calibration import SBCSimulation
from behavio.task.spec import TaskSpec
from behavio.trials import Study


class PyMCBackendError(ValueError):
    """Raised when a model or sampling configuration violates the PyMC adapter contract."""


class PyMCUnavailableError(ImportError):
    """Raised when the optional PyMC backend is requested but not installed."""


@dataclass(frozen=True, slots=True)
class PyMCHierarchicalGLMBackend:
    """Sample a fixed-scale hierarchical penalised-linear model with PyMC NUTS.

    This adapter changes inference, not behavioural semantics. It calls the contract's own
    outcome and design builders, validates the same :class:`TaskSpec`, and uses the priors
    the MAP objective already implies: a quadratic penalty *is* a Gaussian precision, so a
    zero on the penalty diagonal is a flat prior and a positive one is a Normal with
    standard deviation ``1 / sqrt(penalty)``. Group deviations get the same treatment from
    :meth:`~behavio.contracts.compose.PenalisedLinearEstimator.group_penalty`, so a model
    whose parameters vary only on some coordinates samples exactly those.

    ``seed`` is entropy for :class:`numpy.random.SeedSequence`, not a 32-bit sampler seed:
    it is any non-negative integer, and the adapter derives the 32-bit words PyMC needs.
    That is what lets a seed emitted by
    :mod:`behavio.posterior.simulation_based_calibration`, :mod:`behavio.recovery.parameters`, or
    :mod:`behavio.recovery.models` be handed straight to this backend.
    """

    draws: int = 1_000
    tune: int = 1_000
    chains: int = 4
    cores: int = 1
    target_accept: float = 0.9
    seed: int = 0

    def __post_init__(self) -> None:
        for value, label in (
            (self.draws, "draws"),
            (self.tune, "tune"),
            (self.chains, "chains"),
            (self.cores, "cores"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise PyMCBackendError(f"{label} must be a positive integer")
        if self.chains < 2:
            raise PyMCBackendError("chains must be at least two for cross-chain diagnostics")
        if self.cores > self.chains:
            raise PyMCBackendError("cores cannot exceed chains")
        if not np.isfinite(self.target_accept) or not 0.5 <= self.target_accept < 1.0:
            raise PyMCBackendError("target_accept must be finite and in [0.5, 1)")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise PyMCBackendError("seed must be a non-negative integer")

    @property
    def backend_name(self) -> str:
        """Stable backend identifier."""

        return "pymc.NUTS"

    @property
    def backend_config(self) -> MappingProxyType[str, Any]:
        """Immutable, complete sampling configuration."""

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

    def sample(
        self,
        model: HierarchicalModel,
        study: Study,
        *,
        task: TaskSpec,
    ) -> PosteriorResult:
        """Return a task-validated, labelled full-posterior result."""

        if not isinstance(model, HierarchicalModel):
            raise TypeError("model must be a hierarchical() composition")
        if not isinstance(study, Study):
            raise TypeError("study must be a Study")
        if not isinstance(task, TaskSpec):
            raise TypeError("task must be a TaskSpec")
        if model.estimate_scale:
            raise PyMCBackendError(
                "estimate_scale=True has no declared full-posterior scale prior; "
                "use a fixed scale for this adapter"
            )
        inner = model.model
        # `likelihood` is not a universal member: a model whose scores come from a
        # recursion withdraws it deliberately, and reading it raises rather than
        # returning something misleading. Ask, so an unsupported model is refused by
        # this adapter's own message instead of an AttributeError from inside it.
        if not isinstance(getattr(inner, "likelihood", None), BernoulliLikelihood):
            raise PyMCBackendError(
                "this adapter declares a Bernoulli observation model; "
                f"{inner.model_name} does not use one"
            )
        population_precision = _diagonal_precision(inner.penalty_matrix(), "population penalty")
        columns = model.effects.columns(inner.parameter_names)
        deviation_precision = _diagonal_precision(
            inner.group_penalty(columns, model.effects.ordered_scales(inner.parameter_names)),
            "group penalty",
        )
        task.validate_model(model)
        validation = task.validate(study)
        blocks = group_blocks(study, model.grouping)
        subjects = blocks.labels
        if len(subjects) < 2:
            raise PyMCBackendError(
                f"hierarchical sampling requires at least two {model.grouping} groups"
            )
        outcomes = inner.outcomes(study)
        features = inner.design_matrix(study)
        row_subject = np.asarray(blocks.row_block, dtype=np.int32)
        pymc = _import_pymc()
        inference_version = importlib.metadata.version("pymc")
        sampling_seed, predictive_seed = (
            int(value) for value in np.random.SeedSequence(self.seed).generate_state(2)
        )

        parameter_names = tuple(inner.parameter_names)
        varying = model.varying_parameters
        deviation_dim = "coefficient" if varying == parameter_names else "varying_coefficient"
        paired = tuple(zip(parameter_names, population_precision, strict=True))
        flat = tuple(name for name, tau in paired if tau == 0)
        penalised = tuple(name for name, tau in paired if tau > 0)
        if parameter_names[: len(flat)] != flat:
            raise PyMCBackendError(
                "this adapter puts the unpenalised parameters first, as an intercept-plus-"
                f"slopes model does; {model.model_name} interleaves them: {parameter_names}"
            )
        outcome_column = inner.scored_columns[0]

        coords = {
            "trial": np.arange(len(study), dtype=np.int64),
            model.grouping: _object_coordinate(subjects),
            "coefficient": np.asarray(parameter_names),
        }
        if penalised:
            coords["regularized_coefficient"] = np.asarray(penalised)
        if deviation_dim != "coefficient":
            coords[deviation_dim] = np.asarray(varying)

        with pymc.Model(coords=coords) as graph:
            design_matrix = pymc.Data(
                "design_matrix",
                features,
                dims=("trial", "coefficient"),
            )
            subject_index = pymc.Data("subject_index", row_subject, dims="trial")
            blocks = []
            if flat:
                blocks.append(pymc.Flat("population_intercept", shape=len(flat)))
            if penalised:
                blocks.append(
                    pymc.Normal(
                        "population_slope",
                        mu=0.0,
                        sigma=1.0 / np.sqrt(population_precision[len(flat) :]),
                        dims="regularized_coefficient",
                    )
                )
            population_values = (
                blocks[0] if len(blocks) == 1 else pymc.math.concatenate(tuple(blocks))
            )
            population = pymc.Deterministic(
                "population_coefficient",
                population_values,
                dims="coefficient",
            )
            deviations = pymc.Normal(
                "subject_deviation",
                mu=0.0,
                sigma=1.0 / np.sqrt(deviation_precision),
                dims=(model.grouping, deviation_dim),
            )
            pymc.Deterministic(
                "subject_coefficient",
                population[columns][None, :] + deviations,
                dims=(model.grouping, deviation_dim),
            )
            linear_predictor = pymc.math.sum(design_matrix * population[None, :], axis=1)
            linear_predictor = linear_predictor + pymc.math.sum(
                design_matrix[:, columns] * deviations[subject_index], axis=1
            )
            pymc.Deterministic(
                "choice_probability",
                pymc.math.sigmoid(linear_predictor),
                dims="trial",
            )
            pymc.Bernoulli(
                outcome_column,
                logit_p=linear_predictor,
                observed=outcomes.astype(np.int8),
                dims="trial",
            )
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
            inference_data = pymc.compute_log_likelihood(
                inference_data,
                model=graph,
                progressbar=False,
            )
            inference_data = pymc.sample_posterior_predictive(
                inference_data,
                model=graph,
                var_names=[outcome_column],
                random_seed=predictive_seed,
                progressbar=False,
                extend_inferencedata=True,
            )

        result = posterior_result_from_arviz(
            inference_data,
            model_name=model.model_name,
            model_signature=model.signature,
            inference_library="PyMC",
            inference_library_version=inference_version,
            parameter_names=("population_coefficient", "subject_deviation"),
        )
        result = _retain_trial_identity(result, study, outcome_column)
        attrs = {
            **dict(result.attrs),
            "backend": self.backend_name,
            "backend_config": dict(self.backend_config),
            "prediction_mode": "filtered",
            "scored_columns": list(model.scored_columns),
            "trial_coordinate": "study source-row position",
            "task_validation": {
                "n_trials": validation.n_trials,
                "n_observed_choices": validation.n_observed_choices,
                "n_omissions": validation.n_omissions,
            },
            "population_prior": _prior_statement(parameter_names, population_precision),
            "subject_deviation_prior": _prior_statement(varying, deviation_precision),
        }
        return PosteriorResult(
            model_name=result.model_name,
            model_signature=result.model_signature,
            inference_library=result.inference_library,
            inference_library_version=result.inference_library_version,
            parameter_names=result.parameter_names,
            groups=result.groups,
            parameter_space_fingerprint=result.parameter_space_fingerprint,
            attrs=attrs,
        )


@dataclass(frozen=True, slots=True)
class PyMCBinaryQLearning:
    """Proper-prior Bayesian binary Q-learning with filtered posterior prediction.

    The likelihood is the usual conditional action likelihood: each choice is predicted
    from values updated by the choices and rewards observed earlier in the same session.
    Held-out scoring repeats that recursion for every posterior draw and integrates the
    resulting density over draws.  The four priors live in the same
    :class:`~behavio.inference.parameters.ParameterSpace` as the maximum-likelihood model,
    so prior simulation, PyMC sampling, parameter recovery, and SBC use one declaration.

    ``posterior_predictive`` contains one-step-ahead conditional choice replicates.  It
    conditions on the observed history rather than pretending that an independently drawn
    replacement choice also generated the study's next observed reward.  Full recursive
    prior prediction is exposed by :meth:`prior_predictive_simulation`, which uses the
    task's declared action-contingent reward probabilities.
    """

    outcome: str = "choice"
    reward: str = "reward"
    reward_probability_columns: tuple[str, str] = (
        "reward_probability_0",
        "reward_probability_1",
    )
    initial_value: float = 0.5
    draws: int = 1_000
    tune: int = 1_000
    chains: int = 4
    cores: int = 1
    target_accept: float = 0.9
    seed: int = 0

    def __post_init__(self) -> None:
        # Constructing the frequentist twin applies the shared data/model validation.
        self._base_model()
        for value, label in (
            (self.draws, "draws"),
            (self.tune, "tune"),
            (self.chains, "chains"),
            (self.cores, "cores"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise PyMCBackendError(f"{label} must be a positive integer")
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
        return "bayesian-binary-q-learning"

    @property
    def signature(self) -> str:
        environment = ",".join(self.reward_probability_columns)
        return (
            f"{self.model_name}[outcome={self.outcome};reward={self.reward};"
            f"environment={environment};initial_value={self.initial_value};"
            f"session_reset=True;prior={self.parameter_space.fingerprint}]"
        )

    @property
    def parameter_space(self) -> ParameterSpace:
        """Natural parameters, transforms, and normalized priors shared with Q-learning."""

        return self._base_model().parameter_space

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return self.parameter_space.natural_names

    @property
    def posterior_parameter_labels(self) -> Mapping[str, str]:
        return MappingProxyType({name: name for name in self.parameter_names})

    @property
    def scored_columns(self) -> tuple[str, ...]:
        return (self.outcome,)

    @property
    def required_task_columns(self) -> tuple[str, ...]:
        return (self.reward,)

    @property
    def supported_prediction_modes(self) -> tuple[PredictionMode, ...]:
        return (PredictionMode.FILTERED,)

    @property
    def backend_name(self) -> str:
        return "pymc.NUTS"

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

    def sample(self, study: Study) -> PosteriorResult:
        """Sample the conditional sequential likelihood with PyMC NUTS."""

        if not isinstance(study, Study):
            raise TypeError("study must be a Study")
        base = self._base_model()
        choices = base._choices(study).astype(np.int32)
        rewards = base._rewards(study)
        sessions = ordered_session_indices(study)
        order = np.asarray([index for block in sessions for index in block], dtype=np.intp)
        inverse_order = np.argsort(order)
        ordered_choices = choices[order]
        ordered_rewards = rewards[order]
        reset = np.zeros(len(study), dtype=np.int8)
        offset = 0
        for block in sessions:
            reset[offset] = 1
            offset += len(block)

        pymc = _import_pymc()
        pytensor = importlib.import_module("pytensor")
        tensor = importlib.import_module("pytensor.tensor")
        inference_version = importlib.metadata.version("pymc")
        sampling_seed, predictive_seed = (
            int(value) for value in np.random.SeedSequence(self.seed).generate_state(2)
        )
        coords = {"trial": np.arange(len(study), dtype=np.int64)}

        with pymc.Model(coords=coords):
            learning_rate = _pymc_prior(pymc, self.parameter_space, "learning_rate")
            inverse_temperature = _pymc_prior(pymc, self.parameter_space, "inverse_temperature")
            choice_bias = _pymc_prior(pymc, self.parameter_space, "choice_bias")
            perseveration = _pymc_prior(pymc, self.parameter_space, "perseveration")

            def transition(
                choice: Any,
                observed_reward: Any,
                starts_session: Any,
                previous_values: Any,
                previous_choice: Any,
                alpha: Any,
                temperature: Any,
                bias: Any,
                stay: Any,
            ) -> tuple[Any, Any, Any]:
                values = tensor.switch(
                    tensor.neq(starts_session, 0),
                    tensor.full_like(previous_values, self.initial_value),
                    previous_values,
                )
                history = tensor.switch(
                    tensor.neq(starts_session, 0),
                    tensor.zeros_like(previous_choice),
                    previous_choice,
                )
                linear = temperature * (values[1] - values[0]) + bias + stay * history
                numeric_choice = tensor.cast(choice, values.dtype)
                action = tensor.stack((1.0 - numeric_choice, numeric_choice))
                chosen_value = tensor.sum(values * action)
                updated = values + action * alpha * (observed_reward - chosen_value)
                return updated, 2.0 * numeric_choice - 1.0, linear

            _, _, ordered_linear = pytensor.scan(
                fn=transition,
                sequences=(
                    tensor.as_tensor_variable(ordered_choices),
                    tensor.as_tensor_variable(ordered_rewards),
                    tensor.as_tensor_variable(reset),
                ),
                n_steps=len(study),
                outputs_info=(
                    np.full(2, self.initial_value, dtype=np.float64),
                    np.asarray(0.0, dtype=np.float64),
                    None,
                ),
                non_sequences=(
                    learning_rate,
                    inverse_temperature,
                    choice_bias,
                    perseveration,
                ),
                strict=True,
                return_updates=False,
            )
            linear = ordered_linear[inverse_order]
            pymc.Deterministic("choice_probability", pymc.math.sigmoid(linear), dims="trial")
            pymc.Bernoulli(
                self.outcome,
                logit_p=linear,
                observed=choices.astype(np.int8),
                dims="trial",
            )
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

        result = posterior_result_from_arviz(
            inference_data,
            model_name=self.model_name,
            model_signature=self.signature,
            inference_library="PyMC",
            inference_library_version=inference_version,
            parameter_names=self.parameter_names,
            parameter_space_fingerprint=self.parameter_space.fingerprint,
        )
        draw_parameters = self._draw_parameters(result)
        draw_linear = self._draw_linear_predictors(study, draw_parameters).reshape(
            self.chains, self.draws, len(study)
        )
        draw_probability = expit(draw_linear)
        outcome = choices.astype(np.float64)
        draw_log_likelihood = outcome * -np.logaddexp(0.0, -draw_linear)
        draw_log_likelihood += (1.0 - outcome) * -np.logaddexp(0.0, draw_linear)
        generator = np.random.default_rng(predictive_seed)
        predictive = generator.binomial(1, draw_probability).astype(np.int8)
        result = _with_q_learning_evidence(
            result,
            study,
            outcome_name=self.outcome,
            reward_name=self.reward,
            log_likelihood=draw_log_likelihood,
            posterior_predictive=predictive,
        )
        return PosteriorResult(
            model_name=result.model_name,
            model_signature=result.model_signature,
            inference_library=result.inference_library,
            inference_library_version=result.inference_library_version,
            parameter_names=result.parameter_names,
            groups=result.groups,
            parameter_space_fingerprint=result.parameter_space_fingerprint,
            attrs={
                **dict(result.attrs),
                "backend": self.backend_name,
                "backend_config": dict(self.backend_config),
                "prediction_mode": "filtered",
                "posterior_predictive_conditioning": (
                    "one-step-ahead choices conditional on observed earlier choices and rewards"
                ),
                "scored_columns": list(self.scored_columns),
                "parameter_space": self.parameter_space.to_dict(),
                "prior_predictive": "full joint via prior_predictive_simulation(design, seed)",
                "trial_coordinate": "study source-row position",
            },
        )

    def sample_with_seed(self, study: Study, seed: int) -> PosteriorResult:
        """SBC-compatible inference callable with an explicit replicate seed."""

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
        """Integrate filtered, history-replayed probabilities over posterior draws."""

        self._prediction_mode(mode)
        self._validate_posterior(posterior)
        draw_linear = self._draw_linear_predictors(study, self._draw_parameters(posterior))
        probability = np.mean(expit(draw_linear), axis=0)
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
        """Return the draw-averaged held-out choice log predictive density."""

        self._prediction_mode(mode)
        self._validate_posterior(posterior)
        choices = self._base_model()._choices(study)
        linear = self._draw_linear_predictors(study, self._draw_parameters(posterior))
        draw_log_prob = choices * -np.logaddexp(0.0, -linear)
        draw_log_prob += (1.0 - choices) * -np.logaddexp(0.0, linear)
        return posterior_log_predictive_density(draw_log_prob)

    def point_summary(
        self,
        posterior: PosteriorResult,
        *,
        converged: bool,
        centre: PosteriorCentre = PosteriorCentre.MEAN,
    ) -> FitResult:
        self._validate_posterior(posterior)
        return posterior_point_summary(posterior, converged=converged, centre=centre)

    def simulate(
        self,
        design: Study,
        parameters: Mapping[str, float],
        *,
        seed: int | np.random.Generator,
    ) -> Study:
        """Generate the full action/reward recursion from natural-scale parameters."""

        optimizer = self.parameter_space.encode_mapping(parameters)
        return self._base_model().simulate(design, optimizer, seed=seed)

    def prior_predictive_simulation(self, design: Study, seed: int) -> SBCSimulation:
        """Draw parameters from the full prior joint and simulate one SBC replicate."""

        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise PyMCBackendError("seed must be a non-negative integer")
        parameter_seed, simulation_seed = (
            int(value) for value in np.random.SeedSequence(seed).generate_state(2)
        )
        parameters = self.parameter_space.sample_prior(seed=parameter_seed)
        study = self.simulate(design, parameters, seed=simulation_seed)
        return SBCSimulation(study=study, truth=parameters)

    def _base_model(self) -> BinaryQLearning:
        return BinaryQLearning(
            outcome=self.outcome,
            reward=self.reward,
            reward_probability_columns=self.reward_probability_columns,
            initial_value=self.initial_value,
        )

    def _draw_parameters(self, posterior: PosteriorResult) -> NDArray[np.float64]:
        self._validate_posterior(posterior)
        variables = posterior["posterior"]
        values = [
            np.asarray(variables[name].values, dtype=np.float64) for name in self.parameter_names
        ]
        expected = (posterior.n_chains, posterior.n_draws)
        if any(value.shape != expected for value in values):
            raise PyMCBackendError("Q-learning posterior parameters must be scalar per draw")
        return np.column_stack([value.reshape(-1) for value in values])

    def _draw_linear_predictors(
        self, study: Study, parameters: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        base = self._base_model()
        choices = base._choices(study)
        rewards = base._rewards(study)
        draws = np.asarray(parameters, dtype=np.float64)
        if draws.ndim != 2 or draws.shape[1] != len(self.parameter_names):
            raise ValueError("parameters must have shape (posterior_draw, 4)")
        linear = np.empty((len(draws), len(study)), dtype=np.float64)
        learning_rate = draws[:, 0]
        inverse_temperature = draws[:, 1]
        choice_bias = draws[:, 2]
        perseveration = draws[:, 3]
        for session in ordered_session_indices(study):
            values = np.full((len(draws), 2), self.initial_value, dtype=np.float64)
            previous_choice = 0.0
            for index in session:
                linear[:, index] = (
                    inverse_temperature * (values[:, 1] - values[:, 0])
                    + choice_bias
                    + perseveration * previous_choice
                )
                choice = int(choices[index])
                values[:, choice] += learning_rate * (rewards[index] - values[:, choice])
                previous_choice = 2.0 * choice - 1.0
        return linear

    def _validate_posterior(self, posterior: PosteriorResult) -> None:
        if not isinstance(posterior, PosteriorResult):
            raise TypeError("posterior must be a PosteriorResult")
        if posterior.model_signature != self.signature:
            raise ValueError("posterior was produced by a different model specification")
        if posterior.parameter_names != self.parameter_names:
            raise ValueError("posterior parameter names do not match this Q-learning model")

    @staticmethod
    def _prediction_mode(mode: PredictionMode) -> PredictionMode:
        selected = PredictionMode(mode)
        if selected is not PredictionMode.FILTERED:
            raise ValueError("Bayesian Q-learning supports filtered prediction only")
        return selected


def _pymc_prior(pymc: Any, space: ParameterSpace, name: str) -> Any:
    """Build one PyMC random variable from the common natural-scale prior declaration."""

    spec = next((item for item in space.parameters if item.name == name), None)
    if spec is None or spec.prior is None:
        raise PyMCBackendError(f"parameter {name!r} needs a declared proper prior")
    prior = spec.prior
    arguments = prior.arguments
    if prior.family is PriorFamily.NORMAL:
        return pymc.Normal(name, mu=arguments["location"], sigma=arguments["scale"])
    if prior.family is PriorFamily.HALF_NORMAL:
        return pymc.HalfNormal(name, sigma=arguments["scale"])
    if prior.family is PriorFamily.BETA:
        return pymc.Beta(name, alpha=arguments["alpha"], beta=arguments["beta"])
    return pymc.Uniform(name, lower=arguments["lower"], upper=arguments["upper"])


def _with_q_learning_evidence(
    result: PosteriorResult,
    study: Study,
    *,
    outcome_name: str,
    reward_name: str,
    log_likelihood: NDArray[np.float64],
    posterior_predictive: NDArray[np.int8],
) -> PosteriorResult:
    """Attach source-ordered conditional likelihood, predictive draws, and trial identity."""

    posterior_reference = result["posterior"].variables[0]
    sample_coords = {
        "chain": posterior_reference.coords["chain"],
        "draw": posterior_reference.coords["draw"],
    }
    trial = np.arange(len(study), dtype=np.int64)
    coordinates = {**sample_coords, "trial": trial}
    replacements = {
        "log_likelihood": PosteriorGroup(
            "log_likelihood",
            (
                PosteriorVariable(
                    outcome_name,
                    log_likelihood,
                    ("chain", "draw", "trial"),
                    coordinates,
                ),
            ),
        ),
        "posterior_predictive": PosteriorGroup(
            "posterior_predictive",
            (
                PosteriorVariable(
                    outcome_name,
                    posterior_predictive,
                    ("chain", "draw", "trial"),
                    coordinates,
                ),
            ),
        ),
    }
    metadata = (
        PosteriorVariable("trial_subject", study["subject"], ("trial",), {"trial": trial}),
        PosteriorVariable("trial_session", study["session"], ("trial",), {"trial": trial}),
        PosteriorVariable("trial_in_session", study["trial"], ("trial",), {"trial": trial}),
        PosteriorVariable(
            "trial_session_order",
            study["session_order"],
            ("trial",),
            {"trial": trial},
        ),
        PosteriorVariable(f"trial_{reward_name}", study[reward_name], ("trial",), {"trial": trial}),
    )
    existing = {group.name: group for group in result.groups}
    constant = existing.get("constant_data")
    replacements["constant_data"] = PosteriorGroup(
        "constant_data",
        metadata if constant is None else (*constant.variables, *metadata),
        attrs=None if constant is None else constant.attrs,
    )
    existing.update(replacements)
    groups = tuple(existing[name] for name in POSTERIOR_GROUPS if name in existing)
    return PosteriorResult(
        model_name=result.model_name,
        model_signature=result.model_signature,
        inference_library=result.inference_library,
        inference_library_version=result.inference_library_version,
        parameter_names=result.parameter_names,
        groups=groups,
        parameter_space_fingerprint=result.parameter_space_fingerprint,
        attrs=result.attrs,
    )


def _import_pymc() -> Any:
    try:
        return importlib.import_module("pymc")
    except ImportError as error:
        raise PyMCUnavailableError(
            "PyMC inference requires `pip install 'behavio[bayesian]'`"
        ) from error


def _retain_trial_identity(
    result: PosteriorResult,
    study: Study,
    outcome: str,
) -> PosteriorResult:
    trial = result["observed_data"][outcome].coords["trial"]
    metadata = (
        PosteriorVariable("trial_subject", study["subject"], ("trial",), {"trial": trial}),
        PosteriorVariable("trial_session", study["session"], ("trial",), {"trial": trial}),
        PosteriorVariable("trial_in_session", study["trial"], ("trial",), {"trial": trial}),
        PosteriorVariable(
            "trial_session_order",
            study["session_order"],
            ("trial",),
            {"trial": trial},
        ),
    )
    groups = list(result.groups)
    constant_index = next(
        (index for index, group in enumerate(groups) if group.name == "constant_data"),
        None,
    )
    if constant_index is None:
        groups.append(PosteriorGroup("constant_data", metadata))
    else:
        constant = groups[constant_index]
        collisions = set(constant.variable_names) & {variable.name for variable in metadata}
        if collisions:
            raise PyMCBackendError(f"PyMC constant data collides with trial identity: {collisions}")
        groups[constant_index] = PosteriorGroup(
            "constant_data",
            (*constant.variables, *metadata),
            attrs=constant.attrs,
        )
    return PosteriorResult(
        model_name=result.model_name,
        model_signature=result.model_signature,
        inference_library=result.inference_library,
        inference_library_version=result.inference_library_version,
        parameter_names=result.parameter_names,
        groups=tuple(groups),
        parameter_space_fingerprint=result.parameter_space_fingerprint,
        attrs=result.attrs,
    )


def _diagonal_precision(penalty: np.ndarray, label: str) -> np.ndarray:
    """Read a quadratic penalty as a Gaussian precision, refusing correlated priors."""

    diagonal = np.diag(penalty)
    if not np.allclose(penalty, np.diag(diagonal), rtol=0.0, atol=0.0):
        raise PyMCBackendError(
            f"the {label} couples parameters, and this adapter declares independent "
            "priors; a correlated prior needs its own declared full-posterior form"
        )
    if np.any(diagonal < 0):
        raise PyMCBackendError(f"the {label} must be non-negative")
    return np.asarray(diagonal, dtype=np.float64)


def _prior_statement(names: tuple[Any, ...], precision: np.ndarray) -> str:
    return "; ".join(
        f"{name}: flat" if tau == 0 else f"{name}: Normal(0, {1.0 / np.sqrt(tau):.17g})"
        for name, tau in zip(names, precision, strict=True)
    )


def _object_coordinate(values: tuple[Any, ...]) -> np.ndarray:
    coordinate = np.empty(len(values), dtype=object)
    coordinate[:] = values
    return coordinate


def _scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value
