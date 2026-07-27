"""Optional PyMC inference for an established hierarchical behavioural model."""

from __future__ import annotations

import importlib
import importlib.metadata
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from unspool.models.hierarchical_glm import HierarchicalBernoulliHistoryGLM
from unspool.posterior import (
    PosteriorGroup,
    PosteriorResult,
    PosteriorVariable,
    posterior_result_from_arviz,
)
from unspool.study import Study
from unspool.task import TaskSpec


class PyMCBackendError(ValueError):
    """Raised when a model or sampling configuration violates the PyMC adapter contract."""


class PyMCUnavailableError(ImportError):
    """Raised when the optional PyMC backend is requested but not installed."""


@dataclass(frozen=True, slots=True)
class PyMCHierarchicalGLMBackend:
    """Sample the fixed-scale hierarchical Bernoulli history GLM with PyMC NUTS.

    This adapter changes inference, not behavioural semantics. It calls the model's own
    outcome and filtered-history feature builders, validates the same :class:`TaskSpec`,
    and uses the priors implied by the existing MAP objective: a flat intercept, flat or
    L2-equivalent Gaussian population slopes, and Gaussian subject deviations with the
    model's fixed ``subject_scale``.
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
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed < 2**32 - 1
        ):
            raise PyMCBackendError("seed must be an integer in [0, 2**32 - 1)")

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
        model: HierarchicalBernoulliHistoryGLM,
        study: Study,
        *,
        task: TaskSpec,
    ) -> PosteriorResult:
        """Return a task-validated, labelled full-posterior result."""

        if not isinstance(model, HierarchicalBernoulliHistoryGLM):
            raise TypeError("model must be a HierarchicalBernoulliHistoryGLM")
        if not isinstance(study, Study):
            raise TypeError("study must be a Study")
        if not isinstance(task, TaskSpec):
            raise TypeError("task must be a TaskSpec")
        if model.estimate_subject_scale:
            raise PyMCBackendError(
                "estimate_subject_scale=True has no declared full-posterior scale prior; "
                "use a fixed subject_scale for this adapter"
            )
        task.validate_model(model)
        validation = task.validate(study)
        subjects = tuple(_scalar(subject) for subject in study.subjects)
        if len(subjects) < 2:
            raise PyMCBackendError("hierarchical sampling requires at least two subjects")
        outcomes = model._outcomes(study)
        features = model._base_feature_matrix(study, outcomes)
        row_subject = _subject_indices(study, subjects)
        pymc = _import_pymc()
        inference_version = importlib.metadata.version("pymc")
        sampling_seed, predictive_seed = (
            int(value) for value in np.random.SeedSequence(self.seed).generate_state(2)
        )

        coords = {
            "trial": np.arange(len(study), dtype=np.int64),
            "subject": _object_coordinate(subjects),
            "coefficient": np.asarray(model.coefficient_names),
        }
        if len(model.coefficient_names) > 1:
            coords["regularized_coefficient"] = np.asarray(model.coefficient_names[1:])

        with pymc.Model(coords=coords) as graph:
            design_matrix = pymc.Data(
                "design_matrix",
                features,
                dims=("trial", "coefficient"),
            )
            subject_index = pymc.Data("subject_index", row_subject, dims="trial")
            population_intercept = pymc.Flat("population_intercept")
            if len(model.coefficient_names) > 1:
                if model.l2 > 0:
                    population_slope = pymc.Normal(
                        "population_slope",
                        mu=0.0,
                        sigma=1.0 / np.sqrt(model.l2),
                        dims="regularized_coefficient",
                    )
                else:
                    population_slope = pymc.Flat(
                        "population_slope",
                        dims="regularized_coefficient",
                    )
                population_values = pymc.math.concatenate(
                    (pymc.math.stack((population_intercept,)), population_slope)
                )
            else:
                population_values = pymc.math.stack((population_intercept,))
            population = pymc.Deterministic(
                "population_coefficient",
                population_values,
                dims="coefficient",
            )
            deviations = pymc.Normal(
                "subject_deviation",
                mu=0.0,
                sigma=model.subject_scale,
                dims=("subject", "coefficient"),
            )
            pymc.Deterministic(
                "subject_coefficient",
                population[None, :] + deviations,
                dims=("subject", "coefficient"),
            )
            linear_predictor = pymc.math.sum(
                design_matrix * (population + deviations[subject_index]),
                axis=1,
            )
            pymc.Deterministic(
                "choice_probability",
                pymc.math.sigmoid(linear_predictor),
                dims="trial",
            )
            pymc.Bernoulli(
                model.outcome,
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
                var_names=[model.outcome],
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
        result = _retain_trial_identity(result, study, model.outcome)
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
            "population_prior": (
                "flat intercept; flat slopes"
                if model.l2 == 0
                else f"flat intercept; Normal(0, {1.0 / np.sqrt(model.l2):.17g}) slopes"
            ),
            "subject_deviation_prior": f"Normal(0, {model.subject_scale:.17g})",
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


def _import_pymc() -> Any:
    try:
        return importlib.import_module("pymc")
    except ImportError as error:
        raise PyMCUnavailableError(
            "PyMC inference requires `pip install 'unspool[bayesian]'`"
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


def _subject_indices(study: Study, subjects: tuple[Any, ...]) -> np.ndarray:
    index = {subject: position for position, subject in enumerate(subjects)}
    return np.asarray([index[_scalar(subject)] for subject in study["subject"]], dtype=np.int32)


def _object_coordinate(values: tuple[Any, ...]) -> np.ndarray:
    coordinate = np.empty(len(values), dtype=object)
    coordinate[:] = values
    return coordinate


def _scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value
