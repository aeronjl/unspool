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
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from behavio.compose.hierarchy import HierarchicalModel
from behavio.contracts.compose import group_blocks
from behavio.models._kernels.bernoulli import BernoulliLikelihood
from behavio.posterior.result import (
    PosteriorGroup,
    PosteriorResult,
    PosteriorVariable,
    posterior_result_from_arviz,
)
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
        if not isinstance(inner.likelihood, BernoulliLikelihood):
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
