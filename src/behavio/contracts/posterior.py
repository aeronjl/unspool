"""The sampler-backed estimator contract and its point-summary projection.

:class:`behavio.contracts.estimator.BehaviourEstimator` unlocks prospective comparison,
nested selection, protocol execution and model recovery, but its ``fit`` method returns a
:class:`~behavio.contracts.estimator.FitResult`. A sampler returns a
:class:`~behavio.posterior.PosteriorResult`, so today a Bayesian model can have either
prospective comparison/recovery or convergence/LOO/SBC, never both.

:class:`PosteriorBehaviourEstimator` mirrors ``BehaviourEstimator`` exactly, with ``fit``
replaced by ``sample``, and adds ``point_summary``: the explicit, lossy reduction that
lets a sampled model enter the frequentist machinery.
:func:`posterior_point_summary` is the reference implementation of that reduction.

``behavio.evaluation``, ``behavio.comparison``, ``behavio.recovery`` and
``behavio.model_recovery`` accept either contract through the
:data:`AnyBehaviourEstimator` alias and dispatch on
:func:`is_posterior_estimator`. This module stays a leaf, so it declares the contract, the
projection, and the pure helpers those callers need; the convergence gate itself lives in
``behavio.evaluation``, which is allowed to import
:func:`behavio.posterior_diagnostics.audit_posterior`. Wiring the protocol into
``behavio.runner`` remains out of scope.

Scoring with the whole posterior
--------------------------------
``predict`` and ``pointwise_log_prob`` deliberately receive the ``PosteriorResult`` rather
than the projected ``FitResult``. A sampled model must therefore never be scored at its
posterior mean by accident: the held-out log score a Bayesian model is supposed to report
is the log pointwise predictive density ``log((1/S) sum_s p(y_i | theta_s))``, which
:func:`posterior_log_predictive_density` computes from per-draw log densities.
``log p(y_i | mean(theta))`` is a different, smaller-variance and generally larger
quantity, and pooling the two in one comparison would be a category error.

Honesty of the projection
-------------------------
``FitResult`` mandates an estimate vector, matching standard errors and a full p-by-p
covariance. Those have exact posterior analogues -- the posterior mean (or median), the
posterior standard deviation, and the posterior sample covariance -- so the projection
computes them from the draws and records nothing it did not measure.

``FitDiagnostics`` additionally mandates optimizer-shaped fields that a sampler simply does
not have. Fabricating them is not an option: ``behavio.diagnostics`` raises
``nonfinite_gradient`` / ``nonfinite_hessian_condition`` / ``nonfinite_objective`` issues
for NaN values, so every projected fold would carry spurious warnings, and zeros would be a
lie. Instead ``n_iterations``, ``objective``, ``gradient_norm``, ``hessian_condition`` and
``boundary_estimate`` accept ``None`` meaning *inapplicable*, and ``audit_fit`` skips
absent diagnostics rather than warning about them. Every existing maximum-likelihood fit
passes real floats for all five, so the existing MLE audit is bit-identical.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from behavio.contracts.audit import FitDiagnostics
from behavio.contracts.estimator import (
    BehaviourEstimator,
    FitResult,
    GenerativeBehaviourModel,
    ModelCapabilities,
    ModelPrediction,
    PredictionMode,
    model_capabilities,
    validate_model_identity,
    validate_parameter_names,
)
from behavio.posterior import SAMPLE_DIMS, PosteriorError, PosteriorResult, PosteriorVariable
from behavio.study import Study

_SUMMARY_TAIL = "projected to a point summary; optimizer diagnostics are inapplicable"

POSTERIOR_SUMMARY_MESSAGE = f"posterior {_SUMMARY_TAIL}"


class PosteriorCentre(StrEnum):
    """Which posterior central tendency becomes the projected point estimate."""

    MEAN = "mean"
    MEDIAN = "median"


def posterior_summary_message(centre: PosteriorCentre) -> str:
    """Return the projected fit's diagnostic message for one declared centre."""

    return f"posterior {PosteriorCentre(centre).value} {_SUMMARY_TAIL}"


@runtime_checkable
class PosteriorBehaviourEstimator(Protocol):
    """Minimum sampling, prediction, and pointwise-scoring contract.

    Identical to :class:`~behavio.contracts.estimator.BehaviourEstimator` except that the
    fitting method is ``sample`` and returns a full :class:`PosteriorResult`, and that a
    ``point_summary`` projection is required. ``predict`` and ``pointwise_log_prob``
    receive the posterior in place of a ``FitResult``; how they reduce over draws is the
    model's decision and must be stated in its own documentation.
    """

    @property
    def model_name(self) -> str: ...

    @property
    def signature(self) -> str: ...

    @property
    def scored_columns(self) -> tuple[str, ...]: ...

    @property
    def supported_prediction_modes(self) -> tuple[PredictionMode, ...]: ...

    def sample(self, study: Study) -> PosteriorResult: ...

    def predict(
        self,
        study: Study,
        posterior: PosteriorResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> ModelPrediction: ...

    def pointwise_log_prob(
        self,
        study: Study,
        posterior: PosteriorResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> NDArray[np.float64]: ...

    def point_summary(
        self,
        posterior: PosteriorResult,
        *,
        converged: bool,
        centre: PosteriorCentre = PosteriorCentre.MEAN,
    ) -> FitResult: ...


@runtime_checkable
class GenerativePosteriorBehaviourModel(PosteriorBehaviourEstimator, Protocol):
    """A sampled estimator with named parameters and a matching simulator.

    ``parameter_names`` names the scalar quantities ``simulate`` accepts.
    ``point_summary`` names one column per *posterior coordinate*, which is a different
    vocabulary: :func:`posterior_point_summary` flattens a labelled variable into entries
    such as ``beta[coefficient='stimulus']``, and a hierarchical posterior additionally
    carries per-subject coordinates that no scalar simulator argument corresponds to. The
    two therefore do not round-trip by name, and guessing the correspondence would silently
    invalidate parameter recovery. ``posterior_parameter_labels`` is the model's explicit,
    declared mapping from each simulator parameter to its projected column;
    :func:`posterior_parameter_columns` validates it.
    """

    @property
    def parameter_names(self) -> tuple[str, ...]: ...

    @property
    def posterior_parameter_labels(self) -> Mapping[str, str]: ...

    def simulate(
        self,
        design: Study,
        parameters: Mapping[str, float],
        *,
        seed: int | np.random.Generator,
    ) -> Study: ...


#: Either estimator contract, for call sites that accept both.
AnyBehaviourEstimator = BehaviourEstimator | PosteriorBehaviourEstimator

#: Either generative contract, for simulation-backed call sites that accept both.
AnyGenerativeBehaviourModel = GenerativeBehaviourModel | GenerativePosteriorBehaviourModel


def is_posterior_estimator(model: object) -> bool:
    """Whether ``model`` should be driven through ``sample`` rather than ``fit``.

    A model that structurally satisfies both contracts is treated as frequentist, so no
    existing maximum-likelihood estimator can change behaviour by acquiring a ``sample``
    method. Implement exactly one of the two.
    """

    return not isinstance(model, BehaviourEstimator) and isinstance(
        model, PosteriorBehaviourEstimator
    )


def posterior_model_capabilities(model: PosteriorBehaviourEstimator) -> ModelCapabilities:
    """Validate and return the capabilities advertised by a sampled estimator.

    The frequentist counterpart is
    :func:`behavio.contracts.estimator.model_capabilities`; the two return the same
    :class:`ModelCapabilities` record so a sampled model can be described by the same
    capability matrix.
    """

    if not isinstance(model, PosteriorBehaviourEstimator):
        raise TypeError("model must satisfy the PosteriorBehaviourEstimator contract")
    validate_model_identity(model)
    generative = isinstance(model, GenerativePosteriorBehaviourModel)
    if generative:
        validate_parameter_names(model.parameter_names)
        _validated_parameter_labels(model)
    return ModelCapabilities(
        scored_columns=tuple(model.scored_columns),
        prediction_modes=tuple(model.supported_prediction_modes),
        can_simulate=generative,
        can_recover_parameters=generative,
    )


def any_model_capabilities(model: AnyBehaviourEstimator) -> ModelCapabilities:
    """Return the capabilities of a frequentist or a sampled estimator.

    Dispatching here rather than at every call site is what lets ``evaluate_splits``,
    ``compare_models``, ``nested_select_model`` and ``run_model_recovery`` describe both
    kinds of candidate with one capability matrix.
    """

    if is_posterior_estimator(model):
        return posterior_model_capabilities(model)  # type: ignore[arg-type]
    return model_capabilities(model)  # type: ignore[arg-type]


def posterior_log_predictive_density(
    draw_log_probabilities: NDArray[np.float64] | Sequence[Sequence[float]],
) -> NDArray[np.float64]:
    """Average per-draw log densities over draws on the probability scale.

    ``draw_log_probabilities`` has one row per retained posterior draw and one column per
    scored observation. The result is ``log((1/S) sum_s p(y_i | theta_s))``: the log
    pointwise predictive density, which is the quantity a held-out Bayesian log score
    estimates. It is deliberately not ``log p(y_i | mean(theta))``; by Jensen's inequality
    the two differ, and only this one integrates the posterior the sampler produced.

    ``-inf`` is accepted for an impossible draw and propagates only when every draw
    assigns the observation zero probability.
    """

    values = np.asarray(draw_log_probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("draw_log_probabilities must be a non-empty (draw, observation) matrix")
    if np.any(np.isnan(values)) or np.any(np.isposinf(values)):
        raise ValueError("draw log probabilities must be finite or negative infinity")
    maximum = np.max(values, axis=0)
    offset = np.where(np.isfinite(maximum), maximum, 0.0)
    with np.errstate(divide="ignore"):
        return np.log(np.mean(np.exp(values - offset), axis=0)) + offset


def posterior_draw_matrix(result: PosteriorResult) -> tuple[tuple[str, ...], NDArray[np.float64]]:
    """Flatten every declared parameter into one ``(sample, coordinate)`` draw matrix.

    Coordinate labels follow the ``name[dim=label]`` convention shared with
    :func:`behavio.posterior_diagnostics.audit_posterior`, so an audit target and a
    projected parameter name are the same string. Chains are concatenated in order, which
    is correct for any summary that treats the retained draws as one posterior sample and
    is never a substitute for a per-chain convergence diagnostic.
    """

    if not isinstance(result, PosteriorResult):
        raise TypeError("result must be a PosteriorResult")
    names: list[str] = []
    columns: list[NDArray[np.float64]] = []
    posterior = result["posterior"]
    for parameter in result.parameter_names:
        labels, samples = _flatten_variable(posterior[parameter])
        names.extend(labels)
        columns.append(samples)
    draws = np.concatenate(columns, axis=1) if len(columns) > 1 else columns[0]
    return tuple(names), draws


def posterior_parameter_columns(
    model: GenerativePosteriorBehaviourModel,
    projected_names: Sequence[str],
) -> tuple[int, ...]:
    """Locate every simulator parameter within a projected posterior's column order.

    Returns one column index per entry of ``model.parameter_names``, taken from the
    model's declared ``posterior_parameter_labels``. Any parameter whose declared label is
    absent from ``projected_names`` raises rather than being dropped: a recovery report
    that quietly compared a truth value against the wrong posterior coordinate would be
    worse than no report.
    """

    labels = _validated_parameter_labels(model)
    available = tuple(projected_names)
    positions = {name: index for index, name in enumerate(available)}
    if len(positions) != len(available):
        raise ValueError("projected parameter names must be unique")
    missing = {
        name: labels[name] for name in model.parameter_names if labels[name] not in positions
    }
    if missing:
        raise ValueError(
            "posterior_parameter_labels does not round-trip against the projected point "
            f"summary; unmatched {sorted(missing.items())!r}, available {list(available)!r}"
        )
    return tuple(positions[labels[name]] for name in model.parameter_names)


def _validated_parameter_labels(
    model: GenerativePosteriorBehaviourModel,
) -> Mapping[str, str]:
    labels = model.posterior_parameter_labels
    if not isinstance(labels, Mapping):
        raise TypeError("posterior_parameter_labels must be a mapping")
    names = tuple(model.parameter_names)
    if set(labels) != set(names):
        raise ValueError(
            "posterior_parameter_labels must name every simulator parameter exactly once; "
            f"missing={sorted(set(names) - set(labels))}, extra={sorted(set(labels) - set(names))}"
        )
    values = tuple(labels[name] for name in names)
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError("posterior_parameter_labels values must be non-empty strings")
    if len(set(values)) != len(values):
        raise ValueError("posterior_parameter_labels must map to distinct posterior coordinates")
    return labels


def posterior_point_summary(
    result: PosteriorResult,
    *,
    converged: bool,
    n_observations: int | None = None,
    centre: PosteriorCentre = PosteriorCentre.MEAN,
    message: str | None = None,
) -> FitResult:
    """Reduce a labelled posterior to the frequentist :class:`FitResult` contract.

    ``converged`` is required and is never inferred here: convergence of a sampler is the
    verdict of :func:`behavio.posterior_diagnostics.audit_posterior`, which this module
    must not import (it depends on ``behavio.contracts``). Pass
    ``audit.status is not PosteriorAuditStatus.FAIL`` or a stricter rule of your choosing.

    Parameters with intrinsic dimensions are flattened to one entry per coordinate using
    the ``name[dim=label]`` convention shared with
    :func:`behavio.posterior_diagnostics.audit_posterior`. ``n_observations`` is taken
    from the ``log_likelihood`` group, then ``observed_data``, unless given explicitly.

    The default ``message`` names the centre that produced the estimates, so a fold scored
    from a posterior is distinguishable from an optimizer fit in any report that retains
    :class:`~behavio.contracts.audit.FitDiagnostics`.

    The returned estimates, standard errors and covariance are posterior summaries, not
    sampling-theory quantities. Downstream code that reads them as asymptotic optimizer
    output will be interpreting them incorrectly.
    """

    if not isinstance(result, PosteriorResult):
        raise TypeError("result must be a PosteriorResult")
    if not isinstance(converged, bool):
        raise TypeError("converged must be boolean")
    selected_centre = PosteriorCentre(centre)
    summary_message = (
        posterior_summary_message(selected_centre) if message is None else str(message)
    )
    if not summary_message:
        raise ValueError("message must be a non-empty string")

    names, draws = posterior_draw_matrix(result)
    if draws.shape[0] < 2:
        raise PosteriorError("a point summary requires at least two posterior draws")
    if not np.all(np.isfinite(draws)):
        raise PosteriorError("posterior draws must be finite to project a point summary")

    if selected_centre is PosteriorCentre.MEDIAN:
        estimates = np.median(draws, axis=0)
    else:
        estimates = np.mean(draws, axis=0)
    standard_errors = np.std(draws, axis=0, ddof=1)
    covariance = np.atleast_2d(np.cov(draws, rowvar=False, ddof=1))

    observations = _infer_n_observations(result) if n_observations is None else int(n_observations)
    if observations < 1:
        raise PosteriorError("n_observations must be positive")

    return FitResult(
        model_name=result.model_name,
        model_signature=result.model_signature,
        parameter_names=tuple(names),
        estimates=estimates,
        standard_errors=standard_errors,
        covariance=covariance,
        n_observations=observations,
        diagnostics=FitDiagnostics(
            converged=converged,
            optimizer=f"{result.inference_library}/{result.inference_library_version}",
            status=0 if converged else 1,
            message=summary_message,
            n_iterations=None,
            objective=None,
            gradient_norm=None,
            hessian_condition=None,
            boundary_estimate=None,
        ),
    )


def _flatten_variable(
    variable: PosteriorVariable,
) -> tuple[tuple[str, ...], NDArray[np.float64]]:
    dims = variable.dims
    missing = [dim for dim in SAMPLE_DIMS if dim not in dims]
    if missing:
        raise PosteriorError(f"posterior variable {variable.name!r} is missing {missing}")
    values = np.asarray(variable.values, dtype=np.float64)
    sample_axes = [dims.index(dim) for dim in SAMPLE_DIMS]
    ordered = np.moveaxis(values, sample_axes, range(len(SAMPLE_DIMS)))
    intrinsic = variable.intrinsic_dims
    n_samples = int(np.prod(ordered.shape[: len(SAMPLE_DIMS)]))
    flat = ordered.reshape(n_samples, -1)
    if not intrinsic:
        return (variable.name,), flat
    labels = tuple(
        "{}[{}]".format(
            variable.name,
            ",".join(
                f"{dim}={_python_scalar(variable.coords[dim][position])!r}"
                for dim, position in zip(intrinsic, index, strict=True)
            ),
        )
        for index in np.ndindex(*(len(variable.coords[dim]) for dim in intrinsic))
    )
    return labels, flat


def _infer_n_observations(result: PosteriorResult) -> int:
    for group_name in ("log_likelihood", "observed_data"):
        if group_name not in result.group_names:
            continue
        group = result[group_name]
        variable = group.variables[0]
        sizes = [len(variable.coords[dim]) for dim in variable.intrinsic_dims]
        if sizes:
            return int(np.prod(sizes))
    raise PosteriorError(
        "n_observations could not be inferred; supply it explicitly or retain a "
        "log_likelihood or observed_data group"
    )


def _python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value
