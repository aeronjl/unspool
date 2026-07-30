"""A runnable conformance harness for :class:`behavio.contracts.BehaviourEstimator`.

``docs/extensions.md`` lists nine things an estimator extension should test.
:func:`behavio.adapters.conformance.check_study_adapter` executes the six a *data* adapter
owes; this module executes the estimator half, and adds the one check the contract most
needs and could not previously make.

Why a behavioural check for ``SMOOTHED``
----------------------------------------
``AGENTS.md`` requires Behavio to "distinguish filtered predictions from smoothed
descriptions", and :class:`~behavio.contracts.PredictionMode` names the distinction. But the
mode is a *label the model writes on its own output*. Every first-party model declares
``(FILTERED,)`` and none implements ``SMOOTHED``, so the label has never been contradicted;
a wrapper around ``ssm.most_likely_states`` or a ``dynamax`` smoother, both smoothed by
construction, would return smoothed states, stamp them ``FILTERED``, and satisfy every
structural check in the package. A contract that cannot detect its own violation is not a
contract.

Nothing about a single prediction reveals which information set produced it. What does is
the *definition*: a filtered prediction for trial *t* is a function of trials up to *t*, so
changing trials after *t* cannot change it, and a smoothed one is a function of the whole
sequence, so changing trials after *t* generally does. That is directly testable. The
harness *relabels* the observations in the second half of every trial sequence, holding the
fit fixed, and re-predicts:

- a prediction or pointwise score labelled ``FILTERED`` must be unchanged on the first half;
- a prediction labelled ``SMOOTHED``, if the model advertises the mode, must change.

The check is falsifiable in both directions and needs no cooperation from the model beyond
the contract it already implements. It has one honest limit, recorded as
:attr:`~behavio.adapters.conformance.CheckStatus.SKIPPED` rather than hidden: if the
perturbation is a no-op -- a constant column, a two-row sequence -- there is nothing to
detect, and the harness says so instead of passing.

A second, weaker check comes free. Because the fit is held fixed and only the *inputs* move,
any dependence of an early row's prediction on a late row's value is caught, including
preprocessing fitted inside ``predict`` -- a column standardised against the test study's own
mean is leakage of the same kind and fails the same check.

A sampled model reaches the same checks
---------------------------------------
:func:`check_behaviour_estimator` drives ``fit`` and hands the resulting
:class:`~behavio.contracts.FitResult` back to ``predict``.  A
:class:`~behavio.contracts.posterior.PosteriorBehaviourEstimator` has ``sample`` and takes a
:class:`~behavio.posterior.PosteriorResult`, so for a while no sampled model could reach the
two leakage checks or the filtered-versus-smoothed check without its author writing an
adapter -- which is to say, without the *caller* supplying the part of the harness that was
missing. That is backwards, and it withheld the checks from exactly the models that most
need them: a sampled latent-state model is where a smoothed description is most likely to be
labelled a filtered prediction.

:func:`check_posterior_behaviour_estimator` is the sampled entry point. It samples, audits
the posterior, projects a point summary, and then runs the *same* check bodies, because
everything after fitting depends only on two things: the object ``predict`` and
``pointwise_log_prob`` are handed (a ``FitResult`` here, a ``PosteriorResult`` there) and the
:class:`~behavio.contracts.FitResult` the identity check reads. Two checks are specific to
the sampled contract and have no frequentist counterpart -- that ``sample`` returns a
labelled posterior belonging to this specification, and that ``point_summary`` records the
convergence verdict it was *given* rather than one of its own, which is the invariant
``behavio.evaluate.folds`` enforces on every fold.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from behavio.adapters.conformance import CheckStatus, ConformanceCheck
from behavio.contracts.estimator import (
    CategoricalPrediction,
    DensityBehaviourEstimator,
    DensityPrediction,
    FitResult,
    GenerativeBehaviourModel,
    ModelCapabilities,
    Prediction,
    PredictionMode,
    UnsupportedPredictionMode,
    model_capabilities,
)
from behavio.contracts.posterior import (
    GenerativePosteriorBehaviourModel,
    PosteriorCentre,
    posterior_draw_matrix,
    posterior_model_capabilities,
    posterior_parameter_columns,
)
from behavio.posterior.diagnostics import (
    PosteriorAuditPolicy,
    PosteriorAuditStatus,
    audit_posterior,
)
from behavio.posterior.result import PosteriorResult
from behavio.trials import REQUIRED_COLUMNS, Study, sequence_layout

_TOLERANCE = 1e-9
"""Relative agreement two recomputations of the same quantity must reach.

Not exact equality: a vectorised likelihood evaluated over two arrays that differ only in
rows it must ignore may still sum in a different order. A leak, by contrast, moves a
prediction by orders of magnitude more than this.
"""


class EstimatorConformanceError(AssertionError):
    """Raised by :func:`assert_behaviour_estimator_conforms` when a required check fails."""


@dataclass(frozen=True, slots=True)
class EstimatorConformance:
    """The complete result of running the harness against one estimator."""

    capabilities: ModelCapabilities | None
    checks: tuple[ConformanceCheck, ...]

    @property
    def passed(self) -> bool:
        """True when no check failed. Skipped checks do not fail a run."""

        return not self.failures

    @property
    def failures(self) -> tuple[ConformanceCheck, ...]:
        """Checks that failed, in execution order."""

        return tuple(check for check in self.checks if check.status is CheckStatus.FAILED)

    @property
    def skipped(self) -> tuple[ConformanceCheck, ...]:
        """Checks the study or the model did not supply enough evidence to run."""

        return tuple(check for check in self.checks if check.status is CheckStatus.SKIPPED)

    def summary(self) -> str:
        """A single readable line per check, suitable for assertion messages."""

        return "\n".join(
            f"{check.status.value}: {check.name}: {check.detail}" for check in self.checks
        )


def check_behaviour_estimator(
    model: Any,
    study: Study,
    *,
    fit: FitResult | None = None,
    simulation_parameters: Mapping[str, float] | None = None,
    seed: int = 0,
) -> EstimatorConformance:
    """Run the estimator conformance checks and return what each one observed.

    Args:
        model: The estimator under test.
        study: A small study the model can fit and predict. It should contain at least two
            trial sequences of four or more trials each, so the leakage checks have a past
            and a future to separate.
        fit: A fit to predict and score with. Omitted, the harness calls ``model.fit(study)``
            once and reuses the result, which is also what the identity check inspects.
        simulation_parameters: Parameters for the simulator check. Omitted for a generative
            model, the harness uses the fitted estimates.
        seed: Seed for the simulator determinism check.

    Returns:
        An :class:`EstimatorConformance` record; ``passed`` is False if any check failed.
    """

    if not isinstance(study, Study):
        raise TypeError("study must be a Study")
    checks: list[ConformanceCheck] = []
    try:
        capabilities = model_capabilities(model)
    except (TypeError, ValueError) as error:
        checks.append(
            ConformanceCheck(
                "declares-model-identity",
                CheckStatus.FAILED,
                f"model_capabilities() rejected the estimator: {error}",
            )
        )
        return EstimatorConformance(capabilities=None, checks=tuple(checks))
    checks.append(
        ConformanceCheck(
            "declares-model-identity",
            CheckStatus.FAILED if not capabilities.prediction_modes else CheckStatus.PASSED,
            f"{model.model_name} scoring {list(capabilities.scored_columns)} in modes "
            f"{[mode.value for mode in capabilities.prediction_modes]}",
        )
    )

    fitted = fit
    if fitted is None:
        try:
            fitted = model.fit(study)
        except Exception as error:  # any failure here is a conformance failure
            checks.append(
                ConformanceCheck(
                    "fit-reports-the-training-study",
                    CheckStatus.FAILED,
                    f"fit() raised {type(error).__name__}: {error}",
                )
            )
            return EstimatorConformance(capabilities=capabilities, checks=tuple(checks))
    checks.append(_check_fit_identity(model, study, fitted))
    checks.extend(
        _shared_checks(
            model,
            study,
            capabilities,
            context=fitted,
            fit=fitted,
            simulation_parameters=simulation_parameters,
            generative_contract=GenerativeBehaviourModel,
            seed=seed,
        )
    )
    return EstimatorConformance(capabilities=capabilities, checks=tuple(checks))


def check_posterior_behaviour_estimator(
    model: Any,
    study: Study,
    *,
    posterior: PosteriorResult | None = None,
    audit_policy: PosteriorAuditPolicy | None = None,
    centre: PosteriorCentre = PosteriorCentre.MEAN,
    simulation_parameters: Mapping[str, float] | None = None,
    seed: int = 0,
) -> EstimatorConformance:
    """Run the conformance checks against a sampled estimator and report each observation.

    The sampled counterpart of :func:`check_behaviour_estimator`, and the reason a
    :class:`~behavio.contracts.posterior.PosteriorBehaviourEstimator` no longer needs its
    caller to write an adapter before the leakage checks will run. Real sampling happens
    once: ``sample`` is called, :func:`~behavio.posterior.audit_posterior` decides the
    convergence verdict, ``point_summary`` projects it, and every subsequent check receives
    the :class:`~behavio.posterior.PosteriorResult` -- never the projection -- exactly as
    ``behavio.evaluate.folds`` does when it scores a fold.

    Args:
        model: The sampled estimator under test.
        study: A small study the model can sample and predict. It should contain at least
            two trial sequences of four or more trials each, so the leakage checks have a
            past and a future to separate.
        posterior: A posterior to predict and score with. Omitted, the harness calls
            ``model.sample(study)`` once and reuses the result.
        audit_policy: Thresholds for the convergence audit whose verdict is handed to
            ``point_summary``. The audit's *outcome* is not a conformance verdict -- an
            under-sampled study is not a broken contract -- but whether the model records
            the verdict it was given is.
        centre: Posterior central tendency the projection is asked for.
        simulation_parameters: Parameters for the simulator check. Omitted for a generative
            model, the harness reads the projected posterior summary through the model's own
            ``posterior_parameter_labels``, because the projected coordinate names and the
            simulator's scalar names are two vocabularies.
        seed: Seed for the simulator determinism check.

    Returns:
        An :class:`EstimatorConformance` record; ``passed`` is False if any check failed.
    """

    if not isinstance(study, Study):
        raise TypeError("study must be a Study")
    checks: list[ConformanceCheck] = []
    try:
        capabilities = posterior_model_capabilities(model)
    except (TypeError, ValueError) as error:
        checks.append(
            ConformanceCheck(
                "declares-model-identity",
                CheckStatus.FAILED,
                f"posterior_model_capabilities() rejected the estimator: {error}",
            )
        )
        return EstimatorConformance(capabilities=None, checks=tuple(checks))
    checks.append(
        ConformanceCheck(
            "declares-model-identity",
            CheckStatus.FAILED if not capabilities.prediction_modes else CheckStatus.PASSED,
            f"{model.model_name} sampling {list(capabilities.scored_columns)} in modes "
            f"{[mode.value for mode in capabilities.prediction_modes]}",
        )
    )

    sampled = posterior
    if sampled is None:
        try:
            sampled = model.sample(study)
        except Exception as error:  # any failure here is a conformance failure
            checks.append(
                ConformanceCheck(
                    "samples-the-training-study",
                    CheckStatus.FAILED,
                    f"sample() raised {type(error).__name__}: {error}",
                )
            )
            return EstimatorConformance(capabilities=capabilities, checks=tuple(checks))
    identity, verdict = _check_posterior_identity(model, sampled)
    checks.append(identity)
    projection, projected = _check_projected_verdict(
        model, study, sampled, verdict, centre, audit_policy
    )
    checks.append(projection)
    if projected is None:
        return EstimatorConformance(capabilities=capabilities, checks=tuple(checks))
    checks.extend(
        _shared_checks(
            model,
            study,
            capabilities,
            context=sampled,
            fit=projected,
            simulation_parameters=(
                _posterior_simulation_parameters(model, projected)
                if simulation_parameters is None and capabilities.can_simulate
                else simulation_parameters
            ),
            generative_contract=GenerativePosteriorBehaviourModel,
            seed=seed,
        )
    )
    return EstimatorConformance(capabilities=capabilities, checks=tuple(checks))


def _shared_checks(
    model: Any,
    study: Study,
    capabilities: ModelCapabilities,
    *,
    context: Any,
    fit: FitResult,
    simulation_parameters: Mapping[str, float] | None,
    generative_contract: type[Any],
    seed: int,
) -> list[ConformanceCheck]:
    """Every check that depends only on the object ``predict`` is handed.

    ``context`` is a :class:`~behavio.contracts.FitResult` for an optimizer-fitted model and
    a :class:`~behavio.posterior.PosteriorResult` for a sampled one; ``fit`` is the
    :class:`FitResult` either way, because the simulator check reads its estimates. Nothing
    below this line knows which contract it is checking, which is the point: the leakage
    checks a sampled model most needs are the *same* checks, not sampled analogues of them.
    """

    checks = [
        _check_prediction_shape(model, study, context),
        _check_pointwise_scores(model, study, context),
        _check_undeclared_modes_refused(model, study, context, capabilities),
    ]
    checks.extend(_check_information_set(model, study, context, capabilities))
    checks.append(_check_density_agrees_with_choice(model, study, context))
    checks.append(
        _check_simulator(
            model,
            study,
            capabilities,
            simulation_parameters,
            fit,
            seed,
            generative_contract,
        )
    )
    return checks


def assert_behaviour_estimator_conforms(
    model: Any,
    study: Study,
    *,
    fit: FitResult | None = None,
    simulation_parameters: Mapping[str, float] | None = None,
    seed: int = 0,
    require_complete: bool = False,
) -> EstimatorConformance:
    """Run :func:`check_behaviour_estimator` and raise on any failure.

    Set ``require_complete=True`` to also reject a run in which a check could not be
    executed -- a study too short to separate a past from a future, a perturbation that
    changed nothing. That is the strict mode a wrapper's own test suite should use, because
    a skipped leakage check is not evidence of a filtered prediction.
    """

    report = check_behaviour_estimator(
        model,
        study,
        fit=fit,
        simulation_parameters=simulation_parameters,
        seed=seed,
    )
    _raise_on_failure(report, require_complete=require_complete)
    return report


def assert_posterior_behaviour_estimator_conforms(
    model: Any,
    study: Study,
    *,
    posterior: PosteriorResult | None = None,
    audit_policy: PosteriorAuditPolicy | None = None,
    centre: PosteriorCentre = PosteriorCentre.MEAN,
    simulation_parameters: Mapping[str, float] | None = None,
    seed: int = 0,
    require_complete: bool = False,
) -> EstimatorConformance:
    """Run :func:`check_posterior_behaviour_estimator` and raise on any failure."""

    report = check_posterior_behaviour_estimator(
        model,
        study,
        posterior=posterior,
        audit_policy=audit_policy,
        centre=centre,
        simulation_parameters=simulation_parameters,
        seed=seed,
    )
    _raise_on_failure(report, require_complete=require_complete)
    return report


def _raise_on_failure(report: EstimatorConformance, *, require_complete: bool) -> None:
    incomplete = report.skipped if require_complete else ()
    if not report.passed or incomplete:
        raise EstimatorConformanceError(
            "behaviour estimator conformance failed:\n" + report.summary()
        )


def perturb_future_rows(study: Study, *, columns: Sequence[str]) -> tuple[Study, NDArray[np.intp]]:
    """Return a study whose later trials are relabelled, and the rows that were left alone.

    The second half of every trial sequence has each named column mapped through one cyclic
    relabelling of that column's own distinct values: every value becomes the next distinct
    value the column contains, wrapping at the end. Two properties make that the right
    perturbation and a within-block shuffle the wrong one.

    It stays **admissible**. Every replacement is a value the column already holds, so a
    choice stays a choice, a response time stays positive, and a stimulus stays on the
    levels the task used; the study contract still holds, because identity and chronology
    columns are never touched.

    And it changes the block's **multiset**, not merely its order. A shuffle does not: any
    order-insensitive function of the future -- a mean, a count, a proportion correct -- is
    invariant under it, so a leaky prediction built from one would slip through unchanged. A
    relabelling changes what the future *contains*, which no function of the future can
    ignore.

    The returned indices are the rows *before* each sequence's midpoint: the rows whose
    filtered predictions must be unchanged, and whose smoothed predictions must not be.
    """

    if not isinstance(study, Study):
        raise TypeError("study must be a Study")
    perturbable = [
        name for name in columns if name in study.columns and name not in REQUIRED_COLUMNS
    ]
    layout = sequence_layout(study)
    values = {name: np.array(study[name], copy=True) for name in study.columns}
    relabel = {
        name: mapping
        for name in perturbable
        if (mapping := _cyclic_relabelling(np.asarray(study[name]))) is not None
    }
    past: list[int] = []
    for sequence in layout:
        midpoint = len(sequence) // 2
        if midpoint == 0:
            continue
        past.extend(int(index) for index in sequence.indices[:midpoint])
        future = sequence.indices[midpoint:]
        for name, mapping in relabel.items():
            values[name][future] = [mapping[_scalar(value)] for value in values[name][future]]
    return Study(values), np.asarray(sorted(past), dtype=np.intp)


def _cyclic_relabelling(values: NDArray[Any]) -> dict[Any, Any] | None:
    """Map each distinct value of a column to the next one, or ``None`` if it is constant."""

    try:
        pool = list(dict.fromkeys(_scalar(value) for value in values))
    except TypeError:  # pragma: no cover - an unhashable column cannot be relabelled
        return None
    if len(pool) < 2:
        return None
    return {value: pool[(index + 1) % len(pool)] for index, value in enumerate(pool)}


def _scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _check_fit_identity(model: Any, study: Study, fit: Any) -> ConformanceCheck:
    name = "fit-reports-the-training-study"
    if not isinstance(fit, FitResult):
        return ConformanceCheck(
            name, CheckStatus.FAILED, f"fit() returned {type(fit).__name__}, not a FitResult"
        )
    problems = []
    if fit.model_name != model.model_name:
        problems.append(f"model_name {fit.model_name!r} != {model.model_name!r}")
    if fit.model_signature != model.signature:
        problems.append("model_signature does not match the estimator's signature")
    if fit.n_observations != len(study):
        problems.append(f"n_observations {fit.n_observations} != {len(study)} training rows")
    declared = tuple(getattr(model, "parameter_names", fit.parameter_names))
    if declared != fit.parameter_names:
        problems.append("parameter_names disagree with the estimator's declaration")
    if problems:
        return ConformanceCheck(name, CheckStatus.FAILED, "; ".join(problems))
    return ConformanceCheck(
        name,
        CheckStatus.PASSED,
        f"{len(fit.parameter_names)} parameters over {fit.n_observations} rows, "
        f"convergence {fit.diagnostics.convergence.value}",
    )


def _check_posterior_identity(model: Any, posterior: Any) -> tuple[ConformanceCheck, bool]:
    """Check the sampled counterpart of ``fit-reports-the-training-study``.

    The one asymmetry worth stating: a sampled model's ``parameter_names`` are the
    *simulator's* scalar arguments, while the projection names one column per posterior
    coordinate, so the two tuples are not required to be equal and comparing them would be
    wrong. What is required is that the model's declared
    ``posterior_parameter_labels`` locate every simulator parameter inside the posterior it
    just produced, which is what :func:`posterior_parameter_columns` decides.

    The returned flag is whether the posterior converged under the audit, so the projection
    check that follows can hand ``point_summary`` a verdict it did not choose itself.
    """

    name = "samples-the-training-study"
    if not isinstance(posterior, PosteriorResult):
        return (
            ConformanceCheck(
                name,
                CheckStatus.FAILED,
                f"sample() returned {type(posterior).__name__}, not a PosteriorResult",
            ),
            False,
        )
    problems = []
    if posterior.model_name != model.model_name:
        problems.append(f"model_name {posterior.model_name!r} != {model.model_name!r}")
    if posterior.model_signature != model.signature:
        problems.append("model_signature does not match the estimator's signature")
    if not posterior.parameter_names:
        problems.append("the posterior declares no parameters")
    try:
        labels, draws = posterior_draw_matrix(posterior)
    except Exception as error:
        return (
            ConformanceCheck(
                name,
                CheckStatus.FAILED,
                f"the posterior cannot be flattened into draws: {type(error).__name__}: {error}",
            ),
            False,
        )
    if isinstance(model, GenerativePosteriorBehaviourModel):
        try:
            posterior_parameter_columns(model, labels)
        except (TypeError, ValueError) as error:
            problems.append(f"posterior_parameter_labels do not reach this posterior: {error}")
    if problems:
        return ConformanceCheck(name, CheckStatus.FAILED, "; ".join(problems)), False
    audit = audit_posterior(posterior)
    return (
        ConformanceCheck(
            name,
            CheckStatus.PASSED,
            f"{len(labels)} posterior coordinates over {draws.shape[0]} draws from "
            f"{posterior.inference_library}, convergence {audit.status.value}",
        ),
        audit.status is not PosteriorAuditStatus.FAIL,
    )


def _check_projected_verdict(
    model: Any,
    study: Study,
    posterior: Any,
    converged: bool,
    centre: PosteriorCentre,
    audit_policy: PosteriorAuditPolicy | None,
) -> tuple[ConformanceCheck, FitResult | None]:
    """Check that ``point_summary`` records the verdict it is handed, not one of its own.

    This is the invariant :func:`behavio.evaluate.folds.evaluate_splits` enforces on every
    sampled fold, and a model that quietly overrides it would make its own folds eligible
    after a failed convergence audit. It is checked in *both* directions -- the honest
    verdict, and its negation -- because a model that hard-codes ``converged=True`` passes
    the first and fails the second.
    """

    name = "projects-the-convergence-verdict-it-is-given"
    observations = len(study)
    if audit_policy is not None:
        converged = (
            audit_posterior(posterior, policy=audit_policy).status is not PosteriorAuditStatus.FAIL
        )
    try:
        projected = model.point_summary(posterior, converged=converged, centre=centre)
        negated = model.point_summary(posterior, converged=not converged, centre=centre)
    except Exception as error:  # surfaced as a conformance failure
        return (
            ConformanceCheck(
                name, CheckStatus.FAILED, f"point_summary() raised {type(error).__name__}: {error}"
            ),
            None,
        )
    if not isinstance(projected, FitResult) or not isinstance(negated, FitResult):
        return (
            ConformanceCheck(
                name,
                CheckStatus.FAILED,
                f"point_summary() returned {type(projected).__name__}, not a FitResult",
            ),
            None,
        )
    problems = []
    if projected.model_name != model.model_name:
        problems.append(f"model_name {projected.model_name!r} != {model.model_name!r}")
    if projected.model_signature != model.signature:
        problems.append("the projected summary does not carry the estimator's signature")
    if projected.n_observations != observations:
        problems.append(f"n_observations {projected.n_observations} != {observations} sampled rows")
    if projected.diagnostics.converged is not converged:
        problems.append(f"asked for converged={converged}, recorded {projected.diagnostics}")
    if negated.diagnostics.converged is converged:
        problems.append(
            "point_summary reports the same convergence whichever verdict it is handed, so "
            "a failed convergence audit could not make one of its folds ineligible"
        )
    if problems:
        return ConformanceCheck(name, CheckStatus.FAILED, "; ".join(problems)), None
    return (
        ConformanceCheck(
            name,
            CheckStatus.PASSED,
            f"{len(projected.parameter_names)} projected coordinates over "
            f"{projected.n_observations} rows at the posterior {PosteriorCentre(centre).value}, "
            f"convergence recorded as {converged} and as {not converged} when negated",
        ),
        projected,
    )


def _posterior_simulation_parameters(model: Any, fit: FitResult) -> Mapping[str, float] | None:
    """Read the simulator's parameters out of the projection through the declared labels."""

    try:
        columns = posterior_parameter_columns(model, fit.parameter_names)
    except (TypeError, ValueError):  # reported by the identity check, not here
        return None
    return {
        name: float(fit.estimates[column])
        for name, column in zip(model.parameter_names, columns, strict=True)
    }


def _check_prediction_shape(model: Any, study: Study, fit: FitResult) -> ConformanceCheck:
    name = "predicts-one-row-per-trial"
    try:
        prediction = model.predict(study, fit)
    except Exception as error:  # surfaced as a conformance failure
        return ConformanceCheck(
            name, CheckStatus.FAILED, f"predict() raised {type(error).__name__}: {error}"
        )
    if not isinstance(prediction, (Prediction, CategoricalPrediction, DensityPrediction)):
        return ConformanceCheck(
            name,
            CheckStatus.FAILED,
            f"predict() returned {type(prediction).__name__}, which no Behavio consumer reads",
        )
    if prediction.n_observations != len(study):
        return ConformanceCheck(
            name,
            CheckStatus.FAILED,
            f"predict() returned {prediction.n_observations} rows for {len(study)} trials",
        )
    if prediction.mode is not PredictionMode.FILTERED:
        return ConformanceCheck(
            name,
            CheckStatus.FAILED,
            f"predict() defaulted to {prediction.mode.value}, not filtered",
        )
    return ConformanceCheck(
        name,
        CheckStatus.PASSED,
        f"{type(prediction).__name__} over {prediction.n_observations} rows, mode "
        f"{prediction.mode.value}",
    )


def _check_pointwise_scores(model: Any, study: Study, fit: FitResult) -> ConformanceCheck:
    name = "scores-one-row-per-trial"
    try:
        scores = np.asarray(model.pointwise_log_prob(study, fit), dtype=np.float64)
    except Exception as error:  # surfaced as a conformance failure
        return ConformanceCheck(
            name,
            CheckStatus.FAILED,
            f"pointwise_log_prob() raised {type(error).__name__}: {error}",
        )
    if scores.ndim != 1 or scores.size != len(study):
        return ConformanceCheck(
            name,
            CheckStatus.FAILED,
            f"pointwise_log_prob() returned shape {scores.shape} for {len(study)} trials",
        )
    if np.any(np.isnan(scores)) or np.any(np.isposinf(scores)):
        return ConformanceCheck(name, CheckStatus.FAILED, "pointwise scores contain NaN or +inf")
    return ConformanceCheck(
        name,
        CheckStatus.PASSED,
        f"{scores.size} finite-or-floored scores summing to {float(np.sum(scores)):.4f}",
    )


def _check_undeclared_modes_refused(
    model: Any,
    study: Study,
    fit: FitResult,
    capabilities: ModelCapabilities,
) -> ConformanceCheck:
    name = "refuses-undeclared-prediction-modes"
    undeclared = [mode for mode in PredictionMode if mode not in capabilities.prediction_modes]
    if not undeclared:
        return ConformanceCheck(
            name, CheckStatus.SKIPPED, "the model declares every prediction mode"
        )
    for mode in undeclared:
        try:
            model.predict(study, fit, mode=mode)
        except UnsupportedPredictionMode:
            continue
        except Exception as error:
            return ConformanceCheck(
                name,
                CheckStatus.FAILED,
                f"predict(mode={mode.value}) raised {type(error).__name__} rather than "
                "UnsupportedPredictionMode",
            )
        return ConformanceCheck(
            name,
            CheckStatus.FAILED,
            f"predict(mode={mode.value}) returned a prediction the model does not declare",
        )
    return ConformanceCheck(
        name,
        CheckStatus.PASSED,
        f"refused {[mode.value for mode in undeclared]} with UnsupportedPredictionMode",
    )


def _check_information_set(
    model: Any,
    study: Study,
    fit: FitResult,
    capabilities: ModelCapabilities,
) -> list[ConformanceCheck]:
    """The two leakage checks and, when the model claims it, the smoothing check."""

    filtered_name = "filtered-prediction-ignores-future-rows"
    score_name = "filtered-score-ignores-future-rows"
    smoothed_name = "smoothed-prediction-uses-future-rows"
    read = tuple(capabilities.scored_columns) + tuple(capabilities.required_task_columns)
    perturbed, past = perturb_future_rows(study, columns=read)
    if not past.size:
        detail = "no trial sequence in this study has a first half to protect"
        return [
            ConformanceCheck(filtered_name, CheckStatus.SKIPPED, detail),
            ConformanceCheck(score_name, CheckStatus.SKIPPED, detail),
        ]
    moved = [
        name
        for name in read
        if name in study.columns
        and not _columns_equal(np.asarray(study[name]), np.asarray(perturbed[name]))
    ]
    if not moved:
        detail = (
            "permuting the later trials of every sequence changed no column the model "
            f"reads ({list(read)}); the study cannot distinguish filtering from smoothing"
        )
        return [
            ConformanceCheck(filtered_name, CheckStatus.SKIPPED, detail),
            ConformanceCheck(score_name, CheckStatus.SKIPPED, detail),
        ]

    checks = [
        _compare_past_rows(
            filtered_name,
            past,
            lambda source: _prediction_values(model.predict(source, fit)),
            study,
            perturbed,
            moved,
            must_change=False,
        ),
        _compare_past_rows(
            score_name,
            past,
            lambda source: np.asarray(model.pointwise_log_prob(source, fit), dtype=np.float64),
            study,
            perturbed,
            moved,
            must_change=False,
        ),
    ]
    if PredictionMode.SMOOTHED in capabilities.prediction_modes:
        checks.append(
            _compare_past_rows(
                smoothed_name,
                past,
                lambda source: _prediction_values(
                    model.predict(source, fit, mode=PredictionMode.SMOOTHED)
                ),
                study,
                perturbed,
                moved,
                must_change=True,
            )
        )
    return checks


def _compare_past_rows(
    name: str,
    past: NDArray[np.intp],
    evaluate: Callable[[Study], NDArray[np.float64]],
    study: Study,
    perturbed: Study,
    moved: Sequence[str],
    *,
    must_change: bool,
) -> ConformanceCheck:
    try:
        original = evaluate(study)
        replayed = evaluate(perturbed)
    except Exception as error:  # surfaced as a conformance failure
        return ConformanceCheck(
            name, CheckStatus.FAILED, f"re-evaluation raised {type(error).__name__}: {error}"
        )
    if original.shape != replayed.shape or original.shape[0] != len(study):
        return ConformanceCheck(
            name,
            CheckStatus.FAILED,
            f"the two evaluations have shapes {original.shape} and {replayed.shape}",
        )
    left = np.atleast_2d(original[past].T).T
    right = np.atleast_2d(replayed[past].T).T
    scale = np.maximum(np.abs(left), np.abs(right))
    deviation = np.abs(left - right) / np.where(scale > 1.0, scale, 1.0)
    worst = float(np.max(deviation)) if deviation.size else 0.0
    if must_change:
        if worst <= _TOLERANCE:
            return ConformanceCheck(
                name,
                CheckStatus.FAILED,
                f"the model declares SMOOTHED, but relabelling {list(moved)} in the later "
                f"half of every sequence moved its smoothed output on the earlier half by "
                f"at most {worst:.3e}; a smoothed estimate is a function of the whole "
                "sequence, so this is a filtered estimate wearing a smoothed label",
            )
        return ConformanceCheck(
            name,
            CheckStatus.PASSED,
            f"the smoothed output on {past.size} earlier rows responds to later trials "
            f"(largest relative change {worst:.3e})",
        )
    if worst > _TOLERANCE:
        return ConformanceCheck(
            name,
            CheckStatus.FAILED,
            f"relabelling {list(moved)} in the later half of every sequence changed the "
            f"result on {int(np.sum(np.any(deviation > _TOLERANCE, axis=1)))} of "
            f"{past.size} earlier rows (largest relative change {worst:.3e}); a filtered "
            "quantity for trial t cannot depend on trials after t",
        )
    return ConformanceCheck(
        name,
        CheckStatus.PASSED,
        f"{past.size} earlier rows are unchanged when {list(moved)} is relabelled in the "
        "later half of every sequence",
    )


def _check_density_agrees_with_choice(model: Any, study: Study, fit: FitResult) -> ConformanceCheck:
    name = "density-agrees-with-the-choice-prediction"
    if not isinstance(model, DensityBehaviourEstimator):
        return ConformanceCheck(
            name,
            CheckStatus.SKIPPED,
            "the model predicts no continuous outcome, so there is nothing to reconcile",
        )
    try:
        density = model.predict_density(study, fit)
        prediction = model.predict(study, fit)
    except Exception as error:  # surfaced as a conformance failure
        return ConformanceCheck(
            name, CheckStatus.FAILED, f"prediction raised {type(error).__name__}: {error}"
        )
    if not isinstance(density, DensityPrediction):
        return ConformanceCheck(
            name,
            CheckStatus.FAILED,
            f"predict_density() returned {type(density).__name__}, not a DensityPrediction",
        )
    if density.outcome != model.density_outcome:
        return ConformanceCheck(
            name,
            CheckStatus.FAILED,
            f"predict_density() names {density.outcome!r}, the model declares "
            f"{model.density_outcome!r}",
        )
    if not density.is_defective:
        return ConformanceCheck(
            name,
            CheckStatus.PASSED,
            "the density is unlabelled, so it makes no claim about the discrete prediction; "
            f"mass in [{float(np.min(density.total_mass)):.4f}, "
            f"{float(np.max(density.total_mass)):.4f}]",
        )
    implied = density.choice_prediction()
    if isinstance(prediction, DensityPrediction):
        # ``predict`` returns the density itself, which is the shape this check most wants
        # to see: there are not two halves to reconcile because there is only one object.
        # What is still worth checking is that the two calls agree, since a model may build
        # the density twice.
        if prediction.categories != density.categories or not np.array_equal(
            np.asarray(prediction.density), np.asarray(density.density)
        ):
            return ConformanceCheck(
                name,
                CheckStatus.FAILED,
                "predict() and predict_density() returned different densities",
            )
        return ConformanceCheck(
            name,
            CheckStatus.PASSED,
            "predict() returns the density itself, so the discrete and continuous halves "
            "cannot disagree; grid mass in "
            f"[{float(np.min(density.total_mass)):.4f}, "
            f"{float(np.max(density.total_mass)):.4f}]",
        )
    if isinstance(prediction, Prediction):
        if len(implied.categories) != 2:
            return ConformanceCheck(
                name,
                CheckStatus.FAILED,
                "a binary Prediction cannot be reconciled with a density over "
                f"{len(implied.categories)} categories",
            )
        observed = np.asarray(prediction.probability, dtype=np.float64)
        expected = np.asarray(implied.probability[:, -1], dtype=np.float64)
    elif isinstance(prediction, CategoricalPrediction):
        if tuple(prediction.categories) != tuple(implied.categories):
            return ConformanceCheck(
                name,
                CheckStatus.FAILED,
                f"predict() uses categories {list(prediction.categories)} and the density "
                f"uses {list(implied.categories)}",
            )
        observed = np.asarray(prediction.probability, dtype=np.float64)
        expected = np.asarray(implied.probability, dtype=np.float64)
    else:
        return ConformanceCheck(
            name,
            CheckStatus.SKIPPED,
            f"predict() returned {type(prediction).__name__}, which carries no discrete "
            "probability to reconcile",
        )
    worst = float(np.max(np.abs(observed - expected)))
    if worst > 1e-4:
        return ConformanceCheck(
            name,
            CheckStatus.FAILED,
            f"integrating the predicted density gives choice probabilities that differ "
            f"from predict() by up to {worst:.3e}; the two halves of the prediction "
            "describe different models",
        )
    return ConformanceCheck(
        name,
        CheckStatus.PASSED,
        f"the integrated density reproduces predict() to {worst:.3e}, with grid mass in "
        f"[{float(np.min(density.total_mass)):.4f}, {float(np.max(density.total_mass)):.4f}]",
    )


def _check_simulator(
    model: Any,
    study: Study,
    capabilities: ModelCapabilities,
    parameters: Mapping[str, float] | None,
    fit: FitResult,
    seed: int,
    generative_contract: type[Any] = GenerativeBehaviourModel,
) -> ConformanceCheck:
    name = "simulates-the-columns-it-scores"
    if not capabilities.can_simulate or not isinstance(model, generative_contract):
        if capabilities.can_bind_design:
            return ConformanceCheck(
                name,
                CheckStatus.SKIPPED,
                "the model is generative only relative to a design; check bind(design) "
                "instead, which returns the generative model this study would recover",
            )
        return ConformanceCheck(name, CheckStatus.SKIPPED, "the model declares no simulator")
    if parameters is None and capabilities.is_sampled:
        return ConformanceCheck(
            name,
            CheckStatus.FAILED,
            "the projected posterior summary could not be read through the model's declared "
            "posterior_parameter_labels, so there is no truth to simulate from",
        )
    values = dict(fit.parameters) if parameters is None else dict(parameters)
    if set(values) != set(model.parameter_names):
        return ConformanceCheck(
            name,
            CheckStatus.FAILED,
            "the supplied simulation parameters do not match parameter_names",
        )
    try:
        first = model.simulate(study, values, seed=seed)
        second = model.simulate(study, values, seed=seed)
    except Exception as error:  # surfaced as a conformance failure
        return ConformanceCheck(
            name, CheckStatus.FAILED, f"simulate() raised {type(error).__name__}: {error}"
        )
    if not isinstance(first, Study) or len(first) != len(study):
        return ConformanceCheck(
            name, CheckStatus.FAILED, "simulate() must return a Study with one row per design row"
        )
    missing = [column for column in capabilities.scored_columns if column not in first.columns]
    if missing:
        return ConformanceCheck(
            name, CheckStatus.FAILED, f"the simulated study is missing scored columns {missing}"
        )
    for column in capabilities.scored_columns:
        if not _columns_equal(np.asarray(first[column]), np.asarray(second[column])):
            return ConformanceCheck(
                name,
                CheckStatus.FAILED,
                f"two simulations under seed {seed} disagree on {column!r}; the simulator "
                "reads a global random stream",
            )
    return ConformanceCheck(
        name,
        CheckStatus.PASSED,
        f"simulate() reproduces {list(capabilities.scored_columns)} exactly under seed {seed}",
    )


def _prediction_values(prediction: Any) -> NDArray[np.float64]:
    """Flatten any prediction type to the numbers a leakage check can compare."""

    if isinstance(prediction, DensityPrediction):
        return np.asarray(prediction.density, dtype=np.float64).reshape(
            prediction.n_observations, -1
        )
    if isinstance(prediction, (Prediction, CategoricalPrediction)):
        return np.atleast_2d(np.asarray(prediction.probability, dtype=np.float64).T).T
    raise TypeError(f"cannot read prediction values from {type(prediction).__name__}")


def _columns_equal(left: NDArray[Any], right: NDArray[Any]) -> bool:
    if left.shape != right.shape:
        return False
    if left.dtype.kind == "f" and right.dtype.kind == "f":
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


__all__ = [
    "EstimatorConformance",
    "EstimatorConformanceError",
    "assert_behaviour_estimator_conforms",
    "assert_posterior_behaviour_estimator_conforms",
    "check_behaviour_estimator",
    "check_posterior_behaviour_estimator",
    "perturb_future_rows",
]
