"""Fold-aware evaluation for frequentist and sampled behavioural models.

Where the convergence gate lives
-------------------------------
A sampled candidate enters evaluation through
:class:`~behavio.contracts.posterior.PosteriorBehaviourEstimator`. Each fold is sampled,
audited with :func:`behavio.posterior.diagnostics.audit_posterior`, and only then
projected to a :class:`FitResult` with
``converged = audit.status is not PosteriorAuditStatus.FAIL``.

The gate is therefore formed **per fold** and enforced **per candidate** by machinery that
already exists: a non-converged projection earns an ``optimizer_nonconvergence`` issue from
``audit_fit``, which makes the fold's :class:`~behavio.diagnostics.FitAudit` ``FAIL``,
which makes the candidate ineligible in
:attr:`behavio.compare.models.ProspectiveModelResult.audit_status`, in
:attr:`behavio.compare.models.ProspectiveComparisonReport.winner` and in
``behavio.recovery.models``. That mirrors ``behavio.protocol.runner``: a failed audit removes a
candidate from selection rather than raising.

The failing fold's score is still computed and still reported. Silently dropping its rows
would break the invariant that every candidate is scored over identical aggregation units,
which is the only reason a matched comparison is interpretable at all; the honest
alternative is to keep the number and mark it unusable, which is what happens here.

Raising versus retaining
------------------------
A fold can also fail outright -- the optimizer throws, ``predict`` returns the wrong
length, a score is not finite. There used to be two answers to that in this package:
:func:`evaluate_splits` raised, and ``behavio.protocol.runner`` re-implemented the whole loop so it
could catch and retain the failure as evidence. Identical inputs therefore behaved
differently depending on which entry point a caller reached, for reasons no user could
predict.

There is now one loop and one explicit declaration, :class:`FoldFailurePolicy`.
``RAISE`` aborts on the first bad fold and is the default, because an interactive caller
who did not ask for partial results should not silently receive them. ``RETAIN`` records a
:class:`FoldFailure` and continues, which is what a frozen protocol needs: a candidate
that could not be fitted is a finding about that candidate, not a reason to abandon the
other candidates. Both return the same :class:`SplitEvaluation`, whose ``failures`` are
empty under ``RAISE`` by construction.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, overload

import numpy as np
from numpy.typing import NDArray

from behavio._internal.arrays import protected_array
from behavio.contracts.posterior import (
    AnyBehaviourEstimator,
    PosteriorBehaviourEstimator,
    PosteriorCentre,
    any_model_capabilities,
    is_posterior_estimator,
)
from behavio.diagnostics import FitAudit, audit_fit
from behavio.evaluate.splits import EvaluationFold
from behavio.models.base import (
    CategoricalBehaviourEstimator,
    CategoricalPrediction,
    DensityPrediction,
    FitResult,
    ModelPrediction,
    Prediction,
    PredictionMode,
)
from behavio.posterior.diagnostics import (
    PosteriorAudit,
    PosteriorAuditPolicy,
    PosteriorAuditStatus,
    audit_posterior,
)
from behavio.posterior.result import PosteriorResult
from behavio.trials import Study


@dataclass(frozen=True, slots=True)
class PosteriorFoldPolicy:
    """How a sampled model's posterior becomes one scored validation fold.

    ``centre`` selects the posterior central tendency the projected point summary reports.
    ``audit_policy`` is handed to :func:`behavio.posterior.diagnostics.audit_posterior`;
    its verdict, not the caller's optimism, decides whether the projected fit is marked
    converged.
    """

    centre: PosteriorCentre = PosteriorCentre.MEAN
    audit_policy: PosteriorAuditPolicy | None = None

    def __post_init__(self) -> None:
        if self.audit_policy is not None and not isinstance(
            self.audit_policy, PosteriorAuditPolicy
        ):
            raise TypeError("audit_policy must be a PosteriorAuditPolicy")
        object.__setattr__(self, "centre", PosteriorCentre(self.centre))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe record of the declared projection and gate."""

        policy = PosteriorAuditPolicy() if self.audit_policy is None else self.audit_policy
        return {"centre": self.centre.value, "audit_policy": policy.to_dict()}


@dataclass(frozen=True, slots=True)
class PosteriorFoldEvidence:
    """Why one fold's :class:`FitResult` is a projection rather than an optimizer fit."""

    posterior: PosteriorResult
    audit: PosteriorAudit
    centre: PosteriorCentre

    def __post_init__(self) -> None:
        if not isinstance(self.posterior, PosteriorResult):
            raise TypeError("posterior must be a PosteriorResult")
        if not isinstance(self.audit, PosteriorAudit):
            raise TypeError("audit must be a PosteriorAudit")
        object.__setattr__(self, "centre", PosteriorCentre(self.centre))

    @property
    def status(self) -> PosteriorAuditStatus:
        """Convergence verdict that gated this fold."""

        return self.audit.status

    @property
    def converged(self) -> bool:
        """Whether the posterior was usable evidence under the audit policy."""

        return self.audit.status is not PosteriorAuditStatus.FAIL

    def to_dict(self) -> dict[str, Any]:
        """Return the projection and its convergence audit without expanding draws."""

        return {
            "centre": self.centre.value,
            "converged": self.converged,
            "convergence_audit": self.audit.to_dict(),
        }


class CandidateDeclarationError(ValueError):
    """Raised when a candidate could not have completed any fold.

    These are the checks :func:`evaluate_splits` makes *before* the loop: the object does
    not satisfy an estimator contract, it does not support the requested prediction mode,
    the study lacks a column it scores, or a posterior policy was handed to a frequentist
    model. They are raised under either :class:`FoldFailurePolicy`, because retaining the
    same finding once per fold would archive an assertion that those folds were attempted.

    Naming the class is what lets a caller that must not abort -- ``behavio.protocol.runner``, which
    owes the other candidates their evidence -- catch exactly this and record it once
    against the candidate, without also swallowing a genuine per-fold failure.
    """


class FoldStage(StrEnum):
    """Stable stage at which one fold failed.

    The three stages are exactly the three things :func:`evaluate_splits` asks a model to
    do, so a retained failure always names which contract the candidate could not meet.
    """

    FIT = "fit"
    PREDICT = "predict"
    SCORE = "score"


class FoldFailurePolicy(StrEnum):
    """What :func:`evaluate_splits` does when one fold cannot be completed.

    ``RAISE`` is the default: an interactive caller asked for an evaluation, and a partial
    one that silently omits folds is not the thing they asked for. ``RETAIN`` records the
    failure and continues, which is what a frozen protocol needs so that one broken
    candidate does not erase the evidence about the others.
    """

    RAISE = "raise"
    RETAIN = "retain"


@dataclass(frozen=True, slots=True)
class FoldFailure:
    """One fold-stage failure retained instead of aborting the remaining folds."""

    fold: str
    stage: FoldStage
    exception_type: str
    message: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", FoldStage(self.stage))
        if not self.fold or not self.exception_type:
            raise ValueError("a retained fold failure must name its fold and exception type")

    def to_dict(self) -> dict[str, str]:
        """Return a portable record with no traceback and no live exception."""

        return {
            "fold": self.fold,
            "stage": self.stage.value,
            "exception_type": self.exception_type,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class FoldEvaluation:
    """Fit, prediction, and pointwise score for one validation fold.

    ``posterior`` is ``None`` for an optimizer fit and carries the sampling evidence when
    the fold was scored from a posterior, so the two are never confused for one another.

    ``identifier`` is the fold's stable name, taken from
    :attr:`~behavio.contracts.fold.EvaluationFold.identifier` on the split it scored, so
    a report always names the fold the splitter named. ``audit`` is the fold fit's
    normalized numerical audit, computed once here rather than recomputed by every layer
    that needs to know whether the fold is usable.

    ``outcome_codes`` names the observed category of each scored row and is required
    exactly when the prediction *has* categories -- a
    :class:`~behavio.contracts.CategoricalPrediction`, or a
    :class:`~behavio.contracts.DensityPrediction` that is defective across them. A
    defective density is a joint prediction about a discrete choice and a continuous
    latency, and without the codes the discrete half is unscoreable: it is what
    :func:`behavio.compare.compare_models` reads to score the choice margin.
    """

    split: EvaluationFold
    fit: FitResult
    prediction: ModelPrediction
    pointwise_log_probability: NDArray[np.float64]
    outcome_codes: NDArray[np.int64] | None = None
    posterior: PosteriorFoldEvidence | None = None
    identifier: str = ""
    audit: FitAudit | None = None

    def __post_init__(self) -> None:
        scores = protected_array(self.pointwise_log_probability, dtype=np.float64)
        if scores.ndim != 1 or scores.shape != (self.prediction.n_observations,):
            raise ValueError("pointwise scores must match the number of predictions")
        if not np.all(np.isfinite(scores)):
            raise ValueError("pointwise scores must be finite")
        codes = self.outcome_codes
        categories = _prediction_categories(self.prediction)
        if categories is not None:
            if codes is None:
                raise ValueError("categorical predictions require observed outcome codes")
            protected_codes = protected_array(codes, dtype=np.int64)
            if protected_codes.shape != scores.shape or np.any(
                (protected_codes < 0) | (protected_codes >= len(categories))
            ):
                raise ValueError("outcome codes must identify one predicted category per row")
            object.__setattr__(self, "outcome_codes", protected_codes)
        elif codes is not None:
            raise ValueError("binary predictions must not attach categorical outcome codes")
        evidence = self.posterior
        if evidence is not None:
            if not isinstance(evidence, PosteriorFoldEvidence):
                raise TypeError("posterior must be a PosteriorFoldEvidence")
            if self.fit.diagnostics.converged is not evidence.converged:
                raise ValueError("projected fit convergence must equal the posterior audit verdict")
        if self.audit is None:
            object.__setattr__(self, "audit", audit_fit(self.fit))
        elif not isinstance(self.audit, FitAudit):
            raise TypeError("audit must be a FitAudit")
        if not isinstance(self.identifier, str):
            raise TypeError("fold identifier must be a string")
        object.__setattr__(self, "pointwise_log_probability", scores)

    @property
    def fit_audit(self) -> FitAudit:
        """The fold fit's normalized audit, narrowed to non-optional for type checkers.

        ``__post_init__`` always fills ``audit``, so this never recomputes; the field stays
        optional only so a caller can construct a fold without auditing it first.
        """

        if self.audit is None:  # pragma: no cover - established in __post_init__
            raise ValueError("the fold audit was not established")
        return self.audit

    @property
    def from_posterior(self) -> bool:
        """Whether this fold was scored from posterior draws rather than an optimizer."""

        return self.posterior is not None

    @property
    def mean_log_probability(self) -> float:
        return float(np.mean(self.pointwise_log_probability))

    @property
    def mean_log_loss(self) -> float:
        return -self.mean_log_probability

    @property
    def total_log_probability(self) -> float:
        return float(np.sum(self.pointwise_log_probability))


@dataclass(frozen=True, slots=True)
class SplitEvaluation(Sequence[FoldEvaluation]):
    """Every fold one candidate completed, and every fold it did not.

    This is a :class:`~collections.abc.Sequence` of the successful
    :class:`FoldEvaluation` values, so ``for fold in result``, ``result[0]``,
    ``len(result)`` and ``tuple(result)`` all address the completed folds exactly as the
    old ``tuple`` return did. ``failures`` is the new part, and it is empty by
    construction whenever ``policy`` is :attr:`FoldFailurePolicy.RAISE`.
    """

    evaluations: tuple[FoldEvaluation, ...]
    failures: tuple[FoldFailure, ...] = ()
    policy: FoldFailurePolicy = FoldFailurePolicy.RAISE

    def __post_init__(self) -> None:
        evaluations = tuple(self.evaluations)
        failures = tuple(self.failures)
        policy = FoldFailurePolicy(self.policy)
        if policy is FoldFailurePolicy.RAISE and failures:
            raise ValueError("a raising evaluation cannot retain fold failures")
        object.__setattr__(self, "evaluations", evaluations)
        object.__setattr__(self, "failures", failures)
        object.__setattr__(self, "policy", policy)

    def __len__(self) -> int:
        return len(self.evaluations)

    @overload
    def __getitem__(self, index: int) -> FoldEvaluation: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[FoldEvaluation, ...]: ...

    def __getitem__(self, index: int | slice) -> FoldEvaluation | tuple[FoldEvaluation, ...]:
        return self.evaluations[index]

    def __iter__(self) -> Iterator[FoldEvaluation]:
        # The Sequence mixin would synthesize this from __getitem__ with an index counter;
        # delegating to the tuple is both faster and what every caller actually wants.
        return iter(self.evaluations)

    @property
    def complete(self) -> bool:
        """Whether every requested fold produced an evaluation."""

        return not self.failures and bool(self.evaluations)


def evaluate_splits(
    model: AnyBehaviourEstimator,
    study: Study,
    splits: Iterable[EvaluationFold],
    *,
    mode: PredictionMode = PredictionMode.FILTERED,
    require_prospective: bool = True,
    posterior_policy: PosteriorFoldPolicy | None = None,
    on_failure: FoldFailurePolicy = FoldFailurePolicy.RAISE,
) -> SplitEvaluation:
    """Fit or sample and score a model independently within each supplied fold.

    Prospective folds are required by default. Passing a non-prospective splitter therefore
    needs an explicit ``require_prospective=False`` acknowledgement. Prediction-context
    rows initialize filtered history but are removed from returned predictions and scores.

    A :class:`~behavio.contracts.posterior.PosteriorBehaviourEstimator` is sampled once per
    fold, audited, and projected; ``predict`` and ``pointwise_log_prob`` then receive the
    whole :class:`~behavio.posterior.PosteriorResult`, never the projection, so a sampled
    model can and should report the log pointwise predictive density averaged over draws
    (see :func:`~behavio.contracts.posterior.posterior_log_predictive_density`).
    ``posterior_policy`` applies only to sampled models and is rejected for a frequentist
    one rather than silently ignored.

    ``on_failure`` declares what happens when a fold cannot be completed. It governs only
    failures *of the candidate* -- an optimizer that throws, a prediction of the wrong
    length, a non-finite score. Declaration errors that no fold could survive, such as a
    prediction mode the model does not support or a study missing a scored column, are
    raised under either policy, because retaining the same finding once per fold would
    describe the caller's mistake as evidence about the model.
    """

    sampled = is_posterior_estimator(model)
    if posterior_policy is not None:
        if not isinstance(posterior_policy, PosteriorFoldPolicy):
            raise TypeError("posterior_policy must be a PosteriorFoldPolicy")
        if not sampled:
            raise CandidateDeclarationError(
                "posterior_policy applies only to a PosteriorBehaviourEstimator; "
                f"model {model.model_name!r} is fitted by optimization"
            )
    policy = PosteriorFoldPolicy() if posterior_policy is None else posterior_policy
    failure_policy = FoldFailurePolicy(on_failure)
    try:
        capabilities = any_model_capabilities(model)
    except (TypeError, ValueError) as error:
        raise CandidateDeclarationError(str(error)) from error
    prediction_mode = PredictionMode(mode)
    if prediction_mode not in capabilities.prediction_modes:
        raise CandidateDeclarationError(
            f"model {model.model_name!r} does not support {prediction_mode.value!r} predictions"
        )
    missing = set(capabilities.scored_columns) - set(study.columns)
    if missing:
        raise CandidateDeclarationError(f"study is missing scored model columns: {sorted(missing)}")

    evaluations: list[FoldEvaluation] = []
    failures: list[FoldFailure] = []
    seen: dict[str, int] = {}
    for position, split in enumerate(splits):
        identifier = _fold_identifier(split, position, seen)
        if require_prospective and not split.prospective:
            raise ValueError(
                f"split scheme {split.scheme!r} is not prospective; "
                "set require_prospective=False only for an intentional interpolation analysis"
            )
        _validate_positions(split.train_indices, len(study), "train_indices")
        _validate_positions(split.test_indices, len(study), "test_indices")
        _validate_positions(
            split.prediction_context_indices,
            len(study),
            "prediction_context_indices",
        )
        stage = FoldStage.FIT
        try:
            training = study.take(split.train_indices)
            fit, evidence, context = _fold_fit(model, training, policy, sampled=sampled)
            stage = FoldStage.PREDICT
            prediction_study, full_prediction, full_codes = _fold_prediction(
                model,
                study,
                split,
                context,
                prediction_mode,
            )
            stage = FoldStage.SCORE
            evaluations.append(
                _fold_evaluation(
                    model,
                    split,
                    identifier,
                    fit,
                    evidence,
                    context,
                    prediction_study,
                    full_prediction,
                    full_codes,
                    prediction_mode,
                )
            )
        except Exception as error:  # a candidate's failure is evidence about that candidate
            if failure_policy is FoldFailurePolicy.RAISE:
                raise
            failures.append(
                FoldFailure(
                    fold=identifier,
                    stage=stage,
                    exception_type=type(error).__name__,
                    message=str(error),
                )
            )
    return SplitEvaluation(tuple(evaluations), tuple(failures), failure_policy)


def _fold_identifier(split: EvaluationFold, position: int, seen: dict[str, int]) -> str:
    """Read one fold's declared name and refuse a split set that cannot be keyed by it.

    ``identifier`` is a declared member of :class:`~behavio.contracts.fold.EvaluationFold`.
    It used to be read with a ``getattr`` fallback that numbered unnamed folds by position,
    which meant the library depended on a name it had never asked any fold to supply, and
    silently produced ``fold-0003`` for anything that did not. A split that does not name
    itself now fails the contract, loudly, at the fold that broke it.

    Duplicate names are refused for the same reason: a retained failure names its fold, and
    an evidence bundle keys its prediction and audit maps on the name. Two folds sharing one
    would not be an ambiguity to resolve later -- one of them would simply disappear from
    the record.
    """

    try:
        identifier = split.identifier
    except AttributeError:
        raise TypeError(
            f"the split at position {position} declares no identifier and so does not "
            "satisfy behavio.contracts.fold.EvaluationFold"
        ) from None
    if not isinstance(identifier, str) or not identifier:
        raise ValueError(
            f"the split at position {position} must declare a non-empty string identifier; "
            f"got {identifier!r}"
        )
    if identifier in seen:
        raise ValueError(
            f"splits at positions {seen[identifier]} and {position} share the identifier "
            f"{identifier!r}; fold names key retained failures and evidence-bundle records, "
            "so they must be distinct within one split set"
        )
    seen[identifier] = position
    return identifier


def _fold_fit(
    model: AnyBehaviourEstimator,
    training: Study,
    policy: PosteriorFoldPolicy,
    *,
    sampled: bool,
) -> tuple[FitResult, PosteriorFoldEvidence | None, Any]:
    """Fit or sample one training fold and check the result against the estimator."""

    evidence: PosteriorFoldEvidence | None = None
    if sampled:
        fit, evidence = _sampled_fold_fit(model, training, policy)  # type: ignore[arg-type]
        context: Any = evidence.posterior
    else:
        fit = model.fit(training)
        context = fit
    if not isinstance(fit, FitResult):
        raise TypeError("model.fit must return a FitResult")
    if fit.model_name != model.model_name or fit.model_signature != model.signature:
        raise ValueError("fit result does not match the fitted estimator")
    if fit.n_observations != len(training):
        raise ValueError("fit result n_observations must equal the training-study length")
    return fit, evidence, context


def _fold_prediction(
    model: AnyBehaviourEstimator,
    study: Study,
    split: EvaluationFold,
    context: Any,
    prediction_mode: PredictionMode,
) -> tuple[Study, ModelPrediction, NDArray[np.int64] | None]:
    """Predict over context and test rows together and check shape and coordinates.

    All three prediction shapes are admitted, and a
    :class:`~behavio.contracts.DensityPrediction` is checked on exactly the terms a
    :class:`~behavio.contracts.CategoricalPrediction` is: if it names categories, the model
    must name the same ones and must be able to code each row's observed category. A
    density that names none is a prediction about a continuous outcome alone and carries no
    codes.
    """

    prediction_rows = np.concatenate((split.prediction_context_indices, split.test_indices))
    prediction_study = study.take(prediction_rows)
    full_prediction = model.predict(prediction_study, context, mode=prediction_mode)
    if not isinstance(full_prediction, (Prediction, CategoricalPrediction, DensityPrediction)):
        raise TypeError(
            "model.predict must return Prediction, CategoricalPrediction or DensityPrediction"
        )
    if full_prediction.n_observations != len(prediction_study):
        raise ValueError("model.predict must return one prediction per row")
    full_codes: NDArray[np.int64] | None = None
    categories = _prediction_categories(full_prediction)
    if categories is not None:
        if not isinstance(model, CategoricalBehaviourEstimator):
            raise TypeError(
                "categorical predictions require categories and outcome_codes() on the model"
            )
        if tuple(model.categories) != categories:
            raise ValueError("model and prediction category coordinates differ")
        full_codes = np.asarray(model.outcome_codes(prediction_study), dtype=np.int64)
        if full_codes.shape != (len(prediction_study),):
            raise ValueError("outcome_codes must return one code per prediction row")
    return prediction_study, full_prediction, full_codes


def _prediction_categories(prediction: ModelPrediction) -> tuple[Any, ...] | None:
    """Return the category coordinate a prediction declares, or ``None`` for one that does not.

    A :class:`~behavio.contracts.DensityPrediction` may or may not be defective across
    categories, so "does this prediction have a discrete coordinate?" is a question about
    the value rather than about its type.
    """

    if isinstance(prediction, CategoricalPrediction):
        return prediction.categories
    if isinstance(prediction, DensityPrediction):
        return prediction.categories
    return None


def _fold_evaluation(
    model: AnyBehaviourEstimator,
    split: EvaluationFold,
    identifier: str,
    fit: FitResult,
    evidence: PosteriorFoldEvidence | None,
    context: Any,
    prediction_study: Study,
    full_prediction: ModelPrediction,
    full_codes: NDArray[np.int64] | None,
    prediction_mode: PredictionMode,
) -> FoldEvaluation:
    """Score the prediction rows and keep only the scored ones."""

    full_scores = np.asarray(
        model.pointwise_log_prob(prediction_study, context, mode=prediction_mode),
        dtype=np.float64,
    )
    if full_scores.shape != (len(prediction_study),):
        raise ValueError("pointwise_log_prob must return one score per prediction row")
    target = np.arange(
        len(split.prediction_context_indices),
        len(prediction_study),
        dtype=np.intp,
    )
    return FoldEvaluation(
        split=split,
        fit=fit,
        prediction=full_prediction.take(target),
        pointwise_log_probability=full_scores[target],
        outcome_codes=None if full_codes is None else full_codes[target],
        posterior=evidence,
        identifier=identifier,
    )


def _sampled_fold_fit(
    model: PosteriorBehaviourEstimator,
    training: Study,
    policy: PosteriorFoldPolicy,
) -> tuple[FitResult, PosteriorFoldEvidence]:
    """Sample, audit, and project one training fold of a sampled estimator."""

    posterior = model.sample(training)
    if not isinstance(posterior, PosteriorResult):
        raise TypeError("model.sample must return a PosteriorResult")
    audit = audit_posterior(posterior, policy=policy.audit_policy)
    evidence = PosteriorFoldEvidence(posterior=posterior, audit=audit, centre=policy.centre)
    fit = model.point_summary(posterior, converged=evidence.converged, centre=policy.centre)
    if not isinstance(fit, FitResult):
        raise TypeError("model.point_summary must return a FitResult")
    if fit.diagnostics.converged is not evidence.converged:
        raise ValueError(
            "model.point_summary must record the convergence verdict it was given; the "
            "posterior convergence audit, not the model, decides whether a fold is usable"
        )
    return fit, evidence


def _validate_positions(indices: NDArray[np.intp], length: int, name: str) -> None:
    if np.any(indices >= length):
        raise IndexError(f"{name} contains a row position outside the study")
