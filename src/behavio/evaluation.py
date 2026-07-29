"""Fold-aware evaluation for frequentist and sampled behavioural models.

Where the convergence gate lives
-------------------------------
A sampled candidate enters evaluation through
:class:`~behavio.contracts.posterior.PosteriorBehaviourEstimator`. Each fold is sampled,
audited with :func:`behavio.posterior_diagnostics.audit_posterior`, and only then
projected to a :class:`FitResult` with
``converged = audit.status is not PosteriorAuditStatus.FAIL``.

The gate is therefore formed **per fold** and enforced **per candidate** by machinery that
already exists: a non-converged projection earns an ``optimizer_nonconvergence`` issue from
``audit_fit``, which makes the fold's :class:`~behavio.diagnostics.FitAudit` ``FAIL``,
which makes the candidate ineligible in
:attr:`behavio.comparison.ProspectiveModelResult.audit_status`, in
:attr:`behavio.comparison.ProspectiveComparisonReport.winner` and in
``behavio.model_recovery``. That mirrors ``behavio.runner``: a failed audit removes a
candidate from selection rather than raising.

The failing fold's score is still computed and still reported. Silently dropping its rows
would break the invariant that every candidate is scored over identical aggregation units,
which is the only reason a matched comparison is interpretable at all; the honest
alternative is to keep the number and mark it unusable, which is what happens here.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

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
from behavio.models.base import (
    CategoricalBehaviourEstimator,
    CategoricalPrediction,
    FitResult,
    ModelPrediction,
    Prediction,
    PredictionMode,
)
from behavio.posterior import PosteriorResult
from behavio.posterior_diagnostics import (
    PosteriorAudit,
    PosteriorAuditPolicy,
    PosteriorAuditStatus,
    audit_posterior,
)
from behavio.study import Study
from behavio.validation import ValidationFold


@dataclass(frozen=True, slots=True)
class PosteriorFoldPolicy:
    """How a sampled model's posterior becomes one scored validation fold.

    ``centre`` selects the posterior central tendency the projected point summary reports.
    ``audit_policy`` is handed to :func:`behavio.posterior_diagnostics.audit_posterior`;
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


@dataclass(frozen=True, slots=True)
class FoldEvaluation:
    """Fit, prediction, and pointwise score for one validation fold.

    ``posterior`` is ``None`` for an optimizer fit and carries the sampling evidence when
    the fold was scored from a posterior, so the two are never confused for one another.
    """

    split: ValidationFold
    fit: FitResult
    prediction: ModelPrediction
    pointwise_log_probability: NDArray[np.float64]
    outcome_codes: NDArray[np.int64] | None = None
    posterior: PosteriorFoldEvidence | None = None

    def __post_init__(self) -> None:
        scores = protected_array(self.pointwise_log_probability, dtype=np.float64)
        if scores.ndim != 1 or scores.shape != (self.prediction.n_observations,):
            raise ValueError("pointwise scores must match the number of predictions")
        if not np.all(np.isfinite(scores)):
            raise ValueError("pointwise scores must be finite")
        codes = self.outcome_codes
        if isinstance(self.prediction, CategoricalPrediction):
            if codes is None:
                raise ValueError("categorical predictions require observed outcome codes")
            protected_codes = protected_array(codes, dtype=np.int64)
            if protected_codes.shape != scores.shape or np.any(
                (protected_codes < 0) | (protected_codes >= len(self.prediction.categories))
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
        object.__setattr__(self, "pointwise_log_probability", scores)

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


def evaluate_splits(
    model: AnyBehaviourEstimator,
    study: Study,
    splits: Iterable[ValidationFold],
    *,
    mode: PredictionMode = PredictionMode.FILTERED,
    require_prospective: bool = True,
    posterior_policy: PosteriorFoldPolicy | None = None,
) -> tuple[FoldEvaluation, ...]:
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
    """

    sampled = is_posterior_estimator(model)
    if posterior_policy is not None:
        if not isinstance(posterior_policy, PosteriorFoldPolicy):
            raise TypeError("posterior_policy must be a PosteriorFoldPolicy")
        if not sampled:
            raise ValueError(
                "posterior_policy applies only to a PosteriorBehaviourEstimator; "
                f"model {model.model_name!r} is fitted by optimization"
            )
    policy = PosteriorFoldPolicy() if posterior_policy is None else posterior_policy
    capabilities = any_model_capabilities(model)
    prediction_mode = PredictionMode(mode)
    if prediction_mode not in capabilities.prediction_modes:
        raise ValueError(
            f"model {model.model_name!r} does not support {prediction_mode.value!r} predictions"
        )
    missing = set(capabilities.scored_columns) - set(study.columns)
    if missing:
        raise ValueError(f"study is missing scored model columns: {sorted(missing)}")

    evaluations: list[FoldEvaluation] = []
    for split in splits:
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
        training = study.take(split.train_indices)
        evidence: PosteriorFoldEvidence | None = None
        if sampled:
            fit, evidence = _sampled_fold_fit(model, training, policy)
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
        prediction_rows = np.concatenate((split.prediction_context_indices, split.test_indices))
        prediction_study = study.take(prediction_rows)
        full_prediction = model.predict(prediction_study, context, mode=prediction_mode)
        if not isinstance(full_prediction, (Prediction, CategoricalPrediction)):
            raise TypeError("model.predict must return Prediction or CategoricalPrediction")
        if full_prediction.n_observations != len(prediction_study):
            raise ValueError("model.predict must return one prediction per row")
        full_codes: NDArray[np.int64] | None = None
        if isinstance(full_prediction, CategoricalPrediction):
            if not isinstance(model, CategoricalBehaviourEstimator):
                raise TypeError(
                    "categorical predictions require categories and outcome_codes() on the model"
                )
            if tuple(model.categories) != full_prediction.categories:
                raise ValueError("model and prediction category coordinates differ")
            full_codes = np.asarray(model.outcome_codes(prediction_study), dtype=np.int64)
            if full_codes.shape != (len(prediction_study),):
                raise ValueError("outcome_codes must return one code per prediction row")
        full_scores = np.asarray(
            model.pointwise_log_prob(prediction_study, context, mode=prediction_mode),
            dtype=np.float64,
        )
        if full_scores.shape != (len(prediction_study),):
            raise ValueError("pointwise_log_prob must return one score per prediction row")
        target = np.arange(
            len(split.prediction_context_indices),
            len(prediction_rows),
            dtype=np.intp,
        )
        prediction = full_prediction.take(target)
        scores = full_scores[target]
        evaluations.append(
            FoldEvaluation(
                split=split,
                fit=fit,
                prediction=prediction,
                pointwise_log_probability=scores,
                outcome_codes=None if full_codes is None else full_codes[target],
                posterior=evidence,
            )
        )
    return tuple(evaluations)


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
