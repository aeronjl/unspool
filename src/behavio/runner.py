"""Common execution of model candidates under an audited protocol plan.

One execution path
------------------
This module used to re-implement the entire fit-predict-score loop that
:func:`behavio.evaluation.evaluate_splits` already performed, and re-derive the same
statistics under different type names. The two implementations had *opposite failure
semantics*: ``evaluate_splits`` raised on a bad fold while the runner caught and retained
it, so ``model_recovery`` and ``protocol_recovery`` behaved differently on identical
inputs for reasons no user could predict.

There is now one loop. :func:`_run_candidate_folds` adapts each
:class:`~behavio.compiler.CompiledFold` to the
:class:`~behavio.contracts.fold.ValidationFold` contract and calls ``evaluate_splits``
with :attr:`~behavio.evaluation.FoldFailurePolicy.RETAIN`; retaining rather than raising
is now a declared option on the shared primitive rather than a second implementation.
Everything below the fold loop -- pointwise projection, equal-unit scoring, calibration,
ranking -- consumes the shared :class:`~behavio.evaluation.FoldEvaluation`.

The paired comparison, its bootstrap interval, and its simultaneous-inference family come
from :mod:`behavio.comparison`, so a number computed here equals the number computed by
:func:`~behavio.comparison.compare_models` bit for bit.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields, is_dataclass
from enum import StrEnum
from functools import partial
from itertools import chain
from typing import Any

import numpy as np
from numpy.typing import NDArray

from behavio.comparison import (
    BootstrapInterval,
    ComparisonFamily,
    ComparisonMultiplicity,
    PairedComparison,
    bootstrap_interval,
    paired_comparisons,
)
from behavio.compiler import CompiledProtocol
from behavio.diagnostics import FitAuditStatus
from behavio.evaluation import (
    CandidateDeclarationError,
    FoldEvaluation,
    FoldFailure,
    FoldFailurePolicy,
    FoldStage,
    SplitEvaluation,
    evaluate_splits,
)
from behavio.models import (
    BehaviourEstimator,
    CategoricalPrediction,
    PredictionMode,
)
from behavio.protocol import (
    ProtocolState,
    ProtocolValidationError,
    ScoreMetric,
    WinnerPolicy,
)
from behavio.registry import BASE_SETTING, EstimatorRegistry, builtin_estimator_registry

EVALUATION_REPORT_SCHEMA = "behavio.evaluation-report/2"
"""Version 2 collapsed the runner's private per-fold, paired-comparison and bootstrap
types onto the shared ones in :mod:`behavio.comparison` and
:mod:`behavio.evaluation`. Against version 1 a report gains ``ranking.family`` and the
per-contrast ``two_sided_probability``, ``adjusted_probability`` and ``decisive`` members,
and a retained failure no longer repeats the candidate name that already keys it."""

NESTED_EVALUATION_REPORT_SCHEMA = "behavio.nested-evaluation-report/2"
"""Version 2 carries the same collapse through the nested selection record."""


class ProtocolRunError(RuntimeError):
    """Raised when an unaudited plan or mismatched candidate registry is executed."""


class RankingStatus(StrEnum):
    """Whether the declared winner rule supports one candidate."""

    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"
    NO_ELIGIBLE_CANDIDATE = "no-eligible-candidate"


class DeclarationCheck(StrEnum):
    """Outcome of comparing one frozen declaration against the object supplied."""

    VERIFIED = "verified"
    UNVERIFIABLE = "unverifiable"
    CONTRADICTED = "contradicted"


@dataclass(frozen=True, slots=True)
class DeclarationFinding:
    """One declared-versus-supplied comparison for a frozen candidate."""

    candidate: str
    subject: str
    status: DeclarationCheck
    declared: str
    observed: str
    detail: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", DeclarationCheck(self.status))
        if not self.candidate or not self.subject:
            raise ValueError("a declaration finding must name its candidate and subject")

    def to_dict(self) -> dict[str, Any]:
        """Return a portable finding without executable objects."""

        return _json_safe(asdict(self))


@dataclass(frozen=True, slots=True)
class CandidateVerification:
    """Whether the estimator supplied for one candidate is what the protocol froze."""

    candidate: str
    declared_implementation: str
    observed_implementation: str
    model_name: str
    model_signature: str
    findings: tuple[DeclarationFinding, ...]

    @property
    def contradictions(self) -> tuple[DeclarationFinding, ...]:
        """Findings where the supplied object refutes the frozen declaration."""

        return tuple(item for item in self.findings if item.status == DeclarationCheck.CONTRADICTED)

    @property
    def unverifiable(self) -> tuple[DeclarationFinding, ...]:
        """Declarations that could not be checked against the supplied object."""

        return tuple(item for item in self.findings if item.status == DeclarationCheck.UNVERIFIABLE)

    @property
    def verified(self) -> bool:
        """Whether every declaration for this candidate was checked and held."""

        return all(item.status == DeclarationCheck.VERIFIED for item in self.findings)

    def to_dict(self) -> dict[str, Any]:
        """Return portable verification evidence for one candidate."""

        return {
            "candidate": self.candidate,
            "declared_implementation": self.declared_implementation,
            "observed_implementation": self.observed_implementation,
            "model_name": self.model_name,
            "model_signature": self.model_signature,
            "verified": self.verified,
            "findings": [item.to_dict() for item in self.findings],
        }


@dataclass(frozen=True, slots=True)
class PointwisePrediction:
    """Portable scored-row prediction with no copied raw observation."""

    row: int
    probability: float | None
    linear_predictor: float | None
    log_probability: float
    aggregation_unit: str | int | float | bool | None
    category_probabilities: tuple[float, ...] | None = None
    categories: tuple[str | int | float | bool | None, ...] | None = None
    observed_category_index: int | None = None

    def __post_init__(self) -> None:
        if self.row < 0:
            raise ValueError("pointwise prediction row must be non-negative")
        if not math.isfinite(self.log_probability):
            raise ValueError("pointwise log probability must be finite")
        categorical = self.category_probabilities is not None
        if categorical:
            if self.probability is not None or self.linear_predictor is not None:
                raise ValueError("categorical rows must not contain binary prediction fields")
            probabilities = tuple(self.category_probabilities or ())
            categories = tuple(self.categories or ())
            if len(probabilities) < 2 or len(categories) != len(probabilities):
                raise ValueError("categorical rows require aligned categories and probabilities")
            if not all(math.isfinite(value) and 0 <= value <= 1 for value in probabilities):
                raise ValueError("category probabilities must be finite values in [0, 1]")
            if not math.isclose(sum(probabilities), 1.0, rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError("category probabilities must sum to one")
            if self.observed_category_index is None or not (
                0 <= self.observed_category_index < len(categories)
            ):
                raise ValueError("categorical rows require a valid observed category index")
            for category in categories:
                _identifier(category)
            object.__setattr__(self, "category_probabilities", probabilities)
            object.__setattr__(self, "categories", categories)
        else:
            if self.categories is not None or self.observed_category_index is not None:
                raise ValueError("binary rows must not contain categorical prediction fields")
            if self.probability is None or self.linear_predictor is None:
                raise ValueError("binary rows require probability and linear predictor")
            if not all(math.isfinite(value) for value in (self.probability, self.linear_predictor)):
                raise ValueError("binary prediction values must be finite")
            if not 0 <= self.probability <= 1:
                raise ValueError("pointwise probability must lie between zero and one")

    @property
    def is_categorical(self) -> bool:
        return self.category_probabilities is not None

    def to_dict(self) -> dict[str, Any]:
        """Return the binary legacy shape or an explicit categorical row."""

        common: dict[str, Any] = {
            "row": self.row,
            "log_probability": self.log_probability,
            "aggregation_unit": self.aggregation_unit,
        }
        if self.is_categorical:
            common.update(
                {
                    "category_probabilities": list(self.category_probabilities or ()),
                    "categories": list(self.categories or ()),
                    "observed_category_index": self.observed_category_index,
                }
            )
        else:
            common.update(
                {
                    "probability": self.probability,
                    "linear_predictor": self.linear_predictor,
                }
            )
        return common


@dataclass(frozen=True, slots=True)
class CalibrationSummary:
    """Aggregate probability calibration retained independently of model ranking."""

    available: bool
    n_observations: int
    mean_probability: float | None
    observed_rate: float | None
    brier_score: float | None
    expected_calibration_error: float | None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.n_observations < 0:
            raise ValueError("calibration observation count must be non-negative")
        numeric = (
            self.mean_probability,
            self.observed_rate,
            self.brier_score,
            self.expected_calibration_error,
        )
        if self.available:
            if self.n_observations < 1 or any(value is None for value in numeric):
                raise ValueError("available calibration requires observations and finite summaries")
            if self.reason is not None:
                raise ValueError("available calibration cannot declare an unavailability reason")
        else:
            if any(value is not None for value in numeric) or not self.reason:
                raise ValueError("unavailable calibration requires only an explicit reason")
        for value in numeric:
            if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError("calibration summaries must be finite values in [0, 1]")


def fold_to_dict(
    evaluation: FoldEvaluation,
    predictions: Sequence[PointwisePrediction],
    *,
    retain_predictions: bool = True,
) -> dict[str, Any]:
    """Serialize one shared :class:`~behavio.evaluation.FoldEvaluation` for an artifact.

    The runner used to own a second per-fold record, ``FoldRun``, whose only job over and
    above ``FoldEvaluation`` was to be JSON-portable. That is a serialization concern, not
    a second concept, so it is a function here rather than a type.
    """

    fit = evaluation.fit
    result: dict[str, Any] = {
        "fold": evaluation.identifier,
        "fit": {
            "model_name": fit.model_name,
            "model_signature": fit.model_signature,
            "n_observations": fit.n_observations,
            "parameters": {
                name: float(value)
                for name, value in zip(fit.parameter_names, fit.estimates, strict=True)
            },
            "standard_errors": {
                name: float(value)
                for name, value in zip(fit.parameter_names, fit.standard_errors, strict=True)
            },
        },
        "audit": evaluation.fit_audit.to_dict(),
        "n_predictions": len(predictions),
    }
    if retain_predictions:
        result["predictions"] = [_json_safe(item.to_dict()) for item in predictions]
    return result


@dataclass(frozen=True, slots=True)
class UnitScore:
    """Equal-unit score retained before any across-unit aggregation."""

    unit: str | int | float | bool | None
    n_observations: int
    log_loss: float
    brier_score: float | None

    def __post_init__(self) -> None:
        if self.n_observations < 1 or not math.isfinite(self.log_loss):
            raise ValueError("unit score requires observations and finite log loss")
        if self.brier_score is not None and (
            not math.isfinite(self.brier_score) or not 0 <= self.brier_score <= 1
        ):
            raise ValueError("unit Brier score must be null or a finite value in [0, 1]")


@dataclass(frozen=True, slots=True)
class ScoreSummary:
    """Declared proper-score summary at pooled and equal-unit levels."""

    metric: ScoreMetric
    pooled_score: float
    unit_balanced_score: float
    unit_balanced_interval: BootstrapInterval

    def __post_init__(self) -> None:
        object.__setattr__(self, "metric", ScoreMetric(self.metric))
        if not math.isfinite(self.pooled_score) or not math.isfinite(self.unit_balanced_score):
            raise ValueError("score summaries must be finite")
        if self.metric == ScoreMetric.BRIER and not (
            0 <= self.pooled_score <= 1 and 0 <= self.unit_balanced_score <= 1
        ):
            raise ValueError("Brier score summaries must lie in [0, 1]")
        if not math.isclose(
            self.unit_balanced_score,
            self.unit_balanced_interval.estimate,
            rel_tol=0,
            abs_tol=1e-15,
        ):
            raise ValueError("unit-balanced score and interval estimate differ")

    def to_dict(self) -> dict[str, Any]:
        """Return a portable summary with explicit metric semantics."""

        return {
            "metric": self.metric.value,
            "pooled_score": self.pooled_score,
            "unit_balanced_score": self.unit_balanced_score,
            "unit_balanced_interval": self.unit_balanced_interval.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class CandidateRun:
    """All successful folds, failures, scores, uncertainty, and calibration for a candidate."""

    name: str
    model_signature: str
    folds: tuple[FoldEvaluation, ...]
    failures: tuple[FoldFailure, ...]
    fold_predictions: tuple[tuple[PointwisePrediction, ...], ...]
    unit_scores: tuple[UnitScore, ...]
    score: ScoreSummary | None
    calibration: CalibrationSummary
    declaration_failure: FoldFailure | None = None
    """Why this candidate never entered the fold loop, when it did not.

    An object that does not satisfy the estimator contract, or that does not support the
    prediction mode the plan needs, fails before any fold is attempted. That is a finding
    about the candidate, so the run keeps going -- but it is *not* a per-fold finding, and
    manufacturing one ``FoldFailure`` per fold would archive an assertion that N folds were
    attempted when none were. ``fold`` on this record names the candidate, not a fold.
    """

    def __post_init__(self) -> None:
        if len(self.fold_predictions) != len(self.folds):
            raise ValueError("a candidate run must retain one pointwise group per scored fold")
        for fold, points in zip(self.folds, self.fold_predictions, strict=True):
            if len(points) != len(fold.pointwise_log_probability):
                raise ValueError("a scored fold must retain one pointwise record per scored row")
        if self.declaration_failure is not None and self.folds:
            raise ValueError("a candidate that failed its declaration cannot have scored folds")

    @property
    def predictions(self) -> tuple[PointwisePrediction, ...]:
        """Every scored-row record in fold order, flattened."""

        return tuple(chain.from_iterable(self.fold_predictions))

    @property
    def eligible(self) -> bool:
        """Whether every fold completed and no normalized audit failed."""

        return (
            bool(self.folds)
            and not self.failures
            and self.declaration_failure is None
            and self.score is not None
            and all(fold.fit_audit.status != FitAuditStatus.FAIL for fold in self.folds)
        )

    @property
    def audit_status(self) -> FitAuditStatus:
        """Worst numerical status, treating execution failure as failed audit evidence."""

        if self.failures or self.declaration_failure is not None or not self.folds:
            return FitAuditStatus.FAIL
        statuses = {fold.fit_audit.status for fold in self.folds}
        if FitAuditStatus.FAIL in statuses:
            return FitAuditStatus.FAIL
        if FitAuditStatus.WARNING in statuses:
            return FitAuditStatus.WARNING
        return FitAuditStatus.PASS

    @property
    def unit_balanced_log_loss(self) -> float | None:
        """Mean score with each declared aggregation unit weighted equally."""

        if self.score is None or self.score.metric not in (
            ScoreMetric.LOG_LOSS,
            ScoreMetric.JOINT_LOG_LOSS,
        ):
            return None
        return self.score.unit_balanced_score

    @property
    def pooled_log_loss(self) -> float | None:
        """Pooled log loss when log score was declared."""

        return self.score.pooled_score if self.unit_balanced_log_loss is not None else None

    @property
    def unit_balanced_log_loss_interval(self) -> BootstrapInterval | None:
        """Equal-unit log-loss interval when log score was declared."""

        return (
            self.score.unit_balanced_interval if self.unit_balanced_log_loss is not None else None
        )

    @property
    def unit_balanced_brier_score(self) -> float | None:
        """Equal-unit Brier score when Brier scoring was declared."""

        if self.score is None or self.score.metric != ScoreMetric.BRIER:
            return None
        return self.score.unit_balanced_score

    @property
    def pooled_brier_score(self) -> float | None:
        """Pooled Brier score when Brier scoring was declared."""

        return self.score.pooled_score if self.unit_balanced_brier_score is not None else None

    @property
    def unit_balanced_brier_interval(self) -> BootstrapInterval | None:
        """Equal-unit Brier interval when Brier scoring was declared."""

        return (
            self.score.unit_balanced_interval
            if self.unit_balanced_brier_score is not None
            else None
        )

    def to_dict(self, *, retain_predictions: bool = True) -> dict[str, Any]:
        """Return complete candidate evidence without executable serialization."""

        return {
            "name": self.name,
            "model_signature": self.model_signature,
            "eligible": self.eligible,
            "audit_status": self.audit_status.value,
            "folds": [
                fold_to_dict(fold, points, retain_predictions=retain_predictions)
                for fold, points in zip(self.folds, self.fold_predictions, strict=True)
            ],
            "failures": [failure.to_dict() for failure in self.failures],
            "declaration_failure": (
                None if self.declaration_failure is None else self.declaration_failure.to_dict()
            ),
            "unit_scores": [_json_safe(asdict(score)) for score in self.unit_scores],
            "score": self.score.to_dict() if self.score else None,
            "pooled_log_loss": self.pooled_log_loss,
            "unit_balanced_log_loss": self.unit_balanced_log_loss,
            "unit_balanced_log_loss_interval": (
                self.unit_balanced_log_loss_interval.to_dict()
                if self.unit_balanced_log_loss_interval
                else None
            ),
            "pooled_brier_score": self.pooled_brier_score,
            "unit_balanced_brier_score": self.unit_balanced_brier_score,
            "unit_balanced_brier_interval": (
                self.unit_balanced_brier_interval.to_dict()
                if self.unit_balanced_brier_interval
                else None
            ),
            "calibration": _json_safe(asdict(self.calibration)),
        }


@dataclass(frozen=True, slots=True)
class Ranking:
    """Declared winner policy applied without resolving overlapping evidence by fiat.

    ``family`` records the simultaneous family the winner rule was read against. It is
    retained whether or not a winner was named, so a reader can always see how many
    contrasts were examined to produce the verdict.
    """

    status: RankingStatus
    winner: str | None
    eligible_candidates: tuple[str, ...]
    reason: str
    family: ComparisonFamily

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", RankingStatus(self.status))
        if (self.status == RankingStatus.RESOLVED) != (self.winner is not None):
            raise ValueError("ranking winner must exist exactly when status is resolved")
        if not isinstance(self.family, ComparisonFamily):
            raise TypeError("ranking family must be a ComparisonFamily")

    def to_dict(self) -> dict[str, Any]:
        """Return the verdict beside the family it was read against."""

        return {
            "status": self.status.value,
            "winner": self.winner,
            "eligible_candidates": list(self.eligible_candidates),
            "reason": self.reason,
            "family": self.family.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Common runner output over the exact compiled plan."""

    schema_version: str
    protocol_fingerprint: str
    execution_plan_fingerprint: str
    candidates: tuple[CandidateRun, ...]
    paired_comparisons: tuple[PairedComparison, ...]
    ranking: Ranking

    @property
    def fingerprint(self) -> str:
        """Content address of safe summaries and retained pointwise predictions."""

        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def to_dict(self, *, retain_predictions: bool = True) -> dict[str, Any]:
        """Return portable evidence without pickles, functions, or covariance matrices."""

        return {
            "schema_version": self.schema_version,
            "protocol_fingerprint": self.protocol_fingerprint,
            "execution_plan_fingerprint": self.execution_plan_fingerprint,
            "candidates": {
                candidate.name: candidate.to_dict(retain_predictions=retain_predictions)
                for candidate in self.candidates
            },
            "paired_comparisons": [comparison.to_dict() for comparison in self.paired_comparisons],
            "ranking": self.ranking.to_dict(),
        }

    def canonical_json(self) -> str:
        """Serialize the complete retained evaluation deterministically."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )


@dataclass(frozen=True, slots=True)
class ProtocolRun:
    """Evaluated lifecycle state paired with its immutable report.

    ``verification`` records how each supplied estimator was checked against its frozen
    declaration. It is deliberately not part of :class:`EvaluationReport`: a contradiction
    refuses the run outright, so anything retained here is a declaration that held or one
    that could not be checked, and a fully verified run keeps a byte-identical report.
    """

    protocol: Any
    compiled: CompiledProtocol
    report: EvaluationReport
    verification: tuple[CandidateVerification, ...] = ()

    def __post_init__(self) -> None:
        if self.protocol.state != ProtocolState.EVALUATED:
            raise ProtocolValidationError("a protocol run requires evaluated lifecycle state")
        if self.protocol.fingerprint != self.report.protocol_fingerprint:
            raise ProtocolValidationError("run report and protocol fingerprints differ")


@dataclass(frozen=True, slots=True)
class NestedFold:
    """Training-only inner comparison and selected outer evaluation for one fold."""

    outer_fold: str
    selected_candidate: str | None
    inner_candidates: tuple[CandidateRun, ...]
    outer_result: CandidateRun | None
    selection_failure: str | None = None

    def __post_init__(self) -> None:
        if (self.selected_candidate is None) != (self.outer_result is None):
            raise ValueError("selected candidate and outer result must be present together")
        if self.selected_candidate is None and not self.selection_failure:
            raise ValueError("an unresolved nested fold requires a selection failure")

    def to_dict(self, *, retain_predictions: bool = True) -> dict[str, Any]:
        """Return complete inner evidence and the untouched outer evaluation."""

        return {
            "outer_fold": self.outer_fold,
            "selected_candidate": self.selected_candidate,
            "selection_failure": self.selection_failure,
            "inner_candidates": {
                candidate.name: candidate.to_dict(retain_predictions=retain_predictions)
                for candidate in self.inner_candidates
            },
            "outer_result": (
                self.outer_result.to_dict(retain_predictions=retain_predictions)
                if self.outer_result
                else None
            ),
        }


@dataclass(frozen=True, slots=True)
class NestedEvaluationReport:
    """Outer performance of a candidate-selection procedure fitted training-only."""

    schema_version: str
    protocol_fingerprint: str
    execution_plan_fingerprint: str
    candidate_names: tuple[str, ...]
    folds: tuple[NestedFold, ...]
    unit_scores: tuple[UnitScore, ...]
    score: ScoreSummary | None
    calibration: CalibrationSummary

    @property
    def fingerprint(self) -> str:
        """Content address of inner selection and untouched outer evidence."""

        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @property
    def selected_candidates(self) -> tuple[str | None, ...]:
        """Selected candidate for each outer fold in compiled order."""

        return tuple(fold.selected_candidate for fold in self.folds)

    @property
    def selection_counts(self) -> dict[str, int]:
        """Counts over successfully selected outer folds."""

        return {
            name: sum(selected == name for selected in self.selected_candidates)
            for name in self.candidate_names
        }

    @property
    def eligible(self) -> bool:
        """Whether selection and untouched evaluation completed in every outer fold."""

        return bool(self.folds) and all(
            fold.outer_result is not None and fold.outer_result.eligible for fold in self.folds
        )

    @property
    def unit_balanced_log_loss(self) -> float | None:
        """Equal-unit untouched outer performance of the selected procedure."""

        if self.score is None or self.score.metric not in (
            ScoreMetric.LOG_LOSS,
            ScoreMetric.JOINT_LOG_LOSS,
        ):
            return None
        return self.score.unit_balanced_score

    @property
    def pooled_log_loss(self) -> float | None:
        """Pooled outer log loss when a log score was declared."""

        return self.score.pooled_score if self.unit_balanced_log_loss is not None else None

    @property
    def unit_balanced_log_loss_interval(self) -> BootstrapInterval | None:
        """Equal-unit outer log-loss interval when a log score was declared."""

        return (
            self.score.unit_balanced_interval if self.unit_balanced_log_loss is not None else None
        )

    @property
    def unit_balanced_brier_score(self) -> float | None:
        """Equal-unit untouched outer Brier score when declared."""

        if self.score is None or self.score.metric != ScoreMetric.BRIER:
            return None
        return self.score.unit_balanced_score

    @property
    def pooled_brier_score(self) -> float | None:
        """Pooled untouched outer Brier score when declared."""

        return self.score.pooled_score if self.unit_balanced_brier_score is not None else None

    @property
    def unit_balanced_brier_interval(self) -> BootstrapInterval | None:
        """Equal-unit untouched outer Brier interval when declared."""

        return (
            self.score.unit_balanced_interval
            if self.unit_balanced_brier_score is not None
            else None
        )

    def to_dict(self, *, retain_predictions: bool = True) -> dict[str, Any]:
        """Return portable nested-selection evidence."""

        return {
            "schema_version": self.schema_version,
            "protocol_fingerprint": self.protocol_fingerprint,
            "execution_plan_fingerprint": self.execution_plan_fingerprint,
            "candidate_names": list(self.candidate_names),
            "eligible": self.eligible,
            "selected_candidates": list(self.selected_candidates),
            "selection_counts": self.selection_counts,
            "folds": [fold.to_dict(retain_predictions=retain_predictions) for fold in self.folds],
            "unit_scores": [_json_safe(asdict(score)) for score in self.unit_scores],
            "score": self.score.to_dict() if self.score else None,
            "pooled_log_loss": self.pooled_log_loss,
            "unit_balanced_log_loss": self.unit_balanced_log_loss,
            "unit_balanced_log_loss_interval": (
                self.unit_balanced_log_loss_interval.to_dict()
                if self.unit_balanced_log_loss_interval
                else None
            ),
            "pooled_brier_score": self.pooled_brier_score,
            "unit_balanced_brier_score": self.unit_balanced_brier_score,
            "unit_balanced_brier_interval": (
                self.unit_balanced_brier_interval.to_dict()
                if self.unit_balanced_brier_interval
                else None
            ),
            "calibration": _json_safe(asdict(self.calibration)),
        }

    def canonical_json(self) -> str:
        """Serialize all nested evidence deterministically."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )


@dataclass(frozen=True, slots=True)
class NestedProtocolRun:
    """Evaluated lifecycle state for a nested training-only selection procedure."""

    protocol: Any
    compiled: CompiledProtocol
    report: NestedEvaluationReport
    verification: tuple[CandidateVerification, ...] = ()

    def __post_init__(self) -> None:
        if self.protocol.state != ProtocolState.EVALUATED:
            raise ProtocolValidationError("a nested run requires evaluated lifecycle state")
        if self.protocol.fingerprint != self.report.protocol_fingerprint:
            raise ProtocolValidationError("nested report and protocol fingerprints differ")


def verify_candidate_declarations(
    protocol: Any,
    models: Mapping[str, BehaviourEstimator],
    *,
    registry: EstimatorRegistry | None = None,
) -> tuple[CandidateVerification, ...]:
    """Compare each supplied estimator against the candidate declaration frozen for it.

    A frozen protocol is only worth its fingerprint if the object that ran is the object
    it declared. This checks the two things a :class:`~behavio.protocol.CandidateSpec`
    fixes -- ``implementation`` and ``hyperparameters`` -- against the estimator handed to
    the runner.

    ``registry`` is the allowlist the declaration is resolved through, defaulting to
    :func:`~behavio.registry.builtin_estimator_registry`. It is what makes the check
    *decidable*: a registered implementation declares the class it produces, so the
    supplied object either is an instance of it or is not, and a combinator registration
    additionally exposes the model it wraps, so ``base.``-prefixed settings are checked
    against the wrapped model rather than reported as fields that do not exist. Pass a
    registry containing your own registrations to have an extension model verified too.

    A declaration the registry does not know still falls back to the registry-free check,
    which resolves identity without importing the declared string: a frozen protocol is
    data, and importing a module path out of it would turn a declaration into code
    execution. Only modules the process has already imported are consulted, so that
    fallback can prove a contradiction but never manufacture one out of an unread module,
    and it reports ``unverifiable`` where it genuinely cannot tell.

    Findings are returned rather than raised, so the complete comparison is available to a
    caller that wants to record it; :func:`run_protocol` refuses to execute a plan when any
    finding is :attr:`DeclarationCheck.CONTRADICTED`.
    """

    allowlist = builtin_estimator_registry() if registry is None else registry
    verifications: list[CandidateVerification] = []
    for candidate in protocol.candidates:
        model = models[candidate.name]
        findings = [_verify_implementation(candidate, model, allowlist)]
        findings.extend(_verify_hyperparameters(candidate, model, allowlist))
        verifications.append(
            CandidateVerification(
                candidate=candidate.name,
                declared_implementation=candidate.implementation,
                observed_implementation=_implementation_path(type(model)),
                model_name=str(getattr(model, "model_name", "")),
                model_signature=str(getattr(model, "signature", "")),
                findings=tuple(findings),
            )
        )
    return tuple(verifications)


def _verify_implementation(
    candidate: Any, model: Any, registry: EstimatorRegistry
) -> DeclarationFinding:
    declared = candidate.implementation
    model_type = type(model)
    observed = _implementation_path(model_type)
    finding = partial(
        DeclarationFinding,
        candidate=candidate.name,
        subject="implementation",
        declared=declared,
        observed=observed,
    )
    resolved = registry.verify(declared, model)
    if resolved is not None:
        return finding(
            status=DeclarationCheck.VERIFIED if resolved else DeclarationCheck.CONTRADICTED,
            detail=(
                "the registry resolves the declared implementation to this class"
                if resolved
                else "the registry resolves the declared implementation to a different class"
            ),
        )
    if declared in _implementation_aliases(model_type):
        return finding(
            status=DeclarationCheck.VERIFIED,
            detail="the supplied object is an instance of the declared implementation",
        )
    module_path, _, declared_leaf = declared.rpartition(".")
    imported = sys.modules.get(module_path)
    if imported is not None and hasattr(imported, declared_leaf):
        return finding(
            status=DeclarationCheck.CONTRADICTED,
            detail="the declared import path resolves to a different object",
        )
    if declared_leaf != model_type.__qualname__.rpartition(".")[2]:
        return finding(
            status=DeclarationCheck.CONTRADICTED,
            detail="the supplied object is not of the declared class",
        )
    return finding(
        status=DeclarationCheck.UNVERIFIABLE,
        detail=(
            "the class name agrees but the declared module is not imported, and a frozen "
            "declaration is never imported to resolve it"
        ),
    )


def _verify_hyperparameters(
    candidate: Any, model: Any, registry: EstimatorRegistry
) -> tuple[DeclarationFinding, ...]:
    base = registry.base_of(candidate.implementation, model)
    findings: list[DeclarationFinding] = []
    for setting in candidate.hyperparameters:
        finding = partial(
            DeclarationFinding,
            candidate=candidate.name,
            subject=f"hyperparameter:{setting.name}",
            declared=repr(setting.value),
        )
        if setting.name == BASE_SETTING:
            findings.append(_verify_base_reference(finding, setting, base, registry))
            continue
        target, name = (
            (base, setting.name[len(BASE_SETTING) + 1 :])
            if setting.name.startswith(f"{BASE_SETTING}.")
            else (model, setting.name)
        )
        if target is None:
            findings.append(
                finding(
                    status=DeclarationCheck.UNVERIFIABLE,
                    observed="",
                    detail=(
                        "the declaration configures a wrapped model, but the registry does "
                        "not know this implementation to be a combinator"
                    ),
                )
            )
            continue
        findings.append(_verify_field(finding, target, name, setting.value))
    return tuple(findings)


def _verify_base_reference(
    finding: Callable[..., DeclarationFinding],
    setting: Any,
    base: Any,
    registry: EstimatorRegistry,
) -> DeclarationFinding:
    """Decide the ``base`` setting, which names an implementation rather than a value."""

    if base is None:
        return finding(
            status=DeclarationCheck.UNVERIFIABLE,
            observed="",
            detail="the registry does not know this implementation to be a combinator",
        )
    observed = _implementation_path(type(base))
    resolved = registry.verify(setting.value, base) if isinstance(setting.value, str) else None
    if resolved is None:
        return finding(
            status=DeclarationCheck.UNVERIFIABLE,
            observed=observed,
            detail="the declared base implementation is not in the registry",
        )
    return finding(
        status=DeclarationCheck.VERIFIED if resolved else DeclarationCheck.CONTRADICTED,
        observed=observed,
        detail=(
            "the wrapped model is an instance of the declared base implementation"
            if resolved
            else "the wrapped model is not an instance of the declared base implementation"
        ),
    )


def _verify_field(
    finding: Callable[..., DeclarationFinding], model: Any, name: str, value: Any
) -> DeclarationFinding:
    """Decide one declared setting against one field of the object that holds it."""

    if not (is_dataclass(model) and not isinstance(model, type)):
        return finding(
            status=DeclarationCheck.UNVERIFIABLE,
            observed="",
            detail="the supplied object is not a dataclass with declared fields",
        )
    if not any(field.name == name for field in fields(model)):
        return finding(
            status=DeclarationCheck.UNVERIFIABLE,
            observed="",
            detail="the supplied object has no field of that name",
        )
    observed = getattr(model, name)
    comparable, normalized = _comparable_value(observed)
    if not comparable:
        return finding(
            status=DeclarationCheck.UNVERIFIABLE,
            observed=repr(observed),
            detail="the observed field value is not a comparable JSON scalar or tuple",
        )
    agrees = _values_agree(value, normalized)
    return finding(
        status=DeclarationCheck.VERIFIED if agrees else DeclarationCheck.CONTRADICTED,
        observed=repr(normalized),
        detail=(
            "the field value matches the frozen declaration"
            if agrees
            else "the field value differs from the frozen declaration"
        ),
    )


def _implementation_path(model_type: type[Any]) -> str:
    return f"{model_type.__module__}.{model_type.__qualname__}"


def _implementation_aliases(model_type: type[Any]) -> tuple[str, ...]:
    """Return import paths that already resolve to this exact class.

    A declaration names the public path (``behavio.models.BernoulliHistoryGLM``) while the
    class is defined in a private submodule, so the defining path alone is not enough. Only
    ancestor packages of the defining module are consulted, and only if already imported.
    """

    aliases = [_implementation_path(model_type)]
    if "." in model_type.__qualname__:
        return tuple(aliases)
    parts = model_type.__module__.split(".")
    for depth in range(len(parts) - 1, 0, -1):
        prefix = ".".join(parts[:depth])
        package = sys.modules.get(prefix)
        if package is not None and getattr(package, model_type.__qualname__, None) is model_type:
            aliases.append(f"{prefix}.{model_type.__qualname__}")
    return tuple(aliases)


def _comparable_value(value: Any) -> tuple[bool, Any]:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return True, value
    if isinstance(value, float):
        return math.isfinite(value), value
    if isinstance(value, (tuple, list)):
        items = [_comparable_value(item) for item in value]
        if all(comparable for comparable, _ in items):
            return True, tuple(item for _, item in items)
    return False, None


def _values_agree(declared: Any, observed: Any) -> bool:
    if isinstance(declared, tuple) or isinstance(observed, tuple):
        if not isinstance(declared, tuple) or not isinstance(observed, tuple):
            return False
        return len(declared) == len(observed) and all(
            _values_agree(left, right) for left, right in zip(declared, observed, strict=True)
        )
    if isinstance(declared, bool) or isinstance(observed, bool):
        return declared is observed
    if isinstance(declared, (int, float)) and isinstance(observed, (int, float)):
        return math.isclose(float(declared), float(observed), rel_tol=1e-9, abs_tol=1e-12)
    return type(declared) is type(observed) and declared == observed


def _refuse_contradictions(verifications: tuple[CandidateVerification, ...]) -> None:
    contradictions = tuple(
        finding for verification in verifications for finding in verification.contradictions
    )
    if not contradictions:
        return
    details = "; ".join(
        f"{finding.candidate}.{finding.subject}: {finding.detail} "
        f"(declared={finding.declared}, supplied={finding.observed})"
        for finding in contradictions
    )
    raise ProtocolRunError(
        "supplied models contradict the frozen candidate declaration, so the evaluation "
        f"would not be evidence about the declared protocol: {details}"
    )


def run_protocol(
    compiled: CompiledProtocol,
    models: Mapping[str, BehaviourEstimator],
) -> ProtocolRun:
    """Fit, predict, score, audit, compare, and rank every declared candidate.

    Candidate-fold failures are retained and execution continues.  Only an audited plan
    may run, and the exact candidate registry must match the frozen declaration -- by name,
    by declared implementation, and by declared hyperparameter, as
    :func:`verify_candidate_declarations` checks before any fit begins.
    """

    if not compiled.plan.audit.passed or compiled.protocol.state != ProtocolState.AUDITED:
        raise ProtocolRunError("only an audited execution plan may run")
    protocol = compiled.protocol
    if protocol.selection is not None:
        raise ProtocolRunError("nested protocols must use run_nested_protocol")
    declared_names = tuple(candidate.name for candidate in protocol.candidates)
    if set(models) != set(declared_names):
        raise ProtocolRunError(
            "model registry must exactly match declared candidates; "
            f"declared={declared_names!r}, supplied={tuple(models)!r}"
        )
    verification = verify_candidate_declarations(protocol, models)
    _refuse_contradictions(verification)
    aggregation_column = next(
        unit.column for unit in protocol.units if unit.name == protocol.comparison.aggregation_unit
    )
    outcome_column = protocol.candidates[0].scored_columns[0]
    study = compiled.materialized.study
    candidate_runs = tuple(
        _run_candidate(
            candidate_name,
            models[candidate_name],
            compiled,
            aggregation_column,
            outcome_column,
        )
        for candidate_name in declared_names
    )
    comparisons, family = _paired_comparisons(candidate_runs, protocol.comparison)
    ranking = _rank(
        candidate_runs,
        comparisons,
        family,
        protocol.comparison.winner_policy,
        protocol.comparison.metric,
    )
    report = EvaluationReport(
        schema_version=EVALUATION_REPORT_SCHEMA,
        protocol_fingerprint=protocol.fingerprint,
        execution_plan_fingerprint=compiled.plan.fingerprint,
        candidates=candidate_runs,
        paired_comparisons=comparisons,
        ranking=ranking,
    )
    evaluated = protocol.advance(
        ProtocolState.EVALUATED,
        artifact_fingerprint=report.fingerprint,
    )
    del study
    return ProtocolRun(evaluated, compiled, report, verification)


def run_nested_protocol(
    compiled: CompiledProtocol,
    models: Mapping[str, BehaviourEstimator],
) -> NestedProtocolRun:
    """Select candidates inside every outer training study and score untouched rows."""

    if not compiled.plan.audit.passed or compiled.protocol.state != ProtocolState.AUDITED:
        raise ProtocolRunError("only an audited execution plan may run")
    protocol = compiled.protocol
    selection = protocol.selection
    if selection is None:
        raise ProtocolRunError("run_nested_protocol requires a nested selection declaration")
    declared_names = tuple(candidate.name for candidate in protocol.candidates)
    if set(models) != set(declared_names):
        raise ProtocolRunError("model registry must exactly match declared candidates")
    verification = verify_candidate_declarations(protocol, models)
    _refuse_contradictions(verification)
    selection_aggregation_column = next(
        unit.column for unit in protocol.units if unit.name == selection.aggregation_unit
    )
    comparison_aggregation_column = next(
        unit.column for unit in protocol.units if unit.name == protocol.comparison.aggregation_unit
    )
    outcome_column = protocol.candidates[0].scored_columns[0]
    study = compiled.materialized.study
    fold_runs: list[NestedFold] = []
    for outer_index, outer in enumerate(compiled.plan.folds):
        inner_results = tuple(
            _run_candidate_folds(
                name,
                models[name],
                study,
                outer.inner_folds,
                selection_aggregation_column,
                outcome_column,
                bootstrap_repetitions=selection.bootstrap_repetitions,
                bootstrap_seed=selection.seed + outer_index + 1,
                interval_level=protocol.comparison.interval_level,
                metric=selection.metric,
            )
            for name in selection.candidate_names
        )
        eligible = tuple(candidate for candidate in inner_results if candidate.eligible)
        if not eligible:
            fold_runs.append(
                NestedFold(
                    outer.identifier,
                    None,
                    inner_results,
                    None,
                    "no candidate completed every inner fold with an eligible numerical audit",
                )
            )
            continue
        selected = min(
            eligible,
            key=lambda candidate: _candidate_score(candidate, selection.metric),
        ).name
        outer_result = _run_candidate_folds(
            selected,
            models[selected],
            study,
            (outer,),
            comparison_aggregation_column,
            outcome_column,
            bootstrap_repetitions=protocol.comparison.bootstrap_repetitions,
            bootstrap_seed=protocol.comparison.seed + outer_index,
            interval_level=protocol.comparison.interval_level,
            metric=protocol.comparison.metric,
        )
        fold_runs.append(NestedFold(outer.identifier, selected, inner_results, outer_result))
    predictions = tuple(
        point
        for fold in fold_runs
        if fold.outer_result is not None
        for point in fold.outer_result.predictions
    )
    unit_scores = _unit_scores(predictions, study, outcome_column)
    score = (
        _score_summary(
            unit_scores,
            predictions,
            study,
            outcome_column,
            metric=protocol.comparison.metric,
            repetitions=protocol.comparison.bootstrap_repetitions,
            seed=protocol.comparison.seed,
            confidence=protocol.comparison.interval_level,
        )
        if unit_scores
        else None
    )
    report = NestedEvaluationReport(
        NESTED_EVALUATION_REPORT_SCHEMA,
        protocol.fingerprint,
        compiled.plan.fingerprint,
        selection.candidate_names,
        tuple(fold_runs),
        unit_scores,
        score,
        _calibration(predictions, study, outcome_column),
    )
    evaluated = protocol.advance(
        ProtocolState.EVALUATED,
        artifact_fingerprint=report.fingerprint,
    )
    return NestedProtocolRun(evaluated, compiled, report, verification)


def _run_candidate(
    name: str,
    model: BehaviourEstimator,
    compiled: CompiledProtocol,
    aggregation_column: str,
    outcome_column: str,
) -> CandidateRun:
    comparison = compiled.protocol.comparison
    return _run_candidate_folds(
        name,
        model,
        compiled.materialized.study,
        compiled.plan.folds,
        aggregation_column,
        outcome_column,
        bootstrap_repetitions=comparison.bootstrap_repetitions,
        bootstrap_seed=comparison.seed,
        interval_level=comparison.interval_level,
        metric=comparison.metric,
    )


def _run_candidate_folds(
    name: str,
    model: BehaviourEstimator,
    study: Any,
    plan_folds: Sequence[Any],
    aggregation_column: str,
    outcome_column: str,
    *,
    bootstrap_repetitions: int,
    bootstrap_seed: int,
    interval_level: float,
    metric: ScoreMetric,
) -> CandidateRun:
    """Evaluate one candidate over compiled folds, retaining rather than raising failures.

    This is the whole of the runner's former fold loop. It adapts each compiled fold to the
    :class:`~behavio.contracts.fold.ValidationFold` contract, hands the work to
    :func:`~behavio.evaluation.evaluate_splits`, and projects the result into the portable
    records a protocol artifact needs.
    """

    splits = tuple(_compiled_fold_split(fold) for fold in plan_folds)
    declaration_failure: FoldFailure | None = None
    try:
        evaluated = evaluate_splits(
            model,
            study,
            splits,
            mode=PredictionMode.FILTERED,
            # A compiled plan's leakage gate is the pre-fit protocol audit, which checks the
            # exact row sets rather than trusting a splitter's self-description; the
            # fold-level `prospective` flag would be a weaker second opinion about the same
            # question.
            require_prospective=False,
            on_failure=FoldFailurePolicy.RETAIN,
        )
    except CandidateDeclarationError as error:
        # `evaluate_splits` raises this under either fold policy, because it describes
        # something no fold could survive. For an interactive caller that is their own
        # mistake and should stop the call. Under a frozen protocol it is a finding about
        # *this* candidate, and erasing the other candidates' evidence over it would make
        # the run less informative than the failure it is reporting. It is recorded once,
        # against the candidate, rather than fabricated once per fold that never ran.
        evaluated = SplitEvaluation((), (), FoldFailurePolicy.RETAIN)
        declaration_failure = FoldFailure(
            fold=name,
            stage=FoldStage.FIT,
            exception_type=type(error).__name__,
            message=str(error),
        )
    units = study[aggregation_column]
    fold_predictions = tuple(_pointwise_predictions(evaluation, units) for evaluation in evaluated)
    predictions = tuple(chain.from_iterable(fold_predictions))
    unit_scores = _unit_scores(predictions, study, outcome_column)
    score = (
        _score_summary(
            unit_scores,
            predictions,
            study,
            outcome_column,
            metric=metric,
            repetitions=bootstrap_repetitions,
            seed=bootstrap_seed,
            confidence=interval_level,
        )
        if unit_scores
        else None
    )
    return CandidateRun(
        name=name,
        model_signature=model.signature,
        folds=tuple(evaluated),
        failures=evaluated.failures,
        fold_predictions=fold_predictions,
        unit_scores=unit_scores,
        score=score,
        calibration=_calibration(predictions, study, outcome_column),
        declaration_failure=declaration_failure,
    )


@dataclass(frozen=True, slots=True)
class _CompiledFoldSplit:
    """A compiled protocol fold seen through the :class:`ValidationFold` contract.

    The compiler speaks of ``fit_rows``, ``prediction_context_rows`` and ``scored_rows``;
    the fold contract speaks of ``train_indices``, ``prediction_context_indices`` and
    ``test_indices``. They are the same three row sets under two vocabularies, and this
    adapter is the whole of the translation. ``identifier`` is carried through so a
    retained failure and a serialized fold both name the compiled fold rather than a
    position.
    """

    identifier: str
    train_indices: NDArray[np.intp]
    prediction_context_indices: NDArray[np.intp]
    test_indices: NDArray[np.intp]

    @property
    def scheme(self) -> str:
        return "compiled-protocol-fold"

    @property
    def prospective(self) -> bool:
        """Refuse the question rather than answer it wrongly.

        A compiled fold is a set of row positions; whether the split that produced them
        protected a forecast horizon is a property of the *splitter*, which the compiler
        does not carry through. ``_run_candidate_folds`` passes ``require_prospective=False``
        precisely because the plan's pre-fit leakage audit is the stronger gate, so nothing
        on the runner's path reads this. Returning ``True`` anyway would put a fabricated
        claim into any provenance record that later learned to serialize a fold.
        """

        raise NotImplementedError(
            "a compiled protocol fold does not carry its splitter's prospective flag; the "
            "execution plan's pre-fit leakage audit is the gate that applies to it"
        )


def _compiled_fold_split(fold: Any) -> _CompiledFoldSplit:
    return _CompiledFoldSplit(
        identifier=fold.identifier,
        train_indices=np.asarray(fold.fit_rows, dtype=np.intp),
        prediction_context_indices=np.asarray(fold.prediction_context_rows, dtype=np.intp),
        test_indices=np.asarray(fold.scored_rows, dtype=np.intp),
    )


def _pointwise_predictions(
    evaluation: FoldEvaluation,
    units: Any,
) -> tuple[PointwisePrediction, ...]:
    """Project one shared fold evaluation into portable scored-row records.

    ``units`` is the already-resolved aggregation column. ``Study.__getitem__`` builds a
    fresh read-only view per call, so looking it up per scored row would allocate one view
    per trial.
    """

    rows = tuple(int(row) for row in evaluation.split.test_indices)
    prediction = evaluation.prediction
    scores = evaluation.pointwise_log_probability
    if isinstance(prediction, CategoricalPrediction):
        codes = evaluation.outcome_codes
        if codes is None:  # pragma: no cover - established by FoldEvaluation
            raise ProtocolRunError("a categorical fold must retain observed outcome codes")
        categories = tuple(_identifier(value) for value in prediction.categories)
        return tuple(
            PointwisePrediction(
                row=row,
                probability=None,
                linear_predictor=None,
                log_probability=float(log_probability),
                aggregation_unit=_identifier(units[row]),
                category_probabilities=tuple(float(value) for value in probabilities),
                categories=categories,
                observed_category_index=int(code),
            )
            for row, probabilities, code, log_probability in zip(
                rows, prediction.probability, codes, scores, strict=True
            )
        )
    return tuple(
        PointwisePrediction(
            row=row,
            probability=float(probability),
            linear_predictor=float(linear_predictor),
            log_probability=float(log_probability),
            aggregation_unit=_identifier(units[row]),
        )
        for row, probability, linear_predictor, log_probability in zip(
            rows,
            prediction.probability,
            prediction.linear_predictor,
            scores,
            strict=True,
        )
    )


def _unit_scores(
    predictions: Sequence[PointwisePrediction], study: Any, outcome_column: str
) -> tuple[UnitScore, ...]:
    by_unit: dict[str, list[PointwisePrediction]] = {}
    unit_values: dict[str, str | int | float | bool | None] = {}
    for point in predictions:
        key = _identifier_key(point.aggregation_unit)
        by_unit.setdefault(key, []).append(point)
        unit_values[key] = point.aggregation_unit
    result: list[UnitScore] = []
    for key, points in by_unit.items():
        point_brier = [_point_brier(point, study, outcome_column) for point in points]
        brier = (
            float(np.mean(point_brier)) if all(value is not None for value in point_brier) else None
        )
        result.append(
            UnitScore(
                unit=unit_values[key],
                n_observations=len(points),
                log_loss=-float(np.mean([point.log_probability for point in points])),
                brier_score=brier,
            )
        )
    return tuple(result)


def _score_summary(
    unit_scores: Sequence[UnitScore],
    predictions: Sequence[PointwisePrediction],
    study: Any,
    outcome_column: str,
    *,
    metric: ScoreMetric,
    repetitions: int,
    seed: int,
    confidence: float,
) -> ScoreSummary:
    metric = ScoreMetric(metric)
    values = [_unit_score_value(score, metric) for score in unit_scores]
    if metric == ScoreMetric.BRIER:
        brier = [_point_brier(point, study, outcome_column) for point in predictions]
        if any(value is None for value in brier):
            raise ProtocolRunError(
                "Brier scoring requires a binary outcome or explicit categorical codes"
            )
        pooled = float(np.mean(brier))
    else:
        pooled = -float(np.mean([point.log_probability for point in predictions]))
    interval = bootstrap_interval(
        values,
        resamples=repetitions,
        seed=seed,
        confidence_level=confidence,
    )
    return ScoreSummary(metric, pooled, float(np.mean(values)), interval)


def _unit_score_value(score: UnitScore, metric: ScoreMetric) -> float:
    if metric in (ScoreMetric.LOG_LOSS, ScoreMetric.JOINT_LOG_LOSS):
        return score.log_loss
    if score.brier_score is None:
        raise ProtocolRunError(
            "Brier scoring requires a binary outcome or explicit categorical codes"
        )
    return score.brier_score


def _calibration(
    predictions: Sequence[PointwisePrediction], study: Any, outcome_column: str
) -> CalibrationSummary:
    if not predictions:
        return CalibrationSummary(False, 0, None, None, None, None, "no successful predictions")
    if any(point.is_categorical for point in predictions):
        return CalibrationSummary(
            False,
            len(predictions),
            None,
            None,
            None,
            None,
            "binary reliability calibration is not defined for categorical predictions",
        )
    outcomes = [_binary_outcome(study[outcome_column][point.row]) for point in predictions]
    if any(outcome is None for outcome in outcomes):
        return CalibrationSummary(
            False,
            len(predictions),
            None,
            None,
            None,
            None,
            "declared calibration outcome is not binary 0/1",
        )
    observed = np.asarray(outcomes, dtype=np.float64)
    probabilities = np.asarray([point.probability for point in predictions])
    bins = np.minimum((probabilities * 10).astype(int), 9)
    calibration_error = 0.0
    for bin_index in range(10):
        selected = bins == bin_index
        if np.any(selected):
            calibration_error += float(np.mean(selected)) * abs(
                float(np.mean(probabilities[selected]) - np.mean(observed[selected]))
            )
    return CalibrationSummary(
        available=True,
        n_observations=len(predictions),
        mean_probability=float(np.mean(probabilities)),
        observed_rate=float(np.mean(observed)),
        brier_score=float(np.mean((probabilities - observed) ** 2)),
        expected_calibration_error=calibration_error,
    )


def _point_brier(point: PointwisePrediction, study: Any, outcome_column: str) -> float | None:
    if point.is_categorical:
        probabilities = np.asarray(point.category_probabilities, dtype=np.float64)
        target = np.zeros_like(probabilities)
        target[point.observed_category_index] = 1.0  # type: ignore[index]
        return float(0.5 * np.sum((probabilities - target) ** 2))
    outcome = _binary_outcome(study[outcome_column][point.row])
    if outcome is None:
        return None
    return float((point.probability - outcome) ** 2)  # type: ignore[operator]


def _paired_comparisons(
    candidate_runs: tuple[CandidateRun, ...], comparison: Any
) -> tuple[tuple[PairedComparison, ...], ComparisonFamily]:
    """Compare every eligible pair through the one shared paired-bootstrap implementation.

    The adjustment is :attr:`behavio.protocol.ComparisonSpec.multiplicity`, frozen with the
    rest of the design before any data was seen, because which contrasts survive it decides
    which candidate wins. The family-level error rate is ``1 - interval_level``: the rate
    the protocol already declared, now controlled across the whole family of contrasts
    instead of one contrast at a time. No new threshold is introduced.
    """

    eligible = tuple(candidate for candidate in candidate_runs if candidate.eligible)
    return paired_comparisons(
        {
            candidate.name: {
                _identifier_key(score.unit): _unit_score_value(score, comparison.metric)
                for score in candidate.unit_scores
            }
            for candidate in eligible
        },
        metric=comparison.metric,
        resamples=comparison.bootstrap_repetitions,
        seed=comparison.seed,
        confidence_level=comparison.interval_level,
        multiplicity=ComparisonMultiplicity(comparison.multiplicity),
    )


def _rank(
    candidates: tuple[CandidateRun, ...],
    comparisons: tuple[PairedComparison, ...],
    family: ComparisonFamily,
    policy: WinnerPolicy,
    metric: ScoreMetric,
) -> Ranking:
    eligible = tuple(candidate for candidate in candidates if candidate.eligible)
    names = tuple(candidate.name for candidate in eligible)
    if not eligible:
        return Ranking(
            RankingStatus.NO_ELIGIBLE_CANDIDATE,
            None,
            (),
            "every candidate failed execution or numerical audit",
            family,
        )
    if policy == WinnerPolicy.NO_AUTOMATIC_WINNER:
        return Ranking(
            RankingStatus.UNRESOLVED,
            None,
            names,
            "the protocol forbids automatic winner selection",
            family,
        )
    best = min(
        eligible,
        key=lambda candidate: _candidate_score(candidate, metric),
    ).name
    if policy == WinnerPolicy.LOWEST_POINT_ESTIMATE or len(eligible) == 1:
        return Ranking(
            RankingStatus.RESOLVED,
            best,
            names,
            "winner follows the frozen lowest-point-estimate policy",
            family,
        )
    undecided = tuple(
        other
        for other in names
        if other != best and not _decisively_favours(comparisons, best, other)
    )
    if not undecided:
        return Ranking(
            RankingStatus.RESOLVED,
            best,
            names,
            (
                f"the best candidate improves on every eligible competitor across all "
                f"{family.n_comparisons} simultaneous contrasts "
                f"{_adjustment_phrase(family)}"
            ),
            family,
        )
    return Ranking(
        RankingStatus.UNRESOLVED,
        None,
        names,
        (
            f"{len(undecided)} of {len(names) - 1} contrasts against the leading candidate "
            f"do not exclude equal predictive performance across the family of "
            f"{family.n_comparisons} simultaneous contrasts "
            f"({family.n_decisive} decisive {_adjustment_phrase(family)}; "
            f"{family.n_separated} separated on the unadjusted interval alone)"
        ),
        family,
    )


def _adjustment_phrase(family: ComparisonFamily) -> str:
    """Name the declared adjustment, or say plainly that none was applied.

    ``ComparisonMultiplicity.NONE`` and a family of one both leave the reading
    uncorrected; a verdict must say so rather than report ``after none adjustment``.
    """

    if not family.corrected:
        return f"under the uncorrected per-contrast reading at {family.family_error_rate:.3g}"
    return f"after {family.multiplicity.value} adjustment at {family.family_error_rate:.3g}"


def _decisively_favours(comparisons: tuple[PairedComparison, ...], winner: str, other: str) -> bool:
    """Whether a family-adjusted contrast puts ``winner`` strictly ahead of ``other``."""

    for comparison in comparisons:
        pair = (comparison.left_model, comparison.right_model)
        if pair in ((winner, other), (other, winner)):
            return comparison.favours == winner
    return False


def _candidate_score(candidate: CandidateRun, metric: ScoreMetric) -> float:
    if candidate.score is None or candidate.score.metric != metric:
        raise ProtocolRunError(
            f"candidate {candidate.name!r} has no summary for declared metric {metric.value!r}"
        )
    return candidate.score.unit_balanced_score


def _binary_outcome(value: Any) -> float | None:
    value = value.item() if isinstance(value, np.generic) else value
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and value in (0, 1):
        return float(value)
    return None


def _identifier(value: Any) -> str | int | float | bool | None:
    value = value.item() if isinstance(value, np.generic) else value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError(f"aggregation unit is not a finite JSON scalar: {value!r}")


def _identifier_key(value: Any) -> str:
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _json_safe(value: Any) -> Any:
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value
