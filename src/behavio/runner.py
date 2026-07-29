"""Common execution of model candidates under an audited protocol plan."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from behavio.comparison import BootstrapInterval
from behavio.compiler import CompiledProtocol
from behavio.diagnostics import FitAudit, FitAuditStatus, audit_fit
from behavio.models import (
    BehaviourEstimator,
    CategoricalBehaviourEstimator,
    CategoricalPrediction,
    FitResult,
    Prediction,
    PredictionMode,
)
from behavio.protocol import (
    ProtocolState,
    ProtocolValidationError,
    ScoreMetric,
    WinnerPolicy,
)


class ProtocolRunError(RuntimeError):
    """Raised when an unaudited plan or mismatched candidate registry is executed."""


class RunStage(StrEnum):
    """Stable stage at which one retained candidate failure occurred."""

    FIT = "fit"
    PREDICT = "predict"
    SCORE = "score"


class RankingStatus(StrEnum):
    """Whether the declared winner rule supports one candidate."""

    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"
    NO_ELIGIBLE_CANDIDATE = "no-eligible-candidate"


@dataclass(frozen=True, slots=True)
class RunFailure:
    """One fold-stage failure retained without aborting other candidates."""

    candidate: str
    fold: str
    stage: RunStage
    exception_type: str
    message: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", RunStage(self.stage))


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


@dataclass(frozen=True, slots=True)
class FoldRun:
    """Successful fit, numerical audit, and pointwise predictions for one fold."""

    fold: str
    fit: FitResult
    audit: FitAudit
    predictions: tuple[PointwisePrediction, ...]

    def __post_init__(self) -> None:
        if not self.predictions:
            raise ValueError("a successful fold run must retain pointwise predictions")

    def to_dict(self, *, retain_predictions: bool = True) -> dict[str, Any]:
        """Return a safe fit summary, optionally with pointwise predictive evidence."""

        result: dict[str, Any] = {
            "fold": self.fold,
            "fit": {
                "model_name": self.fit.model_name,
                "model_signature": self.fit.model_signature,
                "n_observations": self.fit.n_observations,
                "parameters": {
                    name: float(value)
                    for name, value in zip(
                        self.fit.parameter_names,
                        self.fit.estimates,
                        strict=True,
                    )
                },
                "standard_errors": {
                    name: float(value)
                    for name, value in zip(
                        self.fit.parameter_names,
                        self.fit.standard_errors,
                        strict=True,
                    )
                },
            },
            "audit": self.audit.to_dict(),
            "n_predictions": len(self.predictions),
        }
        if retain_predictions:
            result["predictions"] = [_json_safe(item.to_dict()) for item in self.predictions]
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
class ProtocolPairedComparison:
    """Paired equal-unit difference under the protocol's declared score."""

    left_model: str
    right_model: str
    metric: ScoreMetric
    left_minus_right: BootstrapInterval
    bootstrap_probability_positive: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "metric", ScoreMetric(self.metric))
        if not self.left_model or not self.right_model or self.left_model == self.right_model:
            raise ValueError("paired comparison requires two distinct named models")
        if not 0 <= self.bootstrap_probability_positive <= 1:
            raise ValueError("bootstrap probability must lie in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        """Return an explicit lower-is-better comparison direction."""

        return {
            "left_model": self.left_model,
            "right_model": self.right_model,
            "metric": self.metric.value,
            "direction": (
                f"left {self.metric.value} minus right {self.metric.value}; positive favors right"
            ),
            "left_minus_right": self.left_minus_right.to_dict(),
            "bootstrap_probability_positive": self.bootstrap_probability_positive,
        }


@dataclass(frozen=True, slots=True)
class CandidateRun:
    """All successful folds, failures, scores, uncertainty, and calibration for a candidate."""

    name: str
    model_signature: str
    folds: tuple[FoldRun, ...]
    failures: tuple[RunFailure, ...]
    unit_scores: tuple[UnitScore, ...]
    score: ScoreSummary | None
    calibration: CalibrationSummary

    @property
    def eligible(self) -> bool:
        """Whether every fold completed and no normalized audit failed."""

        return (
            bool(self.folds)
            and not self.failures
            and self.score is not None
            and all(fold.audit.status != FitAuditStatus.FAIL for fold in self.folds)
        )

    @property
    def audit_status(self) -> FitAuditStatus:
        """Worst numerical status, treating execution failure as failed audit evidence."""

        if self.failures or not self.folds:
            return FitAuditStatus.FAIL
        statuses = {fold.audit.status for fold in self.folds}
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
            "folds": [fold.to_dict(retain_predictions=retain_predictions) for fold in self.folds],
            "failures": [_json_safe(asdict(failure)) for failure in self.failures],
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
    """Declared winner policy applied without resolving overlapping evidence by fiat."""

    status: RankingStatus
    winner: str | None
    eligible_candidates: tuple[str, ...]
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", RankingStatus(self.status))
        if (self.status == RankingStatus.RESOLVED) != (self.winner is not None):
            raise ValueError("ranking winner must exist exactly when status is resolved")


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Common runner output over the exact compiled plan."""

    schema_version: str
    protocol_fingerprint: str
    execution_plan_fingerprint: str
    candidates: tuple[CandidateRun, ...]
    paired_comparisons: tuple[ProtocolPairedComparison, ...]
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
            "ranking": _json_safe(asdict(self.ranking)),
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
    """Evaluated lifecycle state paired with its immutable report."""

    protocol: Any
    compiled: CompiledProtocol
    report: EvaluationReport

    def __post_init__(self) -> None:
        if self.protocol.state != ProtocolState.EVALUATED:
            raise ProtocolValidationError("a protocol run requires evaluated lifecycle state")
        if self.protocol.fingerprint != self.report.protocol_fingerprint:
            raise ProtocolValidationError("run report and protocol fingerprints differ")


@dataclass(frozen=True, slots=True)
class NestedFoldRun:
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
    folds: tuple[NestedFoldRun, ...]
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

    def __post_init__(self) -> None:
        if self.protocol.state != ProtocolState.EVALUATED:
            raise ProtocolValidationError("a nested run requires evaluated lifecycle state")
        if self.protocol.fingerprint != self.report.protocol_fingerprint:
            raise ProtocolValidationError("nested report and protocol fingerprints differ")


def run_protocol(
    compiled: CompiledProtocol,
    models: Mapping[str, BehaviourEstimator],
) -> ProtocolRun:
    """Fit, predict, score, audit, compare, and rank every declared candidate.

    Candidate-fold failures are retained and execution continues.  Only an audited plan
    may run, and the exact candidate registry must match the frozen declaration.
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
    comparisons = _paired_comparisons(candidate_runs, protocol.comparison)
    ranking = _rank(
        candidate_runs,
        comparisons,
        protocol.comparison.winner_policy,
        protocol.comparison.metric,
    )
    report = EvaluationReport(
        schema_version="behavio.evaluation-report/1",
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
    return ProtocolRun(evaluated, compiled, report)


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
    selection_aggregation_column = next(
        unit.column for unit in protocol.units if unit.name == selection.aggregation_unit
    )
    comparison_aggregation_column = next(
        unit.column for unit in protocol.units if unit.name == protocol.comparison.aggregation_unit
    )
    outcome_column = protocol.candidates[0].scored_columns[0]
    study = compiled.materialized.study
    fold_runs: list[NestedFoldRun] = []
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
                NestedFoldRun(
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
        fold_runs.append(NestedFoldRun(outer.identifier, selected, inner_results, outer_result))
    predictions = tuple(
        point
        for fold in fold_runs
        if fold.outer_result is not None
        for outer_fold in fold.outer_result.folds
        for point in outer_fold.predictions
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
        "behavio.nested-evaluation-report/1",
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
    return NestedProtocolRun(evaluated, compiled, report)


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
    folds: list[FoldRun] = []
    failures: list[RunFailure] = []
    for fold in plan_folds:
        training = study.take(fold.fit_rows)
        try:
            fit = model.fit(training)
            _validate_fit(model, fit, len(training))
        except Exception as error:  # candidate failures are scientific evidence
            failures.append(_failure(name, fold.identifier, RunStage.FIT, error))
            continue
        prediction_rows = (*fold.prediction_context_rows, *fold.scored_rows)
        prediction_study = study.take(prediction_rows)
        try:
            prediction = model.predict(
                prediction_study,
                fit,
                mode=PredictionMode.FILTERED,
            )
            if not isinstance(prediction, (Prediction, CategoricalPrediction)):
                raise TypeError("model.predict must return Prediction or CategoricalPrediction")
            if prediction.n_observations != len(prediction_study):
                raise ValueError("prediction length differs from compiled prediction rows")
        except Exception as error:
            failures.append(_failure(name, fold.identifier, RunStage.PREDICT, error))
            continue
        try:
            scores = np.asarray(
                model.pointwise_log_prob(
                    prediction_study,
                    fit,
                    mode=PredictionMode.FILTERED,
                ),
                dtype=np.float64,
            )
            if scores.shape != (len(prediction_study),) or not np.all(np.isfinite(scores)):
                raise ValueError("pointwise scores must be one finite value per prediction row")
            offset = len(fold.prediction_context_rows)
            target = slice(offset, None)
            if isinstance(prediction, CategoricalPrediction):
                if not isinstance(model, CategoricalBehaviourEstimator):
                    raise TypeError(
                        "categorical predictions require categories and outcome_codes()"
                    )
                if tuple(model.categories) != prediction.categories:
                    raise ValueError("model and prediction category coordinates differ")
                codes = np.asarray(model.outcome_codes(prediction_study), dtype=np.int64)
                if codes.shape != (len(prediction_study),):
                    raise ValueError("outcome_codes must return one code per prediction row")
                pointwise = tuple(
                    PointwisePrediction(
                        row=row,
                        probability=None,
                        linear_predictor=None,
                        log_probability=float(log_probability),
                        aggregation_unit=_identifier(study[aggregation_column][row]),
                        category_probabilities=tuple(float(value) for value in probabilities),
                        categories=tuple(_identifier(value) for value in prediction.categories),
                        observed_category_index=int(code),
                    )
                    for row, probabilities, code, log_probability in zip(
                        fold.scored_rows,
                        prediction.probability[target],
                        codes[target],
                        scores[target],
                        strict=True,
                    )
                )
            else:
                pointwise = tuple(
                    PointwisePrediction(
                        row=row,
                        probability=float(probability),
                        linear_predictor=float(linear_predictor),
                        log_probability=float(log_probability),
                        aggregation_unit=_identifier(study[aggregation_column][row]),
                    )
                    for row, probability, linear_predictor, log_probability in zip(
                        fold.scored_rows,
                        prediction.probability[target],
                        prediction.linear_predictor[target],
                        scores[target],
                        strict=True,
                    )
                )
        except Exception as error:
            failures.append(_failure(name, fold.identifier, RunStage.SCORE, error))
            continue
        folds.append(FoldRun(fold.identifier, fit, audit_fit(fit), pointwise))

    all_predictions = tuple(point for fold in folds for point in fold.predictions)
    unit_scores = _unit_scores(all_predictions, study, outcome_column)
    score = (
        _score_summary(
            unit_scores,
            all_predictions,
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
    calibration = _calibration(all_predictions, study, outcome_column)
    return CandidateRun(
        name=name,
        model_signature=model.signature,
        folds=tuple(folds),
        failures=tuple(failures),
        unit_scores=unit_scores,
        score=score,
        calibration=calibration,
    )


def _validate_fit(model: BehaviourEstimator, fit: Any, n_observations: int) -> None:
    if not isinstance(fit, FitResult):
        raise TypeError("model.fit must return FitResult")
    if fit.model_name != model.model_name or fit.model_signature != model.signature:
        raise ValueError("fit identity differs from the declared model")
    if fit.n_observations != n_observations:
        raise ValueError("fit n_observations differs from the compiled fitting rows")


def _failure(candidate: str, fold: str, stage: RunStage, error: Exception) -> RunFailure:
    return RunFailure(candidate, fold, stage, type(error).__name__, str(error))


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
    interval = _bootstrap_interval(
        values,
        repetitions=repetitions,
        seed=seed,
        confidence=confidence,
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
) -> tuple[ProtocolPairedComparison, ...]:
    eligible = tuple(candidate for candidate in candidate_runs if candidate.eligible)
    results: list[ProtocolPairedComparison] = []
    for left_index, left in enumerate(eligible):
        left_scores = {
            _identifier_key(score.unit): _unit_score_value(score, comparison.metric)
            for score in left.unit_scores
        }
        for right in eligible[left_index + 1 :]:
            right_scores = {
                _identifier_key(score.unit): _unit_score_value(score, comparison.metric)
                for score in right.unit_scores
            }
            common = tuple(key for key in left_scores if key in right_scores)
            differences = [left_scores[key] - right_scores[key] for key in common]
            if not differences:
                continue
            interval, probability = _paired_interval(
                differences,
                repetitions=comparison.bootstrap_repetitions,
                seed=comparison.seed,
                confidence=comparison.interval_level,
            )
            results.append(
                ProtocolPairedComparison(
                    left.name,
                    right.name,
                    comparison.metric,
                    interval,
                    probability,
                )
            )
    return tuple(results)


def _bootstrap_interval(
    values: Sequence[float], *, repetitions: int, seed: int, confidence: float
) -> BootstrapInterval:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = np.mean(rng.choice(array, size=(repetitions, len(array)), replace=True), axis=1)
    tail = round((1 - confidence) / 2, 15)
    return BootstrapInterval(
        estimate=float(np.mean(array)),
        lower=float(np.quantile(draws, tail)),
        upper=float(np.quantile(draws, 1 - tail)),
        confidence_level=confidence,
    )


def _paired_interval(
    differences: Sequence[float], *, repetitions: int, seed: int, confidence: float
) -> tuple[BootstrapInterval, float]:
    array = np.asarray(differences, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = np.mean(rng.choice(array, size=(repetitions, len(array)), replace=True), axis=1)
    tail = round((1 - confidence) / 2, 15)
    return (
        BootstrapInterval(
            estimate=float(np.mean(array)),
            lower=float(np.quantile(draws, tail)),
            upper=float(np.quantile(draws, 1 - tail)),
            confidence_level=confidence,
        ),
        float(np.mean(draws > 0)),
    )


def _rank(
    candidates: tuple[CandidateRun, ...],
    comparisons: tuple[ProtocolPairedComparison, ...],
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
        )
    if policy == WinnerPolicy.NO_AUTOMATIC_WINNER:
        return Ranking(
            RankingStatus.UNRESOLVED,
            None,
            names,
            "the protocol forbids automatic winner selection",
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
        )
    decisive = True
    for other in names:
        if other == best:
            continue
        difference = _comparison_direction(comparisons, best, other)
        if difference is None or difference.upper >= 0:
            decisive = False
            break
    if decisive:
        return Ranking(
            RankingStatus.RESOLVED,
            best,
            names,
            "the best candidate improves on every eligible competitor with intervals below zero",
        )
    return Ranking(
        RankingStatus.UNRESOLVED,
        None,
        names,
        "at least one paired interval does not exclude equal predictive performance",
    )


def _comparison_direction(
    comparisons: tuple[ProtocolPairedComparison, ...], left: str, right: str
) -> BootstrapInterval | None:
    for comparison in comparisons:
        interval = comparison.left_minus_right
        if comparison.left_model == left and comparison.right_model == right:
            return interval
        if comparison.left_model == right and comparison.right_model == left:
            return BootstrapInterval(
                estimate=-interval.estimate,
                lower=-interval.upper,
                upper=-interval.lower,
                confidence_level=interval.confidence_level,
            )
    return None


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
