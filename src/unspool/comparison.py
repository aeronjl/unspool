"""Prospective model comparison with declared aggregation and uncertainty units."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from unspool.diagnostics import FitAudit, FitAuditStatus, audit_fit
from unspool.evaluation import FoldEvaluation, evaluate_splits
from unspool.models.base import (
    BehaviourEstimator,
    PredictionMode,
    _protected_array,
    model_capabilities,
)
from unspool.study import Study
from unspool.validation import (
    CohortValidationSplit,
    HistoricalCohortForecastSplit,
    PopulationForecastSplit,
    PopulationValidationSplit,
    ValidationFold,
    ValidationSplit,
)


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    """A point estimate and percentile interval over declared resampling units."""

    estimate: float
    lower: float
    upper: float
    confidence_level: float

    def __post_init__(self) -> None:
        values = (self.estimate, self.lower, self.upper, self.confidence_level)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("bootstrap interval values must be finite")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must lie strictly between zero and one")
        if self.lower > self.upper:
            raise ValueError("bootstrap interval lower bound must not exceed its upper bound")

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-safe representation."""

        return {
            "estimate": self.estimate,
            "lower": self.lower,
            "upper": self.upper,
            "confidence_level": self.confidence_level,
        }


@dataclass(frozen=True, slots=True)
class ProspectiveModelResult:
    """Fold fits and scores for one candidate under a shared comparison design."""

    name: str
    model_signature: str
    evaluations: tuple[FoldEvaluation, ...]
    aggregation_units: tuple[Any, ...]
    unit_log_losses: NDArray[np.float64]
    unit_brier_scores: NDArray[np.float64]
    pooled_log_loss: float
    pooled_brier_score: float
    unit_balanced_log_loss_interval: BootstrapInterval
    audits: tuple[FitAudit, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("model result name must be a non-empty string")
        if not isinstance(self.model_signature, str) or not self.model_signature:
            raise ValueError("model signature must be a non-empty string")
        evaluations = tuple(self.evaluations)
        units = _validated_identifiers(self.aggregation_units, "aggregation_units")
        losses = _protected_array(self.unit_log_losses, dtype=np.float64)
        brier = _protected_array(self.unit_brier_scores, dtype=np.float64)
        audits = tuple(self.audits)
        if not evaluations:
            raise ValueError("model results must contain at least one fold evaluation")
        if losses.shape != (len(units),) or brier.shape != losses.shape:
            raise ValueError("unit scores must contain one value per aggregation unit")
        if not np.all(np.isfinite(losses)) or not np.all(np.isfinite(brier)):
            raise ValueError("unit scores must be finite")
        if np.any(losses < 0) or np.any((brier < 0) | (brier > 1)):
            raise ValueError("unit log losses and Brier scores are outside their valid ranges")
        if len(audits) != len(evaluations):
            raise ValueError("every fold evaluation must retain one fit audit")
        if not np.isfinite(self.pooled_log_loss) or self.pooled_log_loss < 0:
            raise ValueError("pooled_log_loss must be finite and non-negative")
        if not np.isfinite(self.pooled_brier_score) or not 0 <= self.pooled_brier_score <= 1:
            raise ValueError("pooled_brier_score must lie between zero and one")
        if not np.isclose(
            self.unit_balanced_log_loss_interval.estimate,
            float(np.mean(losses)),
            rtol=1e-12,
            atol=1e-15,
        ):
            raise ValueError("log-loss interval estimate must equal the unit-balanced mean")
        object.__setattr__(self, "evaluations", evaluations)
        object.__setattr__(self, "aggregation_units", units)
        object.__setattr__(self, "unit_log_losses", losses)
        object.__setattr__(self, "unit_brier_scores", brier)
        object.__setattr__(self, "audits", audits)

    @property
    def unit_balanced_log_loss(self) -> float:
        """Mean log loss after weighting every aggregation unit equally."""

        return float(np.mean(self.unit_log_losses))

    @property
    def unit_balanced_brier_score(self) -> float:
        """Mean Brier score after weighting every aggregation unit equally."""

        return float(np.mean(self.unit_brier_scores))

    @property
    def audit_status(self) -> FitAuditStatus:
        """Worst normalized status among this candidate's fold fits."""

        statuses = {audit.status for audit in self.audits}
        if FitAuditStatus.FAIL in statuses:
            return FitAuditStatus.FAIL
        if FitAuditStatus.WARNING in statuses:
            return FitAuditStatus.WARNING
        return FitAuditStatus.PASS

    @property
    def n_scored_observations(self) -> int:
        """Number of point forecasts included across all folds."""

        return sum(len(evaluation.pointwise_log_probability) for evaluation in self.evaluations)

    def to_dict(self) -> dict[str, Any]:
        """Return scores and audits without expanding fitted covariance arrays."""

        return {
            "name": self.name,
            "model_signature": self.model_signature,
            "n_folds": len(self.evaluations),
            "n_scored_observations": self.n_scored_observations,
            "audit_status": self.audit_status.value,
            "aggregation_units": [_json_value(unit) for unit in self.aggregation_units],
            "unit_scores": [
                {
                    "unit": _json_value(unit),
                    "log_loss": float(log_loss),
                    "brier_score": float(brier),
                }
                for unit, log_loss, brier in zip(
                    self.aggregation_units,
                    self.unit_log_losses,
                    self.unit_brier_scores,
                    strict=True,
                )
            ],
            "unit_balanced_log_loss": self.unit_balanced_log_loss,
            "unit_balanced_brier_score": self.unit_balanced_brier_score,
            "pooled_log_loss": self.pooled_log_loss,
            "pooled_brier_score": self.pooled_brier_score,
            "unit_balanced_log_loss_interval": self.unit_balanced_log_loss_interval.to_dict(),
            "fit_audits": [audit.to_dict() for audit in self.audits],
        }


@dataclass(frozen=True, slots=True)
class PairedModelComparison:
    """Paired unit-level log-loss difference between two candidates."""

    left_model: str
    right_model: str
    left_minus_right: BootstrapInterval
    bootstrap_probability_positive: float

    def __post_init__(self) -> None:
        if not self.left_model or not self.right_model or self.left_model == self.right_model:
            raise ValueError("paired comparison requires two distinct named models")
        if (
            not np.isfinite(self.bootstrap_probability_positive)
            or not 0 <= self.bootstrap_probability_positive <= 1
        ):
            raise ValueError("bootstrap_probability_positive must lie between zero and one")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation with an explicit difference direction."""

        return {
            "left_model": self.left_model,
            "right_model": self.right_model,
            "direction": "left log loss minus right log loss; positive favors right",
            "left_minus_right": self.left_minus_right.to_dict(),
            "bootstrap_probability_positive": self.bootstrap_probability_positive,
        }


@dataclass(frozen=True, slots=True)
class ProspectiveComparisonReport:
    """A matched, cluster-balanced comparison over common prospective folds."""

    aggregation_column: str
    outcome_column: str
    scored_columns: tuple[str, ...]
    splits: tuple[ValidationFold, ...]
    model_results: tuple[ProspectiveModelResult, ...]
    pairwise_comparisons: tuple[PairedModelComparison, ...]
    bootstrap_resamples: int
    bootstrap_seed: int
    confidence_level: float

    def __post_init__(self) -> None:
        if not self.aggregation_column or not self.outcome_column:
            raise ValueError("aggregation and outcome columns must be non-empty")
        scored_columns = tuple(self.scored_columns)
        if not scored_columns or len(set(scored_columns)) != len(scored_columns):
            raise ValueError("scored_columns must be non-empty and unique")
        if self.outcome_column not in scored_columns:
            raise ValueError("outcome_column must be included in scored_columns")
        splits = tuple(self.splits)
        results = tuple(self.model_results)
        comparisons = tuple(self.pairwise_comparisons)
        _require_positive_integer(self.bootstrap_resamples, "bootstrap_resamples")
        _require_nonnegative_integer(self.bootstrap_seed, "bootstrap_seed")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must lie strictly between zero and one")
        if not splits or not results:
            raise ValueError("comparison reports require splits and model results")
        names = tuple(result.name for result in results)
        if len(set(names)) != len(names):
            raise ValueError("comparison model names must be unique")
        expected_pairs = len(results) * (len(results) - 1) // 2
        if len(comparisons) != expected_pairs:
            raise ValueError("pairwise comparisons must cover every unordered model pair")
        expected_pair_names = {
            (left, right)
            for left_index, left in enumerate(names)
            for right in names[left_index + 1 :]
        }
        observed_pair_names = {
            (comparison.left_model, comparison.right_model) for comparison in comparisons
        }
        if observed_pair_names != expected_pair_names:
            raise ValueError("pairwise comparison names must follow declared model order")
        reference_units = results[0].aggregation_units
        if any(result.aggregation_units != reference_units for result in results[1:]):
            raise ValueError("all candidates must be scored over identical aggregation units")
        for result in results:
            if len(result.evaluations) != len(splits) or any(
                evaluation.split is not split
                for evaluation, split in zip(result.evaluations, splits, strict=True)
            ):
                raise ValueError("every candidate must retain the report's exact folds")
            if result.unit_balanced_log_loss_interval.confidence_level != self.confidence_level:
                raise ValueError("model interval confidence levels must match the report")
        results_by_name = {result.name: result for result in results}
        for comparison in comparisons:
            expected_difference = (
                results_by_name[comparison.left_model].unit_balanced_log_loss
                - results_by_name[comparison.right_model].unit_balanced_log_loss
            )
            if not np.isclose(
                comparison.left_minus_right.estimate,
                expected_difference,
                rtol=1e-12,
                atol=1e-15,
            ):
                raise ValueError("pairwise estimate must equal the matched model difference")
            if comparison.left_minus_right.confidence_level != self.confidence_level:
                raise ValueError("pairwise confidence levels must match the report")
        object.__setattr__(self, "splits", splits)
        object.__setattr__(self, "scored_columns", scored_columns)
        object.__setattr__(self, "model_results", results)
        object.__setattr__(self, "pairwise_comparisons", comparisons)

    @property
    def model_order(self) -> tuple[str, ...]:
        """Candidate names in the caller's declared order."""

        return tuple(result.name for result in self.model_results)

    @property
    def eligible_model_order(self) -> tuple[str, ...]:
        """Candidates without a failed fold audit, in declared order."""

        return tuple(
            result.name
            for result in self.model_results
            if result.audit_status is not FitAuditStatus.FAIL
        )

    @property
    def winner(self) -> str | None:
        """Lowest-loss audit-eligible candidate, or ``None`` when every candidate fails."""

        eligible = tuple(
            result
            for result in self.model_results
            if result.audit_status is not FitAuditStatus.FAIL
        )
        if not eligible:
            return None
        return min(
            eligible,
            key=lambda result: result.unit_balanced_log_loss,
        ).name

    def result_for(self, name: str) -> ProspectiveModelResult:
        """Return one named candidate result."""

        for result in self.model_results:
            if result.name == name:
                return result
        raise KeyError(f"unknown comparison model: {name!r}")

    def comparison_for(self, left: str, right: str) -> PairedModelComparison:
        """Return the stored comparison in its original declared direction."""

        for comparison in self.pairwise_comparisons:
            if comparison.left_model == left and comparison.right_model == right:
                return comparison
        raise KeyError(f"no stored comparison from {left!r} to {right!r}")

    def to_dict(self) -> dict[str, Any]:
        """Return a compact JSON-safe scientific record."""

        return {
            "aggregation_column": self.aggregation_column,
            "outcome_column": self.outcome_column,
            "scored_columns": list(self.scored_columns),
            "bootstrap": {
                "unit": self.aggregation_column,
                "resamples": self.bootstrap_resamples,
                "seed": self.bootstrap_seed,
                "confidence_level": self.confidence_level,
                "interval": "percentile",
            },
            "model_order": list(self.model_order),
            "eligible_model_order": list(self.eligible_model_order),
            "winner_policy": "lowest unit-balanced log loss among non-failed audits",
            "winner_by_unit_balanced_log_loss": self.winner,
            "folds": [_split_provenance(split) for split in self.splits],
            "models": {result.name: result.to_dict() for result in self.model_results},
            "pairwise_log_loss_differences": {
                f"{comparison.left_model}_minus_{comparison.right_model}": (comparison.to_dict())
                for comparison in self.pairwise_comparisons
            },
        }


@dataclass(frozen=True, slots=True)
class NestedSelectionFold:
    """Training-only candidate selection and its untouched outer-fold evaluation."""

    outer_split: ValidationFold
    selected_model: str
    inner_report: ProspectiveComparisonReport
    outer_evaluation: FoldEvaluation

    def __post_init__(self) -> None:
        if self.inner_report.winner is None:
            raise ValueError("inner report must have an audit-eligible winner")
        if self.selected_model != self.inner_report.winner:
            raise ValueError("selected_model must equal the inner report winner")
        if self.outer_evaluation.split is not self.outer_split:
            raise ValueError("outer evaluation must retain the declared outer split")

    def to_dict(self) -> dict[str, Any]:
        """Return nested selection evidence without expanding fitted arrays."""

        return {
            "outer_fold": _split_provenance(self.outer_split),
            "selected_model": self.selected_model,
            "selected_fit_audit": audit_fit(self.outer_evaluation.fit).to_dict(),
            "outer_mean_log_loss": self.outer_evaluation.mean_log_loss,
            "inner_comparison": self.inner_report.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class NestedProspectiveSelectionReport:
    """Outer-fold performance of a training-only model-selection procedure."""

    aggregation_column: str
    outcome_column: str
    scored_columns: tuple[str, ...]
    candidate_names: tuple[str, ...]
    folds: tuple[NestedSelectionFold, ...]
    aggregation_units: tuple[Any, ...]
    unit_log_losses: NDArray[np.float64]
    unit_brier_scores: NDArray[np.float64]
    pooled_log_loss: float
    pooled_brier_score: float
    unit_balanced_log_loss_interval: BootstrapInterval
    bootstrap_resamples: int
    bootstrap_seed: int
    confidence_level: float

    def __post_init__(self) -> None:
        candidates = tuple(self.candidate_names)
        scored_columns = tuple(self.scored_columns)
        if not scored_columns or len(set(scored_columns)) != len(scored_columns):
            raise ValueError("scored_columns must be non-empty and unique")
        if self.outcome_column not in scored_columns:
            raise ValueError("outcome_column must be included in scored_columns")
        folds = tuple(self.folds)
        units = _validated_identifiers(self.aggregation_units, "aggregation_units")
        losses = _protected_array(self.unit_log_losses, dtype=np.float64)
        brier = _protected_array(self.unit_brier_scores, dtype=np.float64)
        if not candidates or len(set(candidates)) != len(candidates):
            raise ValueError("candidate_names must be non-empty and unique")
        if not folds:
            raise ValueError("nested selection requires at least one outer fold")
        if losses.shape != (len(units),) or brier.shape != losses.shape:
            raise ValueError("nested unit scores must align with aggregation units")
        if not np.all(np.isfinite(losses)) or not np.all(np.isfinite(brier)):
            raise ValueError("nested unit scores must be finite")
        if np.any(losses < 0) or np.any((brier < 0) | (brier > 1)):
            raise ValueError("nested unit scores are outside their valid ranges")
        if not np.isfinite(self.pooled_log_loss) or self.pooled_log_loss < 0:
            raise ValueError("pooled_log_loss must be finite and non-negative")
        if not np.isfinite(self.pooled_brier_score) or not 0 <= self.pooled_brier_score <= 1:
            raise ValueError("pooled_brier_score must lie between zero and one")
        _require_positive_integer(self.bootstrap_resamples, "bootstrap_resamples")
        _require_nonnegative_integer(self.bootstrap_seed, "bootstrap_seed")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must lie strictly between zero and one")
        if not np.isclose(
            self.unit_balanced_log_loss_interval.estimate,
            float(np.mean(losses)),
            rtol=1e-12,
            atol=1e-15,
        ):
            raise ValueError("log-loss interval estimate must equal the unit-balanced mean")
        if self.unit_balanced_log_loss_interval.confidence_level != self.confidence_level:
            raise ValueError("interval confidence level must match the nested report")
        for fold in folds:
            if fold.selected_model not in candidates:
                raise ValueError("selected models must belong to candidate_names")
            if fold.inner_report.model_order != candidates:
                raise ValueError("every inner report must preserve candidate_names order")
            if (
                fold.inner_report.aggregation_column != self.aggregation_column
                or fold.inner_report.outcome_column != self.outcome_column
                or fold.inner_report.scored_columns != scored_columns
            ):
                raise ValueError("inner report columns must match the nested report")
        object.__setattr__(self, "candidate_names", candidates)
        object.__setattr__(self, "scored_columns", scored_columns)
        object.__setattr__(self, "folds", folds)
        object.__setattr__(self, "aggregation_units", units)
        object.__setattr__(self, "unit_log_losses", losses)
        object.__setattr__(self, "unit_brier_scores", brier)

    @property
    def unit_balanced_log_loss(self) -> float:
        return float(np.mean(self.unit_log_losses))

    @property
    def unit_balanced_brier_score(self) -> float:
        return float(np.mean(self.unit_brier_scores))

    @property
    def selection_counts(self) -> Mapping[str, int]:
        counts = {name: 0 for name in self.candidate_names}
        for fold in self.folds:
            counts[fold.selected_model] += 1
        return MappingProxyType(counts)

    @property
    def audit_status(self) -> FitAuditStatus:
        """Worst normalized status among the selected outer-fold fits."""

        statuses = {audit_fit(fold.outer_evaluation.fit).status for fold in self.folds}
        if FitAuditStatus.FAIL in statuses:
            return FitAuditStatus.FAIL
        if FitAuditStatus.WARNING in statuses:
            return FitAuditStatus.WARNING
        return FitAuditStatus.PASS

    @property
    def n_scored_observations(self) -> int:
        """Number of selected-procedure point forecasts across outer folds."""

        return sum(len(fold.outer_evaluation.pointwise_log_probability) for fold in self.folds)

    def to_dict(self) -> dict[str, Any]:
        """Return the complete inner-selection and outer-evaluation record."""

        return {
            "aggregation_column": self.aggregation_column,
            "outcome_column": self.outcome_column,
            "scored_columns": list(self.scored_columns),
            "candidate_names": list(self.candidate_names),
            "selection_counts": dict(self.selection_counts),
            "n_outer_folds": len(self.folds),
            "n_scored_observations": self.n_scored_observations,
            "audit_status": self.audit_status.value,
            "aggregation_units": [_json_value(unit) for unit in self.aggregation_units],
            "unit_scores": [
                {
                    "unit": _json_value(unit),
                    "log_loss": float(log_loss),
                    "brier_score": float(brier),
                }
                for unit, log_loss, brier in zip(
                    self.aggregation_units,
                    self.unit_log_losses,
                    self.unit_brier_scores,
                    strict=True,
                )
            ],
            "unit_balanced_log_loss": self.unit_balanced_log_loss,
            "unit_balanced_brier_score": self.unit_balanced_brier_score,
            "pooled_log_loss": self.pooled_log_loss,
            "pooled_brier_score": self.pooled_brier_score,
            "unit_balanced_log_loss_interval": self.unit_balanced_log_loss_interval.to_dict(),
            "bootstrap": {
                "unit": self.aggregation_column,
                "resamples": self.bootstrap_resamples,
                "seed": self.bootstrap_seed,
                "confidence_level": self.confidence_level,
                "interval": "percentile",
            },
            "folds": [fold.to_dict() for fold in self.folds],
        }


def compare_models(
    models: Mapping[str, BehaviourEstimator],
    study: Study,
    splits: Iterable[ValidationFold],
    *,
    aggregation_column: str = "subject",
    outcome_column: str = "choice",
    mode: PredictionMode = PredictionMode.FILTERED,
    bootstrap_resamples: int = 5_000,
    bootstrap_seed: int = 0,
    confidence_level: float = 0.95,
) -> ProspectiveComparisonReport:
    """Compare candidates on common folds using equal aggregation-unit weights.

    The same bootstrap draws are reused for every candidate and pairwise difference.
    Point-estimate ties are resolved by the insertion order of ``models``.
    """

    candidates = _validated_models(models)
    capabilities = {name: model_capabilities(model) for name, model in candidates.items()}
    scored_columns = next(iter(capabilities.values())).scored_columns
    for name, candidate_capabilities in capabilities.items():
        if candidate_capabilities.scored_columns != scored_columns:
            raise ValueError(
                "all candidates must score the same observed columns; "
                f"candidate {name!r} scores {candidate_capabilities.scored_columns!r}, "
                f"expected {scored_columns!r}"
            )
    folds = tuple(splits)
    _validate_comparison_inputs(
        study,
        folds,
        aggregation_column=aggregation_column,
        outcome_column=outcome_column,
        scored_columns=scored_columns,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level,
    )
    evaluations = {
        name: evaluate_splits(model, study, folds, mode=mode) for name, model in candidates.items()
    }
    aggregates = {
        name: _aggregate_evaluations(
            study,
            result,
            aggregation_column=aggregation_column,
            outcome_column=outcome_column,
        )
        for name, result in evaluations.items()
    }
    reference_units = next(iter(aggregates.values())).units
    if any(aggregate.units != reference_units for aggregate in aggregates.values()):
        raise ValueError("all candidate evaluations must cover identical aggregation units")

    generator = np.random.default_rng(bootstrap_seed)
    draws = generator.integers(
        0,
        len(reference_units),
        size=(bootstrap_resamples, len(reference_units)),
    )
    model_results: list[ProspectiveModelResult] = []
    for name, model in candidates.items():
        aggregate = aggregates[name]
        bootstrap_means = np.mean(aggregate.unit_log_losses[draws], axis=1)
        model_results.append(
            ProspectiveModelResult(
                name=name,
                model_signature=model.signature,
                evaluations=evaluations[name],
                aggregation_units=aggregate.units,
                unit_log_losses=aggregate.unit_log_losses,
                unit_brier_scores=aggregate.unit_brier_scores,
                pooled_log_loss=aggregate.pooled_log_loss,
                pooled_brier_score=aggregate.pooled_brier_score,
                unit_balanced_log_loss_interval=_bootstrap_interval(
                    aggregate.unit_log_losses,
                    bootstrap_means,
                    confidence_level,
                ),
                audits=tuple(audit_fit(evaluation.fit) for evaluation in evaluations[name]),
            )
        )

    pairwise: list[PairedModelComparison] = []
    names = tuple(candidates)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            difference = aggregates[left].unit_log_losses - aggregates[right].unit_log_losses
            bootstrap_difference = np.mean(difference[draws], axis=1)
            pairwise.append(
                PairedModelComparison(
                    left_model=left,
                    right_model=right,
                    left_minus_right=_bootstrap_interval(
                        difference,
                        bootstrap_difference,
                        confidence_level,
                    ),
                    bootstrap_probability_positive=float(np.mean(bootstrap_difference > 0)),
                )
            )
    return ProspectiveComparisonReport(
        aggregation_column=aggregation_column,
        outcome_column=outcome_column,
        scored_columns=scored_columns,
        splits=folds,
        model_results=tuple(model_results),
        pairwise_comparisons=tuple(pairwise),
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level,
    )


def nested_select_model(
    candidates: Mapping[str, BehaviourEstimator],
    study: Study,
    outer_splits: Iterable[ValidationFold],
    inner_splitter: Callable[[Study], Iterable[ValidationFold]],
    *,
    aggregation_column: str = "subject",
    outcome_column: str = "choice",
    mode: PredictionMode = PredictionMode.FILTERED,
    bootstrap_resamples: int = 5_000,
    bootstrap_seed: int = 0,
    inner_bootstrap_resamples: int = 1_000,
    confidence_level: float = 0.95,
) -> NestedProspectiveSelectionReport:
    """Select a candidate inside each outer training fold, then score untouched test data.

    ``inner_splitter`` receives only ``study.take(outer_split.train_indices)``. Inner split
    positions are therefore relative to that outer-training study and cannot address an
    outer test row. Candidate insertion order breaks exact inner-score ties.
    """

    models = _validated_models(candidates)
    capabilities = {name: model_capabilities(model) for name, model in models.items()}
    scored_columns = next(iter(capabilities.values())).scored_columns
    for name, candidate_capabilities in capabilities.items():
        if candidate_capabilities.scored_columns != scored_columns:
            raise ValueError(
                "all candidates must score the same observed columns; "
                f"candidate {name!r} scores {candidate_capabilities.scored_columns!r}, "
                f"expected {scored_columns!r}"
            )
    folds = tuple(outer_splits)
    _validate_comparison_inputs(
        study,
        folds,
        aggregation_column=aggregation_column,
        outcome_column=outcome_column,
        scored_columns=scored_columns,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level,
    )
    _require_positive_integer(inner_bootstrap_resamples, "inner_bootstrap_resamples")
    if not callable(inner_splitter):
        raise TypeError("inner_splitter must be callable")

    selections: list[NestedSelectionFold] = []
    outer_evaluations: list[FoldEvaluation] = []
    for fold_index, outer_split in enumerate(folds):
        outer_training = study.take(outer_split.train_indices)
        inner_splits = tuple(inner_splitter(outer_training))
        if not inner_splits:
            raise ValueError(f"inner_splitter produced no folds for outer fold {fold_index}")
        inner_report = compare_models(
            models,
            outer_training,
            inner_splits,
            aggregation_column=aggregation_column,
            outcome_column=outcome_column,
            mode=mode,
            bootstrap_resamples=inner_bootstrap_resamples,
            bootstrap_seed=bootstrap_seed + fold_index + 1,
            confidence_level=confidence_level,
        )
        selected = inner_report.winner
        if selected is None:
            raise RuntimeError(f"all candidates failed fit audit inside outer fold {fold_index}")
        outer_evaluation = evaluate_splits(
            models[selected],
            study,
            (outer_split,),
            mode=mode,
        )[0]
        selections.append(
            NestedSelectionFold(
                outer_split=outer_split,
                selected_model=selected,
                inner_report=inner_report,
                outer_evaluation=outer_evaluation,
            )
        )
        outer_evaluations.append(outer_evaluation)

    aggregate = _aggregate_evaluations(
        study,
        tuple(outer_evaluations),
        aggregation_column=aggregation_column,
        outcome_column=outcome_column,
    )
    generator = np.random.default_rng(bootstrap_seed)
    draws = generator.integers(
        0,
        len(aggregate.units),
        size=(bootstrap_resamples, len(aggregate.units)),
    )
    bootstrap_means = np.mean(aggregate.unit_log_losses[draws], axis=1)
    return NestedProspectiveSelectionReport(
        aggregation_column=aggregation_column,
        outcome_column=outcome_column,
        scored_columns=scored_columns,
        candidate_names=tuple(models),
        folds=tuple(selections),
        aggregation_units=aggregate.units,
        unit_log_losses=aggregate.unit_log_losses,
        unit_brier_scores=aggregate.unit_brier_scores,
        pooled_log_loss=aggregate.pooled_log_loss,
        pooled_brier_score=aggregate.pooled_brier_score,
        unit_balanced_log_loss_interval=_bootstrap_interval(
            aggregate.unit_log_losses,
            bootstrap_means,
            confidence_level,
        ),
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level,
    )


@dataclass(frozen=True, slots=True)
class _AggregatedScores:
    units: tuple[Any, ...]
    unit_log_losses: NDArray[np.float64]
    unit_brier_scores: NDArray[np.float64]
    pooled_log_loss: float
    pooled_brier_score: float


def _aggregate_evaluations(
    study: Study,
    evaluations: tuple[FoldEvaluation, ...],
    *,
    aggregation_column: str,
    outcome_column: str,
) -> _AggregatedScores:
    losses_by_unit: dict[Any, list[float]] = {}
    brier_by_unit: dict[Any, list[float]] = {}
    units: list[Any] = []
    pooled_losses: list[NDArray[np.float64]] = []
    pooled_brier: list[NDArray[np.float64]] = []
    for evaluation in evaluations:
        test = study.take(evaluation.split.test_indices)
        raw_outcomes = np.asarray(test[outcome_column])
        try:
            outcomes = np.asarray(raw_outcomes, dtype=np.float64)
        except (TypeError, ValueError):
            raise ValueError(f"outcome column {outcome_column!r} must be binary numeric") from None
        if not np.all(np.isfinite(outcomes)) or not np.all((outcomes == 0) | (outcomes == 1)):
            raise ValueError(f"outcome column {outcome_column!r} must contain only 0 and 1")
        losses = -evaluation.pointwise_log_probability
        brier = (evaluation.prediction.probability - outcomes) ** 2
        if len(test) != len(losses):
            raise ValueError("fold test rows and pointwise scores must align")
        pooled_losses.append(losses)
        pooled_brier.append(brier)
        for position, raw_unit in enumerate(test[aggregation_column]):
            unit = _validated_identifier(raw_unit, aggregation_column)
            if unit not in losses_by_unit:
                units.append(unit)
                losses_by_unit[unit] = []
                brier_by_unit[unit] = []
            losses_by_unit[unit].append(float(losses[position]))
            brier_by_unit[unit].append(float(brier[position]))
    unit_log_losses = np.asarray(
        [np.mean(losses_by_unit[unit]) for unit in units], dtype=np.float64
    )
    unit_brier_scores = np.asarray(
        [np.mean(brier_by_unit[unit]) for unit in units], dtype=np.float64
    )
    return _AggregatedScores(
        units=tuple(units),
        unit_log_losses=unit_log_losses,
        unit_brier_scores=unit_brier_scores,
        pooled_log_loss=float(np.mean(np.concatenate(pooled_losses))),
        pooled_brier_score=float(np.mean(np.concatenate(pooled_brier))),
    )


def _validated_models(
    models: Mapping[str, BehaviourEstimator],
) -> Mapping[str, BehaviourEstimator]:
    if not isinstance(models, Mapping) or not models:
        raise ValueError("models must be a non-empty mapping")
    validated: dict[str, BehaviourEstimator] = {}
    for name, model in models.items():
        if not isinstance(name, str) or not name:
            raise ValueError("model names must be non-empty strings")
        if not isinstance(model, BehaviourEstimator):
            raise TypeError(f"candidate {name!r} does not satisfy BehaviourEstimator")
        model_capabilities(model)
        validated[name] = model
    return MappingProxyType(validated)


def _validate_comparison_inputs(
    study: Study,
    splits: tuple[ValidationFold, ...],
    *,
    aggregation_column: str,
    outcome_column: str,
    scored_columns: tuple[str, ...],
    bootstrap_resamples: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> None:
    if not splits:
        raise ValueError("comparison requires at least one validation split")
    if not isinstance(aggregation_column, str) or not aggregation_column:
        raise ValueError("aggregation_column must be a non-empty string")
    if aggregation_column not in study.columns:
        raise ValueError(f"study does not contain aggregation column {aggregation_column!r}")
    if not isinstance(outcome_column, str) or not outcome_column:
        raise ValueError("outcome_column must be a non-empty string")
    if outcome_column not in study.columns:
        raise ValueError(f"study does not contain outcome column {outcome_column!r}")
    if outcome_column not in scored_columns:
        raise ValueError(
            f"outcome_column {outcome_column!r} is not among the scored columns {scored_columns!r}"
        )
    missing_scored = set(scored_columns) - set(study.columns)
    if missing_scored:
        raise ValueError(f"study is missing scored model columns: {sorted(missing_scored)}")
    _require_positive_integer(bootstrap_resamples, "bootstrap_resamples")
    _require_nonnegative_integer(bootstrap_seed, "bootstrap_seed")
    if not np.isfinite(confidence_level) or not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie strictly between zero and one")
    if any(not split.prospective for split in splits):
        raise ValueError("prospective comparison requires every split to be prospective")


def _bootstrap_interval(
    observed: NDArray[np.float64],
    bootstrap: NDArray[np.float64],
    confidence_level: float,
) -> BootstrapInterval:
    # Decimal confidence levels such as 0.95 otherwise produce a binary tail slightly
    # above 0.025, which creates meaningless last-bit churn in pinned JSON artifacts.
    tail = round((1.0 - confidence_level) / 2.0, 15)
    lower, upper = np.quantile(bootstrap, (tail, 1.0 - tail))
    return BootstrapInterval(
        estimate=float(np.mean(observed)),
        lower=float(lower),
        upper=float(upper),
        confidence_level=confidence_level,
    )


def _split_provenance(split: ValidationFold) -> dict[str, Any]:
    common: dict[str, Any] = {
        "scheme": split.scheme,
        "prospective": split.prospective,
        "n_train_rows": len(split.train_indices),
        "n_test_rows": len(split.test_indices),
        "n_prediction_context_rows": len(split.prediction_context_indices),
    }
    if isinstance(split, ValidationSplit):
        common.update(
            {
                "subject": _json_value(split.subject),
                "train_sessions": [_json_value(value) for value in split.train_sessions],
                "test_sessions": [_json_value(value) for value in split.test_sessions],
                "train_session_orders": list(split.train_session_orders),
                "test_session_orders": list(split.test_session_orders),
                "origin_session": _json_value(split.origin_session),
                "origin_session_order": split.origin_session_order,
                "origin_trial": split.origin_trial,
                "test_trials": list(split.test_trials),
            }
        )
    elif isinstance(split, CohortValidationSplit):
        common.update(
            {
                "subjects": [_json_value(subject) for subject in split.subjects],
                "train_session_count": split.train_session_count,
                "sessions_by_subject": [
                    {
                        "subject": _json_value(subject),
                        "train_sessions": [
                            _json_value(value) for value in split.train_sessions[subject]
                        ],
                        "test_sessions": [
                            _json_value(value) for value in split.test_sessions[subject]
                        ],
                        "train_session_orders": list(split.train_session_orders[subject]),
                        "test_session_orders": list(split.test_session_orders[subject]),
                    }
                    for subject in split.subjects
                ],
            }
        )
    elif isinstance(split, HistoricalCohortForecastSplit):
        common.update(
            {
                "fold_index": split.fold_index,
                "n_folds": split.n_folds,
                "reference_subjects": [_json_value(value) for value in split.reference_subjects],
                "forecast_subjects": [_json_value(value) for value in split.forecast_subjects],
                "reference_session_orders": list(split.reference_session_orders),
                "context_session_orders": list(split.context_session_orders),
                "test_session_orders": list(split.test_session_orders),
            }
        )
    elif isinstance(split, PopulationValidationSplit):
        common.update(
            {
                "train_subjects": [_json_value(value) for value in split.train_subjects],
                "test_subjects": [_json_value(value) for value in split.test_subjects],
                "train_groups": [_json_value(value) for value in split.train_groups],
                "test_groups": [_json_value(value) for value in split.test_groups],
                "held_out_group": _json_value(split.held_out_group),
                "group_column": split.group_column,
            }
        )
    elif isinstance(split, PopulationForecastSplit):
        common.update(
            {
                "train_subjects": [_json_value(value) for value in split.train_subjects],
                "test_subjects": [_json_value(value) for value in split.test_subjects],
                "train_groups": [_json_value(value) for value in split.train_groups],
                "test_groups": [_json_value(value) for value in split.test_groups],
                "held_out_group": _json_value(split.held_out_group),
                "group_column": split.group_column,
                "train_session_orders": list(split.train_session_orders),
                "test_session_orders": list(split.test_session_orders),
                "train_sessions_by_subject": [
                    {
                        "subject": _json_value(subject),
                        "sessions": [_json_value(value) for value in split.train_sessions[subject]],
                    }
                    for subject in split.train_subjects
                ],
                "test_sessions_by_subject": [
                    {
                        "subject": _json_value(subject),
                        "sessions": [_json_value(value) for value in split.test_sessions[subject]],
                    }
                    for subject in split.test_subjects
                ],
            }
        )
    return common


def _validated_identifiers(values: tuple[Any, ...], name: str) -> tuple[Any, ...]:
    identifiers = tuple(
        _validated_identifier(value, f"{name}[{position}]") for position, value in enumerate(values)
    )
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{name} must be non-empty and unique")
    return identifiers


def _validated_identifier(value: Any, name: str) -> Any:
    identifier = value.item() if isinstance(value, np.generic) else value
    if identifier is None:
        raise ValueError(f"{name} must not be missing")
    if isinstance(identifier, (float, complex)) and np.isnan(identifier):
        raise ValueError(f"{name} must not be missing")
    try:
        hash(identifier)
    except TypeError:
        raise TypeError(f"{name} must be hashable") from None
    return identifier


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _require_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _require_nonnegative_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
