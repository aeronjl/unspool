"""Prospective model comparison with declared aggregation and uncertainty units.

Candidates may be frequentist :class:`~behavio.contracts.estimator.BehaviourEstimator`
implementations, sampled
:class:`~behavio.contracts.posterior.PosteriorBehaviourEstimator` implementations, or a
mixture of the two: every candidate is scored on the same folds, over the same aggregation
units, with the same bootstrap draws. A sampled candidate whose posterior fails its
convergence audit on any fold reports ``FAIL`` from
:attr:`ProspectiveModelResult.audit_status` and is therefore excluded from
:attr:`ProspectiveComparisonReport.winner` and
:attr:`ProspectiveComparisonReport.eligible_model_order`, exactly as a non-converged
optimizer fit already is. ``ProspectiveModelResult.to_dict`` gains a ``posterior_folds``
entry only when a candidate actually was sampled, so a maximum-likelihood report is
byte-identical to what it was before sampled candidates existed.

What each metric means for a candidate that predicts a density
--------------------------------------------------------------
A candidate whose ``predict`` returns a
:class:`~behavio.contracts.DensityPrediction` -- a drift diffusion, a race, a continuous
confidence report -- is scored on the same two metrics as every other candidate, but only
one of them is defined the same way.

The **log score is unchanged and is the metric that ranks candidates.**
``pointwise_log_prob`` returns the joint log density of the whole observation, so a
response-time model's log loss and a choice-only model's log loss are both proper scores of
whatever each one claims to predict. That is the number
:attr:`ProspectiveComparisonReport.winner` and every paired contrast read.

The **Brier score is a scoring rule for a probability, and a density is not one.** For a
*defective* density -- one split across named categories, which is what a two-boundary
diffusion produces -- integrating the grid away gives genuine choice probabilities, and
those are scored exactly as a categorical candidate's are. That is deliberate: it is the
only reading under which a diffusion and a logistic regression earn comparable Brier scores
on the same rows. It also means the Brier column reads the **discrete margin only** and
says nothing about the latency half of the prediction. For a density with no categorical
margin there is no probability at all, and the comparison raises
:class:`UnscoreableByBrier` rather than reporting a number; see
:func:`_brier_scoreable_margin` for the full argument.

Simultaneous inference
----------------------
``K`` candidates produce ``K(K-1)/2`` pairwise contrasts, all read against the same
interval level. At five candidates that is ten simultaneous tests, so roughly one of them
excludes zero *by construction* even when every candidate predicts identically well; the
uncorrected reading therefore names a winner about four times in ten under a true null.

:class:`ComparisonFamily` makes the family explicit and is always present, whether or not
anything separates: it records the family size, how many contrasts separate at the
declared interval level, how many were expected to by chance, and the exact binomial
probability of seeing at least that many. Which contrasts count as *decisive* is then
decided by a declared multiplicity adjustment (:class:`ComparisonMultiplicity`,
Benjamini-Hochberg by default) applied to the two-sided bootstrap probabilities. Every
per-contrast interval and probability remains unadjusted and fully retained; adjustment
adds :attr:`PairedComparison.adjusted_probability` and
:attr:`PairedComparison.decisive` beside them. A single contrast is never adjusted, so a
two-candidate comparison behaves exactly as it did.

:class:`ComparisonMultiplicity` and :class:`ComparisonFamily` are re-exported here from
:mod:`behavio._internal.scoring`, which owns the step-up arithmetic. They live there
so that :mod:`behavio.protocol.schema` can freeze the adjustment in
:class:`~behavio.protocol.ComparisonSpec` and
:mod:`behavio.posterior.comparison` can size its own ELPD family, without either module
importing the estimator stack this one rests on. The names, values and behaviour are
unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from behavio._internal.arrays import protected_array
from behavio._internal.parallel import WorkerBackend, map_ordered, resolve_workers
from behavio._internal.scoring import (
    ComparisonFamily,
    ComparisonMultiplicity,
    ScoreMetric,
    adjust_probabilities,
    excess_probability,
)
from behavio.contracts.posterior import (
    AnyBehaviourEstimator,
    any_model_capabilities,
    is_posterior_estimator,
)
from behavio.diagnostics import FitAudit, FitAuditStatus
from behavio.evaluate.folds import (
    FoldEvaluation,
    PosteriorFoldPolicy,
    SplitEvaluation,
    evaluate_splits,
)
from behavio.evaluate.splits import (
    CohortSplit,
    EvaluationFold,
    HistoricalCohortForecastSplit,
    PopulationForecastSplit,
    PopulationSplit,
    Split,
)
from behavio.models.base import (
    BehaviourEstimator,
    CategoricalPrediction,
    DensityPrediction,
    PredictionMode,
)
from behavio.trials import Study


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

    @property
    def excludes_zero(self) -> bool:
        """Whether the whole interval lies strictly on one side of zero."""

        return self.upper < 0 or self.lower > 0


def bootstrap_unit_draws(n_units: int, *, resamples: int, seed: int) -> NDArray[np.intp]:
    """Return the one resampling-index matrix every candidate and contrast shares.

    Drawing unit *indices* once, rather than resampling each candidate's scores
    independently, is what makes the comparison matched: candidate A and candidate B are
    always averaged over the same bootstrap cohort, so their difference is a paired
    difference rather than the difference of two independently noisy means.

    The matrix depends only on ``(n_units, resamples, seed)``, so a caller that recomputes
    it later gets the identical draws. That is why the protocol runner can call
    :func:`bootstrap_interval` once per candidate and still obtain matched intervals.
    """

    if n_units < 1:
        raise ValueError("bootstrap resampling requires at least one aggregation unit")
    _require_positive_integer(resamples, "resamples")
    _require_nonnegative_integer(seed, "seed")
    return np.asarray(
        np.random.default_rng(seed).integers(0, n_units, size=(resamples, n_units)),
        dtype=np.intp,
    )


def bootstrap_interval(
    values: Sequence[float] | NDArray[np.float64],
    *,
    resamples: int,
    seed: int,
    confidence_level: float,
) -> BootstrapInterval:
    """Percentile-bootstrap interval for the equal-unit mean of ``values``.

    This is the package's only bootstrap interval. Both the interactive comparison and the
    protocol runner call it, so a number computed through one entry point equals the
    number computed through the other bit for bit.
    """

    observed = np.asarray(values, dtype=np.float64)
    if observed.ndim != 1:
        raise ValueError("bootstrap values must be one-dimensional")
    draws = bootstrap_unit_draws(len(observed), resamples=resamples, seed=seed)
    return _bootstrap_interval(observed, np.mean(observed[draws], axis=1), confidence_level)


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

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("model result name must be a non-empty string")
        if not isinstance(self.model_signature, str) or not self.model_signature:
            raise ValueError("model signature must be a non-empty string")
        evaluations = tuple(self.evaluations)
        units = _validated_identifiers(self.aggregation_units, "aggregation_units")
        losses = protected_array(self.unit_log_losses, dtype=np.float64)
        brier = protected_array(self.unit_brier_scores, dtype=np.float64)
        if not evaluations:
            raise ValueError("model results must contain at least one fold evaluation")
        if losses.shape != (len(units),) or brier.shape != losses.shape:
            raise ValueError("unit scores must contain one value per aggregation unit")
        if not np.all(np.isfinite(losses)) or not np.all(np.isfinite(brier)):
            raise ValueError("unit scores must be finite")
        if np.any(losses < 0) or np.any((brier < 0) | (brier > 1)):
            raise ValueError("unit log losses and Brier scores are outside their valid ranges")
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

    @property
    def audits(self) -> tuple[FitAudit, ...]:
        """Each fold fit's normalized audit, computed once by ``evaluate_splits``."""

        return tuple(evaluation.fit_audit for evaluation in self.evaluations)

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

    @property
    def posterior_folds(self) -> tuple[tuple[int, FoldEvaluation], ...]:
        """Folds scored from a posterior, paired with their position in ``evaluations``."""

        return tuple(
            (index, evaluation)
            for index, evaluation in enumerate(self.evaluations)
            if evaluation.posterior is not None
        )

    def to_dict(self) -> dict[str, Any]:
        """Return scores and audits without expanding fitted covariance arrays.

        ``posterior_folds`` appears only when at least one fold was projected from
        posterior draws; an optimizer-only candidate keeps its original record exactly.
        """

        payload: dict[str, Any] = {
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
        posterior_folds = self.posterior_folds
        if posterior_folds:
            payload["posterior_folds"] = [
                {"fold": index, **evaluation.posterior.to_dict()}
                for index, evaluation in posterior_folds
                if evaluation.posterior is not None
            ]
        return payload


@dataclass(frozen=True, slots=True)
class PairedComparison:
    """One paired equal-unit score difference between two candidates.

    ``left_minus_right`` and ``bootstrap_probability_positive`` are unadjusted and mean
    exactly what they always meant. ``two_sided_probability`` inverts the percentile
    bootstrap into a probability of a difference at least this extreme under no
    difference, with the Davison-Hinkley ``+1`` correction so a finite resample can never
    report an impossible zero. ``adjusted_probability`` is that probability after the
    family's declared multiplicity adjustment, and ``decisive`` is the only member the
    winner rule reads.
    """

    left_model: str
    right_model: str
    metric: ScoreMetric
    left_minus_right: BootstrapInterval
    bootstrap_probability_positive: float
    two_sided_probability: float = 1.0
    adjusted_probability: float = 1.0
    decisive: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "metric", ScoreMetric(self.metric))
        if not self.left_model or not self.right_model or self.left_model == self.right_model:
            raise ValueError("paired comparison requires two distinct named models")
        for value, label in (
            (self.bootstrap_probability_positive, "bootstrap_probability_positive"),
            (self.two_sided_probability, "two_sided_probability"),
            (self.adjusted_probability, "adjusted_probability"),
        ):
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{label} must lie between zero and one")
        if not isinstance(self.decisive, bool):
            raise TypeError("decisive must be boolean")

    @property
    def favours(self) -> str | None:
        """The named candidate this contrast decisively favours, or ``None``."""

        if not self.decisive:
            return None
        return self.left_model if self.left_minus_right.estimate < 0 else self.right_model

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation with an explicit difference direction."""

        metric = self.metric.value
        return {
            "left_model": self.left_model,
            "right_model": self.right_model,
            "metric": metric,
            "direction": f"left {metric} minus right {metric}; positive favors right",
            "left_minus_right": self.left_minus_right.to_dict(),
            "bootstrap_probability_positive": self.bootstrap_probability_positive,
            "two_sided_probability": self.two_sided_probability,
            "adjusted_probability": self.adjusted_probability,
            "decisive": self.decisive,
        }


def paired_comparisons(
    unit_scores: Mapping[str, Mapping[Any, float]],
    *,
    metric: ScoreMetric,
    resamples: int,
    seed: int,
    confidence_level: float,
    multiplicity: ComparisonMultiplicity = ComparisonMultiplicity.BENJAMINI_HOCHBERG,
    family_error_rate: float | None = None,
) -> tuple[tuple[PairedComparison, ...], ComparisonFamily]:
    """Compare every unordered pair of candidates and size the resulting family.

    ``unit_scores`` maps each candidate name, in the order the caller declared, to that
    candidate's per-aggregation-unit score keyed by unit. A pair is compared over the units
    both candidates scored, in the left candidate's order; a pair with no common unit is
    skipped and does not enter the family.

    ``family_error_rate`` defaults to ``1 - confidence_level``, so the family-level rate is
    the rate the caller already declared rather than a second threshold introduced behind
    their back. Nothing about the per-contrast interval changes.
    """

    metric = ScoreMetric(metric)
    adjustment = ComparisonMultiplicity(multiplicity)
    rate = (1.0 - confidence_level) if family_error_rate is None else family_error_rate
    names = tuple(unit_scores)
    # Every pair over the same number of common units shares one index matrix. Rebuilding
    # it per contrast is pure waste: `bootstrap_unit_draws` is a deterministic function of
    # `(n_units, resamples, seed)`, and in the common case where all candidates cover
    # identical units there is exactly one matrix for the whole family.
    draws_for: dict[int, NDArray[np.intp]] = {}
    unadjusted: list[PairedComparison] = []
    for left_index, left in enumerate(names):
        left_scores = unit_scores[left]
        for right in names[left_index + 1 :]:
            right_scores = unit_scores[right]
            common = tuple(key for key in left_scores if key in right_scores)
            if not common:
                continue
            differences = np.asarray(
                [left_scores[key] - right_scores[key] for key in common],
                dtype=np.float64,
            )
            if len(differences) not in draws_for:
                draws_for[len(differences)] = bootstrap_unit_draws(
                    len(differences), resamples=resamples, seed=seed
                )
            draws = np.mean(differences[draws_for[len(differences)]], axis=1)
            n_positive = int(np.count_nonzero(draws > 0))
            n_negative = int(np.count_nonzero(draws < 0))
            unadjusted.append(
                PairedComparison(
                    left_model=left,
                    right_model=right,
                    metric=metric,
                    left_minus_right=_bootstrap_interval(differences, draws, confidence_level),
                    bootstrap_probability_positive=n_positive / len(draws),
                    # Davison and Hinkley's (1997) +1 correction: a percentile bootstrap
                    # over B draws cannot resolve a probability below 1/(B+1), and
                    # reporting an exact zero would claim a certainty the resample never
                    # established.
                    two_sided_probability=min(
                        1.0, 2.0 * (min(n_positive, n_negative) + 1) / (len(draws) + 1)
                    ),
                )
            )

    probabilities = np.asarray(
        [item.two_sided_probability for item in unadjusted], dtype=np.float64
    )
    separated = np.asarray([item.left_minus_right.excludes_zero for item in unadjusted], dtype=bool)
    per_contrast = 1.0 - confidence_level
    threshold, adjusted = adjust_probabilities(
        probabilities,
        multiplicity=adjustment,
        error_rate=rate,
        unadjusted_threshold=per_contrast,
    )
    # A contrast must both survive the adjustment and have an unadjusted interval that
    # excludes zero. The second condition is what keeps adjustment from ever *adding* a
    # decisive contrast: correction can only take separations away.
    #
    # An unadjusted family reads the interval and nothing else. It must not additionally
    # consult the inverted probability, because the two can disagree at small resample
    # counts: with B draws the two-sided probability cannot fall below 2/(B+1), so at
    # B = 16 it is pinned at 0.118 while the 95% percentile interval can still exclude
    # zero. Correcting is a decision about *multiplicity*; it must not smuggle in a second,
    # stricter reading of a single contrast.
    corrected = len(unadjusted) > 1 and adjustment is not ComparisonMultiplicity.NONE
    decisive = separated & (adjusted <= rate) if corrected else separated
    n_separated = int(np.count_nonzero(separated))
    family = ComparisonFamily(
        n_candidates=len(names),
        n_comparisons=len(unadjusted),
        interval_level=confidence_level,
        multiplicity=adjustment,
        family_error_rate=rate,
        n_separated=n_separated,
        expected_separated=float(len(unadjusted) * per_contrast),
        excess_probability=excess_probability(n_separated, len(unadjusted), per_contrast),
        adjusted_threshold=float(threshold),
        n_decisive=int(np.count_nonzero(decisive)),
    )
    comparisons = tuple(
        replace(
            comparison,
            adjusted_probability=float(adjusted_probability),
            decisive=bool(is_decisive),
        )
        for comparison, adjusted_probability, is_decisive in zip(
            unadjusted, adjusted, decisive, strict=True
        )
    )
    return comparisons, family


@dataclass(frozen=True, slots=True)
class ProspectiveComparisonReport:
    """A matched, cluster-balanced comparison over common prospective folds."""

    aggregation_column: str
    outcome_column: str
    scored_columns: tuple[str, ...]
    splits: tuple[EvaluationFold, ...]
    model_results: tuple[ProspectiveModelResult, ...]
    pairwise_comparisons: tuple[PairedComparison, ...]
    bootstrap_resamples: int
    bootstrap_seed: int
    confidence_level: float
    family: ComparisonFamily

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
        if not isinstance(self.family, ComparisonFamily):
            raise TypeError("family must be a ComparisonFamily")
        if self.family.n_comparisons != len(comparisons):
            raise ValueError("the comparison family must size the retained contrasts")
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

    @property
    def decisive_comparisons(self) -> tuple[PairedComparison, ...]:
        """Contrasts that survive the declared multiplicity adjustment."""

        return tuple(comparison for comparison in self.pairwise_comparisons if comparison.decisive)

    def comparison_for(self, left: str, right: str) -> PairedComparison:
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
            "simultaneous_family": self.family.to_dict(),
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

    outer_split: EvaluationFold
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
        """Return nested selection evidence without expanding fitted arrays.

        ``selected_posterior`` appears only when the selected candidate was sampled.
        """

        payload: dict[str, Any] = {
            "outer_fold": _split_provenance(self.outer_split),
            "selected_model": self.selected_model,
            "selected_fit_audit": self.outer_evaluation.fit_audit.to_dict(),
            "outer_mean_log_loss": self.outer_evaluation.mean_log_loss,
            "inner_comparison": self.inner_report.to_dict(),
        }
        if self.outer_evaluation.posterior is not None:
            payload["selected_posterior"] = self.outer_evaluation.posterior.to_dict()
        return payload


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
        losses = protected_array(self.unit_log_losses, dtype=np.float64)
        brier = protected_array(self.unit_brier_scores, dtype=np.float64)
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

        statuses = {fold.outer_evaluation.fit_audit.status for fold in self.folds}
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


@dataclass(frozen=True, slots=True)
class _CandidateTask:
    """One candidate's whole fold evaluation, holding nothing shared with another.

    ``evaluate_splits`` is already pure over ``(model, study, folds)`` -- it fits inside
    each fold from scratch and reads no state that survives a candidate -- so a candidate is
    a work item without any further adaptation.
    """

    model: AnyBehaviourEstimator
    study: Study
    folds: tuple[EvaluationFold, ...]
    mode: PredictionMode
    posterior_policy: PosteriorFoldPolicy | None


def _evaluate_candidate(task: _CandidateTask) -> SplitEvaluation:
    return evaluate_splits(
        task.model,
        task.study,
        task.folds,
        mode=task.mode,
        posterior_policy=task.posterior_policy,
    )


def _rebind_splits(
    evaluated: SplitEvaluation, folds: tuple[EvaluationFold, ...]
) -> SplitEvaluation:
    """Restore the caller's own fold objects after a round trip through a worker.

    A fold that crosses a process boundary comes back as an equal but distinct object, and
    this package checks fold provenance by *identity*:
    :meth:`ProspectiveComparisonReport.__post_init__` requires that every candidate retained
    the report's exact folds, which is what stops two candidates being compared over folds
    that merely look alike. Rebinding here makes the parallel result the serial result --
    the parent's fold objects, in the parent's order -- rather than weakening the check that
    would otherwise notice the difference.

    ``workers=1`` never reaches the rebinding branch, because the evaluation already holds
    the very objects it was given.
    """

    if all(
        evaluation.split is fold
        for evaluation, fold in zip(evaluated.evaluations, folds, strict=False)
    ):
        return evaluated
    rebound = tuple(
        evaluation if evaluation.split is fold else replace(evaluation, split=fold)
        for evaluation, fold in zip(evaluated.evaluations, folds, strict=True)
    )
    return SplitEvaluation(rebound, evaluated.failures, evaluated.policy)


def compare_models(
    models: Mapping[str, AnyBehaviourEstimator],
    study: Study,
    splits: Iterable[EvaluationFold],
    *,
    aggregation_column: str = "subject",
    outcome_column: str = "choice",
    mode: PredictionMode = PredictionMode.FILTERED,
    bootstrap_resamples: int = 5_000,
    bootstrap_seed: int = 0,
    confidence_level: float = 0.95,
    posterior_policy: PosteriorFoldPolicy | None = None,
    multiplicity: ComparisonMultiplicity = ComparisonMultiplicity.BENJAMINI_HOCHBERG,
    family_error_rate: float | None = None,
    workers: int = 1,
    backend: WorkerBackend | str = WorkerBackend.PROCESS,
) -> ProspectiveComparisonReport:
    """Compare frequentist and sampled candidates on common folds with equal unit weights.

    The same bootstrap draws are reused for every candidate and pairwise difference.
    Point-estimate ties are resolved by the insertion order of ``models``.
    ``posterior_policy`` declares the projection centre and convergence thresholds applied
    to every sampled candidate; it is ignored by candidates fitted by optimization.

    ``multiplicity`` and ``family_error_rate`` govern which pairwise contrasts are marked
    :attr:`PairedComparison.decisive` once more than one contrast exists; they never change
    an interval, a probability, or :attr:`ProspectiveComparisonReport.winner`, which has
    always been the lowest point estimate among audit-eligible candidates.

    ``workers`` evaluates candidates concurrently. Only the fold evaluation is scheduled;
    aggregation, bootstrapping and ranking stay in this process and stay in the insertion
    order of ``models``, which is what breaks point-estimate ties. The report is therefore
    bit-identical for every worker count: the bootstrap seed is a declared constant rather
    than a generator advanced per candidate, so no candidate's draws depend on which
    candidate ran first. Parallelism is bounded by the number of candidates -- two
    candidates cannot use four workers -- so the fold-level and repeat-level entry points
    scale further.
    """

    candidates = _validated_models(models)
    capabilities = {name: any_model_capabilities(model) for name, model in candidates.items()}
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
    completed = map_ordered(
        _evaluate_candidate,
        tuple(
            _CandidateTask(
                model=model,
                study=study,
                folds=folds,
                mode=mode,
                posterior_policy=_policy_for(model, posterior_policy),
            )
            for model in candidates.values()
        ),
        workers=workers,
        backend=WorkerBackend(backend),
    )
    evaluations = {
        name: _rebind_splits(evaluated, folds)
        for name, evaluated in zip(candidates, completed, strict=True)
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

    model_results: list[ProspectiveModelResult] = []
    for name, model in candidates.items():
        aggregate = aggregates[name]
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
                unit_balanced_log_loss_interval=bootstrap_interval(
                    aggregate.unit_log_losses,
                    resamples=bootstrap_resamples,
                    seed=bootstrap_seed,
                    confidence_level=confidence_level,
                ),
            )
        )

    pairwise, family = paired_comparisons(
        {
            name: dict(zip(reference_units, aggregates[name].unit_log_losses, strict=True))
            for name in candidates
        },
        metric=ScoreMetric.LOG_LOSS,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
        confidence_level=confidence_level,
        multiplicity=multiplicity,
        family_error_rate=family_error_rate,
    )
    return ProspectiveComparisonReport(
        aggregation_column=aggregation_column,
        outcome_column=outcome_column,
        scored_columns=scored_columns,
        splits=folds,
        model_results=tuple(model_results),
        pairwise_comparisons=pairwise,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level,
        family=family,
    )


@dataclass(frozen=True, slots=True)
class _OuterFoldTask:
    """One outer fold's whole nested selection: inner comparison, then the outer score.

    ``bootstrap_seed`` is carried as the caller's root rather than as this fold's derived
    seed, because the derived value is ``bootstrap_seed + fold_index + 1`` -- a function of
    where the fold sits, not of when it runs. Deriving it inside the worker keeps that
    visible next to the position it is derived from.
    """

    fold_index: int
    outer_split: EvaluationFold
    models: Mapping[str, AnyBehaviourEstimator]
    study: Study
    inner_splitter: Callable[[Study], Iterable[EvaluationFold]]
    aggregation_column: str
    outcome_column: str
    mode: PredictionMode
    inner_bootstrap_resamples: int
    bootstrap_seed: int
    confidence_level: float
    posterior_policy: PosteriorFoldPolicy | None


def _run_nested_outer_fold(task: _OuterFoldTask) -> NestedSelectionFold:
    """Select inside one outer training study and score that fold's untouched test rows.

    The inner comparison runs with ``workers=1`` on purpose. Outer folds are the outer
    level and already saturate the pool, so handing the inner comparison its own pool would
    nest one executor inside another -- which oversubscribes the machine and, with process
    workers, is not something a worker process can do at all.
    """

    outer_training = task.study.take(task.outer_split.train_indices)
    inner_splits = tuple(task.inner_splitter(outer_training))
    if not inner_splits:
        raise ValueError(f"inner_splitter produced no folds for outer fold {task.fold_index}")
    inner_report = compare_models(
        task.models,
        outer_training,
        inner_splits,
        aggregation_column=task.aggregation_column,
        outcome_column=task.outcome_column,
        mode=task.mode,
        bootstrap_resamples=task.inner_bootstrap_resamples,
        bootstrap_seed=task.bootstrap_seed + task.fold_index + 1,
        confidence_level=task.confidence_level,
        posterior_policy=task.posterior_policy,
    )
    selected = inner_report.winner
    if selected is None:
        raise RuntimeError(f"all candidates failed fit audit inside outer fold {task.fold_index}")
    outer_evaluation = evaluate_splits(
        task.models[selected],
        task.study,
        (task.outer_split,),
        mode=task.mode,
        posterior_policy=_policy_for(task.models[selected], task.posterior_policy),
    )[0]
    return NestedSelectionFold(
        outer_split=task.outer_split,
        selected_model=selected,
        inner_report=inner_report,
        outer_evaluation=outer_evaluation,
    )


def nested_select_model(
    candidates: Mapping[str, AnyBehaviourEstimator],
    study: Study,
    outer_splits: Iterable[EvaluationFold],
    inner_splitter: Callable[[Study], Iterable[EvaluationFold]],
    *,
    aggregation_column: str = "subject",
    outcome_column: str = "choice",
    mode: PredictionMode = PredictionMode.FILTERED,
    bootstrap_resamples: int = 5_000,
    bootstrap_seed: int = 0,
    inner_bootstrap_resamples: int = 1_000,
    confidence_level: float = 0.95,
    posterior_policy: PosteriorFoldPolicy | None = None,
    workers: int = 1,
    backend: WorkerBackend | str = WorkerBackend.PROCESS,
) -> NestedProspectiveSelectionReport:
    """Select a candidate inside each outer training fold, then score untouched test data.

    ``inner_splitter`` receives only ``study.take(outer_split.train_indices)``. Inner split
    positions are therefore relative to that outer-training study and cannot address an
    outer test row. Candidate insertion order breaks exact inner-score ties. A sampled
    candidate whose posterior fails its convergence audit inside an inner fold cannot be
    selected there, because inner selection reuses :attr:`ProspectiveComparisonReport.winner`.

    ``workers`` runs outer folds concurrently, which is the deepest of these loops and the
    one with the most work under it: every candidate is refitted on every inner fold of
    every outer fold. Each outer fold's inner bootstrap seed is ``bootstrap_seed +
    fold_index + 1`` -- determined by the fold's position, not by how many folds preceded it
    at runtime -- so the report is bit-identical for every worker count, and the selected
    model per fold cannot change.

    ``backend="process"`` requires ``inner_splitter`` to be picklable. A lambda is not, and
    a lambda is the natural way to write one, so this is the entry point most likely to
    need either a :func:`functools.partial` of a module-level splitter or
    ``backend="thread"``. The failure is raised before any fitting starts and names the
    argument.
    """

    models = _validated_models(candidates)
    capabilities = {name: any_model_capabilities(model) for name, model in models.items()}
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
    resolve_workers(workers, n_tasks=1)

    selections = map_ordered(
        _run_nested_outer_fold,
        tuple(
            _OuterFoldTask(
                fold_index=fold_index,
                outer_split=outer_split,
                # `_validated_models` returns a MappingProxyType, which cannot be pickled.
                # The proxy protects the caller's mapping from this function, not the worker
                # from itself, so a plain copy is what crosses the boundary.
                models=dict(models),
                study=study,
                inner_splitter=inner_splitter,
                aggregation_column=aggregation_column,
                outcome_column=outcome_column,
                mode=mode,
                inner_bootstrap_resamples=inner_bootstrap_resamples,
                bootstrap_seed=bootstrap_seed,
                confidence_level=confidence_level,
                posterior_policy=posterior_policy,
            )
            for fold_index, outer_split in enumerate(folds)
        ),
        workers=workers,
        backend=WorkerBackend(backend),
    )
    selections = [
        selection
        if selection.outer_split is fold
        else replace(
            selection,
            outer_split=fold,
            outer_evaluation=replace(selection.outer_evaluation, split=fold),
        )
        for selection, fold in zip(selections, folds, strict=True)
    ]
    outer_evaluations = [selection.outer_evaluation for selection in selections]

    aggregate = _aggregate_evaluations(
        study,
        tuple(outer_evaluations),
        aggregation_column=aggregation_column,
        outcome_column=outcome_column,
    )
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
        unit_balanced_log_loss_interval=bootstrap_interval(
            aggregate.unit_log_losses,
            resamples=bootstrap_resamples,
            seed=bootstrap_seed,
            confidence_level=confidence_level,
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
        losses = -evaluation.pointwise_log_probability
        prediction = evaluation.prediction
        if isinstance(prediction, DensityPrediction):
            prediction = _brier_scoreable_margin(prediction)
        if isinstance(prediction, CategoricalPrediction):
            if evaluation.outcome_codes is None:
                raise ValueError("categorical fold is missing observed outcome codes")
            targets = np.zeros_like(prediction.probability)
            targets[np.arange(len(targets)), evaluation.outcome_codes] = 1.0
            # Half the multicategory squared distance preserves the familiar [0, 1]
            # range and exactly matches the existing binary Brier convention.
            brier = 0.5 * np.sum(
                (prediction.probability - targets) ** 2,
                axis=1,
            )
        else:
            raw_outcomes = np.asarray(test[outcome_column])
            try:
                outcomes = np.asarray(raw_outcomes, dtype=np.float64)
            except (TypeError, ValueError):
                raise ValueError(
                    f"outcome column {outcome_column!r} must be binary numeric"
                ) from None
            if not np.all(np.isfinite(outcomes)) or not np.all((outcomes == 0) | (outcomes == 1)):
                raise ValueError(f"outcome column {outcome_column!r} must contain only 0 and 1")
            brier = (prediction.probability - outcomes) ** 2
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


class UnscoreableByBrier(ValueError):
    """Raised when a candidate's prediction carries no probability a Brier score can read.

    Naming the refusal is the point. A caller who reaches this has not made a mistake about
    columns or folds; they have asked a probability scoring rule to score something that is
    not a probability, and the answer is that the metric does not apply -- not a number that
    looks like one.
    """


def _brier_scoreable_margin(prediction: DensityPrediction) -> CategoricalPrediction:
    """Return the discrete margin of a density, or refuse to invent one.

    **A density is not a probability.** ``p(rt = 0.43) = 1.7`` is a perfectly ordinary value
    for a response-time model to report and it is not a number ``(p - y)**2`` means anything
    about: a Brier score is a squared distance to an indicator, so it needs a quantity
    bounded by one and reported against an outcome that either happened or did not. Rescaling
    a density until it fits that shape would produce a number that ranks models by their
    grid resolution.

    What a *defective* density does carry is a genuine probability: integrating each
    category's density over the grid gives the choice probabilities, and those are exactly
    what a choice-only competitor is scored on. Scoring that margin is therefore not a
    concession -- it is the only reading under which a response-time model and a logistic
    regression earn comparable Brier scores on the same rows, which is what a comparison
    table is for. Two things follow, and both are deliberate:

    - the Brier column of a comparison scores the **discrete margin only**. It says nothing
      about the latency half of the prediction, and a model that gets the choice right and
      the response-time distribution badly wrong will look good in it;
    - the **log score does not have this problem**. ``pointwise_log_prob`` is the joint log
      density of the whole observation, choice and latency together, and it is the metric
      :attr:`ProspectiveComparisonReport.winner` and every paired contrast rank on. The
      Brier column is retained beside it as a secondary, deliberately narrower reading.

    A density with no categories -- an unlabelled continuous outcome, a confidence report --
    has no discrete margin at all, so there is nothing to score and no honest number to
    report. That raises :class:`UnscoreableByBrier` rather than returning ``NaN`` or zero.
    """

    if not prediction.is_defective:
        raise UnscoreableByBrier(
            f"the candidate predicts a density over {prediction.outcome!r} with no "
            "categorical margin, and a Brier score is a scoring rule for a probability, "
            "not for a density; there is no number to report. Compare on the log score, "
            "which is the joint log density of the observation and is defined for this "
            "prediction"
        )
    return prediction.choice_prediction()


def _policy_for(
    model: AnyBehaviourEstimator,
    policy: PosteriorFoldPolicy | None,
) -> PosteriorFoldPolicy | None:
    """Forward a posterior policy only to the candidates it can describe."""

    return policy if is_posterior_estimator(model) else None


def _validated_models(
    models: Mapping[str, AnyBehaviourEstimator],
) -> Mapping[str, AnyBehaviourEstimator]:
    if not isinstance(models, Mapping) or not models:
        raise ValueError("models must be a non-empty mapping")
    validated: dict[str, AnyBehaviourEstimator] = {}
    for name, model in models.items():
        if not isinstance(name, str) or not name:
            raise ValueError("model names must be non-empty strings")
        if not isinstance(model, BehaviourEstimator) and not is_posterior_estimator(model):
            raise TypeError(
                f"candidate {name!r} satisfies neither BehaviourEstimator nor "
                "PosteriorBehaviourEstimator"
            )
        any_model_capabilities(model)
        validated[name] = model
    return MappingProxyType(validated)


def _validate_comparison_inputs(
    study: Study,
    splits: tuple[EvaluationFold, ...],
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


def _split_provenance(split: EvaluationFold) -> dict[str, Any]:
    common: dict[str, Any] = {
        "scheme": split.scheme,
        "prospective": split.prospective,
        "n_train_rows": len(split.train_indices),
        "n_test_rows": len(split.test_indices),
        "n_prediction_context_rows": len(split.prediction_context_indices),
    }
    if isinstance(split, Split):
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
    elif isinstance(split, CohortSplit):
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
    elif isinstance(split, PopulationSplit):
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
