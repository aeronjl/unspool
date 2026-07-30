"""Scoring vocabulary shared by every execution path, and the one multiplicity adjustment.

Behavio evaluates several families of simultaneous tests. A posterior-predictive audit
evaluates ``groups x discrepancies`` tail probabilities
(:mod:`behavio.posterior.predictive`); a frequentist model comparison evaluates
``K(K-1)/2`` pairwise score contrasts (:mod:`behavio.compare.models`); a Bayesian model
comparison evaluates ``K(K-1)/2`` pairwise ELPD contrasts
(:mod:`behavio.posterior.comparison`). The domains are unrelated and their records name
different things, but the arithmetic that turns many probabilities into a decision is the
same arithmetic, and it was written three times before this module existed. The copies had
already begun to disagree.

Each caller keeps its own definition of which item counts as *extreme* -- a tail
probability below a threshold in one case, an interval excluding zero in the others -- and
its own way of obtaining a per-contrast probability: a percentile bootstrap for a paired
score difference, a normal approximation for a paired ELPD difference. What they share,
and what lives here, is the step-up itself, the exact binomial tail that says how many
extremes chance alone supplies, the name of the adjustment
(:class:`ComparisonMultiplicity`), and the record of the family it was applied to
(:class:`ComparisonFamily`).

:class:`ComparisonMultiplicity` and :class:`ComparisonFamily` are private only in their
address. Both are re-exported from :mod:`behavio.compare.models` and from the top-level
``behavio`` namespace under those exact names; they live here so that
:mod:`behavio.protocol.schema`, which must declare the adjustment before data is seen and imports
nothing else from the package, and :mod:`behavio.posterior.comparison`, which must not
drag in the estimator stack, can both reach them without a cycle.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray
from scipy.stats import binom

#: Adjustment names, shared so the public enums cannot drift apart in spelling.
NONE = "none"
BENJAMINI_HOCHBERG = "benjamini-hochberg"
BONFERRONI = "bonferroni"


class ScoreMetric(StrEnum):
    """Proper pointwise scoring rules supported by every execution path.

    This lives here rather than in :mod:`behavio.protocol.schema` for the same reason
    :class:`ComparisonMultiplicity` does: a protocol must be able to *declare* the metric
    before any data is seen, and :mod:`behavio.compare.models` must be able to *apply* it
    without importing the protocol package back.
    """

    LOG_LOSS = "log-loss"
    BRIER = "brier"
    JOINT_LOG_LOSS = "joint-log-loss"


LOG_SCORE_METRICS: Final = (ScoreMetric.LOG_LOSS, ScoreMetric.JOINT_LOG_LOSS)
"""The two spellings of the one log score, which every admissible prediction supports.

``pointwise_log_prob`` is the joint log density of the whole scored observation, so both
names describe the same number; a protocol calls it ``joint-log-loss`` when a candidate
scores more than one observed column and ``log-loss`` when it scores one. Neither depends on
the prediction carrying a probability, which is why a candidate that predicts an unlabelled
density supports these and only these.
"""

PROBABILITY_SCORE_METRICS: Final = (*LOG_SCORE_METRICS, ScoreMetric.BRIER)
"""Every rule a prediction with a discrete margin can carry.

The Brier score is the member that separates the two sets: it is a squared distance to an
indicator, so it needs a probability, and a prediction with no discrete margin has none.
"""


def validated_score_metrics(
    metrics: Iterable[ScoreMetric | str], *, label: str
) -> tuple[ScoreMetric, ...]:
    """Return a declared metric set as an ordered, duplicate-free tuple.

    Declaration order is preserved rather than sorted, because every caller that declares
    more than one rule also declares which of them ranks: the first.
    """

    if isinstance(metrics, (str, ScoreMetric)):
        raise TypeError(f"{label} must be a sequence of scoring rules, not one rule")
    values = tuple(ScoreMetric(metric) for metric in metrics)
    if not values:
        raise ValueError(f"{label} must name at least one scoring rule")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must not name the same scoring rule twice")
    return values


class ComparisonMultiplicity(StrEnum):
    """How one comparison turns many simultaneous contrasts into decisive ones.

    ``NONE`` restores the per-comparison reading: every contrast whose unadjusted interval
    excludes zero is decisive, and the expected number of spurious separations grows with
    the family size. ``BENJAMINI_HOCHBERG`` controls the false-discovery rate across the
    family at ``family_error_rate``. ``BONFERRONI`` controls the family-wise error rate at
    the same level.

    The member values are shared with
    :class:`~behavio.posterior.predictive.PredictiveMultiplicity`, so the families the
    package evaluates cannot drift apart in either spelling or arithmetic.
    """

    NONE = NONE
    BENJAMINI_HOCHBERG = BENJAMINI_HOCHBERG
    BONFERRONI = BONFERRONI


@dataclass(frozen=True, slots=True)
class ComparisonFamily:
    """The simultaneous family of pairwise contrasts evaluated in one comparison.

    Retained whether or not anything separates, so a reader can always compare the number
    of contrasts that excluded zero with the number expected there by chance alone.
    ``excess_probability`` is the exact binomial probability of at least ``n_separated``
    separations among ``n_comparisons`` independent contrasts at per-contrast error rate
    ``1 - interval_level``; contrasts sharing candidates are not independent, so it is a
    guide to whether the pattern is remarkable, not a test.

    The same record sizes a frequentist comparison of paired scores
    (:func:`behavio.compare.models.paired_comparisons`) and a Bayesian comparison of paired
    ELPD differences (:func:`behavio.posterior.comparison.compare_posterior_models`). They
    reach ``interval_level`` differently -- one declares a bootstrap percentile level, the
    other converts its standard-error multiplier into the equivalent normal tail mass --
    and the field means the same thing in both: the per-contrast level below which a
    contrast is read as separating.
    """

    n_candidates: int
    n_comparisons: int
    interval_level: float
    multiplicity: ComparisonMultiplicity
    family_error_rate: float
    n_separated: int
    expected_separated: float
    excess_probability: float
    adjusted_threshold: float
    n_decisive: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "multiplicity", ComparisonMultiplicity(self.multiplicity))
        if self.n_candidates < 0 or self.n_comparisons < 0:
            raise ValueError("comparison family counts must be non-negative")
        if not 0 <= self.n_separated <= self.n_comparisons:
            raise ValueError("separated contrasts must lie inside the family")
        if not 0 <= self.n_decisive <= self.n_comparisons:
            raise ValueError("decisive contrasts must lie inside the family")
        if not 0 < self.interval_level < 1:
            raise ValueError("interval_level must lie strictly between zero and one")
        if not 0 < self.family_error_rate < 1:
            raise ValueError("family_error_rate must lie strictly between zero and one")
        for value in (self.expected_separated, self.excess_probability, self.adjusted_threshold):
            if not np.isfinite(value) or value < 0:
                raise ValueError("comparison family rates must be finite and non-negative")

    @property
    def corrected(self) -> bool:
        """Whether an adjustment actually applies to this family."""

        return self.multiplicity is not ComparisonMultiplicity.NONE and self.n_comparisons > 1

    def to_dict(self) -> dict[str, Any]:
        """Return the family size and its chance yield as a JSON-safe record."""

        return {
            "n_candidates": self.n_candidates,
            "n_comparisons": self.n_comparisons,
            "interval_level": self.interval_level,
            "multiplicity": self.multiplicity.value,
            "family_error_rate": self.family_error_rate,
            "n_separated": self.n_separated,
            "expected_separated": self.expected_separated,
            "excess_probability": self.excess_probability,
            "adjusted_threshold": self.adjusted_threshold,
            "n_decisive": self.n_decisive,
        }


def adjust_probabilities(
    probabilities: NDArray[np.float64],
    *,
    multiplicity: str,
    error_rate: float,
    unadjusted_threshold: float,
) -> tuple[float, NDArray[np.float64]]:
    """Return the surviving raw-probability threshold and the adjusted probabilities.

    ``multiplicity`` is one of :data:`NONE`, :data:`BENJAMINI_HOCHBERG` or
    :data:`BONFERRONI`. Adjusted probabilities are on the same scale as the unadjusted ones,
    so a caller decides survival with a single ``adjusted <= error_rate`` and never needs to
    reproduce the step-up to interpret the threshold.

    A family of one is never adjusted: with nothing simultaneous about it there is no
    multiplicity to control, and adjusting anyway would silently move a single test's
    threshold. Its ``unadjusted_threshold`` is returned so the caller can report the
    criterion actually applied.
    """

    n_tests = len(probabilities)
    if n_tests == 0:
        return float(unadjusted_threshold), probabilities
    if n_tests == 1 or multiplicity == NONE:
        return float(unadjusted_threshold), probabilities
    if multiplicity == BONFERRONI:
        return float(error_rate / n_tests), np.minimum(1.0, probabilities * n_tests)
    if multiplicity != BENJAMINI_HOCHBERG:
        raise ValueError(f"unknown multiplicity adjustment: {multiplicity!r}")
    order = np.argsort(probabilities)
    ordered = probabilities[order]
    ranks = np.arange(1, n_tests + 1, dtype=np.float64)
    # Benjamini-Hochberg step-up. The adjusted value is the running minimum of
    # `p * n / rank` taken from the largest probability downwards, which keeps the adjusted
    # sequence monotone in the raw one; `adjusted <= error_rate` then selects exactly the
    # tests below the largest raw probability satisfying `p <= error_rate * rank / n`.
    monotone = np.minimum.accumulate((ordered * n_tests / ranks)[::-1])[::-1]
    adjusted = np.empty_like(monotone)
    adjusted[order] = np.minimum(1.0, monotone)
    surviving = ordered <= error_rate * ranks / n_tests
    threshold = float(ordered[surviving][-1]) if bool(np.any(surviving)) else 0.0
    return threshold, adjusted


def excess_probability(n_extreme: int, n_tests: int, per_test_rate: float) -> float:
    """Exact binomial probability of at least ``n_extreme`` extremes under no effect.

    Returns ``1.0`` when nothing was extreme, which is the correct tail rather than a
    sentinel: at least zero extremes is certain.

    Tests that share an item -- two contrasts against the same candidate, two discrepancies
    over the same group -- are not independent, so this is a guide to whether a pattern is
    remarkable rather than a test in its own right. Every caller says so in its own words.
    """

    if n_extreme <= 0 or n_tests <= 0:
        return 1.0
    return float(binom.sf(n_extreme - 1, n_tests, per_test_rate))
