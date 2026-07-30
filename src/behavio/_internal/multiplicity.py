"""The one step-up multiplicity adjustment the package applies to a family of tests.

Behavio evaluates two families of simultaneous tests. A posterior-predictive audit
evaluates ``groups x discrepancies`` tail probabilities
(:mod:`behavio.posterior_predictive`); a model comparison evaluates ``K(K-1)/2`` pairwise
contrasts (:mod:`behavio.comparison`). The domains are unrelated and their records name
different things, but the arithmetic that turns many probabilities into a decision is the
same arithmetic, and it was written twice before this module existed. The two copies had
already begun to disagree.

The two callers keep their own family records and their own definition of which item counts
as *extreme* -- a tail probability below a threshold in one case, an interval excluding zero
in the other. What they share, and what lives here, is the step-up itself and the exact
binomial tail that says how many extremes chance alone supplies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import binom

#: Adjustment names, shared so the two public enums cannot drift apart in spelling.
NONE = "none"
BENJAMINI_HOCHBERG = "benjamini-hochberg"
BONFERRONI = "bonferroni"


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
    remarkable rather than a test in its own right. Both callers say so in their own words.
    """

    if n_extreme <= 0 or n_tests <= 0:
        return 1.0
    return float(binom.sf(n_extreme - 1, n_tests, per_test_rate))
