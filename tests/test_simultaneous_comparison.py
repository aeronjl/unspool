"""Multiplicity control across the family of pairwise candidate contrasts.

``_rank`` used to name a winner when every pairwise bootstrap interval upper bound fell
below zero. Each interval is an uncorrected 95% percentile bootstrap, so at K candidates
there are K(K-1)/2 simultaneous readings of the same threshold and the leader has that
many chances to clear it. This module holds the correction in place and, most importantly,
holds the case it exists for: a family in which the uncorrected rule names a winner and
the corrected one refuses.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import binom
from test_compiler import capabilities
from test_execution_parity import parity_compiled, parity_protocol

from behavio.compare.models import (
    ComparisonFamily,
    ComparisonMultiplicity,
    paired_comparisons,
)
from behavio.models import BernoulliHistoryGLM
from behavio.protocol.runner import run_protocol
from behavio.protocol.schema import CandidateSpec, ScoreMetric, Setting, WinnerPolicy
from behavio.report.bounded import ReportItem, ReportItemKind, generate_bounded_report

UNITS = tuple(f"subject-{index:02d}" for index in range(24))


def unit_scores(offsets: dict[str, float], *, noise: float = 0.08, seed: int = 3) -> dict:
    """Give each candidate a per-unit score around its own offset, paired across units.

    The same per-unit shocks are shared by every candidate, which is what makes the
    contrasts paired; the offsets are what the contrasts are trying to detect.
    """

    generator = np.random.default_rng(seed)
    shocks = generator.normal(scale=0.25, size=len(UNITS))
    return {
        name: {
            unit: float(offset + shock + wiggle)
            for unit, shock, wiggle in zip(
                UNITS,
                shocks,
                generator.normal(scale=noise, size=len(UNITS)),
                strict=True,
            )
        }
        for name, offset in offsets.items()
    }


def compare(scores, multiplicity=ComparisonMultiplicity.BENJAMINI_HOCHBERG):
    return paired_comparisons(
        scores,
        metric=ScoreMetric.LOG_LOSS,
        resamples=2_000,
        seed=5,
        confidence_level=0.95,
        multiplicity=multiplicity,
    )


def competitors(comparisons, leader: str) -> set[str]:
    return {
        name
        for comparison in comparisons
        for name in (comparison.left_model, comparison.right_model)
    } - {leader}


def leader_beats_everyone(comparisons, leader: str) -> bool:
    """The corrected winner rule: the leader must decisively beat every competitor."""

    others = competitors(comparisons, leader)
    return all(
        any(
            comparison.favours == leader
            and leader in (comparison.left_model, comparison.right_model)
            and other in (comparison.left_model, comparison.right_model)
            for comparison in comparisons
        )
        for other in others
    )


def uncorrected_leader_beats_everyone(comparisons, leader: str) -> bool:
    """The rule this change replaced: every raw interval on the leader's side of zero."""

    for other in competitors(comparisons, leader):
        pair = next(
            comparison
            for comparison in comparisons
            if {comparison.left_model, comparison.right_model} == {leader, other}
        )
        interval = pair.left_minus_right
        excluded = interval.upper < 0 if pair.left_model == leader else interval.lower > 0
        if not excluded:
            return False
    return True


def test_the_family_is_reported_even_when_nothing_separates() -> None:
    scores = unit_scores(dict.fromkeys(("a", "b", "c", "d"), 0.5))

    comparisons, family = compare(scores)

    assert isinstance(family, ComparisonFamily)
    assert family.n_candidates == 4
    assert family.n_comparisons == 6
    assert family.expected_separated == pytest.approx(6 * 0.05)
    assert family.n_decisive == 0
    assert all(not comparison.decisive for comparison in comparisons)
    assert family.multiplicity is ComparisonMultiplicity.BENJAMINI_HOCHBERG


def test_correction_refuses_a_winner_the_uncorrected_rule_would_have_named() -> None:
    """The case the whole change exists for.

    Five candidates, ten simultaneous contrasts. The leader's raw intervals all sit on its
    side of zero, so the old rule resolves and names it. The smallest of those contrasts is
    nowhere near the Benjamini-Hochberg threshold for its rank in a family of ten, so the
    corrected rule refuses -- which is what this package does when the evidence is thin.
    """

    scores = unit_scores(
        {"leader": 0.500, "rival": 0.545, "third": 0.560, "fourth": 0.575, "fifth": 0.590},
        noise=0.30,
        seed=11,
    )

    comparisons, family = compare(scores)
    uncorrected, _ = compare(scores, multiplicity=ComparisonMultiplicity.NONE)

    assert family.n_comparisons == 10
    assert uncorrected_leader_beats_everyone(comparisons, "leader")
    assert leader_beats_everyone(uncorrected, "leader")
    assert not leader_beats_everyone(comparisons, "leader")
    assert family.n_separated > family.n_decisive


def test_a_correction_can_only_remove_decisive_contrasts_never_add_one() -> None:
    scores = unit_scores(
        {"a": 0.50, "b": 0.62, "c": 0.70, "d": 0.55, "e": 0.66},
        noise=0.20,
        seed=29,
    )

    corrected, family = compare(scores)
    uncorrected, plain = compare(scores, multiplicity=ComparisonMultiplicity.NONE)
    bonferroni, strict = compare(scores, multiplicity=ComparisonMultiplicity.BONFERRONI)

    decisive = {(item.left_model, item.right_model) for item in corrected if item.decisive}
    raw = {(item.left_model, item.right_model) for item in uncorrected if item.decisive}
    harsh = {(item.left_model, item.right_model) for item in bonferroni if item.decisive}
    assert decisive <= raw
    assert harsh <= decisive
    assert strict.n_decisive <= family.n_decisive <= plain.n_decisive
    assert plain.n_decisive == plain.n_separated


def test_every_contrast_keeps_its_unadjusted_interval_and_probability() -> None:
    """Adjustment adds members; it must never rewrite the ones already published."""

    scores = unit_scores({"a": 0.50, "b": 0.66, "c": 0.72})

    corrected, _ = compare(scores)
    uncorrected, _ = compare(scores, multiplicity=ComparisonMultiplicity.NONE)

    for left, right in zip(corrected, uncorrected, strict=True):
        assert left.left_minus_right.to_dict() == right.left_minus_right.to_dict()
        assert left.bootstrap_probability_positive == right.bootstrap_probability_positive
        assert left.two_sided_probability == right.two_sided_probability
        assert left.adjusted_probability >= left.two_sided_probability


def test_a_single_contrast_is_never_adjusted() -> None:
    scores = unit_scores({"a": 0.50, "b": 0.80})

    corrected, family = compare(scores)
    uncorrected, plain = compare(scores, multiplicity=ComparisonMultiplicity.NONE)

    assert family.n_comparisons == 1
    assert corrected[0].adjusted_probability == corrected[0].two_sided_probability
    assert corrected[0].decisive == uncorrected[0].decisive
    assert family.n_decisive == plain.n_decisive


def test_an_unadjusted_family_reads_the_interval_and_not_the_inverted_probability() -> None:
    """A small resample pins the two-sided probability above the interval's own reading.

    With ``B`` draws the ``+1``-corrected two-sided probability cannot fall below
    ``2/(B+1)``, so at ``B = 16`` it is stuck at 0.118 while the 95% percentile interval
    can still exclude zero. A single contrast is decided by its interval alone, so it must
    stay decisive here; letting the shared adjustment machinery apply ``p <= 0.05`` would
    silently impose a second, stricter rule on an unadjusted family.
    """

    scores = unit_scores({"a": 0.50, "b": 0.95}, noise=0.02, seed=13)

    comparisons, family = paired_comparisons(
        scores,
        metric=ScoreMetric.LOG_LOSS,
        resamples=16,
        seed=4,
        confidence_level=0.95,
    )

    assert family.n_comparisons == 1
    assert comparisons[0].left_minus_right.excludes_zero
    assert comparisons[0].two_sided_probability == pytest.approx(2 / 17)
    assert comparisons[0].two_sided_probability > family.family_error_rate
    assert comparisons[0].decisive
    assert family.n_decisive == 1


def test_a_finite_resample_never_reports_an_impossible_zero_probability() -> None:
    scores = unit_scores({"a": 0.10, "b": 5.00})

    comparisons, _ = paired_comparisons(
        scores,
        metric=ScoreMetric.LOG_LOSS,
        resamples=200,
        seed=2,
        confidence_level=0.95,
    )

    assert comparisons[0].bootstrap_probability_positive == 0.0
    assert comparisons[0].two_sided_probability == pytest.approx(2 / 201)
    assert comparisons[0].two_sided_probability > 0


def test_the_excess_probability_is_the_exact_binomial_tail() -> None:
    scores = unit_scores(
        {"a": 0.50, "b": 0.90, "c": 1.30, "d": 1.70},
        noise=0.05,
        seed=7,
    )

    _, family = compare(scores)

    assert family.n_separated == 6
    assert family.excess_probability == pytest.approx(
        float(binom.sf(family.n_separated - 1, family.n_comparisons, 0.05))
    )


def test_pairs_with_no_shared_aggregation_unit_are_not_in_the_family() -> None:
    """A contrast that cannot be computed must not inflate the correction's denominator."""

    comparisons, family = paired_comparisons(
        {
            "a": {"x": 0.5, "y": 0.6},
            "b": {"x": 0.7, "y": 0.8},
            "c": {"z": 0.9},
        },
        metric=ScoreMetric.LOG_LOSS,
        resamples=100,
        seed=1,
        confidence_level=0.95,
    )

    assert len(comparisons) == 1
    assert family.n_candidates == 3
    assert family.n_comparisons == 1


def five_candidate_run():
    """A real ``run_protocol`` over five candidates that differ only in ridge penalty."""

    penalties = {"static": 0.01, "b": 0.5, "c": 2.0, "d": 8.0, "e": 32.0}
    candidates = tuple(
        CandidateSpec(
            name=name,
            implementation="behavio.models.BernoulliHistoryGLM",
            hyperparameters=(
                Setting("choice_lags", 0),
                Setting("predictors", ("stimulus",)),
                Setting("l2", penalty),
            ),
            scored_columns=("choice",),
        )
        for name, penalty in penalties.items()
    )
    protocol = parity_protocol(
        candidates,
        with_recovery=False,
        winner_policy=WinnerPolicy.INTERVAL_EXCLUDES_ZERO,
    )
    reference = capabilities()["static"]
    compiled, _, _ = parity_compiled(
        protocol, model_capabilities=dict.fromkeys(penalties, reference)
    )
    models = {
        name: BernoulliHistoryGLM(predictors=("stimulus",), choice_lags=0, l2=penalty)
        for name, penalty in penalties.items()
    }
    return run_protocol(compiled, models)


def test_the_report_shows_the_family_the_winner_rule_was_read_against() -> None:
    run = five_candidate_run()

    family = run.report.ranking.family
    assert family is not None
    assert family.n_candidates == 5
    assert family.n_comparisons == 10
    assert len(run.report.paired_comparisons) == 10
    assert family.expected_separated == pytest.approx(10 * 0.05)
    assert "10 simultaneous contrasts" in run.report.ranking.reason
    assert run.report.to_dict()["ranking"]["family"] == family.to_dict()


def test_the_family_size_survives_serialization_into_the_evidence_artifact() -> None:
    run = five_candidate_run()

    payload = run.report.to_dict(retain_predictions=False)

    assert payload["ranking"]["family"]["n_comparisons"] == 10
    assert payload["ranking"]["family"]["multiplicity"] == "benjamini-hochberg"
    for comparison in payload["paired_comparisons"]:
        assert set(comparison) >= {
            "two_sided_probability",
            "adjusted_probability",
            "decisive",
        }


def test_the_bounded_report_discloses_the_family_beside_the_decision() -> None:
    """A reader must be able to see how many contrasts produced the verdict."""

    run = five_candidate_run()
    reported = generate_bounded_report(
        run,
        items=tuple(
            ReportItem(name=name, kind=kind, summary="required by the frozen protocol")
            for kind, names in (
                (ReportItemKind.TABLE, run.protocol.reporting.required_tables),
                (ReportItemKind.FIGURE, run.protocol.reporting.required_figures),
                (ReportItemKind.DIAGNOSTIC, run.protocol.reporting.required_diagnostics),
            )
            for name in names
        ),
    )

    markdown = reported.report.markdown
    assert "10 contrasts were read simultaneously over 5 eligible candidates" in markdown
    assert "benjamini-hochberg adjustment" in markdown
    assert "Every interval above is unadjusted." in markdown
    assert "| Adjusted p | Decisive |" in markdown
