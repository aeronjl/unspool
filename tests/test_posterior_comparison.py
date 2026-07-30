import json
from dataclasses import replace

import numpy as np
import pytest
from scipy.stats import norm

from behavio.compare.models import ComparisonFamily, ComparisonMultiplicity
from behavio.contracts.audit import AuditSeverity
from behavio.posterior.comparison import (
    ModelComparisonStatus,
    PairedELPDDifference,
    PosteriorModelComparison,
    compare_posterior_models,
)
from behavio.posterior.diagnostics import PosteriorAuditStatus
from behavio.posterior.loo import PSISLOOIssue, PSISLOOResult, psis_loo
from behavio.posterior.result import PosteriorError
from tests.test_posterior_loo import N_SUBJECTS, N_TRIALS, hierarchical_result


def _gate() -> None:
    pytest.importorskip("arviz")


def test_paired_difference_uses_the_pointwise_difference_not_the_marginal_errors() -> None:
    _gate()
    hierarchical = psis_loo(hierarchical_result())
    pooled = psis_loo(hierarchical_result(pooled=True))

    comparison = compare_posterior_models({"hierarchical": hierarchical, "pooled": pooled})

    assert isinstance(comparison, PosteriorModelComparison)
    assert comparison.model_names[0] == "hierarchical"
    difference = comparison.difference("hierarchical", "pooled")
    pointwise = np.asarray(hierarchical.pointwise_elpd) - np.asarray(pooled.pointwise_elpd)
    assert difference.elpd_difference == pytest.approx(float(np.sum(pointwise)))
    assert difference.se == pytest.approx(float(np.sqrt(pointwise.size * np.var(pointwise))))
    # The paired error is far tighter than either naive combination of the marginal errors.
    naive = float(np.hypot(hierarchical.se, pooled.se))
    assert difference.se < naive
    assert comparison.status is ModelComparisonStatus.RESOLVED
    assert comparison.best_model == "hierarchical"
    json.dumps(comparison.to_dict(), allow_nan=False)


def test_reversed_difference_negates_the_interval() -> None:
    _gate()
    comparison = compare_posterior_models(
        {
            "hierarchical": psis_loo(hierarchical_result()),
            "pooled": psis_loo(hierarchical_result(pooled=True)),
        }
    )

    forward = comparison.difference("hierarchical", "pooled")
    reverse = comparison.difference("pooled", "hierarchical")

    assert reverse.elpd_difference == pytest.approx(-forward.elpd_difference)
    assert reverse.lower == pytest.approx(-forward.upper)
    assert reverse.upper == pytest.approx(-forward.lower)
    assert reverse.excludes_zero == forward.excludes_zero


def test_comparison_accepts_posteriors_and_scores_them_identically() -> None:
    _gate()
    left = hierarchical_result()
    right = hierarchical_result(pooled=True)

    from_posteriors = compare_posterior_models(
        {"hierarchical": left, "pooled": right},
        block="trial_subject",
    )
    from_scores = compare_posterior_models(
        {
            "hierarchical": psis_loo(left, block="trial_subject"),
            "pooled": psis_loo(right, block="trial_subject"),
        }
    )

    assert from_posteriors.block == "trial_subject"
    assert from_posteriors.estimand == "leave-one-trial_subject-out"
    assert from_posteriors.n_data_points == N_SUBJECTS
    assert from_posteriors.to_dict() == from_scores.to_dict()
    assert "comparison.few-observations" in from_posteriors.issue_codes


def test_comparison_refuses_mismatched_observation_coordinates() -> None:
    _gate()
    full = psis_loo(hierarchical_result())
    subjects = psis_loo(hierarchical_result(), block="trial_subject")
    animals = psis_loo(
        hierarchical_result(),
        block="animal",
        block_values=np.repeat(np.arange(N_SUBJECTS), N_TRIALS),
    )

    with pytest.raises(PosteriorError, match="different estimands"):
        compare_posterior_models({"trials": full, "subjects": subjects})
    with pytest.raises(PosteriorError, match="different estimands"):
        compare_posterior_models({"animals": animals, "subjects": subjects})
    with pytest.raises(PosteriorError, match="scores dimensions"):
        compare_posterior_models({"flat": full, "nested": _two_dimensional_score(full)})
    with pytest.raises(PosteriorError, match="at least two models"):
        compare_posterior_models({"only": full})


def _two_dimensional_score(reference: PSISLOOResult) -> PSISLOOResult:
    """A hand-built result whose pointwise unit spans two labelled dimensions."""

    shape = (N_SUBJECTS, N_TRIALS)
    return PSISLOOResult(
        model_name=reference.model_name,
        model_signature=reference.model_signature,
        inference_library=reference.inference_library,
        inference_library_version=reference.inference_library_version,
        log_likelihood_name=reference.log_likelihood_name,
        dims=("subject", "trial"),
        coords={"subject": np.arange(N_SUBJECTS), "trial": np.arange(N_TRIALS)},
        elpd_loo=reference.elpd_loo,
        se=reference.se,
        p_loo=reference.p_loo,
        n_samples=reference.n_samples,
        n_data_points=reference.n_data_points,
        good_k=reference.good_k,
        pointwise_elpd=np.asarray(reference.pointwise_elpd).reshape(shape),
        pareto_k=np.asarray(reference.pareto_k).reshape(shape),
    )


def test_comparison_refuses_a_different_number_or_order_of_observations() -> None:
    _gate()
    result = hierarchical_result()
    full = psis_loo(result)
    reversed_labels = np.repeat(np.arange(N_SUBJECTS)[::-1], N_TRIALS)
    reordered = psis_loo(result, block="animal", block_values=reversed_labels)
    forward_labels = np.repeat(np.arange(N_SUBJECTS), N_TRIALS)
    forward = psis_loo(result, block="animal", block_values=forward_labels)
    half = psis_loo(
        result,
        block="animal",
        block_values=np.repeat(np.arange(N_SUBJECTS // 2), N_TRIALS * 2),
    )

    with pytest.raises(PosteriorError, match="scores 3 observations"):
        compare_posterior_models({"forward": forward, "half": half})
    with pytest.raises(PosteriorError, match="identical 'animal' coordinates"):
        compare_posterior_models({"forward": forward, "reordered": reordered})
    assert full.dims == ("trial",)


def test_comparison_surfaces_a_failed_posterior_and_refuses_to_rank_it() -> None:
    _gate()
    healthy = psis_loo(hierarchical_result())
    broken = psis_loo(hierarchical_result(pooled=True, diverging=True))

    comparison = compare_posterior_models({"healthy": healthy, "broken": broken})

    statuses = {model.name: model.status for model in comparison.models}
    assert statuses["broken"] is PosteriorAuditStatus.FAIL
    assert comparison.eligible_models == ("healthy",)
    assert "comparison.posterior-fail" in comparison.issue_codes
    failure = next(
        issue for issue in comparison.issues if issue.code == "comparison.posterior-fail"
    )
    assert failure.severity is AuditSeverity.ERROR
    assert failure.targets == ("broken",)
    # The failed model's ELPD is still reported; it is simply not ranked against.
    assert comparison.status is ModelComparisonStatus.RESOLVED
    assert comparison.best_model == "healthy"
    assert "only model with a usable posterior" in comparison.reason
    assert any(model.name == "broken" for model in comparison.models)


def test_comparison_declares_no_winner_when_every_posterior_fails() -> None:
    _gate()
    left = psis_loo(hierarchical_result(diverging=True))
    right = psis_loo(hierarchical_result(pooled=True, diverging=True))

    comparison = compare_posterior_models({"left": left, "right": right})

    assert comparison.status is ModelComparisonStatus.NO_ELIGIBLE_MODEL
    assert comparison.best_model is None
    assert comparison.eligible_models == ()


def test_comparison_invents_no_winner_when_the_interval_includes_zero() -> None:
    _gate()
    result = hierarchical_result()
    twin = psis_loo(result)
    other = psis_loo(hierarchical_result())

    comparison = compare_posterior_models({"a": twin, "b": other})

    difference = comparison.difference("a", "b")
    assert difference.elpd_difference == pytest.approx(0.0, abs=1e-9)
    assert not difference.excludes_zero
    assert comparison.status is ModelComparisonStatus.UNRESOLVED
    assert comparison.best_model is None
    assert "exclude equal predictive performance" in comparison.reason
    # A single contrast is never adjusted, so the refusal here is the interval's alone and
    # the family record says so rather than crediting a correction that did not run.
    assert comparison.family.n_comparisons == 1
    assert not comparison.family.corrected
    assert not difference.decisive
    assert difference.favours is None


def test_paired_difference_rejects_a_degenerate_pair() -> None:
    with pytest.raises(ValueError, match="two distinct named models"):
        PairedELPDDifference(
            left_model="a",
            right_model="a",
            elpd_difference=0.0,
            se=1.0,
            lower=-2.0,
            upper=2.0,
            interval_scale=2.0,
            n_data_points=10,
        )
    with pytest.raises(ValueError, match="lower bound must not exceed"):
        PairedELPDDifference(
            left_model="a",
            right_model="b",
            elpd_difference=0.0,
            se=1.0,
            lower=2.0,
            upper=-2.0,
            interval_scale=2.0,
            n_data_points=10,
        )


#: Ten-per-cent-of-a-standard-error above the two-SE line. ``MARGINAL_SHIFT / MARGINAL_SPREAD``
#: is chosen so that ``sqrt(n) * mean / sd`` lands at 2.05: both contrasts against the leader
#: put their whole interval on one side of zero, and both invert to a two-sided normal
#: probability of about 0.041. The Benjamini-Hochberg step-up over ``(0.041, 0.041, 1.0)``
#: accepts none of them, because the third contrast -- between two rivals that predict almost
#: identically -- sinks the whole family. The numbers are fixed rather than sampled so the
#: boundary cannot drift.
MARGINAL_OBSERVATIONS = 40
MARGINAL_SHIFT = 0.324
MARGINAL_SPREAD = 1.6892


def marginal_scores() -> dict[str, PSISLOOResult]:
    index = np.arange(MARGINAL_OBSERVATIONS)
    baseline = -0.5 - 0.001 * index
    spread = np.linspace(-MARGINAL_SPREAD, MARGINAL_SPREAD, MARGINAL_OBSERVATIONS)
    advantage = MARGINAL_SHIFT + spread
    twin_noise = np.where(index % 2 == 0, 0.02, -0.02)
    pointwise = {
        "best": baseline,
        "rival_a": baseline - advantage,
        "rival_b": baseline - advantage + twin_noise,
    }
    return {name: _scored(name, values) for name, values in pointwise.items()}


def _scored(name: str, pointwise: np.ndarray) -> PSISLOOResult:
    return PSISLOOResult(
        model_name=name,
        model_signature=f"{name}[prescribed]",
        inference_library="test",
        inference_library_version="0",
        log_likelihood_name="choice",
        dims=("observation",),
        coords={"observation": [f"o{position}" for position in range(len(pointwise))]},
        elpd_loo=float(np.sum(pointwise)),
        se=float(np.sqrt(len(pointwise) * np.var(pointwise))),
        p_loo=1.0,
        n_samples=4000,
        n_data_points=len(pointwise),
        good_k=0.7,
        pointwise_elpd=pointwise,
        pareto_k=np.full(len(pointwise), 0.2),
    )


def test_the_declared_multiplicity_can_refuse_a_winner_the_intervals_alone_would_name() -> None:
    """Same ELPD evidence, same intervals; only the family adjustment differs."""

    scores = marginal_scores()

    uncorrected = compare_posterior_models(scores, multiplicity=ComparisonMultiplicity.NONE)
    corrected = compare_posterior_models(scores)

    leading = [("best", "rival_a"), ("best", "rival_b")]
    # Every interval is identical either way: correction reads the numbers, it does not
    # change them.
    for pair in (*leading, ("rival_a", "rival_b")):
        assert uncorrected.difference(*pair).lower == corrected.difference(*pair).lower
        assert uncorrected.difference(*pair).upper == corrected.difference(*pair).upper
    assert all(uncorrected.difference(*pair).excludes_zero for pair in leading)
    assert not uncorrected.difference("rival_a", "rival_b").excludes_zero
    assert uncorrected.family.n_separated == corrected.family.n_separated == 2

    assert uncorrected.status is ModelComparisonStatus.RESOLVED
    assert uncorrected.best_model == "best"
    assert uncorrected.family.n_decisive == 2
    assert all(uncorrected.difference(*pair).favours == "best" for pair in leading)
    assert "uncorrected per-contrast reading" in uncorrected.reason

    assert corrected.family.multiplicity is ComparisonMultiplicity.BENJAMINI_HOCHBERG
    assert corrected.status is ModelComparisonStatus.UNRESOLVED
    assert corrected.best_model is None
    assert corrected.family.n_decisive == 0
    assert corrected.decisive_differences == ()
    assert all(corrected.difference(*pair).favours is None for pair in leading)
    assert "benjamini-hochberg adjustment" in corrected.reason


def test_the_elpd_family_is_the_one_a_frequentist_comparison_reports() -> None:
    """Same record, same statistics; the standard error underneath is not shared."""

    comparison = compare_posterior_models(marginal_scores())
    family = comparison.family

    assert isinstance(family, ComparisonFamily)
    assert family.n_candidates == 3
    assert family.n_comparisons == 3
    # The per-contrast rate is the normal tail mass outside the declared multiplier, so a
    # 2-SE report is read at 0.0455 rather than at a threshold invented for the occasion.
    assert family.family_error_rate == pytest.approx(2.0 * norm.sf(2.0))
    assert family.interval_level == pytest.approx(1.0 - 2.0 * norm.sf(2.0))
    assert family.expected_separated == pytest.approx(3.0 * family.family_error_rate)
    # Interval and probability are the same normal statement, so they never disagree --
    # which is exactly what a percentile bootstrap over finite resamples cannot promise.
    for item in comparison.differences:
        assert item.excludes_zero == (item.two_sided_probability < family.family_error_rate)
    assert json.dumps(comparison.to_dict(), allow_nan=False)


def test_a_contrast_touching_an_unrankable_model_is_reported_but_never_tested() -> None:
    scores = marginal_scores()
    scores["rival_b"] = replace(
        scores["rival_b"],
        issues=(
            PSISLOOIssue(
                code="psis.unconverged-posterior",
                message="the posterior failed its convergence audit",
                severity=AuditSeverity.ERROR,
            ),
        ),
    )

    comparison = compare_posterior_models(scores)

    assert comparison.eligible_models == ("best", "rival_a")
    assert comparison.family.n_candidates == 2
    assert comparison.family.n_comparisons == 1
    for pair in (("best", "rival_b"), ("rival_a", "rival_b")):
        untested = comparison.difference(*pair)
        assert untested.two_sided_probability is None
        assert untested.adjusted_probability is None
        assert not untested.decisive
    # With one contrast left there is nothing simultaneous to correct, so the surviving
    # interval decides on its own -- and it does exclude zero.
    tested = comparison.difference("best", "rival_a")
    assert tested.decisive
    assert not comparison.family.corrected
    assert comparison.best_model == "best"
