import json

import numpy as np
import pytest

from behavio import (
    PosteriorGroup,
    PosteriorResult,
    PosteriorVariable,
    ReliabilityError,
    ReliabilityPolicy,
    ReliabilityStatistic,
    SubjectEstimates,
    assess_test_retest_reliability,
    posterior_subject_estimates,
)
from behavio.reliability import SubjectPooling

SUBJECTS = tuple(f"s{index:02d}" for index in range(12))
CHAINS = 2
DRAWS = 400


def occasion_result(
    *,
    seed: int,
    posterior_sd: float = 0.6,
    unconverged: bool = False,
) -> PosteriorResult:
    """One occasion's hierarchical fit: per-subject means plus real posterior width."""

    generator = np.random.default_rng(seed)
    truth = np.linspace(-1.0, 1.0, len(SUBJECTS))
    centre = truth + generator.normal(0.0, 0.3, size=len(SUBJECTS))
    values = centre + generator.normal(0.0, posterior_sd, size=(CHAINS, DRAWS, len(SUBJECTS)))
    diverging = np.zeros((CHAINS, DRAWS), dtype=bool)
    if unconverged:
        values[0] += 12.0
        diverging[0, :19] = True
    coords = {
        "chain": np.arange(CHAINS),
        "draw": np.arange(DRAWS),
        "subject": np.asarray(SUBJECTS),
    }
    posterior = PosteriorGroup(
        "posterior",
        (PosteriorVariable("beta", values, ("chain", "draw", "subject"), coords),),
    )
    sample_stats = PosteriorGroup(
        "sample_stats",
        (
            PosteriorVariable(
                "diverging",
                diverging,
                ("chain", "draw"),
                {"chain": coords["chain"], "draw": coords["draw"]},
            ),
        ),
    )
    return PosteriorResult(
        model_name="hierarchical-beta",
        model_signature="hierarchical-beta[v1]",
        inference_library="test",
        inference_library_version="1",
        parameter_names=("beta",),
        groups=(posterior, sample_stats),
    )


def without_draws(estimates: SubjectEstimates) -> SubjectEstimates:
    """The same point estimates with their posterior thrown away, as before the fix."""

    return SubjectEstimates(
        occasion=estimates.occasion,
        target=estimates.target,
        target_signature=estimates.target_signature,
        unit=estimates.unit,
        subjects=estimates.subjects,
        values=estimates.values,
        artifact_signature=estimates.artifact_signature,
    )


def shifted_estimates() -> tuple[SubjectEstimates, SubjectEstimates]:
    subjects = ("a", "b", "c", "d", "e")
    first = SubjectEstimates(
        occasion="test",
        target="learning_rate",
        target_signature="learning-rate[v1]",
        unit="probability",
        subjects=subjects,
        values=[1, 2, 3, 4, 5],
        artifact_signature="test-fit[v1]",
    )
    second = SubjectEstimates(
        occasion="retest",
        target="learning_rate",
        target_signature="learning-rate[v1]",
        unit="probability",
        subjects=tuple(reversed(subjects)),
        values=[15, 14, 13, 12, 11],
        artifact_signature="retest-fit[v1]",
    )
    return first, second


def test_reliability_separates_consistency_from_absolute_agreement() -> None:
    first, second = shifted_estimates()
    policy = ReliabilityPolicy(
        bootstrap_repeats=300,
        interval_probability=0.9,
        minimum_subjects_warning=6,
    )

    report = assess_test_retest_reliability(
        first,
        second,
        seed=81,
        analysis_signature="constant-shift-reliability[v1]",
        policy=policy,
    )

    assert report.subjects == ("a", "b", "c", "d", "e")
    np.testing.assert_allclose(report.second_values, [11, 12, 13, 14, 15])
    assert report[ReliabilityStatistic.PEARSON].estimate == pytest.approx(1.0)
    assert report[ReliabilityStatistic.SPEARMAN].estimate == pytest.approx(1.0)
    assert report[ReliabilityStatistic.ICC_CONSISTENCY].estimate == pytest.approx(1.0)
    assert report[ReliabilityStatistic.ICC_ABSOLUTE_AGREEMENT].estimate == pytest.approx(5 / 105)
    assert report[ReliabilityStatistic.MEAN_DIFFERENCE].estimate == pytest.approx(10.0)
    assert report[ReliabilityStatistic.SD_DIFFERENCE].estimate == pytest.approx(0.0)
    assert report[ReliabilityStatistic.LOWER_LIMIT_OF_AGREEMENT].estimate == pytest.approx(10.0)
    assert report[ReliabilityStatistic.UPPER_LIMIT_OF_AGREEMENT].estimate == pytest.approx(10.0)
    assert report[ReliabilityStatistic.MEAN_ABSOLUTE_ERROR].estimate == pytest.approx(10.0)
    assert report[ReliabilityStatistic.ROOT_MEAN_SQUARED_ERROR].estimate == pytest.approx(10.0)
    assert "reliability.small-sample" in report.issue_codes
    assert all(item.bootstrap_repeats == 300 for item in report.estimates)
    json.dumps(report.to_dict(), allow_nan=False)


def test_paired_bootstrap_is_reproducible_and_retained() -> None:
    first, _ = shifted_estimates()
    second = SubjectEstimates(
        occasion="retest",
        target=first.target,
        target_signature=first.target_signature,
        unit=first.unit,
        subjects=first.subjects,
        values=[1.1, 2.3, 2.7, 4.4, 4.8],
        artifact_signature="retest-fit[noisy]",
    )
    policy = ReliabilityPolicy(bootstrap_repeats=200, minimum_subjects_warning=3)
    kwargs = {
        "first": first,
        "second": second,
        "seed": 17,
        "analysis_signature": "noisy-reliability[v1]",
        "policy": policy,
    }

    first_report = assess_test_retest_reliability(**kwargs)
    second_report = assess_test_retest_reliability(**kwargs)

    assert first_report.to_dict() == second_report.to_dict()
    pearson = first_report[ReliabilityStatistic.PEARSON]
    assert pearson.interval is not None
    assert pearson.effective_bootstrap_repeats > 0
    assert pearson.bootstrap_values.flags.writeable is False
    assert (
        first_report.to_dict(include_bootstrap=False)["estimates"][0].get("bootstrap_values")
        is None
    )


def test_undefined_statistics_and_bootstrap_failures_remain_visible() -> None:
    first = SubjectEstimates(
        "test",
        "bias",
        "bias[v1]",
        "log-odds",
        ("a", "b", "c", "d"),
        [1, 1, 1, 1],
        "test-fit",
    )
    second = SubjectEstimates(
        "retest",
        "bias",
        "bias[v1]",
        "log-odds",
        ("a", "b", "c", "d"),
        [1, 1, 1, 1],
        "retest-fit",
    )

    report = assess_test_retest_reliability(
        first,
        second,
        seed=3,
        analysis_signature="constant-values[v1]",
        policy=ReliabilityPolicy(bootstrap_repeats=40, minimum_subjects_warning=3),
    )

    assert report[ReliabilityStatistic.PEARSON].estimate is None
    assert report[ReliabilityStatistic.SPEARMAN].estimate is None
    assert report[ReliabilityStatistic.ICC_CONSISTENCY].estimate is None
    assert report[ReliabilityStatistic.ICC_ABSOLUTE_AGREEMENT].estimate is None
    assert report[ReliabilityStatistic.MEAN_DIFFERENCE].estimate == 0.0
    assert report[ReliabilityStatistic.PEARSON].invalid_bootstrap_repeats == 40
    assert "reliability.undefined" in report.issue_codes
    assert "reliability.bootstrap-effective" in report.issue_codes
    json.dumps(report.to_dict(), allow_nan=False)


def test_no_bootstrap_is_explicit_and_does_not_fabricate_intervals() -> None:
    first, second = shifted_estimates()
    report = assess_test_retest_reliability(
        first,
        second,
        seed=5,
        analysis_signature="point-estimates-only[v1]",
        policy=ReliabilityPolicy(bootstrap_repeats=0, minimum_subjects_warning=3),
    )

    assert all(item.interval is None for item in report.estimates)
    assert all(item.bootstrap_repeats == 0 for item in report.estimates)
    assert "reliability.bootstrap-effective" not in report.issue_codes


def test_posterior_subject_estimates_selects_and_labels_one_target() -> None:
    values = np.arange(2 * 3 * 3 * 2, dtype=float).reshape(2, 3, 3, 2)
    variable = PosteriorVariable(
        "coefficient",
        values,
        ("chain", "draw", "subject", "term"),
        {
            "chain": [0, 1],
            "draw": [0, 1, 2],
            "subject": ["a", "b", "c"],
            "term": ["bias", "history"],
        },
    )
    posterior = PosteriorResult(
        model_name="hierarchical-history",
        model_signature="hierarchical-history[v1]",
        inference_library="test",
        inference_library_version="1",
        parameter_names=("coefficient",),
        groups=(PosteriorGroup("posterior", (variable,)),),
    )

    estimates = posterior_subject_estimates(
        posterior,
        "coefficient",
        occasion="week-1",
        coordinate={"term": "history"},
    )

    assert estimates.target == "coefficient[term='history']"
    assert estimates.target_signature.endswith("coordinate=term='history']")
    assert estimates.subjects == ("a", "b", "c")
    np.testing.assert_allclose(estimates.values, np.mean(values[:, :, :, 1], axis=(0, 1)))
    assert estimates.values.flags.writeable is False


def test_reliability_requires_exact_subject_and_target_pairing() -> None:
    first, second = shifted_estimates()
    missing = SubjectEstimates(
        occasion="retest",
        target=second.target,
        target_signature=second.target_signature,
        unit=second.unit,
        subjects=("a", "b", "c"),
        values=(11, 12, 13),
        artifact_signature="incomplete",
    )
    with pytest.raises(ReliabilityError, match="exactly the same subjects"):
        assess_test_retest_reliability(
            first,
            missing,
            seed=1,
            analysis_signature="invalid-pair[v1]",
        )

    incompatible = SubjectEstimates(
        occasion="retest",
        target="inverse_temperature",
        target_signature="inverse-temperature[v1]",
        unit=second.unit,
        subjects=second.subjects,
        values=second.values,
        artifact_signature="incompatible",
    )
    with pytest.raises(ReliabilityError, match="same target and unit"):
        assess_test_retest_reliability(
            first,
            incompatible,
            seed=1,
            analysis_signature="invalid-target[v1]",
        )


def test_propagating_posterior_draws_widens_every_reported_interval() -> None:
    pytest.importorskip("arviz")
    first = posterior_subject_estimates(occasion_result(seed=101), "beta", occasion="week-1")
    second = posterior_subject_estimates(occasion_result(seed=202), "beta", occasion="week-3")
    policy = ReliabilityPolicy(bootstrap_repeats=600, minimum_subjects_warning=3)
    kwargs = {"seed": 7, "analysis_signature": "posterior-reliability[v1]", "policy": policy}

    propagated = assess_test_retest_reliability(first, second, **kwargs)
    collapsed = assess_test_retest_reliability(
        without_draws(first), without_draws(second), **kwargs
    )

    assert propagated.posterior_uncertainty_propagated
    assert not collapsed.posterior_uncertainty_propagated
    for statistic in (
        ReliabilityStatistic.PEARSON,
        ReliabilityStatistic.SPEARMAN,
        ReliabilityStatistic.ICC_CONSISTENCY,
        ReliabilityStatistic.ICC_ABSOLUTE_AGREEMENT,
    ):
        wide = propagated[statistic].interval
        narrow = collapsed[statistic].interval
        assert wide is not None and narrow is not None
        assert wide[1] - wide[0] > narrow[1] - narrow[0]
    json.dumps(propagated.to_dict(), allow_nan=False)


def test_posterior_reliability_reports_the_posterior_mean_of_the_statistic() -> None:
    pytest.importorskip("arviz")
    first = posterior_subject_estimates(occasion_result(seed=101), "beta", occasion="week-1")
    second = posterior_subject_estimates(occasion_result(seed=202), "beta", occasion="week-3")
    policy = ReliabilityPolicy(bootstrap_repeats=600, minimum_subjects_warning=3)
    kwargs = {"seed": 7, "analysis_signature": "posterior-reliability[v1]", "policy": policy}

    propagated = assess_test_retest_reliability(first, second, **kwargs)
    collapsed = assess_test_retest_reliability(
        without_draws(first), without_draws(second), **kwargs
    )

    on_means = collapsed[ReliabilityStatistic.PEARSON].estimate
    posterior_mean = propagated[ReliabilityStatistic.PEARSON].estimate
    assert on_means is not None and posterior_mean is not None
    assert posterior_mean < on_means
    assert propagated.to_dict()["uncertainty_sources"] == [
        "subject-resampling",
        "posterior-draws",
    ]


def test_point_estimate_inputs_keep_the_original_subject_bootstrap_exactly() -> None:
    first, second = shifted_estimates()
    policy = ReliabilityPolicy(bootstrap_repeats=300, minimum_subjects_warning=3)

    report = assess_test_retest_reliability(
        first,
        second,
        seed=81,
        analysis_signature="constant-shift-reliability[v1]",
        policy=policy,
    )

    assert not report.posterior_uncertainty_propagated
    assert report.to_dict()["uncertainty_sources"] == ["subject-resampling"]
    assert report[ReliabilityStatistic.PEARSON].estimate == pytest.approx(1.0)
    assert report[ReliabilityStatistic.ICC_ABSOLUTE_AGREEMENT].estimate == pytest.approx(5 / 105)
    assert report[ReliabilityStatistic.MEAN_DIFFERENCE].estimate == pytest.approx(10.0)
    assert "reliability.shrunken-estimates" not in report.issue_codes
    assert "reliability.draws-not-propagated" not in report.issue_codes


def test_partial_pooling_is_declared_and_warns_that_reliability_is_inflated() -> None:
    pytest.importorskip("arviz")
    policy = ReliabilityPolicy(bootstrap_repeats=100, minimum_subjects_warning=3)
    pooled = assess_test_retest_reliability(
        posterior_subject_estimates(occasion_result(seed=101), "beta", occasion="week-1"),
        posterior_subject_estimates(occasion_result(seed=202), "beta", occasion="week-3"),
        seed=3,
        analysis_signature="pooled[v1]",
        policy=policy,
    )
    independent = assess_test_retest_reliability(
        posterior_subject_estimates(
            occasion_result(seed=101),
            "beta",
            occasion="week-1",
            pooling=SubjectPooling.NONE,
        ),
        posterior_subject_estimates(
            occasion_result(seed=202),
            "beta",
            occasion="week-3",
            pooling=SubjectPooling.NONE,
        ),
        seed=3,
        analysis_signature="independent[v1]",
        policy=policy,
    )

    assert "reliability.shrunken-estimates" in pooled.issue_codes
    shrinkage = next(
        item for item in pooled.issues if item.code == "reliability.shrunken-estimates"
    )
    assert shrinkage.statistics == (
        ReliabilityStatistic.PEARSON,
        ReliabilityStatistic.SPEARMAN,
        ReliabilityStatistic.ICC_CONSISTENCY,
        ReliabilityStatistic.ICC_ABSOLUTE_AGREEMENT,
    )
    assert "upper bounds" in shrinkage.message
    assert "reliability.shrunken-estimates" not in independent.issue_codes
    assert pooled.to_dict()["first"]["pooling"] == "partial"
    assert independent.to_dict()["first"]["pooling"] == "none"


def test_partial_pooling_is_the_default_for_posterior_subject_estimates() -> None:
    pytest.importorskip("arviz")
    estimates = posterior_subject_estimates(occasion_result(seed=101), "beta", occasion="week-1")

    assert estimates.pooling is SubjectPooling.PARTIAL
    assert estimates.n_draws == CHAINS * DRAWS
    assert estimates.draws is not None
    assert estimates.draws.shape == (CHAINS * DRAWS, len(SUBJECTS))
    assert estimates.draws.flags.writeable is False
    np.testing.assert_allclose(estimates.values, np.mean(estimates.draws, axis=0))


def test_a_failed_convergence_audit_is_carried_into_the_reliability_report() -> None:
    pytest.importorskip("arviz")
    first = posterior_subject_estimates(
        occasion_result(seed=101, unconverged=True), "beta", occasion="week-1"
    )
    second = posterior_subject_estimates(occasion_result(seed=202), "beta", occasion="week-3")

    assert first.posterior_audit is not None
    assert first.posterior_audit.status.value == "fail"
    assert first.provenance["posterior_audit_status"] == "fail"

    report = assess_test_retest_reliability(
        first,
        second,
        seed=5,
        analysis_signature="unconverged[v1]",
        policy=ReliabilityPolicy(bootstrap_repeats=50, minimum_subjects_warning=3),
    )

    assert "reliability.unconverged-posterior" in report.issue_codes
    failing = next(
        item for item in report.issues if item.code == "reliability.unconverged-posterior"
    )
    assert "week-1" in failing.message
    assert report.to_dict()["first"]["posterior_audit"]["status"] == "fail"
    json.dumps(report.to_dict(), allow_nan=False)


def test_available_draws_that_cannot_be_propagated_are_reported() -> None:
    pytest.importorskip("arviz")
    first = posterior_subject_estimates(occasion_result(seed=101), "beta", occasion="week-1")
    second = without_draws(
        posterior_subject_estimates(occasion_result(seed=202), "beta", occasion="week-3")
    )

    report = assess_test_retest_reliability(
        first,
        second,
        seed=5,
        analysis_signature="half-propagated[v1]",
        policy=ReliabilityPolicy(bootstrap_repeats=50, minimum_subjects_warning=3),
    )

    assert not report.posterior_uncertainty_propagated
    assert "reliability.draws-not-propagated" in report.issue_codes


def test_subject_estimate_draws_must_align_with_their_subjects() -> None:
    with pytest.raises(ValueError, match="sample-by-subject"):
        SubjectEstimates(
            occasion="test",
            target="beta",
            target_signature="beta[v1]",
            unit="log-odds",
            subjects=("a", "b", "c"),
            values=[1.0, 2.0, 3.0],
            artifact_signature="fit[v1]",
            draws=np.zeros((10, 2)),
        )
    with pytest.raises(ValueError, match="draws must be finite"):
        SubjectEstimates(
            occasion="test",
            target="beta",
            target_signature="beta[v1]",
            unit="log-odds",
            subjects=("a", "b", "c"),
            values=[1.0, 2.0, 3.0],
            artifact_signature="fit[v1]",
            draws=np.full((10, 3), np.nan),
        )
