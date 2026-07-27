import json

import numpy as np
import pytest

from unspool import (
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
