"""Signal detection theory: known answers, declared corrections, and the model contract.

The equal-variance checks reproduce the worked example in Macmillan and Creelman (2005),
where a hit rate of .67 against a false-alarm rate of .16 gives d' = 1.43 and c = 0.28. The
forced-choice checks reproduce the published m-alternative table (2AFC 0.954, 3AFC 1.43,
4AFC 1.68 at 75 % correct). The meta-d' check simulates a metacognitively ideal observer
directly from continuous evidence -- not from the estimator's own generative model -- and
requires the recovered M-ratio to be one.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import ndtr

from behavio import Study, model_capabilities, run_parameter_recovery
from behavio.contracts import (
    BehaviourModel,
    CategoricalBehaviourEstimator,
    ModelDataError,
    PredictionMode,
    UnsupportedPredictionMode,
)
from behavio.models import (
    CorrectedRates,
    DetectionCounts,
    EqualVarianceSDT,
    MetaSDT,
    RateCorrection,
    RatingCounts,
    SignalDetectionFitResult,
    UnequalVarianceSDT,
    detection_rates,
    equal_variance_summary,
    forced_choice_d_prime,
    forced_choice_proportion_correct,
    roc_points,
    z_roc_summary,
)

#: Macmillan and Creelman (2005), chapter 1: H = .67, F = .16.
TEXTBOOK = DetectionCounts(hits=67, misses=33, false_alarms=16, correct_rejections=84)


def detection_design(n_trials: int = 1_200, *, seed: int = 4) -> Study:
    generator = np.random.default_rng(seed)
    return Study(
        {
            "subject": ["a"] * n_trials,
            "session": ["s"] * n_trials,
            "trial": list(range(n_trials)),
            "session_order": [0] * n_trials,
            "signal": generator.integers(0, 2, n_trials),
        }
    )


def ideal_observer(
    n_trials: int,
    *,
    d_prime: float = 1.5,
    criterion: float = 0.0,
    type2: tuple[float, ...] = (0.5, 1.0, 1.5),
    confidence_noise: float = 0.0,
    seed: int = 11,
) -> Study:
    """Simulate confidence directly from continuous evidence, independently of the model.

    A metacognitively ideal observer rates confidence by the distance of the *same*
    evidence sample from the decision criterion, so meta-d' must equal type-1 d'. Adding
    ``confidence_noise`` corrupts only the confidence read-out, which must lower meta-d'
    without touching type-1 performance.
    """

    generator = np.random.default_rng(seed)
    signal = generator.integers(0, 2, n_trials)
    evidence = generator.normal(np.where(signal == 1, d_prime / 2, -d_prime / 2), 1.0)
    response = (evidence > criterion).astype(np.int8)
    read_out = evidence
    if confidence_noise > 0:
        read_out = evidence + generator.normal(0.0, confidence_noise, n_trials)
    distance = np.abs(read_out - criterion)
    confidence = np.ones(n_trials, dtype=np.int64)
    for boundary in type2:
        confidence += distance > boundary
    return Study(
        {
            "subject": ["a"] * n_trials,
            "session": ["s"] * n_trials,
            "trial": list(range(n_trials)),
            "session_order": [0] * n_trials,
            "signal": signal,
            "response": response,
            "confidence": confidence,
        }
    )


def test_equal_variance_indices_reproduce_the_textbook_worked_example() -> None:
    summary = equal_variance_summary(TEXTBOOK)

    assert summary.rates.hit_rate == pytest.approx(0.67)
    assert summary.rates.false_alarm_rate == pytest.approx(0.16)
    assert summary.d_prime == pytest.approx(1.43, abs=5e-3)
    assert summary.criterion == pytest.approx(0.28, abs=5e-3)
    assert summary.beta == pytest.approx(1.49, abs=5e-3)
    assert summary.log_beta == pytest.approx(summary.criterion * summary.d_prime)
    assert summary.log10_beta == pytest.approx(summary.log_beta / np.log(10.0))
    assert summary.relative_criterion == pytest.approx(summary.criterion / summary.d_prime)
    assert summary.a_prime == pytest.approx(0.8421, abs=1e-4)
    assert summary.b_double_prime_d == pytest.approx(0.2439, abs=1e-4)
    assert not summary.is_degenerate


def test_criterion_sign_convention_is_conservative_positive() -> None:
    """A high hit rate paired with a high false-alarm rate is a liberal, negative c."""

    liberal = equal_variance_summary(
        DetectionCounts(hits=90, misses=10, false_alarms=40, correct_rejections=60)
    )
    conservative = equal_variance_summary(
        DetectionCounts(hits=60, misses=40, false_alarms=10, correct_rejections=90)
    )

    assert liberal.criterion < 0 < conservative.criterion
    assert liberal.d_prime == pytest.approx(conservative.d_prime)
    assert liberal.beta < 1 < conservative.beta


def test_corrections_disagree_and_never_pass_as_uncorrected() -> None:
    extreme = DetectionCounts(hits=20, misses=0, false_alarms=0, correct_rejections=20)
    ordinary = DetectionCounts(hits=15, misses=5, false_alarms=5, correct_rejections=15)

    uncorrected = detection_rates(extreme)
    log_linear = detection_rates(extreme, correction=RateCorrection.LOG_LINEAR)
    one_over_2n = detection_rates(extreme, correction=RateCorrection.ONE_OVER_2N)

    assert uncorrected.correction is RateCorrection.NONE
    assert uncorrected.correction_applied is False
    assert uncorrected.is_degenerate
    assert np.isposinf(equal_variance_summary(extreme).d_prime)

    assert log_linear.correction_applied is True
    assert one_over_2n.correction_applied is True
    assert log_linear.hit_rate == pytest.approx(20.5 / 21.0)
    assert one_over_2n.hit_rate == pytest.approx(1.0 - 1.0 / 40.0)
    assert log_linear.hit_rate != one_over_2n.hit_rate
    assert not log_linear.is_degenerate and not one_over_2n.is_degenerate

    # The two corrections differ on tables that need no repair at all.
    assert detection_rates(ordinary, correction=RateCorrection.ONE_OVER_2N) == CorrectedRates(
        hit_rate=0.75,
        false_alarm_rate=0.25,
        correction=RateCorrection.ONE_OVER_2N,
        correction_applied=False,
        n_signal=20,
        n_noise=20,
    )
    assert detection_rates(ordinary, correction=RateCorrection.LOG_LINEAR).hit_rate != 0.75
    assert detection_rates(ordinary, correction=RateCorrection.LOG_LINEAR).correction_applied


def test_extreme_rates_force_an_explicit_correction_choice_in_the_estimator() -> None:
    study = Study(
        {
            "subject": ["a"] * 40,
            "session": ["s"] * 40,
            "trial": list(range(40)),
            "session_order": [0] * 40,
            "signal": [1] * 20 + [0] * 20,
            "response": [1] * 20 + [0] * 20,
        }
    )

    with pytest.raises(ModelDataError, match="LOG_LINEAR"):
        EqualVarianceSDT().fit(study)

    corrected = EqualVarianceSDT(correction=RateCorrection.LOG_LINEAR).fit(study)

    assert np.isfinite(corrected.parameters["d_prime"])
    assert corrected.summary.rates.correction is RateCorrection.LOG_LINEAR
    assert corrected.summary.rates.correction_applied is True
    assert corrected.diagnostics.boundary_estimate is True
    assert "correction=log-linear" in corrected.diagnostics.message


def test_equal_variance_estimator_satisfies_the_model_contract() -> None:
    model = EqualVarianceSDT()
    truth = {"d_prime": 1.5, "criterion": -0.25}
    study = model.simulate(detection_design(4_000), truth, seed=3)

    fit = model.fit(study)
    prediction = model.predict(study, fit)

    assert isinstance(model, BehaviourModel)
    assert model_capabilities(model).can_recover_parameters
    assert isinstance(fit, SignalDetectionFitResult)
    assert fit.model_name == model.model_name
    assert fit.parameters["d_prime"] == pytest.approx(1.5, abs=0.15)
    assert fit.parameters["criterion"] == pytest.approx(-0.25, abs=0.1)
    assert fit.standard_error_map["d_prime"] > 0
    assert fit.audit().model_name == model.model_name
    assert prediction.probability.shape == (len(study),)
    assert np.all(np.isfinite(model.pointwise_log_prob(study, fit)))
    assert model.summarize(study).d_prime == pytest.approx(fit.parameters["d_prime"])

    with pytest.raises(UnsupportedPredictionMode):
        model.predict(study, fit, mode=PredictionMode.SMOOTHED)


def test_equal_variance_recovers_its_parameters_under_a_declared_design() -> None:
    model = EqualVarianceSDT()
    truth = {"d_prime": 1.2, "criterion": 0.2}

    report = run_parameter_recovery(model, detection_design(3_000), [truth], repeats=3, seed=8)

    assert report.model_name == model.model_name
    assert np.all(report.converged)
    assert report.estimates[:, 0].mean() == pytest.approx(1.2, abs=0.1)
    assert report.estimates[:, 1].mean() == pytest.approx(0.2, abs=0.1)


def test_two_alternative_forced_choice_uses_the_exact_root_two_relation() -> None:
    assert forced_choice_d_prime(0.75, n_alternatives=2) == pytest.approx(
        float(np.sqrt(2.0) * 0.6744897501960817)
    )
    assert forced_choice_d_prime(0.75, n_alternatives=2) == pytest.approx(0.9539, abs=1e-4)
    assert forced_choice_proportion_correct(0.9539, n_alternatives=2) == pytest.approx(
        0.75, abs=1e-4
    )
    assert forced_choice_d_prime(0.5, n_alternatives=2) == pytest.approx(0.0)


def test_the_forced_choice_integral_agrees_with_the_shortcut_only_at_two_alternatives() -> None:
    from behavio.models.sdt import _forced_choice_integral

    for d_prime in (0.0, 0.5, 1.5, 3.0):
        assert _forced_choice_integral(d_prime, 2) == pytest.approx(
            forced_choice_proportion_correct(d_prime, n_alternatives=2), abs=1e-9
        )

    # Published m-alternative table at 75 % correct: 2AFC 0.954, 3AFC 1.43, 4AFC 1.68.
    assert forced_choice_d_prime(0.75, n_alternatives=3) == pytest.approx(1.4338, abs=1e-3)
    assert forced_choice_d_prime(0.75, n_alternatives=4) == pytest.approx(1.6822, abs=1e-3)
    assert forced_choice_d_prime(0.75, n_alternatives=4) > forced_choice_d_prime(
        0.75, n_alternatives=2
    )
    for alternatives in (3, 4, 8):
        recovered = forced_choice_d_prime(0.6, n_alternatives=alternatives)
        assert forced_choice_proportion_correct(
            recovered, n_alternatives=alternatives
        ) == pytest.approx(0.6, abs=1e-8)


@pytest.mark.parametrize(
    ("arguments", "match"),
    [
        ({"proportion_correct": 0.0, "n_alternatives": 2}, "strictly between"),
        ({"proportion_correct": 1.0, "n_alternatives": 3}, "strictly between"),
        ({"proportion_correct": 0.5, "n_alternatives": 1}, "at least two"),
    ],
)
def test_forced_choice_rejects_impossible_arguments(arguments: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        forced_choice_d_prime(**arguments)


def test_roc_points_run_from_the_origin_to_the_far_corner() -> None:
    counts = RatingCounts(signal=(5, 10, 20, 30, 40, 50), noise=(50, 40, 30, 20, 10, 5))

    false_alarms, hits = roc_points(counts)

    assert false_alarms[0] == 0.0 and hits[0] == 0.0
    assert false_alarms[-1] == pytest.approx(1.0) and hits[-1] == pytest.approx(1.0)
    assert np.all(np.diff(false_alarms) >= 0) and np.all(np.diff(hits) >= 0)
    assert np.all(hits >= false_alarms)


def test_z_roc_summary_recovers_an_equal_variance_generator() -> None:
    generator = np.random.default_rng(5)
    criteria = np.asarray([-1.2, -0.6, 0.0, 0.6, 1.2])
    noise = generator.normal(0.0, 1.0, 40_000)
    signal = generator.normal(1.2, 1.0, 40_000)
    counts = RatingCounts(
        noise=tuple(np.bincount(np.searchsorted(criteria, noise), minlength=6).tolist()),
        signal=tuple(np.bincount(np.searchsorted(criteria, signal), minlength=6).tolist()),
    )

    summary = z_roc_summary(counts)

    assert summary.n_points == 5
    assert summary.slope == pytest.approx(1.0, abs=0.05)
    assert summary.signal_sd == pytest.approx(1.0, abs=0.06)
    assert summary.d_a == pytest.approx(1.2, abs=0.05)
    assert summary.area_under_curve == pytest.approx(float(ndtr(1.2 / np.sqrt(2.0))), abs=0.01)
    # The trapezoid over six rating levels is a chord approximation to a concave ROC, so it
    # is always the smaller of the two. Reporting both keeps that gap visible.
    assert summary.empirical_area < summary.area_under_curve
    assert summary.empirical_area == pytest.approx(summary.area_under_curve, abs=0.03)
    assert summary.correction is RateCorrection.NONE
    assert summary.correction_applied is False


def test_z_roc_summary_needs_usable_operating_points() -> None:
    counts = RatingCounts(signal=(0, 0, 100), noise=(100, 0, 0))

    with pytest.raises(ModelDataError, match="non-degenerate operating points"):
        z_roc_summary(counts)

    repaired = z_roc_summary(counts, correction=RateCorrection.LOG_LINEAR)

    assert repaired.correction_applied is True
    assert np.isfinite(repaired.d_a)


def test_unequal_variance_estimator_recovers_a_wider_signal_distribution() -> None:
    model = UnequalVarianceSDT(ratings=(1, 2, 3, 4, 5, 6))
    truth = model.parameters_from_components(
        signal_mean=1.2, signal_sd=1.4, criteria=[-1.0, -0.4, 0.1, 0.7, 1.4]
    )
    design = Study(
        {
            "subject": ["a"] * 6_000,
            "session": ["s"] * 6_000,
            "trial": list(range(6_000)),
            "session_order": [0] * 6_000,
            "signal": np.tile([0, 1], 3_000),
        }
    )
    study = model.simulate(design, truth, seed=6)

    fit = model.fit(study)
    prediction = model.predict(study, fit)

    assert isinstance(model, BehaviourModel)
    assert isinstance(model, CategoricalBehaviourEstimator)
    assert fit.signal_mean == pytest.approx(1.2, abs=0.15)
    assert fit.signal_sd == pytest.approx(1.4, abs=0.2)
    assert fit.z_roc_slope == pytest.approx(1.0 / fit.signal_sd)
    assert fit.d_a == pytest.approx(
        np.sqrt(2.0) * fit.signal_mean / np.sqrt(1.0 + fit.signal_sd**2)
    )
    assert list(fit.criteria) == sorted(fit.criteria)
    assert prediction.categories == model.ratings
    assert prediction.probability.shape == (len(study), 6)
    assert np.allclose(prediction.probability.sum(axis=1), 1.0)
    assert np.all(np.isfinite(model.pointwise_log_prob(study, fit)))


def test_unequal_variance_maximum_likelihood_and_z_roc_regression_broadly_agree() -> None:
    """Two different estimators of the same quantities: close, but not identical."""

    model = UnequalVarianceSDT(ratings=(1, 2, 3, 4, 5, 6))
    truth = model.parameters_from_components(
        signal_mean=1.1, signal_sd=1.3, criteria=[-1.1, -0.5, 0.0, 0.6, 1.3]
    )
    design = Study(
        {
            "subject": ["a"] * 8_000,
            "session": ["s"] * 8_000,
            "trial": list(range(8_000)),
            "session_order": [0] * 8_000,
            "signal": np.tile([0, 1], 4_000),
        }
    )
    study = model.simulate(design, truth, seed=12)

    fit = model.fit(study)
    regression = z_roc_summary(model.counts(study))

    assert regression.d_a == pytest.approx(fit.d_a, abs=0.1)
    assert regression.slope == pytest.approx(fit.z_roc_slope, abs=0.1)
    assert regression.area_under_curve == pytest.approx(fit.area_under_curve, abs=0.02)


def test_unequal_variance_participates_in_parameter_recovery() -> None:
    model = UnequalVarianceSDT(ratings=(1, 2, 3, 4))
    truth = dict(
        model.parameters_from_components(signal_mean=1.2, signal_sd=1.2, criteria=[-0.8, 0.0, 0.9])
    )
    design = Study(
        {
            "subject": ["a"] * 3_000,
            "session": ["s"] * 3_000,
            "trial": list(range(3_000)),
            "session_order": [0] * 3_000,
            "signal": np.tile([0, 1], 1_500),
        }
    )

    report = run_parameter_recovery(model, design, [truth], repeats=2, seed=21)

    assert np.all(report.converged)
    assert report.estimates[:, 0].mean() == pytest.approx(1.2, abs=0.15)


def test_meta_d_prime_of_an_ideal_observer_is_the_type_one_d_prime() -> None:
    """The check that separates a correct meta-d' from an incorrect one."""

    study = ideal_observer(60_000, d_prime=1.5, seed=11)
    model = MetaSDT()

    fit = model.fit(study)

    assert fit.type1_d_prime == pytest.approx(1.5, abs=0.05)
    assert fit.meta_d_prime == pytest.approx(fit.type1_d_prime, abs=0.05)
    assert fit.m_ratio == pytest.approx(1.0, abs=0.03)
    assert fit.m_diff == pytest.approx(0.0, abs=0.05)
    assert np.allclose(fit.type2_criteria_yes, [0.5, 1.0, 1.5], atol=0.05)
    assert np.allclose(fit.type2_criteria_no, [-0.5, -1.0, -1.5], atol=0.05)


def test_noisy_confidence_lowers_meta_d_prime_without_touching_type_one() -> None:
    ideal = MetaSDT().fit(ideal_observer(60_000, seed=11))
    noisy = MetaSDT().fit(ideal_observer(60_000, confidence_noise=0.9, seed=11))

    assert noisy.type1_d_prime == pytest.approx(ideal.type1_d_prime)
    assert noisy.meta_d_prime < 0.7 * noisy.type1_d_prime
    assert noisy.m_ratio < 0.7
    assert noisy.m_diff < 0


def test_meta_d_prime_holds_the_type_one_fit_and_its_relative_criterion_fixed() -> None:
    study = ideal_observer(40_000, criterion=0.4, seed=17)
    model = MetaSDT()

    fit = model.fit(study)
    closed_form = equal_variance_summary(model.type1_counts(study))

    assert fit.type1_d_prime == pytest.approx(closed_form.d_prime)
    assert fit.type1_criterion == pytest.approx(closed_form.criterion)
    assert fit.parameters["d_prime"] == pytest.approx(closed_form.d_prime)
    assert fit.relative_criterion == pytest.approx(closed_form.criterion / closed_form.d_prime)
    # The criterion the meta model uses keeps c' rather than c: c_meta / meta-d' == c / d'.
    meta_criterion = fit.meta_d_prime * fit.relative_criterion
    assert meta_criterion / fit.meta_d_prime == pytest.approx(fit.relative_criterion)
    assert fit.m_ratio == pytest.approx(1.0, abs=0.05)


def test_meta_d_prime_type_two_criteria_stay_ordered_and_on_their_own_side() -> None:
    fit = MetaSDT().fit(ideal_observer(20_000, criterion=-0.3, seed=31))

    assert all(value < 0 for value in fit.type2_criteria_no)
    assert all(value > 0 for value in fit.type2_criteria_yes)
    assert list(fit.type2_criteria_no) == sorted(fit.type2_criteria_no, reverse=True)
    assert list(fit.type2_criteria_yes) == sorted(fit.type2_criteria_yes)


def test_meta_d_prime_scores_the_joint_response_and_confidence_cell() -> None:
    study = ideal_observer(8_000, seed=5)
    model = MetaSDT()
    fit = model.fit(study)

    prediction = model.predict(study, fit)
    scores = model.pointwise_log_prob(study, fit)

    assert isinstance(model, CategoricalBehaviourEstimator)
    assert model.scored_columns == ("response", "confidence")
    assert prediction.categories == (
        "no-4",
        "no-3",
        "no-2",
        "no-1",
        "yes-1",
        "yes-2",
        "yes-3",
        "yes-4",
    )
    assert np.allclose(prediction.probability.sum(axis=1), 1.0)
    codes = model.outcome_codes(study)
    assert np.allclose(scores, np.log(prediction.probability[np.arange(len(codes)), codes]))
    assert model_capabilities(model).scored_columns == ("response", "confidence")


def test_meta_d_prime_simulation_reproduces_its_own_estimates() -> None:
    model = MetaSDT()
    truth = dict(
        model.parameters_from_components(
            d_prime=1.6,
            criterion=0.15,
            meta_d_prime=1.1,
            type2_criteria_no=[-0.4, -0.9, -1.5],
            type2_criteria_yes=[0.5, 1.0, 1.7],
        )
    )
    design = detection_design(30_000, seed=2)

    study = model.simulate(design, truth, seed=13)
    fit = model.fit(study)

    assert isinstance(model, BehaviourModel)
    assert fit.type1_d_prime == pytest.approx(1.6, abs=0.06)
    assert fit.meta_d_prime == pytest.approx(1.1, abs=0.08)
    assert fit.m_ratio == pytest.approx(1.1 / 1.6, abs=0.06)
    assert set(study.columns) >= {"signal", "response", "confidence"}


def test_meta_d_prime_participates_in_parameter_recovery() -> None:
    model = MetaSDT(confidence_levels=(1, 2, 3))
    truth = dict(
        model.parameters_from_components(
            d_prime=1.5,
            criterion=0.1,
            meta_d_prime=1.2,
            type2_criteria_no=[-0.6, -1.3],
            type2_criteria_yes=[0.6, 1.3],
        )
    )

    report = run_parameter_recovery(model, detection_design(6_000), [truth], repeats=2, seed=7)

    assert np.all(report.converged)
    assert report.parameter_names[:3] == ("d_prime", "criterion", "meta_d_prime")
    assert report.estimates[:, 2].mean() == pytest.approx(1.2, abs=0.15)


def test_meta_d_prime_requires_a_positive_finite_type_one_sensitivity() -> None:
    study = ideal_observer(2_000, d_prime=0.0, seed=3)
    inverted = Study(
        {
            **{name: study[name] for name in study.columns},
            "response": 1 - np.asarray(study["response"]),
        }
    )

    with pytest.raises(ModelDataError, match="positive type-1"):
        MetaSDT().fit(inverted)


@pytest.mark.parametrize(
    ("arguments", "match"),
    [
        ({"signal": "signal", "response": "signal"}, "distinct"),
        ({"signal": "subject"}, "required Study column"),
        ({"correction": "not-a-correction"}, "not a valid RateCorrection"),
    ],
)
def test_equal_variance_configuration_is_validated(arguments: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        EqualVarianceSDT(**arguments)


def test_detection_counts_and_rating_counts_validate_their_tables() -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        DetectionCounts(hits=-1, misses=1, false_alarms=1, correct_rejections=1)
    with pytest.raises(ValueError, match="at least one signal"):
        DetectionCounts(hits=0, misses=0, false_alarms=1, correct_rejections=1)
    with pytest.raises(ValueError, match="at least two rating levels"):
        RatingCounts(signal=(5,), noise=(5,))
    with pytest.raises(ValueError, match="at least three distinct rating levels"):
        UnequalVarianceSDT(ratings=(1, 2))


def test_a_fit_cannot_be_read_under_a_different_specification() -> None:
    model = EqualVarianceSDT()
    other = EqualVarianceSDT(correction=RateCorrection.LOG_LINEAR)
    study = model.simulate(detection_design(600), {"d_prime": 1.0, "criterion": 0.0}, seed=1)
    fit = model.fit(study)

    with pytest.raises(ValueError, match="different model specification"):
        other.predict(study, fit)
