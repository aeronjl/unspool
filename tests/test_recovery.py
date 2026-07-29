import json

import numpy as np
import pytest

from behavio import BernoulliHistoryGLM, Study, run_parameter_recovery


def recovery_design() -> Study:
    generator = np.random.default_rng(5)
    n_sessions = 3
    n_trials = 100
    return Study(
        {
            "subject": ["a"] * (n_sessions * n_trials),
            "session": [
                f"session-{session}" for session in range(n_sessions) for _ in range(n_trials)
            ],
            "trial": list(range(n_trials)) * n_sessions,
            "session_order": [session for session in range(n_sessions) for _ in range(n_trials)],
            "stimulus": generator.normal(size=n_sessions * n_trials),
        }
    )


def test_parameter_recovery_is_reproducible_and_design_specific() -> None:
    model = BernoulliHistoryGLM(covariates=("stimulus",), choice_lags=1, l2=0.05)
    parameter_sets = [
        {"intercept": -0.2, "stimulus": 0.6, "choice_lag_1": 0.2},
        {"intercept": 0.0, "stimulus": 1.0, "choice_lag_1": 0.5},
        {"intercept": 0.2, "stimulus": 1.4, "choice_lag_1": 0.8},
    ]

    first = run_parameter_recovery(model, recovery_design(), parameter_sets, repeats=2, seed=123)
    second = run_parameter_recovery(model, recovery_design(), parameter_sets, repeats=2, seed=123)

    assert first.n_runs == 6
    assert first.n_trials == 300
    assert first.n_subjects == 1
    assert first.repeats == 2
    assert first.root_seed == 123
    assert np.array_equal(first.seeds, second.seeds)
    assert np.array_equal(first.estimates, second.estimates)
    assert first.convergence_rate == 1.0
    assert first.audit_pass_rate == 1.0
    assert first.audit_warning_rate == 0.0
    assert first.audit_failure_rate == 0.0
    assert len(first.audits) == first.n_runs
    assert len(first.summary()) == 3
    assert all(summary.n_successful == 6 for summary in first.summary())
    assert all(summary.n_with_uncertainty == 6 for summary in first.summary())
    assert all(np.isfinite(summary.rmse) for summary in first.summary())
    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        first.estimates.setflags(write=True)
    payload = first.to_dict()
    assert payload["runs"][0]["fit_audit"]["status"] == "pass"
    json.dumps(payload, allow_nan=False)


def test_parameter_recovery_does_not_report_correlation_for_constant_truth() -> None:
    model = BernoulliHistoryGLM(choice_lags=0)
    report = run_parameter_recovery(
        model,
        recovery_design(),
        [{"intercept": 0.2}],
        repeats=3,
        seed=14,
    )

    assert np.isnan(report.summary()[0].correlation)
    assert report.to_dict()["summary"][0]["correlation"] is None


def test_parameter_recovery_keeps_failed_audits_out_of_summaries() -> None:
    model = BernoulliHistoryGLM(
        covariates=("stimulus",),
        choice_lags=1,
        max_iterations=1,
    )
    report = run_parameter_recovery(
        model,
        recovery_design(),
        [{"intercept": -0.2, "stimulus": 1.0, "choice_lag_1": 0.4}],
        seed=19,
    )

    assert report.audit_failure_rate == 1.0
    assert report.audits[0].issue_codes[0] == "optimizer_nonconvergence"
    assert all(summary.n_successful == 0 for summary in report.summary())
    json.dumps(report.to_dict(), allow_nan=False)


@pytest.mark.parametrize(
    ("parameter_sets", "repeats", "seed", "message"),
    [
        ([], 1, 0, "must not be empty"),
        ([{"intercept": 0.0}], 0, 0, "repeats"),
        ([{"intercept": 0.0}], 1, -1, "seed"),
        ([{"intercept": np.nan}], 1, 0, "finite numeric"),
        ([{"intercept": 0.0, "extra": 1.0}], 1, 0, "match the model exactly"),
    ],
)
def test_parameter_recovery_arguments_are_validated(
    parameter_sets: list[dict[str, float]], repeats: int, seed: int, message: str
) -> None:
    model = BernoulliHistoryGLM(choice_lags=0)

    with pytest.raises(ValueError, match=message):
        run_parameter_recovery(
            model,
            recovery_design(),
            parameter_sets,
            repeats=repeats,
            seed=seed,
        )


def test_a_model_with_no_reparameterisation_reports_exactly_one_coordinate() -> None:
    """A model declaring nothing is byte-identical to before the natural coordinate."""

    model = BernoulliHistoryGLM(covariates=("stimulus",), choice_lags=1, l2=0.05)
    report = run_parameter_recovery(
        model,
        recovery_design(),
        [{"intercept": -0.2, "stimulus": 0.6, "choice_lag_1": 0.2}],
        repeats=2,
        seed=7,
    )

    assert report.natural_names == ()
    assert report.natural_true_values is None
    assert report.natural_estimates is None
    assert report.natural_summary() == ()
    payload = report.to_dict()
    assert "natural_summary" not in payload
    assert "coordinate" not in payload
    assert "natural" not in payload["runs"][0]
    assert set(payload["runs"][0]) == {
        "seed",
        "truth",
        "estimate",
        "standard_error",
        "converged",
        "message",
        "fit_audit",
    }


def test_recovery_reports_both_coordinates_without_ever_pooling_them() -> None:
    from behavio.models import PsychometricFunction
    from behavio.recovery import ESTIMATED_COORDINATE, NATURAL_COORDINATE, WALD_INTERVAL

    model = PsychometricFunction(stimulus="stimulus", outcome="choice", n_restarts=2)
    truth = model.parameters_from_components(
        threshold=0.2, width=0.6, guess_rate=0.05, lapse_rate=0.05
    )
    report = run_parameter_recovery(model, recovery_design(), [dict(truth)], repeats=3, seed=11)

    estimated = {summary.parameter: summary for summary in report.summary()}
    natural = {summary.parameter: summary for summary in report.natural_summary()}

    # The estimated coordinate is where coverage is computed, and it is unreadable.
    assert set(estimated) == {"threshold", "log_width", "guess_logit", "lapse_logit"}
    assert all(summary.interval_kind == WALD_INTERVAL for summary in estimated.values())
    # The natural coordinate is readable and carries no coverage field at all.
    assert set(natural) == {"threshold", "width", "guess_rate", "lapse_rate"}
    assert all(summary.coordinate == NATURAL_COORDINATE for summary in natural.values())
    assert not hasattr(natural["width"], "coverage_95")
    assert natural["width"].n_successful == estimated["log_width"].n_successful

    # ``width`` and ``log_width`` are different numbers about the same parameter and are
    # never mixed: one is the exponential of the other, run for run.
    assert report.natural_estimates is not None
    log_width_column = report.parameter_names.index("log_width")
    width_column = report.natural_names.index("width")
    assert np.allclose(
        report.natural_estimates[:, width_column],
        np.exp(report.estimates[:, log_width_column]),
    )

    payload = report.to_dict()
    assert payload["coordinate"] == ESTIMATED_COORDINATE
    assert [row["parameter"] for row in payload["natural_summary"]] == list(report.natural_names)
    assert all("coverage_95" not in row for row in payload["natural_summary"])
    assert set(payload["runs"][0]["natural"]) == {"truth", "estimate"}
    json.dumps(payload, allow_nan=False)


def test_a_failed_run_has_no_natural_image_and_is_excluded_from_both_summaries() -> None:
    from behavio.models import PsychometricFunction
    from behavio.recovery import _natural_coordinate

    model = PsychometricFunction(stimulus="stimulus", outcome="choice", n_restarts=1)
    true_values = np.asarray([[0.2, np.log(0.6), 0.0, 0.0], [np.nan] * 4])
    estimates = np.asarray([[0.2, np.log(0.6), 0.0, 0.0], [np.nan] * 4])

    names, truth, natural = _natural_coordinate(model, true_values, estimates)

    assert names == ("threshold", "width", "guess_rate", "lapse_rate")
    assert truth is not None and natural is not None
    assert np.all(np.isfinite(truth[0])) and np.all(np.isnan(truth[1]))
    assert np.all(np.isnan(natural[1]))
