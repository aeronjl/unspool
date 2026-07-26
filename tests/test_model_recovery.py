import numpy as np
import pytest

from unspool import (
    BernoulliHistoryGLM,
    FitAuditStatus,
    ModelRecoveryScenario,
    SmoothBernoulliHistoryGLM,
    Study,
    run_model_recovery,
    run_model_recovery_grid,
)


def recovery_design(*, n_sessions: int = 10, n_trials: int = 120) -> Study:
    generator = np.random.default_rng(2027)
    n_rows = n_sessions * n_trials
    return Study(
        {
            "subject": ["a"] * n_rows,
            "session": [
                f"session-{session}" for session in range(n_sessions) for _ in range(n_trials)
            ],
            "trial": list(range(n_trials)) * n_sessions,
            "session_order": [session for session in range(n_sessions) for _ in range(n_trials)],
            "stimulus": generator.normal(size=n_rows),
        }
    )


def competing_models(n_sessions: int = 10):
    static = BernoulliHistoryGLM(covariates=("stimulus",), choice_lags=1, l2=0.01)
    smooth = SmoothBernoulliHistoryGLM(
        covariates=("stimulus",),
        choice_lags=1,
        knots=tuple(range(n_sessions)),
        smoothness=10.0,
        l2=0.01,
    )
    return static, smooth


def recovery_scenarios(
    static: BernoulliHistoryGLM,
    smooth: SmoothBernoulliHistoryGLM,
) -> tuple[ModelRecoveryScenario, ModelRecoveryScenario]:
    n_knots = len(smooth.knots)
    return (
        ModelRecoveryScenario(
            name="stationary",
            truth_label="static",
            generator=static,
            parameters={"intercept": -0.2, "stimulus": 1.2, "choice_lag_1": 0.4},
        ),
        ModelRecoveryScenario(
            name="drifting",
            truth_label="smooth",
            generator=smooth,
            parameters=smooth.parameters_from_paths(
                {
                    "intercept": np.linspace(-0.5, 0.5, n_knots),
                    "stimulus": np.linspace(0.2, 2.5, n_knots),
                    "choice_lag_1": np.linspace(0.8, 0.1, n_knots),
                }
            ),
        ),
    )


def test_model_recovery_builds_an_explicit_prospective_confusion_matrix() -> None:
    static, smooth = competing_models()
    report = run_model_recovery(
        recovery_design(),
        recovery_scenarios(static, smooth),
        {"static": static, "smooth": smooth},
        repeats=3,
        seed=12,
        min_train_sessions=3,
        tie_tolerance=0.001,
    )

    assert report.n_runs == 6
    assert report.n_trials == 1_200
    assert report.n_subjects == 1
    assert report.n_folds.tolist() == [7] * 6
    assert report.generator_parameters[0]["stimulus"] == 1.2
    with pytest.raises(TypeError):
        report.generator_parameters[0]["stimulus"] = 99.0
    assert report.truth_labels == ("static", "static", "static", "smooth", "smooth", "smooth")
    assert report.selected_labels == ("static", None, "static", "smooth", "smooth", "smooth")
    assert report.resolution_rate == pytest.approx(5 / 6)
    assert report.overall_accuracy == pytest.approx(5 / 6)
    assert report.resolved_accuracy == 1.0
    assert np.all(report.converged)
    assert all(status is FitAuditStatus.PASS for row in report.audit_statuses for status in row)
    assert all(not codes for row in report.audit_issue_codes for codes in row)
    assert report.audit_warning_rate == 0.0
    assert report.audit_failure_rate == 0.0

    matrix = report.confusion_matrix()
    assert matrix.truth_labels == ("static", "smooth")
    assert matrix.selected_labels == ("static", "smooth", "unresolved")
    assert matrix.counts.tolist() == [[2, 0, 1], [0, 3, 0]]
    assert np.allclose(matrix.rates, [[2 / 3, 0, 1 / 3], [0, 1, 0]])
    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        matrix.counts.setflags(write=True)

    scenario_matrix = report.scenario_confusion_matrix()
    assert scenario_matrix.scenario_names == ("stationary", "drifting")
    assert scenario_matrix.truth_labels == ("static", "smooth")
    assert scenario_matrix.selected_labels == ("static", "smooth", "unresolved")
    assert scenario_matrix.counts.tolist() == [[2, 0, 1], [0, 3, 0]]
    assert np.allclose(scenario_matrix.rates, [[2 / 3, 0, 1 / 3], [0, 1, 0]])


def test_exact_candidate_ties_are_unresolved_and_reproducible() -> None:
    model = BernoulliHistoryGLM(covariates=("stimulus",), choice_lags=1, l2=0.01)
    scenario = ModelRecoveryScenario(
        name="same-model-tie",
        truth_label="first",
        generator=model,
        parameters={"intercept": 0.0, "stimulus": 1.0, "choice_lag_1": 0.2},
    )
    arguments = {
        "design": recovery_design(n_sessions=4, n_trials=60),
        "scenarios": [scenario],
        "candidates": {"first": model, "second": model},
        "repeats": 2,
        "seed": 7,
        "min_train_sessions": 2,
    }

    first = run_model_recovery(**arguments)
    second = run_model_recovery(**arguments)

    assert first.selected_labels == (None, None)
    assert first.resolution_rate == 0.0
    assert np.array_equal(first.seeds, second.seeds)
    assert np.array_equal(first.mean_log_probabilities, second.mean_log_probabilities)


def test_nonconvergence_is_retained_and_excluded_from_selection() -> None:
    good = BernoulliHistoryGLM(covariates=("stimulus",), choice_lags=1, l2=0.01)
    limited = BernoulliHistoryGLM(
        covariates=("stimulus",),
        choice_lags=1,
        l2=0.01,
        max_iterations=1,
    )
    scenario = ModelRecoveryScenario(
        name="optimizer-control",
        truth_label="good",
        generator=good,
        parameters={"intercept": -0.2, "stimulus": 1.2, "choice_lag_1": 0.5},
    )

    report = run_model_recovery(
        recovery_design(n_sessions=4, n_trials=80),
        [scenario],
        {"good": good, "limited": limited},
        seed=1,
        min_train_sessions=2,
    )

    assert report.converged.tolist() == [[True, False]]
    assert report.audit_statuses == ((FitAuditStatus.PASS, FitAuditStatus.FAIL),)
    assert report.audit_issue_codes[0][0] == ()
    assert "optimizer_nonconvergence" in report.audit_issue_codes[0][1]
    assert report.audit_failure_rate == 0.5
    assert report.selected_labels == ("good",)
    assert report.failure_messages[0][0] == ""
    assert "ITERATIONS REACHED LIMIT" in report.failure_messages[0][1]


def test_audit_warnings_are_retained_without_disqualifying_a_candidate() -> None:
    warning_model = BernoulliHistoryGLM(
        covariates=("stimulus",),
        choice_lags=1,
        l2=0.01,
        coefficient_warning_threshold=0.01,
    )
    scenario = ModelRecoveryScenario(
        name="boundary-warning",
        truth_label="warning",
        generator=warning_model,
        parameters={"intercept": -0.2, "stimulus": 1.2, "choice_lag_1": 0.4},
    )

    report = run_model_recovery(
        recovery_design(n_sessions=4, n_trials=80),
        [scenario],
        {"warning": warning_model},
        seed=5,
        min_train_sessions=2,
    )

    assert report.selected_labels == ("warning",)
    assert report.audit_statuses == ((FitAuditStatus.WARNING,),)
    assert report.audit_issue_codes == ((("boundary_estimate",),),)
    assert report.audit_warning_rate == 1.0
    assert report.audit_failure_rate == 0.0


def test_recovery_grid_compares_named_designs_with_independent_seeds() -> None:
    model = BernoulliHistoryGLM(covariates=("stimulus",), choice_lags=1, l2=0.01)
    scenario = ModelRecoveryScenario(
        name="stationary",
        truth_label="static",
        generator=model,
        parameters={"intercept": -0.2, "stimulus": 1.2, "choice_lag_1": 0.4},
    )

    grid = run_model_recovery_grid(
        {
            "short": recovery_design(n_sessions=4, n_trials=40),
            "long": recovery_design(n_sessions=4, n_trials=80),
        },
        [scenario],
        {"static": model},
        seed=91,
        min_train_sessions=2,
    )

    assert grid.design_names == ("short", "long")
    assert len(set(grid.seeds.tolist())) == 2
    assert grid.report_for("short").n_trials == 160
    assert grid.report_for("long").n_trials == 320
    assert [row.design_name for row in grid.summary()] == ["short", "long"]
    assert [row.overall_accuracy for row in grid.summary()] == [1.0, 1.0]
    with pytest.raises(KeyError, match="unknown recovery-grid design"):
        grid.report_for("missing")


def test_scenarios_require_exact_generator_parameters() -> None:
    model = BernoulliHistoryGLM(choice_lags=0)

    with pytest.raises(ValueError, match="match the generator exactly"):
        ModelRecoveryScenario(
            name="invalid",
            truth_label="static",
            generator=model,
            parameters={},
        )


def test_model_recovery_requires_eligible_folds_and_known_truth() -> None:
    model = BernoulliHistoryGLM(choice_lags=0)
    scenario = ModelRecoveryScenario(
        name="one-session",
        truth_label="static",
        generator=model,
        parameters={"intercept": 0.0},
    )
    design = Study(
        {
            "subject": ["a", "a"],
            "session": ["only", "only"],
            "trial": [0, 1],
            "session_order": [0, 0],
        }
    )

    with pytest.raises(ValueError, match="truth_label must name a candidate"):
        run_model_recovery(design, [scenario], {"different": model}, seed=1)

    with pytest.raises(ValueError, match="no eligible prospective folds"):
        run_model_recovery(design, [scenario], {"static": model}, seed=1)
