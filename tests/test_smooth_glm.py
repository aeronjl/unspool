from types import MappingProxyType

import numpy as np
import pytest

from unspool import (
    BernoulliHistoryGLM,
    ModelDataError,
    SmoothBernoulliHistoryGLM,
    Study,
    evaluate_splits,
    forward_session_splits,
    run_parameter_recovery,
)


def make_design(
    *, n_sessions: int = 8, n_trials: int = 200, subjects: tuple[str, ...] = ("a",)
) -> Study:
    generator = np.random.default_rng(123)
    rows_per_subject = n_sessions * n_trials
    return Study(
        {
            "subject": [subject for subject in subjects for _ in range(rows_per_subject)],
            "session": [
                f"session-{session}"
                for _ in subjects
                for session in range(n_sessions)
                for _ in range(n_trials)
            ],
            "trial": list(range(n_trials)) * n_sessions * len(subjects),
            "session_order": [
                session for _ in subjects for session in range(n_sessions) for _ in range(n_trials)
            ],
            "stimulus": generator.normal(size=rows_per_subject * len(subjects)),
        }
    )


def smooth_model(*, n_sessions: int = 8, shared_trajectory: bool = False):
    return SmoothBernoulliHistoryGLM(
        covariates=("stimulus",),
        choice_lags=1,
        knots=tuple(range(n_sessions)),
        smoothness=10.0,
        l2=0.01,
        shared_trajectory=shared_trajectory,
    )


def smooth_truth(model: SmoothBernoulliHistoryGLM) -> MappingProxyType[str, float]:
    n_knots = len(model.knots)
    return model.parameters_from_paths(
        {
            "intercept": np.linspace(-0.5, 0.5, n_knots),
            "stimulus": np.linspace(0.2, 2.5, n_knots),
            "choice_lag_1": np.linspace(0.8, 0.1, n_knots),
        }
    )


def test_smooth_model_has_stable_knot_parameterization() -> None:
    model = smooth_model(n_sessions=3)

    assert model.coefficient_names == ("intercept", "stimulus", "choice_lag_1")
    assert model.parameter_names == (
        "intercept[session_order=0]",
        "intercept[session_order=1]",
        "intercept[session_order=2]",
        "stimulus[session_order=0]",
        "stimulus[session_order=1]",
        "stimulus[session_order=2]",
        "choice_lag_1[session_order=0]",
        "choice_lag_1[session_order=1]",
        "choice_lag_1[session_order=2]",
    )
    parameters = smooth_truth(model)
    assert isinstance(parameters, MappingProxyType)
    assert parameters["stimulus[session_order=2]"] == 2.5


@pytest.mark.parametrize(
    "arguments",
    [
        {"knots": (0.0,)},
        {"knots": (0.0, 0.0)},
        {"knots": (1.0, 0.0)},
        {"knots": (0.0, np.nan)},
        {"smoothness": 0.0},
        {"time": "choice"},
        {"shared_trajectory": 1},
    ],
)
def test_smooth_model_configuration_is_validated(arguments: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        SmoothBernoulliHistoryGLM(**arguments)


def test_simulation_and_fitting_recover_an_inspectable_trajectory() -> None:
    model = smooth_model()
    design = make_design()
    study = model.simulate(design, smooth_truth(model), seed=77)

    fit = model.fit(study)
    trajectory = model.coefficient_trajectory(fit)

    assert fit.diagnostics.converged
    assert trajectory.clock == "session_order"
    assert trajectory.coefficient_names == model.coefficient_names
    assert trajectory.values.shape == (8, 3)
    stimulus_rmse = np.sqrt(np.mean((trajectory.values[:, 1] - np.linspace(0.2, 2.5, 8)) ** 2))
    assert stimulus_rmse < 0.3
    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        trajectory.values.setflags(write=True)


def test_trajectory_can_be_evaluated_between_knots() -> None:
    model = SmoothBernoulliHistoryGLM(choice_lags=0, knots=(0.0, 2.0), smoothness=1.0)
    parameters = model.parameters_from_paths({"intercept": [-1.0, 1.0]})
    design = Study(
        {
            "subject": ["a", "a"],
            "session": ["s0", "s2"],
            "trial": [0, 0],
            "session_order": [0, 2],
        }
    )
    fit = model.fit(model.simulate(design, parameters, seed=1))

    trajectory = model.coefficient_trajectory(fit, times=[0.0, 1.0, 2.0])

    assert trajectory.values[1, 0] == pytest.approx(
        (trajectory.values[0, 0] + trajectory.values[2, 0]) / 2
    )


def test_fixed_knots_must_cover_the_requested_clock() -> None:
    model = smooth_model(n_sessions=3)

    with pytest.raises(ModelDataError, match="fixed knot range"):
        model.simulate(make_design(n_sessions=4), smooth_truth(model), seed=1)


def test_subject_trajectories_are_not_aligned_silently() -> None:
    design = make_design(n_sessions=3, n_trials=20, subjects=("a", "b"))
    model = smooth_model(n_sessions=3)

    with pytest.raises(ModelDataError, match="subject-specific by default"):
        model.simulate(design, smooth_truth(model), seed=1)

    shared = smooth_model(n_sessions=3, shared_trajectory=True)
    simulated = shared.simulate(design, smooth_truth(shared), seed=1)
    assert len(simulated.subjects) == 2


def test_smooth_drift_beats_static_baseline_prospectively_when_truth_drifts() -> None:
    n_sessions = 10
    design = make_design(n_sessions=n_sessions, n_trials=120)
    smooth = smooth_model(n_sessions=n_sessions)
    study = smooth.simulate(design, smooth_truth(smooth), seed=77)
    splits = forward_session_splits(study, min_train_sessions=3)
    static = BernoulliHistoryGLM(covariates=("stimulus",), choice_lags=1, l2=0.01)

    static_evaluations = evaluate_splits(static, study, splits)
    smooth_evaluations = evaluate_splits(smooth, study, splits)
    static_loss = -np.mean(
        np.concatenate([item.pointwise_log_probability for item in static_evaluations])
    )
    smooth_loss = -np.mean(
        np.concatenate([item.pointwise_log_probability for item in smooth_evaluations])
    )

    assert smooth_loss < static_loss - 0.05


def test_stationary_truth_does_not_require_a_smooth_advantage() -> None:
    n_sessions = 8
    design = make_design(n_sessions=n_sessions, n_trials=160)
    smooth = SmoothBernoulliHistoryGLM(
        covariates=("stimulus",),
        choice_lags=1,
        knots=tuple(range(n_sessions)),
        smoothness=30.0,
        l2=0.01,
    )
    truth = smooth.parameters_from_paths(
        {
            "intercept": [-0.2] * n_sessions,
            "stimulus": [1.2] * n_sessions,
            "choice_lag_1": [0.4] * n_sessions,
        }
    )
    study = smooth.simulate(design, truth, seed=17)
    splits = forward_session_splits(study, min_train_sessions=3)
    static = BernoulliHistoryGLM(covariates=("stimulus",), choice_lags=1, l2=0.01)

    static_loss = -np.mean(
        np.concatenate(
            [item.pointwise_log_probability for item in evaluate_splits(static, study, splits)]
        )
    )
    smooth_loss = -np.mean(
        np.concatenate(
            [item.pointwise_log_probability for item in evaluate_splits(smooth, study, splits)]
        )
    )

    assert static_loss <= smooth_loss + 0.02


def test_existing_recovery_engine_accepts_smooth_paths() -> None:
    n_sessions = 4
    model = SmoothBernoulliHistoryGLM(
        covariates=("stimulus",),
        choice_lags=0,
        knots=tuple(range(n_sessions)),
        smoothness=10.0,
        l2=0.01,
    )
    parameter_sets = [
        model.parameters_from_paths(
            {"intercept": [0.0] * n_sessions, "stimulus": [0.5, 0.7, 0.9, 1.1]}
        ),
        model.parameters_from_paths(
            {
                "intercept": [-0.2, -0.1, 0.0, 0.1],
                "stimulus": [1.0, 1.2, 1.4, 1.6],
            }
        ),
    ]

    report = run_parameter_recovery(
        model,
        make_design(n_sessions=n_sessions, n_trials=100),
        parameter_sets,
        seed=13,
    )

    assert report.n_runs == 2
    assert report.convergence_rate == 1.0
    assert report.parameter_names == model.parameter_names
