import numpy as np
import pytest

from unspool import BernoulliHistoryGLM, Study, run_parameter_recovery


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
    assert len(first.summary()) == 3
    assert all(summary.n_successful == 6 for summary in first.summary())
    assert all(np.isfinite(summary.rmse) for summary in first.summary())
    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        first.estimates.setflags(write=True)


@pytest.mark.parametrize(
    ("parameter_sets", "repeats", "seed", "message"),
    [
        ([], 1, 0, "must not be empty"),
        ([{"intercept": 0.0}], 0, 0, "repeats"),
        ([{"intercept": 0.0}], 1, -1, "seed"),
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
