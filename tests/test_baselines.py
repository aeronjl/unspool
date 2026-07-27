import numpy as np
import pytest

from unspool import (
    BehaviourModel,
    BiasOnly,
    LapsePsychometric,
    LapsePsychometricFitResult,
    Perseveration,
    Psychometric,
    Study,
    WinStayLoseShift,
    run_parameter_recovery,
)


def design(n_trials: int = 400) -> Study:
    return Study(
        {
            "subject": ["a"] * n_trials,
            "session": ["s"] * n_trials,
            "trial": list(range(n_trials)),
            "session_order": [0] * n_trials,
            "stimulus": np.linspace(-3.0, 3.0, n_trials),
        }
    )


@pytest.mark.parametrize(
    ("model", "parameters"),
    [
        (BiasOnly(), {"intercept": -0.4}),
        (Psychometric(), {"intercept": -0.2, "stimulus": 1.1}),
        (Perseveration(), {"intercept": -0.2, "choice_lag_1": 0.7}),
    ],
)
def test_named_glm_baselines_satisfy_the_complete_model_contract(model, parameters) -> None:
    study = model.simulate(design(), parameters, seed=10)
    fit = model.fit(study)

    assert isinstance(model, BehaviourModel)
    assert fit.model_name == model.model_name
    assert fit.model_signature == model.signature
    assert model.predict(study, fit).probability.shape == (len(study),)
    assert np.all(np.isfinite(model.pointwise_log_prob(study, fit)))
    assert fit.audit().model_name == model.model_name


def test_lapse_psychometric_retains_natural_rate_and_restart_evidence() -> None:
    model = LapsePsychometric(maximum_lapse=0.3, n_restarts=4)
    parameters = model.parameters_from_components(intercept=-0.1, slope=1.4, lapse_rate=0.08)
    study = model.simulate(design(1_000), parameters, seed=25)

    fit = model.fit(study)
    components = model.parameter_components(fit)

    assert isinstance(model, BehaviourModel)
    assert isinstance(fit, LapsePsychometricFitResult)
    assert len(fit.restart_objectives) == 4
    assert fit.lapse_rate == pytest.approx(components.lapse_rate)
    assert 0 < components.lapse_rate < model.maximum_lapse
    assert components.slope > 0.5
    assert model.predict(study, fit).probability.min() >= components.lapse_rate / 2
    assert model.predict(study, fit).probability.max() <= 1 - components.lapse_rate / 2
    assert np.all(np.isfinite(model.pointwise_log_prob(study, fit)))


def test_lapse_psychometric_participates_in_parameter_recovery() -> None:
    model = LapsePsychometric(n_restarts=2)
    truth = dict(model.parameters_from_components(intercept=0.1, slope=1.0, lapse_rate=0.05))

    report = run_parameter_recovery(model, design(300), [truth], seed=2)

    assert report.model_name == model.model_name
    assert report.audits[0].model_name == model.model_name
    assert report.n_runs == 1


def test_lapse_psychometric_validates_natural_parameters_and_fit_identity() -> None:
    model = LapsePsychometric(maximum_lapse=0.2)
    with pytest.raises(ValueError, match="smaller than maximum_lapse"):
        model.parameters_from_components(intercept=0.0, slope=1.0, lapse_rate=0.2)

    other = LapsePsychometric(maximum_lapse=0.3)
    parameters = other.parameters_from_components(intercept=0.0, slope=1.0, lapse_rate=0.1)
    study = other.simulate(design(100), parameters, seed=4)
    fit = other.fit(study)
    with pytest.raises(ValueError, match="different model"):
        model.predict(study, fit)


def test_win_stay_lose_shift_has_explicit_reward_semantics_and_recovery() -> None:
    model = WinStayLoseShift()
    base = design(600)
    bandit = Study(
        {
            **{name: base[name] for name in base.columns},
            "reward_probability_0": [0.3] * len(base),
            "reward_probability_1": [0.7] * len(base),
        }
    )
    parameters = {"intercept": -0.1, "win_stay": 1.0, "lose_shift": 0.8}
    study = model.simulate(bandit, parameters, seed=9)
    fit = model.fit(study)

    assert isinstance(model, BehaviourModel)
    assert fit.model_name == model.model_name
    assert fit.parameters["win_stay"] > 0
    assert fit.parameters["lose_shift"] > 0
    assert np.all(np.isfinite(model.pointwise_log_prob(study, fit)))
    report = run_parameter_recovery(model, bandit, [parameters], seed=11)
    assert report.model_name == model.model_name
