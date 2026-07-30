import numpy as np
import pytest

from behavio import (
    BiasOnly,
    Perseveration,
    Psychometric,
    Study,
    UniformChoiceGuess,
    WinStayLoseShift,
    mix,
    run_parameter_recovery,
)
from behavio.models import BehaviourModel


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


def lapse_psychometric(maximum_lapse: float = 0.3, **changes: object):
    """What the deleted ``LapsePsychometric`` class is now an expression for."""

    return mix(
        Psychometric(),
        UniformChoiceGuess(),
        weight_bounds=(0.0, maximum_lapse),
        **changes,
    )


def test_a_mixed_psychometric_replaces_the_deleted_lapse_class() -> None:
    model = lapse_psychometric(n_restarts=4)
    parameters = model.from_natural({"intercept": -0.1, "stimulus": 1.4, "lapse_rate": 0.08})
    study = model.simulate(design(1_000), parameters, seed=25)

    fit = model.fit(study)
    natural = model.to_natural(fit.estimates)
    probability = model.predict(study, fit).probability

    assert isinstance(model, BehaviourModel)
    assert fit.derived_value("lapse_rate") == pytest.approx(natural["lapse_rate"])
    assert fit.derived_quantities["lapse_rate"].standard_error > 0
    assert 0 < natural["lapse_rate"] < 0.3
    assert natural["stimulus"] > 0.5
    # The deleted class mixed a logistic with a symmetric coin, so the curve was confined
    # to [lapse/2, 1 - lapse/2]. That is a consequence of the mixture rather than a
    # property of the class, so it survives the class.
    assert probability.min() >= natural["lapse_rate"] / 2
    assert probability.max() <= 1 - natural["lapse_rate"] / 2
    assert np.all(np.isfinite(model.pointwise_log_prob(study, fit)))


def test_a_mixed_psychometric_is_the_closed_form_the_deleted_class_evaluated() -> None:
    """``lapse/2 + (1 - lapse) * expit(a + b x)``, computed here rather than trusted."""

    model = lapse_psychometric()
    parameters = model.from_natural({"intercept": -0.1, "stimulus": 1.4, "lapse_rate": 0.08})
    study = model.simulate(design(500), parameters, seed=25)
    fit = model.fit(study)
    natural = model.to_natural(fit.estimates)

    logistic = 1.0 / (
        1.0 + np.exp(-(natural["intercept"] + natural["stimulus"] * np.asarray(study["stimulus"])))
    )
    expected = natural["lapse_rate"] * 0.5 + (1.0 - natural["lapse_rate"]) * logistic

    assert model.predict(study, fit).probability == pytest.approx(expected)


def test_a_mixed_psychometric_participates_in_parameter_recovery() -> None:
    model = lapse_psychometric(maximum_lapse=0.2, n_restarts=2)
    truth = dict(model.from_natural({"intercept": 0.1, "stimulus": 1.0, "lapse_rate": 0.05}))

    report = run_parameter_recovery(model, design(300), [truth], seed=2)

    assert report.model_name == model.model_name
    assert report.audits[0].model_name == model.model_name
    assert report.n_runs == 1


def test_a_mixed_psychometric_validates_its_declared_weight_range() -> None:
    model = lapse_psychometric(maximum_lapse=0.2)
    with pytest.raises(ValueError, match="strictly inside"):
        model.from_natural({"intercept": 0.0, "stimulus": 1.0, "lapse_rate": 0.2})

    other = lapse_psychometric(maximum_lapse=0.3)
    parameters = other.from_natural({"intercept": 0.0, "stimulus": 1.0, "lapse_rate": 0.1})
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
