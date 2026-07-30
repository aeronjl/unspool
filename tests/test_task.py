import numpy as np
import pytest

from behavio import (
    BernoulliHistoryGLM,
    ChoiceSpec,
    ResponseTimeSpec,
    RewardSpec,
    Study,
    TaskSpec,
    TaskValidationError,
    WinStayLoseShift,
    fit_model,
)


def make_study() -> Study:
    available = np.empty(5, dtype=object)
    available[:] = [
        ("left", "right"),
        ("left", "right"),
        ("left", "right"),
        ("left",),
        ("right",),
    ]
    return Study(
        {
            "subject": ["a"] * 5,
            "session": ["s"] * 5,
            "trial": list(range(5)),
            "session_order": [0] * 5,
            "action": ["left", "right", "omit", "left", "right"],
            "available": available,
            "reward": [0.0, 1.0, np.nan, 1.0, 0.0],
            "stimulus": [-1.0, 1.0, 0.0, -0.5, 0.5],
            "rt_ms": [300.0, 450.0, 500.0, 350.0, 400.0],
        }
    )


def test_task_spec_validates_multialternative_coordinates_and_denominators() -> None:
    task = TaskSpec(
        choice=ChoiceSpec(
            column="action",
            options=("left", "right"),
            omission_values=("omit",),
            available_options_column="available",
        ),
        predictors=("stimulus",),
        reward=RewardSpec(minimum=0.0, maximum=1.0, allow_missing=True),
        response_time=ResponseTimeSpec(column="rt_ms", unit="milliseconds"),
    )

    choices = task.choice.read(make_study())
    validation = task.validate(make_study())

    assert choices.codes.tolist() == [0, 1, -1, 0, 1]
    assert choices.counts == {"left": 2, "right": 2}
    assert choices.available.tolist()[-2:] == [[True, False], [False, True]]
    assert validation.n_trials == 5
    assert validation.n_observed_choices == 4
    assert validation.n_omissions == 1
    assert validation.has_rewards
    assert validation.has_response_times
    with pytest.raises(ValueError, match="cannot set WRITEABLE"):
        choices.codes.setflags(write=True)


def test_task_spec_rejects_undeclared_choices_and_unavailable_actions() -> None:
    study = make_study()
    with pytest.raises(TaskValidationError, match="outside declared"):
        ChoiceSpec(column="action", options=("left", "up")).read(study)

    unavailable = np.empty(len(study), dtype=object)
    unavailable[:] = [("right",)] * len(study)
    invalid = Study(
        {
            **{name: study[name] for name in study.columns if name != "available"},
            "available": unavailable,
        }
    )
    with pytest.raises(TaskValidationError, match="was not available"):
        ChoiceSpec(
            column="action",
            options=("left", "right"),
            omission_values=("omit",),
            available_options_column="available",
        ).read(invalid)


def test_missing_choices_require_an_explicit_omission_policy() -> None:
    study = Study(
        {
            "subject": ["a", "a"],
            "session": ["s", "s"],
            "trial": [0, 1],
            "session_order": [0, 0],
            "choice": [0, np.nan],
        }
    )
    with pytest.raises(TaskValidationError, match="missing_is_omission"):
        ChoiceSpec(options=(0, 1)).read(study)

    choices = ChoiceSpec(options=(0, 1), missing_is_omission=True).read(study)
    assert choices.codes.tolist() == [0, -1]


def test_fit_model_is_the_task_validated_golden_path() -> None:
    design = Study(
        {
            "subject": ["a"] * 80,
            "session": ["s"] * 80,
            "trial": list(range(80)),
            "session_order": [0] * 80,
            "stimulus": np.linspace(-2.0, 2.0, 80),
        }
    )
    model = BernoulliHistoryGLM(predictors=("stimulus",), choice_lags=0)
    study = model.simulate(design, {"intercept": -0.1, "stimulus": 1.2}, seed=42)
    task = TaskSpec(choice=ChoiceSpec(options=(0, 1)), predictors=("stimulus",))

    fitted = fit_model(model, study, task=task)

    assert fitted.validation.n_trials == 80
    assert fitted.result.model_signature == model.signature
    assert fitted.predict(study).probability.shape == (80,)
    assert fitted.pointwise_log_prob(study).shape == (80,)
    assert fitted.audit().model_name == model.model_name


def test_fit_model_rejects_modelled_observations_absent_from_the_task() -> None:
    model = BernoulliHistoryGLM(outcome="choice")
    task = TaskSpec(choice=ChoiceSpec(column="action", options=("left", "right")))

    with pytest.raises(TaskValidationError, match="not declared"):
        task.validate_model(model)


def test_fit_model_rejects_context_columns_without_a_declared_task_role() -> None:
    model = WinStayLoseShift(reward="feedback")
    task = TaskSpec(choice=ChoiceSpec(options=(0, 1)))

    with pytest.raises(TaskValidationError, match="without a declared task role"):
        task.validate_model(model)

    task = TaskSpec(
        choice=ChoiceSpec(options=(0, 1)),
        reward=RewardSpec(column="feedback"),
    )
    task.validate_model(model)
