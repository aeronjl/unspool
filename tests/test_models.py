from types import MappingProxyType

import numpy as np
import pytest
from scipy.special import expit

from behavio import (
    BehaviourEstimator,
    BehaviourModel,
    BernoulliHistoryGLM,
    FitDiagnostics,
    FitResult,
    GenerativeBehaviourModel,
    ModelCapabilities,
    ModelDataError,
    Prediction,
    PredictionMode,
    Study,
    UnsupportedPredictionMode,
    evaluate_splits,
    forward_session_splits,
    model_capabilities,
    run_parameter_recovery,
)


def make_design(*, n_sessions: int = 4, n_trials: int = 300) -> Study:
    generator = np.random.default_rng(90210)
    subject: list[str] = []
    session: list[str] = []
    trial: list[int] = []
    session_order: list[int] = []
    stimulus: list[float] = []
    for order in range(n_sessions):
        subject.extend(["mouse-a"] * n_trials)
        session.extend([f"session-{order}"] * n_trials)
        trial.extend(range(n_trials))
        session_order.extend([order] * n_trials)
        stimulus.extend(generator.normal(size=n_trials))
    return Study(
        {
            "subject": subject,
            "session": session,
            "trial": trial,
            "session_order": session_order,
            "stimulus": stimulus,
        }
    )


def known_fit(model: BernoulliHistoryGLM, estimates: list[float]) -> FitResult:
    n_parameters = len(estimates)
    return FitResult(
        model_name=model.model_name,
        model_signature=model.signature,
        parameter_names=model.parameter_names,
        estimates=np.asarray(estimates),
        standard_errors=np.ones(n_parameters),
        covariance=np.eye(n_parameters),
        n_observations=3,
        diagnostics=FitDiagnostics(
            converged=True,
            optimizer="test",
            status=0,
            message="known test fit",
            n_iterations=0,
            objective=0.0,
            gradient_norm=0.0,
            hessian_condition=1.0,
            boundary_estimate=False,
        ),
    )


class FitOnlyEstimator:
    """Minimal third-party-style estimator without a simulator."""

    def __init__(self, delegate: BernoulliHistoryGLM) -> None:
        self.delegate = delegate

    @property
    def model_name(self) -> str:
        return self.delegate.model_name

    @property
    def signature(self) -> str:
        return self.delegate.signature

    @property
    def scored_columns(self) -> tuple[str, ...]:
        return self.delegate.scored_columns

    @property
    def supported_prediction_modes(self) -> tuple[PredictionMode, ...]:
        return self.delegate.supported_prediction_modes

    def fit(self, study: Study) -> FitResult:
        return self.delegate.fit(study)

    def predict(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> Prediction:
        return self.delegate.predict(study, fit, mode=mode)

    def pointwise_log_prob(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> np.ndarray:
        return self.delegate.pointwise_log_prob(study, fit, mode=mode)


def test_model_satisfies_public_contract() -> None:
    model = BernoulliHistoryGLM(covariates=("stimulus",), choice_lags=2)

    assert isinstance(model, BehaviourModel)
    assert isinstance(model, BehaviourEstimator)
    assert isinstance(model, GenerativeBehaviourModel)
    assert model.scored_columns == ("choice",)
    assert model.parameter_names == (
        "intercept",
        "stimulus",
        "choice_lag_1",
        "choice_lag_2",
    )
    assert model.supported_prediction_modes == (PredictionMode.FILTERED,)
    assert "choice_lags=2" in model.signature

    capabilities = model_capabilities(model)
    assert capabilities == ModelCapabilities(
        scored_columns=("choice",),
        prediction_modes=(PredictionMode.FILTERED,),
        can_simulate=True,
        can_recover_parameters=True,
        required_task_columns=("stimulus",),
    )


def test_fit_only_estimators_can_be_evaluated_but_not_sent_to_recovery() -> None:
    generator = BernoulliHistoryGLM(covariates=("stimulus",), choice_lags=0)
    estimator = FitOnlyEstimator(generator)
    simulated = generator.simulate(
        make_design(n_sessions=3, n_trials=30),
        {"intercept": -0.2, "stimulus": 1.0},
        seed=15,
    )

    assert isinstance(estimator, BehaviourEstimator)
    assert not isinstance(estimator, GenerativeBehaviourModel)
    capabilities = model_capabilities(estimator)
    assert not capabilities.can_simulate
    assert not capabilities.can_recover_parameters
    evaluations = evaluate_splits(
        estimator,
        simulated,
        forward_session_splits(simulated, min_train_sessions=2),
    )
    assert len(evaluations) == 1
    with pytest.raises(TypeError, match="GenerativeBehaviourModel"):
        run_parameter_recovery(
            estimator,  # type: ignore[arg-type]
            make_design(n_sessions=1, n_trials=10),
            [{"intercept": 0.0, "stimulus": 1.0}],
            seed=0,
        )


def test_capability_metadata_rejects_ambiguous_scalar_columns_and_flags() -> None:
    with pytest.raises(ValueError, match="tuple of column names"):
        ModelCapabilities(
            scored_columns="choice",  # type: ignore[arg-type]
            prediction_modes=(PredictionMode.FILTERED,),
            can_simulate=False,
            can_recover_parameters=False,
        )
    with pytest.raises(ValueError, match="requires simulation"):
        ModelCapabilities(
            scored_columns=("choice",),
            prediction_modes=(PredictionMode.FILTERED,),
            can_simulate=False,
            can_recover_parameters=True,
        )


@pytest.mark.parametrize(
    "arguments",
    [
        {"covariates": ("stimulus", "stimulus")},
        {"covariates": ("choice_lag_1",)},
        {"outcome": "trial"},
        {"choice_lags": -1},
        {"l2": -1.0},
        {"max_iterations": 1.5},
    ],
)
def test_model_configuration_is_validated(arguments: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        BernoulliHistoryGLM(**arguments)


def test_simulation_is_reproducible_and_preserves_design() -> None:
    model = BernoulliHistoryGLM(covariates=("stimulus",), choice_lags=1)
    design = make_design(n_sessions=2, n_trials=20)
    parameters = {"intercept": -0.2, "stimulus": 1.1, "choice_lag_1": 0.7}

    first = model.simulate(design, parameters, seed=42)
    second = model.simulate(design, parameters, seed=42)

    assert np.array_equal(first["choice"], second["choice"])
    assert np.array_equal(first["stimulus"], design["stimulus"])
    assert np.array_equal(first["trial"], design["trial"])
    assert set(first["choice"].tolist()) <= {0, 1}
    assert "choice" not in design.columns


def test_filtered_history_respects_trial_order_and_session_boundaries() -> None:
    study = Study(
        {
            "subject": ["a", "a", "a"],
            "session": ["first", "first", "second"],
            "trial": [1, 0, 0],
            "session_order": [0, 0, 1],
            "choice": [1, 0, 1],
        }
    )
    model = BernoulliHistoryGLM(choice_lags=1)
    prediction = model.predict(study, known_fit(model, [0.0, 2.0]))

    assert prediction.mode is PredictionMode.FILTERED
    assert prediction.probability.tolist() == pytest.approx([expit(-2.0), 0.5, 0.5])


def test_filtered_prediction_does_not_use_future_choices() -> None:
    model = BernoulliHistoryGLM(choice_lags=1)
    original = Study(
        {
            "subject": ["a", "a", "a"],
            "session": ["s", "s", "s"],
            "trial": [0, 1, 2],
            "session_order": [0, 0, 0],
            "choice": [0, 0, 0],
        }
    )
    changed_future = Study(
        {
            "subject": ["a", "a", "a"],
            "session": ["s", "s", "s"],
            "trial": [0, 1, 2],
            "session_order": [0, 0, 0],
            "choice": [0, 0, 1],
        }
    )
    fit = known_fit(model, [0.0, 2.0])

    first = model.predict(original, fit)
    second = model.predict(changed_future, fit)

    assert np.array_equal(first.probability, second.probability)


def test_static_glm_recovers_parameters_and_exposes_diagnostics() -> None:
    model = BernoulliHistoryGLM(covariates=("stimulus",), choice_lags=1)
    truth = {"intercept": -0.25, "stimulus": 1.1, "choice_lag_1": 0.55}
    simulated = model.simulate(make_design(), truth, seed=7)

    fit = model.fit(simulated)

    assert fit.diagnostics.converged
    assert fit.n_observations == 1_200
    assert fit.estimates.tolist() == pytest.approx(list(truth.values()), abs=0.2)
    assert isinstance(fit.parameters, MappingProxyType)
    assert fit.parameters["stimulus"] == fit.estimates[1]
    assert np.all(fit.standard_errors > 0)
    assert np.all(np.isfinite(fit.covariance))
    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        fit.estimates.setflags(write=True)


def test_separation_remains_visible_as_a_boundary_diagnostic() -> None:
    n_trials = 40
    study = Study(
        {
            "subject": ["a"] * n_trials,
            "session": ["s"] * n_trials,
            "trial": list(range(n_trials)),
            "session_order": [0] * n_trials,
            "choice": [1] * n_trials,
        }
    )

    fit = BernoulliHistoryGLM(choice_lags=0).fit(study)

    assert fit.diagnostics.converged
    assert fit.diagnostics.boundary_estimate
    assert fit.estimates[0] >= 20.0


def test_pointwise_log_probability_matches_prediction() -> None:
    model = BernoulliHistoryGLM(choice_lags=0)
    study = Study(
        {
            "subject": ["a", "a"],
            "session": ["s", "s"],
            "trial": [0, 1],
            "session_order": [0, 0],
            "choice": [1, 0],
        }
    )
    fit = known_fit(model, [0.4])

    prediction = model.predict(study, fit)
    scores = model.pointwise_log_prob(study, fit)

    assert scores.tolist() == pytest.approx(
        [np.log(prediction.probability[0]), np.log1p(-prediction.probability[1])]
    )


def test_smoothed_prediction_is_rejected_explicitly() -> None:
    model = BernoulliHistoryGLM(choice_lags=0)
    study = Study(
        {
            "subject": ["a"],
            "session": ["s"],
            "trial": [0],
            "session_order": [0],
            "choice": [1],
        }
    )

    with pytest.raises(UnsupportedPredictionMode, match="only filtered"):
        model.predict(study, known_fit(model, [0.0]), mode=PredictionMode.SMOOTHED)


def test_invalid_model_data_remains_visible() -> None:
    design = make_design(n_sessions=1, n_trials=3)
    model = BernoulliHistoryGLM(covariates=("stimulus",))

    with pytest.raises(ModelDataError, match="missing outcome"):
        model.fit(design)

    invalid = Study(
        {
            **{name: design[name] for name in design.columns},
            "choice": [0, 2, 1],
        }
    )
    with pytest.raises(ModelDataError, match="zero and one"):
        model.fit(invalid)


def test_fit_result_must_match_model_specification() -> None:
    first = BernoulliHistoryGLM(choice_lags=0, l2=0.0)
    second = BernoulliHistoryGLM(choice_lags=0, l2=1.0)
    study = Study(
        {
            "subject": ["a"],
            "session": ["s"],
            "trial": [0],
            "session_order": [0],
            "choice": [1],
        }
    )

    with pytest.raises(ValueError, match="different model specification"):
        second.predict(study, known_fit(first, [0.0]))
