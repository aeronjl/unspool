"""The estimator conformance harness, and the SMOOTHED violation it exists to catch."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pytest

from behavio import Study
from behavio.adapters import (
    CheckStatus,
    EstimatorConformanceError,
    assert_behaviour_estimator_conforms,
    check_behaviour_estimator,
    perturb_future_rows,
)
from behavio.contracts import (
    FitDiagnostics,
    FitResult,
    Prediction,
    PredictionMode,
    UnsupportedPredictionMode,
)
from behavio.models.baselines import BiasOnly
from behavio.models.glm import BernoulliHistoryGLM
from behavio.models.glm_hmm import BernoulliGLMHMM


def study(trials: int = 20, subjects: int = 2, sessions: int = 2, seed: int = 3) -> Study:
    return Study.factorial(
        trials=trials,
        subjects=subjects,
        sessions=sessions,
        columns={
            "stimulus": lambda generator, rows: generator.normal(size=rows),
            "choice": lambda generator, rows: generator.integers(0, 2, size=rows),
        },
        seed=seed,
    )


@dataclass(frozen=True, slots=True)
class RunningMeanModel:
    """A minimal estimator whose prediction is a running mean of past choices.

    Deliberately tiny and deliberately honest: the prediction for row *t* reads rows
    strictly before *t* within the same session and nothing else, which is what
    ``FILTERED`` means. ``horizon`` widens the window into the *future*, which is how the
    tests below manufacture the violation the harness has to catch, and ``declared_modes``
    lets a variant lie about which information set it used.
    """

    horizon: int = 0
    declared_modes: tuple[PredictionMode, ...] = (PredictionMode.FILTERED,)
    label_as: PredictionMode | None = None

    @property
    def model_name(self) -> str:
        return "running-mean"

    @property
    def signature(self) -> str:
        return f"running-mean[horizon={self.horizon}]"

    @property
    def scored_columns(self) -> tuple[str, ...]:
        return ("choice",)

    @property
    def required_task_columns(self) -> tuple[str, ...]:
        return ()

    @property
    def supported_prediction_modes(self) -> tuple[PredictionMode, ...]:
        return self.declared_modes

    def fit(self, study: Study) -> FitResult:
        return FitResult(
            model_name=self.model_name,
            model_signature=self.signature,
            parameter_names=("prior",),
            estimates=np.asarray([0.5]),
            standard_errors=np.asarray([0.1]),
            covariance=np.asarray([[0.01]]),
            n_observations=len(study),
            diagnostics=FitDiagnostics.closed_form(procedure="running-mean", message="exact"),
        )

    def _probability(self, study: Study, horizon: int) -> np.ndarray:
        choices = np.asarray(study["choice"], dtype=np.float64)
        subjects = np.asarray(study["subject"])
        sessions = np.asarray(study["session"])
        order = np.asarray(study.chronological_indices())
        probability = np.full(len(study), 0.5)
        for position, row in enumerate(order):
            same = [
                other
                for offset, other in enumerate(order)
                if subjects[other] == subjects[row]
                and sessions[other] == sessions[row]
                and offset < position + horizon
                and offset != position
            ]
            if same:
                probability[row] = np.clip(float(np.mean(choices[same])), 0.02, 0.98)
        return probability

    def predict(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> Prediction:
        requested = PredictionMode(mode)
        if requested not in self.declared_modes:
            raise UnsupportedPredictionMode(f"{self.model_name} cannot report {requested}")
        horizon = self.horizon
        if requested is PredictionMode.SMOOTHED:
            horizon = len(study)
        probability = self._probability(study, horizon)
        return Prediction(
            probability=probability,
            linear_predictor=np.log(probability) - np.log1p(-probability),
            mode=self.label_as or requested,
        )

    def pointwise_log_prob(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> np.ndarray:
        prediction = self.predict(study, fit, mode=mode)
        outcomes = np.asarray(study["choice"], dtype=np.float64)
        probability = np.asarray(prediction.probability)
        return outcomes * np.log(probability) + (1 - outcomes) * np.log1p(-probability)


def named(report, name: str):
    return next(check for check in report.checks if check.name == name)


@pytest.mark.parametrize(
    "model",
    [
        BiasOnly(),
        BernoulliHistoryGLM(predictors=("stimulus",), choice_lags=1),
        BernoulliGLMHMM(predictors=("stimulus",), n_states=2, n_restarts=2),
        RunningMeanModel(),
    ],
    ids=["bias-only", "history-glm", "glm-hmm", "running-mean"],
)
def test_a_filtered_estimator_passes_every_executable_check(model) -> None:
    report = check_behaviour_estimator(model, study())

    assert report.passed, report.summary()
    assert named(report, "filtered-prediction-ignores-future-rows").status is CheckStatus.PASSED
    assert named(report, "filtered-score-ignores-future-rows").status is CheckStatus.PASSED


def test_a_smoothed_prediction_labelled_filtered_is_caught() -> None:
    """The violation the contract could not previously detect.

    ``ssm.most_likely_states`` and a ``dynamax`` smoother both return a state estimate that
    reads the whole sequence. A naive wrapper returns it and stamps it ``FILTERED``. Nothing
    structural distinguishes the two, so this check perturbs the future and looks.
    """

    leaky = RunningMeanModel(horizon=1000)

    report = check_behaviour_estimator(leaky, study())

    assert not report.passed
    failure = named(report, "filtered-prediction-ignores-future-rows")
    assert failure.status is CheckStatus.FAILED
    assert "cannot depend on trials after t" in failure.detail
    assert named(report, "filtered-score-ignores-future-rows").status is CheckStatus.FAILED


def test_a_model_that_advertises_smoothing_must_actually_use_the_future() -> None:
    honest = RunningMeanModel(
        declared_modes=(PredictionMode.FILTERED, PredictionMode.SMOOTHED),
    )

    report = check_behaviour_estimator(honest, study())

    assert report.passed, report.summary()
    smoothed = named(report, "smoothed-prediction-uses-future-rows")
    assert smoothed.status is CheckStatus.PASSED
    assert "responds to later trials" in smoothed.detail


def test_an_advertised_smoothed_mode_that_is_really_filtered_is_caught() -> None:
    """A model may not collect the SMOOTHED label for a filtered computation."""

    class StillFiltered(RunningMeanModel):
        def predict(self, study, fit, *, mode=PredictionMode.FILTERED):
            requested = PredictionMode(mode)
            if requested not in self.declared_modes:
                raise UnsupportedPredictionMode("no")
            probability = self._probability(study, 0)
            return Prediction(
                probability=probability,
                linear_predictor=np.log(probability) - np.log1p(-probability),
                mode=requested,
            )

    report = check_behaviour_estimator(
        StillFiltered(declared_modes=(PredictionMode.FILTERED, PredictionMode.SMOOTHED)),
        study(),
    )

    smoothed = named(report, "smoothed-prediction-uses-future-rows")
    assert smoothed.status is CheckStatus.FAILED
    assert "filtered estimate wearing a smoothed label" in smoothed.detail


def test_a_prediction_stamped_with_the_wrong_mode_is_caught() -> None:
    mislabelled = RunningMeanModel(label_as=PredictionMode.SMOOTHED)

    report = check_behaviour_estimator(mislabelled, study())

    assert named(report, "predicts-one-row-per-trial").status is CheckStatus.FAILED


def test_an_undeclared_mode_must_raise_rather_than_answer() -> None:
    @dataclass(frozen=True, slots=True)
    class Permissive(RunningMeanModel):
        def predict(self, study, fit, *, mode=PredictionMode.FILTERED):
            probability = self._probability(study, 0)
            return Prediction(
                probability=probability,
                linear_predictor=np.log(probability) - np.log1p(-probability),
                mode=PredictionMode.FILTERED,
            )

    report = check_behaviour_estimator(Permissive(), study())

    failure = named(report, "refuses-undeclared-prediction-modes")
    assert failure.status is CheckStatus.FAILED
    assert "does not declare" in failure.detail


def test_a_study_with_nothing_to_perturb_is_skipped_rather_than_passed() -> None:
    """A check the evidence cannot run is not a check the model passed."""

    constant = Study(
        {
            "subject": ["m1"] * 6,
            "session": ["d1"] * 6,
            "trial": list(range(6)),
            "session_order": [0] * 6,
            "choice": [1] * 6,
        }
    )

    report = check_behaviour_estimator(RunningMeanModel(), constant)

    skipped = named(report, "filtered-prediction-ignores-future-rows")
    assert skipped.status is CheckStatus.SKIPPED
    assert "changed no column the model reads" in skipped.detail
    assert report.passed
    with pytest.raises(EstimatorConformanceError, match="skipped"):
        assert_behaviour_estimator_conforms(RunningMeanModel(), constant, require_complete=True)


def test_perturbation_keeps_identity_chronology_and_stays_in_the_column() -> None:
    original = study()

    perturbed, past = perturb_future_rows(original, columns=("choice", "stimulus"))

    assert len(perturbed) == len(original)
    assert past.size == len(original) // 2
    for column in ("subject", "session", "trial", "session_order"):
        assert np.array_equal(np.asarray(perturbed[column]), np.asarray(original[column]))
    for column in ("choice", "stimulus"):
        assert set(np.asarray(perturbed[column]).tolist()) <= set(
            np.asarray(original[column]).tolist()
        )
        assert not np.array_equal(np.asarray(perturbed[column]), np.asarray(original[column]))
    assert np.array_equal(
        np.asarray(perturbed["choice"])[past], np.asarray(original["choice"])[past]
    )


def test_a_broken_fit_identity_is_reported_before_anything_else_runs() -> None:
    @dataclass(frozen=True, slots=True)
    class WrongName(RunningMeanModel):
        def fit(self, study: Study) -> FitResult:
            result = RunningMeanModel.fit(self, study)
            return FitResult(
                model_name="something-else",
                model_signature=result.model_signature,
                parameter_names=result.parameter_names,
                estimates=result.estimates,
                standard_errors=result.standard_errors,
                covariance=result.covariance,
                n_observations=result.n_observations,
                diagnostics=result.diagnostics,
            )

    report = check_behaviour_estimator(WrongName(), study())

    assert named(report, "fit-reports-the-training-study").status is CheckStatus.FAILED


def test_an_object_that_is_not_an_estimator_fails_the_first_check() -> None:
    report = check_behaviour_estimator(object(), study())

    assert report.capabilities is None
    assert len(report.checks) == 1
    assert report.checks[0].status is CheckStatus.FAILED


def test_the_assert_helper_raises_with_the_full_run_in_its_message() -> None:
    with pytest.raises(EstimatorConformanceError) as error:
        assert_behaviour_estimator_conforms(RunningMeanModel(horizon=1000), study())

    assert "filtered-prediction-ignores-future-rows" in str(error.value)


def test_a_simulator_that_reads_a_global_stream_is_caught() -> None:
    @dataclass(frozen=True, slots=True)
    class LeakySimulator(RunningMeanModel):
        @property
        def parameter_names(self) -> tuple[str, ...]:
            return ("prior",)

        def simulate(self, design: Study, parameters: Mapping[str, float], *, seed) -> Study:
            columns = {name: design[name] for name in design.columns}
            columns["choice"] = np.random.default_rng().integers(0, 2, size=len(design))
            return Study(columns)

    report = check_behaviour_estimator(LeakySimulator(), study())

    failure = named(report, "simulates-the-columns-it-scores")
    assert failure.status is CheckStatus.FAILED
    assert "global random stream" in failure.detail
