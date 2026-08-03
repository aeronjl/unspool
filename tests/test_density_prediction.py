"""The prediction type for a continuous outcome, and its passage through the score stack."""

from __future__ import annotations

import numpy as np
import pytest

from behavio import Study, compare_models, forward_session_splits
from behavio.compare.models import UnscoreableByBrier
from behavio.contracts import (
    LOG_DENSITY_FLOOR,
    CategoricalPrediction,
    CensoredDensityPrediction,
    DensityBehaviourEstimator,
    DensityPrediction,
    FitDiagnostics,
    FitResult,
    ModelPrediction,
    Prediction,
    PredictionMode,
)
from behavio.evaluate import evaluate_splits
from behavio.models.baselines import BiasOnly


def exponential_density(rate: float, grid: np.ndarray) -> np.ndarray:
    return rate * np.exp(-rate * grid)


def unlabelled(n_trials: int = 3, points: int = 4001) -> DensityPrediction:
    grid = np.linspace(0.0, 40.0, points)
    density = np.stack([exponential_density(rate, grid) for rate in (0.5, 1.0, 2.0)][:n_trials])
    return DensityPrediction(
        grid=grid, density=density, outcome="response_time", mode=PredictionMode.FILTERED
    )


def defective(points: int = 4001) -> DensityPrediction:
    """Two defective exponentials whose masses are 0.3 and 0.7."""

    grid = np.linspace(0.0, 40.0, points)
    lower = 0.3 * exponential_density(1.0, grid)
    upper = 0.7 * exponential_density(2.0, grid)
    density = np.stack([np.stack([lower, upper]), np.stack([upper, lower])])
    return DensityPrediction(
        grid=grid,
        density=density,
        outcome="response_time",
        mode=PredictionMode.FILTERED,
        categories=(0, 1),
    )


def test_an_unlabelled_density_integrates_to_one_and_reports_its_mean() -> None:
    prediction = unlabelled()

    assert prediction.n_observations == 3
    assert not prediction.is_defective
    assert np.allclose(prediction.total_mass, 1.0, atol=1e-4)
    assert np.allclose(prediction.expected_outcome(), [2.0, 1.0, 0.5], rtol=1e-3)


def test_a_defective_density_reports_the_mass_of_each_category() -> None:
    prediction = defective()

    assert prediction.is_defective
    assert np.allclose(prediction.category_mass, [[0.3, 0.7], [0.7, 0.3]], atol=1e-4)
    assert np.allclose(prediction.total_mass, 1.0, atol=1e-4)


def test_the_choice_prediction_is_the_integrated_density() -> None:
    categorical = defective().choice_prediction()

    assert categorical.categories == (0, 1)
    assert categorical.mode is PredictionMode.FILTERED
    assert np.allclose(categorical.probability, [[0.3, 0.7], [0.7, 0.3]], atol=1e-4)


def test_density_at_interpolates_rather_than_snapping_to_the_grid() -> None:
    """A value between two grid points must not be read as either of them."""

    grid = np.asarray([0.0, 1.0, 2.0])
    density = np.asarray([[0.0, 1.0, 0.0]])
    prediction = DensityPrediction(
        grid=grid, density=density, outcome="rt", mode=PredictionMode.FILTERED
    )

    assert prediction.density_at(np.asarray([0.25])) == pytest.approx(0.25)
    assert prediction.density_at(np.asarray([1.5])) == pytest.approx(0.5)


def test_values_outside_the_grid_evaluate_to_zero_rather_than_the_nearest_edge() -> None:
    grid = np.asarray([1.0, 2.0])
    prediction = DensityPrediction(
        grid=grid,
        density=np.asarray([[0.4, 0.4]]),
        outcome="rt",
        mode=PredictionMode.FILTERED,
    )

    assert prediction.density_at(np.asarray([0.5])) == pytest.approx(0.0)
    assert prediction.density_at(np.asarray([2.5])) == pytest.approx(0.0)
    assert prediction.observed_log_density(np.asarray([0.5])) == pytest.approx(LOG_DENSITY_FLOOR)


def test_observed_log_density_selects_the_category_each_row_actually_produced() -> None:
    prediction = defective()
    values = np.asarray([0.5, 0.5])

    scores = prediction.observed_log_density(values, [1, 0])

    expected = np.log([0.7 * exponential_density(2.0, 0.5), 0.7 * exponential_density(2.0, 0.5)])
    assert np.allclose(scores, expected, rtol=1e-6)


def test_a_defective_density_refuses_to_score_without_observed_categories() -> None:
    prediction = defective()

    with pytest.raises(ValueError, match="observed category of each row"):
        prediction.observed_log_density(np.asarray([0.5, 0.5]))
    with pytest.raises(ValueError, match="not one this prediction declares"):
        prediction.observed_log_density(np.asarray([0.5, 0.5]), [1, 7])


def test_take_keeps_the_grid_and_the_category_coordinate() -> None:
    subset = defective().take([1])

    assert subset.n_observations == 1
    assert subset.categories == (0, 1)
    assert np.allclose(subset.category_mass, [[0.7, 0.3]], atol=1e-4)


def test_a_truncated_grid_reports_the_mass_it_lost_rather_than_hiding_it() -> None:
    grid = np.linspace(0.0, 1.0, 1001)
    prediction = DensityPrediction(
        grid=grid,
        density=np.stack([exponential_density(1.0, grid)]),
        outcome="rt",
        mode=PredictionMode.FILTERED,
    )

    assert float(prediction.total_mass[0]) == pytest.approx(1 - np.exp(-1.0), abs=1e-5)


def test_an_impossible_density_is_rejected() -> None:
    grid = np.asarray([0.0, 1.0])
    with pytest.raises(ValueError, match="may not integrate above one"):
        DensityPrediction(
            grid=grid,
            density=np.asarray([[3.0, 3.0]]),
            outcome="rt",
            mode=PredictionMode.FILTERED,
        )
    with pytest.raises(ValueError, match="must carry positive mass"):
        DensityPrediction(
            grid=grid,
            density=np.asarray([[0.0, 0.0]]),
            outcome="rt",
            mode=PredictionMode.FILTERED,
        )
    with pytest.raises(ValueError, match="finite and non-negative"):
        DensityPrediction(
            grid=grid,
            density=np.asarray([[-0.1, 0.1]]),
            outcome="rt",
            mode=PredictionMode.FILTERED,
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        DensityPrediction(
            grid=np.asarray([1.0, 1.0]),
            density=np.asarray([[0.5, 0.5]]),
            outcome="rt",
            mode=PredictionMode.FILTERED,
        )
    with pytest.raises(ValueError, match="name the outcome column"):
        DensityPrediction(
            grid=grid, density=np.asarray([[0.5, 0.5]]), outcome="", mode=PredictionMode.FILTERED
        )


def test_category_declarations_must_match_the_density_shape() -> None:
    grid = np.asarray([0.0, 1.0])
    with pytest.raises(ValueError, match="at least two categories"):
        DensityPrediction(
            grid=grid,
            density=np.asarray([[[0.5, 0.5]]]),
            outcome="rt",
            mode=PredictionMode.FILTERED,
            categories=(0,),
        )
    with pytest.raises(ValueError, match="n_trials, n_categories, n_grid"):
        DensityPrediction(
            grid=grid,
            density=np.asarray([[0.5, 0.5]]),
            outcome="rt",
            mode=PredictionMode.FILTERED,
            categories=(0, 1),
        )
    with pytest.raises(ValueError, match="no category mass"):
        _ = unlabelled().category_mass


def test_a_density_prediction_is_immutable() -> None:
    prediction = unlabelled()

    assert not prediction.density.flags.writeable
    assert not prediction.grid.flags.writeable
    assert not prediction.density_at(np.asarray([1.0, 1.0, 1.0])).flags.writeable


def test_the_density_estimator_protocol_excludes_a_discrete_only_model() -> None:
    assert not isinstance(BiasOnly(), DensityBehaviourEstimator)


def test_a_density_is_one_of_the_three_shapes_a_consumer_reads_back() -> None:
    """The widening is the point: the union names it, so every consumer must handle it."""

    assert DensityPrediction in ModelPrediction.__args__
    assert CensoredDensityPrediction in ModelPrediction.__args__
    assert set(ModelPrediction.__args__) == {
        Prediction,
        CategoricalPrediction,
        DensityPrediction,
        CensoredDensityPrediction,
    }


def test_a_censored_density_uses_survival_only_for_censored_rows() -> None:
    base = unlabelled(n_trials=2)
    prediction = CensoredDensityPrediction(
        grid=base.grid,
        density=base.density,
        outcome=base.outcome,
        mode=base.mode,
        censoring_time=np.asarray([2.0, 3.0]),
        survival_probability=np.asarray([0.4, 0.2]),
        censoring_column="observation_limit",
    )
    observed = np.asarray([1.0, 3.0])
    censored = np.asarray([False, True])

    scores = prediction.observed_log_density(observed, censored=censored)

    assert scores[0] == pytest.approx(base.observed_log_density(observed)[0])
    assert scores[1] == pytest.approx(np.log(0.2))
    with pytest.raises(ValueError, match="requires the observed censored indicator"):
        prediction.observed_log_density(observed)
    subset = prediction.take([1])
    assert isinstance(subset, CensoredDensityPrediction)
    assert subset.survival_probability == pytest.approx([0.2])


# ---------------------------------------------------------------------------------------
# A density prediction through a fold and into a comparison.
#
# The type existing is not the same as the falsification layer being able to see it. These
# tests drive a real estimator whose ``predict`` returns a density through
# ``evaluate_splits`` and ``compare_models``, which is the whole path the widening was for.
# ---------------------------------------------------------------------------------------

GRID = np.linspace(0.0, 12.0, 1201)


class DefectiveExponential:
    """A two-boundary model whose prediction *is* a density, in twenty lines.

    Not a serious account of anything -- the response time is exponential and independent
    of the choice -- but it is a genuine :class:`DensityPrediction` producer with a
    closed-form fit, so the fold and comparison machinery is exercised without a solver in
    the way. ``behavio.foreign.pyddm`` is the real one.
    """

    model_name = "defective-exponential"
    scored_columns = ("choice", "response_time")
    required_task_columns = ()
    supported_prediction_modes = (PredictionMode.FILTERED,)
    categories = (0, 1)
    density_categories: tuple = (0, 1)
    density_outcome = "response_time"

    def __init__(self, *, label: str = "a", floor: float = 0.0) -> None:
        self.label = label
        self.floor = floor

    @property
    def signature(self) -> str:
        return f"defective-exponential[label={self.label};floor={self.floor}]"

    def fit(self, study: Study) -> FitResult:
        choices = np.asarray(study["choice"], dtype=np.float64)
        times = np.asarray(study["response_time"], dtype=np.float64)
        upper = float(np.clip(np.mean(choices), 0.05, 0.95) + self.floor)
        rate = float(1.0 / max(np.mean(times), 1e-3))
        return FitResult(
            model_name=self.model_name,
            model_signature=self.signature,
            parameter_names=("upper", "rate"),
            estimates=np.asarray([min(upper, 0.95), rate]),
            standard_errors=np.asarray([0.01, 0.01]),
            covariance=np.eye(2) * 1e-4,
            n_observations=len(study),
            diagnostics=FitDiagnostics(
                converged=True,
                optimizer="closed form",
                status=0,
                message="moment matched",
                n_iterations=None,
                objective=None,
                gradient_norm=None,
                hessian_condition=None,
                boundary_estimate=False,
            ),
        )

    def predict(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> DensityPrediction:
        return self.predict_density(study, fit, mode=mode)

    def predict_density(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> DensityPrediction:
        upper, rate = (float(value) for value in fit.estimates)
        shape = exponential_density(rate, GRID)
        row = np.stack([(1.0 - upper) * shape, upper * shape])
        return DensityPrediction(
            grid=GRID,
            density=np.repeat(row[None, :, :], len(study), axis=0),
            outcome="response_time",
            mode=PredictionMode(mode),
            categories=self.density_categories,
        )

    def pointwise_log_prob(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> np.ndarray:
        density = self.predict_density(study, fit, mode=mode)
        return density.observed_log_density(
            np.asarray(study["response_time"], dtype=np.float64),
            np.asarray(study["choice"], dtype=np.int64),
        )

    def outcome_codes(self, study: Study) -> np.ndarray:
        return np.asarray(study["choice"], dtype=np.int64)


class UnlabelledExponential(DefectiveExponential):
    """The same model with no discrete margin at all: a density over time and nothing else."""

    model_name = "unlabelled-exponential"
    scored_columns = ("response_time",)
    categories = ()

    @property
    def signature(self) -> str:
        return "unlabelled-exponential[v1]"

    def predict_density(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> DensityPrediction:
        rate = float(fit.estimates[1])
        shape = exponential_density(rate, GRID)
        return DensityPrediction(
            grid=GRID,
            density=np.repeat(shape[None, :], len(study), axis=0),
            outcome="response_time",
            mode=PredictionMode(mode),
        )

    def pointwise_log_prob(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> np.ndarray:
        density = self.predict_density(study, fit, mode=mode)
        return density.observed_log_density(np.asarray(study["response_time"], dtype=np.float64))


def timed_study(rows: int = 60, sessions: int = 3) -> Study:
    generator = np.random.default_rng(17)
    per_session = rows // sessions
    return Study(
        {
            "subject": ["m1"] * rows,
            "session": [f"s{1 + index // per_session}" for index in range(rows)],
            "session_order": [1 + index // per_session for index in range(rows)],
            "trial": [1 + index % per_session for index in range(rows)],
            "choice": generator.binomial(1, 0.6, size=rows).astype(np.int8),
            "response_time": generator.exponential(0.8, size=rows),
        }
    )


def test_a_density_prediction_survives_a_fold_and_is_scored() -> None:
    """The prediction reaching the fold is the density, sliced to the fold's scored rows."""

    study = timed_study()
    splits = forward_session_splits(study)

    evaluation = evaluate_splits(DefectiveExponential(), study, splits)

    assert evaluation.complete
    for fold in evaluation:
        prediction = fold.prediction
        assert isinstance(prediction, DensityPrediction)
        # Sliced, not passed through whole: only the fold's scored rows survive.
        assert prediction.n_observations == fold.split.test_indices.size
        assert prediction.grid.shape == GRID.shape
        assert prediction.categories == (0, 1)
        # The discrete half is retained beside it, which is what lets a probability
        # scoring rule read the margin later.
        assert fold.outcome_codes is not None
        assert fold.outcome_codes.shape == (prediction.n_observations,)
        # The score is the joint log density, not the choice log probability.
        assert np.isfinite(fold.total_log_probability)
        replayed = prediction.observed_log_density(
            np.asarray(study.take(fold.split.test_indices)["response_time"], dtype=np.float64),
            np.asarray(study.take(fold.split.test_indices)["choice"], dtype=np.int64),
        )
        assert np.allclose(fold.pointwise_log_probability, replayed)


def test_a_fold_refuses_a_defective_density_whose_categories_the_model_disowns() -> None:
    study = timed_study()

    class Mislabelled(DefectiveExponential):
        """Declares one category coordinate and tabulates the density over another."""

        categories = (0, 2)

        def outcome_codes(self, study: Study) -> np.ndarray:
            return np.zeros(len(study), dtype=np.int64)

    with pytest.raises(ValueError, match="category coordinates differ"):
        evaluate_splits(Mislabelled(), study, forward_session_splits(study))


def test_comparing_densities_scores_the_joint_log_loss_and_the_choice_margin_brier() -> None:
    """Two metrics, two different observations, and the difference is deliberate.

    The log loss is the joint density of choice *and* response time. The Brier score is a
    scoring rule for a probability, so it reads the density's discrete margin and nothing
    else -- exactly the number a choice-only competitor would earn on the same rows.
    """

    study = timed_study()
    splits = forward_session_splits(study)

    report = compare_models(
        {"plain": DefectiveExponential(), "shifted": DefectiveExponential(label="b", floor=0.1)},
        study,
        splits,
    )

    result = report.result_for("plain")
    assert result.audit_status.value == "pass"
    assert report.winner in {"plain", "shifted"}
    assert 0.0 <= result.pooled_brier_score <= 1.0

    # The Brier score is the choice margin's, computed here independently.
    expected: list[float] = []
    for evaluation in result.evaluations:
        prediction = evaluation.prediction
        assert isinstance(prediction, DensityPrediction)
        margin = prediction.choice_prediction().probability
        targets = np.zeros_like(margin)
        targets[np.arange(len(targets)), evaluation.outcome_codes] = 1.0
        expected.extend(0.5 * np.sum((margin - targets) ** 2, axis=1))
    assert result.pooled_brier_score == pytest.approx(float(np.mean(expected)))

    # The log loss is *not* the choice margin's: it carries the latency half too.
    choice_only = -float(
        np.mean(
            np.concatenate(
                [
                    np.log(
                        evaluation.prediction.choice_prediction().probability[  # type: ignore[union-attr]
                            np.arange(len(evaluation.outcome_codes)),  # type: ignore[arg-type]
                            evaluation.outcome_codes,
                        ]
                    )
                    for evaluation in result.evaluations
                ]
            )
        )
    )
    assert result.pooled_log_loss != pytest.approx(choice_only)


def test_a_density_with_no_discrete_margin_says_the_brier_score_does_not_apply() -> None:
    """No probability, no Brier score, and no number invented to fill the column."""

    study = timed_study()
    splits = forward_session_splits(study)
    model = UnlabelledExponential()

    # It evaluates and scores perfectly well: the log score applies to a density.
    evaluation = evaluate_splits(model, study, splits)
    assert evaluation.complete
    for fold in evaluation:
        assert isinstance(fold.prediction, DensityPrediction)
        assert not fold.prediction.is_defective
        assert fold.outcome_codes is None
        assert np.isfinite(fold.total_log_probability)

    with pytest.raises(UnscoreableByBrier, match="not for a density"):
        compare_models({"unlabelled": model}, study, splits, outcome_column="response_time")


def test_category_codes_derive_a_folds_outcome_codes_from_the_density_itself() -> None:
    prediction = defective()

    codes = prediction.category_codes([1, 0])

    assert codes.tolist() == [1, 0]
    assert not codes.flags.writeable
    with pytest.raises(ValueError, match="not one this prediction declares"):
        prediction.category_codes([1, 7])
    with pytest.raises(ValueError, match="no categories to code against"):
        unlabelled().category_codes([0, 1, 0])
