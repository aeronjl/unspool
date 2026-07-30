"""The prediction type for a continuous outcome."""

import numpy as np
import pytest

from behavio.adapters import (
    LOG_DENSITY_FLOOR,
    DensityBehaviourEstimator,
    DensityPrediction,
)
from behavio.contracts import PredictionMode
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
