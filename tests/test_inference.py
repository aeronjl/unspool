import json
import math

import numpy as np
import pytest

from unspool import (
    InferenceError,
    ObjectiveTarget,
    OptimizationBackend,
    OptimizationProblem,
    ParameterSpace,
    ParameterSpec,
    ParameterTransform,
    PriorMeasure,
    PriorSpec,
    ScipyMultistart,
)


def quadratic_space() -> ParameterSpace:
    return ParameterSpace(
        (
            ParameterSpec(
                "x",
                bounds=(-10.0, 10.0),
                plausible_bounds=(-5.0, 5.0),
                prior=PriorSpec.normal(2.0, 1.0),
            ),
            ParameterSpec(
                "scale",
                optimizer_name="scale_log",
                transform=ParameterTransform.LOG,
                bounds=(0.0, None),
                plausible_bounds=(0.1, 10.0),
                optimizer_bounds=(-5.0, 5.0),
                prior=PriorSpec.half_normal(2.0),
            ),
        )
    )


def quadratic_objective(vector):
    target = np.asarray([1.5, math.log(3.0)])
    residual = vector - target
    return float(0.5 * residual @ residual), residual


def test_scipy_multistart_retains_every_attempt_and_selects_deterministically() -> None:
    problem = OptimizationProblem(
        parameter_space=quadratic_space(),
        objective=quadratic_objective,
        starts=(np.array([-4.0, -1.0]), np.array([4.0, 2.0]), np.array([0.0, 0.0])),
        has_gradient=True,
    )
    backend = ScipyMultistart(max_iterations=100)

    run = backend.run(problem)

    assert isinstance(backend, OptimizationBackend)
    assert run.backend == "scipy.optimize.minimize/L-BFGS-B"
    assert run.backend_config == {
        "method": "L-BFGS-B",
        "max_iterations": 100,
        "function_tolerance": 1e-9,
        "gradient_tolerance": 1e-9,
    }
    assert len(run.attempts) == 3
    assert tuple(attempt.index for attempt in run.attempts) == (0, 1, 2)
    assert all(attempt.n_evaluations > 0 for attempt in run.attempts)
    assert all(attempt.message for attempt in run.attempts)
    assert run.selected is not None
    np.testing.assert_allclose(run.selected.estimate, [1.5, math.log(3.0)], atol=1e-7)
    assert run.selected.objective == pytest.approx(0.0, abs=1e-12)
    assert not run.selected.estimate.flags.writeable
    assert run.problem["parameter_space_fingerprint"] == quadratic_space().fingerprint
    with pytest.raises(TypeError):
        run.problem["target"] = "changed"
    json.dumps(run.to_dict(), allow_nan=False)


def test_unsuccessful_nonfinite_attempts_remain_visible_and_are_not_selected() -> None:
    def partly_finite(vector):
        if vector[0] < 0:
            return np.inf, np.zeros(2)
        residual = vector - np.asarray([1.0, 0.0])
        return float(residual @ residual), 2.0 * residual

    problem = OptimizationProblem(
        parameter_space=quadratic_space(),
        objective=partly_finite,
        starts=(np.array([-2.0, 0.0]), np.array([2.0, 0.0])),
        has_gradient=True,
    )

    run = ScipyMultistart().run(problem)

    assert not run.attempts[0].finite
    assert not run.attempts[0].converged
    assert run.attempts[0].to_dict()["objective"] is None
    assert run.selected_index == 1
    assert run.selected is run.attempts[1]


def test_all_nonfinite_attempts_produce_an_inspectable_unselected_run() -> None:
    problem = OptimizationProblem(
        parameter_space=quadratic_space(),
        objective=lambda vector: (np.inf, np.zeros_like(vector)),
        starts=(np.zeros(2), np.ones(2)),
        has_gradient=True,
    )

    run = ScipyMultistart().run(problem)

    assert run.selected is None
    assert run.selected_index is None
    assert not run.any_converged
    assert len(run.attempts) == 2


def test_map_targets_apply_declared_priors_and_coordinate_measure_explicitly() -> None:
    space = quadratic_space()
    vector = np.asarray([0.5, math.log(1.5)])
    likelihood_value, likelihood_gradient = quadratic_objective(vector)
    natural_problem = OptimizationProblem(
        parameter_space=space,
        objective=quadratic_objective,
        starts=(vector,),
        has_gradient=True,
        target=ObjectiveTarget.MAXIMUM_A_POSTERIORI,
    )
    optimizer_problem = OptimizationProblem(
        parameter_space=space,
        objective=quadratic_objective,
        starts=(vector,),
        has_gradient=True,
        target=ObjectiveTarget.MAXIMUM_A_POSTERIORI,
        prior_measure=PriorMeasure.OPTIMIZER,
    )

    natural_value, natural_gradient = natural_problem.evaluate(vector)
    optimizer_value, optimizer_gradient = optimizer_problem.evaluate(vector)
    decoded = space.decode(vector)
    expected_natural = likelihood_value - space.log_prior(decoded, require_all=True)

    assert natural_value == pytest.approx(expected_natural)
    np.testing.assert_allclose(
        natural_gradient,
        likelihood_gradient - space.grad_log_prior_optimizer(vector, require_all=True),
    )
    assert optimizer_value == pytest.approx(
        natural_value - space.log_abs_det_inverse_jacobian(vector)
    )
    np.testing.assert_allclose(
        optimizer_gradient,
        natural_gradient - space.grad_log_abs_det_inverse_jacobian(vector),
    )


@pytest.mark.parametrize("measure", [PriorMeasure.NATURAL, PriorMeasure.OPTIMIZER])
def test_map_gradient_matches_complete_objective_finite_difference(measure) -> None:
    problem = OptimizationProblem(
        parameter_space=quadratic_space(),
        objective=quadratic_objective,
        starts=(np.asarray([0.2, 0.3]),),
        has_gradient=True,
        target=ObjectiveTarget.MAXIMUM_A_POSTERIORI,
        prior_measure=measure,
    )
    vector = np.asarray([0.4, -0.2])
    _, analytic = problem.evaluate(vector)
    numeric = np.empty(2)
    for index in range(2):
        positive = vector.copy()
        negative = vector.copy()
        positive[index] += 1e-6
        negative[index] -= 1e-6
        positive_value, _ = problem.evaluate(positive)
        negative_value, _ = problem.evaluate(negative)
        numeric[index] = (positive_value - negative_value) / 2e-6

    np.testing.assert_allclose(analytic, numeric, atol=1e-6, rtol=1e-6)


def test_value_only_objectives_use_backend_numerical_gradients() -> None:
    problem = OptimizationProblem(
        parameter_space=ParameterSpace((ParameterSpec("x", bounds=(-5.0, 5.0)),)),
        objective=lambda vector: float((vector[0] - 2.0) ** 2),
        starts=(np.asarray([-3.0]), np.asarray([3.0])),
        has_gradient=False,
    )

    run = ScipyMultistart().run(problem)

    assert run.selected is not None
    assert run.selected.estimate[0] == pytest.approx(2.0, abs=1e-5)
    assert run.selected.n_evaluations > 0


def test_problem_rejects_semantic_and_objective_contract_drift() -> None:
    space = quadratic_space()
    with pytest.raises(InferenceError, match="outside optimizer bounds"):
        OptimizationProblem(space, quadratic_objective, ((0.0, 6.0),), True)
    with pytest.raises(InferenceError, match="only meaningful"):
        OptimizationProblem(
            space,
            quadratic_objective,
            ((0.0, 0.0),),
            True,
            prior_measure=PriorMeasure.OPTIMIZER,
        )
    with pytest.raises(InferenceError, match="has no prior"):
        OptimizationProblem(
            ParameterSpace((ParameterSpec("x"),)),
            lambda vector: (float(vector[0] ** 2), 2.0 * vector),
            ((0.0,),),
            True,
            target=ObjectiveTarget.MAXIMUM_A_POSTERIORI,
        )

    bad_gradient = OptimizationProblem(
        space,
        lambda vector: float(vector @ vector),
        ((0.0, 0.0),),
        True,
    )
    with pytest.raises(InferenceError, match="must return"):
        bad_gradient.evaluate((0.0, 0.0))
    with pytest.raises(InferenceError, match="only L-BFGS-B"):
        ScipyMultistart(method="Nelder-Mead")
