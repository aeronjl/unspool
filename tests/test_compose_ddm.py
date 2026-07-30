"""The drift-diffusion families as compositions, and the proof they are what they replaced.

``tests/fixtures/pre_combinator_ddm_reference.json`` holds fits produced by the deleted
``SmoothWienerDriftDiffusion`` and ``HierarchicalSmoothWienerDriftDiffusion`` classes at
commit ``88ab198``, the commit immediately before they were replaced. Every model
configuration, design and seed in that fixture is reconstructed here from the combinators,
so the comparison is old-versus-new and not new-versus-itself.

Unlike the generalized linear case, this agreement is **not** bit-for-bit in the fitted
estimates, and that is a fact about the arithmetic rather than about the model. The deleted
classes evaluated a row's drift as ``sum(features * trial_values, axis=1)`` after
interpolating each parameter's path; a composed fit evaluates the same number as one
contraction of a ``(rows, cells, parameters)`` design against the flat coordinate. The two
associate the same products differently.
:func:`test_the_composed_objective_is_the_deleted_objective_to_the_last_bit` measures the
consequence directly: at the deleted class's own optimum the two objectives agree to zero
and one unit in the last place respectively. What the fixture comparisons below then bound
is the *optimizer's* amplification of that -- a bounded quasi-Newton search on a
finite-difference objective, which follows a slightly different path and stops at a
slightly different point. Every quantity that is not downstream of the optimizer --
simulated choices, simulated response times, drawn random effects -- is exactly equal.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from behavio import (
    Study,
    UniformResponseGuess,
    WienerDriftDiffusion,
    cohort_forward_session_splits,
    compare_models,
    evaluate_splits,
    forward_session_splits,
    run_parameter_recovery,
)
from behavio.compose import (
    HierarchicalFitResult,
    HierarchicalModel,
    MixtureModel,
    SmoothModel,
    UnseenGroupPrediction,
    hierarchical,
    mix,
    smooth,
)
from behavio.contracts.compose import PenalisedLinearEstimator, group_blocks
from behavio.evaluate import leave_one_subject_out_splits
from behavio.models import BehaviourModel, DriftDiffusionFitResult, ModelDataError
from behavio.models._kernels.wiener import WienerLikelihood, wiener_cell_bytes, wiener_log_density

REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures" / "pre_combinator_ddm_reference.json").read_text(
        encoding="utf-8"
    )
)
KNOTS = tuple(REFERENCE["knots"])
PATHS = REFERENCE["paths"]
VARYING = tuple(REFERENCE["varying_parameters"])
SMOOTHNESS = float(REFERENCE["smoothness"])


# --------------------------------------------------------------------------------------
# The exact designs and models the reference fixture was generated on
# --------------------------------------------------------------------------------------


def design(
    *, n_subjects: int, n_sessions: int = 2, n_trials: int = 40, seed: int = 20260729
) -> Study:
    generator = np.random.default_rng(seed)
    subject: list[str] = []
    session: list[str] = []
    trial: list[int] = []
    session_order: list[int] = []
    stimulus: list[float] = []
    for index in range(n_subjects):
        for order in range(n_sessions):
            subject.extend([f"mouse-{index}"] * n_trials)
            session.extend([f"session-{order}"] * n_trials)
            trial.extend(range(n_trials))
            session_order.extend([2 * order] * n_trials)
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


def base_ddm(*, n_restarts: int = 2, **changes: Any) -> WienerDriftDiffusion:
    arguments: dict[str, Any] = {
        "predictors": ("stimulus",),
        "n_restarts": n_restarts,
        "max_iterations": 300,
        "simulation_time_step": 0.001,
    }
    arguments.update(changes)
    return WienerDriftDiffusion(**arguments)


def paths_model(*, n_restarts: int = 2, **changes: Any) -> SmoothModel:
    return smooth(
        base_ddm(n_restarts=n_restarts),
        over="session_order",
        knots=KNOTS,
        smoothness=SMOOTHNESS,
        parameters=VARYING,
        group_smoothness=SMOOTHNESS,
        **changes,
    )


def pooled_model(*, n_restarts: int = 2, **changes: Any) -> HierarchicalModel:
    return hierarchical(
        paths_model(n_restarts=n_restarts),
        over="subject",
        parameters=VARYING,
        **changes,
    )


def close(observed: Any, expected: Any, *, tolerance: float) -> None:
    """Assert agreement with a reference the deleted classes produced."""

    observed_array = np.asarray(observed, dtype=np.float64)
    expected_array = np.asarray(expected, dtype=np.float64)
    scale = max(1.0, float(np.max(np.abs(expected_array))))
    np.testing.assert_allclose(
        observed_array, expected_array, rtol=tolerance, atol=tolerance * scale
    )


def identical(observed: Any, expected: Any) -> None:
    """Assert bit-for-bit equality, for quantities no optimizer stands between."""

    np.testing.assert_array_equal(
        np.asarray(observed, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    )


# --------------------------------------------------------------------------------------
# The acceptance test: the combinators reproduce the classes they replaced
# --------------------------------------------------------------------------------------


def test_the_composed_objective_is_the_deleted_objective_to_the_last_bit() -> None:
    """The reassociation, measured where the optimizer cannot hide or amplify it.

    Both objectives are evaluated at the *deleted class's own optimum*, which is a point
    neither implementation chose in agreement with the other. The smooth case agrees
    exactly; the hierarchical case, which additionally re-associates a block-diagonal
    quadratic form into one joint one, agrees to a single unit in the last place.
    """

    paths = paths_model()
    study = paths.simulate(design(n_subjects=1), paths.parameters_from_paths(PATHS), seed=11)
    reference = REFERENCE["reference"]["smooth"]
    point = np.asarray(reference["estimates"], dtype=np.float64)
    predictor = np.einsum("rcp,p->rc", paths.design_matrix(study), point)
    smooth_objective = -float(
        np.sum(paths.likelihood.log_density(predictor, paths.outcomes(study)))
    )
    smooth_objective += 0.5 * float(point @ paths.penalty_matrix() @ point)
    assert smooth_objective == reference["objective"]

    model = pooled_model(scale=0.18)
    joint = model.simulate_with_effects(
        design(n_subjects=3), model.parameters_from_paths(PATHS), seed=13
    ).study
    reference = REFERENCE["reference"]["hierarchical_smooth"]
    population = np.asarray(reference["estimates"], dtype=np.float64)
    deviations = np.asarray(reference["subject_deviations"], dtype=np.float64).reshape(3, -1)
    blocks = group_blocks(joint, "subject")
    columns = model.varying_effects.columns(model.parameter_names)
    rows = np.tile(population, (len(joint), 1))
    rows[:, columns] += deviations[blocks.row_block]
    predictor = np.einsum("rcp,rp->rc", paths.design_matrix(joint), rows)
    pooled_objective = -float(
        np.sum(paths.likelihood.log_density(predictor, paths.outcomes(joint)))
    )
    pooled_objective += 0.5 * float(population @ paths.penalty_matrix() @ population)
    group_penalty = paths.group_penalty(columns, np.full(len(columns), 0.18))
    for row in deviations:
        pooled_objective += 0.5 * float(row @ group_penalty @ row)
    assert abs(pooled_objective - reference["objective"]) <= np.spacing(reference["objective"])


def test_smooth_reproduces_the_hand_written_smooth_ddm() -> None:
    reference = REFERENCE["reference"]["smooth"]
    model = paths_model()

    assert model.parameter_names == tuple(reference["parameter_names"])
    study = model.simulate(design(n_subjects=1), model.parameters_from_paths(PATHS), seed=11)
    fit = model.fit(study)

    identical(study["choice"], reference["choice"])
    identical(study["response_time"], reference["response_time"])
    close(fit.estimates, reference["estimates"], tolerance=1e-5)
    close(fit.standard_errors, reference["standard_errors"], tolerance=1e-5)
    close(fit.diagnostics.objective, reference["objective"], tolerance=1e-9)
    close(
        float(np.sum(model.pointwise_log_prob(study, fit))),
        reference["pointwise_log_prob_sum"],
        tolerance=1e-6,
    )
    close(model.predict(study, fit).probability[:8], reference["probability"], tolerance=1e-5)
    close(
        model.coefficient_trajectory(fit, times=(0.0, 0.5, 1.5, 2.0)).values,
        reference["trajectory"],
        tolerance=1e-5,
    )
    assert isinstance(fit, DriftDiffusionFitResult)
    assert fit.selected_restart == reference["selected_restart"]
    assert fit.likelihood_floor_count == reference["likelihood_floor_count"]
    assert fit.diagnostics.boundary_estimate == reference["boundary_estimate"]


def test_hierarchical_smooth_reproduces_the_hand_written_sibling() -> None:
    reference = REFERENCE["reference"]["hierarchical_smooth"]
    model = pooled_model(scale=0.18)

    assert model.parameter_names == tuple(reference["parameter_names"])
    simulation = model.simulate_with_effects(
        design(n_subjects=3), model.parameters_from_paths(PATHS), seed=13
    )
    study = simulation.study
    fit = model.fit(study)

    identical(study["choice"], reference["choice"])
    identical(study["response_time"], reference["response_time"])
    identical(
        simulation.group_deviations,
        np.asarray(reference["simulated_deviations"]).reshape(3, -1),
    )
    close(fit.estimates, reference["estimates"], tolerance=1e-3)
    close(fit.standard_errors, reference["standard_errors"], tolerance=1e-4)
    close(fit.covariance, reference["covariance"], tolerance=1e-4)
    close(
        fit.group_deviations,
        np.asarray(reference["subject_deviations"]).reshape(3, -1),
        tolerance=1e-3,
    )
    close(
        fit.group_standard_errors,
        np.asarray(reference["subject_standard_errors"]).reshape(3, -1),
        tolerance=1e-4,
    )
    close(fit.diagnostics.objective, reference["objective"], tolerance=1e-6)
    close(
        float(np.sum(model.pointwise_log_prob(study, fit))),
        reference["pointwise_log_prob_sum"],
        tolerance=1e-4,
    )
    close(model.predict(study, fit).probability[:8], reference["probability"], tolerance=1e-3)
    close(
        model.coefficient_trajectory(fit, times=(0.0, 1.0)).values,
        reference["population_trajectory"],
        tolerance=1e-3,
    )
    close(
        model.group_trajectory(fit, "mouse-1", times=(0.0, 1.0)).values,
        reference["subject_trajectory"],
        tolerance=1e-3,
    )


def test_laplace_em_reproduces_the_hand_written_per_parameter_scale_estimate() -> None:
    reference = REFERENCE["reference"]["hierarchical_smooth_estimated_scales"]
    model = pooled_model(
        n_restarts=1,
        parameter_scales={"drift.stimulus": 0.22, "boundary": 0.08},
        estimate_scale=True,
        scale_estimator="laplace-em",
        scale_bounds=(0.04, 0.5),
        scale_max_iterations=6,
        scale_tolerance=0.05,
    )
    study = model.simulate(design(n_subjects=4), model.parameters_from_paths(PATHS), seed=17)

    fit = model.fit(study)

    assert fit.scale_parameters == VARYING
    assert fit.scale_estimation_policy == "laplace-em"
    assert fit.scale_uncertainty_policy == "louis-observed-information"
    assert fit.scale_estimation_iterations == reference["scale_estimation_iterations"]
    assert fit.scale_estimation_converged == reference["scale_estimation_converged"]
    close(fit.scales, reference["subject_parameter_scales"], tolerance=1e-2)
    close(fit.estimates, reference["estimates"], tolerance=1e-3)
    close(fit.scale_standard_errors, reference["subject_scale_standard_errors"], tolerance=1e-2)
    close(
        fit.scale_local_standard_errors,
        reference["subject_scale_local_standard_errors"],
        tolerance=1e-2,
    )
    close(fit.scale_covariance, reference["subject_scale_covariance"], tolerance=1e-2)
    assert np.all(fit.scale_standard_errors >= fit.scale_local_standard_errors)


def test_unseen_group_prediction_reproduces_the_hand_written_monte_carlo() -> None:
    reference = REFERENCE["reference"]["unseen_subjects"]
    model = pooled_model(n_restarts=1, scale=0.18)
    training = model.simulate(design(n_subjects=3), model.parameters_from_paths(PATHS), seed=19)
    fit = model.fit(training)
    unseen_design = design(n_subjects=2, n_trials=20, seed=20260730)
    columns = {name: unseen_design[name] for name in unseen_design.columns}
    columns["subject"] = np.asarray([f"new-{value}" for value in unseen_design["subject"]])
    unseen = model.simulate(Study(columns), model.parameters_from_paths(PATHS), seed=23)

    predicted = model.predict_new_groups(unseen, fit, n_draws=32, seed=29)

    assert isinstance(predicted, UnseenGroupPrediction)
    assert predicted.groups == ("new-mouse-0", "new-mouse-1")
    assert predicted.n_draws == 32
    identical(
        predicted.random_effect_draws[0],
        np.asarray(reference["random_effect_draws_head"]).reshape(2, -1),
    )
    close(fit.estimates, reference["estimates"], tolerance=1e-3)
    close(predicted.probability[:12], reference["probability"], tolerance=1e-3)
    close(
        predicted.pointwise_marginal_log_probability[:12],
        reference["pointwise_marginal_log_probability"],
        tolerance=1e-3,
    )
    close(
        predicted.group_joint_log_probability,
        reference["subject_joint_log_probability"],
        tolerance=1e-3,
    )
    close(predicted.group_effective_draws, reference["subject_effective_draws"], tolerance=1e-3)
    with pytest.raises(ValueError, match="entirely unseen"):
        model.predict_new_groups(training.take([0]), fit, n_draws=8, seed=1)


# --------------------------------------------------------------------------------------
# Hierarchy without smoothness: one of the thirteen cells that never existed
# --------------------------------------------------------------------------------------


def flat_truth() -> dict[str, float]:
    return {
        "drift.intercept": 0.1,
        "drift.stimulus": 1.2,
        "boundary": 1.3,
        "starting_bias": 0.48,
        "nondecision_time": 0.25,
    }


def test_hierarchy_without_smoothness_names_the_wrapped_coordinate() -> None:
    model = hierarchical(base_ddm(), over="subject", parameters=("drift.stimulus", "boundary"))

    assert isinstance(model, BehaviourModel)
    assert model.model_name == "hierarchical-wiener-drift-diffusion"
    assert model.parameter_names == (
        "drift.intercept",
        "drift.stimulus",
        "boundary",
        "starting_bias",
        "nondecision_time",
    )
    assert model.varying_parameters == ("drift.stimulus", "boundary")
    assert model.scored_columns == ("choice", "response_time")
    assert "varying=drift.stimulus,boundary" in model.signature


def test_hierarchy_without_smoothness_fits_and_recovers() -> None:
    """The composition that had no hand-written sibling, and did not need one written."""

    model = hierarchical(
        base_ddm(), over="subject", parameters=("drift.stimulus", "boundary"), scale=0.2
    )
    study_design = design(n_subjects=4, n_trials=90)
    simulation = model.simulate_with_effects(study_design, flat_truth(), seed=7)

    fit = model.fit(simulation.study)

    assert isinstance(fit, HierarchicalFitResult)
    assert fit.diagnostics.converged
    assert fit.groups == tuple(f"mouse-{index}" for index in range(4))
    assert fit.group_deviations.shape == (4, 2)
    assert fit.group_standard_errors.shape == (4, 2)
    assert np.all(np.isfinite(fit.covariance))
    recovered = dict(zip(model.parameter_names, fit.estimates.tolist(), strict=True))
    assert recovered["drift.stimulus"] == pytest.approx(1.2, abs=0.35)
    assert recovered["boundary"] == pytest.approx(1.3, abs=0.25)
    assert recovered["nondecision_time"] == pytest.approx(0.25, abs=0.05)
    # Shrunken deviations track the realised ones they were shrunk towards.
    correlation = np.corrcoef(fit.group_deviations.ravel(), simulation.group_deviations.ravel())[
        0, 1
    ]
    assert correlation > 0.5


def test_hierarchy_without_smoothness_flows_through_every_consumer() -> None:
    model = hierarchical(
        base_ddm(), over="subject", parameters=("drift.stimulus", "boundary"), scale=0.2
    )
    study_design = design(n_subjects=3, n_trials=60)
    study = model.simulate(study_design, flat_truth(), seed=5)
    splits = leave_one_subject_out_splits(study)

    evaluations = evaluate_splits(model, study, splits)
    comparison = compare_models({"pooled": model, "flat": base_ddm()}, study, splits)
    recovery = run_parameter_recovery(model, study_design, [flat_truth()], seed=11)
    description = model.describe()

    assert len(evaluations) == 3
    for evaluation in evaluations:
        assert isinstance(evaluation.fit, HierarchicalFitResult)
        assert not evaluation.fit.group_was_fitted(evaluation.split.held_out_group)
        assert np.all(np.isfinite(evaluation.pointwise_log_probability))
    assert {"pooled", "flat"} == set(comparison.model_order)
    assert recovery.n_runs == 1
    assert recovery.parameter_names == model.parameter_names
    assert recovery.audits[0].status.value in {"pass", "warning"}
    assert description.model_name == "hierarchical-wiener-drift-diffusion"
    assert description.scored_columns == ("choice", "response_time")
    assert description.priors == (
        "deviations by subject: drift.stimulus ~ Normal(0, 0.2), boundary ~ Normal(0, 0.2)",
    )


def test_a_smooth_ddm_flows_through_every_consumer() -> None:
    model = paths_model()
    study_design = design(n_subjects=1, n_trials=80)
    study = model.simulate(study_design, model.parameters_from_paths(PATHS), seed=3)
    splits = forward_session_splits(study, min_train_sessions=1)

    evaluations = evaluate_splits(model, study, splits)
    comparison = compare_models({"smooth": model, "static": base_ddm()}, study, splits)
    recovery = run_parameter_recovery(
        model, study_design, [model.parameters_from_paths(PATHS)], seed=4
    )
    description = model.describe()

    assert len(evaluations) == 1
    assert np.all(np.isfinite(evaluations[0].pointwise_log_probability))
    assert {"smooth", "static"} == set(comparison.model_order)
    assert recovery.parameter_names == model.parameter_names
    assert description.model_name == "smooth-wiener-drift-diffusion"
    assert description.clock == "session_order"


def test_a_hierarchical_smooth_ddm_flows_through_every_consumer() -> None:
    model = pooled_model(scale=0.18)
    study = model.simulate(
        design(n_subjects=4, n_trials=50), model.parameters_from_paths(PATHS), seed=6
    )

    population = evaluate_splits(model, study, leave_one_subject_out_splits(study))
    sessions = evaluate_splits(
        model, study, cohort_forward_session_splits(study, min_train_sessions=1)
    )

    assert len(population) == 4
    for evaluation in population:
        assert not evaluation.fit.group_was_fitted(evaluation.split.held_out_group)
        assert np.all(np.isfinite(evaluation.pointwise_log_probability))
    assert len(sessions) == 1
    assert np.all(np.isfinite(sessions[0].pointwise_log_probability))


# --------------------------------------------------------------------------------------
# What the contract had to grow: joint outcomes, a box, cells
# --------------------------------------------------------------------------------------


def test_the_wiener_family_declares_joint_outcomes_and_four_predictor_cells() -> None:
    model = base_ddm()
    study = model.simulate(design(n_subjects=1, n_trials=20), flat_truth(), seed=2)

    assert isinstance(model, PenalisedLinearEstimator)
    assert model.predictor_cells == ("drift", "boundary", "starting_bias", "nondecision_time")
    assert model.outcome_channels == ("choice", "response_time")
    outcomes = model.outcomes(study)
    assert outcomes.shape == (len(study), 2)
    identical(outcomes[:, 0], np.asarray(study["choice"], dtype=np.float64))
    identical(outcomes[:, 1], np.asarray(study["response_time"], dtype=np.float64))
    matrix = model.design_matrix(study)
    assert matrix.shape == (len(study), 4, 5)
    identical(matrix[:, 1, 2], np.ones(len(study)))
    identical(matrix[:, 0, 2], np.zeros(len(study)))
    box = model.coordinate_box(study)
    assert box.shape == (5, 2)
    assert tuple(box[2]) == model.boundary_bounds
    assert tuple(box[3]) == model.starting_bias_bounds
    assert box[4, 1] < float(np.min(study["response_time"]))
    assert len(model.initial_points(study)) == model.n_restarts


def test_the_wiener_likelihood_differentiates_its_own_density() -> None:
    """The numerical gradient and curvature the contract asks for, against the density."""

    likelihood = WienerLikelihood(density_terms=12)
    cells = np.asarray([[0.8, 1.2, 0.45, 0.2], [-0.4, 1.6, 0.6, 0.15]])
    outcomes = np.asarray([[1.0, 0.7], [0.0, 0.55]])

    value, gradient = likelihood.value_and_gradient(cells, outcomes)
    curvature = likelihood.curvature(cells, outcomes)

    assert value == pytest.approx(-float(np.sum(likelihood.log_density(cells, outcomes))))
    step = 1e-5
    for row in range(cells.shape[0]):
        for cell in range(cells.shape[1]):
            forward = cells.copy()
            backward = cells.copy()
            forward[row, cell] += step
            backward[row, cell] -= step
            expected = (
                -likelihood.log_density(forward, outcomes)[row]
                + likelihood.log_density(backward, outcomes)[row]
            ) / (2.0 * step)
            assert gradient[row, cell] == pytest.approx(expected, rel=1e-3, abs=1e-4)
    assert curvature.shape == (2, 4, 4)
    np.testing.assert_allclose(curvature, np.swapaxes(curvature, 1, 2))


def test_a_dense_wiener_design_reports_the_memory_it_will_cost() -> None:
    """The measurement that decides whether a blocked representation is needed.

    Twenty animals of five thousand trials, four cells, and a joint coordinate of one
    hundred and seventy parameters is the scale the audit named, and the answer is that a
    dense design is over half a gigabyte. Nothing in the test suite or the benchmarks is
    within two orders of magnitude of it, so it is measured rather than optimised.
    """

    assert wiener_cell_bytes(20 * 5_000, 4, 170) == 544_000_000
    model = pooled_model()
    study = model.simulate(
        design(n_subjects=4, n_trials=50), model.parameters_from_paths(PATHS), seed=6
    )
    blocks = group_blocks(study, "subject")
    columns = model.varying_effects.columns(model.parameter_names)
    joint_parameters = len(model.parameter_names) + blocks.n_groups * len(columns)
    assert wiener_cell_bytes(len(study), 4, joint_parameters) < 2_000_000


def test_smoothing_a_wiener_parameter_is_a_declaration_and_not_a_default() -> None:
    everything = smooth(base_ddm(), over="session_order", knots=KNOTS)
    selected = paths_model()

    assert everything.parameter_names == tuple(
        f"{name}[session_order={knot:g}]" for name in everything.coefficient_names for knot in KNOTS
    )
    assert everything.smooths_every_parameter
    assert not selected.smooths_every_parameter
    assert "varying=" not in everything.signature
    assert "varying=drift.stimulus,boundary" in selected.signature
    assert selected.penalty_matrix().shape == (7, 7)
    # A stationary parameter keeps one coordinate and gets no roughness penalty.
    stationary = [
        index for index, (_name, _start, width) in enumerate(selected.layout) if width == 1
    ]
    for index in stationary:
        _name, start, _width = selected.layout[index]
        assert selected.penalty_matrix()[start, start] == 0.0


def test_fixed_knots_cover_the_clock_and_subject_alignment_stays_explicit() -> None:
    model = paths_model()
    parameters = model.parameters_from_paths(PATHS)

    with pytest.raises(ModelDataError, match="fixed knot range"):
        model.simulate(design(n_subjects=1, n_sessions=3), parameters, seed=1)

    with pytest.raises(ModelDataError, match="subject-specific by default"):
        model.simulate(design(n_subjects=2, n_trials=10), parameters, seed=1)

    shared = paths_model(shared_trajectory=True)
    aligned = shared.simulate(design(n_subjects=2, n_trials=10), parameters, seed=1)
    assert len(aligned.subjects) == 2


def test_changing_paths_beat_a_static_ddm_on_future_sessions() -> None:
    model = smooth(
        base_ddm(),
        over="session_order",
        knots=(0.0, 1.0, 2.0, 3.0, 4.0),
        smoothness=10.0,
        parameters=("drift.stimulus", "boundary"),
    )
    study_design = design(n_subjects=1, n_sessions=5, n_trials=100)
    study_design = Study(
        {
            **{name: study_design[name] for name in study_design.columns},
            "session_order": np.asarray(study_design["session_order"]) // 2,
        }
    )
    truth = model.parameters_from_paths(
        {
            "drift.intercept": 0.1,
            "drift.stimulus": np.linspace(0.4, 1.6, 5),
            "boundary": np.linspace(1.5, 1.0, 5),
            "starting_bias": 0.48,
            "nondecision_time": 0.25,
        }
    )
    study = model.simulate(study_design, truth, seed=39)
    splits = forward_session_splits(study, min_train_sessions=3)
    static = base_ddm()

    smooth_loss = -np.mean(
        np.concatenate(
            [item.pointwise_log_probability for item in evaluate_splits(model, study, splits)]
        )
    )
    static_loss = -np.mean(
        np.concatenate(
            [item.pointwise_log_probability for item in evaluate_splits(static, study, splits)]
        )
    )

    assert smooth_loss < static_loss


def test_unobserved_future_knots_carry_forward_without_outcome_leakage() -> None:
    model = smooth(
        base_ddm(),
        over="session_order",
        knots=(0.0, 1.0, 2.0, 3.0),
        smoothness=10.0,
        parameters=("drift.stimulus", "boundary"),
    )
    study_design = design(n_subjects=1, n_sessions=3, n_trials=80)
    study_design = Study(
        {
            **{name: study_design[name] for name in study_design.columns},
            "session_order": np.asarray(study_design["session_order"]) // 2,
        }
    )
    truth = model.parameters_from_paths(
        {
            "drift.intercept": 0.1,
            "drift.stimulus": [0.4, 0.8, 1.2, 1.6],
            "boundary": [1.5, 1.35, 1.2, 1.05],
            "starting_bias": 0.48,
            "nondecision_time": 0.25,
        }
    )
    study = model.simulate(study_design, truth, seed=6)

    trajectory = model.coefficient_trajectory(model.fit(study))

    # A parameter that does not vary is carried forward exactly: no data means no
    # likelihood term, and a constant coefficient has no knot to pull.
    constant = [
        index
        for index, name in enumerate(trajectory.coefficient_names)
        if name not in ("drift.stimulus", "boundary")
    ]
    identical(trajectory.values[-1][constant], trajectory.values[-2][constant])

    # A parameter that *does* vary is held at the unobserved knot by the roughness prior
    # and by nothing else, so the two knots agree only as tightly as that prior is
    # declared -- exact equality would mean the penalty had become infinite. The absence
    # of leakage is the *limit*, not any single tolerance: tighten the prior and the gap
    # goes to zero, which it could not do if the unobserved knot were seeing data.
    gaps = []
    for smoothness in (10.0, 100.0, 1_000.0):
        tightened = smooth(
            base_ddm(),
            over="session_order",
            knots=(0.0, 1.0, 2.0, 3.0),
            smoothness=smoothness,
            parameters=("drift.stimulus", "boundary"),
        )
        path = tightened.coefficient_trajectory(tightened.fit(study))
        gaps.append(float(np.max(np.abs(path.values[-1] - path.values[-2]))))

    assert gaps[0] > gaps[-1]
    assert gaps[-1] < 1e-5


# --------------------------------------------------------------------------------------
# The mixture, which the deleted classes refused and the combinators do not
# --------------------------------------------------------------------------------------


def contaminated_ddm(**changes: Any) -> MixtureModel:
    return mix(
        base_ddm(nondecision_time_bounds=(0.05, 0.4), **changes),
        UniformResponseGuess(time_bounds=(0.15, 3.0)),
        weight_bounds=(0.0, 0.25),
    )


def test_a_mixture_weight_is_a_predictor_cell_and_therefore_composes() -> None:
    """``SmoothWienerDriftDiffusion`` refused contaminants; the composition does not.

    The deleted class raised ``"does not yet support contaminants"`` from its constructor:
    the field composed syntactically and the model refused semantically. The mixture is now
    a combinator of its own, and its weight is a predictor cell like any other, so a
    drifting or group-varying lapse rate is an ordinary instance of the other two.
    """

    model = contaminated_ddm()

    assert model.predictor_cells == (
        "drift",
        "boundary",
        "starting_bias",
        "nondecision_time",
        "mixture_weight",
        "component_log_density",
        "component_probability[0]",
    )
    assert model.parameter_names[-1] == "mixture_logit"

    drifting = smooth(
        model,
        over="session_order",
        knots=KNOTS,
        smoothness=SMOOTHNESS,
        parameters=("mixture_logit",),
    )
    assert drifting.parameter_names[-2:] == (
        "mixture_logit[session_order=0]",
        "mixture_logit[session_order=2]",
    )
    truth = drifting.parameters_from_paths(
        {
            "drift.intercept": 0.1,
            "drift.stimulus": 1.1,
            "boundary": 1.3,
            "starting_bias": 0.48,
            "nondecision_time": 0.25,
            "mixture_logit": [-3.0, -0.5],
        }
    )
    study = drifting.simulate(design(n_subjects=1, n_trials=120), truth, seed=15)
    fit = drifting.fit(study)

    assert fit.diagnostics.converged
    assert np.all(np.isfinite(drifting.pointwise_log_prob(study, fit)))
    early, late = fit.estimates[-2:]
    assert early < late


def test_a_group_varying_mixture_weight_fits() -> None:
    model = hierarchical(
        contaminated_ddm(),
        over="subject",
        parameters=("mixture_logit",),
        scale=0.4,
    )
    truth = dict(
        model.model.from_natural(
            {
                "drift.intercept": 0.1,
                "drift.stimulus": 1.1,
                "boundary": 1.3,
                "starting_bias": 0.48,
                "nondecision_time": 0.25,
                "contaminant_rate": 0.1,
            }
        )
    )
    study = model.simulate(design(n_subjects=3, n_trials=60), truth, seed=21)

    fit = model.fit(study)

    assert fit.varying_parameters == ("mixture_logit",)
    assert fit.group_deviations.shape == (3, 1)
    assert np.all(np.isfinite(fit.group_deviations))


# --------------------------------------------------------------------------------------
# Configuration errors, on the combinators rather than on a variant class
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "arguments",
    [
        {"knots": (0.0,)},
        {"knots": (0.0, 0.0)},
        {"knots": (1.0, 0.0)},
        {"knots": (0.0, np.nan)},
        {"smoothness": 0.0},
        {"over": "choice"},
        {"shared_trajectory": 1},
        {"parameters": ("no-such-parameter",)},
        {"parameters": ()},
        {"parameters": ("boundary", "boundary")},
    ],
)
def test_smoothing_a_wiener_model_is_validated(arguments: dict[str, Any]) -> None:
    # Merge rather than splat alongside the defaults: half these cases override `over` or
    # `knots`, and passing both spellings raises TypeError before the validation under test
    # can raise ValueError.
    settings: dict[str, Any] = {"over": "session_order", "knots": KNOTS, **arguments}
    with pytest.raises(ValueError):
        smooth(base_ddm(), **settings)


@pytest.mark.parametrize(
    "arguments",
    [
        {"scale": 0.0},
        {"parameters": ("nonexistent",)},
        {"parameters": ("boundary", "boundary")},
        {"scale_bounds": (0.2, 0.1)},
        {"estimate_scale": True, "scale": 0.01},
        {"scale_estimator": "variational"},
        {"scale_max_iterations": 0},
        {"scale_tolerance": 0.0},
        {"scale_uncertainty": "posterior"},
        {"scale_uncertainty": "supplemented"},
    ],
)
def test_pooling_a_wiener_model_is_validated(arguments: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        hierarchical(base_ddm(), over="subject", **arguments)


def test_hierarchy_is_the_outer_combinator_for_the_wiener_family_too() -> None:
    pooled = hierarchical(base_ddm(), over="subject")

    with pytest.raises(TypeError, match="hierarchy is the outer combinator"):
        smooth(pooled, over="session_order", knots=KNOTS)


def test_hierarchical_fitting_needs_more_than_one_group() -> None:
    model = pooled_model(scale=0.18)
    study = model.simulate(design(n_subjects=1), model.parameters_from_paths(PATHS), seed=1)

    with pytest.raises(ModelDataError, match="at least two subject groups"):
        model.fit(study)


def test_the_wiener_density_accepts_aligned_trialwise_boundaries_and_biases() -> None:
    times = np.asarray([0.2, 0.5, 0.8])
    choices = np.asarray([0.0, 1.0, 1.0])
    drifts = np.asarray([-0.2, 0.5, 1.1])
    boundaries = np.asarray([0.8, 1.1, 1.4])
    biases = np.asarray([0.3, 0.5, 0.7])

    vectorized = wiener_log_density(
        times, choices, drifts, boundary=boundaries, starting_bias=biases, terms=12
    )
    scalar = np.asarray(
        [
            wiener_log_density(
                times[index : index + 1],
                choices[index : index + 1],
                drifts[index : index + 1],
                boundary=float(boundaries[index]),
                starting_bias=float(biases[index]),
                terms=12,
            )[0]
            for index in range(len(times))
        ]
    )

    np.testing.assert_allclose(vectorized, scalar)
