"""The PyDDM wrapper: parity with Behavio's own density, the contract, and where it strains.

Every test here is gated on the optional extra, exactly as the NWB, DANDI, ONE, PyBADS and
Parquet suites are: ``pytest.importorskip`` at module scope, so a checkout without
``behavio[pyddm]`` skips this file cleanly rather than failing it.
"""

from __future__ import annotations

import numpy as np
import pytest

from behavio import Study, compare_models, forward_session_splits
from behavio.adapters import assert_behaviour_estimator_conforms
from behavio.contracts import (
    BehaviourEstimator,
    CategoricalBehaviourEstimator,
    ConvergenceStatus,
    DensityBehaviourEstimator,
    DensityPrediction,
    FitAuditStatus,
    FitDiagnostics,
    GenerativeBehaviourModel,
    PredictionMode,
    UnsupportedPredictionMode,
    model_capabilities,
)
from behavio.design import DesignSpec
from behavio.evaluate import evaluate_splits
from behavio.models._kernels.wiener import wiener_log_density
from behavio.models.ddm import WienerDriftDiffusion
from behavio.recovery import run_parameter_recovery
from behavio.task import ResponseTimeSpec, ResponseTimeUnit

pytest.importorskip("pyddm")

from behavio.foreign.pyddm import (
    PARAMETER_CORRESPONDENCE,
    PyDDMDriftDiffusion,
    PyDDMFitResult,
)

TRUTH = {
    "drift": {"drift.intercept": 0.2, "drift.stimulus": 1.8},
    "boundary": 1.4,
    "starting_bias": 0.52,
    "nondecision_time": 0.22,
}


def wrapper(**overrides) -> PyDDMDriftDiffusion:
    settings = {
        "predictors": ("stimulus",),
        "time_step": 0.01,
        "max_time": 4.0,
        "max_iterations": 40,
        "population_size": 8,
        "seed": 3,
    }
    settings.update(overrides)
    return PyDDMDriftDiffusion(**settings)


@pytest.fixture(scope="module")
def simulated() -> Study:
    """Three sessions of a two-level task, simulated by Behavio's own Wiener model."""

    generator = np.random.default_rng(0)
    design = Study.factorial(
        trials=80,
        subjects=1,
        sessions=3,
        columns={"stimulus": generator.choice([-1.0, 1.0], 240)},
    )
    reference = WienerDriftDiffusion(predictors=("stimulus",))
    return reference.simulate(design, reference.parameters_from_components(**TRUTH), seed=1)


@pytest.fixture(scope="module")
def fitted(simulated: Study) -> tuple[PyDDMDriftDiffusion, PyDDMFitResult]:
    model = wrapper()
    return model, model.fit(simulated)


# ---------------------------------------------------------------------------------------
# The parameter map, which is the whole of what recovery could get wrong.
# ---------------------------------------------------------------------------------------


def test_the_pyddm_density_matches_behavios_own_wiener_density(simulated: Study) -> None:
    """The claim the parameter map makes, checked against the density it maps onto.

    Behavio's Wiener density is a fixed twelve-term series with a hardcoded expansion
    switch; PyDDM's is adaptive Navarro-Fuss. If the boundary, start and non-decision
    conversions are right, the two agree to solver precision over the body of the
    distribution, and any disagreement is a mapping error rather than a numerical one.
    """

    model = PyDDMDriftDiffusion(predictors=("stimulus",), time_step=0.001, max_time=6.0)
    values = dict(model.parameters_from_components(**TRUTH))
    conditions, _ = model._condition_rows(model._features(simulated))
    grid, table = model._density_table(values, conditions)

    for row, (intercept, stimulus) in enumerate(conditions):
        drift = TRUTH["drift"]["drift.intercept"] * intercept
        drift += TRUTH["drift"]["drift.stimulus"] * stimulus
        for time in (0.3, 0.5, 1.0, 2.0):
            for category, outcome in ((0, 0.0), (1, 1.0)):
                behavio = float(
                    np.exp(
                        wiener_log_density(
                            np.asarray([time - TRUTH["nondecision_time"]]),
                            np.asarray([outcome]),
                            np.asarray([drift]),
                            boundary=TRUTH["boundary"],
                            starting_bias=TRUTH["starting_bias"],
                            terms=12,
                        )[0]
                    )
                )
                foreign = float(np.interp(time, grid, table[row, category]))
                assert foreign == pytest.approx(behavio, rel=1e-6)


def test_the_parameter_correspondence_is_published_and_complete() -> None:
    assert set(PARAMETER_CORRESPONDENCE) == {
        "boundary",
        "starting_bias",
        "nondecision_time",
        "drift.<feature>",
    }


def test_parameter_names_match_behavios_own_drift_diffusion_exactly() -> None:
    """A recovery study is worthless if the simulator and the fitter disagree on a name."""

    assert (
        wrapper().parameter_names == WienerDriftDiffusion(predictors=("stimulus",)).parameter_names
    )


# ---------------------------------------------------------------------------------------
# Identity and the contract.
# ---------------------------------------------------------------------------------------


def test_the_wrapper_satisfies_the_estimator_protocols() -> None:
    model = wrapper()

    assert isinstance(model, BehaviourEstimator)
    assert isinstance(model, GenerativeBehaviourModel)
    assert isinstance(model, DensityBehaviourEstimator)
    capabilities = model_capabilities(model)
    assert capabilities.scored_columns == ("choice", "response_time")
    assert capabilities.required_task_columns == ("stimulus",)
    assert capabilities.prediction_modes == (PredictionMode.FILTERED,)
    assert capabilities.can_recover_parameters


def test_the_signature_is_a_fingerprint_over_every_setting_that_changes_the_numbers() -> None:
    base = wrapper()

    assert base.signature == wrapper().signature
    assert "backend=pyddm:0.9" in base.signature
    for changed in (
        wrapper(time_step=0.02),
        wrapper(max_time=5.0),
        wrapper(seed=4),
        wrapper(max_iterations=41),
        wrapper(population_size=9),
        wrapper(contaminant_probability=0.02),
        wrapper(loss="likelihood"),
        wrapper(boundary_bounds=(0.1, 4.0)),
    ):
        assert changed.signature != base.signature


def test_the_signature_does_not_move_with_settings_that_only_cost_time() -> None:
    """``max_conditions`` is a refusal threshold, not a claim about the model."""

    assert wrapper(max_conditions=8).signature == wrapper(max_conditions=256).signature


def test_describe_reports_the_design_the_parameters_and_the_bounds(simulated: Study) -> None:
    description = wrapper().describe(simulated)

    assert description.model_name == "pyddm-drift-diffusion"
    assert description.design_columns == ("intercept", "stimulus")
    assert description.parameter_names == (
        "drift.intercept",
        "drift.stimulus",
        "boundary",
        "starting_bias",
        "nondecision_time",
    )
    assert description.parameter_bounds["boundary"] == (0.1, 5.0)
    assert not description.errors


def test_describe_refuses_a_study_the_solver_grid_cannot_reach(simulated: Study) -> None:
    description = wrapper(max_time=0.5).describe(simulated)

    codes = [finding.code for finding in description.errors]
    assert "response_time_beyond_grid" in codes


def test_describe_refuses_a_design_with_more_condition_grids_than_declared() -> None:
    generator = np.random.default_rng(1)
    design = Study.factorial(
        trials=40,
        subjects=1,
        sessions=1,
        columns={"stimulus": generator.normal(size=40), "response_time": np.full(40, 0.5)},
    )
    study = Study({**{name: design[name] for name in design.columns}, "choice": [1, 0] * 20})

    description = wrapper(max_conditions=8).describe(study)

    finding = next(f for f in description.errors if f.code == "too_many_condition_grids")
    assert "40 distinct rows" in finding.message
    assert "bin it" in finding.message


# ---------------------------------------------------------------------------------------
# Fitting.
# ---------------------------------------------------------------------------------------


def test_the_fit_is_a_behavio_fit_result_with_the_foreign_evidence_kept(
    fitted: tuple[PyDDMDriftDiffusion, PyDDMFitResult], simulated: Study
) -> None:
    model, fit = fitted

    assert fit.model_name == model.model_name
    assert fit.model_signature == model.signature
    assert fit.n_observations == len(simulated)
    assert fit.pyddm_version.startswith("0.9")
    assert fit.pyddm_loss
    assert fit.pyddm_solver == "analytical"
    assert fit.n_conditions == 2
    assert fit.likelihood_floor_count == 0
    assert fit.diagnostics.optimizer == "pyddm:differential_evolution"


def test_the_fit_recovers_the_truth_it_was_simulated_from(
    fitted: tuple[PyDDMDriftDiffusion, PyDDMFitResult],
) -> None:
    _, fit = fitted
    estimates = fit.parameters

    assert estimates["drift.stimulus"] == pytest.approx(1.8, abs=0.5)
    assert estimates["boundary"] == pytest.approx(1.4, abs=0.3)
    assert estimates["nondecision_time"] == pytest.approx(0.22, abs=0.05)


def test_a_covariance_is_estimated_and_agrees_with_behavios_own_to_an_order(
    fitted: tuple[PyDDMDriftDiffusion, PyDDMFitResult], simulated: Study
) -> None:
    """PyDDM reports no uncertainty; the wrapper differences its loss to get one."""

    _, fit = fitted
    reference = WienerDriftDiffusion(predictors=("stimulus",)).fit(simulated)

    assert fit.covariance_is_estimated
    assert np.all(np.isfinite(fit.standard_errors))
    for name in fit.parameter_names:
        ratio = fit.standard_error_map[name] / reference.standard_error_map[name]
        assert 0.25 < ratio < 4.0, name


def test_the_fit_audit_passes_and_convergence_is_a_verified_claim(
    fitted: tuple[PyDDMDriftDiffusion, PyDDMFitResult],
) -> None:
    _, fit = fitted

    assert fit.diagnostics.converged is True
    assert "solver-lattice step" in fit.diagnostics.message
    assert fit.audit().status.value == "pass"


def test_an_underpowered_search_is_reported_as_not_converged(simulated: Study) -> None:
    """The convergence claim has to be falsifiable, so an unfinished search must fail it."""

    starved = wrapper(max_iterations=1, population_size=4)

    fit = starved.fit(simulated)

    assert fit.diagnostics.converged is False
    assert fit.audit().status.value == "fail"


def test_fitting_twice_under_one_seed_gives_the_same_estimates(simulated: Study) -> None:
    model = wrapper()

    first, second = model.fit(simulated), model.fit(simulated)

    assert np.array_equal(first.estimates, second.estimates)


def test_a_different_seed_is_a_different_configuration(simulated: Study) -> None:
    assert wrapper(seed=3).signature != wrapper(seed=11).signature


# ---------------------------------------------------------------------------------------
# Prediction, density and scoring.
# ---------------------------------------------------------------------------------------


def test_predict_returns_the_density_itself_not_the_choice_probability(
    fitted: tuple[PyDDMDriftDiffusion, PyDDMFitResult], simulated: Study
) -> None:
    """A diffusion predicts a joint distribution, and ``predict`` now says so.

    ``predict`` used to return the upper-boundary probability alone, which threw the
    latency half away at exactly the point where a fold, a comparison and an evidence
    bundle pick a prediction up. The choice probabilities are still available and are
    still derived from the density, so the two halves cannot disagree.
    """

    model, fit = fitted

    prediction = model.predict(simulated, fit)
    density = model.predict_density(simulated, fit)

    assert isinstance(prediction, DensityPrediction)
    assert isinstance(density, DensityPrediction)
    assert density.is_defective and density.categories == (0, 1)
    assert density.outcome == "response_time"
    assert density.n_observations == len(simulated)
    assert np.array_equal(prediction.density, density.density)
    margin = model.choice_probability(simulated, fit)
    assert margin.categories == (0, 1)
    assert np.allclose(margin.probability, density.choice_prediction().probability)
    assert np.all(density.total_mass > 0.999)
    # The category coordinate the fold retains codes against is the density's own.
    assert isinstance(model, CategoricalBehaviourEstimator)
    assert tuple(model.categories) == density.categories
    codes = model.outcome_codes(simulated)
    assert np.array_equal(codes, np.asarray(simulated["choice"], dtype=np.int64))


def test_the_pointwise_score_is_the_density_read_at_the_observation(
    fitted: tuple[PyDDMDriftDiffusion, PyDDMFitResult], simulated: Study
) -> None:
    model, fit = fitted

    scores = model.pointwise_log_prob(simulated, fit)
    density = model.predict_density(simulated, fit)
    replayed = density.observed_log_density(
        np.asarray(simulated["response_time"], dtype=np.float64),
        np.asarray(simulated["choice"], dtype=np.int64),
    )

    assert scores.shape == (len(simulated),)
    assert np.allclose(scores, replayed)


def test_the_interpolated_score_differs_from_pyddms_binned_loss_and_says_so(
    fitted: tuple[PyDDMDriftDiffusion, PyDDMFitResult], simulated: Study
) -> None:
    """PyDDM scores by rounding each response time to a grid index; the wrapper interpolates.

    The two are different functions of the same density, so they disagree. The size of the
    disagreement is retained rather than hidden, and it shrinks when the grid is refined.
    """

    model, fit = fitted
    scores = model.pointwise_log_prob(simulated, fit)

    assert fit.interpolation_gap == pytest.approx(
        -float(np.sum(scores)) - fit.diagnostics.objective, abs=1e-9
    )
    fine = wrapper(time_step=0.002).fit(simulated)
    assert abs(fine.interpolation_gap) < abs(fit.interpolation_gap)


def test_a_smoothed_prediction_is_refused_rather_than_faked(
    fitted: tuple[PyDDMDriftDiffusion, PyDDMFitResult], simulated: Study
) -> None:
    model, fit = fitted

    with pytest.raises(UnsupportedPredictionMode, match="only filtered"):
        model.predict(simulated, fit, mode=PredictionMode.SMOOTHED)
    with pytest.raises(UnsupportedPredictionMode):
        model.pointwise_log_prob(simulated, fit, mode=PredictionMode.SMOOTHED)


def test_a_fit_from_another_specification_is_refused(simulated: Study) -> None:
    model, other = wrapper(), wrapper(time_step=0.02)
    fit = other.fit(simulated)

    with pytest.raises(ValueError, match="different model specification"):
        model.predict(simulated, fit)


# ---------------------------------------------------------------------------------------
# Convergence: what PyDDM does not report, and what the wrapper checks instead.
# ---------------------------------------------------------------------------------------


def test_the_wrapper_still_verifies_local_optimality_rather_than_shrugging(
    fitted: tuple[PyDDMDriftDiffusion, PyDDMFitResult],
) -> None:
    """``UNREPORTED`` exists now, and the self-check survived it, because it says more.

    PyDDM reports no convergence flag, so the honest floor is
    :attr:`~behavio.contracts.ConvergenceStatus.UNREPORTED`. Coordinate-wise local
    optimality on the solver lattice is a strictly stronger, checkable claim, and the
    probe evaluations are the ones the Hessian already takes, so it costs nothing to keep.
    """

    _, fit = fitted

    assert fit.diagnostics.convergence is ConvergenceStatus.CONVERGED
    assert fit.diagnostics.converged is True
    assert fit.diagnostics.status == 0
    assert "solver-lattice step" in fit.diagnostics.message
    assert "pyddm 0.9" in fit.diagnostics.message
    assert fit.audit().status is not FitAuditStatus.FAIL


def test_a_convergence_check_that_could_not_run_reports_unreported_not_failure() -> None:
    """The one branch where neither PyDDM nor the wrapper measured anything.

    Every coordinate pinned against a bound leaves no admissible local move, so the probe
    cannot be taken. That used to be recorded as ``converged=False``, which failed the
    audit and evicted the fit from every comparison on the strength of a test that never
    executed.
    """

    from behavio.foreign.pyddm import _unknown_curvature

    curvature = _unknown_curvature(
        2,
        "no covariance and no convergence check: every parameter is pinned against its bound",
        converged=ConvergenceStatus.UNREPORTED,
    )

    assert curvature.status is None
    diagnostics = FitDiagnostics(
        converged=curvature.converged,
        optimizer="pyddm:differential_evolution",
        status=curvature.status,
        message=curvature.message,
        n_iterations=None,
        objective=1.0,
        gradient_norm=None,
        hessian_condition=None,
        boundary_estimate=True,
    )

    assert diagnostics.convergence is ConvergenceStatus.UNREPORTED
    assert not diagnostics.failed_to_converge
    # A refused covariance is still a refused covariance, and still only a warning.
    assert not np.all(np.isfinite(curvature.covariance))
    assert not curvature.estimated


# ---------------------------------------------------------------------------------------
# Simulation.
# ---------------------------------------------------------------------------------------


def test_the_simulator_reproduces_the_solved_choice_probabilities() -> None:
    model = wrapper(time_step=0.005, max_time=6.0)
    design = Study.factorial(
        trials=4000, subjects=1, sessions=1, columns={"stimulus": np.zeros(4000)}
    )
    values = model.parameters_from_components(
        drift={"drift.intercept": 0.5, "drift.stimulus": 0.0},
        boundary=1.2,
        starting_bias=0.5,
        nondecision_time=0.2,
    )

    simulated = model.simulate(design, values, seed=11)
    conditions, _ = model._condition_rows(model._features(design))
    grid, table = model._density_table(dict(values), conditions)
    expected = np.trapezoid(table[0], grid, axis=-1)

    assert float(np.mean(simulated["choice"])) == pytest.approx(expected[1], abs=0.02)
    assert float(np.min(simulated["response_time"])) > 0.2


def test_the_simulator_is_seeded_and_touches_no_global_stream() -> None:
    model = wrapper()
    design = Study.factorial(trials=50, subjects=1, sessions=1, columns={"stimulus": np.ones(50)})
    values = model.parameters_from_components(**TRUTH)

    np.random.seed(0)
    first = model.simulate(design, values, seed=5)
    np.random.seed(999)
    second = model.simulate(design, values, seed=5)

    assert np.array_equal(np.asarray(first["choice"]), np.asarray(second["choice"]))
    assert np.array_equal(np.asarray(first["response_time"]), np.asarray(second["response_time"]))


def test_the_simulator_refuses_to_discard_undecided_mass() -> None:
    model = wrapper(max_time=0.4)
    design = Study.factorial(trials=10, subjects=1, sessions=1, columns={"stimulus": np.zeros(10)})
    values = model.parameters_from_components(
        drift={"drift.intercept": 0.0, "drift.stimulus": 0.0},
        boundary=3.0,
        starting_bias=0.5,
        nondecision_time=0.0,
    )

    with pytest.raises(ValueError, match="has not terminated"):
        model.simulate(design, values, seed=1)


# ---------------------------------------------------------------------------------------
# The consumers the wrapper exists to reach.
# ---------------------------------------------------------------------------------------


def test_the_wrapper_passes_the_estimator_conformance_harness(simulated: Study) -> None:
    report = assert_behaviour_estimator_conforms(wrapper(), simulated)

    names = {check.name for check in report.checks if check.status.value == "passed"}
    assert "filtered-prediction-ignores-future-rows" in names
    assert "filtered-score-ignores-future-rows" in names
    assert "density-agrees-with-the-choice-prediction" in names
    assert "simulates-the-columns-it-scores" in names


def test_the_wrapper_flows_through_prospective_evaluation(simulated: Study) -> None:
    splits = forward_session_splits(simulated)

    evaluation = evaluate_splits(wrapper(), simulated, splits)

    assert evaluation.complete
    assert len(evaluation.evaluations) == len(splits)
    for fold in evaluation.evaluations:
        assert np.isfinite(fold.total_log_probability)
        assert fold.prediction.n_observations == fold.split.test_indices.size
        # The density itself survives the fold, sliced to the scored rows, with the
        # observed boundary of each row retained beside it.
        assert isinstance(fold.prediction, DensityPrediction)
        assert fold.prediction.is_defective
        assert fold.outcome_codes is not None


def test_the_wrapper_is_comparable_against_behavios_own_drift_diffusion(
    simulated: Study,
) -> None:
    splits = forward_session_splits(simulated)

    report = compare_models(
        {"pyddm": wrapper(), "wiener": WienerDriftDiffusion(predictors=("stimulus",))},
        simulated,
        splits,
    )

    assert set(report.model_order) == {"pyddm", "wiener"}
    assert report.scored_columns == ("choice", "response_time")
    for name in report.model_order:
        assert report.result_for(name).audit_status is not FitAuditStatus.FAIL
    assert set(report.eligible_model_order) == {"pyddm", "wiener"}
    assert report.winner in {"pyddm", "wiener"}
    # The Brier score of the density candidate is its choice margin's, which is the only
    # part of a density a probability scoring rule can read; the log loss is joint.
    pyddm_result = report.result_for("pyddm")
    assert 0.0 <= pyddm_result.pooled_brier_score <= 1.0
    expected = []
    for evaluation in pyddm_result.evaluations:
        margin = evaluation.prediction.choice_prediction().probability
        targets = np.zeros_like(margin)
        targets[np.arange(len(targets)), evaluation.outcome_codes] = 1.0
        expected.extend(0.5 * np.sum((margin - targets) ** 2, axis=1))
    assert pyddm_result.pooled_brier_score == pytest.approx(float(np.mean(expected)))


def test_parameter_recovery_runs_against_the_wrappers_own_simulator() -> None:
    model = wrapper()
    generator = np.random.default_rng(4)
    design = Study.factorial(
        trials=150,
        subjects=1,
        sessions=1,
        columns={"stimulus": generator.choice([-1.0, 1.0], 150)},
    )

    report = run_parameter_recovery(
        model, design, [model.parameters_from_components(**TRUTH)], repeats=2, seed=5
    )

    assert report.model_name == "pyddm-drift-diffusion"
    assert report.model_signature == model.signature
    assert report.convergence_rate == 1.0
    assert report.interval_kind == "wald"
    summaries = {summary.parameter: summary for summary in report.summary()}
    assert summaries["drift.stimulus"].n_with_uncertainty == 2


def test_recovery_exposes_pyddms_truncated_non_decision_shift() -> None:
    """The one bias the wrapper cannot remove, made visible by the machinery around it.

    ``OverlayNonDecision`` shifts by ``int(nondecision_time / dt)`` bins, truncating rather
    than rounding, so every value inside one solver time step is observationally identical
    and the fitted value drifts to the top of its cell. Behavio's recovery machinery is what
    surfaces that; the wrapper only declares the quantum it happens in.
    """

    model = wrapper(time_step=0.02)
    generator = np.random.default_rng(6)
    design = Study.factorial(
        trials=200,
        subjects=1,
        sessions=1,
        columns={"stimulus": generator.choice([-1.0, 1.0], 200)},
    )

    report = run_parameter_recovery(
        model, design, [model.parameters_from_components(**TRUTH)], repeats=2, seed=8
    )

    bias = {summary.parameter: summary.bias for summary in report.summary()}
    assert abs(bias["nondecision_time"]) < 3 * model.time_step
    fit = model.fit(model.simulate(design, model.parameters_from_components(**TRUTH), seed=2))
    assert fit.derived_value("nondecision_time_grid_quantum") == model.time_step


# ---------------------------------------------------------------------------------------
# Configuration and refusals.
# ---------------------------------------------------------------------------------------


def test_millisecond_response_times_are_read_through_the_task_contract() -> None:
    """The response-time unit is declared, not inferred, and the fit is unit-invariant."""

    generator = np.random.default_rng(2)
    design = Study.factorial(
        trials=120, subjects=1, sessions=1, columns={"stimulus": generator.choice([-1.0, 1.0], 120)}
    )
    reference = WienerDriftDiffusion(predictors=("stimulus",))
    seconds = reference.simulate(design, reference.parameters_from_components(**TRUTH), seed=7)
    milliseconds = Study(
        {
            **{name: seconds[name] for name in seconds.columns if name != "response_time"},
            "response_time_ms": np.asarray(seconds["response_time"]) * 1000.0,
        }
    )

    in_seconds = wrapper().fit(seconds)
    in_milliseconds = wrapper(
        response_time=ResponseTimeSpec(
            column="response_time_ms", unit=ResponseTimeUnit.MILLISECONDS
        )
    ).fit(milliseconds)

    assert np.allclose(in_seconds.estimates, in_milliseconds.estimates, atol=1e-8)


def test_an_invalid_configuration_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="time_step must be smaller"):
        PyDDMDriftDiffusion(time_step=5.0, max_time=1.0)
    with pytest.raises(ValueError, match="loss must be one of"):
        PyDDMDriftDiffusion(loss="squared-error")
    with pytest.raises(ValueError, match="fitting_method must be one of"):
        PyDDMDriftDiffusion(fitting_method="bads")
    with pytest.raises(ValueError, match="contaminant_probability"):
        PyDDMDriftDiffusion(contaminant_probability=1.5)
    with pytest.raises(ValueError, match="seed must be a non-negative integer"):
        PyDDMDriftDiffusion(seed=-1)
    with pytest.raises(ValueError, match="starting_bias_bounds"):
        PyDDMDriftDiffusion(starting_bias_bounds=(0.0, 1.0))
    with pytest.raises(ValueError, match="either predictors or design"):
        PyDDMDriftDiffusion(predictors=("stimulus",), design=DesignSpec(terms=()))


def test_the_wrapper_refuses_a_study_it_cannot_score(simulated: Study) -> None:
    without_stimulus = Study(
        {name: simulated[name] for name in simulated.columns if name != "stimulus"}
    )

    with pytest.raises(ValueError, match="cannot fit this study"):
        wrapper().fit(without_stimulus)


def test_a_contaminant_mixture_changes_the_density_and_the_signature(simulated: Study) -> None:
    plain, mixed = wrapper(), wrapper(contaminant_probability=0.05)
    values = dict(plain.parameters_from_components(**TRUTH))
    conditions, _ = plain._condition_rows(plain._features(simulated))

    _, clean_table = plain._density_table(values, conditions)
    _, mixed_table = mixed._density_table(values, conditions)

    assert plain.signature != mixed.signature
    assert not np.allclose(clean_table, mixed_table)


def test_a_contaminant_mixture_still_fits_and_maps_back(simulated: Study) -> None:
    """The mixture nests the non-decision overlay inside a chain; the map must survive it."""

    model = wrapper(contaminant_probability=0.02, max_iterations=25, population_size=6)

    fit = model.fit(simulated)

    assert set(fit.parameters) == set(model.parameter_names)
    assert np.all(np.isfinite(fit.estimates))
    assert fit.parameters["boundary"] > 0
    assert 0 < fit.parameters["starting_bias"] < 1


def test_the_simplex_fitter_is_available_and_reports_its_own_identity(
    simulated: Study,
) -> None:
    model = wrapper(fitting_method="simplex")

    fit = model.fit(simulated)

    assert fit.diagnostics.optimizer == "pyddm:simplex"
    assert model.signature != wrapper().signature
