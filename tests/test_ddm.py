from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

from behavio import (
    BehaviourModel,
    DriftDiffusionFitResult,
    PredictionMode,
    ResponseTimeSpec,
    ResponseTimeUnit,
    Study,
    WienerDriftDiffusion,
    evaluate_splits,
    forward_session_splits,
    model_capabilities,
    run_parameter_recovery,
)
from behavio.models.ddm import (
    _numerical_hessian,
    _upper_boundary_probability,
    _wiener_log_density,
)


def design(
    *,
    n_sessions: int = 4,
    trials_per_session: int = 200,
    seed: int = 61,
) -> Study:
    generator = np.random.default_rng(seed)
    n_rows = n_sessions * trials_per_session
    return Study(
        {
            "subject": ["a"] * n_rows,
            "session": [
                f"s{session}" for session in range(n_sessions) for _ in range(trials_per_session)
            ],
            "trial": list(range(trials_per_session)) * n_sessions,
            "session_order": [
                session for session in range(n_sessions) for _ in range(trials_per_session)
            ],
            "stimulus": generator.normal(size=n_rows),
        }
    )


def model(**changes: object) -> WienerDriftDiffusion:
    arguments = {
        "covariates": ("stimulus",),
        "n_restarts": 3,
        "max_iterations": 300,
        "simulation_time_step": 0.0005,
    }
    arguments.update(changes)
    return WienerDriftDiffusion(**arguments)  # type: ignore[arg-type]


def truth(ddm: WienerDriftDiffusion):
    return ddm.parameters_from_components(
        drift={"drift.intercept": 0.2, "drift.stimulus": 1.2},
        boundary=1.2,
        starting_bias=0.45,
        nondecision_time=0.25,
    )


@pytest.mark.parametrize(
    ("drift", "boundary", "starting_bias"),
    [(0.0, 1.0, 0.5), (1.2, 1.1, 0.4), (-1.5, 0.8, 0.65)],
)
def test_wiener_density_integrates_to_analytic_boundary_probabilities(
    drift: float,
    boundary: float,
    starting_bias: float,
) -> None:
    masses = []
    for choice in (0.0, 1.0):

        def density(decision_time: float, observed_choice: float = choice) -> float:
            return float(
                np.exp(
                    _wiener_log_density(
                        np.asarray([decision_time]),
                        np.asarray([observed_choice]),
                        np.asarray([drift]),
                        boundary=boundary,
                        starting_bias=starting_bias,
                        terms=12,
                    )[0]
                )
            )

        mass, _ = quad(density, 1e-8, 20.0, epsabs=1e-9)
        masses.append(mass)

    upper = float(
        _upper_boundary_probability(
            np.asarray([drift]),
            boundary=boundary,
            starting_bias=starting_bias,
        )[0]
    )
    assert masses[0] == pytest.approx(1.0 - upper, abs=2e-8)
    assert masses[1] == pytest.approx(upper, abs=2e-8)
    assert sum(masses) == pytest.approx(1.0, abs=2e-8)


def test_ddm_satisfies_joint_observation_contract_and_round_trips_parameters() -> None:
    ddm = model()
    parameters = truth(ddm)
    components = ddm.parameter_components(parameters)

    assert isinstance(ddm, BehaviourModel)
    assert ddm.scored_columns == ("choice", "response_time")
    assert ddm.supported_prediction_modes == (PredictionMode.FILTERED,)
    assert model_capabilities(ddm).can_recover_parameters
    assert components.drift_map["drift.stimulus"] == 1.2
    assert components.boundary == 1.2
    assert components.starting_bias == 0.45
    assert components.nondecision_time == 0.25


def test_simulation_is_reproducible_and_respects_declared_time_units() -> None:
    seconds_model = model()
    source_design = design(n_sessions=1, trials_per_session=200)
    first = seconds_model.simulate(source_design, truth(seconds_model), seed=9)
    second = seconds_model.simulate(source_design, truth(seconds_model), seed=9)

    np.testing.assert_array_equal(first["choice"], second["choice"])
    np.testing.assert_array_equal(first["response_time"], second["response_time"])
    assert set(first["choice"].tolist()) <= {0, 1}
    assert np.min(first["response_time"]) > 0.25

    milliseconds_model = model(
        response_time=ResponseTimeSpec(column="rt_ms", unit=ResponseTimeUnit.MILLISECONDS)
    )
    milliseconds = milliseconds_model.simulate(
        design(n_sessions=1, trials_per_session=200),
        truth(milliseconds_model),
        seed=9,
    )
    np.testing.assert_allclose(milliseconds["rt_ms"], first["response_time"] * 1000.0)


def test_fit_recovers_joint_choice_rt_parameters_and_retains_restart_diagnostics() -> None:
    ddm = model()
    simulated = ddm.simulate(design(), truth(ddm), seed=12)

    fit = ddm.fit(simulated)
    recovered = ddm.parameter_components(fit)

    assert isinstance(fit, DriftDiffusionFitResult)
    assert fit.diagnostics.converged
    assert fit.restart_objectives.shape == (ddm.n_restarts,)
    assert fit.restart_converged.shape == (ddm.n_restarts,)
    assert fit.likelihood_floor_count == 0
    assert fit.audit().status.value in {"pass", "warning"}
    assert recovered.drift_map["drift.intercept"] == pytest.approx(0.2, abs=0.155)
    assert recovered.drift_map["drift.stimulus"] == pytest.approx(1.2, abs=0.145)
    assert recovered.boundary == pytest.approx(1.2, abs=0.043)
    assert recovered.starting_bias == pytest.approx(0.45, abs=0.024)
    assert recovered.nondecision_time == pytest.approx(0.25, abs=0.0063)
    assert np.all(np.isfinite(ddm.pointwise_log_prob(simulated, fit)))


def test_ddm_supports_prospective_evaluation_and_generic_parameter_recovery() -> None:
    ddm = model(n_restarts=2)
    simulated = ddm.simulate(design(n_sessions=4, trials_per_session=100), truth(ddm), seed=21)
    evaluations = evaluate_splits(
        ddm,
        simulated,
        forward_session_splits(simulated, min_train_sessions=3),
    )
    recovery = run_parameter_recovery(
        ddm,
        design(n_sessions=2, trials_per_session=200),
        [truth(ddm)],
        seed=22,
    )

    assert len(evaluations) == 1
    assert len(evaluations[0].pointwise_log_probability) == 100
    assert recovery.n_runs == 1
    assert recovery.parameter_names == ddm.parameter_names
    assert recovery.audit_failure_rate == 0.0


def test_response_times_at_or_below_nondecision_time_have_finite_floor_scores() -> None:
    ddm = model()
    simulated = ddm.simulate(design(n_sessions=1, trials_per_session=20), truth(ddm), seed=31)
    fit = ddm.fit(simulated)
    columns = {name: simulated[name] for name in simulated.columns}
    response_times = np.array(simulated["response_time"], copy=True)
    response_times[0] = ddm.parameter_components(fit).nondecision_time
    columns["response_time"] = response_times

    scores = ddm.pointwise_log_prob(Study(columns), fit)

    assert np.isfinite(scores[0])
    assert scores[0] < -700


def test_numerical_hessian_respects_a_narrow_bounded_interval() -> None:
    point = np.asarray([2.5e-5])

    hessian = _numerical_hessian(
        lambda values: float((values[0] - 2.0e-5) ** 2),
        point,
        [(0.0, 5.0e-5)],
    )

    assert hessian[0, 0] == pytest.approx(2.0)


@pytest.mark.parametrize(
    "arguments",
    [
        {"density_terms": 7},
        {"simulation_time_step": -1.0},
        {"starting_bias_bounds": (0.0, 0.9)},
        {"response_time": "response_time"},
    ],
)
def test_ddm_configuration_is_validated(arguments: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        WienerDriftDiffusion(**arguments)
