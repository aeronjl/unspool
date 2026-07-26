from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

from unspool import (
    BehaviourModel,
    DriftDiffusionFitResult,
    ResponseTimeSpec,
    ResponseTimeUnit,
    Study,
    UniformResponseTimeContaminant,
    WienerDriftDiffusion,
    evaluate_splits,
    forward_session_splits,
    run_parameter_recovery,
)


def design(*, n_sessions: int = 4, trials_per_session: int = 300, seed: int = 401) -> Study:
    generator = np.random.default_rng(seed)
    n_trials = n_sessions * trials_per_session
    return Study(
        {
            "subject": ["synthetic-subject"] * n_trials,
            "session": [
                f"session-{session}"
                for session in range(n_sessions)
                for _ in range(trials_per_session)
            ],
            "trial": list(range(trials_per_session)) * n_sessions,
            "session_order": [
                session for session in range(n_sessions) for _ in range(trials_per_session)
            ],
            "stimulus": generator.normal(size=n_trials),
        }
    )


def model(**changes: object) -> WienerDriftDiffusion:
    arguments = {
        "covariates": ("stimulus",),
        "contaminant": UniformResponseTimeContaminant(time_bounds=(0.05, 3.0)),
        "nondecision_time_bounds": (0.1, 0.6),
        "n_restarts": 4,
        "max_iterations": 400,
        "simulation_time_step": 0.0002,
    }
    arguments.update(changes)
    return WienerDriftDiffusion(**arguments)  # type: ignore[arg-type]


def truth(ddm: WienerDriftDiffusion, *, probability: float = 0.06):
    return ddm.parameters_from_components(
        drift={"drift.intercept": 0.2, "drift.stimulus": 1.2},
        boundary=1.2,
        starting_bias=0.45,
        nondecision_time=0.25,
        contaminant_probability=probability,
    )


def test_uniform_contaminant_is_a_normalized_fixed_support_joint_density() -> None:
    contaminant = UniformResponseTimeContaminant(
        time_bounds=(0.1, 2.1),
        choice_probability=0.3,
    )

    lower = contaminant.pointwise_log_density(
        np.asarray([1.0]),
        np.asarray([0.0]),
    )[0]
    upper = contaminant.pointwise_log_density(
        np.asarray([1.0]),
        np.asarray([1.0]),
    )[0]
    outside = contaminant.pointwise_log_density(
        np.asarray([3.0]),
        np.asarray([1.0]),
    )[0]

    assert np.exp(lower) == pytest.approx(0.7 / 2.0)
    assert np.exp(upper) == pytest.approx(0.3 / 2.0)
    assert np.isneginf(outside)
    assert (np.exp(lower) + np.exp(upper)) * 2.0 == pytest.approx(1.0)


def test_contaminant_configuration_and_parameters_are_explicit() -> None:
    ddm = model()
    parameters = truth(ddm)
    components = ddm.parameter_components(parameters)

    assert isinstance(ddm, BehaviourModel)
    assert ddm.model_name == "uniform-contaminant-wiener-drift-diffusion"
    assert ddm.parameter_names[-1] == "contaminant_probability"
    assert components.contaminant_probability == 0.06
    assert "contaminant_time_bounds=(0.05, 3.0)" in ddm.signature
    with pytest.raises(ValueError, match="requires fixed nondecision_time_bounds"):
        WienerDriftDiffusion(contaminant=UniformResponseTimeContaminant(time_bounds=(0.05, 3.0)))
    with pytest.raises(ValueError, match="requires a contaminant model"):
        WienerDriftDiffusion().parameters_from_components(
            drift={"drift.intercept": 0.0},
            boundary=1.0,
            contaminant_probability=0.1,
        )


def test_contaminant_mixture_density_integrates_to_one() -> None:
    ddm = model()
    components = ddm.parameter_components(truth(ddm, probability=0.1))

    def density(response_time: float, choice: float) -> float:
        return float(
            np.exp(
                ddm._joint_log_density(
                    np.asarray([response_time]),
                    np.asarray([choice]),
                    np.asarray([0.8]),
                    components,
                )[0]
            )
        )

    masses = [
        quad(
            density,
            1e-8,
            20.0,
            args=(choice,),
            points=(0.05, 0.25, 3.0),
            epsabs=1e-9,
        )[0]
        for choice in (0.0, 1.0)
    ]

    assert sum(masses) == pytest.approx(1.0, abs=2e-8)


def test_simulation_retains_unexposed_contaminant_truth_and_is_reproducible() -> None:
    ddm = model()
    source = design(n_sessions=1, trials_per_session=1_000)
    parameters = truth(ddm, probability=0.2)

    first = ddm.simulate_with_contaminants(source, parameters, seed=402)
    second = ddm.simulate_with_contaminants(source, parameters, seed=402)

    np.testing.assert_array_equal(first.study["choice"], second.study["choice"])
    np.testing.assert_array_equal(first.study["response_time"], second.study["response_time"])
    np.testing.assert_array_equal(first.contaminants, second.contaminants)
    assert "contaminant" not in first.study.columns
    assert np.mean(first.contaminants) == pytest.approx(0.2, abs=0.04)
    contaminated_times = first.study["response_time"][first.contaminants]
    assert np.all((contaminated_times >= 0.05) & (contaminated_times <= 3.0))
    observed = ddm.simulate(source, parameters, seed=402)
    np.testing.assert_array_equal(observed["response_time"], first.study["response_time"])

    milliseconds_model = model(
        response_time=ResponseTimeSpec(column="rt_ms", unit=ResponseTimeUnit.MILLISECONDS)
    )
    milliseconds = milliseconds_model.simulate_with_contaminants(
        source,
        truth(milliseconds_model, probability=0.2),
        seed=402,
    )
    np.testing.assert_array_equal(milliseconds.contaminants, first.contaminants)
    np.testing.assert_allclose(
        milliseconds.study["rt_ms"],
        first.study["response_time"] * 1000.0,
    )


def test_fit_recovers_contaminant_weight_and_retains_trial_responsibilities() -> None:
    ddm = model()
    simulation = ddm.simulate_with_contaminants(design(), truth(ddm), seed=403)

    fit = ddm.fit(simulation.study)
    recovered = ddm.parameter_components(fit)
    posterior = ddm.contaminant_responsibility(simulation.study, fit)

    assert isinstance(fit, DriftDiffusionFitResult)
    assert fit.audit().status.value in {"pass", "warning"}
    assert fit.likelihood_floor_count == 0
    assert recovered.contaminant_probability == pytest.approx(0.06, abs=0.025)
    assert recovered.drift_map["drift.stimulus"] == pytest.approx(1.2, abs=0.25)
    assert recovered.boundary == pytest.approx(1.2, abs=0.15)
    assert recovered.nondecision_time == pytest.approx(0.25, abs=0.025)
    np.testing.assert_allclose(fit.posterior_contaminant_probability, posterior)
    assert fit.expected_contaminant_count == pytest.approx(float(np.sum(posterior)))
    assert np.mean(posterior[simulation.contaminants]) > 0.5
    assert np.mean(posterior[~simulation.contaminants]) < 0.1


def test_contaminant_model_supports_prospective_joint_scoring() -> None:
    ddm = model(n_restarts=2)
    simulation = ddm.simulate(
        design(n_sessions=4, trials_per_session=100),
        truth(ddm),
        seed=404,
    )

    evaluations = evaluate_splits(
        ddm,
        simulation,
        forward_session_splits(simulation, min_train_sessions=3),
    )

    assert len(evaluations) == 1
    assert len(evaluations[0].pointwise_log_probability) == 100
    assert np.all(np.isfinite(evaluations[0].pointwise_log_probability))


def test_contaminant_model_supports_generic_parameter_recovery() -> None:
    ddm = model(n_restarts=2)

    report = run_parameter_recovery(
        ddm,
        design(n_sessions=2, trials_per_session=200),
        [truth(ddm)],
        seed=405,
    )

    assert report.n_runs == 1
    assert report.parameter_names[-1] == "contaminant_probability"
    assert report.audit_failure_rate == 0.0


@pytest.mark.parametrize(
    "arguments",
    [
        {"time_bounds": (0.0, 2.0)},
        {"time_bounds": (0.1, 2.0), "probability_bounds": (0.0, 1.0)},
        {"time_bounds": (0.1, 2.0), "choice_probability": 1.0},
    ],
)
def test_contaminant_configuration_is_validated(arguments: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        UniformResponseTimeContaminant(**arguments)  # type: ignore[arg-type]
