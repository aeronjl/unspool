"""A drift-diffusion model mixed with a uniform response process.

This file used to test a ``contaminant=`` slot on ``WienerDriftDiffusion``. The slot is
gone: a contaminant is a mixture with a process that emits a response without making a
decision, and that is :func:`behavio.compose.mix` with
:class:`~behavio.compose.UniformResponseGuess`. What is tested here is that the general
combinator does everything the special case did -- a normalized joint component density, a
mixture density that still integrates to one, recoverable weights, retained per-trial
responsibilities, and reproducible simulation that keeps the mixture truth out of the study.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scipy.integrate import quad

from behavio import (
    BehaviourModel,
    DriftDiffusionFitResult,
    ResponseTimeSpec,
    ResponseTimeUnit,
    Study,
    UniformResponseGuess,
    WienerDriftDiffusion,
    evaluate_splits,
    forward_session_splits,
    mix,
    run_parameter_recovery,
)
from behavio.compose import MixtureModel
from behavio.contracts.compose import linear_predictor


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


def model(*, weight_bounds: tuple[float, float] = (0.0, 0.25), **changes: Any) -> MixtureModel:
    arguments: dict[str, Any] = {
        "covariates": ("stimulus",),
        "nondecision_time_bounds": (0.1, 0.6),
        "n_restarts": 4,
        "max_iterations": 400,
        "simulation_time_step": 0.0002,
    }
    arguments.update(changes)
    response_time = arguments.get("response_time", ResponseTimeSpec())
    return mix(
        WienerDriftDiffusion(**arguments),
        UniformResponseGuess(time_bounds=(0.05, 3.0), response_time=response_time),
        weight_bounds=weight_bounds,
        n_restarts=4,
    )


def truth(mixed: MixtureModel, *, rate: float = 0.06):
    return mixed.from_natural(
        {
            "drift.intercept": 0.2,
            "drift.stimulus": 1.2,
            "boundary": 1.2,
            "starting_bias": 0.45,
            "nondecision_time": 0.25,
            "contaminant_rate": rate,
        }
    )


def test_the_uniform_response_process_is_a_normalized_fixed_support_joint_density() -> None:
    component = UniformResponseGuess(time_bounds=(0.1, 2.1), choice_probability=0.3)
    outcomes = np.asarray([[0.0, 1.0], [1.0, 1.0], [1.0, 3.0]])

    density = component.pointwise_log_density(design(n_sessions=1, trials_per_session=3), outcomes)

    assert np.exp(density[0]) == pytest.approx(0.7 / 2.0)
    assert np.exp(density[1]) == pytest.approx(0.3 / 2.0)
    assert np.isneginf(density[2])
    assert (np.exp(density[0]) + np.exp(density[1])) * 2.0 == pytest.approx(1.0)


def test_the_mixed_configuration_and_its_one_extra_parameter_are_explicit() -> None:
    mixed = model()

    assert isinstance(mixed, BehaviourModel)
    assert mixed.model_name == "contaminant-wiener-drift-diffusion"
    assert mixed.parameter_names[-1] == "mixture_logit"
    assert mixed.natural_names[-1] == "contaminant_rate"
    assert "time_bounds=(0.05, 3.0)" in mixed.signature
    assert mixed.to_natural(mixed.parameter_vector(truth(mixed)))[
        "contaminant_rate"
    ] == pytest.approx(0.06)
    with pytest.raises(ValueError, match="strictly inside"):
        model(weight_bounds=(0.0, 0.05)).from_natural(
            {
                "drift.intercept": 0.0,
                "drift.stimulus": 0.0,
                "boundary": 1.0,
                "starting_bias": 0.5,
                "nondecision_time": 0.1,
                "contaminant_rate": 0.1,
            }
        )


def test_the_mixed_density_integrates_to_one() -> None:
    mixed = model()
    study = design(n_sessions=1, trials_per_session=1)
    likelihood = mixed.likelihood
    coordinate = mixed.parameter_vector(truth(mixed, rate=0.1))

    def density(response_time: float, choice: float) -> float:
        one_row = Study(
            {
                **{name: study[name] for name in study.columns},
                "stimulus": np.asarray([0.5]),
                "choice": np.asarray([choice], dtype=np.int8),
                "response_time": np.asarray([response_time]),
            }
        )
        cells = linear_predictor(
            mixed.design_matrix(one_row), coordinate, mixed.predictor_offsets(one_row)
        )
        return float(np.exp(likelihood.pointwise_log_prob(cells, mixed.outcomes(one_row))[0]))

    masses = [
        quad(density, 1e-8, 20.0, args=(choice,), points=(0.05, 0.25, 3.0), epsabs=1e-9)[0]
        for choice in (0.0, 1.0)
    ]

    assert sum(masses) == pytest.approx(1.0, abs=2e-8)


def test_simulation_retains_unexposed_mixture_truth_and_is_reproducible() -> None:
    mixed = model()
    source = design(n_sessions=1, trials_per_session=1_000)
    parameters = truth(mixed, rate=0.2)

    first = mixed.simulate_with_component(source, parameters, seed=402)
    second = mixed.simulate_with_component(source, parameters, seed=402)

    np.testing.assert_array_equal(first.study["choice"], second.study["choice"])
    np.testing.assert_array_equal(first.study["response_time"], second.study["response_time"])
    np.testing.assert_array_equal(first.from_component, second.from_component)
    assert "from_component" not in first.study.columns
    assert np.mean(first.from_component) == pytest.approx(0.2, abs=0.04)
    contaminated = first.study["response_time"][first.from_component]
    assert np.all((contaminated >= 0.05) & (contaminated <= 3.0))
    observed = mixed.simulate(source, parameters, seed=402)
    np.testing.assert_array_equal(observed["response_time"], first.study["response_time"])

    milliseconds_model = model(
        response_time=ResponseTimeSpec(column="rt_ms", unit=ResponseTimeUnit.MILLISECONDS)
    )
    milliseconds = milliseconds_model.simulate_with_component(
        source, truth(milliseconds_model, rate=0.2), seed=402
    )
    np.testing.assert_array_equal(milliseconds.from_component, first.from_component)
    np.testing.assert_allclose(milliseconds.study["rt_ms"], first.study["response_time"] * 1000.0)


def test_fitting_recovers_the_weight_and_assigns_per_trial_responsibility() -> None:
    mixed = model()
    simulation = mixed.simulate_with_component(design(), truth(mixed), seed=403)

    fit = mixed.fit(simulation.study)
    recovered = mixed.to_natural(fit.estimates)
    posterior = mixed.component_responsibility(simulation.study, fit)

    assert isinstance(fit, DriftDiffusionFitResult)
    assert fit.audit().status.value in {"pass", "warning"}
    assert recovered["contaminant_rate"] == pytest.approx(0.06, abs=0.0197)
    assert recovered["drift.stimulus"] == pytest.approx(1.2, abs=0.129)
    assert recovered["boundary"] == pytest.approx(1.2, abs=0.042)
    assert recovered["nondecision_time"] == pytest.approx(0.25, abs=0.0058)
    assert mixed.weight(fit) == pytest.approx(recovered["contaminant_rate"])
    assert np.mean(posterior[simulation.from_component]) > 0.5
    assert np.mean(posterior[~simulation.from_component]) < 0.1


def test_a_mixed_drift_diffusion_model_supports_prospective_joint_scoring() -> None:
    mixed = model(n_restarts=2)
    simulation = mixed.simulate(
        design(n_sessions=4, trials_per_session=100), truth(mixed), seed=404
    )

    evaluations = evaluate_splits(
        mixed, simulation, forward_session_splits(simulation, min_train_sessions=3)
    )

    assert len(evaluations) == 1
    assert len(evaluations[0].pointwise_log_probability) == 100
    assert np.all(np.isfinite(evaluations[0].pointwise_log_probability))


def test_a_mixed_drift_diffusion_model_supports_generic_parameter_recovery() -> None:
    mixed = model(n_restarts=2)

    report = run_parameter_recovery(
        mixed, design(n_sessions=2, trials_per_session=200), [truth(mixed)], seed=405
    )

    assert report.n_runs == 1
    assert report.parameter_names[-1] == "mixture_logit"
    assert report.audit_failure_rate == 0.0


@pytest.mark.parametrize(
    "arguments",
    [
        {"time_bounds": (0.0, 2.0)},
        {"time_bounds": (2.0, 0.1)},
        {"time_bounds": (0.1, 2.0), "choice_probability": 1.0},
    ],
)
def test_the_component_configuration_is_validated(arguments: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        UniformResponseGuess(**arguments)
