from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pytest

from unspool import (
    ModelDataError,
    SmoothDriftDiffusionFitResult,
    SmoothWienerDriftDiffusion,
    Study,
    WienerDriftDiffusion,
    evaluate_splits,
    forward_session_splits,
    run_parameter_recovery,
)
from unspool.models.ddm import _wiener_log_density


def make_design(
    *,
    n_sessions: int = 5,
    trials_per_session: int = 100,
    subjects: tuple[str, ...] = ("a",),
    seed: int = 123,
) -> Study:
    generator = np.random.default_rng(seed)
    rows_per_subject = n_sessions * trials_per_session
    return Study(
        {
            "subject": [subject for subject in subjects for _ in range(rows_per_subject)],
            "session": [
                f"session-{session}"
                for _ in subjects
                for session in range(n_sessions)
                for _ in range(trials_per_session)
            ],
            "trial": list(range(trials_per_session)) * n_sessions * len(subjects),
            "session_order": [
                session
                for _ in subjects
                for session in range(n_sessions)
                for _ in range(trials_per_session)
            ],
            "stimulus": generator.normal(size=rows_per_subject * len(subjects)),
        }
    )


def smooth_model(
    *,
    n_sessions: int = 5,
    varying_parameters: tuple[str, ...] = ("drift.stimulus", "boundary"),
    shared_trajectory: bool = False,
    tolerance: float = 1e-8,
) -> SmoothWienerDriftDiffusion:
    return SmoothWienerDriftDiffusion(
        covariates=("stimulus",),
        knots=tuple(float(value) for value in range(n_sessions)),
        varying_parameters=varying_parameters,
        smoothness=10.0,
        n_restarts=2,
        max_iterations=300,
        simulation_time_step=0.001,
        shared_trajectory=shared_trajectory,
        tolerance=tolerance,
    )


def smooth_truth(model: SmoothWienerDriftDiffusion):
    n_knots = len(model.knots)
    return model.parameters_from_paths(
        {
            "drift.intercept": 0.1,
            "drift.stimulus": np.linspace(0.4, 1.6, n_knots),
            "boundary": np.linspace(1.5, 1.0, n_knots),
            "starting_bias": 0.48,
            "nondecision_time": 0.25,
        }
    )


def test_wiener_density_accepts_aligned_trialwise_boundaries_and_biases() -> None:
    times = np.asarray([0.2, 0.5, 0.8])
    choices = np.asarray([0.0, 1.0, 1.0])
    drifts = np.asarray([-0.2, 0.5, 1.1])
    boundaries = np.asarray([0.8, 1.1, 1.4])
    biases = np.asarray([0.3, 0.5, 0.7])

    vectorized = _wiener_log_density(
        times,
        choices,
        drifts,
        boundary=boundaries,
        starting_bias=biases,
        terms=12,
    )
    scalar = np.asarray(
        [
            _wiener_log_density(
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


def test_smooth_ddm_has_stable_mixed_path_parameterization() -> None:
    model = smooth_model(n_sessions=3)

    assert model.base_parameter_names == (
        "drift.intercept",
        "drift.stimulus",
        "boundary",
        "starting_bias",
        "nondecision_time",
    )
    assert model.parameter_names == (
        "drift.intercept",
        "drift.stimulus[session_order=0]",
        "drift.stimulus[session_order=1]",
        "drift.stimulus[session_order=2]",
        "boundary[session_order=0]",
        "boundary[session_order=1]",
        "boundary[session_order=2]",
        "starting_bias",
        "nondecision_time",
    )
    parameters = smooth_truth(model)
    assert isinstance(parameters, MappingProxyType)
    assert parameters["drift.stimulus[session_order=2]"] == 1.6


@pytest.mark.parametrize(
    "arguments",
    [
        {"knots": (0.0,)},
        {"knots": (0.0, 0.0)},
        {"knots": (1.0, 0.0)},
        {"knots": (0.0, np.nan)},
        {"smoothness": 0.0},
        {"time": "choice"},
        {"shared_trajectory": 1},
        {"varying_parameters": ("nondecision_time",)},
        {"varying_parameters": ("boundary", "boundary")},
    ],
)
def test_smooth_ddm_configuration_is_validated(arguments: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        SmoothWienerDriftDiffusion(**arguments)  # type: ignore[arg-type]


def test_simulation_and_fitting_recover_inspectable_parameter_trajectories() -> None:
    model = smooth_model()
    design = make_design(trials_per_session=150)
    study = model.simulate(design, smooth_truth(model), seed=77)

    fit = model.fit(study)
    trajectory = model.parameter_trajectory(fit)

    assert isinstance(fit, SmoothDriftDiffusionFitResult)
    assert fit.diagnostics.converged
    assert trajectory.clock == "session_order"
    assert trajectory.parameter_names == model.base_parameter_names
    assert trajectory.values.shape == (5, 5)
    stimulus = trajectory.path("drift.stimulus")
    boundary = trajectory.path("boundary")
    assert np.sqrt(np.mean((stimulus - np.linspace(0.4, 1.6, 5)) ** 2)) < 0.35
    assert np.sqrt(np.mean((boundary - np.linspace(1.5, 1.0, 5)) ** 2)) < 0.2
    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        trajectory.values.setflags(write=True)


def test_unobserved_future_knots_carry_forward_without_outcome_leakage() -> None:
    model = smooth_model(n_sessions=5, tolerance=1e-12)
    design = make_design(n_sessions=4, trials_per_session=100)
    parameters = smooth_truth(model)
    study = model.simulate(design, parameters, seed=6)

    trajectory = model.parameter_trajectory(model.fit(study))

    np.testing.assert_allclose(trajectory.values[-1], trajectory.values[-2], atol=1e-5)


def test_fixed_knots_cover_the_clock_and_subject_alignment_is_explicit() -> None:
    model = smooth_model(n_sessions=3)
    parameters = smooth_truth(model)

    with pytest.raises(ModelDataError, match="fixed knot range"):
        model.simulate(make_design(n_sessions=4), parameters, seed=1)

    multi_subject = make_design(
        n_sessions=3,
        trials_per_session=20,
        subjects=("a", "b"),
    )
    with pytest.raises(ModelDataError, match="subject-specific by default"):
        model.simulate(multi_subject, parameters, seed=1)

    shared = smooth_model(n_sessions=3, shared_trajectory=True)
    assert len(shared.simulate(multi_subject, smooth_truth(shared), seed=1).subjects) == 2


def test_smooth_ddm_supports_prospective_evaluation_and_parameter_recovery() -> None:
    model = smooth_model(n_sessions=4)
    design = make_design(n_sessions=4, trials_per_session=70)
    simulated = model.simulate(design, smooth_truth(model), seed=21)

    evaluations = evaluate_splits(
        model,
        simulated,
        forward_session_splits(simulated, min_train_sessions=3),
    )
    recovery = run_parameter_recovery(
        model,
        make_design(n_sessions=4, trials_per_session=60),
        [smooth_truth(model)],
        seed=22,
    )

    assert len(evaluations) == 1
    assert len(evaluations[0].pointwise_log_probability) == 70
    assert recovery.n_runs == 1
    assert recovery.parameter_names == model.parameter_names


def test_changing_paths_can_beat_a_static_ddm_on_future_sessions() -> None:
    n_sessions = 7
    design = make_design(n_sessions=n_sessions, trials_per_session=100)
    smooth = smooth_model(n_sessions=n_sessions)
    study = smooth.simulate(design, smooth_truth(smooth), seed=39)
    splits = forward_session_splits(study, min_train_sessions=4)
    static = WienerDriftDiffusion(
        covariates=("stimulus",),
        n_restarts=2,
        max_iterations=300,
        simulation_time_step=0.001,
    )

    static_loss = -np.mean(
        np.concatenate(
            [item.pointwise_log_probability for item in evaluate_splits(static, study, splits)]
        )
    )
    smooth_loss = -np.mean(
        np.concatenate(
            [item.pointwise_log_probability for item in evaluate_splits(smooth, study, splits)]
        )
    )

    assert smooth_loss < static_loss
