import numpy as np
import pytest

from behavio import (
    BernoulliGLMHMM,
    HierarchicalSessionDynamicBernoulliGLMHMM,
    LabHierarchicalSessionDynamicBernoulliGLMHMM,
    PyMCBernoulliGLMHMM,
    SessionDynamicBernoulliGLMHMM,
    Study,
)
from behavio.contracts.posterior import PosteriorBehaviourEstimator
from behavio.pymc_backend import PyMCBackendError


def lab_design(*, sessions: int = 2, trials: int = 1) -> Study:
    subjects = ("north-0", "north-1", "south-0", "south-1")
    study = Study.factorial(trials=trials, subjects=subjects, sessions=sessions)
    columns = {name: study[name] for name in study.columns}
    columns["lab"] = np.repeat(
        ("north", "south"),
        2 * sessions * trials,
    )
    return Study(columns)


def test_configuration_declares_only_normalized_priors_and_sampled_contract() -> None:
    sampled = PyMCBernoulliGLMHMM(
        BernoulliGLMHMM(choice_lags=0, n_restarts=1),
        draws=5,
        tune=6,
        chains=2,
        cores=1,
        seed=4,
    )

    assert isinstance(sampled, PosteriorBehaviourEstimator)
    assert sampled.backend_name == "pymc.NUTS/marginalized-HMM"
    assert sampled.prior_specification == {
        "emission_normal_scale": 2.5,
        "initial_dirichlet_concentration": 1.0,
        "transition_dirichlet_concentration": 1.0,
    }
    assert "priors=" in sampled.signature
    with pytest.raises(PyMCBackendError, match="emission_prior_scale"):
        PyMCBernoulliGLMHMM(sampled.model, emission_prior_scale=0.0)
    with pytest.raises(PyMCBackendError, match="at least two"):
        PyMCBernoulliGLMHMM(sampled.model, chains=1)


@pytest.mark.parametrize(
    ("model", "design", "expected_truth"),
    [
        (
            BernoulliGLMHMM(choice_lags=0, n_restarts=1),
            Study.factorial(trials=3, subjects="animal", sessions=2),
            {"emission_coefficients"},
        ),
        (
            SessionDynamicBernoulliGLMHMM(choice_lags=0, n_restarts=1),
            Study.factorial(trials=3, subjects="animal", sessions=2),
            {"emission_step_scale", "session_emission_coefficients"},
        ),
        (
            HierarchicalSessionDynamicBernoulliGLMHMM(choice_lags=0, n_restarts=1),
            Study.factorial(trials=2, subjects=("a", "b"), sessions=2),
            {
                "population_emission_coefficients",
                "population_emission_step_scale",
                "subject_deviation_coefficients",
                "subject_emission_scale",
            },
        ),
        (
            LabHierarchicalSessionDynamicBernoulliGLMHMM(choice_lags=0, n_restarts=1),
            lab_design(),
            {
                "population_emission_coefficients",
                "lab_deviation_coefficients",
                "lab_emission_scale",
                "lab_emission_step_scale",
                "subject_deviation_coefficients",
                "subject_emission_scale",
            },
        ),
    ],
)
def test_prior_joint_simulates_every_supported_hierarchy_depth(
    model: BernoulliGLMHMM,
    design: Study,
    expected_truth: set[str],
) -> None:
    sampled = PyMCBernoulliGLMHMM(model, draws=2, tune=2, chains=2)

    first = sampled.prior_predictive_simulation(design, seed=21)
    second = sampled.prior_predictive_simulation(design, seed=21)

    assert expected_truth.issubset(first.truth)
    assert {"initial_probabilities", "global_transition_matrix"}.issubset(first.truth)
    np.testing.assert_array_equal(first.study["choice"], second.study["choice"])
    for name in first.truth:
        np.testing.assert_allclose(first.truth[name], second.truth[name])


def test_real_static_posterior_marginalizes_states_and_relabels_complete_draws() -> None:
    pytest.importorskip("pymc")
    sampled = PyMCBernoulliGLMHMM(
        BernoulliGLMHMM(choice_lags=0, n_restarts=1),
        draws=5,
        tune=5,
        chains=2,
        cores=1,
        seed=28,
    )
    simulation = sampled.prior_predictive_simulation(
        Study.factorial(trials=4, subjects="animal", sessions=2),
        seed=29,
    )

    posterior = sampled.sample(simulation.study)

    assert posterior.parameter_names == (
        "initial_probabilities",
        "global_transition_matrix",
        "emission_coefficients",
    )
    assert {
        "posterior",
        "sample_stats",
        "log_likelihood",
        "posterior_predictive",
        "observed_data",
        "constant_data",
    }.issubset(posterior.group_names)
    emissions = posterior["posterior"]["emission_coefficients"].values[..., 0]
    assert np.all(np.diff(emissions, axis=-1) >= 0.0)
    probability = posterior["posterior"]["choice_probability"].values
    choice = np.asarray(simulation.study["choice"])
    expected = choice * np.log(probability) + (1 - choice) * np.log1p(-probability)
    np.testing.assert_allclose(
        posterior["log_likelihood"]["choice"].values,
        expected,
        rtol=1e-10,
        atol=1e-10,
    )
    assert posterior.attrs["state_sequence"].startswith("analytically marginalized")
    assert posterior.attrs["all_declared_priors_normalized"]
    prediction = sampled.predict(simulation.study, posterior)
    score = sampled.pointwise_log_prob(simulation.study, posterior)
    assert prediction.probability.shape == score.shape == (len(simulation.study),)
    assert np.all(np.isfinite(score))


def test_real_lab_hierarchy_samples_paths_scales_transitions_and_label_ambiguity() -> None:
    pytest.importorskip("pymc")
    sampled = PyMCBernoulliGLMHMM(
        LabHierarchicalSessionDynamicBernoulliGLMHMM(
            choice_lags=0,
            n_restarts=1,
        ),
        draws=3,
        tune=3,
        chains=2,
        cores=1,
        seed=31,
    )
    simulation = sampled.prior_predictive_simulation(lab_design(sessions=2), seed=32)

    posterior = sampled.sample(simulation.study)

    expected = {
        "initial_probabilities",
        "global_transition_matrix",
        "session_transition_matrices",
        "session_emission_coefficients",
        "population_emission_coefficients",
        "subject_deviation_coefficients",
        "lab_deviation_coefficients",
        "population_emission_step_scale",
        "lab_emission_scale",
        "lab_emission_step_scale",
        "subject_emission_scale",
        "emission_step_scale",
        "session_transition_concentration",
    }
    assert expected == set(posterior.parameter_names)
    assert posterior["posterior"]["session_emission_coefficients"].dims == (
        "chain",
        "draw",
        "path_session",
        "state",
        "coefficient",
    )
    assert posterior["posterior"]["lab_deviation_coefficients"].dims == (
        "chain",
        "draw",
        "lab_path",
        "state",
        "coefficient",
    )
    assert posterior["posterior"]["label_permutation"].dims == (
        "chain",
        "draw",
        "state",
    )
    assert posterior["posterior"]["label_path_crossing"].values.shape == (2, 3)
    assert posterior["posterior"]["label_ambiguous"].values.shape == (2, 3)
    assert posterior.attrs["hierarchy_parameterization"] == ("non-centred Gaussian innovations")
    assert posterior.attrs["dynamic_prediction_scope"] == "fitted subject-session blocks only"
    assert np.all(np.isfinite(sampled.pointwise_log_prob(simulation.study, posterior)))
    future = sampled.prior_predictive_simulation(lab_design(sessions=3), seed=42).study

    with pytest.raises(PyMCBackendError, match="unseen-session"):
        sampled.predict(future, posterior)
