import importlib

import numpy as np
import pytest

from behavio import (
    BernoulliHistoryGLM,
    CategoryRateDiscrepancy,
    ChoiceSpec,
    PyMCBackendError,
    PyMCHierarchicalGLMBackend,
    PyMCUnavailableError,
    Study,
    TaskSpec,
    TaskValidationError,
    posterior_predictive_check,
)
from behavio.compose import HierarchicalModel, hierarchical


def hierarchical_study() -> tuple[HierarchicalModel, Study, TaskSpec]:
    generator = np.random.default_rng(17)
    n_trials = 30
    design = Study(
        {
            "subject": ["mouse-a"] * n_trials + ["mouse-b"] * n_trials,
            "session": ["session-0"] * (2 * n_trials),
            "trial": list(range(n_trials)) * 2,
            "session_order": [0] * (2 * n_trials),
            "stimulus": generator.normal(size=2 * n_trials),
        }
    )
    model = hierarchical(
        BernoulliHistoryGLM(covariates=("stimulus",), choice_lags=1, l2=0.25),
        over="subject",
        scale=0.4,
    )
    study = model.simulate(
        design,
        {"intercept": -0.2, "stimulus": 1.0, "choice_lag_1": 0.25},
        seed=18,
    )
    task = TaskSpec(choice=ChoiceSpec(options=(0, 1)), predictors=("stimulus",))
    return model, study, task


def test_backend_configuration_is_complete_immutable_and_validated() -> None:
    backend = PyMCHierarchicalGLMBackend(
        draws=50,
        tune=60,
        chains=2,
        cores=1,
        target_accept=0.85,
        seed=4,
    )

    assert backend.backend_name == "pymc.NUTS"
    assert backend.backend_config == {
        "draws": 50,
        "tune": 60,
        "chains": 2,
        "cores": 1,
        "target_accept": 0.85,
        "seed": 4,
        "nuts_sampler": "pymc",
        "init": "jitter+adapt_diag",
    }
    with pytest.raises(TypeError):
        backend.backend_config["draws"] = 1
    with pytest.raises(PyMCBackendError, match="at least two"):
        PyMCHierarchicalGLMBackend(chains=1)
    with pytest.raises(PyMCBackendError, match="cannot exceed"):
        PyMCHierarchicalGLMBackend(chains=2, cores=3)
    with pytest.raises(PyMCBackendError, match="target_accept"):
        PyMCHierarchicalGLMBackend(target_accept=1.0)


def test_adapter_rejects_undeclared_covariates_and_empirical_bayes_scale() -> None:
    model, study, _ = hierarchical_study()
    undeclared = TaskSpec(choice=ChoiceSpec(options=(0, 1)))

    with pytest.raises(TaskValidationError, match="without a declared task role"):
        PyMCHierarchicalGLMBackend(draws=10, tune=10, chains=2).sample(
            model, study, task=undeclared
        )

    empirical_bayes = hierarchical(
        BernoulliHistoryGLM(covariates=("stimulus",), choice_lags=1),
        over="subject",
        scale=0.4,
        estimate_scale=True,
    )
    with pytest.raises(PyMCBackendError, match="no declared full-posterior scale prior"):
        PyMCHierarchicalGLMBackend(draws=10, tune=10, chains=2).sample(
            empirical_bayes,
            study,
            task=TaskSpec(choice=ChoiceSpec(options=(0, 1)), predictors=("stimulus",)),
        )


def test_real_pymc_fit_preserves_model_task_likelihood_and_predictive_evidence() -> None:
    pytest.importorskip("pymc")
    model, study, task = hierarchical_study()
    backend = PyMCHierarchicalGLMBackend(
        draws=30,
        tune=30,
        chains=2,
        cores=1,
        target_accept=0.8,
        seed=29,
    )

    result = backend.sample(model, study, task=task)

    assert result.model_name == model.model_name
    assert result.model_signature == model.signature
    assert result.inference_library == "PyMC"
    assert result.n_chains == 2
    assert result.n_draws == 30
    assert {
        "posterior",
        "sample_stats",
        "log_likelihood",
        "posterior_predictive",
        "observed_data",
        "constant_data",
    }.issubset(result.group_names)
    assert result.parameter_names == ("population_coefficient", "subject_deviation")
    assert result["posterior"]["population_coefficient"].dims == (
        "chain",
        "draw",
        "coefficient",
    )
    assert result["posterior"]["subject_deviation"].dims == (
        "chain",
        "draw",
        "subject",
        "coefficient",
    )
    np.testing.assert_array_equal(
        result["posterior"]["subject_deviation"].coords["subject"],
        ["mouse-a", "mouse-b"],
    )
    np.testing.assert_array_equal(
        result["posterior"]["population_coefficient"].coords["coefficient"],
        model.parameter_names,
    )
    assert not np.any(result["sample_stats"]["diverging"].values)
    assert result.attrs["backend_config"]["draws"] == 30
    assert result.attrs["task_validation"]["n_trials"] == len(study)
    assert result.attrs["scored_columns"] == ("choice",)

    probability = result["posterior"]["choice_probability"].values
    outcomes = np.asarray(study["choice"], dtype=np.float64)
    expected_log_likelihood = outcomes * np.log(probability)
    expected_log_likelihood += (1.0 - outcomes) * np.log1p(-probability)
    np.testing.assert_allclose(
        result["log_likelihood"]["choice"].values,
        expected_log_likelihood,
        rtol=1e-10,
        atol=1e-10,
    )
    predictive = result["posterior_predictive"]["choice"].values
    assert predictive.shape == (2, 30, len(study))
    assert set(np.unique(predictive)) <= {0, 1}
    np.testing.assert_array_equal(
        result["constant_data"]["trial_subject"].values,
        study["subject"],
    )
    np.testing.assert_array_equal(
        result["constant_data"]["trial_session"].values,
        study["session"],
    )
    np.testing.assert_array_equal(
        result["constant_data"]["trial_in_session"].values,
        study["trial"],
    )
    predictive_audit = posterior_predictive_check(
        result,
        (CategoryRateDiscrepancy(1),),
        groupby=("trial_subject",),
    )
    assert len(predictive_audit.checks) == 2
    assert {check.group for check in predictive_audit.checks} == {
        (("trial_subject", "mouse-a"),),
        (("trial_subject", "mouse-b"),),
    }


def test_pymc_remains_an_optional_dependency(monkeypatch) -> None:
    backend_module = importlib.import_module("behavio.pymc_backend")
    real_import = backend_module.importlib.import_module
    model, study, task = hierarchical_study()

    def unavailable(name):
        if name == "pymc":
            raise ImportError("not installed")
        return real_import(name)

    monkeypatch.setattr(backend_module.importlib, "import_module", unavailable)

    with pytest.raises(PyMCUnavailableError, match=r"behavio\[bayesian\]"):
        PyMCHierarchicalGLMBackend(draws=10, tune=10, chains=2).sample(model, study, task=task)
