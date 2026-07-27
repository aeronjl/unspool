import json

import numpy as np
import pytest

from unspool import (
    PosteriorGroup,
    PosteriorParameterQuantity,
    PosteriorResult,
    PosteriorVariable,
    SBCSimulation,
    Study,
    run_simulation_based_calibration,
)


def beta_binomial_simulator(seed: int) -> SBCSimulation:
    generator = np.random.default_rng(seed)
    probability = generator.uniform()
    n_trials = 20
    design = Study(
        {
            "subject": ["mouse"] * n_trials,
            "session": ["session"] * n_trials,
            "trial": np.arange(n_trials),
            "session_order": np.zeros(n_trials, dtype=int),
            "choice": generator.binomial(1, probability, size=n_trials),
        }
    )
    return SBCSimulation(design, {"probability": probability})


def beta_binomial_inference(study: Study, seed: int) -> PosteriorResult:
    generator = np.random.default_rng(seed)
    successes = int(np.sum(study["choice"]))
    failures = len(study) - successes
    draws = generator.beta(1 + successes, 1 + failures, size=(2, 200))
    variable = PosteriorVariable(
        "probability",
        draws,
        ("chain", "draw"),
        {"chain": np.arange(2), "draw": np.arange(200)},
    )
    return PosteriorResult(
        model_name="beta-binomial",
        model_signature="beta-binomial[uniform-prior]",
        inference_library="numpy",
        inference_library_version=np.__version__,
        parameter_names=("probability",),
        groups=(PosteriorGroup("posterior", (variable,)),),
    )


def test_sbc_conjugate_pipeline_is_reproducible_and_retains_raw_ranks() -> None:
    kwargs = {
        "simulator": beta_binomial_simulator,
        "inference": beta_binomial_inference,
        "quantities": (PosteriorParameterQuantity("probability"),),
        "repeats": 120,
        "seed": 419,
        "simulation_signature": "beta-binomial-prior-predictive[v1]",
        "inference_signature": "beta-binomial-conjugate[v1]",
        "thin": 2,
        "interval_probability": 0.9,
    }

    first = run_simulation_based_calibration(**kwargs)
    second = run_simulation_based_calibration(**kwargs)

    assert first == second
    assert first.n_successful == 120
    assert first.n_failed == 0
    assert len(first.ranks) == 120
    assert all(rank.n_posterior_draws == 200 for rank in first.ranks)
    assert all(0 <= rank.rank <= rank.n_posterior_draws for rank in first.ranks)
    summary = first.summary(bins=10)[0]
    assert summary.target == "probability"
    assert summary.n_replicates == 120
    assert abs(summary.mean_normalized_rank - 0.5) < 0.1
    assert 0.78 < summary.interval_coverage < 0.99
    assert sum(summary.histogram_counts) == 120
    json.dumps(first.to_dict(), allow_nan=False)


def test_sbc_vector_quantity_preserves_coordinate_targets() -> None:
    def simulator(seed: int) -> SBCSimulation:
        study = Study(
            {
                "subject": ["a", "b"],
                "session": ["s", "s"],
                "trial": [0, 0],
                "session_order": [0, 0],
            }
        )
        return SBCSimulation(study, {"bias": np.asarray([0.1, -0.1])})

    def inference(study: Study, seed: int) -> PosteriorResult:
        generator = np.random.default_rng(seed + 1)
        values = generator.normal(
            loc=np.asarray([0.1, -0.1]),
            scale=1.0,
            size=(2, 20, 2),
        )
        variable = PosteriorVariable(
            "bias",
            values,
            ("chain", "draw", "subject"),
            {"chain": [0, 1], "draw": np.arange(20), "subject": study.subjects},
        )
        return PosteriorResult(
            model_name="vector-test",
            model_signature="vector-test[v1]",
            inference_library="test",
            inference_library_version="1",
            parameter_names=("bias",),
            groups=(PosteriorGroup("posterior", (variable,)),),
        )

    report = run_simulation_based_calibration(
        simulator,
        inference,
        (PosteriorParameterQuantity("bias"),),
        repeats=2,
        seed=2,
        simulation_signature="vector-simulator[v1]",
        inference_signature="vector-inference[v1]",
    )

    assert {rank.target for rank in report.ranks} == {
        "bias[subject='a']",
        "bias[subject='b']",
    }
    assert all(rank.coordinate for rank in report.ranks)


def test_sbc_retains_failures_without_fabricating_ranks() -> None:
    calls = 0

    def inference(study: Study, seed: int) -> PosteriorResult:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("deliberate sampler failure")
        return beta_binomial_inference(study, seed)

    report = run_simulation_based_calibration(
        beta_binomial_simulator,
        inference,
        (PosteriorParameterQuantity("probability"),),
        repeats=3,
        seed=9,
        simulation_signature="beta-binomial-prior-predictive[v1]",
        inference_signature="failure-test[v1]",
    )

    assert report.n_successful == 2
    assert report.n_failed == 1
    assert report.failures[0].replicate == 1
    assert report.failures[0].stage == "inference"
    assert report.failures[0].error_type == "RuntimeError"
    assert "deliberate" in report.failures[0].message


def test_sbc_randomizes_exact_ties_deterministically() -> None:
    def tied_inference(study: Study, seed: int) -> PosteriorResult:
        del study, seed
        variable = PosteriorVariable(
            "probability",
            np.full((2, 20), 0.5),
            ("chain", "draw"),
            {"chain": [0, 1], "draw": np.arange(20)},
        )
        return PosteriorResult(
            model_name="tied-test",
            model_signature="tied-test[v1]",
            inference_library="test",
            inference_library_version="1",
            parameter_names=("probability",),
            groups=(PosteriorGroup("posterior", (variable,)),),
        )

    def fixed_simulator(seed: int) -> SBCSimulation:
        del seed
        simulation = beta_binomial_simulator(1)
        return SBCSimulation(simulation.study, {"probability": 0.5})

    kwargs = {
        "simulator": fixed_simulator,
        "inference": tied_inference,
        "quantities": (PosteriorParameterQuantity("probability"),),
        "repeats": 40,
        "seed": 31,
        "simulation_signature": "fixed-tie[v1]",
        "inference_signature": "fixed-tie-inference[v1]",
    }
    first = run_simulation_based_calibration(**kwargs)
    second = run_simulation_based_calibration(**kwargs)

    assert first == second
    assert len({rank.rank for rank in first.ranks}) > 1
    assert all(0 <= rank.rank <= 40 for rank in first.ranks)


def test_sbc_shape_errors_are_retained_as_evaluation_failures() -> None:
    def scalar_truth(seed: int) -> SBCSimulation:
        simulation = beta_binomial_simulator(seed)
        return SBCSimulation(simulation.study, {"probability": 0.5})

    def vector_posterior(study: Study, seed: int) -> PosteriorResult:
        result = beta_binomial_inference(study, seed)
        draws = np.repeat(result["posterior"]["probability"].values[..., None], 2, axis=-1)
        variable = PosteriorVariable(
            "probability",
            draws,
            ("chain", "draw", "condition"),
            {"chain": [0, 1], "draw": np.arange(200), "condition": ["a", "b"]},
        )
        return PosteriorResult(
            model_name="shape-test",
            model_signature="shape-test[v1]",
            inference_library="test",
            inference_library_version="1",
            parameter_names=("probability",),
            groups=(PosteriorGroup("posterior", (variable,)),),
        )

    report = run_simulation_based_calibration(
        scalar_truth,
        vector_posterior,
        (PosteriorParameterQuantity("probability"),),
        repeats=1,
        seed=4,
        simulation_signature="shape-simulation[v1]",
        inference_signature="shape-inference[v1]",
    )

    assert report.n_successful == 0
    assert report.n_failed == 1
    assert report.ranks == ()
    assert report.summary() == ()
    assert report.failures[0].stage == "evaluation"
    assert "intrinsic shape" in report.failures[0].message


def test_sbc_validates_pipeline_and_quantity_contracts() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        run_simulation_based_calibration(
            beta_binomial_simulator,
            beta_binomial_inference,
            (PosteriorParameterQuantity("probability"),),
            repeats=0,
            seed=1,
            simulation_signature="simulation",
            inference_signature="inference",
        )
    with pytest.raises(ValueError, match="posterior_name"):
        PosteriorParameterQuantity("")
