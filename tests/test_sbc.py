import json

import numpy as np
import pytest
from scipy import stats

from behavio import (
    PosteriorGroup,
    PosteriorParameterQuantity,
    PosteriorResult,
    PosteriorVariable,
    SBCSimulation,
    Study,
    run_simulation_based_calibration,
)
from behavio.posterior_diagnostics import PosteriorAuditPolicy
from behavio.sbc import SBCRank, SBCReport, _simultaneous_band

pytest.importorskip("arviz")


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
    assert first.n_successful + first.n_failed == 120
    assert first.n_failed == first.n_unconverged
    assert first.n_other_failures == 0
    assert len(first.ranks) == first.n_successful
    assert all(rank.n_posterior_draws == 200 for rank in first.ranks)
    assert all(0 <= rank.rank <= rank.n_posterior_draws for rank in first.ranks)
    summary = first.summary(bins=10)[0]
    assert summary.target == "probability"
    assert summary.n_replicates == first.n_successful
    assert summary.repeats_requested == 120
    assert abs(summary.mean_normalized_rank - 0.5) < 0.1
    assert 0.78 < summary.interval_coverage < 0.99
    assert sum(summary.histogram_counts) == first.n_successful
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
        audit_policy=None,
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
        # A constant chain has an undefined R-hat, so the convergence audit would exclude
        # every replicate; this test is about the tie arithmetic, not about convergence.
        "audit_policy": None,
    }
    first = run_simulation_based_calibration(**kwargs)
    second = run_simulation_based_calibration(**kwargs)

    assert first == second
    assert first.audit_policy is None
    assert first.n_unconverged == 0
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
    with pytest.raises(TypeError, match="audit_policy"):
        run_simulation_based_calibration(
            beta_binomial_simulator,
            beta_binomial_inference,
            (PosteriorParameterQuantity("probability"),),
            repeats=1,
            seed=1,
            simulation_signature="simulation",
            inference_signature="inference",
            audit_policy="strict",
        )


def divergent_inference(study: Study, seed: int, *, divergent: bool) -> PosteriorResult:
    """Return the conjugate posterior, optionally flagged as a divergent sampler run."""

    result = beta_binomial_inference(study, seed)
    variable = result["posterior"]["probability"]
    n_chains, n_draws = variable.values.shape
    diverging = np.zeros((n_chains, n_draws), dtype=bool)
    if divergent:
        diverging[0, :4] = True
    sample_stats = PosteriorGroup(
        "sample_stats",
        (
            PosteriorVariable(
                "diverging",
                diverging,
                ("chain", "draw"),
                {"chain": np.arange(n_chains), "draw": np.arange(n_draws)},
            ),
        ),
    )
    return PosteriorResult(
        model_name=result.model_name,
        model_signature=result.model_signature,
        inference_library=result.inference_library,
        inference_library_version=result.inference_library_version,
        parameter_names=result.parameter_names,
        groups=(result["posterior"], sample_stats),
    )


def test_sbc_excludes_divergent_replicates_from_the_rank_histogram() -> None:
    """Unconverged replicates are retained as coded failures, never pooled into ranks."""

    calls = 0

    def inference(study: Study, seed: int) -> PosteriorResult:
        nonlocal calls
        replicate = calls
        calls += 1
        return divergent_inference(study, seed, divergent=replicate in {1, 3})

    report = run_simulation_based_calibration(
        beta_binomial_simulator,
        inference,
        (PosteriorParameterQuantity("probability"),),
        repeats=6,
        seed=77,
        simulation_signature="beta-binomial-prior-predictive[v1]",
        inference_signature="divergent[v1]",
        audit_policy=PosteriorAuditPolicy(max_rhat=1.5, min_ess_bulk=1.0, min_ess_tail=1.0),
    )

    excluded = report.unconverged_replicates
    assert report.n_unconverged == 2
    assert report.n_other_failures == 0
    assert report.n_successful == 4
    assert set(excluded) == {1, 3}
    assert {rank.replicate for rank in report.ranks}.isdisjoint(excluded)
    for failure in report.failures:
        assert failure.stage == "audit"
        assert failure.error_type == "PosteriorAuditFailure"
        assert failure.audit_issue_codes == ("posterior.divergences",)
    summary = report.summary(bins=5)[0]
    assert summary.n_replicates == 4
    assert summary.repeats_requested == 6
    assert summary.n_unconverged == 2
    assert sum(summary.histogram_counts) == 4
    assert summary.retained_fraction == pytest.approx(4 / 6)
    payload = report.to_dict(bins=5)
    assert payload["n_unconverged"] == 2
    assert payload["unconverged_replicates"] == [1, 3]
    assert payload["audit_policy"]["max_rhat"] == 1.5
    json.dumps(payload, allow_nan=False)


def autocorrelated_report(*, thin: int):
    def simulator(seed: int) -> SBCSimulation:
        study = Study(
            {
                "subject": ["a"],
                "session": ["s"],
                "trial": [0],
                "session_order": [0],
            }
        )
        return SBCSimulation(study, {"theta": 0.0})

    def inference(study: Study, seed: int) -> PosteriorResult:
        del study
        generator = np.random.default_rng(seed)
        n_chains, n_draws, rho = 2, 400, 0.95
        values = np.empty((n_chains, n_draws))
        innovation = np.sqrt(1.0 - rho**2)
        for chain in range(n_chains):
            state = generator.normal()
            for draw in range(n_draws):
                state = rho * state + innovation * generator.normal()
                values[chain, draw] = state
        variable = PosteriorVariable(
            "theta",
            values,
            ("chain", "draw"),
            {"chain": np.arange(n_chains), "draw": np.arange(n_draws)},
        )
        return PosteriorResult(
            model_name="ar1",
            model_signature="ar1[rho=0.95]",
            inference_library="numpy",
            inference_library_version=np.__version__,
            parameter_names=("theta",),
            groups=(PosteriorGroup("posterior", (variable,)),),
        )

    return run_simulation_based_calibration(
        simulator,
        inference,
        (PosteriorParameterQuantity("theta"),),
        repeats=4,
        seed=5,
        simulation_signature="ar1-simulator[v1]",
        inference_signature="ar1-inference[v1]",
        thin=thin,
        audit_policy=None,
    )


def test_sbc_records_whether_thinning_achieved_near_independence() -> None:
    """An autocorrelated chain at thin=1 is recorded, not silently ranked as independent."""

    unthinned = autocorrelated_report(thin=1)
    thinned = autocorrelated_report(thin=20)

    assert all(rank.thinned_ess is not None for rank in unthinned.ranks)
    assert unthinned.summary()[0].mean_relative_ess < 0.15
    assert thinned.summary()[0].mean_relative_ess > 0.3
    assert thinned.summary()[0].min_relative_ess > 0.2
    assert unthinned.to_dict()["ranks"][0]["relative_ess"] < 0.15


def report_from_ranks(rank_values, *, n_posterior_draws: int) -> SBCReport:
    """Build a report whose ranks are fixed, so uniformity can be tested in isolation."""

    ranks = tuple(
        SBCRank(
            replicate=replicate,
            quantity_name="theta",
            quantity_signature="fixed-rank[v1]",
            target="theta",
            coordinate=(),
            truth=0.0,
            rank=int(value),
            n_posterior_draws=n_posterior_draws,
            posterior_mean=0.0,
            posterior_sd=1.0,
            interval=(-1.0, 1.0),
            covered=True,
        )
        for replicate, value in enumerate(rank_values)
    )
    return SBCReport(
        simulation_signature="fixed[v1]",
        inference_signature="fixed[v1]",
        quantity_signatures=("fixed-rank[v1]",),
        repeats_requested=len(ranks),
        root_seed=0,
        thin=1,
        interval_probability=0.9,
        ranks=ranks,
        failures=(),
    )


def test_sbc_ecdf_band_is_simultaneous_rather_than_pointwise() -> None:
    """Validate the null: the band must cover the whole curve at its nominal rate."""

    n_replicates, n_cells = 100, 100
    null_cdf = (np.arange(n_cells) + 1.0) / n_cells
    level, lower, upper = _simultaneous_band(
        null_cdf,
        n_replicates=n_replicates,
        confidence_level=0.95,
        n_band_simulations=4_000,
        band_seed=0,
    )
    generator = np.random.default_rng(20_240_517)
    cells = generator.integers(0, n_cells, size=(20_000, n_replicates))
    counts = np.cumsum(
        np.apply_along_axis(np.bincount, 1, cells, minlength=n_cells),
        axis=1,
    )
    pointwise_lower = stats.binom.ppf(0.025, n_replicates, null_cdf)
    pointwise_upper = stats.binom.isf(0.025, n_replicates, null_cdf)

    simultaneous_coverage = float(np.mean(np.all((counts >= lower) & (counts <= upper), axis=1)))
    pointwise_coverage = float(
        np.mean(np.all((counts >= pointwise_lower) & (counts <= pointwise_upper), axis=1))
    )

    assert level < 0.05
    assert 0.93 <= simultaneous_coverage <= 0.97
    assert pointwise_coverage < 0.75


def test_sbc_uniformity_sees_dispersion_errors_that_the_mean_rank_cannot() -> None:
    """A symmetric U-shape and cap both leave the mean at one half; the band does not."""

    n_posterior_draws = 99
    generator = np.random.default_rng(4)
    overdispersed = np.rint(generator.beta(0.35, 0.35, size=400) * n_posterior_draws)
    underdispersed = np.rint(generator.beta(4.0, 4.0, size=400) * n_posterior_draws)

    for values in (overdispersed, underdispersed):
        report = report_from_ranks(values, n_posterior_draws=n_posterior_draws)
        summary = report.summary()[0]
        uniformity = report.uniformity()[0]

        assert abs(summary.mean_normalized_rank - 0.5) < 0.03
        assert uniformity.null == "discrete-uniform"
        assert uniformity.n_posterior_draws == n_posterior_draws
        assert uniformity.chi_square_dof == 9
        assert uniformity.chi_square_p_value < 1e-6
        assert uniformity.n_points_outside_band > 0
        assert uniformity.max_absolute_difference > 0.05


def test_sbc_uniformity_keeps_calibrated_ranks_inside_its_band() -> None:
    generator = np.random.default_rng(11)
    calibrated = generator.integers(0, 100, size=400)

    report = report_from_ranks(calibrated, n_posterior_draws=99)
    uniformity = report.uniformity(bins=10, confidence_level=0.95)

    assert len(uniformity) == 1
    assert uniformity[0].n_points_outside_band == 0
    assert uniformity[0].chi_square_p_value > 0.01
    assert uniformity[0].min_expected_bin_count == pytest.approx(40.0)
    assert all(
        lower <= difference <= upper
        for lower, difference, upper in zip(
            uniformity[0].lower_difference_band,
            uniformity[0].ecdf_difference,
            uniformity[0].upper_difference_band,
            strict=True,
        )
    )
    assert report.uniformity() == uniformity
