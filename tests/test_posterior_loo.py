import json

import numpy as np
import pytest

from behavio import (
    PosteriorAuditStatus,
    PosteriorGroup,
    PosteriorResult,
    PosteriorVariable,
    PSISLOOResult,
    psis_loo,
)
from behavio.posterior import PosteriorError


def posterior_result(*, pathological: bool = False) -> PosteriorResult:
    generator = np.random.default_rng(281)
    chains = 4
    draws = 500
    trials = 12
    posterior_values = generator.normal(scale=0.2, size=(chains, draws))
    probability = 1.0 / (1.0 + np.exp(-posterior_values[..., None]))
    outcomes = np.asarray([0, 1] * (trials // 2))
    log_likelihood = np.where(
        outcomes[None, None, :] == 1,
        np.log(probability),
        np.log1p(-probability),
    )
    if pathological:
        log_likelihood[:, :, 3] = -np.exp(generator.normal(scale=1.2, size=(chains, draws)))
    sample_coords = {"chain": np.arange(chains), "draw": np.arange(draws)}
    posterior = PosteriorGroup(
        "posterior",
        (
            PosteriorVariable(
                "intercept",
                posterior_values,
                ("chain", "draw"),
                sample_coords,
            ),
        ),
    )
    likelihood_coords = {**sample_coords, "trial": np.arange(trials)}
    likelihood = PosteriorGroup(
        "log_likelihood",
        (
            PosteriorVariable(
                "choice",
                log_likelihood,
                ("chain", "draw", "trial"),
                likelihood_coords,
            ),
        ),
    )
    return PosteriorResult(
        model_name="test-model",
        model_signature="test-signature",
        inference_library="test-sampler",
        inference_library_version="1",
        parameter_names=("intercept",),
        groups=(posterior, likelihood),
    )


def test_psis_loo_retains_labelled_pointwise_evidence() -> None:
    pytest.importorskip("arviz")

    result = psis_loo(posterior_result())

    assert isinstance(result, PSISLOOResult)
    assert result.status is PosteriorAuditStatus.PASS
    assert result.issue_codes == ()
    assert result.log_likelihood_name == "choice"
    assert result.dims == ("trial",)
    assert result.n_samples == 2_000
    assert result.n_data_points == 12
    assert result.pointwise_elpd.shape == (12,)
    assert result.pareto_k.shape == (12,)
    assert not result.pointwise_elpd.flags.writeable
    assert np.sum(result.pointwise_elpd) == pytest.approx(result.elpd_loo)
    json.dumps(result.to_dict(), allow_nan=False)


def test_psis_loo_localizes_unreliable_importance_sampling() -> None:
    pytest.importorskip("arviz")

    with pytest.warns(UserWarning, match="Pareto"):
        result = psis_loo(posterior_result(pathological=True))

    assert result.status is PosteriorAuditStatus.WARNING
    assert "psis.high-pareto-k" in result.issue_codes
    issue = next(item for item in result.issues if item.code == "psis.high-pareto-k")
    assert "choice[trial=3]" in issue.targets
    assert result.pareto_k[3] > result.good_k


def test_psis_loo_requires_pointwise_finite_named_likelihood() -> None:
    result = posterior_result()
    posterior = result["posterior"]
    sample_coords = {"chain": np.arange(4), "draw": np.arange(500)}
    aggregated = PosteriorResult(
        model_name=result.model_name,
        model_signature=result.model_signature,
        inference_library=result.inference_library,
        inference_library_version=result.inference_library_version,
        parameter_names=result.parameter_names,
        groups=(
            posterior,
            PosteriorGroup(
                "log_likelihood",
                (
                    PosteriorVariable(
                        "choice",
                        np.zeros((4, 500)),
                        ("chain", "draw"),
                        sample_coords,
                    ),
                ),
            ),
        ),
    )
    with pytest.raises(PosteriorError, match="pointwise rather than aggregated"):
        psis_loo(aggregated)

    values = np.array(result["log_likelihood"]["choice"].values, copy=True)
    values[0, 0, 0] = np.nan
    nonfinite = PosteriorResult(
        model_name=result.model_name,
        model_signature=result.model_signature,
        inference_library=result.inference_library,
        inference_library_version=result.inference_library_version,
        parameter_names=result.parameter_names,
        groups=(
            posterior,
            PosteriorGroup(
                "log_likelihood",
                (
                    PosteriorVariable(
                        "choice",
                        values,
                        ("chain", "draw", "trial"),
                        {
                            **sample_coords,
                            "trial": np.arange(12),
                        },
                    ),
                ),
            ),
        ),
    )
    with pytest.raises(PosteriorError, match="only finite"):
        psis_loo(nonfinite)

    with pytest.raises(PosteriorError, match="no variable"):
        psis_loo(result, log_likelihood_name="response_time")
