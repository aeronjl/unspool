import json

import numpy as np
import pytest

from unspool import (
    CategoryRateDiscrepancy,
    MeanDiscrepancy,
    PosteriorAuditStatus,
    PosteriorGroup,
    PosteriorPredictivePolicy,
    PosteriorResult,
    PosteriorVariable,
    PredictiveTail,
    SwitchRateDiscrepancy,
    VarianceDiscrepancy,
    posterior_predictive_check,
)
from unspool.posterior import PosteriorError


def variable(name, values, dims, **coords):
    return PosteriorVariable(name=name, values=np.asarray(values), dims=dims, coords=coords)


def predictive_result(*, mismatch: bool = False) -> PosteriorResult:
    generator = np.random.default_rng(317)
    chain = np.arange(2)
    draw = np.arange(200)
    trial = np.arange(8)
    observed_values = np.asarray([0, 1, 0, 1, 1, 0, 1, 0])
    probability = 0.98 if mismatch else 0.5
    predictive_values = generator.binomial(1, probability, size=(2, 200, 8))
    posterior = PosteriorGroup(
        "posterior",
        (variable("bias", np.zeros((2, 200)), ("chain", "draw"), chain=chain, draw=draw),),
    )
    observed = PosteriorGroup(
        "observed_data",
        (variable("choice", observed_values, ("trial",), trial=trial),),
    )
    predictive = PosteriorGroup(
        "posterior_predictive",
        (
            variable(
                "choice",
                predictive_values,
                ("chain", "draw", "trial"),
                chain=chain,
                draw=draw,
                trial=trial,
            ),
        ),
    )
    constants = PosteriorGroup(
        "constant_data",
        (
            variable(
                "trial_subject",
                ["a", "a", "a", "a", "b", "b", "b", "b"],
                ("trial",),
                trial=trial,
            ),
        ),
    )
    return PosteriorResult(
        model_name="test-model",
        model_signature="test-signature",
        inference_library="test-backend",
        inference_library_version="1",
        parameter_names=("bias",),
        groups=(posterior, observed, predictive, constants),
    )


def test_global_checks_retain_reference_distributions_and_explicit_tails() -> None:
    audit = posterior_predictive_check(
        predictive_result(),
        (
            MeanDiscrepancy(),
            VarianceDiscrepancy(),
            CategoryRateDiscrepancy(1),
            SwitchRateDiscrepancy(),
        ),
    )

    assert audit.status is PosteriorAuditStatus.PASS
    assert audit.variable_name == "choice"
    assert len(audit.checks) == 4
    mean = audit.checks[0]
    assert mean.observed == pytest.approx(0.5)
    assert mean.tail is PredictiveTail.TWO_SIDED
    assert mean.replicated.shape == (2, 200)
    assert not mean.replicated.flags.writeable
    assert mean.n_observations == 8
    assert mean.interval[0] <= mean.observed <= mean.interval[1]
    json.dumps(audit.to_dict(), allow_nan=False)


def test_grouped_checks_preserve_labels_and_localize_mismatch() -> None:
    audit = posterior_predictive_check(
        predictive_result(mismatch=True),
        (CategoryRateDiscrepancy(1),),
        groupby=("trial_subject",),
    )

    assert audit.status is PosteriorAuditStatus.WARNING
    assert audit.issue_codes == (
        "ppc.extreme-discrepancy",
        "ppc.extreme-discrepancy",
    )
    assert [check.group for check in audit.checks] == [
        (("trial_subject", "a"),),
        (("trial_subject", "b"),),
    ]
    assert all(check.n_observations == 4 for check in audit.checks)
    assert all(issue.group for issue in audit.issues)


def test_policy_and_contract_fail_loudly() -> None:
    with pytest.raises(ValueError, match="interval_probability"):
        PosteriorPredictivePolicy(interval_probability=1.0)
    with pytest.raises(ValueError, match="tail_probability_warning"):
        PosteriorPredictivePolicy(tail_probability_warning=0.5)
    with pytest.raises(ValueError, match="unique"):
        posterior_predictive_check(
            predictive_result(),
            (MeanDiscrepancy(), MeanDiscrepancy()),
        )
    with pytest.raises(PosteriorError, match="no grouping variable"):
        posterior_predictive_check(
            predictive_result(),
            (MeanDiscrepancy(),),
            groupby=("session",),
        )
