import json

import numpy as np
import pytest

from behavio import (
    CategoryRateDiscrepancy,
    MeanDiscrepancy,
    PosteriorAuditPolicy,
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
from behavio.contracts import AuditSeverity
from behavio.posterior import PosteriorError
from behavio.posterior_predictive import PredictiveMultiplicity

CHAINS = 2
DRAWS = 500


def variable(name, values, dims, **coords):
    return PosteriorVariable(name=name, values=np.asarray(values), dims=dims, coords=coords)


def unaudited_policy() -> PosteriorAuditPolicy:
    """Declare, check by check, that unusable draws are being accepted anyway."""

    return PosteriorAuditPolicy(
        divergence_severity=AuditSeverity.WARNING,
        rhat_severity=AuditSeverity.WARNING,
        nonfinite_severity=AuditSeverity.WARNING,
    )


def predictive_result(*, mismatch: bool = False, unconverged: bool = False) -> PosteriorResult:
    generator = np.random.default_rng(317)
    chain = np.arange(CHAINS)
    draw = np.arange(DRAWS)
    trial = np.arange(8)
    observed_values = np.asarray([0, 1, 0, 1, 1, 0, 1, 0])
    probability = 0.98 if mismatch else 0.5
    predictive_values = generator.binomial(1, probability, size=(CHAINS, DRAWS, 8))
    bias_values = generator.normal(size=(CHAINS, DRAWS))
    diverging = np.zeros((CHAINS, DRAWS), dtype=bool)
    if unconverged:
        bias_values[0] += 8.0
        diverging[0, :17] = True
    posterior = PosteriorGroup(
        "posterior",
        (variable("bias", bias_values, ("chain", "draw"), chain=chain, draw=draw),),
    )
    sample_stats = PosteriorGroup(
        "sample_stats",
        (variable("diverging", diverging, ("chain", "draw"), chain=chain, draw=draw),),
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
        groups=(posterior, sample_stats, observed, predictive, constants),
    )


def many_group_result(
    *,
    n_groups: int,
    n_trials: int,
    seed: int,
    shift: float = 0.0,
) -> PosteriorResult:
    """Build a continuous outcome whose per-group tail probabilities are near-uniform.

    ``shift`` displaces every observed group by the same amount, so a non-zero shift is
    diffuse misfit that no single group carries.
    """

    generator = np.random.default_rng(seed)
    size = n_groups * n_trials
    chain = np.arange(CHAINS)
    draw = np.arange(DRAWS)
    trial = np.arange(size)
    labels = np.repeat([f"s{index:02d}" for index in range(n_groups)], n_trials)
    observed_values = generator.normal(shift, 1.0, size=size)
    predictive_values = generator.normal(0.0, 1.0, size=(CHAINS, DRAWS, size))
    posterior = PosteriorGroup(
        "posterior",
        (
            variable(
                "bias",
                generator.normal(size=(CHAINS, DRAWS)),
                ("chain", "draw"),
                chain=chain,
                draw=draw,
            ),
        ),
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
        (variable("trial_subject", labels, ("trial",), trial=trial),),
    )
    return PosteriorResult(
        model_name="well-specified",
        model_signature="well-specified[v1]",
        inference_library="test-backend",
        inference_library_version="1",
        parameter_names=("bias",),
        groups=(posterior, observed, predictive, constants),
    )


def test_global_checks_retain_reference_distributions_and_explicit_tails() -> None:
    pytest.importorskip("arviz")
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
    assert mean.replicated.shape == (CHAINS, DRAWS)
    assert not mean.replicated.flags.writeable
    assert mean.n_observations == 8
    assert mean.interval[0] <= mean.observed <= mean.interval[1]
    assert audit.convergence is not None
    assert audit.convergence.status is PosteriorAuditStatus.PASS
    json.dumps(audit.to_dict(), allow_nan=False)


def test_grouped_checks_preserve_labels_and_localize_mismatch() -> None:
    pytest.importorskip("arviz")
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
    assert audit.family.n_checks == 2
    assert audit.family.n_flagged == 2


def test_policy_and_contract_fail_loudly() -> None:
    with pytest.raises(ValueError, match="interval_probability"):
        PosteriorPredictivePolicy(interval_probability=1.0)
    with pytest.raises(ValueError, match="tail_probability_warning"):
        PosteriorPredictivePolicy(tail_probability_warning=0.5)
    with pytest.raises(ValueError, match="family_discovery_rate"):
        PosteriorPredictivePolicy(family_discovery_rate=0.0)
    with pytest.raises(TypeError, match="audit_policy"):
        posterior_predictive_check(
            predictive_result(),
            (MeanDiscrepancy(),),
            audit_policy=PosteriorPredictivePolicy(),  # type: ignore[arg-type]
        )
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


def test_unconverged_posterior_fails_the_check_without_discarding_the_evidence() -> None:
    pytest.importorskip("arviz")
    audit = posterior_predictive_check(
        predictive_result(unconverged=True),
        (MeanDiscrepancy(), VarianceDiscrepancy()),
    )

    assert audit.status is PosteriorAuditStatus.FAIL
    assert "ppc.unconverged-posterior" in audit.issue_codes
    failing = next(item for item in audit.issues if item.code == "ppc.unconverged-posterior")
    assert failing.severity is AuditSeverity.ERROR
    assert failing.discrepancy_signature is None
    assert audit.convergence is not None
    assert audit.convergence.status is PosteriorAuditStatus.FAIL
    assert set(audit.convergence.issue_codes) >= {"posterior.rhat", "posterior.divergences"}
    assert len(audit.checks) == 2
    assert audit.checks[0].replicated.shape == (CHAINS, DRAWS)
    json.dumps(audit.to_dict(), allow_nan=False)


def test_convergence_gate_is_injectable_and_a_downgrade_stays_in_the_artifact() -> None:
    pytest.importorskip("arviz")
    audit = posterior_predictive_check(
        predictive_result(unconverged=True),
        (MeanDiscrepancy(),),
        audit_policy=unaudited_policy(),
    )

    assert audit.status is PosteriorAuditStatus.WARNING
    assert "ppc.unconverged-posterior" not in audit.issue_codes
    assert "ppc.posterior-diagnostic-warning" in audit.issue_codes
    payload = audit.to_dict()
    assert payload["convergence"]["policy"]["rhat_severity"] == "warning"
    assert payload["convergence"]["policy"]["divergence_severity"] == "warning"
    assert payload["convergence"]["status"] == "warning"


def test_family_size_and_chance_expectation_are_always_reported() -> None:
    pytest.importorskip("arviz")
    audit = posterior_predictive_check(
        predictive_result(),
        (MeanDiscrepancy(), VarianceDiscrepancy(), CategoryRateDiscrepancy(1)),
        groupby=("trial_subject",),
    )

    family = audit.family
    assert family.n_checks == 6
    assert family.n_groups == 2
    assert family.n_discrepancies == 3
    assert family.multiplicity is PredictiveMultiplicity.BENJAMINI_HOCHBERG
    assert family.expected_extreme == pytest.approx(6 * 0.05)
    assert 0.0 <= family.excess_probability <= 1.0
    assert audit.to_dict()["family"]["n_checks"] == 6


def test_chance_level_extreme_checks_do_not_become_one_warning_each() -> None:
    pytest.importorskip("arviz")
    result = many_group_result(n_groups=60, n_trials=8, seed=23)
    discrepancies = (MeanDiscrepancy(),)

    adjusted = posterior_predictive_check(result, discrepancies, groupby=("trial_subject",))
    unadjusted = posterior_predictive_check(
        result,
        discrepancies,
        groupby=("trial_subject",),
        policy=PosteriorPredictivePolicy(multiplicity=PredictiveMultiplicity.NONE),
    )

    assert adjusted.family.n_checks == 60
    assert adjusted.family.n_extreme == 3
    assert adjusted.family.expected_extreme == pytest.approx(3.0)
    assert adjusted.family.excess_probability > 0.05
    assert adjusted.family.n_flagged == 0
    assert "ppc.extreme-discrepancy" not in adjusted.issue_codes
    assert "ppc.extreme-discrepancy-rate" not in adjusted.issue_codes
    assert adjusted.status is PosteriorAuditStatus.PASS

    assert unadjusted.family.n_extreme == adjusted.family.n_extreme
    assert unadjusted.issue_codes.count("ppc.extreme-discrepancy") == 3
    assert unadjusted.status is PosteriorAuditStatus.WARNING
    assert [check.tail_probability for check in unadjusted.checks] == [
        check.tail_probability for check in adjusted.checks
    ]


def test_diffuse_excess_is_promoted_to_one_summary_issue() -> None:
    pytest.importorskip("arviz")
    audit = posterior_predictive_check(
        many_group_result(n_groups=60, n_trials=8, seed=11, shift=0.4),
        (MeanDiscrepancy(),),
        groupby=("trial_subject",),
    )

    assert audit.family.n_extreme > audit.family.expected_extreme
    assert audit.family.excess_probability < 0.05
    assert audit.family.n_flagged == 0
    assert audit.issue_codes == ("ppc.extreme-discrepancy-rate",)
    summary = audit.issues[0]
    assert summary.discrepancy_signature is None
    assert "60 simultaneous checks" in summary.message


def test_bonferroni_is_available_and_stricter_than_the_raw_threshold() -> None:
    pytest.importorskip("arviz")
    audit = posterior_predictive_check(
        predictive_result(mismatch=True),
        (CategoryRateDiscrepancy(1),),
        groupby=("trial_subject",),
        policy=PosteriorPredictivePolicy(multiplicity=PredictiveMultiplicity.BONFERRONI),
    )

    assert audit.family.adjusted_threshold == pytest.approx(0.025)
    assert audit.family.n_flagged == 2
