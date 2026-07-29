import json

import numpy as np
import pytest

from behavio import (
    PosteriorAuditPolicy,
    PosteriorAuditStatus,
    PosteriorGroup,
    PosteriorResult,
    PosteriorVariable,
    audit_posterior,
)
from behavio.contracts import AuditSeverity
from behavio.posterior_diagnostics import PosteriorAuditIssue


def posterior_result(*, shifted: bool = False) -> PosteriorResult:
    generator = np.random.default_rng(209)
    chains = 4
    draws = 500
    values = generator.normal(size=(chains, draws, 2))
    if shifted:
        values[0, :, 0] += 4.0
    coords = {
        "chain": np.arange(chains),
        "draw": np.arange(draws),
        "coefficient": np.asarray(["intercept", "stimulus"]),
    }
    posterior = PosteriorGroup(
        "posterior",
        (
            PosteriorVariable(
                "beta",
                values,
                ("chain", "draw", "coefficient"),
                coords,
            ),
        ),
    )
    diverging = np.zeros((chains, draws), dtype=bool)
    treedepth = np.zeros((chains, draws), dtype=bool)
    if shifted:
        diverging[0, :3] = True
        treedepth[1, :2] = True
    sample_stats = PosteriorGroup(
        "sample_stats",
        (
            PosteriorVariable(
                "diverging",
                diverging,
                ("chain", "draw"),
                {"chain": coords["chain"], "draw": coords["draw"]},
            ),
            PosteriorVariable(
                "reached_max_treedepth",
                treedepth,
                ("chain", "draw"),
                {"chain": coords["chain"], "draw": coords["draw"]},
            ),
        ),
    )
    return PosteriorResult(
        model_name="test-model",
        model_signature="test-signature",
        inference_library="test-sampler",
        inference_library_version="1",
        parameter_names=("beta",),
        groups=(posterior, sample_stats),
    )


def test_well_mixed_posterior_passes_explicit_policy() -> None:
    pytest.importorskip("arviz")
    result = posterior_result()

    audit = audit_posterior(result)

    assert audit.status == PosteriorAuditStatus.PASS
    assert audit.issue_codes == ()
    assert audit.n_chains == 4
    assert audit.n_draws == 500
    assert audit.n_samples == 2_000
    assert audit.divergences == 0
    assert audit.max_treedepth_hits == 0
    assert len(audit.diagnostics) == 1
    diagnostic = audit.diagnostics[0]
    assert diagnostic.name == "beta"
    assert diagnostic.dims == ("coefficient",)
    assert np.max(diagnostic.rhat) <= 1.01
    assert np.min(diagnostic.ess_bulk) >= 400
    assert np.min(diagnostic.ess_tail) >= 400
    assert not diagnostic.rhat.flags.writeable
    json.dumps(audit.to_dict(), allow_nan=False)


def test_audit_retains_mixing_and_hmc_warnings_with_labelled_targets() -> None:
    pytest.importorskip("arviz")
    result = posterior_result(shifted=True)

    audit = audit_posterior(result)

    assert audit.status == PosteriorAuditStatus.FAIL
    assert audit.divergences == 3
    assert audit.max_treedepth_hits == 2
    assert set(audit.issue_codes) >= {
        "posterior.divergences",
        "posterior.max-treedepth",
        "posterior.rhat",
        "posterior.ess-bulk",
    }
    rhat_issue = next(issue for issue in audit.issues if issue.code == "posterior.rhat")
    assert "beta[coefficient='intercept']" in rhat_issue.targets


def test_policy_validation_and_thresholds_are_explicit() -> None:
    with pytest.raises(ValueError, match="greater than one"):
        PosteriorAuditPolicy(max_rhat=1.0)
    with pytest.raises(ValueError, match="positive"):
        PosteriorAuditPolicy(min_ess_bulk=0.0)
    with pytest.raises(ValueError, match="non-negative"):
        PosteriorAuditPolicy(max_divergences=-1)

    pytest.importorskip("arviz")
    audit = audit_posterior(
        posterior_result(),
        policy=PosteriorAuditPolicy(
            max_rhat=1.1,
            min_ess_bulk=10_000,
            min_ess_tail=10_000,
            max_divergences=0,
            max_treedepth_hits=0,
        ),
    )
    assert set(audit.issue_codes) == {"posterior.ess-bulk", "posterior.ess-tail"}


def test_divergences_and_rhat_are_errors_while_low_ess_is_only_a_warning() -> None:
    pytest.importorskip("arviz")

    failing = audit_posterior(posterior_result(shifted=True))
    severities = {issue.code: issue.severity for issue in failing.issues}
    assert severities["posterior.divergences"] is AuditSeverity.ERROR
    assert severities["posterior.rhat"] is AuditSeverity.ERROR
    assert severities["posterior.ess-bulk"] is AuditSeverity.WARNING
    assert severities["posterior.max-treedepth"] is AuditSeverity.WARNING
    assert failing.status is PosteriorAuditStatus.FAIL

    warning_only = audit_posterior(
        posterior_result(),
        policy=PosteriorAuditPolicy(min_ess_bulk=10_000, min_ess_tail=10_000),
    )
    assert set(warning_only.issue_codes) == {"posterior.ess-bulk", "posterior.ess-tail"}
    assert all(issue.severity is AuditSeverity.WARNING for issue in warning_only.issues)
    assert warning_only.status is PosteriorAuditStatus.WARNING


def test_severity_classification_is_an_overridable_policy_decision() -> None:
    pytest.importorskip("arviz")

    audit = audit_posterior(
        posterior_result(shifted=True),
        policy=PosteriorAuditPolicy(
            divergence_severity=AuditSeverity.WARNING,
            rhat_severity=AuditSeverity.WARNING,
            nonfinite_severity=AuditSeverity.WARNING,
        ),
    )

    assert audit.issue_codes  # the same issues are still detected
    assert audit.status is PosteriorAuditStatus.WARNING
    assert audit.policy.to_dict()["divergence_severity"] == "warning"
    assert audit.policy.to_dict()["ess_severity"] == "warning"
    assert audit.policy.to_dict()["rhat_severity"] == "warning"
    json.dumps(audit.to_dict(), allow_nan=False)


def test_strict_ess_policy_can_be_promoted_to_a_failure() -> None:
    pytest.importorskip("arviz")

    audit = audit_posterior(
        posterior_result(),
        policy=PosteriorAuditPolicy(
            min_ess_bulk=10_000,
            min_ess_tail=10_000,
            ess_severity=AuditSeverity.ERROR,
        ),
    )

    assert audit.status is PosteriorAuditStatus.FAIL


def test_issue_severity_defaults_to_warning_and_is_serialised() -> None:
    issue = PosteriorAuditIssue("posterior.custom", "something to look at")

    assert issue.severity is AuditSeverity.WARNING
    assert (
        PosteriorAuditIssue("posterior.custom", "worse", severity="error").severity
        is AuditSeverity.ERROR
    )
