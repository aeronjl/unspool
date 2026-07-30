import json

import numpy as np
import pytest

from behavio import audit_fit
from behavio.diagnostics import AuditSeverity, FitAuditPolicy, FitAuditStatus
from behavio.models import FitDiagnostics, FitResult, GLMHMMFitResult


def _diagnostics(**changes: object) -> FitDiagnostics:
    values = {
        "converged": True,
        "optimizer": "test optimizer",
        "status": 0,
        "message": "converged",
        "n_iterations": 4,
        "objective": 10.0,
        "gradient_norm": 1e-7,
        "hessian_condition": 20.0,
        "boundary_estimate": False,
    }
    values.update(changes)
    return FitDiagnostics(**values)  # type: ignore[arg-type]


def _fit(*, diagnostics: FitDiagnostics | None = None) -> FitResult:
    return FitResult(
        model_name="test-model",
        model_signature="test-model[v1]",
        parameter_names=("intercept",),
        estimates=np.array([0.1]),
        standard_errors=np.array([0.2]),
        covariance=np.array([[0.04]]),
        n_observations=100,
        diagnostics=_diagnostics() if diagnostics is None else diagnostics,
    )


def test_clean_fit_has_a_machine_readable_pass_audit() -> None:
    fit = _fit()

    audit = fit.audit()
    payload = audit.to_dict()

    assert audit.status is FitAuditStatus.PASS
    assert audit.issue_codes == ()
    assert audit.restarts is None
    assert audit.latent_states is None
    assert payload["status"] == "pass"
    assert payload["numerical"]["objective"] == 10.0
    json.dumps(payload)


def test_common_numerical_failures_and_warnings_remain_distinct() -> None:
    fit = _fit(
        diagnostics=_diagnostics(
            converged=False,
            message="iteration limit",
            hessian_condition=np.inf,
            boundary_estimate=True,
        )
    )

    audit = audit_fit(fit)

    assert audit.status is FitAuditStatus.FAIL
    assert audit.issue_codes == (
        "optimizer_nonconvergence",
        "nonfinite_hessian_condition",
        "boundary_estimate",
    )
    assert audit.issues[0].severity is AuditSeverity.ERROR
    assert all(issue.severity is AuditSeverity.WARNING for issue in audit.issues[1:])


def test_nonfinite_parameters_objective_gradient_and_uncertainty_are_audited() -> None:
    fit = FitResult(
        model_name="test-model",
        model_signature="test-model[nonfinite]",
        parameter_names=("intercept",),
        estimates=np.array([np.nan]),
        standard_errors=np.array([np.inf]),
        covariance=np.array([[np.nan]]),
        n_observations=100,
        diagnostics=_diagnostics(objective=np.inf, gradient_norm=np.nan),
    )

    audit = fit.audit()

    assert audit.status is FitAuditStatus.FAIL
    assert audit.issue_codes == (
        "nonfinite_estimates",
        "nonfinite_objective",
        "nonfinite_gradient",
        "nonfinite_uncertainty",
    )


def test_explicit_policy_controls_condition_and_restart_thresholds() -> None:
    fit = _fit(diagnostics=_diagnostics(hessian_condition=100.0))

    default = fit.audit()
    strict = fit.audit(policy=FitAuditPolicy(hessian_condition_warning=50.0))

    assert default.status is FitAuditStatus.PASS
    assert strict.status is FitAuditStatus.WARNING
    assert strict.issue_codes == ("ill_conditioned_hessian",)


def test_restart_and_latent_state_evidence_share_one_audit() -> None:
    fit = GLMHMMFitResult(
        model_name="bernoulli-glm-hmm",
        model_signature="bernoulli-glm-hmm[test]",
        parameter_names=("parameter",),
        estimates=np.array([0.1]),
        standard_errors=np.array([0.2]),
        covariance=np.array([[0.04]]),
        n_observations=100,
        diagnostics=_diagnostics(),
        restart_objectives=np.array([10.0, 10.5, np.inf]),
        restart_converged=np.array([True, True, False]),
        restart_messages=("converged", "converged", "iteration limit"),
        selected_restart=0,
        canonical_permutation=(1, 0),
        state_occupancy=np.array([0.995, 0.005]),
        state_separation=0.2,
        label_order_gap=0.0,
        label_ambiguous=True,
        low_occupancy=True,
    )

    audit = fit.audit()

    assert audit.status is FitAuditStatus.WARNING
    assert audit.issue_codes == (
        "restart_nonconvergence",
        "restart_objective_disagreement",
        "low_state_occupancy",
        "label_ambiguity",
    )
    assert audit.restarts is not None
    assert audit.restarts.n_converged == 2
    assert audit.restarts.failed_messages == ("iteration limit",)
    assert audit.restarts.relative_objective_range == pytest.approx(0.05)
    assert audit.latent_states is not None
    assert audit.latent_states.minimum_occupancy == pytest.approx(0.005)
    assert audit.latent_states.label_ambiguous


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("hessian_condition_warning", 0.0, "hessian_condition_warning"),
        ("restart_relative_objective_warning", -1.0, "restart_relative_objective_warning"),
    ],
)
def test_audit_policy_rejects_invalid_thresholds(field: str, value: float, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        FitAuditPolicy(**{field: value})
