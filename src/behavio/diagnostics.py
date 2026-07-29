"""Model-agnostic auditing of numerical and model-specific fit evidence.

The audit *vocabulary* -- :class:`~behavio.contracts.audit.AuditSeverity`,
:class:`~behavio.contracts.audit.FitAuditStatus`,
:class:`~behavio.contracts.audit.FitAuditPolicy`,
:class:`~behavio.contracts.audit.FitIssue`,
:class:`~behavio.contracts.audit.RestartAudit`,
:class:`~behavio.contracts.audit.LatentStateAudit` and
:class:`~behavio.contracts.audit.FitAudit` -- now lives in :mod:`behavio.contracts.audit`
and is re-exported here, so every existing ``from behavio.diagnostics import ...`` keeps
working. This module owns the *derivation*: :func:`audit_fit`.

Breaking the cycle
------------------
``behavio.diagnostics`` used to import ``FitResult``/``FitDiagnostics`` from
``behavio.models.base``, while ``behavio.models.base`` type-imported ``FitAudit`` and
``FitAuditPolicy`` back from here and ran a function-local ``from behavio.diagnostics
import audit_fit`` inside ``FitResult.audit()``. That was the package's only import cycle.

The dependency is now inverted rather than deferred. ``behavio.contracts`` is a leaf and
declares both the audit vocabulary and a :class:`~behavio.contracts.estimator.FitAuditor`
protocol. This module depends on ``behavio.contracts`` in one direction only and registers
:func:`audit_fit` as the implementation at import time, so ``FitResult.audit()`` dispatches
through the registry instead of importing anything at call time. There is no module-level
cycle and no function-local import left.
"""

from __future__ import annotations

from typing import Any, Protocol, cast

import numpy as np

from behavio.contracts.audit import (
    AuditSeverity,
    FitAudit,
    FitAuditPolicy,
    FitAuditStatus,
    FitDiagnostics,
    FitIssue,
    LatentStateAudit,
    RestartAudit,
)
from behavio.contracts.estimator import FitResult, register_fit_auditor

__all__ = [
    "AuditSeverity",
    "FitAudit",
    "FitAuditPolicy",
    "FitAuditStatus",
    "FitDiagnostics",
    "FitIssue",
    "LatentStateAudit",
    "RestartAudit",
    "audit_fit",
]


class _RestartFit(Protocol):
    restart_objectives: Any
    restart_converged: Any
    restart_messages: tuple[str, ...]
    selected_restart: int


class _LatentStateFit(Protocol):
    state_occupancy: Any
    state_separation: float
    label_order_gap: float
    label_ambiguous: bool
    low_occupancy: bool


def audit_fit(fit: FitResult, *, policy: FitAuditPolicy | None = None) -> FitAudit:
    """Derive one audit while leaving every raw diagnostic attached to ``fit``."""

    if not isinstance(fit, FitResult):
        raise TypeError("fit must be a FitResult")
    selected_policy = FitAuditPolicy() if policy is None else policy
    if not isinstance(selected_policy, FitAuditPolicy):
        raise TypeError("policy must be a FitAuditPolicy")

    issues = _common_issues(fit, selected_policy)
    restarts = _restart_audit(fit)
    if restarts is not None:
        if restarts.n_converged < restarts.n_restarts:
            severity = AuditSeverity.ERROR if restarts.n_converged == 0 else AuditSeverity.WARNING
            issues.append(
                FitIssue(
                    code="restart_nonconvergence",
                    severity=severity,
                    message=(
                        f"{restarts.n_restarts - restarts.n_converged} of "
                        f"{restarts.n_restarts} optimizer restarts did not converge"
                    ),
                )
            )
        if restarts.relative_objective_range > selected_policy.restart_relative_objective_warning:
            issues.append(
                FitIssue(
                    code="restart_objective_disagreement",
                    severity=AuditSeverity.WARNING,
                    message=(
                        "converged restarts reached materially different objective values "
                        f"(relative range {restarts.relative_objective_range:.3g})"
                    ),
                )
            )

    latent_states = _latent_state_audit(fit)
    if latent_states is not None:
        if latent_states.low_occupancy:
            issues.append(
                FitIssue(
                    code="low_state_occupancy",
                    severity=AuditSeverity.WARNING,
                    message=(
                        "at least one latent state has low fitted occupancy "
                        f"(minimum {latent_states.minimum_occupancy:.3g})"
                    ),
                )
            )
        if latent_states.label_ambiguous:
            issues.append(
                FitIssue(
                    code="label_ambiguity",
                    severity=AuditSeverity.WARNING,
                    message=(
                        "the fitted label-order gap is too small for stable state naming "
                        f"({latent_states.label_order_gap:.3g})"
                    ),
                )
            )

    return FitAudit(
        model_name=fit.model_name,
        model_signature=fit.model_signature,
        n_observations=fit.n_observations,
        numerical=fit.diagnostics,
        issues=tuple(issues),
        restarts=restarts,
        latent_states=latent_states,
    )


def _common_issues(fit: FitResult, policy: FitAuditPolicy) -> list[FitIssue]:
    """Derive the issues common to every fit.

    Optimizer-shaped diagnostics are optional: ``None`` means the procedure that
    produced the fit has no such quantity -- a sampler projected by
    :func:`behavio.contracts.posterior.posterior_point_summary`, for example. Absent
    diagnostics are skipped rather than reported, while non-finite ones are still
    reported exactly as before, so maximum-likelihood audits are unchanged.
    """

    diagnostics = fit.diagnostics
    issues: list[FitIssue] = []
    if not diagnostics.converged:
        issues.append(
            FitIssue(
                code="optimizer_nonconvergence",
                severity=AuditSeverity.ERROR,
                message=f"optimizer did not converge: {diagnostics.message}",
            )
        )
    if not np.all(np.isfinite(fit.estimates)):
        issues.append(
            FitIssue(
                code="nonfinite_estimates",
                severity=AuditSeverity.ERROR,
                message="one or more parameter estimates are non-finite",
            )
        )
    if diagnostics.objective is not None and not np.isfinite(diagnostics.objective):
        issues.append(
            FitIssue(
                code="nonfinite_objective",
                severity=AuditSeverity.ERROR,
                message="the selected objective value is non-finite",
            )
        )
    if diagnostics.gradient_norm is not None and not np.isfinite(diagnostics.gradient_norm):
        issues.append(
            FitIssue(
                code="nonfinite_gradient",
                severity=AuditSeverity.ERROR,
                message="the selected gradient norm is non-finite",
            )
        )
    if not np.all(np.isfinite(fit.standard_errors)) or not np.all(np.isfinite(fit.covariance)):
        issues.append(
            FitIssue(
                code="nonfinite_uncertainty",
                severity=AuditSeverity.WARNING,
                message="the local covariance or standard errors contain non-finite values",
            )
        )
    elif np.any(fit.standard_errors < 0):
        issues.append(
            FitIssue(
                code="invalid_standard_errors",
                severity=AuditSeverity.WARNING,
                message="one or more approximate standard errors are negative",
            )
        )
    condition = diagnostics.hessian_condition
    if condition is not None and not np.isfinite(condition):
        issues.append(
            FitIssue(
                code="nonfinite_hessian_condition",
                severity=AuditSeverity.WARNING,
                message="the local Hessian condition number is non-finite",
            )
        )
    elif condition is not None and condition >= policy.hessian_condition_warning:
        issues.append(
            FitIssue(
                code="ill_conditioned_hessian",
                severity=AuditSeverity.WARNING,
                message=f"the local Hessian is ill-conditioned ({condition:.3g})",
            )
        )
    if diagnostics.boundary_estimate is True:
        issues.append(
            FitIssue(
                code="boundary_estimate",
                severity=AuditSeverity.WARNING,
                message="one or more estimates meet the model's boundary-warning rule",
            )
        )
    return issues


def _restart_audit(fit: FitResult) -> RestartAudit | None:
    required = (
        "restart_objectives",
        "restart_converged",
        "restart_messages",
        "selected_restart",
    )
    if not all(hasattr(fit, field) for field in required):
        return None
    restarted_fit = cast(_RestartFit, fit)
    objectives = np.asarray(restarted_fit.restart_objectives, dtype=np.float64)
    converged = np.asarray(restarted_fit.restart_converged, dtype=np.bool_)
    messages = tuple(restarted_fit.restart_messages)
    selected = int(restarted_fit.selected_restart)
    eligible = objectives[converged & np.isfinite(objectives)]
    if len(eligible) >= 2:
        objective_range = float(np.max(eligible) - np.min(eligible))
        relative_range = objective_range / max(1.0, abs(float(np.min(eligible))))
    else:
        objective_range = 0.0
        relative_range = 0.0
    return RestartAudit(
        n_restarts=len(objectives),
        n_converged=int(np.count_nonzero(converged)),
        selected_restart=selected,
        selected_converged=bool(converged[selected]),
        selected_objective=float(objectives[selected]),
        objective_range=objective_range,
        relative_objective_range=relative_range,
        failed_messages=tuple(
            message
            for message, successful in zip(messages, converged, strict=True)
            if not successful
        ),
    )


def _latent_state_audit(fit: FitResult) -> LatentStateAudit | None:
    required = (
        "state_occupancy",
        "state_separation",
        "label_order_gap",
        "label_ambiguous",
        "low_occupancy",
    )
    if not all(hasattr(fit, field) for field in required):
        return None
    latent_fit = cast(_LatentStateFit, fit)
    occupancy = np.asarray(latent_fit.state_occupancy, dtype=np.float64)
    return LatentStateAudit(
        n_states=len(occupancy),
        minimum_occupancy=float(np.min(occupancy)),
        state_separation=float(latent_fit.state_separation),
        label_order_gap=float(latent_fit.label_order_gap),
        label_ambiguous=bool(latent_fit.label_ambiguous),
        low_occupancy=bool(latent_fit.low_occupancy),
    )


register_fit_auditor(audit_fit)
