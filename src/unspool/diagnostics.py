"""Model-agnostic auditing of numerical and model-specific fit evidence."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, cast

import numpy as np

from unspool.models.base import FitDiagnostics, FitResult


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


class AuditSeverity(StrEnum):
    """Consequence of one fit-audit issue."""

    WARNING = "warning"
    ERROR = "error"


class FitAuditStatus(StrEnum):
    """Overall numerical status derived from all retained fit evidence."""

    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"


@dataclass(frozen=True, slots=True)
class FitAuditPolicy:
    """Explicit thresholds used to turn continuous diagnostics into warnings."""

    hessian_condition_warning: float = 1e12
    restart_relative_objective_warning: float = 1e-3

    def __post_init__(self) -> None:
        if not np.isfinite(self.hessian_condition_warning) or self.hessian_condition_warning <= 0:
            raise ValueError("hessian_condition_warning must be finite and positive")
        if (
            not np.isfinite(self.restart_relative_objective_warning)
            or self.restart_relative_objective_warning < 0
        ):
            raise ValueError("restart_relative_objective_warning must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class FitIssue:
    """One stable, machine-readable reason that a fit needs attention."""

    code: str
    severity: AuditSeverity
    message: str

    def __post_init__(self) -> None:
        if not self.code or not self.message:
            raise ValueError("fit issues require a non-empty code and message")
        object.__setattr__(self, "severity", AuditSeverity(self.severity))


@dataclass(frozen=True, slots=True)
class RestartAudit:
    """Comparable summary of a multi-restart optimizer's retained outcomes."""

    n_restarts: int
    n_converged: int
    selected_restart: int
    selected_converged: bool
    selected_objective: float
    objective_range: float
    relative_objective_range: float
    failed_messages: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.n_restarts < 1:
            raise ValueError("n_restarts must be positive")
        if not 0 <= self.n_converged <= self.n_restarts:
            raise ValueError("n_converged must lie between zero and n_restarts")
        if not 0 <= self.selected_restart < self.n_restarts:
            raise ValueError("selected_restart must identify one restart")
        if self.objective_range < 0 or self.relative_objective_range < 0:
            raise ValueError("restart objective ranges must be non-negative")
        object.__setattr__(self, "failed_messages", tuple(self.failed_messages))


@dataclass(frozen=True, slots=True)
class LatentStateAudit:
    """Comparable summary of evidence specific to a latent-state fit."""

    n_states: int
    minimum_occupancy: float
    state_separation: float
    label_order_gap: float
    label_ambiguous: bool
    low_occupancy: bool

    def __post_init__(self) -> None:
        if self.n_states < 2:
            raise ValueError("latent-state audits require at least two states")
        if not 0 <= self.minimum_occupancy <= 1:
            raise ValueError("minimum_occupancy must lie between zero and one")
        if self.state_separation < 0 or self.label_order_gap < 0:
            raise ValueError("latent-state separation diagnostics must be non-negative")


@dataclass(frozen=True, slots=True)
class FitAudit:
    """One normalized audit of common, restart, and latent-state fit evidence."""

    model_name: str
    model_signature: str
    n_observations: int
    numerical: FitDiagnostics
    issues: tuple[FitIssue, ...]
    restarts: RestartAudit | None = None
    latent_states: LatentStateAudit | None = None

    def __post_init__(self) -> None:
        issues = tuple(self.issues)
        if not self.model_name or not self.model_signature:
            raise ValueError("fit audits require model name and signature")
        if self.n_observations < 1:
            raise ValueError("n_observations must be positive")
        codes = [issue.code for issue in issues]
        if len(set(codes)) != len(codes):
            raise ValueError("fit issue codes must be unique within one audit")
        object.__setattr__(self, "issues", issues)

    @property
    def status(self) -> FitAuditStatus:
        """Return fail, warning, or pass without collapsing the underlying issues."""

        if any(issue.severity is AuditSeverity.ERROR for issue in self.issues):
            return FitAuditStatus.FAIL
        if self.issues:
            return FitAuditStatus.WARNING
        return FitAuditStatus.PASS

    @property
    def issue_codes(self) -> tuple[str, ...]:
        """Stable issue codes suitable for filtering reports and recovery grids."""

        return tuple(issue.code for issue in self.issues)

    def to_dict(self) -> dict[str, Any]:
        """Return a new JSON-serializable representation of the complete audit."""

        diagnostics = self.numerical
        payload: dict[str, Any] = {
            "model_name": self.model_name,
            "model_signature": self.model_signature,
            "n_observations": self.n_observations,
            "status": self.status.value,
            "numerical": {
                "converged": diagnostics.converged,
                "optimizer": diagnostics.optimizer,
                "status": diagnostics.status,
                "message": diagnostics.message,
                "n_iterations": diagnostics.n_iterations,
                "objective": diagnostics.objective,
                "gradient_norm": diagnostics.gradient_norm,
                "hessian_condition": diagnostics.hessian_condition,
                "boundary_estimate": diagnostics.boundary_estimate,
            },
            "issues": [
                {
                    "code": issue.code,
                    "severity": issue.severity.value,
                    "message": issue.message,
                }
                for issue in self.issues
            ],
            "restarts": None,
            "latent_states": None,
        }
        if self.restarts is not None:
            payload["restarts"] = {
                "n_restarts": self.restarts.n_restarts,
                "n_converged": self.restarts.n_converged,
                "selected_restart": self.restarts.selected_restart,
                "selected_converged": self.restarts.selected_converged,
                "selected_objective": self.restarts.selected_objective,
                "objective_range": self.restarts.objective_range,
                "relative_objective_range": self.restarts.relative_objective_range,
                "failed_messages": list(self.restarts.failed_messages),
            }
        if self.latent_states is not None:
            payload["latent_states"] = {
                "n_states": self.latent_states.n_states,
                "minimum_occupancy": self.latent_states.minimum_occupancy,
                "state_separation": self.latent_states.state_separation,
                "label_order_gap": self.latent_states.label_order_gap,
                "label_ambiguous": self.latent_states.label_ambiguous,
                "low_occupancy": self.latent_states.low_occupancy,
            }
        return payload


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
    if not np.isfinite(diagnostics.objective):
        issues.append(
            FitIssue(
                code="nonfinite_objective",
                severity=AuditSeverity.ERROR,
                message="the selected objective value is non-finite",
            )
        )
    if not np.isfinite(diagnostics.gradient_norm):
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
    if not np.isfinite(diagnostics.hessian_condition):
        issues.append(
            FitIssue(
                code="nonfinite_hessian_condition",
                severity=AuditSeverity.WARNING,
                message="the local Hessian condition number is non-finite",
            )
        )
    elif diagnostics.hessian_condition >= policy.hessian_condition_warning:
        issues.append(
            FitIssue(
                code="ill_conditioned_hessian",
                severity=AuditSeverity.WARNING,
                message=(
                    f"the local Hessian is ill-conditioned ({diagnostics.hessian_condition:.3g})"
                ),
            )
        )
    if diagnostics.boundary_estimate:
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
