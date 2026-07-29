"""Fit evidence and its normalized audit vocabulary.

This module holds the *declarations* that both a fitting model and an auditing routine
have to agree on: the raw optimizer diagnostics attached to a fit
(:class:`FitDiagnostics`), the severity and status vocabulary, the audit policy, and the
audit record itself. ``behavio.diagnostics`` owns the *derivation* (``audit_fit``) and
re-exports every name defined here, so ``from behavio.diagnostics import FitAudit`` keeps
working.

``FitDiagnostics`` lives beside the audit types rather than beside :class:`FitResult`
because :class:`FitAudit` needs it structurally and :mod:`behavio.contracts.estimator`
imports this module, not the other way round.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np


class AuditSeverity(StrEnum):
    """Consequence of one audit issue."""

    WARNING = "warning"
    ERROR = "error"


class FitAuditStatus(StrEnum):
    """Overall numerical status derived from all retained fit evidence."""

    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"


@dataclass(frozen=True, slots=True)
class FitDiagnostics:
    """Optimizer and numerical diagnostics that remain attached to a fit.

    ``n_iterations``, ``objective``, ``gradient_norm``, ``hessian_condition`` and
    ``boundary_estimate`` are optimizer-shaped. They accept ``None`` to mean *this
    quantity does not exist for the procedure that produced the fit* -- for example a
    posterior projected to a point summary by
    :func:`behavio.contracts.posterior.posterior_point_summary`. ``None`` is deliberately
    distinct from a non-finite value: ``audit_fit`` reports non-finite diagnostics as
    issues and skips absent ones. Every field remains a required constructor argument, so
    a fit can never omit a diagnostic by accident.
    """

    converged: bool
    optimizer: str
    status: int
    message: str
    n_iterations: int | None
    objective: float | None
    gradient_norm: float | None
    hessian_condition: float | None
    boundary_estimate: bool | None


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
