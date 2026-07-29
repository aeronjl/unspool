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
from typing import Any, Protocol, runtime_checkable

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

    @classmethod
    def closed_form(
        cls,
        *,
        procedure: str,
        message: str,
        objective: float | None = None,
        boundary_estimate: bool | None = None,
    ) -> FitDiagnostics:
        """Diagnostics for an estimator that solves rather than searches.

        A closed-form estimator -- the equal-variance z-transform, say -- has no optimizer,
        no iterations, no gradient and no Hessian, and forcing it to fill those fields in
        makes it advertise a convergence that was never in question. This constructor
        names the *procedure* instead of an optimizer and leaves every optimizer-shaped
        diagnostic absent. ``converged`` and ``status`` remain ``True`` and ``0``: there
        was nothing to converge, so the audit must not report a failure, and
        :func:`behavio.diagnostics.audit_fit` reads them exactly as it always has. The
        record is bit-identical to writing the same fields by hand, so no maximum-
        likelihood behaviour changes.
        """

        if not isinstance(procedure, str) or not procedure:
            raise ValueError("a closed-form procedure needs a non-empty name")
        return cls(
            converged=True,
            optimizer=procedure,
            status=0,
            message=message,
            n_iterations=None,
            objective=objective,
            gradient_norm=None,
            hessian_condition=None,
            boundary_estimate=boundary_estimate,
        )


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


@runtime_checkable
class MultistartFit(Protocol):
    """A fit result that retains every restart of a deterministic multistart optimizer.

    This is the contract behind :class:`RestartAudit`. It was previously a *private*
    protocol inside ``behavio.diagnostics``, so eight models populated these four exact
    attribute names by copying an implementation rather than by reading a contract, and a
    ninth got a restart audit only because its author matched the names character for
    character. The names are the contract, and they are now declared where a model author
    will find them.

    The raw restart evidence stays on the fit rather than being replaced by its summary:
    ``RestartAudit`` is a comparable digest, and a scientific requirement of this package
    is that optimization failures remain visible in full.
    """

    restart_objectives: Any
    """One objective value per restart, in restart order."""

    restart_converged: Any
    """One convergence flag per restart, aligned with ``restart_objectives``."""

    restart_messages: tuple[str, ...]
    """One optimizer message per restart, aligned with ``restart_objectives``."""

    selected_restart: int
    """Index of the restart whose estimates the fit reports."""


@runtime_checkable
class LatentStateFit(Protocol):
    """A fit result that retains the label and occupancy evidence of a latent-state model.

    The counterpart of :class:`MultistartFit` for :class:`LatentStateAudit`, promoted out
    of ``behavio.diagnostics`` for the same reason and on the same terms.
    """

    state_occupancy: Any
    """Fitted occupancy of each latent state."""

    state_separation: float
    """Non-negative separation between the fitted state emissions."""

    label_order_gap: float
    """Non-negative margin by which the canonical state ordering is decided."""

    label_ambiguous: bool
    """Whether the fitted ordering is too close to name states stably."""

    low_occupancy: bool
    """Whether any state's fitted occupancy is too low to interpret."""


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

    @classmethod
    def from_fit(cls, fit: Any) -> RestartAudit | None:
        """Summarize a :class:`MultistartFit`, or return ``None`` for any other fit.

        The relative objective range is scaled by the best retained objective, floored at
        one, so a model whose objective is near zero cannot manufacture a disagreement
        warning out of a rounding difference. Only converged restarts with finite
        objectives are eligible, and fewer than two of them means there is nothing to
        disagree about.
        """

        if not isinstance(fit, MultistartFit):
            return None
        objectives = np.asarray(fit.restart_objectives, dtype=np.float64)
        converged = np.asarray(fit.restart_converged, dtype=np.bool_)
        messages = tuple(fit.restart_messages)
        selected = int(fit.selected_restart)
        eligible = objectives[converged & np.isfinite(objectives)]
        if len(eligible) >= 2:
            objective_range = float(np.max(eligible) - np.min(eligible))
            relative_range = objective_range / max(1.0, abs(float(np.min(eligible))))
        else:
            objective_range = 0.0
            relative_range = 0.0
        return cls(
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

    @classmethod
    def from_fit(cls, fit: Any) -> LatentStateAudit | None:
        """Summarize a :class:`LatentStateFit`, or return ``None`` for any other fit."""

        if not isinstance(fit, LatentStateFit):
            return None
        occupancy = np.asarray(fit.state_occupancy, dtype=np.float64)
        return cls(
            n_states=len(occupancy),
            minimum_occupancy=float(np.min(occupancy)),
            state_separation=float(fit.state_separation),
            label_order_gap=float(fit.label_order_gap),
            label_ambiguous=bool(fit.label_ambiguous),
            low_occupancy=bool(fit.low_occupancy),
        )


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
