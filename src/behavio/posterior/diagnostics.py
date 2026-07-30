"""Common convergence diagnostics for backend-neutral posterior results.

Severity policy
---------------
The maximum-likelihood side has a three-valued verdict
(:class:`~behavio.contracts.audit.FitAuditStatus`) driven by a two-valued severity
(:class:`~behavio.contracts.audit.AuditSeverity`), so a non-convergent optimizer produces
``FAIL`` rather than a warning. This module now mirrors it: ``PosteriorAuditStatus`` gains
``FAIL`` and every :class:`PosteriorAuditIssue` carries an ``AuditSeverity``.

The default classification is a property of :class:`PosteriorAuditPolicy`, because it is a
threshold decision like ``max_rhat``, not a fact about the sampler:

- **divergences** are ``ERROR``. A divergent transition means the sampler failed to
  explore the target geometry; the draws are biased in an unknown direction, so the
  posterior is not usable evidence.
- **R-hat exceedance** is ``ERROR``. Chains that have not mixed are not draws from one
  distribution at all, so every downstream summary is meaningless.
- **non-finite diagnostics** are ``ERROR``: the audit itself could not be computed.
- **low bulk or tail ESS** is a ``WARNING``. Low ESS means the Monte Carlo error on a
  summary is larger than the policy wants, which is a precision problem, not a validity
  problem; it is fixed by drawing more samples and does not by itself invalidate the fit.
- **maximum tree-depth saturation** is a ``WARNING``: it costs efficiency, and it is
  routinely benign when R-hat and ESS are otherwise healthy.

Callers that disagree can pass their own severities on the policy.
``behavio.posterior.loo`` and ``behavio.posterior.predictive`` reuse
``PosteriorAuditStatus`` for their own status properties and deliberately keep their
existing ``PASS``/``WARNING`` semantics; adding ``FAIL`` does not change them.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from behavio.contracts.audit import AuditSeverity
from behavio.posterior.result import PosteriorError, PosteriorResult


class PosteriorAuditStatus(StrEnum):
    """Overall status of a posterior convergence audit.

    ``FAIL`` mirrors :class:`~behavio.contracts.audit.FitAuditStatus`: at least one issue
    is severe enough that the posterior should not be used as evidence.
    """

    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"


@dataclass(frozen=True, slots=True)
class PosteriorAuditPolicy:
    """Explicit thresholds for standard MCMC diagnostics."""

    max_rhat: float = 1.01
    min_ess_bulk: float = 400.0
    min_ess_tail: float = 400.0
    max_divergences: int = 0
    max_treedepth_hits: int = 0
    divergence_severity: AuditSeverity = AuditSeverity.ERROR
    rhat_severity: AuditSeverity = AuditSeverity.ERROR
    nonfinite_severity: AuditSeverity = AuditSeverity.ERROR
    ess_severity: AuditSeverity = AuditSeverity.WARNING
    treedepth_severity: AuditSeverity = AuditSeverity.WARNING

    def __post_init__(self) -> None:
        if not np.isfinite(self.max_rhat) or self.max_rhat <= 1.0:
            raise ValueError("max_rhat must be finite and greater than one")
        for value, label in (
            (self.min_ess_bulk, "min_ess_bulk"),
            (self.min_ess_tail, "min_ess_tail"),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{label} must be finite and positive")
        for value, label in (
            (self.max_divergences, "max_divergences"),
            (self.max_treedepth_hits, "max_treedepth_hits"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{label} must be a non-negative integer")
        for label in (
            "divergence_severity",
            "rhat_severity",
            "nonfinite_severity",
            "ess_severity",
            "treedepth_severity",
        ):
            object.__setattr__(self, label, AuditSeverity(getattr(self, label)))

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "max_rhat": self.max_rhat,
            "min_ess_bulk": self.min_ess_bulk,
            "min_ess_tail": self.min_ess_tail,
            "max_divergences": self.max_divergences,
            "max_treedepth_hits": self.max_treedepth_hits,
            "divergence_severity": self.divergence_severity.value,
            "rhat_severity": self.rhat_severity.value,
            "nonfinite_severity": self.nonfinite_severity.value,
            "ess_severity": self.ess_severity.value,
            "treedepth_severity": self.treedepth_severity.value,
        }


@dataclass(frozen=True, slots=True)
class PosteriorDiagnostic:
    """R-hat and effective sample sizes on one labelled parameter variable."""

    name: str
    dims: tuple[str, ...]
    coords: Mapping[str, Sequence[Any] | NDArray[Any]]
    rhat: NDArray[np.float64] | Sequence[float] | float
    ess_bulk: NDArray[np.float64] | Sequence[float] | float
    ess_tail: NDArray[np.float64] | Sequence[float] | float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("diagnostic name must be a non-empty string")
        dims = tuple(self.dims)
        if len(set(dims)) != len(dims):
            raise ValueError("diagnostic dimensions must be unique")
        if set(self.coords) != set(dims):
            raise ValueError("diagnostic coordinates must name exactly its dimensions")
        coords = {dim: _protected(np.asarray(self.coords[dim])) for dim in dims}
        shape = tuple(len(coords[dim]) for dim in dims)
        arrays = {
            "rhat": _protected(np.asarray(self.rhat, dtype=np.float64)),
            "ess_bulk": _protected(np.asarray(self.ess_bulk, dtype=np.float64)),
            "ess_tail": _protected(np.asarray(self.ess_tail, dtype=np.float64)),
        }
        if any(array.shape != shape for array in arrays.values()):
            raise ValueError("diagnostic arrays must align with their labelled dimensions")
        object.__setattr__(self, "dims", dims)
        object.__setattr__(self, "coords", MappingProxyType(coords))
        for name, array in arrays.items():
            object.__setattr__(self, name, array)


@dataclass(frozen=True, slots=True)
class PosteriorAuditIssue:
    """One stable posterior diagnostic warning and its labelled targets."""

    code: str
    message: str
    targets: tuple[str, ...] = ()
    severity: AuditSeverity = AuditSeverity.WARNING

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code:
            raise ValueError("issue code must be a non-empty string")
        if not isinstance(self.message, str) or not self.message:
            raise ValueError("issue message must be a non-empty string")
        targets = tuple(self.targets)
        if any(not isinstance(target, str) or not target for target in targets):
            raise ValueError("issue targets must be non-empty strings")
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "severity", AuditSeverity(self.severity))


@dataclass(frozen=True, slots=True)
class PosteriorAudit:
    """Complete retained evidence from one policy-driven posterior audit."""

    model_name: str
    model_signature: str
    inference_library: str
    inference_library_version: str
    n_chains: int
    n_draws: int
    divergences: int | None
    max_treedepth_hits: int | None
    policy: PosteriorAuditPolicy
    diagnostics: tuple[PosteriorDiagnostic, ...]
    issues: tuple[PosteriorAuditIssue, ...]

    @property
    def status(self) -> PosteriorAuditStatus:
        """Return fail, warning, or pass without collapsing the underlying issues."""

        if any(issue.severity is AuditSeverity.ERROR for issue in self.issues):
            return PosteriorAuditStatus.FAIL
        if self.issues:
            return PosteriorAuditStatus.WARNING
        return PosteriorAuditStatus.PASS

    @property
    def n_samples(self) -> int:
        return self.n_chains * self.n_draws

    @property
    def issue_codes(self) -> tuple[str, ...]:
        return tuple(issue.code for issue in self.issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_signature": self.model_signature,
            "inference_library": self.inference_library,
            "inference_library_version": self.inference_library_version,
            "n_chains": self.n_chains,
            "n_draws": self.n_draws,
            "n_samples": self.n_samples,
            "divergences": self.divergences,
            "max_treedepth_hits": self.max_treedepth_hits,
            "status": self.status.value,
            "policy": self.policy.to_dict(),
            "issues": [
                {
                    "code": issue.code,
                    "message": issue.message,
                    "targets": list(issue.targets),
                    "severity": issue.severity.value,
                }
                for issue in self.issues
            ],
        }


def audit_posterior(
    result: PosteriorResult,
    *,
    policy: PosteriorAuditPolicy | None = None,
) -> PosteriorAudit:
    """Compute rank-normalized R-hat, bulk/tail ESS, and retained HMC warnings."""

    if not isinstance(result, PosteriorResult):
        raise TypeError("result must be a PosteriorResult")
    active_policy = PosteriorAuditPolicy() if policy is None else policy
    if not isinstance(active_policy, PosteriorAuditPolicy):
        raise TypeError("policy must be a PosteriorAuditPolicy")
    arviz_data = result.to_arviz()
    rhat_data, bulk_data, tail_data = _sampling_diagnostics(arviz_data, result.parameter_names)
    diagnostics = tuple(
        _diagnostic_for(name, rhat_data, bulk_data, tail_data) for name in result.parameter_names
    )
    divergences = _sample_stat_count(result, "diverging")
    treedepth = _sample_stat_count(result, "reached_max_treedepth")
    issues: list[PosteriorAuditIssue] = []
    if divergences is not None and divergences > active_policy.max_divergences:
        issues.append(
            PosteriorAuditIssue(
                "posterior.divergences",
                f"observed {divergences} divergent transitions",
                severity=active_policy.divergence_severity,
            )
        )
    if treedepth is not None and treedepth > active_policy.max_treedepth_hits:
        issues.append(
            PosteriorAuditIssue(
                "posterior.max-treedepth",
                f"observed {treedepth} transitions at maximum tree depth",
                severity=active_policy.treedepth_severity,
            )
        )
    nonfinite = _targets(diagnostics, lambda item: ~np.isfinite(item.rhat))
    nonfinite += _targets(diagnostics, lambda item: ~np.isfinite(item.ess_bulk))
    nonfinite += _targets(diagnostics, lambda item: ~np.isfinite(item.ess_tail))
    if nonfinite:
        issues.append(
            PosteriorAuditIssue(
                "posterior.nonfinite-diagnostic",
                "one or more sampling diagnostics are non-finite",
                tuple(dict.fromkeys(nonfinite)),
                severity=active_policy.nonfinite_severity,
            )
        )
    rhat = _targets(diagnostics, lambda item: item.rhat > active_policy.max_rhat)
    if rhat:
        issues.append(
            PosteriorAuditIssue(
                "posterior.rhat",
                f"rank-normalized R-hat exceeds {active_policy.max_rhat}",
                rhat,
                severity=active_policy.rhat_severity,
            )
        )
    bulk = _targets(diagnostics, lambda item: item.ess_bulk < active_policy.min_ess_bulk)
    if bulk:
        issues.append(
            PosteriorAuditIssue(
                "posterior.ess-bulk",
                f"bulk ESS is below {active_policy.min_ess_bulk}",
                bulk,
                severity=active_policy.ess_severity,
            )
        )
    tail = _targets(diagnostics, lambda item: item.ess_tail < active_policy.min_ess_tail)
    if tail:
        issues.append(
            PosteriorAuditIssue(
                "posterior.ess-tail",
                f"tail ESS is below {active_policy.min_ess_tail}",
                tail,
                severity=active_policy.ess_severity,
            )
        )
    return PosteriorAudit(
        model_name=result.model_name,
        model_signature=result.model_signature,
        inference_library=result.inference_library,
        inference_library_version=result.inference_library_version,
        n_chains=result.n_chains,
        n_draws=result.n_draws,
        divergences=divergences,
        max_treedepth_hits=treedepth,
        policy=active_policy,
        diagnostics=diagnostics,
        issues=tuple(issues),
    )


def _sampling_diagnostics(data: Any, parameter_names: tuple[str, ...]) -> tuple[Any, Any, Any]:
    try:
        stats = importlib.import_module("arviz_stats")
    except ImportError:
        stats = importlib.import_module("arviz")
    kwargs = {"var_names": list(parameter_names)}
    return (
        _dataset(stats.rhat(data, **kwargs)),
        _dataset(stats.ess(data, method="bulk", **kwargs)),
        _dataset(stats.ess(data, method="tail", **kwargs)),
    )


def _dataset(value: Any) -> Any:
    if hasattr(value, "data_vars"):
        return value
    if hasattr(value, "to_dataset"):
        return value.to_dataset()
    raise PosteriorError("ArviZ diagnostic output is not an xarray Dataset or DataTree")


def _diagnostic_for(name: str, rhat: Any, bulk: Any, tail: Any) -> PosteriorDiagnostic:
    if name not in rhat or name not in bulk or name not in tail:
        raise PosteriorError(f"sampling diagnostics omitted declared parameter {name!r}")
    source = rhat[name]
    dims = tuple(str(dim) for dim in source.dims)
    coords = {dim: np.asarray(source.coords[dim].values) for dim in dims}
    return PosteriorDiagnostic(
        name=name,
        dims=dims,
        coords=coords,
        rhat=np.asarray(source.values),
        ess_bulk=np.asarray(bulk[name].values),
        ess_tail=np.asarray(tail[name].values),
    )


def _sample_stat_count(result: PosteriorResult, name: str) -> int | None:
    if "sample_stats" not in result.group_names:
        return None
    group = result["sample_stats"]
    if name not in group.variable_names:
        return None
    values = np.asarray(group[name].values)
    if values.dtype.kind not in "biu":
        raise PosteriorError(f"sample_stats.{name} must contain boolean or integer flags")
    return int(np.count_nonzero(values))


def _targets(
    diagnostics: tuple[PosteriorDiagnostic, ...],
    predicate: Callable[[PosteriorDiagnostic], NDArray[np.bool_]],
) -> tuple[str, ...]:
    labels: list[str] = []
    for diagnostic in diagnostics:
        mask = np.asarray(predicate(diagnostic), dtype=bool)
        for raw_index in np.argwhere(mask):
            index = tuple(int(value) for value in raw_index)
            labels.append(_target_label(diagnostic, index))
    return tuple(labels)


def _target_label(diagnostic: PosteriorDiagnostic, index: tuple[int, ...]) -> str:
    if not diagnostic.dims:
        return diagnostic.name
    coordinates = ",".join(
        f"{dim}={_python_scalar(diagnostic.coords[dim][position])!r}"
        for dim, position in zip(diagnostic.dims, index, strict=True)
    )
    return f"{diagnostic.name}[{coordinates}]"


def _python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _protected(values: NDArray[Any]) -> NDArray[Any]:
    protected = np.array(values, copy=True)
    protected.setflags(write=False)
    return protected
