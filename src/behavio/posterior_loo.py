"""Pointwise PSIS-LOO evaluation for backend-neutral posterior results."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from behavio.posterior import PosteriorError, PosteriorResult
from behavio.posterior_diagnostics import PosteriorAuditStatus


@dataclass(frozen=True, slots=True)
class PSISLOOIssue:
    """One stable warning from pointwise PSIS-LOO evaluation."""

    code: str
    message: str
    targets: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code:
            raise ValueError("PSIS-LOO issue code must be a non-empty string")
        if not isinstance(self.message, str) or not self.message:
            raise ValueError("PSIS-LOO issue message must be a non-empty string")
        targets = tuple(self.targets)
        if any(not isinstance(target, str) or not target for target in targets):
            raise ValueError("PSIS-LOO issue targets must be non-empty strings")
        object.__setattr__(self, "targets", targets)


@dataclass(frozen=True, slots=True)
class PSISLOOResult:
    """Immutable pointwise PSIS-LOO evidence in labelled observation coordinates."""

    model_name: str
    model_signature: str
    inference_library: str
    inference_library_version: str
    log_likelihood_name: str
    dims: tuple[str, ...]
    coords: Mapping[str, Sequence[Any] | NDArray[Any]]
    elpd_loo: float
    se: float
    p_loo: float
    n_samples: int
    n_data_points: int
    good_k: float
    pointwise_elpd: NDArray[np.float64] | Sequence[float] | float
    pareto_k: NDArray[np.float64] | Sequence[float] | float
    issues: tuple[PSISLOOIssue, ...] = ()

    def __post_init__(self) -> None:
        for value, label in (
            (self.model_name, "model_name"),
            (self.model_signature, "model_signature"),
            (self.inference_library, "inference_library"),
            (self.inference_library_version, "inference_library_version"),
            (self.log_likelihood_name, "log_likelihood_name"),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{label} must be a non-empty string")
        dims = tuple(self.dims)
        if not dims or any(not isinstance(dim, str) or not dim for dim in dims):
            raise ValueError("PSIS-LOO dimensions must be non-empty strings")
        if len(set(dims)) != len(dims):
            raise ValueError("PSIS-LOO dimensions must be unique")
        if set(self.coords) != set(dims):
            raise ValueError("PSIS-LOO coordinates must name exactly its dimensions")
        coords = {dim: _protected(np.asarray(self.coords[dim])) for dim in dims}
        shape = tuple(len(coords[dim]) for dim in dims)
        pointwise = _protected(np.asarray(self.pointwise_elpd, dtype=np.float64))
        pareto_k = _protected(np.asarray(self.pareto_k, dtype=np.float64))
        if pointwise.shape != shape or pareto_k.shape != shape:
            raise ValueError("PSIS-LOO arrays must align with labelled observation dimensions")
        for value, label in (
            (self.elpd_loo, "elpd_loo"),
            (self.se, "se"),
            (self.p_loo, "p_loo"),
            (self.good_k, "good_k"),
        ):
            if not isinstance(value, (int, float, np.integer, np.floating)):
                raise ValueError(f"{label} must be numeric")
        if not np.isfinite(self.good_k) or self.good_k <= 0:
            raise ValueError("good_k must be finite and positive")
        for value, label in (
            (self.n_samples, "n_samples"),
            (self.n_data_points, "n_data_points"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{label} must be a positive integer")
        if self.n_data_points != pointwise.size:
            raise ValueError("n_data_points must equal the pointwise array size")
        issues = tuple(self.issues)
        if any(not isinstance(issue, PSISLOOIssue) for issue in issues):
            raise ValueError("issues must contain PSISLOOIssue records")
        codes = tuple(issue.code for issue in issues)
        if len(set(codes)) != len(codes):
            raise ValueError("PSIS-LOO issue codes must be unique within one result")
        object.__setattr__(self, "dims", dims)
        object.__setattr__(self, "coords", MappingProxyType(coords))
        object.__setattr__(self, "elpd_loo", float(self.elpd_loo))
        object.__setattr__(self, "se", float(self.se))
        object.__setattr__(self, "p_loo", float(self.p_loo))
        object.__setattr__(self, "good_k", float(self.good_k))
        object.__setattr__(self, "pointwise_elpd", pointwise)
        object.__setattr__(self, "pareto_k", pareto_k)
        object.__setattr__(self, "issues", issues)

    @property
    def status(self) -> PosteriorAuditStatus:
        return PosteriorAuditStatus.PASS if not self.issues else PosteriorAuditStatus.WARNING

    @property
    def issue_codes(self) -> tuple[str, ...]:
        return tuple(issue.code for issue in self.issues)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation including pointwise evidence."""

        return {
            "model_name": self.model_name,
            "model_signature": self.model_signature,
            "inference_library": self.inference_library,
            "inference_library_version": self.inference_library_version,
            "log_likelihood_name": self.log_likelihood_name,
            "dims": list(self.dims),
            "coords": {
                dim: [_json_scalar(value) for value in self.coords[dim]] for dim in self.dims
            },
            "elpd_loo": _json_float(self.elpd_loo),
            "se": _json_float(self.se),
            "p_loo": _json_float(self.p_loo),
            "n_samples": self.n_samples,
            "n_data_points": self.n_data_points,
            "good_k": self.good_k,
            "pointwise_elpd": [_json_float(value) for value in self.pointwise_elpd.flat],
            "pareto_k": [_json_float(value) for value in self.pareto_k.flat],
            "status": self.status.value,
            "issues": [
                {"code": issue.code, "message": issue.message, "targets": list(issue.targets)}
                for issue in self.issues
            ],
        }


def psis_loo(
    result: PosteriorResult,
    *,
    log_likelihood_name: str | None = None,
) -> PSISLOOResult:
    """Estimate pointwise log-scale ELPD with PSIS-LOO and retain Pareto-k evidence."""

    if not isinstance(result, PosteriorResult):
        raise TypeError("result must be a PosteriorResult")
    variable = _log_likelihood_variable(result, log_likelihood_name)
    if not variable.intrinsic_dims:
        raise PosteriorError("PSIS-LOO requires pointwise rather than aggregated log likelihood")
    if not np.all(np.isfinite(variable.values)):
        raise PosteriorError("pointwise log likelihood must contain only finite values")

    output = _compute_loo(result, variable.name)
    pointwise = _attribute(output, "elpd_i", "loo_i")
    pareto = _attribute(output, "pareto_k")
    dims = tuple(str(dim) for dim in pointwise.dims)
    if dims != variable.intrinsic_dims:
        raise PosteriorError(
            "PSIS-LOO pointwise dimensions do not match the declared log likelihood"
        )
    coords = {dim: np.asarray(pointwise.coords[dim].values) for dim in dims}
    for dim in dims:
        if not np.array_equal(coords[dim], variable.coords[dim]):
            raise PosteriorError(
                "PSIS-LOO pointwise coordinates do not match the declared log likelihood"
            )
    values = np.asarray(pointwise.values, dtype=np.float64)
    pareto_values = np.asarray(pareto.values, dtype=np.float64)
    if pareto_values.shape != values.shape:
        raise PosteriorError("PSIS-LOO Pareto-k values do not align with pointwise ELPD")

    good_k = float(_attribute(output, "good_k"))
    issues: list[PSISLOOIssue] = []
    nonfinite = (~np.isfinite(values)) | (~np.isfinite(pareto_values))
    if np.any(nonfinite):
        issues.append(
            PSISLOOIssue(
                code="psis.nonfinite",
                message="one or more pointwise PSIS-LOO estimates are non-finite",
                targets=_targets(variable.name, dims, coords, nonfinite),
            )
        )
    high_k = pareto_values > good_k
    if np.any(high_k):
        issues.append(
            PSISLOOIssue(
                code="psis.high-pareto-k",
                message=f"Pareto k exceeds the sample-size threshold {good_k:.3g}",
                targets=_targets(variable.name, dims, coords, high_k),
            )
        )
    backend_warning = bool(_attribute(output, "warning"))
    if backend_warning and not issues:
        issues.append(
            PSISLOOIssue(
                code="psis.backend-warning",
                message="the PSIS-LOO backend reported an unlocalized reliability warning",
            )
        )
    return PSISLOOResult(
        model_name=result.model_name,
        model_signature=result.model_signature,
        inference_library=result.inference_library,
        inference_library_version=result.inference_library_version,
        log_likelihood_name=variable.name,
        dims=dims,
        coords=coords,
        elpd_loo=float(_attribute(output, "elpd", "elpd_loo")),
        se=float(_attribute(output, "se")),
        p_loo=float(_attribute(output, "p", "p_loo")),
        n_samples=int(_attribute(output, "n_samples")),
        n_data_points=int(_attribute(output, "n_data_points")),
        good_k=good_k,
        pointwise_elpd=values,
        pareto_k=pareto_values,
        issues=tuple(issues),
    )


def _log_likelihood_variable(result: PosteriorResult, name: str | None) -> Any:
    if "log_likelihood" not in result.group_names:
        raise PosteriorError("PSIS-LOO requires a log_likelihood group")
    group = result["log_likelihood"]
    if name is None:
        if len(group.variable_names) != 1:
            raise PosteriorError(
                "log_likelihood_name is required when multiple likelihood variables exist"
            )
        return group.variables[0]
    if not isinstance(name, str) or not name:
        raise ValueError("log_likelihood_name must be null or a non-empty string")
    if name not in group.variable_names:
        raise PosteriorError(f"log_likelihood has no variable {name!r}")
    return group[name]


def _compute_loo(result: PosteriorResult, variable_name: str) -> Any:
    try:
        stats = importlib.import_module("arviz_stats")
    except ImportError:
        stats = importlib.import_module("arviz")
    return stats.loo(result.to_arviz(), pointwise=True, var_name=variable_name)


def _attribute(value: Any, *names: str) -> Any:
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
        try:
            return value[name]
        except (KeyError, TypeError):
            continue
    raise PosteriorError(f"PSIS-LOO output is missing required field {names[0]!r}")


def _targets(
    variable_name: str,
    dims: tuple[str, ...],
    coords: Mapping[str, NDArray[Any]],
    mask: NDArray[np.bool_],
) -> tuple[str, ...]:
    labels = []
    for raw_index in np.argwhere(mask):
        index = tuple(int(value) for value in raw_index)
        coordinates = ",".join(
            f"{dim}={_json_scalar(coords[dim][position])!r}"
            for dim, position in zip(dims, index, strict=True)
        )
        labels.append(f"{variable_name}[{coordinates}]")
    return tuple(labels)


def _json_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _json_float(value: Any) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


def _protected(values: NDArray[Any]) -> NDArray[Any]:
    protected = np.array(values, copy=True)
    protected.setflags(write=False)
    return protected
