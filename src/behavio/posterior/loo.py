"""Pointwise and blocked PSIS-LOO evaluation for backend-neutral posterior results.

Estimand
--------
By default :func:`psis_loo` scores the estimand ArviZ computes: **leave-one-observation-out**
ELPD over whatever intrinsic dimension the declared log likelihood carries. For the
hierarchical backends in this package that dimension is ``trial``, so the default estimand is
leave-one-*trial*-out.

Leave-one-trial-out is the wrong estimand for most multi-subject, multi-session behavioural
questions. The held-out trial's own subject remains in the fit, its subject-level parameters
were estimated partly from it, and neighbouring trials carry history features derived from the
held-out response. Trial-level ELPD therefore systematically overstates predictive performance
relative to the question "how well does this model predict a subject, or a session, it has not
seen?".

``block`` fixes the estimand rather than the estimate. Naming a grouping variable sums the
pointwise log likelihood **within each group before** importance sampling, so PSIS reweights
draws for the removal of a whole subject or session. The result is leave-one-subject-out or
leave-one-session-out ELPD, and the pointwise unit of the returned arrays becomes the block.

Status
------
:attr:`PSISLOOResult.status` combines two independent sources of doubt:

- **importance-sampling reliability**, owned by this module (Pareto-:math:`k`, non-finite
  pointwise output, backend warnings); and
- **posterior convergence**, delegated to :func:`behavio.posterior.diagnostics.audit_posterior`.

They combine by worst severity, following
:class:`~behavio.posterior.diagnostics.PosteriorAuditStatus`: any ``ERROR``-severity issue
gives ``FAIL``, any remaining issue gives ``WARNING``, otherwise ``PASS``. ELPD from a
non-converged posterior is not evidence, but the package idiom is to retain the number and mark
it unusable rather than to raise, mirroring how ``FitAuditStatus.FAIL`` makes a candidate
ineligible in :mod:`behavio.protocol.runner`. The full :class:`~behavio.posterior.diagnostics.
PosteriorAudit` is retained on :attr:`PSISLOOResult.convergence`.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from behavio.contracts.audit import AuditSeverity
from behavio.posterior.diagnostics import (
    PosteriorAudit,
    PosteriorAuditPolicy,
    PosteriorAuditStatus,
    audit_posterior,
)
from behavio.posterior.result import (
    SAMPLE_DIMS,
    PosteriorError,
    PosteriorGroup,
    PosteriorResult,
    PosteriorVariable,
)

MIN_RELIABLE_BLOCKS: Final = 10
"""Blocks below which the normal approximation behind ``se`` is not trustworthy.

``se`` is ``sqrt(n * var(pointwise_elpd))``, a central-limit approximation over the pointwise
unit. Blocking collapses thousands of trials into a handful of subjects, so the same formula
that is unremarkable over 20,000 trials is a strong assumption over eight subjects. Crossing
this threshold emits ``psis.few-blocks`` rather than silently reporting a confident interval.
"""

_BLOCK_LOOKUP_GROUPS: Final = ("constant_data", "observed_data")


@dataclass(frozen=True, slots=True)
class PSISLOOIssue:
    """One stable warning from pointwise PSIS-LOO evaluation.

    ``severity`` mirrors :class:`~behavio.posterior.diagnostics.PosteriorAuditIssue`: an
    ``ERROR`` marks the ELPD estimate unusable as evidence, a ``WARNING`` marks it imprecise or
    locally unreliable.
    """

    code: str
    message: str
    targets: tuple[str, ...] = ()
    severity: AuditSeverity = AuditSeverity.WARNING

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code:
            raise ValueError("PSIS-LOO issue code must be a non-empty string")
        if not isinstance(self.message, str) or not self.message:
            raise ValueError("PSIS-LOO issue message must be a non-empty string")
        targets = tuple(self.targets)
        if any(not isinstance(target, str) or not target for target in targets):
            raise ValueError("PSIS-LOO issue targets must be non-empty strings")
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "severity", AuditSeverity(self.severity))


@dataclass(frozen=True, slots=True)
class PSISLOOResult:
    """Immutable PSIS-LOO evidence in labelled observation or block coordinates.

    ``block`` records the estimand. ``None`` means leave-one-observation-out over the declared
    log-likelihood dimension; a name means leave-one-*block*-out, and the pointwise arrays are
    then one value per block rather than one per observation. Two results carrying different
    ``block`` values are estimates of different quantities and must never be differenced; that
    refusal is enforced in :func:`behavio.posterior.comparison.compare_posterior_models`.
    """

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
    block: str | None = None
    convergence: PosteriorAudit | None = None

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
        if self.block is not None and (not isinstance(self.block, str) or not self.block):
            raise ValueError("block must be null or a non-empty string")
        if self.convergence is not None and not isinstance(self.convergence, PosteriorAudit):
            raise ValueError("convergence must be null or a PosteriorAudit")
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
        """Worst severity across importance-sampling and convergence evidence.

        ``FAIL`` when any retained issue is an ``ERROR`` (which includes a failed convergence
        audit), ``WARNING`` when issues exist but none is fatal, ``PASS`` otherwise.
        """

        if any(issue.severity is AuditSeverity.ERROR for issue in self.issues):
            return PosteriorAuditStatus.FAIL
        if self.issues:
            return PosteriorAuditStatus.WARNING
        return PosteriorAuditStatus.PASS

    @property
    def issue_codes(self) -> tuple[str, ...]:
        return tuple(issue.code for issue in self.issues)

    @property
    def estimand(self) -> str:
        """Human-readable name of the cross-validation estimand this result computed."""

        if self.block is None:
            return "leave-one-observation-out"
        return f"leave-one-{self.block}-out"

    @property
    def convergence_status(self) -> PosteriorAuditStatus | None:
        """Status of the retained convergence audit, or ``None`` when it was not run."""

        return None if self.convergence is None else self.convergence.status

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation including pointwise evidence."""

        return {
            "model_name": self.model_name,
            "model_signature": self.model_signature,
            "inference_library": self.inference_library,
            "inference_library_version": self.inference_library_version,
            "log_likelihood_name": self.log_likelihood_name,
            "block": self.block,
            "estimand": self.estimand,
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
            "convergence": None if self.convergence is None else self.convergence.to_dict(),
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


def psis_loo(
    result: PosteriorResult,
    *,
    log_likelihood_name: str | None = None,
    block: str | None = None,
    block_values: Sequence[Any] | NDArray[Any] | None = None,
    policy: PosteriorAuditPolicy | None = None,
) -> PSISLOOResult:
    """Estimate log-scale ELPD with PSIS-LOO, optionally blocked, gated on convergence.

    Args:
        result: Labelled posterior carrying a pointwise ``log_likelihood`` group.
        log_likelihood_name: Which likelihood variable to score. Required when the group holds
            more than one.
        block: Name of the grouping that defines the leave-one-out unit. ``None`` keeps the
            default leave-one-observation-out estimand and reproduces the unblocked
            calculation exactly. A name selects leave-one-*block*-out: the pointwise log
            likelihood is summed within each group before importance sampling. Unless
            ``block_values`` is given, the name must match a one-dimensional variable in
            ``constant_data`` or ``observed_data`` laid out over the scored dimension --- for
            example ``"trial_subject"`` or ``"trial_session"`` from the PyMC backend. Nothing
            is inferred from coordinate names or heuristics.
        block_values: Explicit per-observation group labels, for posteriors that do not retain
            the grouping. Its length must equal the scored dimension. Requires ``block``, which
            then names the estimand rather than a stored variable.
        policy: Thresholds for the convergence audit. ``None`` uses the default
            :class:`~behavio.posterior.diagnostics.PosteriorAuditPolicy`.

    Returns:
        A :class:`PSISLOOResult` whose pointwise arrays are one value per observation when
        ``block`` is ``None`` and one value per block otherwise.

    Raises:
        PosteriorError: If the log likelihood is aggregated or non-finite, if blocking is
            requested on a multi-dimensional pointwise likelihood, or if the grouping labels
            cannot be resolved or do not span the scored dimension.
    """

    if not isinstance(result, PosteriorResult):
        raise TypeError("result must be a PosteriorResult")
    variable = _log_likelihood_variable(result, log_likelihood_name)
    if not variable.intrinsic_dims:
        raise PosteriorError("PSIS-LOO requires pointwise rather than aggregated log likelihood")
    if not np.all(np.isfinite(variable.values)):
        raise PosteriorError("pointwise log likelihood must contain only finite values")

    block_name = _validated_block(block, block_values)
    if block_name is None:
        scored, scored_dims = result, variable.intrinsic_dims
    else:
        labels = _block_labels(result, variable, block_name, block_values)
        scored = _blocked_result(result, variable, block_name, labels)
        scored_dims = (block_name,)

    convergence = audit_posterior(result, policy=policy)
    output = _compute_loo(scored, variable.name)
    pointwise = _attribute(output, "elpd_i", "loo_i")
    pareto = _attribute(output, "pareto_k")
    dims = tuple(str(dim) for dim in pointwise.dims)
    if dims != scored_dims:
        raise PosteriorError(
            "PSIS-LOO pointwise dimensions do not match the declared log likelihood"
        )
    reference = scored["log_likelihood"][variable.name]
    coords = {dim: np.asarray(pointwise.coords[dim].values) for dim in dims}
    for dim in dims:
        if not np.array_equal(coords[dim], reference.coords[dim]):
            raise PosteriorError(
                "PSIS-LOO pointwise coordinates do not match the declared log likelihood"
            )
    values = np.asarray(pointwise.values, dtype=np.float64)
    pareto_values = np.asarray(pareto.values, dtype=np.float64)
    if pareto_values.shape != values.shape:
        raise PosteriorError("PSIS-LOO Pareto-k values do not align with pointwise ELPD")

    good_k = float(_attribute(output, "good_k"))
    issues = _importance_sampling_issues(
        variable.name,
        dims,
        coords,
        values,
        pareto_values,
        good_k,
        block_name,
        bool(_attribute(output, "warning")),
    )
    convergence_issue = _convergence_issue(convergence)
    if convergence_issue is not None:
        issues.append(convergence_issue)
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
        block=block_name,
        convergence=convergence,
    )


def _importance_sampling_issues(
    variable_name: str,
    dims: tuple[str, ...],
    coords: Mapping[str, NDArray[Any]],
    values: NDArray[np.float64],
    pareto_values: NDArray[np.float64],
    good_k: float,
    block_name: str | None,
    backend_warning: bool,
) -> list[PSISLOOIssue]:
    """Localize importance-sampling failures, at whatever scale the estimand uses.

    ``good_k`` is ``min(1 - 1/log10(S), 0.7)`` in the number of posterior draws ``S``, so it is
    unchanged by blocking. What blocking changes is the meaning of crossing it: leaving out a
    whole subject is a far larger perturbation of the posterior than leaving out one trial, so
    high :math:`k` is both more likely and more consequential, and there are far fewer values
    in which to hide it. The message therefore reports the affected share when blocked.
    """

    issues: list[PSISLOOIssue] = []
    nonfinite = (~np.isfinite(values)) | (~np.isfinite(pareto_values))
    if np.any(nonfinite):
        issues.append(
            PSISLOOIssue(
                code="psis.nonfinite",
                message="one or more pointwise PSIS-LOO estimates are non-finite",
                targets=_targets(variable_name, dims, coords, nonfinite),
                severity=AuditSeverity.ERROR,
            )
        )
    high_k = pareto_values > good_k
    if np.any(high_k):
        share = ""
        if block_name is not None:
            share = f" for {int(np.count_nonzero(high_k))} of {values.size} {block_name} blocks"
        issues.append(
            PSISLOOIssue(
                code="psis.high-pareto-k",
                message=f"Pareto k exceeds the sample-size threshold {good_k:.3g}{share}",
                targets=_targets(variable_name, dims, coords, high_k),
            )
        )
    if backend_warning and not issues:
        issues.append(
            PSISLOOIssue(
                code="psis.backend-warning",
                message="the PSIS-LOO backend reported an unlocalized reliability warning",
            )
        )
    if block_name is not None and values.size < MIN_RELIABLE_BLOCKS:
        issues.append(
            PSISLOOIssue(
                code="psis.few-blocks",
                message=(
                    f"blocked ELPD rests on {values.size} {block_name} blocks, below the "
                    f"{MIN_RELIABLE_BLOCKS} needed for the normal approximation behind se"
                ),
            )
        )
    return issues


def _convergence_issue(audit: PosteriorAudit) -> PSISLOOIssue | None:
    """Fold the convergence verdict into this module's issue vocabulary."""

    status = audit.status
    if status is PosteriorAuditStatus.PASS:
        return None
    codes = ", ".join(audit.issue_codes)
    if status is PosteriorAuditStatus.FAIL:
        return PSISLOOIssue(
            code="psis.posterior-not-converged",
            message=(
                "the posterior failed its convergence audit, so this ELPD estimate is not "
                f"usable evidence ({codes})"
            ),
            targets=audit.issue_codes,
            severity=AuditSeverity.ERROR,
        )
    return PSISLOOIssue(
        code="psis.posterior-warning",
        message=f"the posterior convergence audit reported warnings ({codes})",
        targets=audit.issue_codes,
    )


def _validated_block(block: str | None, block_values: Any) -> str | None:
    if block is None:
        if block_values is not None:
            raise ValueError("block_values requires an explicit block name for the estimand")
        return None
    if not isinstance(block, str) or not block:
        raise ValueError("block must be null or a non-empty string")
    return block


def _block_labels(
    result: PosteriorResult,
    variable: PosteriorVariable,
    block_name: str,
    block_values: Sequence[Any] | NDArray[Any] | None,
) -> NDArray[Any]:
    """Resolve one group label per scored observation, explicitly and without guessing."""

    if len(variable.intrinsic_dims) != 1:
        raise PosteriorError(
            "blocked PSIS-LOO requires a log likelihood with exactly one observation dimension"
        )
    observation_dim = variable.intrinsic_dims[0]
    size = variable.values.shape[variable.dims.index(observation_dim)]
    if block_values is not None:
        labels = np.asarray(block_values)
    else:
        labels = _stored_block_labels(result, block_name, observation_dim)
    if labels.ndim != 1:
        raise PosteriorError(f"block {block_name!r} must supply one label per scored observation")
    if len(labels) != size:
        raise PosteriorError(
            f"block {block_name!r} supplies {len(labels)} labels for {size} scored observations"
        )
    return labels


def _stored_block_labels(
    result: PosteriorResult,
    block_name: str,
    observation_dim: str,
) -> NDArray[Any]:
    available: list[str] = []
    for group_name in _BLOCK_LOOKUP_GROUPS:
        if group_name not in result.group_names:
            continue
        group = result[group_name]
        for candidate in group.variables:
            if candidate.dims == (observation_dim,):
                available.append(f"{group_name}.{candidate.name}")
        if block_name in group.variable_names:
            stored = group[block_name]
            if stored.dims != (observation_dim,):
                raise PosteriorError(
                    f"block {block_name!r} is not laid out over the scored {observation_dim!r} "
                    "dimension"
                )
            return np.asarray(stored.values)
    raise PosteriorError(
        f"block {block_name!r} is not retained in constant_data or observed_data; pass "
        f"block_values explicitly or choose one of {sorted(available)}"
    )


def _blocked_result(
    result: PosteriorResult,
    variable: PosteriorVariable,
    block_name: str,
    labels: NDArray[Any],
) -> PosteriorResult:
    """Sum the pointwise log likelihood within each block, before any importance sampling.

    Summing on the log scale before PSIS is what changes the estimand: the importance ratio
    that PSIS smooths is then the ratio for removing the whole block's joint contribution, not
    one observation's. Summing the *pointwise ELPD* afterwards would only re-aggregate
    leave-one-trial-out numbers and would keep the optimistic estimand.
    """

    posterior = result["posterior"]
    reserved = {dim for item in posterior.variables for dim in item.dims}
    if block_name in reserved:
        raise PosteriorError(
            f"block {block_name!r} collides with a posterior dimension of the same name; "
            "pass a distinct block name"
        )
    blocks, inverse = _first_appearance_unique(labels)
    values = np.asarray(variable.values, dtype=np.float64)
    axis = variable.dims.index(variable.intrinsic_dims[0])
    moved = np.moveaxis(values, axis, -1)
    summed = np.stack(
        [moved[..., inverse == position].sum(axis=-1) for position in range(len(blocks))],
        axis=-1,
    )
    coords = {dim: variable.coords[dim] for dim in SAMPLE_DIMS}
    blocked = PosteriorVariable(
        variable.name,
        summed,
        (*SAMPLE_DIMS, block_name),
        {**coords, block_name: blocks},
    )
    return PosteriorResult(
        model_name=result.model_name,
        model_signature=result.model_signature,
        inference_library=result.inference_library,
        inference_library_version=result.inference_library_version,
        parameter_names=result.parameter_names,
        groups=(posterior, PosteriorGroup("log_likelihood", (blocked,))),
        parameter_space_fingerprint=result.parameter_space_fingerprint,
    )


def _first_appearance_unique(labels: NDArray[Any]) -> tuple[NDArray[Any], NDArray[np.intp]]:
    """Return block labels in first-appearance order with each observation's block index.

    First appearance rather than sort order keeps subject and session boundaries in the order
    the study presents them, which is the ordering every other labelled array in this package
    preserves.
    """

    unique, first_index, inverse = np.unique(labels, return_index=True, return_inverse=True)
    order = np.argsort(first_index, kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(order))
    return unique[order], ranks[inverse.reshape(-1)]


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
