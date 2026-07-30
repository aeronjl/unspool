"""Composable posterior-predictive checks over labelled replicated observations.

``PredictiveDiscrepancy`` and ``PredictiveTail`` now live in
:mod:`behavio.contracts.discrepancy` and are re-exported here, so
``from behavio.posterior_predictive import PredictiveDiscrepancy`` keeps working.

Convergence gating
------------------
A posterior-predictive check summarises draws. If those draws are not draws from the
posterior at all -- chains that never mixed, or a sampler that diverged -- every tail
probability below is meaningless. :func:`posterior_predictive_check` therefore runs
:func:`behavio.posterior_diagnostics.audit_posterior` on the same result and retains the
audit. It follows the package idiom of *retaining the evidence and marking the status*
rather than raising: the returned :class:`PosteriorPredictiveAudit` still holds every
reference distribution, and its ``status`` becomes ``FAIL`` so the layer above can refuse
to publish it. A caller who disagrees with the default gate injects a
:class:`~behavio.posterior_diagnostics.PosteriorAuditPolicy` whose per-check severities
name exactly which failures they are willing to tolerate; the downgraded policy is
recorded verbatim in ``to_dict()``, so the decision stays in the artifact.

Simultaneous inference
----------------------
One call evaluates ``len(groupby groups) x len(discrepancies)`` checks against the same
tail-probability threshold. Thirty subjects and four discrepancies is one hundred and
twenty simultaneous tests, so roughly six of them fall below ``0.05`` *by construction*
even when the model is perfect. Emitting one warning per extreme check therefore trains
readers to ignore the warning.

The family is now explicit. :class:`PredictiveFamily` records its size, the number of
checks below the per-check threshold, and how many were expected there by chance, and it
is always present in the output whether or not anything is flagged. Per-check issues are
emitted only for checks that survive a declared multiplicity adjustment
(:class:`PredictiveMultiplicity`, Benjamini-Hochberg by default) at the declared
``family_discovery_rate``. Nothing about ``tail_probability_warning`` changed: it still
defines ``n_extreme`` exactly as before, and every per-check ``tail_probability`` is
unadjusted and fully retained. A single check is never adjusted, so an ungrouped
one-discrepancy audit behaves exactly as it did.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
from numpy.typing import NDArray

from behavio._internal import multiplicity as _multiplicity
from behavio._internal.multiplicity import adjust_probabilities, excess_probability
from behavio.contracts.audit import AuditSeverity
from behavio.contracts.discrepancy import (
    PredictiveDiscrepancy,
    PredictiveTail,
)
from behavio.posterior import ArviZUnavailableError, PosteriorError, PosteriorResult
from behavio.posterior_diagnostics import (
    PosteriorAudit,
    PosteriorAuditPolicy,
    PosteriorAuditStatus,
    audit_posterior,
)


@dataclass(frozen=True, slots=True)
class MeanDiscrepancy:
    """Compare the observed and replicated arithmetic mean."""

    tail: PredictiveTail = PredictiveTail.TWO_SIDED

    def __post_init__(self) -> None:
        object.__setattr__(self, "tail", PredictiveTail(self.tail))

    @property
    def name(self) -> str:
        return "mean"

    @property
    def signature(self) -> str:
        return f"mean[v1;tail={self.tail.value}]"

    def evaluate(self, values: NDArray[Any]) -> float:
        return float(np.mean(_numeric(values, self.name)))


@dataclass(frozen=True, slots=True)
class VarianceDiscrepancy:
    """Compare observed and replicated population variance."""

    tail: PredictiveTail = PredictiveTail.UPPER

    def __post_init__(self) -> None:
        object.__setattr__(self, "tail", PredictiveTail(self.tail))

    @property
    def name(self) -> str:
        return "variance"

    @property
    def signature(self) -> str:
        return f"variance[v1;ddof=0;tail={self.tail.value}]"

    def evaluate(self, values: NDArray[Any]) -> float:
        return float(np.var(_numeric(values, self.name), ddof=0))


@dataclass(frozen=True, slots=True)
class CategoryRateDiscrepancy:
    """Compare the rate of one declared categorical response."""

    category: Any
    tail: PredictiveTail = PredictiveTail.TWO_SIDED

    def __post_init__(self) -> None:
        object.__setattr__(self, "tail", PredictiveTail(self.tail))
        try:
            hash(self.category)
        except TypeError as error:
            raise ValueError("category must be a scalar hashable value") from error

    @property
    def name(self) -> str:
        return f"rate[{self.category!r}]"

    @property
    def signature(self) -> str:
        return f"category-rate[v1;category={self.category!r};tail={self.tail.value}]"

    def evaluate(self, values: NDArray[Any]) -> float:
        return float(np.mean(np.asarray(values) == self.category))


@dataclass(frozen=True, slots=True)
class SwitchRateDiscrepancy:
    """Compare adjacent-response switch rate in the supplied row order."""

    tail: PredictiveTail = PredictiveTail.TWO_SIDED

    def __post_init__(self) -> None:
        object.__setattr__(self, "tail", PredictiveTail(self.tail))

    @property
    def name(self) -> str:
        return "switch-rate"

    @property
    def signature(self) -> str:
        return f"switch-rate[v1;tail={self.tail.value}]"

    def evaluate(self, values: NDArray[Any]) -> float:
        array = np.asarray(values)
        if array.ndim != 1 or len(array) < 2:
            raise ValueError("switch-rate requires at least two ordered observations")
        return float(np.mean(array[1:] != array[:-1]))


class PredictiveMultiplicity(StrEnum):
    """How one audit converts many simultaneous tail probabilities into warnings.

    ``NONE`` restores the per-comparison behaviour: every check below
    ``tail_probability_warning`` raises its own issue, and the expected number of
    spurious issues grows linearly with the family size. ``BENJAMINI_HOCHBERG`` controls
    the false-discovery rate across the family at ``family_discovery_rate``.
    ``BONFERRONI`` controls the family-wise error rate at the same level.

    The member values are shared with
    :class:`~behavio.comparison.ComparisonMultiplicity` through
    :mod:`behavio._internal.multiplicity`, which also owns the step-up itself, so the two
    families the package evaluates cannot drift apart in either spelling or arithmetic.
    """

    NONE = _multiplicity.NONE
    BENJAMINI_HOCHBERG = _multiplicity.BENJAMINI_HOCHBERG
    BONFERRONI = _multiplicity.BONFERRONI


@dataclass(frozen=True, slots=True)
class PosteriorPredictivePolicy:
    """Explicit interval, tail-probability, and simultaneous-inference thresholds.

    ``tail_probability_warning`` keeps its original per-check meaning and still defines
    which checks count as extreme. ``multiplicity`` and ``family_discovery_rate`` govern
    only which of those extreme checks become issues once more than one check is
    evaluated in the same call.
    """

    interval_probability: float = 0.9
    tail_probability_warning: float = 0.05
    multiplicity: PredictiveMultiplicity = PredictiveMultiplicity.BENJAMINI_HOCHBERG
    family_discovery_rate: float = 0.05

    def __post_init__(self) -> None:
        if not np.isfinite(self.interval_probability) or not 0 < self.interval_probability < 1:
            raise ValueError("interval_probability must be finite and lie between zero and one")
        if (
            not np.isfinite(self.tail_probability_warning)
            or not 0 < self.tail_probability_warning < 0.5
        ):
            raise ValueError("tail_probability_warning must be finite and lie between zero and 0.5")
        if not np.isfinite(self.family_discovery_rate) or not 0 < self.family_discovery_rate < 0.5:
            raise ValueError("family_discovery_rate must be finite and lie between zero and 0.5")
        object.__setattr__(self, "multiplicity", PredictiveMultiplicity(self.multiplicity))

    def to_dict(self) -> dict[str, float | str]:
        return {
            "interval_probability": self.interval_probability,
            "tail_probability_warning": self.tail_probability_warning,
            "multiplicity": self.multiplicity.value,
            "family_discovery_rate": self.family_discovery_rate,
        }


@dataclass(frozen=True, slots=True)
class PosteriorPredictiveCheck:
    """One observed discrepancy placed in its posterior-predictive reference distribution."""

    discrepancy_name: str
    discrepancy_signature: str
    tail: PredictiveTail
    group: tuple[tuple[str, Any], ...]
    n_observations: int
    observed: float
    replicated: NDArray[np.float64] | Sequence[float]
    interval: tuple[float, float]
    lower_probability: float
    upper_probability: float
    tail_probability: float

    def __post_init__(self) -> None:
        if not self.discrepancy_name or not self.discrepancy_signature:
            raise ValueError("predictive checks require discrepancy identity")
        object.__setattr__(self, "tail", PredictiveTail(self.tail))
        if self.n_observations < 1:
            raise ValueError("n_observations must be positive")
        replicated = _protected(np.asarray(self.replicated, dtype=np.float64))
        if replicated.ndim != 2 or not np.all(np.isfinite(replicated)):
            raise ValueError("replicated discrepancies must be a finite chain-by-draw array")
        if not np.isfinite(self.observed):
            raise ValueError("observed discrepancy must be finite")
        if len(self.interval) != 2 or not np.all(np.isfinite(self.interval)):
            raise ValueError("predictive interval must contain two finite values")
        if self.interval[0] > self.interval[1]:
            raise ValueError("predictive interval must be ordered")
        for value in (
            self.lower_probability,
            self.upper_probability,
            self.tail_probability,
        ):
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("predictive probabilities must lie between zero and one")
        group = tuple(self.group)
        if any(len(item) != 2 or not isinstance(item[0], str) or not item[0] for item in group):
            raise ValueError("predictive-check groups must contain named scalar labels")
        object.__setattr__(self, "replicated", replicated)
        object.__setattr__(self, "group", group)
        object.__setattr__(self, "observed", float(self.observed))
        object.__setattr__(self, "interval", tuple(float(value) for value in self.interval))


@dataclass(frozen=True, slots=True)
class PosteriorPredictiveIssue:
    """One stable warning about a predictive audit.

    ``discrepancy_signature`` is ``None`` for issues that are properties of the whole
    audit -- the convergence gate and the family-level discrepancy rate -- rather than of
    one check. ``severity`` follows :class:`~behavio.contracts.audit.AuditSeverity`, so an
    unusable posterior is an ``ERROR`` while an extreme discrepancy is a ``WARNING``.
    """

    code: str
    message: str
    discrepancy_signature: str | None = None
    group: tuple[tuple[str, Any], ...] = ()
    severity: AuditSeverity = AuditSeverity.WARNING

    def __post_init__(self) -> None:
        if not self.code or not self.message:
            raise ValueError("predictive-check issues require identity and a message")
        if self.discrepancy_signature is not None and not self.discrepancy_signature:
            raise ValueError("predictive-check issues require identity and a message")
        object.__setattr__(self, "group", tuple(self.group))
        object.__setattr__(self, "severity", AuditSeverity(self.severity))


@dataclass(frozen=True, slots=True)
class PredictiveFamily:
    """The simultaneous family of discrepancy checks evaluated in one audit.

    Retained whether or not anything is flagged, so a reader can always compare the
    observed number of extreme checks with the number expected at the declared per-check
    threshold by chance alone.
    """

    n_checks: int
    n_groups: int
    n_discrepancies: int
    tail_probability_warning: float
    multiplicity: PredictiveMultiplicity
    family_discovery_rate: float
    n_extreme: int
    expected_extreme: float
    excess_probability: float
    adjusted_threshold: float
    n_flagged: int

    def __post_init__(self) -> None:
        if self.n_checks < 1 or self.n_groups < 1 or self.n_discrepancies < 1:
            raise ValueError("a predictive family must contain at least one check")
        if self.n_checks != self.n_groups * self.n_discrepancies:
            raise ValueError("predictive family size must be groups times discrepancies")
        if not 0 <= self.n_extreme <= self.n_checks or not 0 <= self.n_flagged <= self.n_checks:
            raise ValueError("predictive family counts must lie inside the family")
        for value in (
            self.expected_extreme,
            self.excess_probability,
            self.adjusted_threshold,
        ):
            if not np.isfinite(value) or value < 0:
                raise ValueError("predictive family rates must be finite and non-negative")
        object.__setattr__(self, "multiplicity", PredictiveMultiplicity(self.multiplicity))

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_checks": self.n_checks,
            "n_groups": self.n_groups,
            "n_discrepancies": self.n_discrepancies,
            "tail_probability_warning": self.tail_probability_warning,
            "multiplicity": self.multiplicity.value,
            "family_discovery_rate": self.family_discovery_rate,
            "n_extreme": self.n_extreme,
            "expected_extreme": self.expected_extreme,
            "excess_probability": self.excess_probability,
            "adjusted_threshold": self.adjusted_threshold,
            "n_flagged": self.n_flagged,
        }


@dataclass(frozen=True, slots=True)
class PosteriorPredictiveAudit:
    """All requested discrepancy checks for one replicated outcome."""

    model_name: str
    model_signature: str
    variable_name: str
    policy: PosteriorPredictivePolicy
    checks: tuple[PosteriorPredictiveCheck, ...]
    issues: tuple[PosteriorPredictiveIssue, ...]
    family: PredictiveFamily
    convergence: PosteriorAudit | None = None

    def __post_init__(self) -> None:
        if not self.model_name or not self.model_signature or not self.variable_name:
            raise ValueError("predictive audits require model and variable identity")
        checks = tuple(self.checks)
        issues = tuple(self.issues)
        if not checks or any(not isinstance(item, PosteriorPredictiveCheck) for item in checks):
            raise ValueError("predictive audits require PosteriorPredictiveCheck records")
        if any(not isinstance(item, PosteriorPredictiveIssue) for item in issues):
            raise ValueError("predictive audit issues must be PosteriorPredictiveIssue records")
        if not isinstance(self.family, PredictiveFamily):
            raise TypeError("predictive audits require a PredictiveFamily record")
        if self.family.n_checks != len(checks):
            raise ValueError("predictive family size must match the retained checks")
        if self.convergence is not None and not isinstance(self.convergence, PosteriorAudit):
            raise TypeError("predictive audit convergence must be a PosteriorAudit")
        object.__setattr__(self, "checks", checks)
        object.__setattr__(self, "issues", issues)

    @property
    def status(self) -> PosteriorAuditStatus:
        """Return fail, warning, or pass without collapsing the underlying issues.

        ``FAIL`` means at least one issue is an ``ERROR`` -- in practice a posterior that
        failed its convergence audit, whose replicated draws are therefore not draws from
        the posterior and whose tail probabilities cannot be interpreted.
        """

        if any(issue.severity is AuditSeverity.ERROR for issue in self.issues):
            return PosteriorAuditStatus.FAIL
        if self.issues:
            return PosteriorAuditStatus.WARNING
        return PosteriorAuditStatus.PASS

    @property
    def issue_codes(self) -> tuple[str, ...]:
        return tuple(issue.code for issue in self.issues)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible record of every requested predictive check."""

        return {
            "model_name": self.model_name,
            "model_signature": self.model_signature,
            "variable_name": self.variable_name,
            "status": self.status.value,
            "policy": self.policy.to_dict(),
            "family": self.family.to_dict(),
            "convergence": None if self.convergence is None else self.convergence.to_dict(),
            "checks": [
                {
                    "discrepancy_name": check.discrepancy_name,
                    "discrepancy_signature": check.discrepancy_signature,
                    "tail": check.tail.value,
                    "group": dict(check.group),
                    "n_observations": check.n_observations,
                    "observed": check.observed,
                    "replicated": check.replicated.tolist(),
                    "interval": list(check.interval),
                    "lower_probability": check.lower_probability,
                    "upper_probability": check.upper_probability,
                    "tail_probability": check.tail_probability,
                }
                for check in self.checks
            ],
            "issues": [
                {
                    "code": issue.code,
                    "message": issue.message,
                    "discrepancy_signature": issue.discrepancy_signature,
                    "group": dict(issue.group),
                    "severity": issue.severity.value,
                }
                for issue in self.issues
            ],
        }


def posterior_predictive_check(
    result: PosteriorResult,
    discrepancies: Sequence[PredictiveDiscrepancy],
    *,
    variable_name: str | None = None,
    groupby: Sequence[str] = (),
    policy: PosteriorPredictivePolicy | None = None,
    audit_policy: PosteriorAuditPolicy | None = None,
) -> PosteriorPredictiveAudit:
    """Evaluate declared discrepancies against labelled posterior-predictive draws.

    The result's convergence is audited first with ``audit_policy`` and the audit is
    retained on :attr:`PosteriorPredictiveAudit.convergence`. A failing audit does not
    raise: it marks the returned audit ``FAIL`` so a caller cannot publish tail
    probabilities computed from chains that never mixed without stepping over an explicit
    status. To accept such a posterior deliberately, pass a
    :class:`~behavio.posterior_diagnostics.PosteriorAuditPolicy` that names the severities
    you are downgrading; the policy is recorded in ``to_dict()``.

    Issues are emitted for the subset of extreme checks that survive the declared
    multiplicity adjustment over the whole ``groups x discrepancies`` family, whose size
    and expected chance yield are always reported on :attr:`PosteriorPredictiveAudit.family`.
    """

    if not isinstance(result, PosteriorResult):
        raise TypeError("result must be a PosteriorResult")
    selected_policy = PosteriorPredictivePolicy() if policy is None else policy
    if not isinstance(selected_policy, PosteriorPredictivePolicy):
        raise TypeError("policy must be a PosteriorPredictivePolicy")
    if audit_policy is not None and not isinstance(audit_policy, PosteriorAuditPolicy):
        raise TypeError("audit_policy must be a PosteriorAuditPolicy")
    declared = tuple(discrepancies)
    if not declared or any(not isinstance(item, PredictiveDiscrepancy) for item in declared):
        raise TypeError("discrepancies must contain PredictiveDiscrepancy objects")
    for item in declared:
        if not isinstance(item.name, str) or not item.name:
            raise ValueError("discrepancy names must be non-empty strings")
        if not isinstance(item.signature, str) or not item.signature:
            raise ValueError("discrepancy signatures must be non-empty strings")
        if not isinstance(item.tail, PredictiveTail):
            raise ValueError("discrepancy tails must be PredictiveTail values")
    signatures = tuple(item.signature for item in declared)
    if len(set(signatures)) != len(signatures):
        raise ValueError("discrepancy signatures must be unique")
    observed, predictive = _paired_variables(result, variable_name)
    groups = _groups(result, observed, tuple(groupby))
    checks = tuple(
        _check(discrepancy, labels, mask, observed.values, predictive.values, selected_policy)
        for labels, mask in groups
        for discrepancy in declared
    )
    convergence, unavailable = _convergence(result, audit_policy)
    family, flagged = _family(checks, len(groups), len(declared), selected_policy)
    issues = [
        *_convergence_issues(convergence, unavailable),
        *(
            PosteriorPredictiveIssue(
                code="ppc.extreme-discrepancy",
                message=(
                    f"observed {check.discrepancy_name} has tail probability "
                    f"{check.tail_probability:.3g} and survives "
                    f"{family.multiplicity.value} adjustment over "
                    f"{family.n_checks} simultaneous checks"
                ),
                discrepancy_signature=check.discrepancy_signature,
                group=check.group,
            )
            for check, is_flagged in zip(checks, flagged, strict=True)
            if is_flagged
        ),
    ]
    if (
        family.n_flagged == 0
        and family.n_checks > 1
        and family.excess_probability < selected_policy.family_discovery_rate
    ):
        issues.append(
            PosteriorPredictiveIssue(
                code="ppc.extreme-discrepancy-rate",
                message=(
                    f"{family.n_extreme} of {family.n_checks} simultaneous checks fall below "
                    f"{selected_policy.tail_probability_warning:.3g} where "
                    f"{family.expected_extreme:.3g} are expected by chance "
                    f"(p={family.excess_probability:.3g}), but no single check survives "
                    f"{family.multiplicity.value} adjustment"
                ),
            )
        )
    return PosteriorPredictiveAudit(
        model_name=result.model_name,
        model_signature=result.model_signature,
        variable_name=observed.name,
        policy=selected_policy,
        checks=checks,
        issues=tuple(issues),
        family=family,
        convergence=convergence,
    )


def _convergence(
    result: PosteriorResult,
    policy: PosteriorAuditPolicy | None,
) -> tuple[PosteriorAudit | None, bool]:
    """Audit convergence, reporting rather than hiding a missing diagnostics backend."""

    try:
        return audit_posterior(result, policy=policy), False
    except ArviZUnavailableError:
        return None, True


def _convergence_issues(
    convergence: PosteriorAudit | None,
    unavailable: bool,
) -> tuple[PosteriorPredictiveIssue, ...]:
    if unavailable:
        return (
            PosteriorPredictiveIssue(
                code="ppc.posterior-audit-unavailable",
                message=(
                    "convergence could not be audited because the ArviZ diagnostics "
                    "backend is unavailable; install `behavio[probabilistic]`"
                ),
            ),
        )
    if convergence is None or convergence.status is PosteriorAuditStatus.PASS:
        return ()
    codes = ", ".join(convergence.issue_codes)
    if convergence.status is PosteriorAuditStatus.FAIL:
        return (
            PosteriorPredictiveIssue(
                code="ppc.unconverged-posterior",
                message=(
                    "the replicated draws come from a posterior that failed its "
                    f"convergence audit ({codes}); the tail probabilities below are not "
                    "interpretable"
                ),
                severity=AuditSeverity.ERROR,
            ),
        )
    return (
        PosteriorPredictiveIssue(
            code="ppc.posterior-diagnostic-warning",
            message=f"the source posterior carries convergence warnings ({codes})",
        ),
    )


def _family(
    checks: tuple[PosteriorPredictiveCheck, ...],
    n_groups: int,
    n_discrepancies: int,
    policy: PosteriorPredictivePolicy,
) -> tuple[PredictiveFamily, tuple[bool, ...]]:
    """Size the simultaneous family and decide which checks survive adjustment."""

    probabilities = np.asarray([check.tail_probability for check in checks], dtype=np.float64)
    n_checks = len(probabilities)
    alpha = policy.tail_probability_warning
    rate = policy.family_discovery_rate
    extreme = probabilities < alpha
    n_extreme = int(np.count_nonzero(extreme))
    expected = float(n_checks * alpha)
    excess = excess_probability(n_extreme, n_checks, alpha)
    threshold, adjusted = adjust_probabilities(
        probabilities,
        multiplicity=policy.multiplicity.value,
        error_rate=rate,
        unadjusted_threshold=alpha,
    )
    unadjusted = n_checks == 1 or policy.multiplicity is PredictiveMultiplicity.NONE
    flagged = extreme if unadjusted else adjusted <= rate
    family = PredictiveFamily(
        n_checks=n_checks,
        n_groups=n_groups,
        n_discrepancies=n_discrepancies,
        tail_probability_warning=alpha,
        multiplicity=policy.multiplicity,
        family_discovery_rate=rate,
        n_extreme=n_extreme,
        expected_extreme=expected,
        excess_probability=excess,
        adjusted_threshold=float(threshold),
        n_flagged=int(np.count_nonzero(flagged)),
    )
    return family, tuple(bool(value) for value in flagged)


def _paired_variables(result: PosteriorResult, name: str | None) -> tuple[Any, Any]:
    if (
        "observed_data" not in result.group_names
        or "posterior_predictive" not in result.group_names
    ):
        raise PosteriorError("predictive checks require observed_data and posterior_predictive")
    observed = result["observed_data"]
    predictive = result["posterior_predictive"]
    common = tuple(item for item in observed.variable_names if item in predictive.variable_names)
    if name is None:
        if len(common) != 1:
            raise PosteriorError("variable_name is required unless exactly one outcome is shared")
        name = common[0]
    if not isinstance(name, str) or not name:
        raise ValueError("variable_name must be null or a non-empty string")
    if name not in common:
        raise PosteriorError(f"no paired observed and predictive variable {name!r}")
    observed_variable = observed[name]
    predictive_variable = predictive[name]
    if predictive_variable.intrinsic_dims != observed_variable.dims:
        raise PosteriorError("observed and posterior-predictive dimensions must match")
    for dim in observed_variable.dims:
        if not np.array_equal(observed_variable.coords[dim], predictive_variable.coords[dim]):
            raise PosteriorError("observed and posterior-predictive coordinates must match")
    return observed_variable, predictive_variable


def _groups(
    result: PosteriorResult,
    observed: Any,
    groupby: tuple[str, ...],
) -> tuple[tuple[tuple[tuple[str, Any], ...], NDArray[np.bool_]], ...]:
    size = np.asarray(observed.values).size
    if not groupby:
        return (((), np.ones(size, dtype=bool)),)
    if len(set(groupby)) != len(groupby) or any(not name for name in groupby):
        raise ValueError("groupby names must be non-empty and unique")
    if "constant_data" not in result.group_names:
        raise PosteriorError("grouped predictive checks require constant_data")
    constants = result["constant_data"]
    arrays = []
    for name in groupby:
        if name not in constants.variable_names:
            raise PosteriorError(f"constant_data has no grouping variable {name!r}")
        variable = constants[name]
        if variable.dims != observed.dims:
            raise PosteriorError(f"grouping variable {name!r} must match observed dimensions")
        arrays.append(np.asarray(variable.values).reshape(-1))
    keys: list[tuple[Any, ...]] = []
    for values in zip(*arrays, strict=True):
        key = tuple(_scalar(value) for value in values)
        if key not in keys:
            keys.append(key)
    rows = tuple(zip(*arrays, strict=True))
    return tuple(
        (
            tuple(zip(groupby, key, strict=True)),
            np.asarray(
                [
                    all(value == expected for value, expected in zip(values, key, strict=True))
                    for values in rows
                ],
                dtype=bool,
            ),
        )
        for key in keys
    )


def _check(
    discrepancy: PredictiveDiscrepancy,
    labels: tuple[tuple[str, Any], ...],
    mask: NDArray[np.bool_],
    observed: NDArray[Any],
    predictive: NDArray[Any],
    policy: PosteriorPredictivePolicy,
) -> PosteriorPredictiveCheck:
    flat_observed = np.asarray(observed).reshape(-1)[mask]
    flat_predictive = np.asarray(predictive).reshape(*predictive.shape[:2], -1)[..., mask]
    observed_value = discrepancy.evaluate(flat_observed)
    replicated = np.empty(flat_predictive.shape[:2], dtype=np.float64)
    for chain in range(replicated.shape[0]):
        for draw in range(replicated.shape[1]):
            replicated[chain, draw] = discrepancy.evaluate(flat_predictive[chain, draw])
    if not np.isfinite(observed_value) or not np.all(np.isfinite(replicated)):
        raise PosteriorError(f"discrepancy {discrepancy.name!r} returned non-finite values")
    lower = float(np.mean(replicated <= observed_value))
    upper = float(np.mean(replicated >= observed_value))
    tail_probability = {
        PredictiveTail.LOWER: lower,
        PredictiveTail.UPPER: upper,
        PredictiveTail.TWO_SIDED: min(1.0, 2.0 * min(lower, upper)),
    }[discrepancy.tail]
    alpha = (1.0 - policy.interval_probability) / 2.0
    interval = tuple(np.quantile(replicated, (alpha, 1.0 - alpha)))
    return PosteriorPredictiveCheck(
        discrepancy_name=discrepancy.name,
        discrepancy_signature=discrepancy.signature,
        tail=discrepancy.tail,
        group=labels,
        n_observations=int(np.count_nonzero(mask)),
        observed=observed_value,
        replicated=replicated,
        interval=interval,
        lower_probability=lower,
        upper_probability=upper,
        tail_probability=tail_probability,
    )


def _numeric(values: NDArray[Any], name: str) -> NDArray[np.float64]:
    try:
        numeric = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} requires numeric observations") from error
    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"{name} requires finite observations")
    return numeric


def _scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _protected(values: NDArray[Any]) -> NDArray[Any]:
    protected = np.array(values, copy=True)
    protected.setflags(write=False)
    return protected
