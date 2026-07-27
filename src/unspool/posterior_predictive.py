"""Composable posterior-predictive checks over labelled replicated observations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from unspool.posterior import PosteriorError, PosteriorResult
from unspool.posterior_diagnostics import PosteriorAuditStatus


class PredictiveTail(StrEnum):
    """Reference tail used to summarize a replicated discrepancy."""

    LOWER = "lower"
    UPPER = "upper"
    TWO_SIDED = "two-sided"


@runtime_checkable
class PredictiveDiscrepancy(Protocol):
    """A named, provenance-bearing scalar summary of one observation vector."""

    @property
    def name(self) -> str: ...

    @property
    def signature(self) -> str: ...

    @property
    def tail(self) -> PredictiveTail: ...

    def evaluate(self, values: NDArray[Any]) -> float: ...


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


@dataclass(frozen=True, slots=True)
class PosteriorPredictivePolicy:
    """Explicit interval and tail-probability thresholds for predictive checks."""

    interval_probability: float = 0.9
    tail_probability_warning: float = 0.05

    def __post_init__(self) -> None:
        if not np.isfinite(self.interval_probability) or not 0 < self.interval_probability < 1:
            raise ValueError("interval_probability must be finite and lie between zero and one")
        if (
            not np.isfinite(self.tail_probability_warning)
            or not 0 < self.tail_probability_warning < 0.5
        ):
            raise ValueError("tail_probability_warning must be finite and lie between zero and 0.5")


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
    """One stable warning for an extreme observed discrepancy."""

    code: str
    message: str
    discrepancy_signature: str
    group: tuple[tuple[str, Any], ...]

    def __post_init__(self) -> None:
        if not self.code or not self.message or not self.discrepancy_signature:
            raise ValueError("predictive-check issues require identity and a message")
        object.__setattr__(self, "group", tuple(self.group))


@dataclass(frozen=True, slots=True)
class PosteriorPredictiveAudit:
    """All requested discrepancy checks for one replicated outcome."""

    model_name: str
    model_signature: str
    variable_name: str
    policy: PosteriorPredictivePolicy
    checks: tuple[PosteriorPredictiveCheck, ...]
    issues: tuple[PosteriorPredictiveIssue, ...]

    def __post_init__(self) -> None:
        if not self.model_name or not self.model_signature or not self.variable_name:
            raise ValueError("predictive audits require model and variable identity")
        checks = tuple(self.checks)
        issues = tuple(self.issues)
        if not checks or any(not isinstance(item, PosteriorPredictiveCheck) for item in checks):
            raise ValueError("predictive audits require PosteriorPredictiveCheck records")
        if any(not isinstance(item, PosteriorPredictiveIssue) for item in issues):
            raise ValueError("predictive audit issues must be PosteriorPredictiveIssue records")
        object.__setattr__(self, "checks", checks)
        object.__setattr__(self, "issues", issues)

    @property
    def status(self) -> PosteriorAuditStatus:
        return PosteriorAuditStatus.PASS if not self.issues else PosteriorAuditStatus.WARNING

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
            "policy": {
                "interval_probability": self.policy.interval_probability,
                "tail_probability_warning": self.policy.tail_probability_warning,
            },
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
) -> PosteriorPredictiveAudit:
    """Evaluate declared discrepancies against labelled posterior-predictive draws."""

    if not isinstance(result, PosteriorResult):
        raise TypeError("result must be a PosteriorResult")
    selected_policy = PosteriorPredictivePolicy() if policy is None else policy
    if not isinstance(selected_policy, PosteriorPredictivePolicy):
        raise TypeError("policy must be a PosteriorPredictivePolicy")
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
    issues = tuple(
        PosteriorPredictiveIssue(
            code="ppc.extreme-discrepancy",
            message=(
                f"observed {check.discrepancy_name} has tail probability "
                f"{check.tail_probability:.3g}"
            ),
            discrepancy_signature=check.discrepancy_signature,
            group=check.group,
        )
        for check in checks
        if check.tail_probability < selected_policy.tail_probability_warning
    )
    return PosteriorPredictiveAudit(
        model_name=result.model_name,
        model_signature=result.model_signature,
        variable_name=observed.name,
        policy=selected_policy,
        checks=checks,
        issues=issues,
    )


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
