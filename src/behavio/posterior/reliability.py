"""Test-retest consistency and absolute agreement for labelled subject estimates.

Two sources of uncertainty
--------------------------
A subject's estimate on one occasion is not a measurement, it is an inference. Collapsing
a posterior to one number per subject and then bootstrapping only over subjects propagates
sampling uncertainty and throws posterior uncertainty away, which makes every reported
interval too narrow. When both occasions carry posterior draws
(:attr:`SubjectEstimates.draws`, attached automatically by
:func:`posterior_subject_estimates`), each repetition now resamples subjects *and* draws a
fresh posterior draw for each occasion, so the retained interval covers both.

**Draw pairing.** The two occasions are separate fits. Draw ``d`` of the test posterior and
draw ``d`` of the retest posterior are not one joint sample of anything: there is no
coupling between them to preserve, and inventing one -- pairing by index, or ordering draws
to match -- would fabricate a correlation. Each repetition therefore samples a draw index
independently and uniformly from each occasion, which treats the draw index as an
independent resample of that occasion's own uncertainty. The reported statistic is the
posterior mean of the per-draw statistic, not the statistic of the posterior means.

Shrinkage
---------
Posterior means from a partial-pooling model are shrunk toward a common mean, and
correlating two sets of shrunken estimates inflates Pearson, Spearman, and both ICCs: the
shared prior pulls both occasions toward the same point, which reads as agreement. This
module *warns* about that with ``reliability.shrunken-estimates``; it does not correct it.
There is no correction available here, because undoing shrinkage requires the hierarchical
model's variance components, which a :class:`SubjectEstimates` record does not carry.
Treat a flagged reliability estimate as an upper bound.

Whether the source model pools is a fact about the model, not about the numbers, so it is
declared rather than guessed: see :class:`SubjectPooling`. Estimates built by hand default
to :attr:`SubjectPooling.NONE` (independently fitted subjects, the maximum-likelihood
path), and :func:`posterior_subject_estimates` defaults to
:attr:`SubjectPooling.PARTIAL`, the conservative side.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from behavio.posterior.diagnostics import (
    PosteriorAudit,
    PosteriorAuditPolicy,
    PosteriorAuditStatus,
    audit_posterior,
)
from behavio.posterior.result import ArviZUnavailableError, PosteriorResult

_SHRINKABLE: tuple[str, ...] = (
    "pearson-r",
    "spearman-rho",
    "icc-c-1",
    "icc-a-1",
)


class ReliabilityError(ValueError):
    """Raised when a reliability analysis violates its pairing contract."""


class SubjectPooling(StrEnum):
    """Whether the model behind one occasion's estimates pooled across subjects.

    ``NONE`` declares that each subject was estimated independently, so the estimates
    carry no shrinkage. ``PARTIAL`` declares a hierarchical or partial-pooling model,
    whose per-subject estimates are shrunk toward a common mean and therefore inflate
    every consistency and agreement statistic computed from them.
    """

    NONE = "none"
    PARTIAL = "partial"


class ReliabilityStatistic(StrEnum):
    """Scalar consistency or agreement statistic retained by the report."""

    PEARSON = "pearson-r"
    SPEARMAN = "spearman-rho"
    ICC_CONSISTENCY = "icc-c-1"
    ICC_ABSOLUTE_AGREEMENT = "icc-a-1"
    MEAN_DIFFERENCE = "mean-difference"
    SD_DIFFERENCE = "sd-difference"
    WITHIN_SUBJECT_SD = "within-subject-sd"
    LOWER_LIMIT_OF_AGREEMENT = "lower-limit-of-agreement"
    UPPER_LIMIT_OF_AGREEMENT = "upper-limit-of-agreement"
    MEAN_ABSOLUTE_ERROR = "mean-absolute-error"
    ROOT_MEAN_SQUARED_ERROR = "root-mean-squared-error"


@dataclass(frozen=True, slots=True)
class ReliabilityPolicy:
    """Explicit uncertainty and estimability policy for paired measurements."""

    bootstrap_repeats: int = 2_000
    interval_probability: float = 0.95
    agreement_multiplier: float = 1.96
    minimum_subjects_warning: int = 30
    minimum_bootstrap_fraction: float = 0.8

    def __post_init__(self) -> None:
        for value, label, minimum in (
            (self.bootstrap_repeats, "bootstrap_repeats", 0),
            (self.minimum_subjects_warning, "minimum_subjects_warning", 3),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{label} must be an integer of at least {minimum}")
        if not np.isfinite(self.interval_probability) or not 0 < self.interval_probability < 1:
            raise ValueError("interval_probability must be finite and lie between zero and one")
        if not np.isfinite(self.agreement_multiplier) or self.agreement_multiplier <= 0:
            raise ValueError("agreement_multiplier must be finite and positive")
        if (
            not np.isfinite(self.minimum_bootstrap_fraction)
            or not 0 < self.minimum_bootstrap_fraction <= 1
        ):
            raise ValueError(
                "minimum_bootstrap_fraction must be finite and lie between zero and one"
            )


@dataclass(frozen=True, slots=True)
class SubjectEstimates:
    """One labelled scalar estimate per subject on one occasion.

    ``values`` is the point estimate per subject and is what every non-posterior caller
    supplies. ``draws`` optionally retains the full ``(sample, subject)`` posterior behind
    those points so that :func:`assess_test_retest_reliability` can propagate posterior
    uncertainty instead of discarding it; ``values`` remains the posterior mean, so the
    reported point summary of the estimates themselves does not change. ``pooling``
    declares whether those estimates are shrunk, and ``posterior_audit`` retains the
    convergence evidence for the fit they came from.
    """

    occasion: str
    target: str
    target_signature: str
    unit: str
    subjects: tuple[Any, ...]
    values: NDArray[np.float64] | Sequence[float]
    artifact_signature: str
    provenance: Mapping[str, Any] = field(default_factory=dict)
    draws: NDArray[np.float64] | Sequence[Sequence[float]] | None = None
    pooling: SubjectPooling = SubjectPooling.NONE
    posterior_audit: PosteriorAudit | None = None

    def __post_init__(self) -> None:
        for value, label in (
            (self.occasion, "occasion"),
            (self.target, "target"),
            (self.target_signature, "target_signature"),
            (self.unit, "unit"),
            (self.artifact_signature, "artifact_signature"),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"reliability {label} must be a non-empty string")
        subjects = tuple(
            _json_scalar(subject, label="reliability subject") for subject in self.subjects
        )
        if len(subjects) < 3:
            raise ValueError("reliability estimates require at least three subjects")
        if len(set(subjects)) != len(subjects):
            raise ValueError("reliability subjects must be unique")
        values = _protected(np.asarray(self.values, dtype=np.float64))
        if values.ndim != 1 or len(values) != len(subjects) or not np.all(np.isfinite(values)):
            raise ValueError("reliability values must be one finite estimate per subject")
        if not isinstance(self.provenance, Mapping):
            raise TypeError("reliability provenance must be a mapping")
        provenance: dict[str, Any] = {}
        for key, value in self.provenance.items():
            if not isinstance(key, str) or not key:
                raise ValueError("reliability provenance names must be non-empty strings")
            provenance[key] = _json_scalar(value, label=f"reliability provenance {key!r}")
        if self.draws is not None:
            draws = _protected(np.asarray(self.draws, dtype=np.float64))
            if draws.ndim != 2 or draws.shape[1] != len(subjects) or draws.shape[0] < 1:
                raise ValueError("reliability draws must be a sample-by-subject matrix")
            if not np.all(np.isfinite(draws)):
                raise ValueError("reliability draws must be finite")
            object.__setattr__(self, "draws", draws)
        audit = self.posterior_audit
        if audit is not None and not isinstance(audit, PosteriorAudit):
            raise TypeError("reliability posterior_audit must be a PosteriorAudit")
        object.__setattr__(self, "subjects", subjects)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "pooling", SubjectPooling(self.pooling))
        object.__setattr__(self, "provenance", MappingProxyType(provenance))

    @property
    def n_draws(self) -> int:
        """Number of retained posterior samples, or zero when only points are available."""

        return 0 if self.draws is None else int(np.asarray(self.draws).shape[0])


@dataclass(frozen=True, slots=True)
class ReliabilityEstimate:
    """One point statistic with paired-bootstrap uncertainty and failures."""

    statistic: ReliabilityStatistic
    estimate: float | None
    interval: tuple[float, float] | None
    bootstrap_values: NDArray[np.float64] | Sequence[float]
    invalid_bootstrap_repeats: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "statistic", ReliabilityStatistic(self.statistic))
        if self.estimate is not None and not np.isfinite(self.estimate):
            raise ValueError("reliability estimate must be finite or null")
        interval = _interval(self.interval)
        values = _protected(np.asarray(self.bootstrap_values, dtype=np.float64))
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise ValueError("reliability bootstrap values must be a finite vector")
        if (
            isinstance(self.invalid_bootstrap_repeats, bool)
            or not isinstance(self.invalid_bootstrap_repeats, int)
            or self.invalid_bootstrap_repeats < 0
        ):
            raise ValueError("invalid_bootstrap_repeats must be a non-negative integer")
        if self.estimate is None and interval is not None:
            raise ValueError("an undefined reliability estimate cannot have an interval")
        object.__setattr__(
            self,
            "estimate",
            None if self.estimate is None else float(self.estimate),
        )
        object.__setattr__(self, "interval", interval)
        object.__setattr__(self, "bootstrap_values", values)

    @property
    def effective_bootstrap_repeats(self) -> int:
        return len(self.bootstrap_values)

    @property
    def bootstrap_repeats(self) -> int:
        return self.effective_bootstrap_repeats + self.invalid_bootstrap_repeats


@dataclass(frozen=True, slots=True)
class ReliabilityIssue:
    """One stable estimability or finite-sample warning."""

    code: str
    message: str
    statistics: tuple[ReliabilityStatistic, ...] = ()

    def __post_init__(self) -> None:
        if not self.code or not self.message:
            raise ValueError("reliability issues require a code and message")
        statistics = tuple(ReliabilityStatistic(value) for value in self.statistics)
        if len(set(statistics)) != len(statistics):
            raise ValueError("reliability issue statistics must be unique")
        object.__setattr__(self, "statistics", statistics)


@dataclass(frozen=True, slots=True)
class TestRetestReliabilityReport:
    """Paired raw estimates, consistency, agreement, uncertainty, and warnings."""

    first: SubjectEstimates
    second: SubjectEstimates
    analysis_signature: str
    seed: int
    policy: ReliabilityPolicy
    estimates: tuple[ReliabilityEstimate, ...]
    issues: tuple[ReliabilityIssue, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.first, SubjectEstimates) or not isinstance(
            self.second, SubjectEstimates
        ):
            raise TypeError("test-retest report requires two SubjectEstimates objects")
        _validate_pair(self.first, self.second)
        if not isinstance(self.analysis_signature, str) or not self.analysis_signature:
            raise ValueError("reliability report requires an analysis signature")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("reliability seed must be a non-negative integer")
        if not isinstance(self.policy, ReliabilityPolicy):
            raise TypeError("reliability report policy must be ReliabilityPolicy")
        estimates = tuple(self.estimates)
        if not estimates or any(not isinstance(item, ReliabilityEstimate) for item in estimates):
            raise ValueError("reliability report requires ReliabilityEstimate records")
        expected = tuple(ReliabilityStatistic)
        if tuple(item.statistic for item in estimates) != expected:
            raise ValueError("reliability report must contain every statistic in canonical order")
        if any(item.bootstrap_repeats != self.policy.bootstrap_repeats for item in estimates):
            raise ValueError("reliability bootstrap counts must match the declared policy")
        issues = tuple(self.issues)
        if any(not isinstance(item, ReliabilityIssue) for item in issues):
            raise ValueError("reliability report issues must be ReliabilityIssue records")
        if len({item.code for item in issues}) != len(issues):
            raise ValueError("reliability issue codes must be unique")
        object.__setattr__(self, "estimates", estimates)
        object.__setattr__(self, "issues", issues)

    @property
    def subjects(self) -> tuple[Any, ...]:
        return self.first.subjects

    @property
    def n_subjects(self) -> int:
        return len(self.subjects)

    @property
    def first_values(self) -> NDArray[np.float64]:
        return self.first.values

    @property
    def second_values(self) -> NDArray[np.float64]:
        lookup = dict(zip(self.second.subjects, self.second.values, strict=True))
        return _protected(np.asarray([lookup[subject] for subject in self.subjects]))

    @property
    def pair_means(self) -> NDArray[np.float64]:
        return _protected((self.first_values + self.second_values) / 2.0)

    @property
    def differences(self) -> NDArray[np.float64]:
        """Return second-minus-first differences in first-occasion subject order."""

        return _protected(self.second_values - self.first_values)

    @property
    def issue_codes(self) -> tuple[str, ...]:
        return tuple(issue.code for issue in self.issues)

    @property
    def posterior_uncertainty_propagated(self) -> bool:
        """Whether each repetition also resampled a posterior draw per occasion."""

        return (
            self.first.draws is not None
            and self.second.draws is not None
            and self.policy.bootstrap_repeats > 0
        )

    def __getitem__(self, statistic: ReliabilityStatistic | str) -> ReliabilityEstimate:
        selected = ReliabilityStatistic(statistic)
        return next(item for item in self.estimates if item.statistic is selected)

    def to_dict(self, *, include_bootstrap: bool = True) -> dict[str, Any]:
        return {
            "analysis_signature": self.analysis_signature,
            "seed": self.seed,
            "target": self.first.target,
            "target_signature": self.first.target_signature,
            "unit": self.first.unit,
            "n_subjects": self.n_subjects,
            "posterior_uncertainty_propagated": self.posterior_uncertainty_propagated,
            "uncertainty_sources": (
                ["subject-resampling", "posterior-draws"]
                if self.posterior_uncertainty_propagated
                else ["subject-resampling"]
            ),
            "first": _subject_estimates_dict(self.first),
            "second": _subject_estimates_dict(self.second),
            "paired": [
                {
                    "subject": subject,
                    "first": float(first),
                    "second": float(second),
                    "mean": float(mean),
                    "difference": float(difference),
                }
                for subject, first, second, mean, difference in zip(
                    self.subjects,
                    self.first_values,
                    self.second_values,
                    self.pair_means,
                    self.differences,
                    strict=True,
                )
            ],
            "difference_direction": "second-minus-first",
            "policy": {
                "bootstrap_repeats": self.policy.bootstrap_repeats,
                "interval_probability": self.policy.interval_probability,
                "agreement_multiplier": self.policy.agreement_multiplier,
                "minimum_subjects_warning": self.policy.minimum_subjects_warning,
                "minimum_bootstrap_fraction": self.policy.minimum_bootstrap_fraction,
            },
            "estimates": [
                {
                    "statistic": item.statistic.value,
                    "estimate": item.estimate,
                    "interval": None if item.interval is None else list(item.interval),
                    "effective_bootstrap_repeats": item.effective_bootstrap_repeats,
                    "invalid_bootstrap_repeats": item.invalid_bootstrap_repeats,
                    **(
                        {"bootstrap_values": item.bootstrap_values.tolist()}
                        if include_bootstrap
                        else {}
                    ),
                }
                for item in self.estimates
            ],
            "issues": [
                {
                    "code": issue.code,
                    "message": issue.message,
                    "statistics": [value.value for value in issue.statistics],
                }
                for issue in self.issues
            ],
        }


def assess_test_retest_reliability(
    first: SubjectEstimates,
    second: SubjectEstimates,
    *,
    seed: int,
    analysis_signature: str,
    policy: ReliabilityPolicy | None = None,
) -> TestRetestReliabilityReport:
    """Estimate paired consistency and agreement without silent complete-case filtering.

    When both occasions carry posterior draws and ``bootstrap_repeats`` is positive, each
    repetition resamples subjects with replacement *and* takes one independently sampled
    posterior draw from each occasion, so the retained interval covers sampling and
    posterior uncertainty together and each reported statistic is a posterior mean rather
    than a statistic of posterior means. Draw indices are sampled independently for the two
    occasions because they come from separate fits whose draws are not jointly distributed;
    no coupling between them is assumed or manufactured.

    When either occasion carries only point estimates the original paired subject bootstrap
    runs unchanged, so maximum-likelihood inputs produce exactly the numbers they did
    before.
    """

    if not isinstance(first, SubjectEstimates) or not isinstance(second, SubjectEstimates):
        raise TypeError("first and second must be SubjectEstimates")
    _validate_pair(first, second)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("reliability seed must be a non-negative integer")
    if not isinstance(analysis_signature, str) or not analysis_signature:
        raise ValueError("analysis_signature must be a non-empty string")
    selected_policy = ReliabilityPolicy() if policy is None else policy
    if not isinstance(selected_policy, ReliabilityPolicy):
        raise TypeError("policy must be ReliabilityPolicy")

    order = [second.subjects.index(subject) for subject in first.subjects]
    first_values = np.asarray(first.values)
    second_values = np.asarray(second.values)[order]
    paired_draws = _paired_draws(first, second, order, selected_policy.bootstrap_repeats)
    propagate = paired_draws is not None
    n_subjects = len(first_values)
    multiplier = selected_policy.agreement_multiplier
    bootstraps = {statistic: [] for statistic in ReliabilityStatistic}
    invalid = {statistic: 0 for statistic in ReliabilityStatistic}
    posterior_draws = {statistic: [] for statistic in ReliabilityStatistic}
    generator = np.random.default_rng(seed)
    for _ in range(selected_policy.bootstrap_repeats):
        left, right = first_values, second_values
        if paired_draws is not None:
            first_draws, second_draws = paired_draws
            left = first_draws[generator.integers(0, len(first_draws))]
            right = second_draws[generator.integers(0, len(second_draws))]
            for statistic, value in _statistics(left, right, multiplier).items():
                if value is not None:
                    posterior_draws[statistic].append(value)
        index = generator.integers(0, n_subjects, size=n_subjects)
        for statistic, value in _statistics(left[index], right[index], multiplier).items():
            if value is None:
                invalid[statistic] += 1
            else:
                bootstraps[statistic].append(value)
    if propagate:
        point = {
            statistic: (float(np.mean(values)) if values else None)
            for statistic, values in posterior_draws.items()
        }
    else:
        point = _statistics(first_values, second_values, multiplier)

    alpha = (1.0 - selected_policy.interval_probability) / 2.0
    required = max(
        2,
        int(
            np.ceil(selected_policy.bootstrap_repeats * selected_policy.minimum_bootstrap_fraction)
        ),
    )
    estimates = []
    issues = []
    undefined = tuple(statistic for statistic, value in point.items() if value is None)
    if undefined:
        issues.append(
            ReliabilityIssue(
                code="reliability.undefined",
                message="one or more reliability statistics are undefined for these paired values",
                statistics=undefined,
            )
        )
    ineffective = []
    for statistic in ReliabilityStatistic:
        values = np.asarray(bootstraps[statistic], dtype=np.float64)
        interval = None
        if point[statistic] is not None and len(values) >= required:
            quantiles = np.quantile(values, (alpha, 1.0 - alpha))
            interval = (float(quantiles[0]), float(quantiles[1]))
        elif selected_policy.bootstrap_repeats > 0:
            ineffective.append(statistic)
        estimates.append(
            ReliabilityEstimate(
                statistic=statistic,
                estimate=point[statistic],
                interval=interval,
                bootstrap_values=values,
                invalid_bootstrap_repeats=invalid[statistic],
            )
        )
    if ineffective:
        issues.append(
            ReliabilityIssue(
                code="reliability.bootstrap-effective",
                message=(
                    "too few finite paired-bootstrap repetitions remain for one or more "
                    "requested intervals"
                ),
                statistics=tuple(ineffective),
            )
        )
    if len(first_values) < selected_policy.minimum_subjects_warning:
        issues.append(
            ReliabilityIssue(
                code="reliability.small-sample",
                message=(
                    f"the paired sample has {len(first_values)} subjects, below the declared "
                    f"warning threshold of {selected_policy.minimum_subjects_warning}"
                ),
            )
        )
    issues.extend(_source_issues(first, second, propagate))
    return TestRetestReliabilityReport(
        first=first,
        second=second,
        analysis_signature=analysis_signature,
        seed=seed,
        policy=selected_policy,
        estimates=tuple(estimates),
        issues=tuple(issues),
    )


def posterior_subject_estimates(
    result: PosteriorResult,
    variable_name: str,
    *,
    occasion: str,
    subject_dim: str = "subject",
    coordinate: Mapping[str, Any] | None = None,
    unit: str = "natural parameter",
    artifact_signature: str | None = None,
    pooling: SubjectPooling | str = SubjectPooling.PARTIAL,
    audit_policy: PosteriorAuditPolicy | None = None,
) -> SubjectEstimates:
    """Extract one scalar parameter target per subject, with its draws and its audit.

    ``values`` remains the per-subject posterior mean and is numerically unchanged. The
    full ``(sample, subject)`` draw matrix is retained alongside it so that
    :func:`assess_test_retest_reliability` can propagate posterior uncertainty rather than
    treating posterior means as observed data.

    ``pooling`` declares whether the source model pooled across subjects. A
    :class:`PosteriorResult` records model identity but nothing about its pooling
    structure, and sniffing the model name would be a guess, so the caller declares it.
    The default is :attr:`SubjectPooling.PARTIAL` -- the conservative side, because most
    posteriors that produce per-subject parameters come from hierarchical models and the
    cost of a false warning is far below the cost of an unflagged inflated reliability.
    Pass :attr:`SubjectPooling.NONE` for independently fitted subjects.

    The posterior's convergence is audited under ``audit_policy`` and the audit is retained
    on the returned record. A failing audit does not raise here: the estimates are still
    evidence, and :func:`assess_test_retest_reliability` gates on them with
    ``reliability.unconverged-posterior``.
    """

    if not isinstance(result, PosteriorResult):
        raise TypeError("result must be PosteriorResult")
    if audit_policy is not None and not isinstance(audit_policy, PosteriorAuditPolicy):
        raise TypeError("audit_policy must be a PosteriorAuditPolicy")
    selected_pooling = SubjectPooling(pooling)
    for value, label in (
        (variable_name, "variable_name"),
        (occasion, "occasion"),
        (subject_dim, "subject_dim"),
        (unit, "unit"),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label} must be a non-empty string")
    posterior = result["posterior"]
    if variable_name not in posterior.variable_names:
        raise ReliabilityError(f"posterior has no variable {variable_name!r}")
    variable = posterior[variable_name]
    if variable.dims[:2] != ("chain", "draw") or subject_dim not in variable.intrinsic_dims:
        raise ReliabilityError(
            "posterior reliability variables must lead with chain/draw and contain subject_dim"
        )
    selected = {} if coordinate is None else dict(coordinate)
    required = tuple(dim for dim in variable.intrinsic_dims if dim != subject_dim)
    if set(selected) != set(required):
        raise ReliabilityError(
            "coordinate must select every non-subject intrinsic posterior dimension exactly"
        )
    index: list[Any] = [slice(None), slice(None)]
    target_coordinate = []
    for dim in variable.intrinsic_dims:
        if dim == subject_dim:
            index.append(slice(None))
            continue
        label = _json_scalar(selected[dim], label=f"posterior coordinate {dim!r}")
        labels = tuple(
            _json_scalar(value, label=f"posterior coordinate {dim!r}")
            for value in variable.coords[dim]
        )
        matches = tuple(position for position, value in enumerate(labels) if value == label)
        if len(matches) != 1:
            raise ReliabilityError(f"coordinate {dim!r} has no unique label {label!r}")
        index.append(matches[0])
        target_coordinate.append((dim, label))
    values = np.asarray(variable.values, dtype=np.float64)[tuple(index)]
    if values.ndim != 3 or not np.all(np.isfinite(values)):
        raise ReliabilityError("selected posterior target must be finite and scalar per subject")
    subjects = tuple(
        _json_scalar(value, label="posterior subject") for value in variable.coords[subject_dim]
    )
    signature = artifact_signature or (
        f"posterior-result[v1;model={result.model_signature};"
        f"backend={result.inference_library}@{result.inference_library_version};"
        f"parameter-space={result.parameter_space_fingerprint or 'none'}]"
    )
    coordinate_signature = ",".join(f"{dim}={value!r}" for dim, value in target_coordinate)
    try:
        audit: PosteriorAudit | None = audit_posterior(result, policy=audit_policy)
    except ArviZUnavailableError:
        audit = None
        audit_status = "unavailable"
    else:
        audit_status = audit.status.value
    return SubjectEstimates(
        occasion=occasion,
        target=_target(variable_name, tuple(target_coordinate)),
        target_signature=(
            f"posterior-subject-mean[v1;variable={variable_name};"
            f"coordinate={coordinate_signature or 'none'}]"
        ),
        unit=unit,
        subjects=subjects,
        values=np.mean(values, axis=(0, 1)),
        artifact_signature=signature,
        provenance={
            "model_name": result.model_name,
            "model_signature": result.model_signature,
            "inference_library": result.inference_library,
            "inference_library_version": result.inference_library_version,
            "parameter_space_fingerprint": result.parameter_space_fingerprint,
            "pooling": selected_pooling.value,
            "posterior_audit_status": audit_status,
        },
        draws=values.reshape(-1, len(subjects)),
        pooling=selected_pooling,
        posterior_audit=audit,
    )


def _paired_draws(
    first: SubjectEstimates,
    second: SubjectEstimates,
    order: Sequence[int],
    bootstrap_repeats: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """Return both occasions' draws in first-occasion subject order, or nothing."""

    if first.draws is None or second.draws is None or bootstrap_repeats < 1:
        return None
    return np.asarray(first.draws), np.asarray(second.draws)[:, list(order)]


def _source_issues(
    first: SubjectEstimates,
    second: SubjectEstimates,
    propagated: bool,
) -> tuple[ReliabilityIssue, ...]:
    """Warn about shrinkage, refused convergence, and uncorrected posterior width."""

    issues: list[ReliabilityIssue] = []
    shrunken = tuple(
        estimates.occasion
        for estimates in (first, second)
        if estimates.pooling is SubjectPooling.PARTIAL
    )
    if shrunken:
        issues.append(
            ReliabilityIssue(
                code="reliability.shrunken-estimates",
                message=(
                    "per-subject estimates from a partial-pooling model are shrunk toward a "
                    f"common mean on occasion(s) {', '.join(shrunken)}; correlating shrunken "
                    "estimates inflates consistency and agreement, so treat these as upper "
                    "bounds. This is a warning, not a correction: undoing shrinkage needs the "
                    "hierarchical variance components, which subject estimates do not carry"
                ),
                statistics=tuple(ReliabilityStatistic(value) for value in _SHRINKABLE),
            )
        )
    failed = tuple(
        estimates.occasion
        for estimates in (first, second)
        if estimates.posterior_audit is not None
        and estimates.posterior_audit.status is PosteriorAuditStatus.FAIL
    )
    if failed:
        issues.append(
            ReliabilityIssue(
                code="reliability.unconverged-posterior",
                message=(
                    "the posterior behind occasion(s) "
                    f"{', '.join(failed)} failed its convergence audit, so the paired "
                    "estimates are not draws from the model being reported"
                ),
            )
        )
    unaudited = tuple(
        estimates.occasion
        for estimates in (first, second)
        if estimates.provenance.get("posterior_audit_status") == "unavailable"
    )
    if unaudited:
        issues.append(
            ReliabilityIssue(
                code="reliability.posterior-audit-unavailable",
                message=(
                    "convergence could not be audited for occasion(s) "
                    f"{', '.join(unaudited)} because the ArviZ diagnostics backend is "
                    "unavailable; install `behavio[probabilistic]`"
                ),
            )
        )
    carriers = tuple(item for item in (first, second) if item.draws is not None)
    if carriers and not propagated:
        issues.append(
            ReliabilityIssue(
                code="reliability.draws-not-propagated",
                message=(
                    "posterior draws were available but not propagated, so the reported "
                    "intervals cover subject sampling only and are narrower than the "
                    "evidence supports"
                ),
            )
        )
    return tuple(issues)


def _statistics(
    first: NDArray[np.float64],
    second: NDArray[np.float64],
    agreement_multiplier: float,
) -> dict[ReliabilityStatistic, float | None]:
    difference = second - first
    mean_difference = float(np.mean(difference))
    sd_difference = float(np.std(difference, ddof=1))
    pearson = _correlation(first, second)
    spearman = _correlation(_average_ranks(first), _average_ranks(second))
    icc_consistency, icc_agreement = _icc(first, second)
    return {
        ReliabilityStatistic.PEARSON: pearson,
        ReliabilityStatistic.SPEARMAN: spearman,
        ReliabilityStatistic.ICC_CONSISTENCY: icc_consistency,
        ReliabilityStatistic.ICC_ABSOLUTE_AGREEMENT: icc_agreement,
        ReliabilityStatistic.MEAN_DIFFERENCE: mean_difference,
        ReliabilityStatistic.SD_DIFFERENCE: sd_difference,
        ReliabilityStatistic.WITHIN_SUBJECT_SD: sd_difference / np.sqrt(2.0),
        ReliabilityStatistic.LOWER_LIMIT_OF_AGREEMENT: (
            mean_difference - agreement_multiplier * sd_difference
        ),
        ReliabilityStatistic.UPPER_LIMIT_OF_AGREEMENT: (
            mean_difference + agreement_multiplier * sd_difference
        ),
        ReliabilityStatistic.MEAN_ABSOLUTE_ERROR: float(np.mean(np.abs(difference))),
        ReliabilityStatistic.ROOT_MEAN_SQUARED_ERROR: float(
            np.sqrt(np.mean(np.square(difference)))
        ),
    }


def _correlation(first: NDArray[np.float64], second: NDArray[np.float64]) -> float | None:
    centered_first = first - np.mean(first)
    centered_second = second - np.mean(second)
    denominator = float(
        np.sqrt(np.sum(np.square(centered_first)) * np.sum(np.square(centered_second)))
    )
    if denominator <= 0 or not np.isfinite(denominator):
        return None
    return float(np.sum(centered_first * centered_second) / denominator)


def _average_ranks(values: NDArray[np.float64]) -> NDArray[np.float64]:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return ranks


def _icc(
    first: NDArray[np.float64],
    second: NDArray[np.float64],
) -> tuple[float | None, float | None]:
    values = np.column_stack((first, second))
    n_subjects, n_occasions = values.shape
    grand_mean = float(np.mean(values))
    subject_means = np.mean(values, axis=1)
    occasion_means = np.mean(values, axis=0)
    subject_mean_square = float(
        n_occasions * np.sum(np.square(subject_means - grand_mean)) / (n_subjects - 1)
    )
    occasion_mean_square = float(
        n_subjects * np.sum(np.square(occasion_means - grand_mean)) / (n_occasions - 1)
    )
    residuals = values - subject_means[:, None] - occasion_means[None, :] + grand_mean
    error_mean_square = float(np.sum(np.square(residuals)) / ((n_subjects - 1) * (n_occasions - 1)))
    numerator = subject_mean_square - error_mean_square
    consistency_denominator = subject_mean_square + (n_occasions - 1) * error_mean_square
    agreement_denominator = (
        consistency_denominator
        + n_occasions * (occasion_mean_square - error_mean_square) / n_subjects
    )
    consistency = (
        None
        if consistency_denominator <= 0 or not np.isfinite(consistency_denominator)
        else float(numerator / consistency_denominator)
    )
    agreement = (
        None
        if agreement_denominator <= 0 or not np.isfinite(agreement_denominator)
        else float(numerator / agreement_denominator)
    )
    return consistency, agreement


def _validate_pair(first: SubjectEstimates, second: SubjectEstimates) -> None:
    if first.occasion == second.occasion:
        raise ReliabilityError("test and retest occasions must have distinct names")
    if (
        first.target != second.target
        or first.target_signature != second.target_signature
        or first.unit != second.unit
    ):
        raise ReliabilityError("test and retest must measure the same target and unit")
    if set(first.subjects) != set(second.subjects):
        missing_second = tuple(
            subject for subject in first.subjects if subject not in second.subjects
        )
        missing_first = tuple(
            subject for subject in second.subjects if subject not in first.subjects
        )
        raise ReliabilityError(
            "test and retest must contain exactly the same subjects; "
            f"missing from retest={missing_second!r}, missing from test={missing_first!r}"
        )


def _subject_estimates_dict(estimates: SubjectEstimates) -> dict[str, Any]:
    return {
        "occasion": estimates.occasion,
        "artifact_signature": estimates.artifact_signature,
        "provenance": dict(estimates.provenance),
        "pooling": estimates.pooling.value,
        "n_draws": estimates.n_draws,
        "posterior_audit": (
            None if estimates.posterior_audit is None else estimates.posterior_audit.to_dict()
        ),
    }


def _target(name: str, coordinate: tuple[tuple[str, Any], ...]) -> str:
    if not coordinate:
        return name
    labels = ",".join(f"{dim}={value!r}" for dim, value in coordinate)
    return f"{name}[{labels}]"


def _interval(values: tuple[float, float] | None) -> tuple[float, float] | None:
    if values is None:
        return None
    if len(values) != 2 or not np.all(np.isfinite(values)):
        raise ValueError("reliability interval must contain two finite values")
    interval = (float(values[0]), float(values[1]))
    if interval[0] > interval[1]:
        raise ValueError("reliability interval must be ordered")
    return interval


def _json_scalar(value: Any, *, label: str) -> Any:
    scalar = value.item() if isinstance(value, np.generic) else value
    if scalar is None or isinstance(scalar, str | bool | int):
        return scalar
    if isinstance(scalar, float) and np.isfinite(scalar):
        return scalar
    raise ValueError(f"{label} must be a finite JSON scalar")


def _protected(values: NDArray[np.float64]) -> NDArray[np.float64]:
    protected = np.array(values, dtype=np.float64, copy=True)
    protected.setflags(write=False)
    return protected
