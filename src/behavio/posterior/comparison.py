"""Paired ELPD comparison of two or more scored posteriors.

Comparing models on ELPD is a paired calculation, not a comparison of two summaries. The
standard error of a total ELPD is large because it is dominated by how hard the observations
are; the standard error of the *difference* is small because that difficulty cancels
observation by observation. Differencing the totals and combining their marginal standard
errors overstates the uncertainty and is the mistake this module exists to remove:

.. math::

    \\widehat{\\mathrm{elpd}}_{A} - \\widehat{\\mathrm{elpd}}_{B}
        = \\sum_i \\left(\\widehat{\\mathrm{elpd}}_{A,i} - \\widehat{\\mathrm{elpd}}_{B,i}\\right),
    \\qquad
    \\mathrm{se}_{\\mathrm{diff}} = \\sqrt{n \\operatorname{Var}_i(
        \\widehat{\\mathrm{elpd}}_{A,i} - \\widehat{\\mathrm{elpd}}_{B,i})}.

The pairing is only meaningful if index :math:`i` means the same observation in both models, so
:func:`compare_posterior_models` aligns labelled coordinates and refuses to broadcast. It also
refuses to compare results computed under different blocking, because leave-one-subject-out and
leave-one-trial-out ELPD are different quantities that happen to share units.

Following :mod:`behavio.protocol.runner`, no winner is declared when a paired interval
fails to exclude zero, and a model whose posterior failed its convergence audit is never ranked.

Simultaneous inference
----------------------
``K`` eligible models produce ``K(K-1)/2`` paired differences read against one interval, so
this ranking has exactly the multiplicity problem :mod:`behavio.compare.models` has, and it is
sized and corrected by the same record:
:class:`~behavio._internal.scoring.ComparisonFamily`, reported on
:attr:`PosteriorModelComparison.family`, with the adjustment declared through
``multiplicity`` and Benjamini-Hochberg by default.

What it deliberately does **not** share is the machinery underneath. A paired score
comparison inverts a percentile bootstrap over resampled aggregation units; a paired ELPD
comparison has no bootstrap, only :math:`\\mathrm{se}_{\\mathrm{diff}}` and the normal
approximation the reporting scale of two standard errors already assumes. So this module
keeps :class:`PairedELPDDifference` and its own standard error, and converts *that* into
the per-contrast probability the shared step-up consumes:
:math:`p = 2\\Phi(-|\\Delta| / \\mathrm{se})`, with the interval level
:math:`1 - 2\\Phi(-\\text{interval\\_scale})` implied by the declared multiplier. Because
both the interval and the probability come from the same normal, ``excludes_zero`` and
``two_sided_probability < 1 - interval_level`` agree exactly -- unlike the bootstrap case,
where a finite resample can pin the probability above a threshold its interval has already
crossed.

Correction can only remove a separation, so a comparison that named a best model on
uncorrected intervals may now decline to. That is the intended direction: this package
refuses to name a winner on insufficient evidence, and a family of ten contrasts read at
one interval level *is* insufficient evidence for the one that happens to clear it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from behavio._internal.scoring import (
    ComparisonFamily,
    ComparisonMultiplicity,
    adjust_probabilities,
    excess_probability,
)
from behavio.contracts.audit import AuditSeverity
from behavio.posterior.diagnostics import PosteriorAuditPolicy, PosteriorAuditStatus
from behavio.posterior.loo import MIN_RELIABLE_BLOCKS, PSISLOOResult, psis_loo
from behavio.posterior.result import PosteriorError, PosteriorResult

DEFAULT_INTERVAL_SCALE: Final = 2.0
"""Standard-error multiplier for the paired difference interval.

Two standard errors is the conventional ELPD-difference reporting scale (Vehtari, Gelman and
Gabry 2017). It is a normal approximation over the pointwise unit, which is why
``comparison.few-observations`` fires when there are too few units to support it.
"""

#: The smallest step below one a float can represent. A per-contrast rate is clamped into
#: the open unit interval with it, because ``2 * norm.sf(scale)`` underflows to exactly zero
#: for a large enough multiplier and a family record cannot describe a rate of zero or one.
_SMALLEST_STEP: Final = float(1.0 - np.nextafter(1.0, 0.0))


class ModelComparisonStatus(StrEnum):
    """Whether the paired ELPD evidence supports one model."""

    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"
    NO_ELIGIBLE_MODEL = "no-eligible-model"


@dataclass(frozen=True, slots=True)
class ModelComparisonIssue:
    """One stable machine-readable warning from a posterior model comparison."""

    code: str
    message: str
    targets: tuple[str, ...] = ()
    severity: AuditSeverity = AuditSeverity.WARNING

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code:
            raise ValueError("comparison issue code must be a non-empty string")
        if not isinstance(self.message, str) or not self.message:
            raise ValueError("comparison issue message must be a non-empty string")
        targets = tuple(self.targets)
        if any(not isinstance(target, str) or not target for target in targets):
            raise ValueError("comparison issue targets must be non-empty strings")
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "severity", AuditSeverity(self.severity))

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "targets": list(self.targets),
            "severity": self.severity.value,
        }


@dataclass(frozen=True, slots=True)
class ScoredModel:
    """One model's ELPD summary and the reasons it may not be usable evidence."""

    name: str
    model_signature: str
    log_likelihood_name: str
    elpd_loo: float
    se: float
    p_loo: float
    n_data_points: int
    max_pareto_k: float
    good_k: float
    status: PosteriorAuditStatus
    issue_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("scored model name must be a non-empty string")
        object.__setattr__(self, "status", PosteriorAuditStatus(self.status))
        object.__setattr__(self, "issue_codes", tuple(self.issue_codes))

    @property
    def eligible(self) -> bool:
        """Whether this model may be ranked at all.

        A ``FAIL`` status means either the posterior did not converge or the importance
        sampling produced non-finite ELPD. Neither is a close call to be resolved by the size
        of a difference.
        """

        return self.status is not PosteriorAuditStatus.FAIL

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model_signature": self.model_signature,
            "log_likelihood_name": self.log_likelihood_name,
            "elpd_loo": _json_float(self.elpd_loo),
            "se": _json_float(self.se),
            "p_loo": _json_float(self.p_loo),
            "n_data_points": self.n_data_points,
            "max_pareto_k": _json_float(self.max_pareto_k),
            "good_k": self.good_k,
            "status": self.status.value,
            "eligible": self.eligible,
            "issue_codes": list(self.issue_codes),
        }


@dataclass(frozen=True, slots=True)
class PairedELPDDifference:
    """A paired ELPD difference and its own standard error, on aligned observations.

    ``elpd_difference``, ``se``, ``lower`` and ``upper`` are unadjusted and mean exactly
    what they always meant. ``two_sided_probability`` is the normal-approximation
    probability of a difference at least this extreme under no difference, computed from
    this contrast's own standard error rather than from a bootstrap.
    ``adjusted_probability`` is that probability after the family's declared multiplicity
    adjustment, and ``decisive`` is the only member the ranking reads.

    All three are ``None``/``False`` for a contrast that was not part of the tested family
    -- that is, one involving a model whose posterior or importance sampling failed. Such a
    model is never ranked, so its contrasts are reported as evidence but are not tests, and
    inflating the family with them would correct the eligible contrasts against readings
    that were never going to decide anything.
    """

    left_model: str
    right_model: str
    elpd_difference: float
    se: float
    lower: float
    upper: float
    interval_scale: float
    n_data_points: int
    two_sided_probability: float | None = None
    adjusted_probability: float | None = None
    decisive: bool = False

    def __post_init__(self) -> None:
        if not self.left_model or not self.right_model or self.left_model == self.right_model:
            raise ValueError("a paired difference requires two distinct named models")
        if self.se < 0:
            raise ValueError("a paired difference standard error must be non-negative")
        if self.lower > self.upper:
            raise ValueError("paired difference lower bound must not exceed its upper bound")
        for value, label in (
            (self.two_sided_probability, "two_sided_probability"),
            (self.adjusted_probability, "adjusted_probability"),
        ):
            if value is not None and (not np.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f"{label} must be null or lie between zero and one")
        if not isinstance(self.decisive, bool):
            raise TypeError("decisive must be boolean")
        if self.decisive and self.adjusted_probability is None:
            raise ValueError(
                "a decisive contrast must record the adjusted probability that made it"
            )

    @property
    def excludes_zero(self) -> bool:
        """Whether the interval separates the two models at the declared scale."""

        return self.lower > 0.0 or self.upper < 0.0

    @property
    def favours(self) -> str | None:
        """The named model this contrast decisively favours, or ``None``.

        ELPD is higher-is-better, so a positive difference favours the left model. A
        contrast that did not survive the family adjustment favours nobody.
        """

        if not self.decisive:
            return None
        return self.left_model if self.elpd_difference > 0.0 else self.right_model

    def to_dict(self) -> dict[str, Any]:
        return {
            "left_model": self.left_model,
            "right_model": self.right_model,
            "direction": (
                f"left elpd minus right elpd; positive favors {self.left_model} "
                "because ELPD is higher-is-better"
            ),
            "elpd_difference": _json_float(self.elpd_difference),
            "se": _json_float(self.se),
            "lower": _json_float(self.lower),
            "upper": _json_float(self.upper),
            "interval_scale": self.interval_scale,
            "excludes_zero": self.excludes_zero,
            "two_sided_probability": self.two_sided_probability,
            "adjusted_probability": self.adjusted_probability,
            "decisive": self.decisive,
            "n_data_points": self.n_data_points,
        }


@dataclass(frozen=True, slots=True)
class PosteriorModelComparison:
    """Complete retained evidence from one paired ELPD comparison."""

    block: str | None
    estimand: str
    dims: tuple[str, ...]
    coords: Mapping[str, Sequence[Any] | NDArray[Any]]
    n_data_points: int
    interval_scale: float
    models: tuple[ScoredModel, ...]
    differences: tuple[PairedELPDDifference, ...]
    status: ModelComparisonStatus
    best_model: str | None
    reason: str
    family: ComparisonFamily
    issues: tuple[ModelComparisonIssue, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", ModelComparisonStatus(self.status))
        if not isinstance(self.family, ComparisonFamily):
            raise TypeError("comparison family must be a ComparisonFamily")
        if (self.status is ModelComparisonStatus.RESOLVED) != (self.best_model is not None):
            raise ValueError("a comparison names a best model exactly when it is resolved")
        dims = tuple(self.dims)
        if set(self.coords) != set(dims):
            raise ValueError("comparison coordinates must name exactly its dimensions")
        coords = {dim: _protected(np.asarray(self.coords[dim])) for dim in dims}
        models = tuple(self.models)
        if len(models) < 2:
            raise ValueError("a comparison retains at least two scored models")
        names = tuple(model.name for model in models)
        if len(set(names)) != len(names):
            raise ValueError("compared model names must be unique")
        codes = tuple(issue.code for issue in self.issues)
        if len(set(codes)) != len(codes):
            raise ValueError("comparison issue codes must be unique within one report")
        object.__setattr__(self, "dims", dims)
        object.__setattr__(self, "coords", MappingProxyType(coords))
        object.__setattr__(self, "models", models)
        object.__setattr__(self, "differences", tuple(self.differences))
        object.__setattr__(self, "issues", tuple(self.issues))

    @property
    def model_names(self) -> tuple[str, ...]:
        """Compared models ordered best ELPD first, including ineligible ones."""

        return tuple(model.name for model in self.models)

    @property
    def eligible_models(self) -> tuple[str, ...]:
        """Models whose posterior and importance sampling permit ranking."""

        return tuple(model.name for model in self.models if model.eligible)

    @property
    def decisive_differences(self) -> tuple[PairedELPDDifference, ...]:
        """Contrasts that survive the declared multiplicity adjustment."""

        return tuple(item for item in self.differences if item.decisive)

    @property
    def issue_codes(self) -> tuple[str, ...]:
        return tuple(issue.code for issue in self.issues)

    def difference(self, left: str, right: str) -> PairedELPDDifference:
        """Return the retained paired difference for an ordered pair of models."""

        for item in self.differences:
            if item.left_model == left and item.right_model == right:
                return item
            if item.left_model == right and item.right_model == left:
                # Reversing the direction negates the difference and reflects the interval;
                # the probabilities and the verdict are direction-free and carry across
                # unchanged.
                return PairedELPDDifference(
                    left_model=left,
                    right_model=right,
                    elpd_difference=-item.elpd_difference,
                    se=item.se,
                    lower=-item.upper,
                    upper=-item.lower,
                    interval_scale=item.interval_scale,
                    n_data_points=item.n_data_points,
                    two_sided_probability=item.two_sided_probability,
                    adjusted_probability=item.adjusted_probability,
                    decisive=item.decisive,
                )
        raise KeyError((left, right))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible record of the comparison and its refusals."""

        return {
            "block": self.block,
            "estimand": self.estimand,
            "dims": list(self.dims),
            "coords": {
                dim: [_json_scalar(value) for value in self.coords[dim]] for dim in self.dims
            },
            "n_data_points": self.n_data_points,
            "interval_scale": self.interval_scale,
            "status": self.status.value,
            "best_model": self.best_model,
            "reason": self.reason,
            "simultaneous_family": self.family.to_dict(),
            "models": [model.to_dict() for model in self.models],
            "eligible_models": list(self.eligible_models),
            "differences": [item.to_dict() for item in self.differences],
            "issues": [issue.to_dict() for issue in self.issues],
        }


def compare_posterior_models(
    results: Mapping[str, PSISLOOResult | PosteriorResult],
    *,
    log_likelihood_name: str | None = None,
    block: str | None = None,
    block_values: Sequence[Any] | NDArray[Any] | None = None,
    policy: PosteriorAuditPolicy | None = None,
    interval_scale: float = DEFAULT_INTERVAL_SCALE,
    multiplicity: ComparisonMultiplicity = ComparisonMultiplicity.BENJAMINI_HOCHBERG,
) -> PosteriorModelComparison:
    """Compare two or more models on paired ELPD over identically labelled observations.

    Args:
        results: Mapping of model name to an already-computed :class:`PSISLOOResult` or to a
            :class:`~behavio.posterior.PosteriorResult`. Posteriors are scored here with
            :func:`~behavio.posterior.loo.psis_loo` under the shared ``log_likelihood_name``,
            ``block``, ``block_values``, and ``policy``, so every model is scored the same way.
        log_likelihood_name: Passed to :func:`~behavio.posterior.loo.psis_loo` for posteriors.
        block: Passed to :func:`~behavio.posterior.loo.psis_loo` for posteriors. Pre-computed
            :class:`PSISLOOResult` inputs must already agree with it.
        block_values: Passed to :func:`~behavio.posterior.loo.psis_loo` for posteriors.
        policy: Convergence thresholds applied to every posterior scored here.
        interval_scale: Standard-error multiplier for the reported difference interval. It
            also fixes the per-contrast error rate the family is corrected at, which is the
            normal tail mass outside it: ``2 * norm.sf(interval_scale)``, or about 0.045 at
            the default of two standard errors.
        multiplicity: Adjustment applied across the ``K(K-1)/2`` contrasts among eligible
            models, defaulting to Benjamini-Hochberg so a Bayesian comparison is corrected
            on the same terms as :func:`behavio.compare.models.compare_models`.
            :attr:`ComparisonMultiplicity.NONE` restores the uncorrected per-contrast
            reading and will name a best model strictly more often.

    Returns:
        A :class:`PosteriorModelComparison` carrying every model's status, every paired
        difference, the simultaneous family they were read against, and a ranking that
        stays unresolved when the evidence is not separating.

    Raises:
        PosteriorError: If fewer than two models are supplied, if any two results were computed
            under different blocking, or if their pointwise coordinates are not identical in
            the same order. A mismatch is never resolved by broadcasting, reindexing, or
            truncation, because every such repair silently changes the estimand.
    """

    if not isinstance(results, Mapping):
        raise TypeError("results must be a mapping of model name to scored evidence")
    if not np.isfinite(interval_scale) or interval_scale <= 0:
        raise ValueError("interval_scale must be finite and positive")
    scored = _scored_inputs(results, log_likelihood_name, block, block_values, policy)
    if len(scored) < 2:
        raise PosteriorError("comparing predictive performance requires at least two models")
    reference_name, reference = next(iter(scored.items()))
    for name, item in scored.items():
        _require_alignment(reference_name, reference, name, item)

    models = tuple(
        sorted(
            (_scored_model(name, item) for name, item in scored.items()),
            key=lambda model: (-model.elpd_loo, model.name),
        )
    )
    adjustment = ComparisonMultiplicity(multiplicity)
    unadjusted = _paired_differences(scored, models, interval_scale)
    differences, family = _corrected_family(unadjusted, models, interval_scale, adjustment)
    issues = _comparison_issues(scored, models, reference)
    status, best, reason = _rank(models, differences, family)
    return PosteriorModelComparison(
        block=reference.block,
        estimand=reference.estimand,
        dims=reference.dims,
        coords={dim: reference.coords[dim] for dim in reference.dims},
        n_data_points=reference.n_data_points,
        interval_scale=float(interval_scale),
        models=models,
        differences=differences,
        status=status,
        best_model=best,
        reason=reason,
        family=family,
        issues=issues,
    )


def _scored_inputs(
    results: Mapping[str, PSISLOOResult | PosteriorResult],
    log_likelihood_name: str | None,
    block: str | None,
    block_values: Sequence[Any] | NDArray[Any] | None,
    policy: PosteriorAuditPolicy | None,
) -> dict[str, PSISLOOResult]:
    scored: dict[str, PSISLOOResult] = {}
    for name, value in results.items():
        if not isinstance(name, str) or not name:
            raise ValueError("compared model names must be non-empty strings")
        if isinstance(value, PSISLOOResult):
            scored[name] = value
        elif isinstance(value, PosteriorResult):
            scored[name] = psis_loo(
                value,
                log_likelihood_name=log_likelihood_name,
                block=block,
                block_values=block_values,
                policy=policy,
            )
        else:
            raise TypeError(f"model {name!r} is neither a PSISLOOResult nor a PosteriorResult")
    return scored


def _require_alignment(
    reference_name: str,
    reference: PSISLOOResult,
    name: str,
    candidate: PSISLOOResult,
) -> None:
    """Refuse anything but exact agreement on what was scored, and in which order."""

    if candidate.block != reference.block:
        raise PosteriorError(
            f"model {name!r} computed {candidate.estimand} ELPD but model {reference_name!r} "
            f"computed {reference.estimand} ELPD; these are different estimands and cannot be "
            "differenced"
        )
    if candidate.dims != reference.dims:
        raise PosteriorError(
            f"model {name!r} scores dimensions {list(candidate.dims)} but model "
            f"{reference_name!r} scores {list(reference.dims)}"
        )
    if candidate.n_data_points != reference.n_data_points:
        raise PosteriorError(
            f"model {name!r} scores {candidate.n_data_points} observations but model "
            f"{reference_name!r} scores {reference.n_data_points}"
        )
    for dim in reference.dims:
        if not np.array_equal(candidate.coords[dim], reference.coords[dim]):
            raise PosteriorError(
                f"model {name!r} and model {reference_name!r} do not share identical "
                f"{dim!r} coordinates in the same order"
            )


def _scored_model(name: str, item: PSISLOOResult) -> ScoredModel:
    pareto = np.asarray(item.pareto_k, dtype=np.float64)
    finite = pareto[np.isfinite(pareto)]
    return ScoredModel(
        name=name,
        model_signature=item.model_signature,
        log_likelihood_name=item.log_likelihood_name,
        elpd_loo=item.elpd_loo,
        se=item.se,
        p_loo=item.p_loo,
        n_data_points=item.n_data_points,
        max_pareto_k=float(finite.max()) if finite.size else float("nan"),
        good_k=item.good_k,
        status=item.status,
        issue_codes=item.issue_codes,
    )


def _paired_differences(
    scored: Mapping[str, PSISLOOResult],
    models: tuple[ScoredModel, ...],
    interval_scale: float,
) -> tuple[PairedELPDDifference, ...]:
    """Difference every ordered pair once, better-ranked model on the left."""

    differences: list[PairedELPDDifference] = []
    for position, left in enumerate(models):
        for right in models[position + 1 :]:
            differences.append(
                _difference(
                    left.name,
                    scored[left.name],
                    right.name,
                    scored[right.name],
                    interval_scale,
                )
            )
    return tuple(differences)


def _difference(
    left_name: str,
    left: PSISLOOResult,
    right_name: str,
    right: PSISLOOResult,
    interval_scale: float,
) -> PairedELPDDifference:
    """Compute the difference and its standard error from the paired pointwise values.

    ``se`` uses ``sqrt(n * var(pointwise difference))`` with the population variance, matching
    ArviZ's ``compare`` and the reference implementation. It is emphatically not
    ``left.se - right.se`` or ``sqrt(left.se**2 + right.se**2)``: the pointwise values are
    positively correlated across models, so both of those inflate the interval.
    """

    pointwise = np.asarray(left.pointwise_elpd, dtype=np.float64).reshape(-1) - np.asarray(
        right.pointwise_elpd, dtype=np.float64
    ).reshape(-1)
    difference = float(np.sum(pointwise))
    se = float(np.sqrt(pointwise.size * np.var(pointwise)))
    half_width = interval_scale * se
    return PairedELPDDifference(
        left_model=left_name,
        right_model=right_name,
        elpd_difference=difference,
        se=se,
        lower=difference - half_width,
        upper=difference + half_width,
        interval_scale=float(interval_scale),
        n_data_points=int(pointwise.size),
    )


def _per_contrast_rate(interval_scale: float) -> float:
    """Normal tail mass outside ``+/- interval_scale`` standard errors.

    This is the per-contrast error rate the declared reporting scale already implies. It is
    not a second threshold: at the conventional two standard errors it is 0.0455, which is
    what "two SE" has always meant.
    """

    rate = float(2.0 * norm.sf(interval_scale))
    return min(max(rate, _SMALLEST_STEP), 1.0 - _SMALLEST_STEP)


def _two_sided_probability(difference: PairedELPDDifference) -> float:
    """Normal-approximation probability of a difference at least this extreme.

    The interval and this probability come from the same normal with the same standard
    error, so ``excludes_zero`` holds exactly when this falls below the per-contrast rate.
    A zero standard error is not a division to guard against silently: identical pointwise
    ELPD in both models is complete agreement, and anything else with no spread at all is
    a difference the data resolve exactly.
    """

    if difference.se == 0.0:
        return 0.0 if difference.elpd_difference != 0.0 else 1.0
    return float(2.0 * norm.sf(abs(difference.elpd_difference) / difference.se))


def _corrected_family(
    differences: tuple[PairedELPDDifference, ...],
    models: tuple[ScoredModel, ...],
    interval_scale: float,
    multiplicity: ComparisonMultiplicity,
) -> tuple[tuple[PairedELPDDifference, ...], ComparisonFamily]:
    """Size the simultaneous family and mark the contrasts that survive its adjustment.

    The family is the contrasts among *eligible* models, which is the same rule
    :func:`behavio.compare.models.paired_comparisons` follows when the runner hands it only
    eligible candidates. A contrast touching a model that failed convergence or importance
    sampling is retained and reported, but its probabilities stay ``None``: it was never a
    test, because the model on one side of it cannot be ranked at whatever the answer.
    """

    eligible = {model.name for model in models if model.eligible}
    tested = tuple(
        item for item in differences if item.left_model in eligible and item.right_model in eligible
    )
    rate = _per_contrast_rate(interval_scale)
    interval_level = 1.0 - rate
    probabilities = np.asarray([_two_sided_probability(item) for item in tested], dtype=np.float64)
    separated = np.asarray([item.excludes_zero for item in tested], dtype=bool)
    threshold, adjusted = adjust_probabilities(
        probabilities,
        multiplicity=multiplicity,
        error_rate=rate,
        unadjusted_threshold=rate,
    )
    # As in `behavio.compare.models`, a contrast must both survive the adjustment and have an
    # unadjusted interval that excludes zero, so correction can only take a separation
    # away. Here the two conditions cannot disagree -- interval and probability are the
    # same normal statement -- but the conjunction is kept so the two comparison paths
    # define `decisive` identically.
    corrected = len(tested) > 1 and multiplicity is not ComparisonMultiplicity.NONE
    decisive = separated & (adjusted <= rate) if corrected else separated
    by_pair = {
        (item.left_model, item.right_model): (probability, adjustment_value, bool(is_decisive))
        for item, probability, adjustment_value, is_decisive in zip(
            tested, probabilities, adjusted, decisive, strict=True
        )
    }
    n_separated = int(np.count_nonzero(separated))
    family = ComparisonFamily(
        n_candidates=len(eligible),
        n_comparisons=len(tested),
        interval_level=interval_level,
        multiplicity=multiplicity,
        family_error_rate=rate,
        n_separated=n_separated,
        expected_separated=float(len(tested) * rate),
        excess_probability=excess_probability(n_separated, len(tested), rate),
        adjusted_threshold=float(threshold),
        n_decisive=int(np.count_nonzero(decisive)),
    )
    return (
        tuple(
            _with_family_verdict(item, by_pair.get((item.left_model, item.right_model)))
            for item in differences
        ),
        family,
    )


def _with_family_verdict(
    difference: PairedELPDDifference,
    verdict: tuple[float, float, bool] | None,
) -> PairedELPDDifference:
    """Attach a family reading to one contrast, or leave it untested."""

    if verdict is None:
        return difference
    probability, adjusted, decisive = verdict
    return replace(
        difference,
        two_sided_probability=float(probability),
        adjusted_probability=float(adjusted),
        decisive=decisive,
    )


def _comparison_issues(
    scored: Mapping[str, PSISLOOResult],
    models: tuple[ScoredModel, ...],
    reference: PSISLOOResult,
) -> tuple[ModelComparisonIssue, ...]:
    issues: list[ModelComparisonIssue] = []
    failed = tuple(model.name for model in models if not model.eligible)
    if failed:
        issues.append(
            ModelComparisonIssue(
                code="comparison.posterior-fail",
                message=(
                    "one or more models failed convergence or importance-sampling checks and "
                    "were excluded from ranking"
                ),
                targets=failed,
                severity=AuditSeverity.ERROR,
            )
        )
    warned = tuple(model.name for model in models if model.status is PosteriorAuditStatus.WARNING)
    if warned:
        issues.append(
            ModelComparisonIssue(
                code="comparison.posterior-warning",
                message="one or more models carry unresolved diagnostic warnings",
                targets=warned,
            )
        )
    unreliable = tuple(model.name for model in models if "psis.high-pareto-k" in model.issue_codes)
    if unreliable:
        issues.append(
            ModelComparisonIssue(
                code="comparison.high-pareto-k",
                message=(
                    "one or more models rely on unstable importance weights, so their "
                    "pointwise contributions to the difference are approximate"
                ),
                targets=unreliable,
            )
        )
    names = {item.log_likelihood_name for item in scored.values()}
    if len(names) > 1:
        issues.append(
            ModelComparisonIssue(
                code="comparison.likelihood-name-mismatch",
                message=(
                    "compared models score differently named likelihood variables; confirm "
                    "they describe the same observed quantity"
                ),
                targets=tuple(sorted(names)),
            )
        )
    if reference.n_data_points < MIN_RELIABLE_BLOCKS:
        issues.append(
            ModelComparisonIssue(
                code="comparison.few-observations",
                message=(
                    f"the paired difference rests on {reference.n_data_points} units, below "
                    f"the {MIN_RELIABLE_BLOCKS} needed for the normal approximation behind se"
                ),
            )
        )
    return tuple(issues)


def _rank(
    models: tuple[ScoredModel, ...],
    differences: tuple[PairedELPDDifference, ...],
    family: ComparisonFamily,
) -> tuple[ModelComparisonStatus, str | None, str]:
    """Name a best model only when the corrected paired evidence separates it.

    The rule reads :attr:`PairedELPDDifference.decisive`, so the family adjustment is the
    only thing standing between an interval above zero and a named winner. This used to
    read ``item.lower <= 0.0``: a third implementation of "declare a winner when every
    pairwise interval excludes zero", with no multiplicity control at all, which named a
    best model roughly four times in ten among five equally good models.
    """

    eligible = tuple(model for model in models if model.eligible)
    if not eligible:
        return (
            ModelComparisonStatus.NO_ELIGIBLE_MODEL,
            None,
            "every model failed its convergence or importance-sampling checks",
        )
    best = eligible[0].name
    if len(eligible) == 1:
        return (
            ModelComparisonStatus.RESOLVED,
            best,
            f"{best} is the only model with a usable posterior; no comparison was possible",
        )
    by_pair = {(item.left_model, item.right_model): item for item in differences}
    undecided = tuple(
        other.name
        for other in eligible[1:]
        if (item := by_pair.get((best, other.name))) is None or item.favours != best
    )
    reading = (
        f"after {family.multiplicity.value} adjustment at {family.family_error_rate:.3g}"
        if family.corrected
        else f"under the uncorrected per-contrast reading at {family.family_error_rate:.3g}"
    )
    if undecided:
        return (
            ModelComparisonStatus.UNRESOLVED,
            None,
            (
                f"{len(undecided)} of {len(eligible) - 1} contrasts against {best} do not "
                f"exclude equal predictive performance across the family of "
                f"{family.n_comparisons} simultaneous contrasts "
                f"({family.n_decisive} decisive {reading}; "
                f"{family.n_separated} separated on the unadjusted interval alone)"
            ),
        )
    return (
        ModelComparisonStatus.RESOLVED,
        best,
        (
            f"{best} improves on every eligible competitor across all "
            f"{family.n_comparisons} simultaneous contrasts {reading}"
        ),
    )


def _json_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _json_float(value: Any) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


def _protected(values: NDArray[Any]) -> NDArray[Any]:
    protected = np.array(values, copy=True)
    protected.setflags(write=False)
    return protected
