"""Simulation-based calibration for backend-neutral posterior pipelines.

Convergence policy
------------------
SBC's validity argument is conditional: ranks are uniform when the sampler draws from the
posterior it claims to. A divergent or unmixed replicate therefore contributes a rank whose
distribution is unknown, and pooling it into the histogram produces exactly the deviation
SBC exists to detect, with no way to tell the two apart. Every replicate is consequently
audited with :func:`behavio.posterior_diagnostics.audit_posterior`; a replicate whose audit
status is ``FAIL`` is retained as a coded :class:`SBCFailure` with stage ``"audit"`` and
excluded from the ranks, so the histogram reports the sample that actually survived rather
than the sample that was requested. Pass ``audit_policy=None`` to opt out; the choice is
recorded on the report so a reader can see that the histogram was not filtered.

Uniformity
----------
:meth:`SBCReport.uniformity` reports the ECDF-difference statistic against a simultaneous
confidence band (Säilynoja, Bürkner & Vehtari, 2022, https://arxiv.org/abs/2103.10522)
alongside a binned chi-square. Neither is collapsed into a verdict: the statistic, its
reference distribution, and the band are surfaced so the reader judges.
"""

from __future__ import annotations

import functools
import importlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from behavio._internal.parallel import WorkerBackend, map_ordered, resolve_workers
from behavio.contracts.audit import AuditSeverity
from behavio.posterior import PosteriorResult, PosteriorVariable
from behavio.posterior_diagnostics import (
    PosteriorAuditPolicy,
    PosteriorAuditStatus,
    audit_posterior,
)
from behavio.study import Study

DEFAULT_SBC_AUDIT_POLICY = PosteriorAuditPolicy()
"""Convergence policy applied to every SBC replicate unless one is passed explicitly."""

SBC_FAILURE_STAGES = ("simulation", "inference", "evaluation", "audit")
"""Pipeline stages that can retain an SBC failure instead of a rank."""


class SBCError(ValueError):
    """Raised when an SBC pipeline violates its declared contract."""


@dataclass(frozen=True, slots=True)
class SBCSimulation:
    """One prior-predictive study and the latent truth used to generate it."""

    study: Study
    truth: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.study, Study):
            raise TypeError("SBC simulation study must be a Study")
        if not isinstance(self.truth, Mapping) or not self.truth:
            raise ValueError("SBC simulation truth must be a non-empty mapping")
        truth: dict[str, NDArray[Any]] = {}
        for name, value in self.truth.items():
            if not isinstance(name, str) or not name:
                raise ValueError("SBC truth names must be non-empty strings")
            array = np.array(value, copy=True)
            if array.size < 1 or array.dtype.kind not in "biuf" or not np.all(np.isfinite(array)):
                raise ValueError(f"SBC truth {name!r} must contain finite real values")
            array.setflags(write=False)
            truth[name] = array
        object.__setattr__(self, "truth", MappingProxyType(truth))


@runtime_checkable
class SBCTestQuantity(Protocol):
    """A declared scalar or labelled vector quantity checked by SBC."""

    @property
    def name(self) -> str: ...

    @property
    def signature(self) -> str: ...

    def truth_values(self, simulation: SBCSimulation) -> NDArray[Any]: ...

    def posterior_values(self, result: PosteriorResult) -> PosteriorVariable: ...


@dataclass(frozen=True, slots=True)
class PosteriorParameterQuantity:
    """Match one named simulation truth to one natural posterior variable."""

    posterior_name: str
    truth_name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.posterior_name, str) or not self.posterior_name:
            raise ValueError("posterior_name must be a non-empty string")
        if self.truth_name is not None and (
            not isinstance(self.truth_name, str) or not self.truth_name
        ):
            raise ValueError("truth_name must be null or a non-empty string")

    @property
    def name(self) -> str:
        return self.posterior_name

    @property
    def signature(self) -> str:
        return (
            f"posterior-parameter[v1;posterior={self.posterior_name};"
            f"truth={self.truth_name or self.posterior_name}]"
        )

    def truth_values(self, simulation: SBCSimulation) -> NDArray[Any]:
        name = self.posterior_name if self.truth_name is None else self.truth_name
        if name not in simulation.truth:
            raise SBCError(f"simulation truth has no quantity {name!r}")
        return np.asarray(simulation.truth[name])

    def posterior_values(self, result: PosteriorResult) -> PosteriorVariable:
        if self.posterior_name not in result["posterior"].variable_names:
            raise SBCError(f"posterior has no quantity {self.posterior_name!r}")
        return result["posterior"][self.posterior_name]


@dataclass(frozen=True, slots=True)
class SBCRank:
    """One randomized truth rank for a scalar labelled test quantity."""

    replicate: int
    quantity_name: str
    quantity_signature: str
    target: str
    coordinate: tuple[tuple[str, Any], ...]
    truth: float
    rank: int
    n_posterior_draws: int
    posterior_mean: float
    posterior_sd: float
    interval: tuple[float, float]
    covered: bool
    thinned_ess: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.replicate, bool) or not isinstance(self.replicate, int):
            raise TypeError("SBC replicate index must be an integer")
        if self.replicate < 0:
            raise ValueError("SBC replicate index must be non-negative")
        if not self.quantity_name or not self.quantity_signature or not self.target:
            raise ValueError("SBC ranks require quantity and target identity")
        if self.n_posterior_draws < 1 or not 0 <= self.rank <= self.n_posterior_draws:
            raise ValueError("SBC rank must lie between zero and the posterior draw count")
        for value, label in (
            (self.truth, "truth"),
            (self.posterior_mean, "posterior_mean"),
            (self.posterior_sd, "posterior_sd"),
        ):
            if not np.isfinite(value):
                raise ValueError(f"SBC {label} must be finite")
        if len(self.interval) != 2 or not np.all(np.isfinite(self.interval)):
            raise ValueError("SBC interval must contain two finite values")
        if self.interval[0] > self.interval[1]:
            raise ValueError("SBC interval must be ordered")
        if not isinstance(self.covered, bool):
            raise ValueError("SBC covered flag must be boolean")
        if self.thinned_ess is not None and (
            not np.isfinite(self.thinned_ess) or self.thinned_ess <= 0
        ):
            raise ValueError("SBC thinned_ess must be null or a finite positive value")
        coordinate: list[tuple[str, Any]] = []
        for item in self.coordinate:
            if not isinstance(item, Sequence) or len(item) != 2:
                raise ValueError("SBC coordinates must contain dimension/value pairs")
            dim, value = item
            if not isinstance(dim, str) or not dim:
                raise ValueError("SBC coordinate dimensions must be non-empty strings")
            coordinate.append((dim, _scalar(value)))
        if len({dim for dim, _ in coordinate}) != len(coordinate):
            raise ValueError("SBC coordinate dimensions must be unique")
        object.__setattr__(self, "coordinate", tuple(coordinate))
        object.__setattr__(self, "interval", tuple(float(value) for value in self.interval))

    @property
    def normalized_rank(self) -> float:
        """Map the discrete rank to the midpoint of its unit-interval cell."""

        return (self.rank + 0.5) / (self.n_posterior_draws + 1.0)

    @property
    def relative_ess(self) -> float | None:
        """Bulk ESS of the thinned draws per thinned draw, or null when unmeasured.

        SBC assumes the retained draws are near-independent. A value close to one means the
        declared ``thin`` stride achieved that; a small value means the histogram may be
        non-uniform because the chain is autocorrelated rather than because the model is
        wrong.
        """

        if self.thinned_ess is None:
            return None
        return self.thinned_ess / self.n_posterior_draws


@dataclass(frozen=True, slots=True)
class SBCFailure:
    """One retained simulation, inference, evaluation, or convergence failure."""

    replicate: int
    stage: str
    error_type: str
    message: str
    simulation_seed: int
    inference_seed: int
    audit_issue_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.replicate, bool) or not isinstance(self.replicate, int):
            raise TypeError("SBC failure replicate index must be an integer")
        if self.replicate < 0:
            raise ValueError("SBC failure replicate index must be non-negative")
        if self.stage not in set(SBC_FAILURE_STAGES):
            raise ValueError("SBC failure stage is not recognized")
        if not self.error_type or not self.message:
            raise ValueError("SBC failures require an error type and message")
        if any(
            isinstance(seed, bool) or not isinstance(seed, int)
            for seed in (self.simulation_seed, self.inference_seed)
        ):
            raise TypeError("SBC failure seeds must be integers")
        if self.simulation_seed < 0 or self.inference_seed < 0:
            raise ValueError("SBC failure seeds must be non-negative")
        codes = tuple(self.audit_issue_codes)
        if any(not isinstance(code, str) or not code for code in codes):
            raise ValueError("SBC audit issue codes must be non-empty strings")
        if codes and self.stage != "audit":
            raise ValueError("only an SBC audit failure can carry audit issue codes")
        object.__setattr__(self, "audit_issue_codes", codes)


@dataclass(frozen=True, slots=True)
class SBCSummary:
    """Finite-sample descriptive calibration summary for one labelled target.

    ``n_replicates`` counts the replicates that reached the histogram, not the replicates
    that were requested. ``repeats_requested``, ``n_unconverged`` and ``n_other_failures``
    carry the rest of the accounting, so a histogram built from forty of one hundred
    intended replicates cannot present as though it came from one hundred.
    """

    target: str
    n_replicates: int
    mean_normalized_rank: float
    interval_coverage: float
    histogram_counts: tuple[int, ...]
    repeats_requested: int
    n_unconverged: int
    n_other_failures: int
    mean_relative_ess: float | None
    min_relative_ess: float | None

    def __post_init__(self) -> None:
        counts = tuple(self.histogram_counts)
        if not self.target or self.n_replicates < 1:
            raise ValueError("SBC summaries require a target and at least one replicate")
        if len(counts) < 2 or any(value < 0 for value in counts):
            raise ValueError("SBC histogram counts must contain at least two non-negative bins")
        if sum(counts) != self.n_replicates:
            raise ValueError("SBC histogram counts must sum to n_replicates")
        if not 0 <= self.mean_normalized_rank <= 1:
            raise ValueError("SBC mean normalized rank must lie between zero and one")
        if not 0 <= self.interval_coverage <= 1:
            raise ValueError("SBC interval coverage must lie between zero and one")
        if self.n_unconverged < 0 or self.n_other_failures < 0:
            raise ValueError("SBC exclusion counts must be non-negative")
        if self.repeats_requested < self.n_replicates + self.n_unconverged + self.n_other_failures:
            raise ValueError("SBC summary accounting cannot exceed repeats_requested")
        for value, label in (
            (self.mean_relative_ess, "mean_relative_ess"),
            (self.min_relative_ess, "min_relative_ess"),
        ):
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError(f"SBC {label} must be null or a finite positive value")
        object.__setattr__(self, "histogram_counts", counts)

    @property
    def expected_bin_count(self) -> float:
        return self.n_replicates / len(self.histogram_counts)

    @property
    def retained_fraction(self) -> float:
        """Fraction of the requested replicates that reached this histogram."""

        return self.n_replicates / self.repeats_requested


@dataclass(frozen=True, slots=True)
class SBCUniformity:
    """ECDF-difference band and binned chi-square for one labelled target.

    The primary statistic is the difference between the empirical CDF of the normalized
    ranks and the CDF they would have under the exact null, evaluated on a grid. The
    reference is a *simultaneous* band: the pointwise level ``pointwise_level`` is
    calibrated so that, under the null, the whole difference curve stays inside the band
    with probability ``confidence_level``. It is not a pointwise band evaluated at many
    places, which would be exceeded far more often than its nominal level.

    ``null`` is ``"discrete-uniform"`` when every retained replicate has the same posterior
    draw count, in which case the reference is the exact discrete uniform over the
    ``n_posterior_draws + 1`` rank cells rather than a continuous approximation of it. It is
    ``"continuous-uniform"`` when the draw counts differ and no single discrete null exists.
    """

    target: str
    n_replicates: int
    null: str
    n_posterior_draws: int | None
    confidence_level: float
    pointwise_level: float
    n_band_simulations: int
    band_seed: int
    evaluation_points: tuple[float, ...]
    null_cdf: tuple[float, ...]
    ecdf_difference: tuple[float, ...]
    lower_difference_band: tuple[float, ...]
    upper_difference_band: tuple[float, ...]
    n_points_outside_band: int
    bins: int
    chi_square: float
    chi_square_dof: int
    chi_square_p_value: float
    min_expected_bin_count: float

    def __post_init__(self) -> None:
        if not self.target or self.n_replicates < 1:
            raise ValueError("SBC uniformity requires a target and at least one replicate")
        if self.null not in {"discrete-uniform", "continuous-uniform"}:
            raise ValueError("SBC uniformity null is not recognized")
        if not 0 < self.confidence_level < 1:
            raise ValueError("SBC confidence_level must lie between zero and one")
        if not 0 < self.pointwise_level <= 1 - self.confidence_level:
            raise ValueError("SBC pointwise_level must be positive and at most one minus the level")
        curves = ("evaluation_points", "null_cdf", "ecdf_difference")
        curves += ("lower_difference_band", "upper_difference_band")
        values = {name: tuple(float(item) for item in getattr(self, name)) for name in curves}
        if len({len(item) for item in values.values()}) != 1 or not values["null_cdf"]:
            raise ValueError("SBC uniformity curves must be non-empty and aligned")
        if any(not np.all(np.isfinite(item)) for item in values.values()):
            raise ValueError("SBC uniformity curves must be finite")
        if any(
            lower > upper
            for lower, upper in zip(
                values["lower_difference_band"],
                values["upper_difference_band"],
                strict=True,
            )
        ):
            raise ValueError("SBC uniformity band must be ordered")
        if self.n_points_outside_band < 0 or self.n_points_outside_band > len(values["null_cdf"]):
            raise ValueError("SBC band exceedance count is out of range")
        if self.chi_square < 0 or not np.isfinite(self.chi_square) or self.chi_square_dof < 1:
            raise ValueError("SBC chi-square requires a finite statistic and positive dof")
        if not 0 <= self.chi_square_p_value <= 1:
            raise ValueError("SBC chi-square p-value must lie between zero and one")
        for name, item in values.items():
            object.__setattr__(self, name, item)

    @property
    def max_absolute_difference(self) -> float:
        """Largest absolute ECDF difference over the evaluation grid."""

        return float(np.max(np.abs(self.ecdf_difference)))


@dataclass(frozen=True, slots=True)
class SBCReport:
    """Raw SBC ranks and failures tied to declared pipeline provenance."""

    simulation_signature: str
    inference_signature: str
    quantity_signatures: tuple[str, ...]
    repeats_requested: int
    root_seed: int
    thin: int
    interval_probability: float
    ranks: tuple[SBCRank, ...]
    failures: tuple[SBCFailure, ...]
    audit_policy: PosteriorAuditPolicy | None = None

    def __post_init__(self) -> None:
        if not self.simulation_signature or not self.inference_signature:
            raise ValueError("SBC report requires simulation and inference signatures")
        signatures = tuple(self.quantity_signatures)
        if not signatures or len(set(signatures)) != len(signatures):
            raise ValueError("SBC quantity signatures must be non-empty and unique")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (self.repeats_requested, self.root_seed, self.thin)
        ):
            raise TypeError("SBC repeats, root_seed, and thin must be integers")
        if self.repeats_requested < 1 or self.thin < 1 or self.root_seed < 0:
            raise ValueError("SBC repeats/thin must be positive and root_seed non-negative")
        if not 0 < self.interval_probability < 1:
            raise ValueError("SBC interval_probability must lie between zero and one")
        ranks = tuple(self.ranks)
        failures = tuple(self.failures)
        rank_replicates = {item.replicate for item in ranks}
        failed_replicates = {item.replicate for item in failures}
        if len(failed_replicates) != len(failures):
            raise ValueError("an SBC replicate can retain at most one failure")
        if rank_replicates & failed_replicates:
            raise ValueError("an SBC replicate cannot be both successful and failed")
        if any(
            index < 0 or index >= self.repeats_requested
            for index in rank_replicates | failed_replicates
        ):
            raise ValueError("SBC replicate index exceeds repeats_requested")
        if len(rank_replicates | failed_replicates) != self.repeats_requested:
            raise ValueError("SBC report must account for every requested replicate")
        if self.audit_policy is not None and not isinstance(
            self.audit_policy, PosteriorAuditPolicy
        ):
            raise TypeError("SBC audit_policy must be a PosteriorAuditPolicy or null")
        if self.audit_policy is None and any(item.stage == "audit" for item in failures):
            raise ValueError("SBC audit failures require a declared audit policy")
        if any(rank.quantity_signature not in signatures for rank in ranks):
            raise ValueError("SBC rank has an undeclared quantity signature")
        target_sets: dict[int, set[tuple[str, str]]] = {}
        for rank in ranks:
            identity = (rank.quantity_signature, rank.target)
            selected = target_sets.setdefault(rank.replicate, set())
            if identity in selected:
                raise ValueError("an SBC replicate cannot contain duplicate labelled targets")
            selected.add(identity)
        if target_sets and len({frozenset(targets) for targets in target_sets.values()}) != 1:
            raise ValueError("successful SBC replicates must contain the same labelled targets")
        object.__setattr__(self, "quantity_signatures", signatures)
        object.__setattr__(self, "ranks", ranks)
        object.__setattr__(self, "failures", failures)

    @property
    def n_successful(self) -> int:
        return len({rank.replicate for rank in self.ranks})

    @property
    def n_failed(self) -> int:
        return len(self.failures)

    @property
    def success_rate(self) -> float:
        return self.n_successful / self.repeats_requested

    @property
    def unconverged_replicates(self) -> tuple[int, ...]:
        """Replicates excluded from the ranks because their posterior audit failed."""

        return tuple(item.replicate for item in self.failures if item.stage == "audit")

    @property
    def n_unconverged(self) -> int:
        return len(self.unconverged_replicates)

    @property
    def n_other_failures(self) -> int:
        """Failures that never produced an auditable posterior at all."""

        return self.n_failed - self.n_unconverged

    def summary(self, *, bins: int = 10) -> tuple[SBCSummary, ...]:
        """Summarize ranks without converting finite simulations into a pass/fail claim."""

        if isinstance(bins, bool) or not isinstance(bins, int) or bins < 2:
            raise ValueError("SBC summary bins must be an integer of at least two")
        targets = tuple(dict.fromkeys(rank.target for rank in self.ranks))
        summaries = []
        for target in targets:
            selected = tuple(rank for rank in self.ranks if rank.target == target)
            normalized = np.asarray([rank.normalized_rank for rank in selected])
            counts, _ = np.histogram(normalized, bins=bins, range=(0.0, 1.0))
            relative = [rank.relative_ess for rank in selected if rank.relative_ess is not None]
            summaries.append(
                SBCSummary(
                    target=target,
                    n_replicates=len(selected),
                    mean_normalized_rank=float(np.mean(normalized)),
                    interval_coverage=float(np.mean([rank.covered for rank in selected])),
                    histogram_counts=tuple(int(value) for value in counts),
                    repeats_requested=self.repeats_requested,
                    n_unconverged=self.n_unconverged,
                    n_other_failures=self.n_other_failures,
                    mean_relative_ess=float(np.mean(relative)) if relative else None,
                    min_relative_ess=float(np.min(relative)) if relative else None,
                )
            )
        return tuple(summaries)

    def uniformity(
        self,
        *,
        bins: int = 10,
        confidence_level: float = 0.95,
        n_band_simulations: int = 2_000,
        band_seed: int = 0,
        n_evaluation_points: int = 100,
    ) -> tuple[SBCUniformity, ...]:
        """Assess rank uniformity against its exact null, still without a verdict.

        Returns, per labelled target, the ECDF-difference curve with a simultaneous
        confidence band and a binned chi-square statistic with its degrees of freedom and
        p-value. The mean normalized rank alone is blind to the two commonest SBC failure
        modes -- the symmetric U-shape of an over-dispersed posterior and the symmetric
        cap of an under-dispersed one -- because both leave the mean at one half.

        The band is calibrated by simulating ``n_band_simulations`` rank sets from the
        exact null with ``band_seed``, so repeated calls on the same report agree.
        """

        if isinstance(bins, bool) or not isinstance(bins, int) or bins < 2:
            raise ValueError("SBC uniformity bins must be an integer of at least two")
        if not np.isfinite(confidence_level) or not 0 < confidence_level < 1:
            raise ValueError("SBC confidence_level must be finite and lie between zero and one")
        for value, label in (
            (n_band_simulations, "n_band_simulations"),
            (n_evaluation_points, "n_evaluation_points"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ValueError(f"{label} must be an integer of at least two")
        if isinstance(band_seed, bool) or not isinstance(band_seed, int) or band_seed < 0:
            raise ValueError("band_seed must be a non-negative integer")
        targets = tuple(dict.fromkeys(rank.target for rank in self.ranks))
        return tuple(
            _uniformity_for(
                target,
                tuple(rank for rank in self.ranks if rank.target == target),
                bins=bins,
                confidence_level=confidence_level,
                n_band_simulations=n_band_simulations,
                band_seed=band_seed,
                n_evaluation_points=n_evaluation_points,
            )
            for target in targets
        )

    def to_dict(self, *, bins: int = 10, confidence_level: float = 0.95) -> dict[str, Any]:
        summaries = self.summary(bins=bins)
        uniformity = self.uniformity(bins=bins, confidence_level=confidence_level)
        return {
            "simulation_signature": self.simulation_signature,
            "inference_signature": self.inference_signature,
            "quantity_signatures": list(self.quantity_signatures),
            "repeats_requested": self.repeats_requested,
            "root_seed": self.root_seed,
            "thin": self.thin,
            "interval_probability": self.interval_probability,
            "audit_policy": None if self.audit_policy is None else self.audit_policy.to_dict(),
            "n_successful": self.n_successful,
            "n_failed": self.n_failed,
            "n_unconverged": self.n_unconverged,
            "n_other_failures": self.n_other_failures,
            "unconverged_replicates": list(self.unconverged_replicates),
            "success_rate": self.success_rate,
            "summary": [
                {
                    "target": item.target,
                    "n_replicates": item.n_replicates,
                    "mean_normalized_rank": item.mean_normalized_rank,
                    "interval_coverage": item.interval_coverage,
                    "histogram_counts": list(item.histogram_counts),
                    "expected_bin_count": item.expected_bin_count,
                    "repeats_requested": item.repeats_requested,
                    "n_unconverged": item.n_unconverged,
                    "n_other_failures": item.n_other_failures,
                    "retained_fraction": item.retained_fraction,
                    "mean_relative_ess": item.mean_relative_ess,
                    "min_relative_ess": item.min_relative_ess,
                }
                for item in summaries
            ],
            "uniformity": [
                {
                    "target": item.target,
                    "n_replicates": item.n_replicates,
                    "null": item.null,
                    "n_posterior_draws": item.n_posterior_draws,
                    "confidence_level": item.confidence_level,
                    "pointwise_level": item.pointwise_level,
                    "n_band_simulations": item.n_band_simulations,
                    "band_seed": item.band_seed,
                    "evaluation_points": list(item.evaluation_points),
                    "null_cdf": list(item.null_cdf),
                    "ecdf_difference": list(item.ecdf_difference),
                    "lower_difference_band": list(item.lower_difference_band),
                    "upper_difference_band": list(item.upper_difference_band),
                    "n_points_outside_band": item.n_points_outside_band,
                    "max_absolute_difference": item.max_absolute_difference,
                    "bins": item.bins,
                    "chi_square": item.chi_square,
                    "chi_square_dof": item.chi_square_dof,
                    "chi_square_p_value": item.chi_square_p_value,
                    "min_expected_bin_count": item.min_expected_bin_count,
                }
                for item in uniformity
            ],
            "ranks": [
                {
                    "replicate": item.replicate,
                    "quantity_name": item.quantity_name,
                    "quantity_signature": item.quantity_signature,
                    "target": item.target,
                    "coordinate": dict(item.coordinate),
                    "truth": item.truth,
                    "rank": item.rank,
                    "normalized_rank": item.normalized_rank,
                    "n_posterior_draws": item.n_posterior_draws,
                    "posterior_mean": item.posterior_mean,
                    "posterior_sd": item.posterior_sd,
                    "interval": list(item.interval),
                    "covered": item.covered,
                    "thinned_ess": item.thinned_ess,
                    "relative_ess": item.relative_ess,
                }
                for item in self.ranks
            ],
            "failures": [
                {
                    "replicate": item.replicate,
                    "stage": item.stage,
                    "error_type": item.error_type,
                    "message": item.message,
                    "simulation_seed": item.simulation_seed,
                    "inference_seed": item.inference_seed,
                    "audit_issue_codes": list(item.audit_issue_codes),
                }
                for item in self.failures
            ],
        }


SBCSimulator = Callable[[int], SBCSimulation]
SBCInference = Callable[[Study, int], PosteriorResult]


@dataclass(frozen=True, slots=True)
class _ReplicateTask:
    """Everything one SBC replicate needs, addressed by its position in the run.

    ``seeds`` is the replicate's ``(simulation, inference, rank)`` triple, drawn from
    ``SeedSequence(seed).spawn(repeats)[replicate]`` before any work began. No worker draws
    randomness of its own, so a replicate produces the same ranks wherever it runs.
    """

    replicate: int
    simulator: SBCSimulator
    inference: SBCInference
    quantities: tuple[SBCTestQuantity, ...]
    seeds: tuple[int, int, int]
    thin: int
    interval_probability: float
    audit_policy: PosteriorAuditPolicy | None


@dataclass(frozen=True, slots=True)
class _ReplicateOutcome:
    """One replicate's ranks, or the failure that replaced them, plus its target identity.

    The posterior itself is deliberately not carried back. A replicate's draws are the
    largest thing it produces and nothing outside the worker reads them, so returning the
    ranks and the audit verdict alone keeps the inter-process traffic proportional to the
    evidence rather than to the sampler.
    """

    replicate: int
    simulation_seed: int
    inference_seed: int
    failure: SBCFailure | None
    ranks: tuple[SBCRank, ...]
    identity: tuple[tuple[str, tuple[str, ...], tuple[tuple[Any, ...], ...]], ...] | None
    audit_failure: SBCFailure | None


def _run_replicate(task: _ReplicateTask) -> _ReplicateOutcome:
    """Simulate, fit, rank and audit one replicate as a pure function of its position.

    This mirrors the serial stage order exactly -- simulate, infer, rank, audit -- with one
    deliberate omission: the cross-replicate target-identity check cannot be made here,
    because it compares this replicate against whichever earlier replicate established the
    targets. The identity is returned instead and
    :func:`run_simulation_based_calibration` settles it in replicate order.

    Auditing still happens here even though a later identity mismatch would discard the
    result. ``audit_posterior`` is a pure function of the draws, so computing it and
    throwing it away costs time and changes nothing; the alternative would be to carry the
    whole posterior back to the parent just so it could be audited there.
    """

    simulation_seed, inference_seed, rank_seed = task.seeds
    replicate = task.replicate
    try:
        simulation = task.simulator(simulation_seed)
        if not isinstance(simulation, SBCSimulation):
            raise TypeError("simulator must return SBCSimulation")
    except Exception as error:
        return _outcome(
            task, failure=_failure(replicate, "simulation", error, simulation_seed, inference_seed)
        )
    try:
        posterior = task.inference(simulation.study, inference_seed)
        if not isinstance(posterior, PosteriorResult):
            raise TypeError("inference must return PosteriorResult")
    except Exception as error:
        return _outcome(
            task, failure=_failure(replicate, "inference", error, simulation_seed, inference_seed)
        )
    try:
        local, identity = _rank_replicate(
            replicate,
            simulation,
            posterior,
            task.quantities,
            thin=task.thin,
            interval_probability=task.interval_probability,
            rank_seed=rank_seed,
        )
    except Exception as error:
        return _outcome(
            task, failure=_failure(replicate, "evaluation", error, simulation_seed, inference_seed)
        )
    audit_failure: SBCFailure | None = None
    if task.audit_policy is not None:
        try:
            audit = audit_posterior(posterior, policy=task.audit_policy)
        except Exception as error:
            audit_failure = _failure(replicate, "audit", error, simulation_seed, inference_seed)
        else:
            if audit.status is PosteriorAuditStatus.FAIL:
                audit_failure = _unconverged(replicate, audit, simulation_seed, inference_seed)
    return _outcome(task, ranks=local, identity=identity, audit_failure=audit_failure)


def _outcome(
    task: _ReplicateTask,
    *,
    failure: SBCFailure | None = None,
    ranks: tuple[SBCRank, ...] = (),
    identity: tuple[tuple[str, tuple[str, ...], tuple[tuple[Any, ...], ...]], ...] | None = None,
    audit_failure: SBCFailure | None = None,
) -> _ReplicateOutcome:
    return _ReplicateOutcome(
        replicate=task.replicate,
        simulation_seed=task.seeds[0],
        inference_seed=task.seeds[1],
        failure=failure,
        ranks=ranks,
        identity=identity,
        audit_failure=audit_failure,
    )


def run_simulation_based_calibration(
    simulator: SBCSimulator,
    inference: SBCInference,
    quantities: Sequence[SBCTestQuantity],
    *,
    repeats: int,
    seed: int,
    simulation_signature: str,
    inference_signature: str,
    thin: int = 1,
    interval_probability: float = 0.9,
    audit_policy: PosteriorAuditPolicy | None = DEFAULT_SBC_AUDIT_POLICY,
    workers: int = 1,
    backend: WorkerBackend | str = WorkerBackend.PROCESS,
) -> SBCReport:
    """Run a prior-predictive SBC pipeline and retain every rank or failure.

    Every replicate whose posterior audit fails under ``audit_policy`` is retained as an
    ``"audit"``-stage :class:`SBCFailure` and kept out of the ranks: SBC is only a valid
    check conditional on convergence, so a divergent or unmixed replicate cannot be pooled
    into the histogram it is supposed to test. ``audit_policy=None`` disables the check and
    is recorded on the report.

    ``workers`` runs replicates concurrently. The report is **bit-identical** for every
    worker count: each replicate's ``(simulation, inference, rank)`` seeds come from
    ``SeedSequence(seed).spawn(repeats)`` indexed by replicate, and both ``ranks`` and
    ``failures`` are assembled in replicate order rather than in completion order, so a
    retained failure sits where it sat serially. ``workers=1`` is the default and builds no
    executor.

    ``backend`` selects processes or threads. Processes require ``simulator`` and
    ``inference`` to be picklable -- a lambda or a locally defined closure is the usual
    thing that is not, and :class:`~behavio._internal.parallel.UnpicklableTaskError` says so
    before any replicate runs. ``backend="thread"`` pickles nothing and is the fallback;
    it is also the right choice when inference is a sampler that already releases the GIL.
    """

    if not callable(simulator) or not callable(inference):
        raise TypeError("simulator and inference must be callable")
    declared = tuple(quantities)
    if not declared or any(not isinstance(item, SBCTestQuantity) for item in declared):
        raise TypeError("quantities must contain SBCTestQuantity objects")
    names = tuple(item.name for item in declared)
    signatures = tuple(item.signature for item in declared)
    if any(not isinstance(value, str) or not value for value in names):
        raise ValueError("SBC quantity names must be non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError("SBC quantity names must be unique")
    if any(not isinstance(value, str) or not value for value in signatures):
        raise ValueError("SBC quantity signatures must be non-empty strings")
    if len(set(signatures)) != len(signatures):
        raise ValueError("SBC quantity signatures must be unique")
    for value, label in (
        (simulation_signature, "simulation_signature"),
        (inference_signature, "inference_signature"),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label} must be a non-empty string")
    for value, label in ((repeats, "repeats"), (thin, "thin")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{label} must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not np.isfinite(interval_probability) or not 0 < interval_probability < 1:
        raise ValueError("interval_probability must be finite and lie between zero and one")
    if audit_policy is not None and not isinstance(audit_policy, PosteriorAuditPolicy):
        raise TypeError("audit_policy must be a PosteriorAuditPolicy or None")
    backend = WorkerBackend(backend)
    resolve_workers(workers, n_tasks=1)

    child_sequences = np.random.SeedSequence(seed).spawn(repeats)
    tasks = tuple(
        _ReplicateTask(
            replicate=replicate,
            simulator=simulator,
            inference=inference,
            quantities=declared,
            seeds=tuple(int(value) for value in sequence.generate_state(3, dtype=np.uint64)),
            thin=thin,
            interval_probability=interval_probability,
            audit_policy=audit_policy,
        )
        for replicate, sequence in enumerate(child_sequences)
    )
    outcomes = map_ordered(_run_replicate, tasks, workers=workers, backend=backend)

    ranks: list[SBCRank] = []
    failures: list[SBCFailure] = []
    expected_targets: (
        tuple[tuple[str, tuple[str, ...], tuple[tuple[Any, ...], ...]], ...] | None
    ) = None
    # The cross-replicate identity check is the one part of a replicate that is *not*
    # independent of the others, so it is settled here, in replicate order, rather than in a
    # worker that cannot see what the earlier replicates declared. Everything a worker does
    # is a pure function of its own replicate; this loop replays the serial decision tree
    # over those results and so retains failures in replicate order under any worker count.
    for outcome in outcomes:
        if outcome.failure is not None:
            failures.append(outcome.failure)
            continue
        if expected_targets is None:
            expected_targets = outcome.identity
        elif outcome.identity != expected_targets:
            failures.append(
                _failure(
                    outcome.replicate,
                    "evaluation",
                    SBCError("SBC target dimensions or coordinates changed across replicates"),
                    outcome.simulation_seed,
                    outcome.inference_seed,
                )
            )
            continue
        if outcome.audit_failure is not None:
            failures.append(outcome.audit_failure)
            continue
        ranks.extend(outcome.ranks)

    return SBCReport(
        simulation_signature=simulation_signature,
        inference_signature=inference_signature,
        quantity_signatures=signatures,
        repeats_requested=repeats,
        root_seed=seed,
        thin=thin,
        interval_probability=interval_probability,
        ranks=tuple(ranks),
        failures=tuple(failures),
        audit_policy=audit_policy,
    )


def _rank_replicate(
    replicate: int,
    simulation: SBCSimulation,
    posterior: PosteriorResult,
    quantities: tuple[SBCTestQuantity, ...],
    *,
    thin: int,
    interval_probability: float,
    rank_seed: int,
) -> tuple[
    tuple[SBCRank, ...],
    tuple[tuple[str, tuple[str, ...], tuple[tuple[Any, ...], ...]], ...],
]:
    generator = np.random.default_rng(rank_seed)
    local: list[SBCRank] = []
    identities = []
    alpha = (1.0 - interval_probability) / 2.0
    for quantity in quantities:
        truth = np.asarray(quantity.truth_values(simulation), dtype=np.float64)
        variable = quantity.posterior_values(posterior)
        if not isinstance(variable, PosteriorVariable):
            raise TypeError("SBC posterior_values must return PosteriorVariable")
        if variable.dims[:2] != ("chain", "draw"):
            raise SBCError("SBC posterior quantities must lead with chain and draw")
        if truth.shape != variable.values.shape[2:]:
            raise SBCError(f"truth for {quantity.name!r} does not match posterior intrinsic shape")
        if not np.all(np.isfinite(truth)) or not np.all(np.isfinite(variable.values)):
            raise SBCError(f"SBC quantity {quantity.name!r} must be finite")
        dims = variable.intrinsic_dims
        coordinates = tuple(tuple(_scalar(value) for value in variable.coords[dim]) for dim in dims)
        identities.append((quantity.signature, dims, coordinates))
        posterior_values = np.asarray(variable.values, dtype=np.float64).reshape(-1, *truth.shape)[
            ::thin
        ]
        # The same stride, kept in (chain, draw) layout, so ESS can see the chain structure
        # that the flattened ranking view deliberately discards.
        thinned_chains = np.asarray(variable.values, dtype=np.float64)[:, ::thin]
        for index in np.ndindex(truth.shape):
            draws = posterior_values[(slice(None), *index)]
            true_value = float(truth[index])
            less = int(np.count_nonzero(draws < true_value))
            ties = int(np.count_nonzero(draws == true_value))
            rank = less + int(generator.integers(0, ties + 1))
            coordinate = tuple(
                (dim, coordinates[axis][position])
                for axis, (dim, position) in enumerate(zip(dims, index, strict=True))
            )
            target = _target(quantity.name, coordinate)
            interval = tuple(float(value) for value in np.quantile(draws, (alpha, 1 - alpha)))
            local.append(
                SBCRank(
                    replicate=replicate,
                    quantity_name=quantity.name,
                    quantity_signature=quantity.signature,
                    target=target,
                    coordinate=coordinate,
                    truth=true_value,
                    rank=rank,
                    n_posterior_draws=len(draws),
                    posterior_mean=float(np.mean(draws)),
                    posterior_sd=float(np.std(draws, ddof=0)),
                    interval=interval,
                    covered=bool(interval[0] <= true_value <= interval[1]),
                    thinned_ess=_thinned_ess(thinned_chains[(slice(None), slice(None), *index)]),
                )
            )
    return tuple(local), tuple(identities)


def _target(name: str, coordinate: tuple[tuple[str, Any], ...]) -> str:
    if not coordinate:
        return name
    labels = ",".join(f"{dim}={value!r}" for dim, value in coordinate)
    return f"{name}[{labels}]"


def _failure(
    replicate: int,
    stage: str,
    error: Exception,
    simulation_seed: int,
    inference_seed: int,
) -> SBCFailure:
    return SBCFailure(
        replicate=replicate,
        stage=stage,
        error_type=type(error).__name__,
        message=str(error) or "<exception had no message>",
        simulation_seed=simulation_seed,
        inference_seed=inference_seed,
    )


def _unconverged(
    replicate: int,
    audit: Any,
    simulation_seed: int,
    inference_seed: int,
) -> SBCFailure:
    """Retain an unconverged replicate as coded evidence instead of a rank."""

    codes = tuple(issue.code for issue in audit.issues if issue.severity is AuditSeverity.ERROR)
    return SBCFailure(
        replicate=replicate,
        stage="audit",
        error_type="PosteriorAuditFailure",
        message=(
            "posterior audit failed and the replicate was excluded from the ranks: "
            + ", ".join(
                f"{issue.code}: {issue.message}"
                for issue in audit.issues
                if issue.severity is AuditSeverity.ERROR
            )
        ),
        simulation_seed=simulation_seed,
        inference_seed=inference_seed,
        audit_issue_codes=codes,
    )


@functools.cache
def _bulk_ess() -> Callable[[NDArray[np.float64]], float] | None:
    """Return a bulk-ESS callable for a ``(chain, draw)`` array, or null when unavailable.

    ArviZ 1.x moved the diagnostics into ``arviz_stats`` while ArviZ 0.x keeps them on the
    top-level module, so sniff for the first installed module that exposes ``ess``. Both
    accept a raw ``(chain, draw)`` array with ``method="bulk"``. ESS is a recorded
    diagnostic here, never a hard requirement, so a missing ArviZ yields null rather than
    an error.
    """

    for name in ("arviz_stats", "arviz"):
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        function = getattr(module, "ess", None)
        if callable(function):
            return lambda values, _function=function: float(_function(values, method="bulk"))
    return None


def _thinned_ess(values: NDArray[np.float64]) -> float | None:
    """Bulk ESS of one target's thinned ``(chain, draw)`` draws, or null when unmeasurable."""

    function = _bulk_ess()
    if function is None or values.shape[1] < 4:
        return None
    try:
        result = function(np.ascontiguousarray(values))
    except Exception:
        return None
    return float(result) if np.isfinite(result) and result > 0 else None


def _uniformity_for(
    target: str,
    ranks: tuple[SBCRank, ...],
    *,
    bins: int,
    confidence_level: float,
    n_band_simulations: int,
    band_seed: int,
    n_evaluation_points: int,
) -> SBCUniformity:
    normalized = np.asarray([rank.normalized_rank for rank in ranks], dtype=np.float64)
    n_replicates = normalized.size
    draw_counts = {int(rank.n_posterior_draws) for rank in ranks}
    if len(draw_counts) == 1:
        null = "discrete-uniform"
        n_posterior_draws: int | None = draw_counts.pop()
        n_cells = n_posterior_draws + 1
        cells = np.arange(n_cells)
        midpoints = (cells + 0.5) / n_cells
        cell_cdf = (cells + 1.0) / n_cells
        if n_cells > n_evaluation_points:
            picks = np.unique(np.rint(np.linspace(0, n_cells - 1, n_evaluation_points)).astype(int))
        else:
            picks = cells
        points = midpoints[picks]
        null_cdf = cell_cdf[picks]
    else:
        null = "continuous-uniform"
        n_posterior_draws = None
        midpoints = None
        n_cells = 0
        points = np.linspace(0.0, 1.0, n_evaluation_points + 2)[1:-1]
        null_cdf = points.copy()

    observed = np.count_nonzero(normalized[:, None] <= points[None, :], axis=0)
    pointwise_level, lower_counts, upper_counts = _simultaneous_band(
        null_cdf,
        n_replicates=n_replicates,
        confidence_level=confidence_level,
        n_band_simulations=n_band_simulations,
        band_seed=band_seed,
    )
    counts, _ = np.histogram(normalized, bins=bins, range=(0.0, 1.0))
    if midpoints is None:
        expected = np.full(bins, n_replicates / bins, dtype=np.float64)
    else:
        cells_per_bin, _ = np.histogram(midpoints, bins=bins, range=(0.0, 1.0))
        expected = n_replicates * cells_per_bin / n_cells
    support = expected > 0
    chi_square = float(np.sum((counts[support] - expected[support]) ** 2 / expected[support]))
    dof = int(np.count_nonzero(support)) - 1
    if dof < 1:
        raise SBCError("SBC chi-square needs at least two bins with positive expected counts")
    return SBCUniformity(
        target=target,
        n_replicates=n_replicates,
        null=null,
        n_posterior_draws=n_posterior_draws,
        confidence_level=confidence_level,
        pointwise_level=pointwise_level,
        n_band_simulations=n_band_simulations,
        band_seed=band_seed,
        evaluation_points=tuple(float(value) for value in points),
        null_cdf=tuple(float(value) for value in null_cdf),
        ecdf_difference=tuple(float(value) for value in observed / n_replicates - null_cdf),
        lower_difference_band=tuple(
            float(value) for value in lower_counts / n_replicates - null_cdf
        ),
        upper_difference_band=tuple(
            float(value) for value in upper_counts / n_replicates - null_cdf
        ),
        n_points_outside_band=int(
            np.count_nonzero((observed < lower_counts) | (observed > upper_counts))
        ),
        bins=bins,
        chi_square=chi_square,
        chi_square_dof=dof,
        chi_square_p_value=float(stats.chi2.sf(chi_square, dof)),
        min_expected_bin_count=float(np.min(expected[support])),
    )


def _simultaneous_band(
    null_cdf: NDArray[np.float64],
    *,
    n_replicates: int,
    confidence_level: float,
    n_band_simulations: int,
    band_seed: int,
) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Calibrate the pointwise level that gives simultaneous coverage over the whole grid.

    At each evaluation point the ECDF count is marginally binomial, so a pointwise interval
    is exact there. The counts across the grid are strongly dependent, however, so applying
    a pointwise level to every point at once would be exceeded far more often than its
    nominal rate. The pointwise level ``gamma`` is therefore lowered until the *whole*
    curve stays inside the resulting envelope with probability ``confidence_level`` under
    the null. That null is simulated exactly: the counts between consecutive evaluation
    points are multinomial with the null cell probabilities, so cumulating them reproduces
    the joint law of the ECDF without any Gaussian or asymptotic approximation.
    """

    probabilities = np.diff(np.concatenate(([0.0], null_cdf, [1.0])))
    probabilities = np.clip(probabilities, 0.0, None)
    probabilities /= probabilities.sum()
    generator = np.random.default_rng(band_seed)
    draws = generator.multinomial(n_replicates, probabilities, size=n_band_simulations)
    simulated = np.cumsum(draws[:, : null_cdf.size], axis=1)
    low, high = 0.0, 1.0 - confidence_level
    for _ in range(40):
        middle = 0.5 * (low + high)
        lower = stats.binom.ppf(middle / 2.0, n_replicates, null_cdf)
        upper = stats.binom.isf(middle / 2.0, n_replicates, null_cdf)
        inside = np.all((simulated >= lower) & (simulated <= upper), axis=1)
        if float(np.mean(inside)) >= confidence_level:
            low = middle
        else:
            high = middle
    pointwise_level = max(low, np.finfo(np.float64).tiny)
    return (
        float(pointwise_level),
        np.asarray(stats.binom.ppf(pointwise_level / 2.0, n_replicates, null_cdf)),
        np.asarray(stats.binom.isf(pointwise_level / 2.0, n_replicates, null_cdf)),
    )


def _scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value
