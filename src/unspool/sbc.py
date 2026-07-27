"""Simulation-based calibration for backend-neutral posterior pipelines."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from unspool.posterior import PosteriorResult, PosteriorVariable
from unspool.study import Study


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


@dataclass(frozen=True, slots=True)
class SBCFailure:
    """One retained simulation, inference, or evaluation failure."""

    replicate: int
    stage: str
    error_type: str
    message: str
    simulation_seed: int
    inference_seed: int

    def __post_init__(self) -> None:
        if isinstance(self.replicate, bool) or not isinstance(self.replicate, int):
            raise TypeError("SBC failure replicate index must be an integer")
        if self.replicate < 0:
            raise ValueError("SBC failure replicate index must be non-negative")
        if self.stage not in {"simulation", "inference", "evaluation"}:
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


@dataclass(frozen=True, slots=True)
class SBCSummary:
    """Finite-sample descriptive calibration summary for one labelled target."""

    target: str
    n_replicates: int
    mean_normalized_rank: float
    interval_coverage: float
    histogram_counts: tuple[int, ...]

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
        object.__setattr__(self, "histogram_counts", counts)

    @property
    def expected_bin_count(self) -> float:
        return self.n_replicates / len(self.histogram_counts)


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
            summaries.append(
                SBCSummary(
                    target=target,
                    n_replicates=len(selected),
                    mean_normalized_rank=float(np.mean(normalized)),
                    interval_coverage=float(np.mean([rank.covered for rank in selected])),
                    histogram_counts=tuple(int(value) for value in counts),
                )
            )
        return tuple(summaries)

    def to_dict(self, *, bins: int = 10) -> dict[str, Any]:
        summaries = self.summary(bins=bins)
        return {
            "simulation_signature": self.simulation_signature,
            "inference_signature": self.inference_signature,
            "quantity_signatures": list(self.quantity_signatures),
            "repeats_requested": self.repeats_requested,
            "root_seed": self.root_seed,
            "thin": self.thin,
            "interval_probability": self.interval_probability,
            "n_successful": self.n_successful,
            "n_failed": self.n_failed,
            "success_rate": self.success_rate,
            "summary": [
                {
                    "target": item.target,
                    "n_replicates": item.n_replicates,
                    "mean_normalized_rank": item.mean_normalized_rank,
                    "interval_coverage": item.interval_coverage,
                    "histogram_counts": list(item.histogram_counts),
                    "expected_bin_count": item.expected_bin_count,
                }
                for item in summaries
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
                }
                for item in self.failures
            ],
        }


SBCSimulator = Callable[[int], SBCSimulation]
SBCInference = Callable[[Study, int], PosteriorResult]


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
) -> SBCReport:
    """Run a prior-predictive SBC pipeline and retain every rank or failure."""

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

    child_sequences = np.random.SeedSequence(seed).spawn(repeats)
    ranks: list[SBCRank] = []
    failures: list[SBCFailure] = []
    expected_targets: (
        tuple[tuple[str, tuple[str, ...], tuple[tuple[Any, ...], ...]], ...] | None
    ) = None
    for replicate, sequence in enumerate(child_sequences):
        simulation_seed, inference_seed, rank_seed = (
            int(value) for value in sequence.generate_state(3, dtype=np.uint64)
        )
        try:
            simulation = simulator(simulation_seed)
            if not isinstance(simulation, SBCSimulation):
                raise TypeError("simulator must return SBCSimulation")
        except Exception as error:
            failures.append(
                _failure(replicate, "simulation", error, simulation_seed, inference_seed)
            )
            continue
        try:
            posterior = inference(simulation.study, inference_seed)
            if not isinstance(posterior, PosteriorResult):
                raise TypeError("inference must return PosteriorResult")
        except Exception as error:
            failures.append(
                _failure(replicate, "inference", error, simulation_seed, inference_seed)
            )
            continue
        try:
            local, identity = _rank_replicate(
                replicate,
                simulation,
                posterior,
                declared,
                thin=thin,
                interval_probability=interval_probability,
                rank_seed=rank_seed,
            )
            if expected_targets is None:
                expected_targets = identity
            elif identity != expected_targets:
                raise SBCError("SBC target dimensions or coordinates changed across replicates")
        except Exception as error:
            failures.append(
                _failure(replicate, "evaluation", error, simulation_seed, inference_seed)
            )
            continue
        ranks.extend(local)

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


def _scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value
