"""Declared refit sensitivity analyses over scalar scientific summaries."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np

from unspool.posterior import PosteriorResult


class SensitivityError(ValueError):
    """Raised when a sensitivity analysis violates its declared contract."""


@dataclass(frozen=True, slots=True)
class SensitivityScenario:
    """One reference or alternative analysis specification."""

    name: str
    signature: str
    changes: Mapping[str, Any]
    is_reference: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("sensitivity scenario name must be a non-empty string")
        if not isinstance(self.signature, str) or not self.signature:
            raise ValueError("sensitivity scenario signature must be a non-empty string")
        if not isinstance(self.is_reference, bool):
            raise TypeError("is_reference must be boolean")
        if not isinstance(self.changes, Mapping):
            raise TypeError("sensitivity scenario changes must be a mapping")
        changes: dict[str, Any] = {}
        for key, value in self.changes.items():
            if not isinstance(key, str) or not key:
                raise ValueError("sensitivity change names must be non-empty strings")
            changes[key] = _json_scalar(value, label=f"sensitivity change {key!r}")
        if not self.is_reference and not changes:
            raise ValueError("alternative sensitivity scenarios must declare at least one change")
        object.__setattr__(self, "changes", MappingProxyType(changes))


@dataclass(frozen=True, slots=True)
class SensitivityMetric:
    """One scalar estimate whose meaning stays fixed across scenarios."""

    name: str
    signature: str
    estimate: float
    coordinate: tuple[tuple[str, Any], ...] = ()
    interval: tuple[float, float] | None = None
    unit: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("sensitivity metric name must be a non-empty string")
        if not isinstance(self.signature, str) or not self.signature:
            raise ValueError("sensitivity metric signature must be a non-empty string")
        if not np.isfinite(self.estimate):
            raise ValueError("sensitivity metric estimate must be finite")
        if self.unit is not None and (not isinstance(self.unit, str) or not self.unit):
            raise ValueError("sensitivity metric unit must be null or a non-empty string")
        coordinate = _coordinate(self.coordinate)
        interval = _interval(self.interval)
        object.__setattr__(self, "estimate", float(self.estimate))
        object.__setattr__(self, "coordinate", coordinate)
        object.__setattr__(self, "interval", interval)

    @property
    def target(self) -> str:
        return _target(self.name, self.coordinate)

    @property
    def identity(self) -> tuple[str, tuple[tuple[str, Any], ...]]:
        return self.signature, self.coordinate


@dataclass(frozen=True, slots=True)
class SensitivityOutcome:
    """Scalar summaries and provenance returned by one completed refit."""

    artifact_signature: str
    metrics: tuple[SensitivityMetric, ...]
    diagnostic_codes: tuple[str, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_signature, str) or not self.artifact_signature:
            raise ValueError("sensitivity outcome requires an artifact signature")
        metrics = tuple(self.metrics)
        if not metrics or any(not isinstance(item, SensitivityMetric) for item in metrics):
            raise ValueError("sensitivity outcome requires SensitivityMetric records")
        identities = [item.identity for item in metrics]
        if len(set(identities)) != len(identities):
            raise ValueError("sensitivity outcome metric identities must be unique")
        diagnostic_codes = tuple(self.diagnostic_codes)
        if any(not isinstance(code, str) or not code for code in diagnostic_codes):
            raise ValueError("sensitivity diagnostic codes must be non-empty strings")
        if len(set(diagnostic_codes)) != len(diagnostic_codes):
            raise ValueError("sensitivity diagnostic codes must be unique")
        if not isinstance(self.provenance, Mapping):
            raise TypeError("sensitivity provenance must be a mapping")
        provenance: dict[str, Any] = {}
        for key, value in self.provenance.items():
            if not isinstance(key, str) or not key:
                raise ValueError("sensitivity provenance names must be non-empty strings")
            provenance[key] = _json_scalar(value, label=f"sensitivity provenance {key!r}")
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "diagnostic_codes", diagnostic_codes)
        object.__setattr__(self, "provenance", MappingProxyType(provenance))


@dataclass(frozen=True, slots=True)
class SensitivityFailure:
    """One retained analysis or cross-scenario evaluation failure."""

    scenario_name: str
    scenario_signature: str
    stage: str
    error_type: str
    message: str
    seed: int

    def __post_init__(self) -> None:
        if not self.scenario_name or not self.scenario_signature:
            raise ValueError("sensitivity failures require scenario identity")
        if self.stage not in {"analysis", "evaluation"}:
            raise ValueError("sensitivity failure stage is not recognized")
        if not self.error_type or not self.message:
            raise ValueError("sensitivity failures require an error type and message")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("sensitivity failure seed must be an integer")
        if self.seed < 0:
            raise ValueError("sensitivity failure seed must be non-negative")


@dataclass(frozen=True, slots=True)
class SensitivityRun:
    """One declared scenario paired with either its outcome or retained failure."""

    scenario: SensitivityScenario
    seed: int
    outcome: SensitivityOutcome | None = None
    failure: SensitivityFailure | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.scenario, SensitivityScenario):
            raise TypeError("sensitivity run scenario must be SensitivityScenario")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("sensitivity run seed must be an integer")
        if self.seed < 0:
            raise ValueError("sensitivity run seed must be non-negative")
        if (self.outcome is None) == (self.failure is None):
            raise ValueError("sensitivity run requires exactly one outcome or failure")
        if self.outcome is not None and not isinstance(self.outcome, SensitivityOutcome):
            raise TypeError("sensitivity run outcome must be SensitivityOutcome")
        if self.failure is not None:
            if not isinstance(self.failure, SensitivityFailure):
                raise TypeError("sensitivity run failure must be SensitivityFailure")
            if (
                self.failure.scenario_name != self.scenario.name
                or self.failure.scenario_signature != self.scenario.signature
                or self.failure.seed != self.seed
            ):
                raise ValueError("sensitivity failure identity must match its scenario run")


@dataclass(frozen=True, slots=True)
class SensitivityContrast:
    """One alternative-minus-reference scalar contrast."""

    scenario_name: str
    scenario_signature: str
    metric_name: str
    metric_signature: str
    target: str
    coordinate: tuple[tuple[str, Any], ...]
    unit: str | None
    reference_estimate: float
    estimate: float
    difference: float
    reference_interval: tuple[float, float] | None
    interval: tuple[float, float] | None

    def __post_init__(self) -> None:
        if not self.scenario_name or not self.scenario_signature:
            raise ValueError("sensitivity contrasts require scenario identity")
        if not self.metric_name or not self.metric_signature or not self.target:
            raise ValueError("sensitivity contrasts require metric identity")
        if self.unit is not None and (not isinstance(self.unit, str) or not self.unit):
            raise ValueError("sensitivity contrast unit must be null or a non-empty string")
        if not np.all(np.isfinite((self.reference_estimate, self.estimate, self.difference))):
            raise ValueError("sensitivity contrast estimates must be finite")
        coordinate = _coordinate(self.coordinate)
        if self.target != _target(self.metric_name, coordinate):
            raise ValueError("sensitivity contrast target does not match its coordinate")
        expected = self.estimate - self.reference_estimate
        if not np.isclose(self.difference, expected, rtol=1e-12, atol=1e-15):
            raise ValueError("sensitivity contrast difference is inconsistent")
        object.__setattr__(self, "coordinate", coordinate)
        object.__setattr__(self, "reference_interval", _interval(self.reference_interval))
        object.__setattr__(self, "interval", _interval(self.interval))
        object.__setattr__(self, "reference_estimate", float(self.reference_estimate))
        object.__setattr__(self, "estimate", float(self.estimate))
        object.__setattr__(self, "difference", float(self.difference))


@dataclass(frozen=True, slots=True)
class SensitivitySummary:
    """Descriptive spread across successful scenarios for one metric."""

    metric_name: str
    metric_signature: str
    target: str
    unit: str | None
    n_scenarios: int
    reference_estimate: float
    minimum: float
    maximum: float
    range: float
    maximum_absolute_difference: float
    largest_difference_scenario: str

    def __post_init__(self) -> None:
        if not self.metric_name or not self.metric_signature or not self.target:
            raise ValueError("sensitivity summaries require metric identity")
        if self.unit is not None and (not isinstance(self.unit, str) or not self.unit):
            raise ValueError("sensitivity summary unit must be null or a non-empty string")
        if self.n_scenarios < 1 or not self.largest_difference_scenario:
            raise ValueError("sensitivity summaries require represented scenarios")
        values = (
            self.reference_estimate,
            self.minimum,
            self.maximum,
            self.range,
            self.maximum_absolute_difference,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("sensitivity summary estimates must be finite")
        if self.minimum > self.maximum or self.range < 0 or self.maximum_absolute_difference < 0:
            raise ValueError("sensitivity summary spread is inconsistent")
        if not self.minimum <= self.reference_estimate <= self.maximum:
            raise ValueError("sensitivity reference estimate must lie inside scenario extrema")
        if not np.isclose(self.range, self.maximum - self.minimum, rtol=1e-12, atol=1e-15):
            raise ValueError("sensitivity summary range is inconsistent")


@dataclass(frozen=True, slots=True)
class SensitivityReport:
    """All declared scenario outcomes, failures, and reference contrasts."""

    analysis_signature: str
    root_seed: int
    runs: tuple[SensitivityRun, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.analysis_signature, str) or not self.analysis_signature:
            raise ValueError("sensitivity report requires an analysis signature")
        if isinstance(self.root_seed, bool) or not isinstance(self.root_seed, int):
            raise TypeError("sensitivity root_seed must be an integer")
        if self.root_seed < 0:
            raise ValueError("sensitivity root_seed must be non-negative")
        runs = tuple(self.runs)
        if not runs or any(not isinstance(item, SensitivityRun) for item in runs):
            raise ValueError("sensitivity report requires SensitivityRun records")
        names = [item.scenario.name for item in runs]
        signatures = [item.scenario.signature for item in runs]
        if len(set(names)) != len(names) or len(set(signatures)) != len(signatures):
            raise ValueError("sensitivity scenario names and signatures must be unique")
        if sum(item.scenario.is_reference for item in runs) != 1:
            raise ValueError("sensitivity report requires exactly one reference scenario")
        if any(run.seed != _scenario_seed(self.root_seed, run.scenario.signature) for run in runs):
            raise ValueError("sensitivity run seeds must match root_seed and scenario signatures")
        reference = next(run for run in runs if run.scenario.is_reference)
        if reference.outcome is not None:
            expected = _metric_descriptors(reference.outcome)
            if any(
                run.outcome is not None and _metric_descriptors(run.outcome) != expected
                for run in runs
            ):
                raise ValueError("successful sensitivity runs must contain the reference metrics")
        object.__setattr__(self, "runs", runs)

    @property
    def reference(self) -> SensitivityRun:
        return next(run for run in self.runs if run.scenario.is_reference)

    @property
    def is_interpretable(self) -> bool:
        return self.reference.outcome is not None

    @property
    def n_successful(self) -> int:
        return sum(run.outcome is not None for run in self.runs)

    @property
    def n_failed(self) -> int:
        return len(self.runs) - self.n_successful

    @property
    def success_rate(self) -> float:
        return self.n_successful / len(self.runs)

    @property
    def contrasts(self) -> tuple[SensitivityContrast, ...]:
        if self.reference.outcome is None:
            return ()
        reference = {metric.identity: metric for metric in self.reference.outcome.metrics}
        contrasts: list[SensitivityContrast] = []
        for run in self.runs:
            if run.scenario.is_reference or run.outcome is None:
                continue
            for metric in run.outcome.metrics:
                baseline = reference[metric.identity]
                contrasts.append(
                    SensitivityContrast(
                        scenario_name=run.scenario.name,
                        scenario_signature=run.scenario.signature,
                        metric_name=metric.name,
                        metric_signature=metric.signature,
                        target=metric.target,
                        coordinate=metric.coordinate,
                        unit=metric.unit,
                        reference_estimate=baseline.estimate,
                        estimate=metric.estimate,
                        difference=metric.estimate - baseline.estimate,
                        reference_interval=baseline.interval,
                        interval=metric.interval,
                    )
                )
        return tuple(contrasts)

    def summary(self) -> tuple[SensitivitySummary, ...]:
        """Describe scenario spread without assigning a universal robustness threshold."""

        if self.reference.outcome is None:
            return ()
        successful = tuple(run for run in self.runs if run.outcome is not None)
        by_run = {
            run.scenario.name: {metric.identity: metric for metric in run.outcome.metrics}
            for run in successful
            if run.outcome is not None
        }
        summaries = []
        for baseline in self.reference.outcome.metrics:
            selected = tuple(
                (name, metrics[baseline.identity].estimate) for name, metrics in by_run.items()
            )
            values = np.asarray([estimate for _, estimate in selected], dtype=np.float64)
            differences = np.abs(values - baseline.estimate)
            largest_difference = selected[int(np.argmax(differences))][0]
            summaries.append(
                SensitivitySummary(
                    metric_name=baseline.name,
                    metric_signature=baseline.signature,
                    target=baseline.target,
                    unit=baseline.unit,
                    n_scenarios=len(selected),
                    reference_estimate=baseline.estimate,
                    minimum=float(np.min(values)),
                    maximum=float(np.max(values)),
                    range=float(np.max(values) - np.min(values)),
                    maximum_absolute_difference=float(np.max(differences)),
                    largest_difference_scenario=largest_difference,
                )
            )
        return tuple(summaries)

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_signature": self.analysis_signature,
            "root_seed": self.root_seed,
            "reference_scenario": self.reference.scenario.name,
            "is_interpretable": self.is_interpretable,
            "n_successful": self.n_successful,
            "n_failed": self.n_failed,
            "success_rate": self.success_rate,
            "runs": [_run_dict(run) for run in self.runs],
            "contrasts": [_contrast_dict(item) for item in self.contrasts],
            "summary": [_summary_dict(item) for item in self.summary()],
        }


SensitivityAnalysis = Callable[[SensitivityScenario, int], SensitivityOutcome]


def run_sensitivity_analysis(
    scenarios: Sequence[SensitivityScenario],
    analysis: SensitivityAnalysis,
    *,
    seed: int,
    analysis_signature: str,
) -> SensitivityReport:
    """Refit all declared scenarios and retain every outcome or failure."""

    declared = tuple(scenarios)
    if not declared or any(not isinstance(item, SensitivityScenario) for item in declared):
        raise TypeError("scenarios must contain SensitivityScenario objects")
    if len({item.name for item in declared}) != len(declared):
        raise ValueError("sensitivity scenario names must be unique")
    if len({item.signature for item in declared}) != len(declared):
        raise ValueError("sensitivity scenario signatures must be unique")
    if sum(item.is_reference for item in declared) != 1:
        raise ValueError("sensitivity analysis requires exactly one reference scenario")
    if not callable(analysis):
        raise TypeError("sensitivity analysis callback must be callable")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("sensitivity seed must be a non-negative integer")
    if not isinstance(analysis_signature, str) or not analysis_signature:
        raise ValueError("analysis_signature must be a non-empty string")

    runs = []
    for scenario in declared:
        scenario_seed = _scenario_seed(seed, scenario.signature)
        try:
            outcome = analysis(scenario, scenario_seed)
        except Exception as error:
            run = SensitivityRun(
                scenario=scenario,
                seed=scenario_seed,
                failure=_failure(scenario, scenario_seed, "analysis", error),
            )
        else:
            if not isinstance(outcome, SensitivityOutcome):
                error = TypeError("sensitivity analysis must return SensitivityOutcome")
                run = SensitivityRun(
                    scenario=scenario,
                    seed=scenario_seed,
                    failure=_failure(scenario, scenario_seed, "evaluation", error),
                )
            else:
                run = SensitivityRun(scenario=scenario, seed=scenario_seed, outcome=outcome)
        runs.append(run)

    reference = next(run for run in runs if run.scenario.is_reference)
    if reference.outcome is not None:
        expected = _metric_descriptors(reference.outcome)
        for index, run in enumerate(runs):
            if run.outcome is None or run.scenario.is_reference:
                continue
            if _metric_descriptors(run.outcome) != expected:
                error = SensitivityError(
                    "sensitivity metric identities, names, units, or order changed "
                    "relative to the reference scenario"
                )
                runs[index] = SensitivityRun(
                    scenario=run.scenario,
                    seed=run.seed,
                    failure=_failure(run.scenario, run.seed, "evaluation", error),
                )

    return SensitivityReport(
        analysis_signature=analysis_signature,
        root_seed=seed,
        runs=tuple(runs),
    )


def posterior_sensitivity_outcome(
    result: PosteriorResult,
    *,
    variable_names: Sequence[str] | None = None,
    interval_probability: float = 0.9,
    artifact_signature: str | None = None,
    diagnostic_codes: Sequence[str] = (),
) -> SensitivityOutcome:
    """Summarize labelled natural posterior parameters for exact-refit sensitivity."""

    if not isinstance(result, PosteriorResult):
        raise TypeError("result must be PosteriorResult")
    if not np.isfinite(interval_probability) or not 0 < interval_probability < 1:
        raise ValueError("interval_probability must be finite and lie between zero and one")
    names = result.parameter_names if variable_names is None else tuple(variable_names)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("variable_names must contain non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError("variable_names must be unique")
    posterior = result["posterior"]
    missing = tuple(name for name in names if name not in posterior.variable_names)
    if missing:
        raise SensitivityError(f"posterior has no requested variables: {missing!r}")
    alpha = (1.0 - interval_probability) / 2.0
    metrics: list[SensitivityMetric] = []
    for name in names:
        variable = posterior[name]
        if variable.dims[:2] != ("chain", "draw"):
            raise SensitivityError("posterior sensitivity variables must lead with chain and draw")
        values = np.asarray(variable.values, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise SensitivityError(f"posterior sensitivity variable {name!r} must be finite")
        intrinsic_shape = values.shape[2:]
        draws = values.reshape(-1, *intrinsic_shape)
        coordinates = tuple(
            tuple(
                _json_scalar(value, label=f"coordinate {dim!r}") for value in variable.coords[dim]
            )
            for dim in variable.intrinsic_dims
        )
        for index in np.ndindex(intrinsic_shape):
            selected = draws[(slice(None), *index)]
            coordinate = tuple(
                (dim, coordinates[axis][position])
                for axis, (dim, position) in enumerate(
                    zip(variable.intrinsic_dims, index, strict=True)
                )
            )
            interval = np.quantile(selected, (alpha, 1.0 - alpha))
            metrics.append(
                SensitivityMetric(
                    name=name,
                    signature=(
                        f"posterior-mean[v1;variable={name};"
                        f"interval_probability={float(interval_probability)!r}]"
                    ),
                    estimate=float(np.mean(selected)),
                    coordinate=coordinate,
                    interval=(float(interval[0]), float(interval[1])),
                    unit="natural parameter",
                )
            )
    signature = artifact_signature or (
        f"posterior-result[v1;model={result.model_signature};"
        f"backend={result.inference_library}@{result.inference_library_version};"
        f"parameter-space={result.parameter_space_fingerprint or 'none'}]"
    )
    return SensitivityOutcome(
        artifact_signature=signature,
        metrics=tuple(metrics),
        diagnostic_codes=tuple(diagnostic_codes),
        provenance={
            "model_name": result.model_name,
            "model_signature": result.model_signature,
            "inference_library": result.inference_library,
            "inference_library_version": result.inference_library_version,
            "parameter_space_fingerprint": result.parameter_space_fingerprint,
        },
    )


def _metric_descriptors(
    outcome: SensitivityOutcome,
) -> tuple[tuple[tuple[str, tuple[tuple[str, Any], ...]], str, str | None], ...]:
    return tuple((metric.identity, metric.name, metric.unit) for metric in outcome.metrics)


def _scenario_seed(root_seed: int, signature: str) -> int:
    payload = f"unspool-sensitivity-v1\0{root_seed}\0{signature}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _failure(
    scenario: SensitivityScenario,
    seed: int,
    stage: str,
    error: Exception,
) -> SensitivityFailure:
    return SensitivityFailure(
        scenario_name=scenario.name,
        scenario_signature=scenario.signature,
        stage=stage,
        error_type=type(error).__name__,
        message=str(error) or "<exception had no message>",
        seed=seed,
    )


def _coordinate(values: Sequence[Sequence[Any]]) -> tuple[tuple[str, Any], ...]:
    coordinate: list[tuple[str, Any]] = []
    for item in values:
        if not isinstance(item, Sequence) or len(item) != 2:
            raise ValueError("sensitivity coordinates must contain dimension/value pairs")
        dim, value = item
        if not isinstance(dim, str) or not dim:
            raise ValueError("sensitivity coordinate dimensions must be non-empty strings")
        coordinate.append((dim, _json_scalar(value, label=f"coordinate {dim!r}")))
    if len({dim for dim, _ in coordinate}) != len(coordinate):
        raise ValueError("sensitivity coordinate dimensions must be unique")
    return tuple(coordinate)


def _target(name: str, coordinate: tuple[tuple[str, Any], ...]) -> str:
    if not coordinate:
        return name
    labels = ",".join(f"{dim}={value!r}" for dim, value in coordinate)
    return f"{name}[{labels}]"


def _interval(values: tuple[float, float] | None) -> tuple[float, float] | None:
    if values is None:
        return None
    if len(values) != 2 or not np.all(np.isfinite(values)):
        raise ValueError("sensitivity interval must contain two finite values")
    interval = (float(values[0]), float(values[1]))
    if interval[0] > interval[1]:
        raise ValueError("sensitivity interval must be ordered")
    return interval


def _json_scalar(value: Any, *, label: str) -> Any:
    scalar = value.item() if isinstance(value, np.generic) else value
    if scalar is None or isinstance(scalar, str | bool | int):
        return scalar
    if isinstance(scalar, float) and np.isfinite(scalar):
        return scalar
    raise ValueError(f"{label} must be a finite JSON scalar")


def _metric_dict(metric: SensitivityMetric) -> dict[str, Any]:
    return {
        "name": metric.name,
        "signature": metric.signature,
        "target": metric.target,
        "coordinate": dict(metric.coordinate),
        "estimate": metric.estimate,
        "interval": None if metric.interval is None else list(metric.interval),
        "unit": metric.unit,
    }


def _run_dict(run: SensitivityRun) -> dict[str, Any]:
    outcome = run.outcome
    failure = run.failure
    return {
        "scenario": {
            "name": run.scenario.name,
            "signature": run.scenario.signature,
            "changes": dict(run.scenario.changes),
            "is_reference": run.scenario.is_reference,
        },
        "seed": run.seed,
        "outcome": None
        if outcome is None
        else {
            "artifact_signature": outcome.artifact_signature,
            "metrics": [_metric_dict(metric) for metric in outcome.metrics],
            "diagnostic_codes": list(outcome.diagnostic_codes),
            "provenance": dict(outcome.provenance),
        },
        "failure": None
        if failure is None
        else {
            "stage": failure.stage,
            "error_type": failure.error_type,
            "message": failure.message,
        },
    }


def _contrast_dict(contrast: SensitivityContrast) -> dict[str, Any]:
    return {
        "scenario_name": contrast.scenario_name,
        "scenario_signature": contrast.scenario_signature,
        "metric_name": contrast.metric_name,
        "metric_signature": contrast.metric_signature,
        "target": contrast.target,
        "coordinate": dict(contrast.coordinate),
        "unit": contrast.unit,
        "reference_estimate": contrast.reference_estimate,
        "estimate": contrast.estimate,
        "difference": contrast.difference,
        "reference_interval": None
        if contrast.reference_interval is None
        else list(contrast.reference_interval),
        "interval": None if contrast.interval is None else list(contrast.interval),
    }


def _summary_dict(summary: SensitivitySummary) -> dict[str, Any]:
    return {
        "metric_name": summary.metric_name,
        "metric_signature": summary.metric_signature,
        "target": summary.target,
        "unit": summary.unit,
        "n_scenarios": summary.n_scenarios,
        "reference_estimate": summary.reference_estimate,
        "minimum": summary.minimum,
        "maximum": summary.maximum,
        "range": summary.range,
        "maximum_absolute_difference": summary.maximum_absolute_difference,
        "largest_difference_scenario": summary.largest_difference_scenario,
    }
