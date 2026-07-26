"""Design-specific parameter-recovery experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from unspool.diagnostics import FitAudit, FitAuditPolicy, FitAuditStatus
from unspool.models.base import (
    FitResult,
    GenerativeBehaviourModel,
    _protected_array,
    model_capabilities,
)
from unspool.study import Study


@dataclass(frozen=True, slots=True)
class ParameterRecoverySummary:
    """Recovery metrics for one parameter across audit-eligible fits."""

    parameter: str
    bias: float
    rmse: float
    correlation: float
    coverage_95: float
    n_successful: int
    n_with_uncertainty: int


@dataclass(frozen=True, slots=True)
class ParameterRecoveryReport:
    """Raw results and summaries tied to a particular study design."""

    model_name: str
    model_signature: str
    parameter_names: tuple[str, ...]
    true_values: NDArray[np.float64]
    estimates: NDArray[np.float64]
    standard_errors: NDArray[np.float64]
    converged: NDArray[np.bool_]
    messages: tuple[str, ...]
    audits: tuple[FitAudit, ...]
    seeds: NDArray[np.uint64]
    n_trials: int
    n_subjects: int
    repeats: int
    root_seed: int

    def __post_init__(self) -> None:
        names = tuple(self.parameter_names)
        true_values = _protected_array(self.true_values, dtype=np.float64)
        estimates = _protected_array(self.estimates, dtype=np.float64)
        standard_errors = _protected_array(self.standard_errors, dtype=np.float64)
        converged = _protected_array(self.converged, dtype=np.bool_)
        seeds = _protected_array(self.seeds, dtype=np.uint64)
        audits = tuple(self.audits)
        expected_shape = (len(self.messages), len(names))
        if not names or len(set(names)) != len(names):
            raise ValueError("parameter_names must be non-empty and unique")
        if true_values.shape != expected_shape:
            raise ValueError("true_values must have one row per recovery run")
        if estimates.shape != expected_shape or standard_errors.shape != expected_shape:
            raise ValueError("estimate arrays must match true_values")
        if converged.shape != (len(self.messages),) or seeds.shape != (len(self.messages),):
            raise ValueError("run metadata must have one value per recovery run")
        if len(audits) != len(self.messages):
            raise ValueError("every recovery run must retain one fit audit")
        if any(
            audit.model_name != self.model_name or audit.model_signature != self.model_signature
            for audit in audits
        ):
            raise ValueError("recovery audits must match the report model")
        if self.n_trials < 1 or self.n_subjects < 1 or self.repeats < 1:
            raise ValueError("design counts and repeats must be positive")
        object.__setattr__(self, "parameter_names", names)
        object.__setattr__(self, "true_values", true_values)
        object.__setattr__(self, "estimates", estimates)
        object.__setattr__(self, "standard_errors", standard_errors)
        object.__setattr__(self, "converged", converged)
        object.__setattr__(self, "messages", tuple(self.messages))
        object.__setattr__(self, "audits", audits)
        object.__setattr__(self, "seeds", seeds)

    @property
    def n_runs(self) -> int:
        return len(self.messages)

    @property
    def convergence_rate(self) -> float:
        return float(np.mean(self.converged))

    @property
    def audit_pass_rate(self) -> float:
        return float(np.mean([audit.status is FitAuditStatus.PASS for audit in self.audits]))

    @property
    def audit_warning_rate(self) -> float:
        return float(np.mean([audit.status is FitAuditStatus.WARNING for audit in self.audits]))

    @property
    def audit_failure_rate(self) -> float:
        return float(np.mean([audit.status is FitAuditStatus.FAIL for audit in self.audits]))

    def summary(self) -> tuple[ParameterRecoverySummary, ...]:
        """Summarize bias, RMSE, association, and Wald-interval coverage.

        Warnings remain eligible; failed audits do not enter estimation summaries. Coverage
        uses only eligible runs with finite, non-negative standard errors and reports that
        denominator separately.
        """

        summaries: list[ParameterRecoverySummary] = []
        audit_eligible = np.asarray(
            [audit.status is not FitAuditStatus.FAIL for audit in self.audits],
            dtype=np.bool_,
        )
        for column, name in enumerate(self.parameter_names):
            valid = audit_eligible & np.isfinite(self.estimates[:, column])
            n_successful = int(np.sum(valid))
            if n_successful:
                truth = self.true_values[valid, column]
                estimate = self.estimates[valid, column]
                error = estimate - truth
                bias = float(np.mean(error))
                rmse = float(np.sqrt(np.mean(error**2)))
            else:
                truth = np.array([], dtype=np.float64)
                estimate = np.array([], dtype=np.float64)
                bias = rmse = float("nan")
            uncertainty_valid = valid & np.isfinite(self.standard_errors[:, column])
            uncertainty_valid &= self.standard_errors[:, column] >= 0
            n_with_uncertainty = int(np.sum(uncertainty_valid))
            if n_with_uncertainty:
                uncertainty_error = (
                    self.estimates[uncertainty_valid, column]
                    - self.true_values[uncertainty_valid, column]
                )
                covered = np.abs(uncertainty_error) <= (
                    1.959963984540054 * self.standard_errors[uncertainty_valid, column]
                )
                coverage = float(np.mean(covered))
            else:
                coverage = float("nan")
            if n_successful >= 2 and np.std(truth) > 0 and np.std(estimate) > 0:
                correlation = float(np.corrcoef(truth, estimate)[0, 1])
            else:
                correlation = float("nan")
            summaries.append(
                ParameterRecoverySummary(
                    parameter=name,
                    bias=bias,
                    rmse=rmse,
                    correlation=correlation,
                    coverage_95=coverage,
                    n_successful=n_successful,
                    n_with_uncertainty=n_with_uncertainty,
                )
            )
        return tuple(summaries)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable record of every run and summary denominator."""

        summaries = self.summary()
        return {
            "model_name": self.model_name,
            "model_signature": self.model_signature,
            "parameter_names": list(self.parameter_names),
            "design": {
                "n_trials": self.n_trials,
                "n_subjects": self.n_subjects,
            },
            "repeats": self.repeats,
            "root_seed": self.root_seed,
            "n_runs": self.n_runs,
            "convergence_rate": self.convergence_rate,
            "audit_rates": {
                "pass": self.audit_pass_rate,
                "warning": self.audit_warning_rate,
                "fail": self.audit_failure_rate,
            },
            "summary": [
                {
                    "parameter": summary.parameter,
                    "bias": _json_float(summary.bias),
                    "rmse": _json_float(summary.rmse),
                    "correlation": _json_float(summary.correlation),
                    "coverage_95": _json_float(summary.coverage_95),
                    "n_successful": summary.n_successful,
                    "n_with_uncertainty": summary.n_with_uncertainty,
                }
                for summary in summaries
            ],
            "runs": [
                {
                    "seed": int(self.seeds[index]),
                    "truth": {
                        name: float(self.true_values[index, column])
                        for column, name in enumerate(self.parameter_names)
                    },
                    "estimate": {
                        name: _json_float(self.estimates[index, column])
                        for column, name in enumerate(self.parameter_names)
                    },
                    "standard_error": {
                        name: _json_float(self.standard_errors[index, column])
                        for column, name in enumerate(self.parameter_names)
                    },
                    "converged": bool(self.converged[index]),
                    "message": self.messages[index],
                    "fit_audit": _json_safe(self.audits[index].to_dict()),
                }
                for index in range(self.n_runs)
            ],
        }


def run_parameter_recovery(
    model: GenerativeBehaviourModel,
    design: Study,
    parameter_sets: Sequence[Mapping[str, float]],
    *,
    repeats: int = 1,
    seed: int,
    audit_policy: FitAuditPolicy | None = None,
) -> ParameterRecoveryReport:
    """Simulate and refit explicit parameter sets under one observed design.

    Every parameter set is repeated ``repeats`` times. The report retains the design size,
    random seeds, convergence flags, and optimizer messages so recovery is never detached
    from the conditions under which it was assessed.
    """

    if not isinstance(model, GenerativeBehaviourModel):
        raise TypeError("model must satisfy the GenerativeBehaviourModel contract")
    model_capabilities(model)
    parameter_sets = tuple(parameter_sets)
    if not parameter_sets:
        raise ValueError("parameter_sets must not be empty")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if audit_policy is not None and not isinstance(audit_policy, FitAuditPolicy):
        raise TypeError("audit_policy must be a FitAuditPolicy")

    validated_parameters: list[dict[str, float]] = []
    expected = set(model.parameter_names)
    for index, parameters in enumerate(parameter_sets):
        if not isinstance(parameters, Mapping):
            raise TypeError(f"parameter set {index} must be a mapping")
        observed = set(parameters)
        if observed != expected:
            raise ValueError(
                f"parameter set {index} must match the model exactly; "
                f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
            )
        try:
            values = {name: float(parameters[name]) for name in model.parameter_names}
        except (TypeError, ValueError):
            raise ValueError(f"parameter set {index} must contain finite numeric values") from None
        if not np.all(np.isfinite(tuple(values.values()))):
            raise ValueError(f"parameter set {index} must contain finite numeric values")
        validated_parameters.append(values)

    n_runs = len(parameter_sets) * repeats
    child_sequences = np.random.SeedSequence(seed).spawn(n_runs)
    true_values = np.empty((n_runs, len(model.parameter_names)), dtype=np.float64)
    estimates = np.empty_like(true_values)
    standard_errors = np.empty_like(true_values)
    converged = np.empty(n_runs, dtype=np.bool_)
    seeds = np.empty(n_runs, dtype=np.uint64)
    messages: list[str] = []
    audits: list[FitAudit] = []

    run = 0
    for parameters in validated_parameters:
        for _ in range(repeats):
            child_seed = int(child_sequences[run].generate_state(1, dtype=np.uint64)[0])
            simulated = model.simulate(design, parameters, seed=child_seed)
            fit = model.fit(simulated)
            if not isinstance(fit, FitResult):
                raise TypeError("model.fit must return a FitResult")
            if (
                fit.model_name != model.model_name
                or fit.model_signature != model.signature
                or fit.parameter_names != model.parameter_names
            ):
                raise ValueError("fit result does not match the recovery model")
            if fit.n_observations != len(simulated):
                raise ValueError("fit result n_observations must equal the simulated-study length")
            audit = fit.audit(policy=audit_policy)
            true_values[run] = [parameters[name] for name in model.parameter_names]
            estimates[run] = fit.estimates
            standard_errors[run] = fit.standard_errors
            converged[run] = fit.diagnostics.converged
            seeds[run] = child_seed
            messages.append(fit.diagnostics.message)
            audits.append(audit)
            run += 1

    return ParameterRecoveryReport(
        model_name=model.model_name,
        model_signature=model.signature,
        parameter_names=model.parameter_names,
        true_values=true_values,
        estimates=estimates,
        standard_errors=standard_errors,
        converged=converged,
        messages=tuple(messages),
        audits=tuple(audits),
        seeds=seeds,
        n_trials=len(design),
        n_subjects=len(design.subjects),
        repeats=repeats,
        root_seed=seed,
    )


def _json_float(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return _json_float(float(value))
    return value
