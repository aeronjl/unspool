"""Design-specific parameter-recovery experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from unspool.models.base import BehaviourModel, _protected_array
from unspool.study import Study


@dataclass(frozen=True, slots=True)
class ParameterRecoverySummary:
    """Recovery metrics for one parameter across successful fits."""

    parameter: str
    bias: float
    rmse: float
    correlation: float
    coverage_95: float
    n_successful: int


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
        expected_shape = (len(self.messages), len(names))
        if not names or len(set(names)) != len(names):
            raise ValueError("parameter_names must be non-empty and unique")
        if true_values.shape != expected_shape:
            raise ValueError("true_values must have one row per recovery run")
        if estimates.shape != expected_shape or standard_errors.shape != expected_shape:
            raise ValueError("estimate arrays must match true_values")
        if converged.shape != (len(self.messages),) or seeds.shape != (len(self.messages),):
            raise ValueError("run metadata must have one value per recovery run")
        if self.n_trials < 1 or self.n_subjects < 1 or self.repeats < 1:
            raise ValueError("design counts and repeats must be positive")
        object.__setattr__(self, "parameter_names", names)
        object.__setattr__(self, "true_values", true_values)
        object.__setattr__(self, "estimates", estimates)
        object.__setattr__(self, "standard_errors", standard_errors)
        object.__setattr__(self, "converged", converged)
        object.__setattr__(self, "messages", tuple(self.messages))
        object.__setattr__(self, "seeds", seeds)

    @property
    def n_runs(self) -> int:
        return len(self.messages)

    @property
    def convergence_rate(self) -> float:
        return float(np.mean(self.converged))

    def summary(self) -> tuple[ParameterRecoverySummary, ...]:
        """Summarize bias, RMSE, association, and Wald-interval coverage."""

        summaries: list[ParameterRecoverySummary] = []
        for column, name in enumerate(self.parameter_names):
            valid = self.converged & np.isfinite(self.estimates[:, column])
            n_successful = int(np.sum(valid))
            if n_successful:
                truth = self.true_values[valid, column]
                estimate = self.estimates[valid, column]
                standard_error = self.standard_errors[valid, column]
                error = estimate - truth
                bias = float(np.mean(error))
                rmse = float(np.sqrt(np.mean(error**2)))
                covered = np.abs(error) <= 1.959963984540054 * standard_error
                coverage = float(np.mean(covered))
            else:
                truth = np.array([], dtype=np.float64)
                estimate = np.array([], dtype=np.float64)
                bias = rmse = coverage = float("nan")
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
                )
            )
        return tuple(summaries)


def run_parameter_recovery(
    model: BehaviourModel,
    design: Study,
    parameter_sets: Sequence[Mapping[str, float]],
    *,
    repeats: int = 1,
    seed: int,
) -> ParameterRecoveryReport:
    """Simulate and refit explicit parameter sets under one observed design.

    Every parameter set is repeated ``repeats`` times. The report retains the design size,
    random seeds, convergence flags, and optimizer messages so recovery is never detached
    from the conditions under which it was assessed.
    """

    parameter_sets = tuple(parameter_sets)
    if not parameter_sets:
        raise ValueError("parameter_sets must not be empty")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    n_runs = len(parameter_sets) * repeats
    child_sequences = np.random.SeedSequence(seed).spawn(n_runs)
    true_values = np.empty((n_runs, len(model.parameter_names)), dtype=np.float64)
    estimates = np.empty_like(true_values)
    standard_errors = np.empty_like(true_values)
    converged = np.empty(n_runs, dtype=np.bool_)
    seeds = np.empty(n_runs, dtype=np.uint64)
    messages: list[str] = []

    run = 0
    for parameters in parameter_sets:
        for _ in range(repeats):
            child_seed = int(child_sequences[run].generate_state(1, dtype=np.uint64)[0])
            simulated = model.simulate(design, parameters, seed=child_seed)
            fit = model.fit(simulated)
            true_values[run] = [parameters[name] for name in model.parameter_names]
            estimates[run] = fit.estimates
            standard_errors[run] = fit.standard_errors
            converged[run] = fit.diagnostics.converged
            seeds[run] = child_seed
            messages.append(fit.diagnostics.message)
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
        seeds=seeds,
        n_trials=len(design),
        n_subjects=len(design.subjects),
        repeats=repeats,
        root_seed=seed,
    )
