"""Prospective recovery experiments across competing model families."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from unspool.evaluation import FoldEvaluation, evaluate_splits
from unspool.models.base import BehaviourModel, _protected_array
from unspool.study import Study
from unspool.validation import forward_session_splits

UNRESOLVED_LABEL = "unresolved"


@dataclass(frozen=True, slots=True)
class ModelRecoveryScenario:
    """One named generative condition with an expected candidate label."""

    name: str
    truth_label: str
    generator: BehaviourModel
    parameters: Mapping[str, float]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("scenario name must be a non-empty string")
        if not isinstance(self.truth_label, str) or not self.truth_label:
            raise ValueError("truth_label must be a non-empty string")
        if not isinstance(self.generator, BehaviourModel):
            raise TypeError("generator must satisfy the BehaviourModel contract")
        expected = set(self.generator.parameter_names)
        observed = set(self.parameters)
        if observed != expected:
            raise ValueError(
                "scenario parameters must match the generator exactly; "
                f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
            )
        parameters = {name: float(self.parameters[name]) for name in self.generator.parameter_names}
        if not np.all(np.isfinite(tuple(parameters.values()))):
            raise ValueError("scenario parameters must be finite")
        object.__setattr__(self, "parameters", MappingProxyType(parameters))


@dataclass(frozen=True, slots=True)
class ModelRecoveryMatrix:
    """Counts and row-normalized selection rates by generating family."""

    truth_labels: tuple[str, ...]
    selected_labels: tuple[str, ...]
    counts: NDArray[np.int64]
    rates: NDArray[np.float64]

    def __post_init__(self) -> None:
        truth_labels = tuple(self.truth_labels)
        selected_labels = tuple(self.selected_labels)
        counts = _protected_array(self.counts, dtype=np.int64)
        rates = _protected_array(self.rates, dtype=np.float64)
        expected_shape = (len(truth_labels), len(selected_labels))
        if counts.shape != expected_shape or rates.shape != expected_shape:
            raise ValueError("recovery matrix dimensions must match its labels")
        if np.any(counts < 0):
            raise ValueError("recovery counts must be non-negative")
        object.__setattr__(self, "truth_labels", truth_labels)
        object.__setattr__(self, "selected_labels", selected_labels)
        object.__setattr__(self, "counts", counts)
        object.__setattr__(self, "rates", rates)


@dataclass(frozen=True, slots=True)
class ModelRecoveryReport:
    """Raw prospective scores and selections for a model-recovery experiment."""

    candidate_labels: tuple[str, ...]
    candidate_signatures: tuple[str, ...]
    scenario_names: tuple[str, ...]
    generator_signatures: tuple[str, ...]
    generator_parameters: tuple[Mapping[str, float], ...]
    truth_labels: tuple[str, ...]
    selected_labels: tuple[str | None, ...]
    mean_log_probabilities: NDArray[np.float64]
    converged: NDArray[np.bool_]
    failure_messages: tuple[tuple[str, ...], ...]
    seeds: NDArray[np.uint64]
    n_folds: NDArray[np.int64]
    n_trials: int
    n_subjects: int
    repeats: int
    root_seed: int
    min_train_sessions: int
    horizon: int
    step: int
    tie_tolerance: float

    def __post_init__(self) -> None:
        candidates = tuple(self.candidate_labels)
        candidate_signatures = tuple(self.candidate_signatures)
        scenario_names = tuple(self.scenario_names)
        generator_signatures = tuple(self.generator_signatures)
        generator_parameters = tuple(
            MappingProxyType({name: float(value) for name, value in parameters.items()})
            for parameters in self.generator_parameters
        )
        truth_labels = tuple(self.truth_labels)
        selected_labels = tuple(self.selected_labels)
        failure_messages = tuple(tuple(row) for row in self.failure_messages)
        scores = _protected_array(self.mean_log_probabilities, dtype=np.float64)
        converged = _protected_array(self.converged, dtype=np.bool_)
        seeds = _protected_array(self.seeds, dtype=np.uint64)
        n_folds = _protected_array(self.n_folds, dtype=np.int64)
        n_runs = len(scenario_names)
        expected_shape = (n_runs, len(candidates))

        if not candidates or len(set(candidates)) != len(candidates):
            raise ValueError("candidate labels must be non-empty and unique")
        if len(candidate_signatures) != len(candidates):
            raise ValueError("every candidate must retain its model signature")
        if not (
            len(generator_signatures)
            == len(generator_parameters)
            == len(truth_labels)
            == len(selected_labels)
            == len(failure_messages)
            == n_runs
        ):
            raise ValueError("run metadata must have one value per recovery run")
        if any(
            not np.all(np.isfinite(tuple(parameters.values())))
            for parameters in generator_parameters
        ):
            raise ValueError("generator parameters must be finite")
        if scores.shape != expected_shape or converged.shape != expected_shape:
            raise ValueError("candidate results must have one column per candidate")
        if seeds.shape != (n_runs,) or n_folds.shape != (n_runs,):
            raise ValueError("run arrays must have one value per recovery run")
        if not np.all(np.isfinite(scores)):
            raise ValueError("candidate scores must be finite")
        if any(len(row) != len(candidates) for row in failure_messages):
            raise ValueError("failure messages must have one column per candidate")
        if any(label not in candidates for label in truth_labels):
            raise ValueError("every truth label must name a candidate")
        if any(label is not None and label not in candidates for label in selected_labels):
            raise ValueError("selected labels must name a candidate or be unresolved")
        if np.any(n_folds < 1):
            raise ValueError("every recovery run must contain at least one fold")
        if self.n_trials < 1 or self.n_subjects < 1 or self.repeats < 1:
            raise ValueError("design counts and repeats must be positive")

        object.__setattr__(self, "candidate_labels", candidates)
        object.__setattr__(self, "candidate_signatures", candidate_signatures)
        object.__setattr__(self, "scenario_names", scenario_names)
        object.__setattr__(self, "generator_signatures", generator_signatures)
        object.__setattr__(self, "generator_parameters", generator_parameters)
        object.__setattr__(self, "truth_labels", truth_labels)
        object.__setattr__(self, "selected_labels", selected_labels)
        object.__setattr__(self, "mean_log_probabilities", scores)
        object.__setattr__(self, "converged", converged)
        object.__setattr__(self, "failure_messages", failure_messages)
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "n_folds", n_folds)

    @property
    def n_runs(self) -> int:
        return len(self.scenario_names)

    @property
    def resolution_rate(self) -> float:
        return float(np.mean([label is not None for label in self.selected_labels]))

    @property
    def overall_accuracy(self) -> float:
        correct = sum(
            selected == truth
            for selected, truth in zip(self.selected_labels, self.truth_labels, strict=True)
        )
        return correct / self.n_runs

    @property
    def resolved_accuracy(self) -> float:
        resolved = [
            selected == truth
            for selected, truth in zip(self.selected_labels, self.truth_labels, strict=True)
            if selected is not None
        ]
        return float(np.mean(resolved)) if resolved else float("nan")

    def confusion_matrix(self) -> ModelRecoveryMatrix:
        """Return candidate selections plus an explicit unresolved column."""

        selected_columns = (*self.candidate_labels, UNRESOLVED_LABEL)
        counts = np.zeros((len(self.candidate_labels), len(selected_columns)), dtype=np.int64)
        truth_index = {label: index for index, label in enumerate(self.candidate_labels)}
        selected_index = {label: index for index, label in enumerate(selected_columns)}
        for truth, selected in zip(self.truth_labels, self.selected_labels, strict=True):
            selected_key = UNRESOLVED_LABEL if selected is None else selected
            counts[truth_index[truth], selected_index[selected_key]] += 1
        row_totals = counts.sum(axis=1, keepdims=True)
        rates = np.full(counts.shape, np.nan, dtype=np.float64)
        np.divide(counts, row_totals, out=rates, where=row_totals > 0)
        return ModelRecoveryMatrix(
            truth_labels=self.candidate_labels,
            selected_labels=selected_columns,
            counts=counts,
            rates=rates,
        )


def run_model_recovery(
    design: Study,
    scenarios: Sequence[ModelRecoveryScenario],
    candidates: Mapping[str, BehaviourModel],
    *,
    repeats: int = 1,
    seed: int,
    min_train_sessions: int = 1,
    horizon: int = 1,
    step: int = 1,
    tie_tolerance: float = 1e-8,
) -> ModelRecoveryReport:
    """Simulate scenarios and select candidates by prospective mean log probability."""

    scenarios = tuple(scenarios)
    if not scenarios:
        raise ValueError("scenarios must not be empty")
    if len({scenario.name for scenario in scenarios}) != len(scenarios):
        raise ValueError("scenario names must be unique")
    candidate_models = _validated_candidates(candidates)
    candidate_labels = tuple(candidate_models)
    if any(scenario.truth_label not in candidate_models for scenario in scenarios):
        raise ValueError("every scenario truth_label must name a candidate")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    _require_positive_integer(min_train_sessions, "min_train_sessions")
    _require_positive_integer(horizon, "horizon")
    _require_positive_integer(step, "step")
    if not np.isfinite(tie_tolerance) or tie_tolerance < 0:
        raise ValueError("tie_tolerance must be finite and non-negative")

    n_runs = len(scenarios) * repeats
    child_sequences = np.random.SeedSequence(seed).spawn(n_runs)
    scores = np.empty((n_runs, len(candidate_labels)), dtype=np.float64)
    converged = np.empty((n_runs, len(candidate_labels)), dtype=np.bool_)
    seeds = np.empty(n_runs, dtype=np.uint64)
    n_folds = np.empty(n_runs, dtype=np.int64)
    scenario_names: list[str] = []
    generator_signatures: list[str] = []
    generator_parameters: list[Mapping[str, float]] = []
    truth_labels: list[str] = []
    selected_labels: list[str | None] = []
    failure_messages: list[tuple[str, ...]] = []

    run = 0
    for scenario in scenarios:
        for _ in range(repeats):
            child_seed = int(child_sequences[run].generate_state(1, dtype=np.uint64)[0])
            simulated = scenario.generator.simulate(design, scenario.parameters, seed=child_seed)
            splits = forward_session_splits(
                simulated,
                min_train_sessions=min_train_sessions,
                horizon=horizon,
                step=step,
            )
            if not splits:
                raise ValueError(
                    f"scenario {scenario.name!r} produced no eligible prospective folds"
                )
            n_folds[run] = len(splits)
            run_failures: list[str] = []
            for column, model in enumerate(candidate_models.values()):
                evaluations = evaluate_splits(model, simulated, splits)
                pointwise = np.concatenate(
                    [evaluation.pointwise_log_probability for evaluation in evaluations]
                )
                scores[run, column] = float(np.mean(pointwise))
                candidate_converged = all(
                    evaluation.fit.diagnostics.converged for evaluation in evaluations
                )
                converged[run, column] = candidate_converged
                run_failures.append(_failure_message(evaluations))

            selected_labels.append(
                _select_candidate(scores[run], converged[run], candidate_labels, tie_tolerance)
            )
            seeds[run] = child_seed
            scenario_names.append(scenario.name)
            generator_signatures.append(scenario.generator.signature)
            generator_parameters.append(scenario.parameters)
            truth_labels.append(scenario.truth_label)
            failure_messages.append(tuple(run_failures))
            run += 1

    return ModelRecoveryReport(
        candidate_labels=candidate_labels,
        candidate_signatures=tuple(model.signature for model in candidate_models.values()),
        scenario_names=tuple(scenario_names),
        generator_signatures=tuple(generator_signatures),
        generator_parameters=tuple(generator_parameters),
        truth_labels=tuple(truth_labels),
        selected_labels=tuple(selected_labels),
        mean_log_probabilities=scores,
        converged=converged,
        failure_messages=tuple(failure_messages),
        seeds=seeds,
        n_folds=n_folds,
        n_trials=len(design),
        n_subjects=len(design.subjects),
        repeats=repeats,
        root_seed=seed,
        min_train_sessions=min_train_sessions,
        horizon=horizon,
        step=step,
        tie_tolerance=tie_tolerance,
    )


def _validated_candidates(
    candidates: Mapping[str, BehaviourModel],
) -> Mapping[str, BehaviourModel]:
    if not isinstance(candidates, Mapping) or not candidates:
        raise ValueError("candidates must be a non-empty mapping")
    validated: dict[str, BehaviourModel] = {}
    for label, model in candidates.items():
        if not isinstance(label, str) or not label or label == UNRESOLVED_LABEL:
            raise ValueError(
                f"candidate labels must be non-empty and cannot be {UNRESOLVED_LABEL!r}"
            )
        if not isinstance(model, BehaviourModel):
            raise TypeError(f"candidate {label!r} does not satisfy the BehaviourModel contract")
        validated[label] = model
    return MappingProxyType(validated)


def _select_candidate(
    scores: NDArray[np.float64],
    converged: NDArray[np.bool_],
    labels: tuple[str, ...],
    tie_tolerance: float,
) -> str | None:
    valid = np.flatnonzero(converged & np.isfinite(scores))
    if not valid.size:
        return None
    ordered = valid[np.argsort(scores[valid])[::-1]]
    if len(ordered) > 1 and scores[ordered[0]] - scores[ordered[1]] <= tie_tolerance:
        return None
    return labels[int(ordered[0])]


def _failure_message(evaluations: tuple[FoldEvaluation, ...]) -> str:
    failures: list[str] = []
    for fold, evaluation in enumerate(evaluations):
        diagnostics = evaluation.fit.diagnostics
        if not diagnostics.converged:
            failures.append(f"fold {fold}: {diagnostics.message}")
    return "; ".join(failures)


def _require_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
