"""Prospective recovery experiments across competing model families.

Generators and candidates may be frequentist or sampled. A sampled candidate is driven
through :meth:`~behavio.contracts.posterior.PosteriorBehaviourEstimator.sample` and its
posterior convergence audit becomes the fold's ``converged`` flag, so a candidate whose
posterior failed cannot win a recovery run: ``_aggregate_audits`` already reports ``FAIL``
for it and ``_select_candidate`` only considers numerically usable columns.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from behavio._internal.arrays import protected_array
from behavio._internal.parallel import WorkerBackend, map_ordered, resolve_workers
from behavio.contracts.posterior import (
    AnyBehaviourEstimator,
    AnyGenerativeBehaviourModel,
    GenerativePosteriorBehaviourModel,
    any_model_capabilities,
    is_posterior_estimator,
)
from behavio.diagnostics import FitAuditStatus
from behavio.evaluate.folds import FoldEvaluation, PosteriorFoldPolicy, evaluate_splits
from behavio.evaluate.splits import EvaluationFold, forward_session_splits
from behavio.models.base import (
    BehaviourEstimator,
    GenerativeBehaviourModel,
)
from behavio.trials import Study

UNRESOLVED_LABEL = "unresolved"


@dataclass(frozen=True, slots=True)
class ModelRecoveryScenario:
    """One named generative condition with an expected candidate label."""

    name: str
    truth_label: str
    generator: AnyGenerativeBehaviourModel
    parameters: Mapping[str, float]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("scenario name must be a non-empty string")
        if not isinstance(self.truth_label, str) or not self.truth_label:
            raise ValueError("truth_label must be a non-empty string")
        if not isinstance(
            self.generator, (GenerativeBehaviourModel, GenerativePosteriorBehaviourModel)
        ):
            raise TypeError(
                "generator must satisfy the GenerativeBehaviourModel or "
                "GenerativePosteriorBehaviourModel contract"
            )
        any_model_capabilities(self.generator)
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

    def __getstate__(self) -> dict[str, object]:
        """Send the parameters as a plain mapping so a scenario reaches a worker process.

        ``parameters`` is wrapped in a :class:`~types.MappingProxyType` to make the frozen
        scenario deeply immutable, and a mapping proxy cannot be pickled. The proxy is a
        storage detail rather than part of the declaration, so it is unwrapped on the way
        out and re-established on the way in.
        """

        return {
            "name": self.name,
            "truth_label": self.truth_label,
            "generator": self.generator,
            "parameters": dict(self.parameters),
        }

    def __setstate__(self, state: Mapping[str, object]) -> None:
        object.__setattr__(self, "name", state["name"])
        object.__setattr__(self, "truth_label", state["truth_label"])
        object.__setattr__(self, "generator", state["generator"])
        parameters = dict(state["parameters"])  # type: ignore[arg-type]
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
        counts = protected_array(self.counts, dtype=np.int64)
        rates = protected_array(self.rates, dtype=np.float64)
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
class ModelRecoveryScenarioMatrix:
    """Counts and rates by named parameter regime rather than model family."""

    scenario_names: tuple[str, ...]
    truth_labels: tuple[str, ...]
    selected_labels: tuple[str, ...]
    counts: NDArray[np.int64]
    rates: NDArray[np.float64]

    def __post_init__(self) -> None:
        scenario_names = tuple(self.scenario_names)
        truth_labels = tuple(self.truth_labels)
        selected_labels = tuple(self.selected_labels)
        counts = protected_array(self.counts, dtype=np.int64)
        rates = protected_array(self.rates, dtype=np.float64)
        expected_shape = (len(scenario_names), len(selected_labels))
        if not scenario_names or len(set(scenario_names)) != len(scenario_names):
            raise ValueError("scenario names must be non-empty and unique")
        if len(truth_labels) != len(scenario_names):
            raise ValueError("every scenario row must retain its truth label")
        if counts.shape != expected_shape or rates.shape != expected_shape:
            raise ValueError("scenario matrix dimensions must match its labels")
        if np.any(counts < 0):
            raise ValueError("scenario recovery counts must be non-negative")
        object.__setattr__(self, "scenario_names", scenario_names)
        object.__setattr__(self, "truth_labels", truth_labels)
        object.__setattr__(self, "selected_labels", selected_labels)
        object.__setattr__(self, "counts", counts)
        object.__setattr__(self, "rates", rates)


@dataclass(frozen=True, slots=True)
class ModelRecoveryReport:
    """Raw prospective scores and selections for a model-recovery experiment."""

    candidate_labels: tuple[str, ...]
    candidate_signatures: tuple[str, ...]
    scored_columns: tuple[str, ...]
    scenario_names: tuple[str, ...]
    generator_signatures: tuple[str, ...]
    generator_parameters: tuple[Mapping[str, float], ...]
    truth_labels: tuple[str, ...]
    selected_labels: tuple[str | None, ...]
    mean_log_probabilities: NDArray[np.float64]
    converged: NDArray[np.bool_]
    failure_messages: tuple[tuple[str, ...], ...]
    audit_statuses: tuple[tuple[FitAuditStatus, ...], ...]
    audit_issue_codes: tuple[tuple[tuple[str, ...], ...], ...]
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
    validation_scheme: str
    aggregation_column: str | None
    sampled_candidate_labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        candidates = tuple(self.candidate_labels)
        sampled = tuple(self.sampled_candidate_labels)
        candidate_signatures = tuple(self.candidate_signatures)
        scored_columns = tuple(self.scored_columns)
        scenario_names = tuple(self.scenario_names)
        generator_signatures = tuple(self.generator_signatures)
        generator_parameters = tuple(
            MappingProxyType({name: float(value) for name, value in parameters.items()})
            for parameters in self.generator_parameters
        )
        truth_labels = tuple(self.truth_labels)
        selected_labels = tuple(self.selected_labels)
        failure_messages = tuple(tuple(row) for row in self.failure_messages)
        audit_statuses = tuple(
            tuple(FitAuditStatus(status) for status in row) for row in self.audit_statuses
        )
        audit_issue_codes = tuple(
            tuple(tuple(codes) for codes in row) for row in self.audit_issue_codes
        )
        scores = protected_array(self.mean_log_probabilities, dtype=np.float64)
        converged = protected_array(self.converged, dtype=np.bool_)
        seeds = protected_array(self.seeds, dtype=np.uint64)
        n_folds = protected_array(self.n_folds, dtype=np.int64)
        n_runs = len(scenario_names)
        expected_shape = (n_runs, len(candidates))

        if not candidates or len(set(candidates)) != len(candidates):
            raise ValueError("candidate labels must be non-empty and unique")
        if len(candidate_signatures) != len(candidates):
            raise ValueError("every candidate must retain its model signature")
        if len(set(sampled)) != len(sampled) or any(label not in candidates for label in sampled):
            raise ValueError("sampled_candidate_labels must name distinct declared candidates")
        if not scored_columns or len(set(scored_columns)) != len(scored_columns):
            raise ValueError("scored_columns must be non-empty and unique")
        if not (
            len(generator_signatures)
            == len(generator_parameters)
            == len(truth_labels)
            == len(selected_labels)
            == len(failure_messages)
            == len(audit_statuses)
            == len(audit_issue_codes)
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
        if any(len(row) != len(candidates) for row in audit_statuses):
            raise ValueError("audit statuses must have one column per candidate")
        if any(len(row) != len(candidates) for row in audit_issue_codes):
            raise ValueError("audit issue codes must have one column per candidate")
        if any(len(set(codes)) != len(codes) for row in audit_issue_codes for codes in row):
            raise ValueError("audit issue codes must be unique within each candidate run")
        if any(label not in candidates for label in truth_labels):
            raise ValueError("every truth label must name a candidate")
        if any(label is not None and label not in candidates for label in selected_labels):
            raise ValueError("selected labels must name a candidate or be unresolved")
        if np.any(n_folds < 1):
            raise ValueError("every recovery run must contain at least one fold")
        if self.n_trials < 1 or self.n_subjects < 1 or self.repeats < 1:
            raise ValueError("design counts and repeats must be positive")
        if not isinstance(self.validation_scheme, str) or not self.validation_scheme:
            raise ValueError("validation_scheme must be a non-empty string")
        if self.aggregation_column is not None and (
            not isinstance(self.aggregation_column, str) or not self.aggregation_column
        ):
            raise ValueError("aggregation_column must be None or a non-empty string")

        object.__setattr__(self, "candidate_labels", candidates)
        object.__setattr__(self, "candidate_signatures", candidate_signatures)
        object.__setattr__(self, "sampled_candidate_labels", sampled)
        object.__setattr__(self, "scored_columns", scored_columns)
        object.__setattr__(self, "scenario_names", scenario_names)
        object.__setattr__(self, "generator_signatures", generator_signatures)
        object.__setattr__(self, "generator_parameters", generator_parameters)
        object.__setattr__(self, "truth_labels", truth_labels)
        object.__setattr__(self, "selected_labels", selected_labels)
        object.__setattr__(self, "mean_log_probabilities", scores)
        object.__setattr__(self, "converged", converged)
        object.__setattr__(self, "failure_messages", failure_messages)
        object.__setattr__(self, "audit_statuses", audit_statuses)
        object.__setattr__(self, "audit_issue_codes", audit_issue_codes)
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

    @property
    def audit_warning_rate(self) -> float:
        """Fraction of candidate-run cells with at least one non-failing warning."""

        statuses = [status for row in self.audit_statuses for status in row]
        return float(np.mean([status is FitAuditStatus.WARNING for status in statuses]))

    @property
    def audit_failure_rate(self) -> float:
        """Fraction of candidate-run cells with a failing numerical audit."""

        statuses = [status for row in self.audit_statuses for status in row]
        return float(np.mean([status is FitAuditStatus.FAIL for status in statuses]))

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

    def scenario_confusion_matrix(self) -> ModelRecoveryScenarioMatrix:
        """Return selections separately for every named parameter regime."""

        scenario_names = tuple(dict.fromkeys(self.scenario_names))
        selected_columns = (*self.candidate_labels, UNRESOLVED_LABEL)
        scenario_index = {name: index for index, name in enumerate(scenario_names)}
        selected_index = {label: index for index, label in enumerate(selected_columns)}
        truth_by_scenario: dict[str, str] = {}
        counts = np.zeros((len(scenario_names), len(selected_columns)), dtype=np.int64)
        for scenario, truth, selected in zip(
            self.scenario_names,
            self.truth_labels,
            self.selected_labels,
            strict=True,
        ):
            previous_truth = truth_by_scenario.setdefault(scenario, truth)
            if previous_truth != truth:
                raise ValueError("a named scenario cannot map to multiple truth labels")
            selected_key = UNRESOLVED_LABEL if selected is None else selected
            counts[scenario_index[scenario], selected_index[selected_key]] += 1
        row_totals = counts.sum(axis=1, keepdims=True)
        rates = np.full(counts.shape, np.nan, dtype=np.float64)
        np.divide(counts, row_totals, out=rates, where=row_totals > 0)
        return ModelRecoveryScenarioMatrix(
            scenario_names=scenario_names,
            truth_labels=tuple(truth_by_scenario[name] for name in scenario_names),
            selected_labels=selected_columns,
            counts=counts,
            rates=rates,
        )


@dataclass(frozen=True, slots=True)
class ModelRecoveryGridSummary:
    """Comparable recovery and audit rates for one named design cell."""

    design_name: str
    n_trials: int
    n_subjects: int
    n_runs: int
    resolution_rate: float
    overall_accuracy: float
    resolved_accuracy: float
    audit_warning_rate: float
    audit_failure_rate: float


@dataclass(frozen=True, slots=True)
class ModelRecoveryGridReport:
    """Recovery reports across named designs with fixed scenarios and candidates."""

    design_names: tuple[str, ...]
    reports: tuple[ModelRecoveryReport, ...]
    seeds: NDArray[np.uint64]
    root_seed: int

    def __post_init__(self) -> None:
        names = tuple(self.design_names)
        reports = tuple(self.reports)
        seeds = protected_array(self.seeds, dtype=np.uint64)
        if not names or len(set(names)) != len(names):
            raise ValueError("design names must be non-empty and unique")
        if len(reports) != len(names) or seeds.shape != (len(names),):
            raise ValueError("grid reports and seeds must align with design names")
        if (
            isinstance(self.root_seed, bool)
            or not isinstance(self.root_seed, int)
            or self.root_seed < 0
        ):
            raise ValueError("root_seed must be a non-negative integer")
        reference = reports[0]
        for report in reports[1:]:
            if (
                report.candidate_labels != reference.candidate_labels
                or report.candidate_signatures != reference.candidate_signatures
                or report.scored_columns != reference.scored_columns
                or report.scenario_names != reference.scenario_names
                or report.truth_labels != reference.truth_labels
                or report.repeats != reference.repeats
                or report.validation_scheme != reference.validation_scheme
                or report.aggregation_column != reference.aggregation_column
                or report.sampled_candidate_labels != reference.sampled_candidate_labels
            ):
                raise ValueError(
                    "every design cell must use the same candidate and scenario contract"
                )
        object.__setattr__(self, "design_names", names)
        object.__setattr__(self, "reports", reports)
        object.__setattr__(self, "seeds", seeds)

    def report_for(self, design_name: str) -> ModelRecoveryReport:
        """Return one named design cell without relying on tuple position."""

        try:
            index = self.design_names.index(design_name)
        except ValueError:
            raise KeyError(f"unknown recovery-grid design: {design_name!r}") from None
        return self.reports[index]

    def summary(self) -> tuple[ModelRecoveryGridSummary, ...]:
        """Return one compact recovery and audit summary per design cell."""

        return tuple(
            ModelRecoveryGridSummary(
                design_name=name,
                n_trials=report.n_trials,
                n_subjects=report.n_subjects,
                n_runs=report.n_runs,
                resolution_rate=report.resolution_rate,
                overall_accuracy=report.overall_accuracy,
                resolved_accuracy=report.resolved_accuracy,
                audit_warning_rate=report.audit_warning_rate,
                audit_failure_rate=report.audit_failure_rate,
            )
            for name, report in zip(self.design_names, self.reports, strict=True)
        )


@dataclass(frozen=True, slots=True)
class _RecoveryTask:
    """Everything one ``(scenario, repeat)`` cell needs, and nothing it shares.

    The cell is addressed by position: ``child_seed`` was drawn from
    ``SeedSequence(seed).spawn(n_runs)[position]`` before any work started, so it depends on
    where the cell sits and not on when it runs. That is the whole of why these cells can
    be executed in any order, or in parallel, without changing a number.
    """

    design: Study
    scenario: ModelRecoveryScenario
    candidates: tuple[AnyBehaviourEstimator, ...]
    child_seed: int
    min_train_sessions: int
    horizon: int
    step: int
    splitter: Callable[[Study], Iterable[EvaluationFold]] | None
    aggregation_column: str | None
    posterior_policy: PosteriorFoldPolicy | None


@dataclass(frozen=True, slots=True)
class _RecoveryCell:
    """One recovery cell's complete result, with one entry per candidate in column order."""

    seed: int
    n_folds: int
    scores: tuple[float, ...]
    converged: tuple[bool, ...]
    failure_messages: tuple[str, ...]
    audit_statuses: tuple[FitAuditStatus, ...]
    audit_issue_codes: tuple[tuple[str, ...], ...]


def _run_recovery_cell(task: _RecoveryTask) -> _RecoveryCell:
    """Simulate one scenario repeat and score every candidate against it.

    This is a pure function of ``task``. It draws no randomness of its own -- the only seed
    it uses is the position-determined one it was handed -- and it touches nothing outside
    its argument, which is what lets :func:`run_model_recovery` schedule these cells freely.
    """

    scenario = task.scenario
    simulated = scenario.generator.simulate(task.design, scenario.parameters, seed=task.child_seed)
    if task.splitter is None:
        splits: tuple[EvaluationFold, ...] = forward_session_splits(
            simulated,
            min_train_sessions=task.min_train_sessions,
            horizon=task.horizon,
            step=task.step,
        )
    else:
        splits = tuple(task.splitter(simulated))
    if not splits:
        raise ValueError(f"scenario {scenario.name!r} produced no eligible prospective folds")

    scores: list[float] = []
    converged: list[bool] = []
    failure_messages: list[str] = []
    audit_statuses: list[FitAuditStatus] = []
    audit_issue_codes: list[tuple[str, ...]] = []
    for model in task.candidates:
        evaluations = evaluate_splits(
            model,
            simulated,
            splits,
            posterior_policy=(task.posterior_policy if is_posterior_estimator(model) else None),
        )
        scores.append(
            _mean_log_probability(
                simulated,
                evaluations,
                aggregation_column=task.aggregation_column,
            )
        )
        converged.append(
            all(not evaluation.fit.diagnostics.failed_to_converge for evaluation in evaluations)
        )
        failure_messages.append(_failure_message(evaluations))
        audit_status, issue_codes = _aggregate_audits(evaluations)
        audit_statuses.append(audit_status)
        audit_issue_codes.append(issue_codes)
    return _RecoveryCell(
        seed=task.child_seed,
        n_folds=len(splits),
        scores=tuple(scores),
        converged=tuple(converged),
        failure_messages=tuple(failure_messages),
        audit_statuses=tuple(audit_statuses),
        audit_issue_codes=tuple(audit_issue_codes),
    )


def run_model_recovery(
    design: Study,
    scenarios: Sequence[ModelRecoveryScenario],
    candidates: Mapping[str, AnyBehaviourEstimator],
    *,
    repeats: int = 1,
    seed: int,
    min_train_sessions: int = 1,
    horizon: int = 1,
    step: int = 1,
    tie_tolerance: float = 1e-8,
    splitter: Callable[[Study], Iterable[EvaluationFold]] | None = None,
    splitter_name: str | None = None,
    aggregation_column: str | None = None,
    posterior_policy: PosteriorFoldPolicy | None = None,
    workers: int = 1,
    backend: WorkerBackend | str = WorkerBackend.PROCESS,
) -> ModelRecoveryReport:
    """Simulate scenarios and select candidates by prospective mean log probability.

    By default recovery uses expanding within-subject forward-session folds. ``splitter``
    permits an exact experimental validation geometry to be reapplied after each
    simulation; ``splitter_name`` then provides stable provenance in the report.
    ``posterior_policy`` declares the projection and convergence gate applied to sampled
    candidates, which the report names in ``sampled_candidate_labels``.

    ``workers`` runs the ``scenarios x repeats`` cells concurrently and is the one place in
    this function where anything is scheduled. The report it returns is **bit-identical**
    for every worker count, because each cell's simulation seed comes from
    ``SeedSequence(seed).spawn(scenarios * repeats)`` indexed by the cell's position and
    every result is written back by that same position -- including ``failure_messages``,
    ``audit_statuses`` and ``audit_issue_codes``, which are built inside a cell and never
    appended in completion order. ``workers=1`` is the default and runs a plain loop with
    no executor, so a small run pays nothing.

    ``backend`` selects processes (the default, which sidesteps the GIL) or threads.
    Processes require ``design``, the scenarios, the candidates and any ``splitter`` to be
    picklable, which rules out a lambda splitter; threads pickle nothing and are the
    fallback for one. The ``Parallelism and determinism`` guide records measured speedups
    and the crossover below which ``workers=1`` wins.
    """

    scenarios = tuple(scenarios)
    if not scenarios:
        raise ValueError("scenarios must not be empty")
    if len({scenario.name for scenario in scenarios}) != len(scenarios):
        raise ValueError("scenario names must be unique")
    candidate_models = _validated_candidates(candidates)
    candidate_labels = tuple(candidate_models)
    sampled_labels = tuple(
        label for label, model in candidate_models.items() if is_posterior_estimator(model)
    )
    scored_columns = any_model_capabilities(next(iter(candidate_models.values()))).scored_columns
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
    if splitter is None:
        if splitter_name is not None:
            raise ValueError("splitter_name requires a custom splitter")
        validation_scheme = "forward-session"
    else:
        if not callable(splitter):
            raise TypeError("splitter must be callable")
        if splitter_name is None:
            splitter_name = getattr(splitter, "__name__", type(splitter).__qualname__)
        if not isinstance(splitter_name, str) or not splitter_name:
            raise ValueError("splitter_name must be a non-empty string")
        validation_scheme = splitter_name
    if aggregation_column is not None:
        if not isinstance(aggregation_column, str) or not aggregation_column:
            raise ValueError("aggregation_column must be None or a non-empty string")
        if aggregation_column not in design.columns:
            raise ValueError(f"design is missing aggregation column {aggregation_column!r}")
    backend = WorkerBackend(backend)
    # Validated alongside every other argument, rather than after the simulation setup that
    # `map_ordered` sits behind: a bad `workers` is a caller's typo and should be reported
    # with the other caller's typos.
    resolve_workers(workers, n_tasks=1)

    n_runs = len(scenarios) * repeats
    child_sequences = np.random.SeedSequence(seed).spawn(n_runs)
    # One cell per (scenario, repeat), in the order the report will report them. Every seed
    # is drawn here, from the cell's position, before any cell runs.
    cells = tuple(scenario for scenario in scenarios for _ in range(repeats))
    models = tuple(candidate_models.values())
    tasks = tuple(
        _RecoveryTask(
            design=design,
            scenario=scenario,
            candidates=models,
            child_seed=int(child_sequences[position].generate_state(1, dtype=np.uint64)[0]),
            min_train_sessions=min_train_sessions,
            horizon=horizon,
            step=step,
            splitter=splitter,
            aggregation_column=aggregation_column,
            posterior_policy=posterior_policy,
        )
        for position, scenario in enumerate(cells)
    )
    completed = map_ordered(_run_recovery_cell, tasks, workers=workers, backend=backend)

    scores = np.asarray([run.scores for run in completed], dtype=np.float64).reshape(
        n_runs, len(candidate_labels)
    )
    converged = np.asarray([run.converged for run in completed], dtype=np.bool_).reshape(
        n_runs, len(candidate_labels)
    )
    seeds = np.asarray([run.seed for run in completed], dtype=np.uint64)
    n_folds = np.asarray([run.n_folds for run in completed], dtype=np.int64)
    selected_labels = [
        _select_candidate(
            scores[position],
            np.asarray(
                [status is not FitAuditStatus.FAIL for status in run.audit_statuses],
                dtype=np.bool_,
            ),
            candidate_labels,
            tie_tolerance,
        )
        for position, run in enumerate(completed)
    ]
    scenario_names = [scenario.name for scenario in cells]
    generator_signatures = [scenario.generator.signature for scenario in cells]
    generator_parameters: list[Mapping[str, float]] = [scenario.parameters for scenario in cells]
    truth_labels = [scenario.truth_label for scenario in cells]
    failure_messages = [run.failure_messages for run in completed]
    audit_statuses = [run.audit_statuses for run in completed]
    audit_issue_codes = [run.audit_issue_codes for run in completed]

    return ModelRecoveryReport(
        candidate_labels=candidate_labels,
        candidate_signatures=tuple(model.signature for model in candidate_models.values()),
        scored_columns=scored_columns,
        scenario_names=tuple(scenario_names),
        generator_signatures=tuple(generator_signatures),
        generator_parameters=tuple(generator_parameters),
        truth_labels=tuple(truth_labels),
        selected_labels=tuple(selected_labels),
        mean_log_probabilities=scores,
        converged=converged,
        failure_messages=tuple(failure_messages),
        audit_statuses=tuple(audit_statuses),
        audit_issue_codes=tuple(audit_issue_codes),
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
        validation_scheme=validation_scheme,
        aggregation_column=aggregation_column,
        sampled_candidate_labels=sampled_labels,
    )


def run_model_recovery_grid(
    designs: Mapping[str, Study],
    scenarios: Sequence[ModelRecoveryScenario],
    candidates: Mapping[str, AnyBehaviourEstimator],
    *,
    repeats: int = 1,
    seed: int,
    min_train_sessions: int = 1,
    horizon: int = 1,
    step: int = 1,
    tie_tolerance: float = 1e-8,
    splitter: Callable[[Study], Iterable[EvaluationFold]] | None = None,
    splitter_name: str | None = None,
    aggregation_column: str | None = None,
    posterior_policy: PosteriorFoldPolicy | None = None,
    workers: int = 1,
    backend: WorkerBackend | str = WorkerBackend.PROCESS,
) -> ModelRecoveryGridReport:
    """Run one fixed recovery contract across named study-design cells.

    ``workers`` is forwarded to each design cell rather than used to run the cells against
    one another. Design cells are the coarser level, but they are also the level at which
    the work is least even -- a grid usually varies trial count, so one cell can cost many
    times another -- and parallelising the cells would leave the largest one running alone
    at the end. Parallelising inside each cell keeps every worker busy for the whole grid,
    and avoids nesting one pool inside another.
    """

    if not isinstance(designs, Mapping) or not designs:
        raise ValueError("designs must be a non-empty mapping")
    validated_designs: dict[str, Study] = {}
    for name, design in designs.items():
        if not isinstance(name, str) or not name:
            raise ValueError("design names must be non-empty strings")
        if not isinstance(design, Study):
            raise TypeError(f"design {name!r} must be a Study")
        validated_designs[name] = design
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    child_sequences = np.random.SeedSequence(seed).spawn(len(validated_designs))
    seeds = np.asarray(
        [sequence.generate_state(1, dtype=np.uint64)[0] for sequence in child_sequences],
        dtype=np.uint64,
    )
    reports = tuple(
        run_model_recovery(
            design,
            scenarios,
            candidates,
            repeats=repeats,
            seed=int(child_seed),
            min_train_sessions=min_train_sessions,
            horizon=horizon,
            step=step,
            tie_tolerance=tie_tolerance,
            splitter=splitter,
            splitter_name=splitter_name,
            aggregation_column=aggregation_column,
            posterior_policy=posterior_policy,
            workers=workers,
            backend=backend,
        )
        for design, child_seed in zip(validated_designs.values(), seeds, strict=True)
    )
    return ModelRecoveryGridReport(
        design_names=tuple(validated_designs),
        reports=reports,
        seeds=seeds,
        root_seed=seed,
    )


def _validated_candidates(
    candidates: Mapping[str, AnyBehaviourEstimator],
) -> Mapping[str, AnyBehaviourEstimator]:
    if not isinstance(candidates, Mapping) or not candidates:
        raise ValueError("candidates must be a non-empty mapping")
    validated: dict[str, AnyBehaviourEstimator] = {}
    for label, model in candidates.items():
        if not isinstance(label, str) or not label or label == UNRESOLVED_LABEL:
            raise ValueError(
                f"candidate labels must be non-empty and cannot be {UNRESOLVED_LABEL!r}"
            )
        if not isinstance(model, BehaviourEstimator) and not is_posterior_estimator(model):
            raise TypeError(
                f"candidate {label!r} satisfies neither the BehaviourEstimator nor the "
                "PosteriorBehaviourEstimator contract"
            )
        any_model_capabilities(model)
        validated[label] = model
    scored_columns = {any_model_capabilities(model).scored_columns for model in validated.values()}
    if len(scored_columns) != 1:
        raise ValueError("all model-recovery candidates must score identical observed columns")
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
        if diagnostics.failed_to_converge:
            failures.append(f"fold {fold}: {diagnostics.message}")
    return "; ".join(failures)


def _mean_log_probability(
    study: Study,
    evaluations: tuple[FoldEvaluation, ...],
    *,
    aggregation_column: str | None,
) -> float:
    if aggregation_column is None:
        pointwise = np.concatenate(
            [evaluation.pointwise_log_probability for evaluation in evaluations]
        )
        return float(np.mean(pointwise))

    totals: dict[object, float] = {}
    counts: dict[object, int] = {}
    for evaluation in evaluations:
        units = study[aggregation_column][evaluation.split.test_indices]
        for unit, score in zip(units, evaluation.pointwise_log_probability, strict=True):
            key = unit.item() if isinstance(unit, np.generic) else unit
            totals[key] = totals.get(key, 0.0) + float(score)
            counts[key] = counts.get(key, 0) + 1
    if not totals:
        raise ValueError("model recovery produced no aggregation units")
    return float(np.mean([totals[key] / counts[key] for key in totals]))


def _aggregate_audits(
    evaluations: tuple[FoldEvaluation, ...],
) -> tuple[FitAuditStatus, tuple[str, ...]]:
    audits = tuple(evaluation.fit.audit() for evaluation in evaluations)
    if any(audit.status is FitAuditStatus.FAIL for audit in audits):
        status = FitAuditStatus.FAIL
    elif any(audit.status is FitAuditStatus.WARNING for audit in audits):
        status = FitAuditStatus.WARNING
    else:
        status = FitAuditStatus.PASS
    codes = tuple(dict.fromkeys(code for audit in audits for code in audit.issue_codes))
    return status, codes


def _require_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
