"""Opt-in parallelism must not change a result.

Every test here asks one question in a different place: is the artifact a parallel run
produces the artifact a serial run produces, byte for byte? Nothing is compared by
tolerance. A recovery report's scores are compared through ``tobytes()``, retained failures
are compared as ordered tuples, and the whole-report comparisons lean on the frozen
dataclasses' structural equality rather than on a hand-written field list.
"""

from __future__ import annotations

import functools
import hashlib
import pickle

import numpy as np
import pytest

from behavio import (
    BernoulliHistoryGLM,
    ModelRecoveryScenario,
    PosteriorGroup,
    PosteriorParameterQuantity,
    PosteriorResult,
    PosteriorVariable,
    SBCSimulation,
    Study,
    compare_models,
    forward_session_splits,
    nested_select_model,
    run_model_recovery,
    run_model_recovery_grid,
    run_simulation_based_calibration,
)
from behavio._internal.parallel import (
    ParallelWorkerError,
    UnpicklableTaskError,
    WorkerBackend,
    map_ordered,
    resolve_workers,
)
from behavio.compose import smooth as make_smooth

# --------------------------------------------------------------------------------------
# Fixtures shared by the recovery, comparison and selection cases
# --------------------------------------------------------------------------------------


def recovery_design(*, n_sessions: int = 6, n_trials: int = 60) -> Study:
    generator = np.random.default_rng(2027)
    n_rows = n_sessions * n_trials
    return Study(
        {
            "subject": ["a"] * n_rows,
            "session": [f"session-{s}" for s in range(n_sessions) for _ in range(n_trials)],
            "trial": list(range(n_trials)) * n_sessions,
            "session_order": [s for s in range(n_sessions) for _ in range(n_trials)],
            "stimulus": generator.normal(size=n_rows),
        }
    )


def competing_models(n_sessions: int = 6):
    static = BernoulliHistoryGLM(covariates=("stimulus",), choice_lags=1, l2=0.01)
    smooth = make_smooth(
        BernoulliHistoryGLM(covariates=("stimulus",), choice_lags=1, l2=0.01),
        over="session_order",
        knots=tuple(float(knot) for knot in range(n_sessions)),
        smoothness=10.0,
    )
    return static, smooth


def recovery_scenarios(static, smooth):
    n_knots = len(smooth.knots)
    return (
        ModelRecoveryScenario(
            name="stationary",
            truth_label="static",
            generator=static,
            parameters={"intercept": -0.2, "stimulus": 1.2, "choice_lag_1": 0.4},
        ),
        ModelRecoveryScenario(
            name="drifting",
            truth_label="smooth",
            generator=smooth,
            parameters=smooth.parameters_from_paths(
                {
                    "intercept": np.linspace(-0.5, 0.5, n_knots),
                    "stimulus": np.linspace(0.2, 2.5, n_knots),
                    "choice_lag_1": np.linspace(0.8, 0.1, n_knots),
                }
            ),
        ),
    )


def recovery_arguments(repeats: int = 3) -> dict[str, object]:
    static, smooth = competing_models()
    return {
        "design": recovery_design(),
        "scenarios": recovery_scenarios(static, smooth),
        "candidates": {"static": static, "smooth": smooth},
        "repeats": repeats,
        "seed": 12,
        "min_train_sessions": 3,
        "tie_tolerance": 0.001,
    }


def report_bytes(report) -> bytes:
    """A byte image of every retained number, label, failure and diagnostic.

    The arrays go in through ``tobytes()`` rather than through ``allclose``: the claim under
    test is that a parallel run reproduces a serial one exactly, and a tolerance would let
    exactly the drift this is meant to catch through unnoticed.
    """

    payload = (
        report.mean_log_probabilities.tobytes(),
        report.converged.tobytes(),
        report.seeds.tobytes(),
        report.n_folds.tobytes(),
        report.candidate_labels,
        report.candidate_signatures,
        report.scenario_names,
        report.generator_signatures,
        tuple(dict(parameters) for parameters in report.generator_parameters),
        report.truth_labels,
        report.selected_labels,
        report.failure_messages,
        report.audit_statuses,
        report.audit_issue_codes,
    )
    return hashlib.sha256(pickle.dumps(payload)).digest()


# --------------------------------------------------------------------------------------
# The scheduler itself
# --------------------------------------------------------------------------------------


def double(value: int) -> int:
    return value * 2


def fail_on_odd(value: int) -> int:
    if value % 2:
        raise ValueError(f"refusing {value}")
    return value


class _Unpicklable:
    def __reduce__(self):
        raise TypeError("this object refuses to be pickled")


def test_map_ordered_returns_input_order_for_every_worker_count() -> None:
    tasks = list(range(16))
    expected = [double(task) for task in tasks]
    for workers in (1, 2, 4, 8):
        for backend in (WorkerBackend.PROCESS, WorkerBackend.THREAD):
            assert map_ordered(double, tasks, workers=workers, backend=backend) == expected


def test_map_ordered_raises_the_lowest_indexed_failure_not_the_first_to_occur() -> None:
    # 3, 5 and 7 all fail. A serial loop stops at 3, so a parallel one must too, however
    # the tasks happen to interleave.
    with pytest.raises(ValueError, match="refusing 3"):
        map_ordered(fail_on_odd, [0, 2, 3, 5, 7], workers=4, backend=WorkerBackend.THREAD)
    with pytest.raises(ValueError, match="refusing 3"):
        map_ordered(fail_on_odd, [0, 2, 3, 5, 7], workers=4, backend=WorkerBackend.PROCESS)


def test_a_worker_exception_keeps_its_type_and_carries_a_real_traceback() -> None:
    with pytest.raises(ValueError) as caught:
        map_ordered(fail_on_odd, [1, 2], workers=2, backend=WorkerBackend.PROCESS)

    # The type survives, so `except ValueError` keeps working under any worker count...
    assert "refusing 1" in str(caught.value)
    # ...and the worker's own frames are attached, rather than a bare BrokenProcessPool
    # naming only the parent's `result()` call.
    notes = "\n".join(getattr(caught.value, "__notes__", []))
    assert "fail_on_odd" in notes
    assert 'raise ValueError(f"refusing {value}")' in notes
    assert "parallel task 0" in notes


def test_an_unpicklable_payload_is_refused_before_any_work_starts() -> None:
    with pytest.raises(UnpicklableTaskError, match="position 1"):
        map_ordered(double, [1, _Unpicklable(), 3], workers=2, backend=WorkerBackend.PROCESS)


def test_an_unpicklable_payload_is_fine_on_the_thread_backend() -> None:
    payload = _Unpicklable()
    assert map_ordered(type, [payload, payload], workers=2, backend=WorkerBackend.THREAD) == [
        _Unpicklable,
        _Unpicklable,
    ]


def test_a_lambda_task_function_is_named_rather_than_dying_as_a_broken_pool() -> None:
    with pytest.raises(UnpicklableTaskError, match="cannot be pickled"):
        map_ordered(lambda value: value, [1, 2], workers=2, backend=WorkerBackend.PROCESS)


def test_worker_counts_are_validated_and_clamped_to_the_available_work() -> None:
    assert resolve_workers(8, n_tasks=3) == 3
    assert resolve_workers(1, n_tasks=100) == 1
    for bad in (0, -1, True, 2.0, "4"):
        with pytest.raises(ValueError, match="workers must be a positive integer"):
            resolve_workers(bad, n_tasks=4)  # type: ignore[arg-type]


def test_an_empty_task_list_never_starts_a_pool() -> None:
    assert map_ordered(double, [], workers=8) == []


def test_parallel_worker_error_is_raised_when_an_exception_cannot_travel() -> None:
    with pytest.raises(ParallelWorkerError, match="could not be sent back"):
        map_ordered(raise_unpicklable_error, [0, 1], workers=2, backend=WorkerBackend.PROCESS)


class UnpicklableError(Exception):
    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail

    def __reduce__(self):
        raise TypeError("this exception refuses to be pickled")


def raise_unpicklable_error(_: int) -> int:
    raise UnpicklableError("the worker failed in a way that cannot be shipped home")


# --------------------------------------------------------------------------------------
# Study, the payload every worker receives
# --------------------------------------------------------------------------------------


def test_a_study_survives_a_round_trip_and_stays_immutable() -> None:
    study = recovery_design()
    restored = pickle.loads(pickle.dumps(study))

    assert restored.columns == study.columns
    assert len(restored) == len(study)
    assert restored.subjects == study.subjects
    for column in study.columns:
        assert np.array_equal(restored[column], study[column])
        # A study that arrived from another process must be as immutable as one built here.
        assert not restored[column].flags.writeable
    with pytest.raises(AttributeError):
        restored._length = 0


def test_a_recovery_scenario_survives_a_round_trip_with_a_frozen_parameter_map() -> None:
    static, _ = competing_models()
    scenario = ModelRecoveryScenario(
        name="stationary",
        truth_label="static",
        generator=static,
        parameters={"intercept": -0.2, "stimulus": 1.2, "choice_lag_1": 0.4},
    )
    restored = pickle.loads(pickle.dumps(scenario))

    assert restored == scenario
    assert dict(restored.parameters) == dict(scenario.parameters)
    with pytest.raises(TypeError):
        restored.parameters["stimulus"] = 99.0


@pytest.mark.parametrize(
    "model",
    [
        BernoulliHistoryGLM(covariates=("stimulus",), choice_lags=1, l2=0.01),
        make_smooth(
            BernoulliHistoryGLM(covariates=("stimulus",), choice_lags=1),
            over="session_order",
            knots=(0.0, 1.0, 2.0),
        ),
    ],
    ids=["plain", "smooth"],
)
def test_composed_models_reach_a_worker_process_intact(model) -> None:
    """`smooth(...)` wraps a model rather than closing over one, so it pickles.

    The combinators build frozen dataclasses whose fields are the wrapped estimator and
    plain scalars -- no closures, no lambdas, no bound methods -- which is what makes the
    process backend usable with composed candidates at all.
    """

    restored = pickle.loads(pickle.dumps(model))
    assert restored == model
    assert restored.signature == model.signature
    assert restored.parameter_names == model.parameter_names


# --------------------------------------------------------------------------------------
# Model recovery
# --------------------------------------------------------------------------------------


def test_model_recovery_is_bit_identical_under_every_worker_count() -> None:
    arguments = recovery_arguments()
    serial = run_model_recovery(**arguments)
    expected = report_bytes(serial)

    for workers in (2, 3, 6):
        parallel = run_model_recovery(**arguments, workers=workers)
        assert report_bytes(parallel) == expected, f"workers={workers} changed the report"
        assert parallel.selected_labels == serial.selected_labels
        assert parallel.failure_messages == serial.failure_messages
        assert parallel.audit_statuses == serial.audit_statuses
        assert parallel.audit_issue_codes == serial.audit_issue_codes


def test_model_recovery_is_bit_identical_on_the_thread_backend() -> None:
    arguments = recovery_arguments()
    serial = run_model_recovery(**arguments)
    threaded = run_model_recovery(**arguments, workers=4, backend="thread")

    assert report_bytes(threaded) == report_bytes(serial)


def test_two_separate_parallel_recovery_runs_agree_with_each_other() -> None:
    """Determinism across runs, not merely against the serial baseline.

    A scheduler that reordered results consistently would pass a serial comparison only by
    luck; running the parallel path twice and requiring the same bytes closes that gap.
    """

    arguments = recovery_arguments()
    first = run_model_recovery(**arguments, workers=4)
    second = run_model_recovery(**arguments, workers=4)

    assert report_bytes(first) == report_bytes(second)
    assert first.seeds.tobytes() == second.seeds.tobytes()


def test_recovery_seeds_are_position_determined_rather_than_order_determined() -> None:
    """The property the whole guarantee rests on, asserted directly.

    Each cell's simulation seed comes from ``SeedSequence(seed).spawn(n_runs)`` indexed by
    position. If a shared generator were advanced per cell instead, running the cells in a
    different order would deal them different seeds and every downstream number would move.
    """

    arguments = recovery_arguments()
    serial = run_model_recovery(**arguments)
    parallel = run_model_recovery(**arguments, workers=4)

    n_runs = len(arguments["scenarios"]) * arguments["repeats"]  # type: ignore[arg-type]
    expected = np.asarray(
        [
            int(sequence.generate_state(1, dtype=np.uint64)[0])
            for sequence in np.random.SeedSequence(arguments["seed"]).spawn(n_runs)
        ],
        dtype=np.uint64,
    )
    assert serial.seeds.tolist() == expected.tolist()
    assert parallel.seeds.tolist() == expected.tolist()


def test_a_recovery_grid_is_bit_identical_when_its_cells_run_in_parallel() -> None:
    arguments = recovery_arguments(repeats=2)
    design = arguments.pop("design")
    designs = {"small": design, "large": recovery_design(n_sessions=6, n_trials=80)}

    arguments.pop("seed")
    serial = run_model_recovery_grid(designs, **arguments, seed=5)
    parallel = run_model_recovery_grid(designs, **arguments, seed=5, workers=4)

    assert parallel.design_names == serial.design_names
    assert parallel.seeds.tobytes() == serial.seeds.tobytes()
    for expected_report, observed in zip(serial.reports, parallel.reports, strict=True):
        assert report_bytes(observed) == report_bytes(expected_report)


def test_a_failing_recovery_cell_reports_the_scenario_that_failed() -> None:
    """A worker's exception must name the science, not the scheduler."""

    arguments = recovery_arguments(repeats=1)
    arguments["min_train_sessions"] = 99
    with pytest.raises(ValueError, match="no eligible prospective folds"):
        run_model_recovery(**arguments, workers=2)


def test_a_lambda_splitter_is_refused_by_name_and_works_on_threads() -> None:
    """The most likely way to reach the process backend with something it cannot send."""

    arguments = recovery_arguments(repeats=1)
    with pytest.raises(UnpicklableTaskError, match="cannot be pickled"):
        run_model_recovery(
            **arguments,
            splitter=lambda study: forward_session_splits(study, min_train_sessions=3),
            splitter_name="lambda-splitter",
            workers=2,
        )

    picklable = functools.partial(forward_session_splits, min_train_sessions=3)
    serial = run_model_recovery(**arguments, splitter=picklable, splitter_name="partial")
    parallel = run_model_recovery(
        **arguments, splitter=picklable, splitter_name="partial", workers=2
    )
    assert report_bytes(parallel) == report_bytes(serial)


def test_recovery_rejects_an_invalid_worker_count_before_running_anything() -> None:
    arguments = recovery_arguments(repeats=1)
    with pytest.raises(ValueError, match="workers must be a positive integer"):
        run_model_recovery(**arguments, workers=0)


# --------------------------------------------------------------------------------------
# Simulation-based calibration
# --------------------------------------------------------------------------------------


def beta_binomial_simulator(seed: int) -> SBCSimulation:
    generator = np.random.default_rng(seed)
    probability = generator.uniform()
    n_trials = 20
    design = Study(
        {
            "subject": ["mouse"] * n_trials,
            "session": ["session"] * n_trials,
            "trial": np.arange(n_trials),
            "session_order": np.zeros(n_trials, dtype=int),
            "choice": generator.binomial(1, probability, size=n_trials),
        }
    )
    return SBCSimulation(design, {"probability": probability})


def beta_binomial_inference(study: Study, seed: int) -> PosteriorResult:
    generator = np.random.default_rng(seed)
    successes = int(np.sum(study["choice"]))
    failures = len(study) - successes
    draws = generator.beta(1 + successes, 1 + failures, size=(2, 200))
    variable = PosteriorVariable(
        "probability",
        draws,
        ("chain", "draw"),
        {"chain": np.arange(2), "draw": np.arange(200)},
    )
    return PosteriorResult(
        model_name="beta-binomial",
        model_signature="beta-binomial[uniform-prior]",
        inference_library="numpy",
        inference_library_version=np.__version__,
        parameter_names=("probability",),
        groups=(PosteriorGroup("posterior", (variable,)),),
    )


def flaky_inference(study: Study, seed: int) -> PosteriorResult:
    """Fail for a deterministic subset of seeds, so failures are reproducible evidence."""

    if seed % 3 == 0:
        raise RuntimeError(f"sampler refused seed {seed}")
    return beta_binomial_inference(study, seed)


def sbc_arguments(inference=beta_binomial_inference, repeats: int = 24) -> dict[str, object]:
    return {
        "simulator": beta_binomial_simulator,
        "inference": inference,
        "quantities": (PosteriorParameterQuantity("probability"),),
        "repeats": repeats,
        "seed": 419,
        "simulation_signature": "beta-binomial-prior-predictive[v1]",
        "inference_signature": "beta-binomial-conjugate[v1]",
        "thin": 2,
    }


def test_sbc_is_identical_under_every_worker_count() -> None:
    pytest.importorskip("arviz")
    arguments = sbc_arguments()
    serial = run_simulation_based_calibration(**arguments)

    for workers in (2, 4, 8):
        parallel = run_simulation_based_calibration(**arguments, workers=workers)
        assert parallel == serial, f"workers={workers} changed the report"


def test_sbc_retains_failures_in_replicate_order_under_every_worker_count() -> None:
    """The order of retained evidence is part of the artifact, not an implementation detail.

    A parallel run that appended failures as workers finished would produce the same *set*
    of failures in a scrambled order -- and a scrambled artifact is a changed artifact.
    """

    pytest.importorskip("arviz")
    arguments = sbc_arguments(inference=flaky_inference)
    serial = run_simulation_based_calibration(**arguments)

    assert serial.failures, "the fixture must actually produce retained failures"
    replicates = [failure.replicate for failure in serial.failures]
    assert replicates == sorted(replicates)

    for workers in (2, 4, 8):
        parallel = run_simulation_based_calibration(**arguments, workers=workers)
        assert parallel.failures == serial.failures
        assert parallel.ranks == serial.ranks
        assert parallel == serial


def test_two_separate_parallel_sbc_runs_agree_with_each_other() -> None:
    pytest.importorskip("arviz")
    arguments = sbc_arguments(inference=flaky_inference)
    first = run_simulation_based_calibration(**arguments, workers=4)
    second = run_simulation_based_calibration(**arguments, workers=4)

    assert first == second


def test_sbc_runs_a_closure_on_the_thread_backend() -> None:
    """Closures are the normal way to write an SBC pipeline, and threads accept them."""

    pytest.importorskip("arviz")

    scale = 200

    def closure_inference(study: Study, seed: int) -> PosteriorResult:
        assert scale == 200
        return beta_binomial_inference(study, seed)

    arguments = sbc_arguments(inference=closure_inference, repeats=8)
    serial = run_simulation_based_calibration(**arguments)
    threaded = run_simulation_based_calibration(**arguments, workers=4, backend="thread")

    assert threaded == serial

    with pytest.raises(UnpicklableTaskError, match="cannot be pickled"):
        run_simulation_based_calibration(**arguments, workers=4)


# --------------------------------------------------------------------------------------
# Comparison and nested selection
# --------------------------------------------------------------------------------------


def comparison_study() -> Study:
    static, _ = competing_models()
    design = recovery_design()
    return static.simulate(
        design, {"intercept": -0.2, "stimulus": 1.2, "choice_lag_1": 0.4}, seed=31
    )


def test_compare_models_is_identical_when_candidates_run_in_parallel() -> None:
    study = comparison_study()
    static, smooth = competing_models()
    splits = forward_session_splits(study, min_train_sessions=3)
    arguments = {
        "models": {"static": static, "smooth": smooth},
        "study": study,
        "splits": splits,
        "aggregation_column": "session",
        "bootstrap_resamples": 200,
    }

    serial = compare_models(**arguments)
    for workers, backend in ((2, "process"), (2, "thread")):
        parallel = compare_models(**arguments, workers=workers, backend=backend)
        assert parallel.winner == serial.winner
        assert parallel.pairwise_comparisons == serial.pairwise_comparisons
        for expected, observed in zip(serial.model_results, parallel.model_results, strict=True):
            assert observed.name == expected.name
            assert observed.pooled_log_loss == expected.pooled_log_loss
            assert observed.unit_log_losses.tobytes() == expected.unit_log_losses.tobytes()
            assert observed.aggregation_units == expected.aggregation_units


def test_nested_selection_is_identical_when_outer_folds_run_in_parallel() -> None:
    """The four-deep loop: outer folds x candidates x inner folds x fits."""

    study = comparison_study()
    static, smooth = competing_models()
    inner_splitter = functools.partial(forward_session_splits, min_train_sessions=2)
    arguments = {
        "candidates": {"static": static, "smooth": smooth},
        "study": study,
        "outer_splits": forward_session_splits(study, min_train_sessions=4),
        "inner_splitter": inner_splitter,
        "aggregation_column": "session",
        "bootstrap_resamples": 200,
        "inner_bootstrap_resamples": 100,
    }

    serial = nested_select_model(**arguments)
    for workers in (2, 4):
        parallel = nested_select_model(**arguments, workers=workers)
        assert [fold.selected_model for fold in parallel.folds] == [
            fold.selected_model for fold in serial.folds
        ]
        assert parallel.aggregation_units == serial.aggregation_units
        assert parallel.unit_log_losses.tobytes() == serial.unit_log_losses.tobytes()
        assert parallel.pooled_log_loss == serial.pooled_log_loss
        assert parallel.pooled_brier_score == serial.pooled_brier_score
        assert parallel.unit_balanced_log_loss_interval == (serial.unit_balanced_log_loss_interval)


def test_a_lambda_inner_splitter_is_refused_by_name() -> None:
    study = comparison_study()
    static, smooth = competing_models()
    with pytest.raises(UnpicklableTaskError, match="cannot be pickled"):
        nested_select_model(
            {"static": static, "smooth": smooth},
            study,
            forward_session_splits(study, min_train_sessions=4),
            lambda inner: forward_session_splits(inner, min_train_sessions=2),
            aggregation_column="session",
            bootstrap_resamples=200,
            inner_bootstrap_resamples=100,
            workers=2,
        )


# --------------------------------------------------------------------------------------
# What a memo on `(signature, fold rows)` would have done
# --------------------------------------------------------------------------------------


def test_a_model_signature_does_not_capture_everything_that_changes_a_fit() -> None:
    """Why nested selection's repeated refits are *not* memoised.

    The obvious cache key for the refits is ``(model signature, fold row indices)``, and
    ``signature`` is the package's scientific fingerprint, so it looks like exactly the
    right key. It is not sufficient: a model's optimizer settings change its estimates and
    do not change its signature. Two candidates that differ only in ``max_iterations`` are
    a legitimate thing to compare -- convergence behaviour is a real question about a model
    -- and a memo keyed this way would serve the first one's fit to the second and report
    them as identical.

    This test exists to keep that reasoning falsifiable. If the signature is ever widened to
    cover these fields, it fails, and caching can be reconsidered on the evidence.
    """

    n_rows = 600
    generator = np.random.default_rng(7)
    study = Study(
        {
            "subject": ["a"] * n_rows,
            "session": [f"s{i // 60}" for i in range(n_rows)],
            "trial": list(range(60)) * 10,
            "session_order": [i // 60 for i in range(n_rows)],
            "stimulus": generator.normal(size=n_rows),
            "choice": generator.integers(0, 2, size=n_rows),
        }
    )
    loose = BernoulliHistoryGLM(covariates=("stimulus",), choice_lags=1, max_iterations=2)
    tight = BernoulliHistoryGLM(covariates=("stimulus",), choice_lags=1, max_iterations=1_000)

    assert loose.signature == tight.signature
    assert not np.array_equal(loose.fit(study).estimates, tight.fit(study).estimates)


def test_a_repeated_fit_of_one_model_is_bitwise_deterministic() -> None:
    """The other half of the caching question: the fits themselves are reproducible.

    Determinism is what makes the *parallel* guarantee possible at all -- the same model on
    the same rows in a worker process gives the same estimates as it does here. It is also
    what would have made a memo correct, had there been a sufficient key to hang one on.
    """

    study = comparison_study()
    model = BernoulliHistoryGLM(covariates=("stimulus",), choice_lags=1, l2=0.01)
    first = model.fit(study)
    second = model.fit(study)

    assert first.estimates.tobytes() == second.estimates.tobytes()
    assert first.standard_errors.tobytes() == second.standard_errors.tobytes()


def drifting_coordinate_simulator(seed: int) -> SBCSimulation:
    """A two-element vector quantity, so the run has coordinates that could disagree."""

    generator = np.random.default_rng(seed)
    truth = generator.uniform(size=2)
    n_trials = 12
    design = Study(
        {
            "subject": ["mouse"] * n_trials,
            "session": ["session"] * n_trials,
            "trial": np.arange(n_trials),
            "session_order": np.zeros(n_trials, dtype=int),
            "choice": generator.binomial(1, truth.mean(), size=n_trials),
        }
    )
    return SBCSimulation(design, {"theta": truth})


def relabelling_inference(study: Study, seed: int) -> PosteriorResult:
    """Rename the second coordinate for a deterministic subset of replicates.

    The identity of an SBC target is `(signature, dims, coordinates)`. Changing a coordinate
    label mid-run means the ranks are no longer about the same thing, which the serial loop
    catches by comparing every replicate against the first one that succeeded.
    """

    generator = np.random.default_rng(seed)
    draws = generator.beta(2.0, 2.0, size=(2, 120, 2))
    labels = ["a", "b"] if seed % 4 else ["a", "SHIFTED"]
    variable = PosteriorVariable(
        "theta",
        draws,
        ("chain", "draw", "theta_dim"),
        {"chain": np.arange(2), "draw": np.arange(120), "theta_dim": labels},
    )
    return PosteriorResult(
        model_name="vector-toy",
        model_signature="vector-toy[v1]",
        inference_library="numpy",
        inference_library_version=np.__version__,
        parameter_names=("theta",),
        groups=(PosteriorGroup("posterior", (variable,)),),
    )


def test_the_cross_replicate_identity_check_is_settled_in_replicate_order() -> None:
    """The one part of a replicate that is not independent of the others.

    Each worker can rank its own replicate alone, but *whether that ranking is comparable*
    is a question about the earlier replicates, so the check is replayed in the parent. If a
    worker decided it locally, which replicate established the targets would depend on which
    finished first, and a different set of replicates would be rejected on every run.
    """

    pytest.importorskip("arviz")
    arguments = {
        "simulator": drifting_coordinate_simulator,
        "inference": relabelling_inference,
        "quantities": (PosteriorParameterQuantity("theta"),),
        "repeats": 24,
        "seed": 91,
        "simulation_signature": "vector-toy-prior-predictive[v1]",
        "inference_signature": "vector-toy[v1]",
        "audit_policy": None,
    }
    serial = run_simulation_based_calibration(**arguments)

    mismatches = [failure for failure in serial.failures if failure.stage == "evaluation"]
    assert mismatches, "the fixture must actually produce an identity mismatch"

    for workers in (2, 4, 8):
        parallel = run_simulation_based_calibration(**arguments, workers=workers)
        assert parallel.failures == serial.failures
        assert parallel.ranks == serial.ranks
        assert parallel == serial
