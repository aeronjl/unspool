"""Measure what ``workers`` actually buys, and where it starts costing instead.

Run this rather than trusting a rule of thumb::

    uv run python scripts/parallel_scaling.py
    uv run python scripts/parallel_scaling.py --quick

It sweeps :func:`behavio.run_model_recovery` across per-cell workloads and worker counts,
on both backends, and reports wall time, speedup against ``workers=1``, and the digest of
every retained number in the report -- so a row that is faster but produces a different
digest is visible as the bug it would be.

The crossover is the smallest per-cell cost at which ``workers>1`` beats ``workers=1``.
Below it, process startup and the pickling of the design and the candidates cost more than
the fits they are meant to overlap, and the honest recommendation is the default.

This is deliberately not in ``benchmarks/``: that directory pins committed numbers and is
quarantined behind a marker. Nothing here is asserted against; it is an instrument.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import pickle
import time
from dataclasses import dataclass

import numpy as np

from behavio import BernoulliHistoryGLM, ModelRecoveryScenario, Study, run_model_recovery
from behavio.compose import smooth as make_smooth


def design(n_sessions: int, n_trials: int) -> Study:
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


def candidates(n_sessions: int):
    static = BernoulliHistoryGLM(covariates=("stimulus",), choice_lags=1, l2=0.01)
    smooth = make_smooth(
        BernoulliHistoryGLM(covariates=("stimulus",), choice_lags=1, l2=0.01),
        over="session_order",
        knots=tuple(float(knot) for knot in range(n_sessions)),
        smoothness=10.0,
    )
    return static, smooth


def scenarios(static, smooth):
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


def digest(report) -> str:
    payload = (
        report.mean_log_probabilities.tobytes(),
        report.converged.tobytes(),
        report.seeds.tobytes(),
        report.n_folds.tobytes(),
        report.selected_labels,
        report.failure_messages,
        report.audit_statuses,
        report.audit_issue_codes,
    )
    return hashlib.sha256(pickle.dumps(payload)).hexdigest()[:12]


@dataclass(frozen=True, slots=True)
class Measurement:
    n_sessions: int
    n_trials: int
    repeats: int
    workers: int
    backend: str
    seconds: float
    digest: str

    @property
    def n_cells(self) -> int:
        return 2 * self.repeats


def measure(
    *, n_sessions: int, n_trials: int, repeats: int, workers: int, backend: str, replicates: int
) -> Measurement:
    static, smooth = candidates(n_sessions)
    arguments = {
        "design": design(n_sessions, n_trials),
        "scenarios": scenarios(static, smooth),
        "candidates": {"static": static, "smooth": smooth},
        "repeats": repeats,
        "seed": 12,
        "min_train_sessions": 3,
        "tie_tolerance": 0.001,
    }
    best = float("inf")
    fingerprint = ""
    for _ in range(replicates):
        start = time.perf_counter()
        report = run_model_recovery(**arguments, workers=workers, backend=backend)
        best = min(best, time.perf_counter() - start)
        fingerprint = digest(report)
    return Measurement(
        n_sessions=n_sessions,
        n_trials=n_trials,
        repeats=repeats,
        workers=workers,
        backend=backend,
        seconds=best,
        digest=fingerprint,
    )


def sweep(quick: bool) -> list[Measurement]:
    # Each row scales the cost of one recovery cell. The cell is the unit of scheduling, so
    # what decides whether parallelism pays is the cost of a cell, not the size of the run.
    workloads = (
        [(6, 40, 8), (8, 200, 8), (10, 800, 8)]
        if quick
        else [
            (6, 25, 8),
            (6, 50, 8),
            (8, 100, 8),
            (8, 200, 8),
            (10, 400, 8),
            (10, 800, 8),
            (12, 1_600, 8),
        ]
    )
    worker_counts = (1, 2, 4, 8)
    replicates = 1 if quick else 3
    results: list[Measurement] = []
    for n_sessions, n_trials, repeats in workloads:
        for backend in ("process", "thread"):
            for workers in worker_counts:
                if workers == 1 and backend == "thread":
                    continue
                results.append(
                    measure(
                        n_sessions=n_sessions,
                        n_trials=n_trials,
                        repeats=repeats,
                        workers=workers,
                        backend=backend,
                        replicates=replicates,
                    )
                )
    return results


def report(results: list[Measurement]) -> None:
    serial = {
        (item.n_sessions, item.n_trials, item.repeats): item
        for item in results
        if item.workers == 1
    }
    print(f"cpus={os.cpu_count()}  numpy={np.__version__}")
    print()
    header = (
        f"{'sessions':>8} {'trials':>7} {'cells':>6} {'backend':>8} "
        f"{'workers':>7} {'seconds':>9} {'speedup':>8}  digest"
    )
    print(header)
    print("-" * len(header))
    identical = True
    for item in results:
        key = (item.n_sessions, item.n_trials, item.repeats)
        baseline = serial[key]
        speedup = baseline.seconds / item.seconds
        identical &= item.digest == baseline.digest
        marker = "" if item.digest == baseline.digest else "  <-- DIGEST CHANGED"
        print(
            f"{item.n_sessions:>8} {item.n_trials:>7} {item.n_cells:>6} "
            f"{item.backend:>8} {item.workers:>7} {item.seconds:>9.3f} "
            f"{speedup:>7.2f}x  {item.digest}{marker}"
        )
    print()
    print(
        "every worker count reproduced the serial digest"
        if identical
        else "!! a parallel run changed the report -- this is a bug, not a tuning result"
    )
    print()

    print("crossover (process backend, best worker count per workload):")
    crossover_found = False
    for key, baseline in serial.items():
        cell_cost = baseline.seconds / (2 * baseline.repeats)
        best = min(
            (
                item
                for item in results
                if (item.n_sessions, item.n_trials, item.repeats) == key
                and item.backend == "process"
                and item.workers > 1
            ),
            key=lambda item: item.seconds,
        )
        speedup = baseline.seconds / best.seconds
        verdict = "parallel wins" if speedup > 1.0 else "workers=1 wins"
        if speedup > 1.0 and not crossover_found:
            crossover_found = True
            verdict += "   <-- crossover"
        print(
            f"  {key[1]:>5} trials x {key[0]} sessions | cell ~{cell_cost * 1e3:7.1f} ms | "
            f"best={best.workers} workers | {speedup:.2f}x | {verdict}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="fewer workloads, one replicate")
    arguments = parser.parse_args()
    report(sweep(quick=arguments.quick))


if __name__ == "__main__":
    main()
