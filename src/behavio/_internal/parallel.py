"""Order-preserving parallel execution for position-indexed, order-independent work.

Behavio's value is reproducible scientific evidence, so the only parallelism it offers is
parallelism that cannot change a result. Everything in this module exists to make one
guarantee enforceable: **for any ``workers``, the returned list is the list a serial
comprehension would have produced, element for element.**

Three properties together give that guarantee, and a caller has to supply the first two.

Positional dispatch
    :func:`map_ordered` submits one task per input position and reads the results back by
    position, never by completion. A task that finishes first does not move. Callers that
    accumulate into a list therefore keep the order the inputs declared, which is what
    makes retained failures -- the evidence several of these loops exist to collect --
    land in the same order under every worker count.

Purity of the task
    The mapped function must depend only on its argument. In particular it must not read
    or advance a shared random generator: a loop that draws from one generator as it goes
    is *order-dependent by construction* and cannot be parallelised without changing its
    results. Every caller in this package instead derives each task's seed from
    ``numpy.random.SeedSequence(root).spawn(n)`` indexed by position, which is
    position-determined and so survives being run in any order, or in none.

First failure wins
    Results are collected in index order, so the exception that escapes is the one the
    serial loop would have raised -- the lowest-indexed failure, not the earliest one to
    occur. Later tasks are cancelled once it is known.

``workers=1`` does not construct an executor at all. It is a plain loop, so the default
path costs nothing and cannot be made slower by a scheduler.

Process versus thread
---------------------
:attr:`WorkerBackend.PROCESS` is the default because these are NumPy-bound workloads whose
Python-level fitting code holds the GIL. Threads help only where the work is genuinely
inside a released-GIL BLAS call for most of its duration, which a per-trial recursive
filter is not; :attr:`WorkerBackend.THREAD` is offered for tasks that cannot be pickled and
for measuring the difference rather than assuming it.

Worker processes are started with the ``spawn`` context on every platform. ``fork`` is not
an option: these processes are created after NumPy and its BLAS have already started
threads, and forking a threaded process is unsafe. ``spawn`` also makes the two platforms
behave the same way, which a reproducibility claim needs.

BLAS thread counts are deliberately **not** touched. A spawned worker inherits the parent's
environment and therefore the parent's BLAS threading, so a reduction runs the same way in
a worker as it does in the parent process. Forcing single-threaded BLAS in workers would be
a plausible-looking performance tweak that could change the last bits of a matrix product,
which is exactly the guarantee this module exists to keep.
"""

from __future__ import annotations

import multiprocessing
import pickle
import traceback
from collections.abc import Callable, Sequence
from concurrent.futures import Executor, Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, TypeVar

TaskT = TypeVar("TaskT")
ResultT = TypeVar("ResultT")

__all__ = [
    "ParallelWorkerError",
    "UnpicklableTaskError",
    "WorkerBackend",
    "map_ordered",
    "resolve_workers",
]


class WorkerBackend(StrEnum):
    """Where parallel tasks run.

    ``PROCESS`` sidesteps the GIL and is the default for model fitting. ``THREAD`` shares
    the interpreter, so it pickles nothing and can run closures, lambdas and locally
    defined splitters -- at the cost of contending for the GIL on the Python-level parts of
    a fit.
    """

    PROCESS = "process"
    THREAD = "thread"


class ParallelWorkerError(RuntimeError):
    """A task failed in a worker and its exception could not be sent back intact.

    The common case never reaches this class: a worker's exception is normally picklable,
    so :func:`map_ordered` re-raises *that* object, annotated with the worker's own
    traceback, and a caller's ``except ValueError`` keeps working under any worker count.
    This is the fallback for an exception that cannot cross the process boundary, and for a
    worker that died without raising anything at all -- a segfaulting native library, or an
    out-of-memory kill. In both cases the message says which task position was responsible.
    """


class UnpicklableTaskError(TypeError):
    """A task or its payload cannot be sent to a worker process.

    Raised *before* any work starts, rather than surfacing later as a ``BrokenProcessPool``
    with nothing in it to read. The usual cause is a lambda or a locally defined function
    passed as a splitter, a simulator, or an inference callable; the usual fixes are a
    module-level function, a :func:`functools.partial` of one, or
    ``backend=WorkerBackend.THREAD``, which pickles nothing.
    """


@dataclass(frozen=True, slots=True)
class _WorkerFailure:
    """A task's exception, carried back to the parent by value rather than by raising."""

    error: BaseException | None
    error_type: str
    message: str
    remote_traceback: str


def _run_task(
    function: Callable[[TaskT], ResultT], task: TaskT
) -> ResultT | _WorkerFailure:  # pragma: no cover - the body runs in a worker process
    """Run one task, converting an exception into a value the parent can inspect.

    Returning the failure rather than raising it is what preserves the worker's traceback.
    An exception raised out of a worker is pickled, and pickling drops ``__traceback__``,
    so the parent would otherwise see the right exception type with a traceback pointing
    only at its own ``result()`` call. Formatting the traceback inside the worker captures
    the frames that actually failed.
    """

    try:
        return function(task)
    except BaseException as error:  # re-raised in the parent, in index order
        detail = traceback.format_exc()
        error_type = type(error).__name__
        message = str(error)
        try:
            pickle.loads(pickle.dumps(error))
        except Exception:  # an exception that cannot travel is reported by text
            return _WorkerFailure(None, error_type, message, detail)
        return _WorkerFailure(error, error_type, message, detail)


def resolve_workers(workers: int, *, n_tasks: int) -> int:
    """Validate a worker count and clamp it to the work that actually exists.

    Requesting more workers than tasks is not an error -- a caller sizing a pool from
    ``os.cpu_count()`` should not have to know how many scenarios a run produced -- but
    starting the surplus would pay process startup for nothing.
    """

    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    return max(1, min(workers, n_tasks))


def _executor(backend: WorkerBackend, workers: int) -> Executor:
    if backend is WorkerBackend.THREAD:
        return ThreadPoolExecutor(max_workers=workers)
    return ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
    )


def _require_picklable(function: Callable[[TaskT], ResultT], tasks: Sequence[TaskT]) -> None:
    """Fail early, and by name, on a payload a worker process could never receive."""

    try:
        pickle.dumps(function)
    except Exception as error:
        raise UnpicklableTaskError(
            f"the parallel task function {getattr(function, '__qualname__', function)!r} "
            f"cannot be pickled, so it cannot run in a worker process: {error}"
        ) from error
    for position, task in enumerate(tasks):
        try:
            pickle.dumps(task)
        except Exception as error:
            raise UnpicklableTaskError(
                f"the work item at position {position} cannot be pickled, so it cannot be "
                f"sent to a worker process: {error}. Lambdas, closures and locally defined "
                "functions are the usual cause; pass a module-level function or a "
                "functools.partial of one, or run with backend='thread', which pickles "
                "nothing."
            ) from error


def _raise_worker_failure(position: int, failure: _WorkerFailure) -> None:
    """Re-raise a worker's exception in the parent, keeping its type and its traceback."""

    detail = (
        f"the above exception was raised by parallel task {position}; "
        f"the worker's traceback follows.\n{failure.remote_traceback}"
    )
    error = failure.error
    if error is None:
        raise ParallelWorkerError(
            f"parallel task {position} raised {failure.error_type}: {failure.message}. "
            f"The exception could not be sent back from the worker process, so it is "
            f"reported by text.\n{failure.remote_traceback}"
        )
    error.add_note(detail)
    raise error


def map_ordered(
    function: Callable[[TaskT], ResultT],
    tasks: Sequence[TaskT],
    *,
    workers: int = 1,
    backend: WorkerBackend | str = WorkerBackend.PROCESS,
) -> list[ResultT]:
    """Apply ``function`` to every task and return the results in **input** order.

    The result is required to equal ``[function(task) for task in tasks]`` for every
    ``workers`` and every backend, including the exception that escapes: failures are read
    in index order, so the lowest-indexed one wins exactly as it would in a serial loop.

    ``function`` must be pure in its argument and must not touch shared mutable state or a
    shared random generator. That is a precondition this module cannot check and callers
    must supply; see the module docstring.
    """

    resolved_backend = WorkerBackend(backend)
    items = tuple(tasks)
    resolved = resolve_workers(workers, n_tasks=len(items))
    if not items:
        return []
    if resolved == 1:
        return [function(item) for item in items]
    if resolved_backend is WorkerBackend.PROCESS:
        _require_picklable(function, items)

    futures: list[Future[Any]] = []
    results: list[ResultT] = []
    with _executor(resolved_backend, resolved) as executor:
        futures = [executor.submit(_run_task, function, item) for item in items]
        try:
            for position, future in enumerate(futures):
                try:
                    outcome = future.result()
                except Exception as error:
                    raise ParallelWorkerError(
                        f"parallel task {position} did not return a result and its worker "
                        f"did not raise a reportable exception: {type(error).__name__}: "
                        f"{error}. A worker that dies this way was killed rather than "
                        f"failing -- a segmenting native library or an out-of-memory kill "
                        f"are the usual causes. If this is a script rather than a library "
                        f"call, check that the work is guarded by "
                        f'`if __name__ == "__main__":`, which worker processes require '
                        f"because they are started with spawn and re-import the main "
                        f"module. Re-running with workers=1 executes the task in this "
                        f"process, where it can be debugged."
                    ) from error
                if isinstance(outcome, _WorkerFailure):
                    _raise_worker_failure(position, outcome)
                results.append(outcome)
        finally:
            for pending in futures:
                pending.cancel()
    return results
