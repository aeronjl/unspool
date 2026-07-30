# Running work in parallel

Recovery experiments, simulation-based calibration and nested selection are the expensive
things Behavio does, and they are expensive in the same way: the same fit, repeated over
independent cells. Those loops accept a `workers` argument.

Parallelism here is **opt-in and result-preserving**. It is not a mode, an approximation,
or a speed/accuracy trade. `workers=8` must produce the same artifact as `workers=1`, and
if it ever does not, that is a bug in Behavio rather than a tuning problem.

```python
report = run_model_recovery(
    design,
    scenarios,
    candidates,
    repeats=200,
    seed=12,
    workers=8,
)
```

## The determinism guarantee

For every entry point below, and for any `workers` and any `backend`:

> The returned report is **bit-identical** to the report `workers=1` produces, including
> the order of every retained failure and every diagnostic.

That is a stronger claim than "statistically equivalent" or "equal within tolerance", and
it is what the package's evidence bundles need: a content-addressed artifact whose
fingerprint changed because you used more cores would not be reproducible evidence.

Three properties hold it up.

**Seeds come from position, not from iteration order.** Every parallelised loop derives its
randomness from `numpy.random.SeedSequence(seed).spawn(n)` indexed by the cell's position,
before any work begins. Cell 7 gets cell 7's seed whether it runs first, last, or on
another core. A loop that instead drew from one shared generator as it went would be
order-dependent by construction, and no amount of careful scheduling could parallelise it
without changing its numbers — so those loops are not parallelised. See
[what is not parallelised](#what-is-not-parallelised).

**Results are collected by position, never by completion.** Work is dispatched one task per
index and read back by index. A task that finishes first does not move to the front. This
is what keeps *retained failures* in order: several of these loops deliberately keep
failures as evidence rather than aborting, and a run that reported them in completion order
would have changed the artifact even though it found the same failures.

**The first failure is the lowest-indexed one.** When a cell raises, the exception that
escapes is the one a serial loop would have raised — the earliest cell in input order, not
the earliest one to fail in wall-clock time.

### The limits of the guarantee

The guarantee covers Behavio's scheduling. It does not extend to:

- **A different machine, BLAS, or NumPy build.** Floating-point reductions depend on the
  library doing them. Behavio deliberately does *not* change BLAS thread counts in workers
  — a worker inherits the parent's environment, so a matrix product runs the same way in
  both — but two different machines were never going to agree bit-for-bit and parallelism
  does not change that.
- **A model whose `fit` is not deterministic.** Everything here assumes that fitting the
  same model to the same rows twice gives the same answer. Behavio's own estimators do; a
  custom estimator that seeds itself from the clock, or reads global mutable state, breaks
  the guarantee at `workers=1` too.
- **A model that mutates shared state.** Under the thread backend, candidates share one
  interpreter. A well-behaved estimator is a frozen declaration and has nothing to mutate,
  which is why the contracts require one.

## Where `workers` is accepted

| Function | Parallel over | Notes |
| --- | --- | --- |
| `run_model_recovery` | `scenarios x repeats` cells | The embarrassingly parallel level; most work per task. |
| `run_model_recovery_grid` | forwarded into each design cell | Design cells differ wildly in cost, so parallelising *within* a cell keeps workers busy. |
| `run_simulation_based_calibration` | replicates | Ranks and failures both assembled in replicate order. |
| `compare_models` | candidates | Bounded by the candidate count; two candidates cannot use four workers. |
| `nested_select_model` | outer folds | The deepest loop: every candidate refitted on every inner fold of every outer fold. |

`nested_select_model` runs its inner comparison with `workers=1` on purpose. Outer folds
already saturate the pool, and nesting a second executor inside a worker oversubscribes the
machine.

## Process or thread

`backend="process"` is the default. Measure before choosing anything else.

Model fitting in Behavio is a mix of NumPy calls and Python-level loops — recursive history
updates, per-fold bookkeeping — and the Python-level part holds the GIL. Threads therefore
contend where processes overlap. Measured on a 10-core Apple Silicon machine (NumPy 2.3.5),
16 recovery cells per row, best of three runs:

| Cell cost | `workers=1` | best process | best thread |
| --- | --- | --- | --- |
| 8.7 ms | 1.00x | 0.26x (2 workers) | 0.92x (2 workers) |
| 13.2 ms | 1.00x | 0.33x (2 workers) | 0.90x (2 workers) |
| 44.8 ms | 1.00x | 0.84x (4 workers) | 0.95x (2 workers) |
| 81.2 ms | 1.00x | **1.29x** (4 workers) | 0.93x (2 workers) |
| 416 ms | 1.00x | **2.77x** (4 workers) | 1.58x (2 workers) |
| 477 ms | 1.00x | **2.11x** (8 workers) | 0.98x (2 workers) |
| 1723 ms | 1.00x | **3.05x** (8 workers) | 1.13x (2 workers) |

Every one of those 49 configurations reproduced the serial run's digest exactly.

Threads beat serial only twice, on the two largest cells, and never beat processes at any
size. Reach for `backend="thread"` when the process backend is not available to you — when
something in the call cannot be pickled — or when your inference callable spends its time
inside a sampler that releases the GIL, which the table above does not measure.

Reproduce the table with:

```bash
uv run python scripts/parallel_scaling.py
```

### The crossover

**Below roughly 50 ms of work per cell, `workers=1` is faster.** The measured crossover sits
between 45 ms (0.84x — still losing) and 81 ms (1.29x — winning).

Process startup uses the `spawn` method, so each worker is a fresh interpreter that
re-imports Behavio and NumPy, and the design study and candidate models are pickled to each
one. For a fast fit that overhead dominates: at 8.7 ms per cell, two workers made the run
nearly *four times slower*.

This is why `workers=1` is the default. A small run cannot be made slower by a scheduler it
never starts — at `workers=1` no executor is constructed at all, and the code path is a
plain loop.

Speedup stays sublinear well above the crossover: 3.05x from 8 workers on 10 cores, at cells
costing 1.7 s each. Pickling the design out to each worker and the results back is real work
that a serial run never does, and more workers do not make it smaller.

## Pickling

The process backend sends your objects to worker processes, so they must pickle.

**These do pickle**, and are tested to:

- `Study` — its columns cross the boundary and the immutability contract is re-established
  on arrival, so a study that arrived from a worker is as read-only as one built locally.
- Every estimator in the catalogue.
- The combinators. `smooth(...)` and `hierarchical(...)` build frozen dataclasses whose
  fields are the wrapped estimator and plain scalars — they are wrappers, not closures, and
  hold no bound methods or lambdas.
- `ModelRecoveryScenario`, validation folds, fit results and posterior results.

**These do not**: lambdas, closures, and functions defined inside another function. The
places you are most likely to meet this are the callable arguments — `splitter`,
`inner_splitter`, `simulator`, `inference` — because a lambda is the natural way to write
one:

```python
# Refused, with an error naming the argument, before any fitting starts.
by_lambda = lambda study: forward_session_splits(study, min_train_sessions=2)
nested_select_model(..., inner_splitter=by_lambda, workers=4)

# Fine: a partial of a module-level function.
by_partial = functools.partial(forward_session_splits, min_train_sessions=2)
nested_select_model(..., inner_splitter=by_partial, workers=4)

# Also fine: threads pickle nothing.
nested_select_model(..., inner_splitter=by_lambda, workers=4, backend="thread")
```

Behavio checks picklability *before* dispatching any work and raises `UnpicklableTaskError`
naming what failed, rather than letting the pool break later with nothing in it to read.

### Scripts need a `__main__` guard

Worker processes are started with `spawn`, which re-imports the main module. A script that
calls a parallel entry point at import time will recurse. Guard it:

```python
if __name__ == "__main__":
    report = run_model_recovery(..., workers=8)
```

## When a worker fails

A worker's exception is re-raised in the parent **with its original type**, so
`except ValueError` keeps working at any worker count, and with the worker's own traceback
attached as a note — the frames that actually failed, not just the parent's `result()`
call.

Two cases fall back to `ParallelWorkerError`: an exception that cannot be pickled, which is
then reported by text; and a worker that was *killed* rather than raising — a segfaulting
native library, an out-of-memory kill, or the missing `__main__` guard above. The message
names the task position and suggests re-running that task with `workers=1`, where it
executes in this process and can be debugged normally.

## Why there is no fit cache

Nested selection refits every candidate on every inner fold of every outer fold, and those
training sets genuinely repeat across outer folds. A memo keyed on
`(model signature, fold row indices)` is the obvious optimisation, and `signature` is the
package's scientific fingerprint, so it looks like exactly the right key.

**It is not a sufficient key, so there is no cache.** A signature records the scientific
declaration, not the numerical procedure. Two `BernoulliHistoryGLM` candidates that differ
only in `max_iterations` or `tolerance` have identical signatures and produce different
estimates:

```python
loose = BernoulliHistoryGLM(covariates=("stimulus",), max_iterations=2)
tight = BernoulliHistoryGLM(covariates=("stimulus",), max_iterations=1_000)

loose.signature == tight.signature  # True
loose.fit(study).estimates == tight.fit(study).estimates  # not equal
```

Comparing two such candidates is a legitimate thing to do — convergence behaviour is a real
question about a model — and a memo keyed this way would silently serve the first one's fit
to the second and report them as identical. Silently returning the wrong fit is a worse
failure than a slow one, and it is the specific failure this package exists to prevent.

The constraint is checked by a test, so if the signature is ever widened to cover the
numerical settings, that test fails and caching can be reconsidered on the evidence. Two
further points would still need settling: a memo does not survive a process pool, so it
would compete with the parallelism above rather than compose with it; and the memory cost
of retaining fit results for a long run is unbounded.

The fits themselves *are* deterministic — the same model on the same rows gives bitwise
identical estimates, which is what makes the parallel guarantee possible. The missing piece
is a key, not reproducibility.

## What is not parallelised

**The fold loop inside `evaluate_splits`.** Folds are cheap relative to process startup, so
this is below the crossover by construction. It is also where the ordering guarantees live:
the loop assigns fold identifiers and refuses a split set with duplicate names, reporting
the position that broke it. Parallelising it would change which error a caller sees for a
malformed split set, in exchange for a speedup that measurement says is negative.

**`run_protocol` and `run_nested_protocol`.** These are reachable only from an audited,
frozen protocol, and they build a content-addressed evidence bundle. The candidate loop is
structurally the same shape as `compare_models` and could be parallelised the same way, but
the value at stake — a fingerprinted artifact — is highest exactly where the confidence is
lowest, and the protocol path has no benchmark yet establishing that its candidates cost
more than the crossover. Users who want candidate-level parallelism can get it today
through `compare_models`.
