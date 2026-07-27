# Prospective model recovery

Parameter recovery asks whether a model can estimate known quantities when its own family
generated the data. Model recovery asks a different question: under a specified design,
can an evaluation procedure distinguish competing generative families?

Unspool answers that question by simulation and forward-session prediction. It does not
select the model that best redescribes the complete simulated record.

## Experiment contract

Each `ModelRecoveryScenario` has:

- a unique condition name;
- a `truth_label` identifying the candidate family expected to win;
- a generator satisfying `GenerativeBehaviourModel`;
- one exact, named parameter set for that generator.

Candidate labels and configurations are fixed before simulation. For every scenario and
repeat, `run_model_recovery` then:

1. derives and records an independent child seed;
2. simulates choices under the scenario generator;
3. constructs expanding forward-session folds;
4. refits every candidate independently at every forecasting origin;
5. pools pointwise test log probabilities across folds;
6. aggregates every fold's `FitAudit` without discarding warnings;
7. selects the numerically usable candidate with the highest prospective mean log
   probability.

An unresolved result is retained when every candidate has a failing audit or when the best
two usable scores fall within `tie_tolerance`. Warnings remain eligible because they can
limit uncertainty or interpretation without invalidating filtered prediction. Failing
candidates are excluded from selection, while their fold-level optimizer messages, audit
statuses, and stable issue codes remain in the report.

## Static versus smooth example

```python
from unspool import ModelRecoveryScenario, run_model_recovery

scenarios = [
    ModelRecoveryScenario(
        name="stationary",
        truth_label="static",
        generator=static_model,
        parameters=static_truth,
    ),
    ModelRecoveryScenario(
        name="drifting",
        truth_label="smooth",
        generator=smooth_model,
        parameters=smooth_truth,
    ),
]

report = run_model_recovery(
    design,
    scenarios,
    {"static": static_model, "smooth": smooth_model},
    repeats=100,
    seed=123,
    min_train_sessions=3,
    tie_tolerance=0.001,
)
matrix = report.confusion_matrix()
```

The confusion matrix has one row per candidate truth and one column per selected candidate,
plus an explicit `unresolved` column. Rates use every generated run as the row denominator;
ties and failures therefore cannot disappear through conditional reporting. The report also
provides overall accuracy, resolution rate, and accuracy conditional on resolution.

Raw run-level state is retained: scenario and truth labels, generator parameters, generator
and candidate signatures, the common scored columns, child seeds, mean log probabilities,
convergence flags, failure messages, audit statuses and issue codes, fold counts, and all
splitter settings. This
makes alternative tie rules or summaries possible without rerunning the simulations.

When multiple parameter regimes share a generating family,
`report.scenario_confusion_matrix()` retains one row per named scenario rather than
collapsing them into the family-level confusion matrix. This is essential for testing
limiting cases: aggregate family accuracy can otherwise hide which parameter regimes are
actually distinguishable.

## Named design grids

`run_model_recovery_grid` applies the same scenarios, candidates, split settings, and tie
rule to a mapping of named `Study` designs. Each design receives an independent recorded
child seed. `ModelRecoveryGridReport` retains the complete per-design reports and provides
one summary row per cell with trial/subject counts, resolution and accuracy, plus audit
warning and failure rates.

```python
from unspool import run_model_recovery_grid

grid = run_model_recovery_grid(
    {"sparse": sparse_design, "dense": dense_design},
    scenarios,
    candidates,
    repeats=20,
    seed=123,
    min_train_sessions=3,
)

for row in grid.summary():
    print(row.design_name, row.overall_accuracy, row.audit_warning_rate)
```

The first bounded [four-family recovery benchmark](https://github.com/aeronjl/unspool/tree/main/benchmarks/recovery_grid)
uses static, smooth, GLM-HMM, and Q-learning candidates on nested 150- and 300-trial
designs. The smaller cell recovers two of four generating families; the larger recovers
all four for the exact single-run parameter regimes. The contrast is evidence that the
answer changes with the design—not an estimate of a general sample-size threshold.

The follow-up [weak-signal benchmark](https://github.com/aeronjl/unspool/tree/main/benchmarks/weak_signal_recovery) fixes
the 300-trial design and repeats each stronger and boundary-near regime ten times. Recovery
falls from 70.0% to 32.5%; the scenario matrix shows subtle drift collapsing toward static
fits and overlapping HMM emissions collapsing toward static or smooth fits. Wilson
intervals quantify finite-simulation uncertainty without treating the chosen parameter
regimes as a population sample.

## What the matrix does—and does not—show

Recovery is conditional on all of the following:

- parameter sets and their frequency;
- subjects, sessions, trials, covariates, and missingness in the supplied design;
- candidate hyperparameters, including smoothness and temporal knots;
- forecasting origins, horizon, and step;
- the candidate set and score tolerance.

Easy recovery from extreme stationary and drifting paths is weak evidence. Useful grids
must concentrate on scientifically plausible paths and on the boundary where families
mimic one another. Conversely, poor recovery does not prove that both mechanisms are false;
it shows that this design and evaluation procedure cannot reliably distinguish them.

This is a closed-world experiment: one listed family generated every run. Real behaviour
may be generated by none of them. Static and smooth models should later compete with task
history controls, observable strategies, reinforcement learning, and discrete latent-state
models under the same prospective interface.

`BernoulliGLMHMM` can be used as either a scenario generator or candidate. Its packed
parameters are canonically label-ordered, but recovery conclusions must still inspect
restart convergence, state occupancy, emission separation, and label-order ambiguity. See
the [GLM-HMM guide](glm-hmm.md). When latent simulation truth is available,
`state_recovery()` performs a separate permutation-invariant assignment and reports its
best-versus-runner-up gap. Canonical parameter order and recovered state identity are not
treated as the same claim.

`BinaryQLearning` likewise satisfies both recovery interfaces. Generative designs must
include explicit action-contingent reward probabilities, while fitting conditions only on
the resulting observed choices and rewards. Recovery should vary reward volatility because
learning rate and inverse temperature are often weakly distinguished by stationary tasks.
See the [Q-learning guide](q-learning.md).

Smoothness, knots, preprocessing, and landmarks must be chosen a priori or inside a nested
training procedure. Choosing them on the same test folds used in the recovery matrix makes
the matrix optimistic. With `horizon > 1` and `step < horizon`, test sessions may also occur
at more than one forecasting origin; pointwise scores then intentionally weight those
repeated forecast targets more than once.

Run the complete example with:

```bash
uv run python examples/model_recovery.py
uv run python -m benchmarks.recovery_grid.benchmark
uv run python -m benchmarks.weak_signal_recovery.benchmark
uv run python -m benchmarks.state_alignment.benchmark
```
