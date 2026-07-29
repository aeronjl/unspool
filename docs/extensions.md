# Extend Behavio without forking it

Downstream packages should own domain-specific readers, models, solvers, and diagnostics.
Behavio owns the small contracts that let those components enter the same longitudinal
validation and evidence workflow. Compatibility is structural; subclassing an internal
base class is not required.

## Choose the smallest extension surface

| You already have | Implement | What becomes available |
| --- | --- | --- |
| A task-specific table or API | a function returning `Study`, `TaskSpec`, and source provenance | common validation, clocks, splitters, and models |
| A fitted predictive model | `BehaviourEstimator` | prospective evaluation and matched comparison |
| A model that can also simulate | `GenerativeBehaviourModel` | parameter and model recovery |
| A natural/optimizer parameter description | `ParameterSpaceProvider` | portable transforms, bounds, priors, and backend adapters |
| An optimizer | `OptimizationBackend` | identical deterministic problems with complete attempt records |
| Posterior samples | `PosteriorResult` or ArviZ adapter | convergence audit, PPC, PSIS-LOO, SBC, and sensitivity |
| A behavioural summary | `PredictiveDiscrepancy` | grouped posterior-predictive checks |
| A complete public analysis | literature-recipe contract | documentation and evidence-bundle integration |

Do not implement simulation merely to satisfy a protocol. A prediction-only external
estimator can be compared prospectively; it becomes eligible for recovery only when its
simulator represents the same named parameters and task semantics.

## Task adapters

A task adapter should return ordinary public objects rather than a package-specific
subclass:

```python
from behavio import ChoiceSpec, RewardSpec, Study, TaskSpec


def read_my_bandit(rows) -> tuple[Study, TaskSpec, dict[str, str]]:
    study = Study(
        {
            "subject": rows["participant"],
            "session": rows["visit"],
            "trial": rows["trial_in_visit"],
            "session_order": rows["visit_order"],
            "choice": rows["action"],
            "reward": rows["outcome"],
        }
    )
    task = TaskSpec(
        choice=ChoiceSpec(options=(0, 1)),
        reward=RewardSpec(minimum=0.0, maximum=1.0),
    )
    task.validate(study)
    return (
        study,
        task,
        {
            "provider": "my-bandit-adapter",
            "source_version": "1.0",
        },
    )
```

The adapter must not sort away source order silently. Map identity and chronology
explicitly, retain source identifiers where licensing permits, validate units and choice
coding, and test duplicated, missing, or contradictory rows. Network retrieval and local
normalization should be separate functions so a checksum-pinned fixture can test parsing.

## Estimator adapters

An estimator supplies stable identity, the complete observed event, supported prediction
modes, and three methods:

```python
from behavio import BehaviourEstimator, model_capabilities

assert isinstance(external_model, BehaviourEstimator)
capabilities = model_capabilities(external_model)

fit = external_model.fit(train_study)
prediction = external_model.predict(test_study, fit)
scores = external_model.pointwise_log_prob(test_study, fit)
```

`fit()` must return an Behavio `FitResult` whose model name, signature, and training-row
count match the estimator. `predict()` returns `Prediction` or `CategoricalPrediction` in
the requested study's source row order. `pointwise_log_prob()` returns one finite value per
row for exactly `scored_columns`.

If an upstream package uses sequence arrays, xarray objects, or its own sample class, keep
that conversion inside the adapter and test boundary resets and row restoration directly.
Rich native results can remain available on a model-specific result subtype; the common
fields are the interoperability floor, not a demand to discard evidence.

## Local registration

Use `EstimatorRegistry` when a protocol or command line needs to create models from
explicit JSON-like configuration:

```python
from behavio import EstimatorRegistry

registry = EstimatorRegistry()
registry.add(
    "my-external-model",
    lambda config: MyExternalModel(**dict(config)),
    provider="my-behaviour-package",
    version="2.1.0",
)

model = registry.create("my-external-model", {"history_lags": 2})
manifest = registry.manifest()
```

The factory's `model_name` must equal its registration name. Registries are instance-scoped,
reject replacement, and serialize provider/version metadata without serializing executable
factories. Extension packages should not mutate a process-global registry at import time.

## Optimization backends

Implement `OptimizationBackend.run(problem)` and return a complete `OptimizationRun`.
Every declared start must produce an `OptimizationAttempt`, including non-finite or failed
attempts. The run records backend name and immutable configuration, selects one finite
attempt deterministically, and never changes the supplied `OptimizationProblem`, parameter
space, objective measure, or task semantics.

An optimizer adapter is not allowed to reinterpret plausible bounds as hard bounds, drop
the MAP Jacobian, change natural versus optimizer density measure, or reseed unrelated
global state without restoration. Test the adapter against the SciPy reference on fixed
problems rather than demanding identical trajectories.

## Posterior and sampler adapters

Convert posterior output into labelled `PosteriorResult` groups with leading `chain` and
`draw` dimensions. Retain, when available:

- natural posterior parameters and their coordinates;
- posterior predictive and observed-data groups;
- pointwise log likelihood aligned to observations;
- sampler diagnostics such as divergences and tree-depth saturation;
- model, inference-library, version, and parameter-space provenance.

The ArviZ/xarray interchange helpers are preferable to handwritten dimension guessing.
Do not flatten chain and draw before convergence diagnostics, and do not represent an
empirical-Bayes fixed quantity as a posterior variable. A sampler becomes eligible for
simulation-based calibration only when a prior simulator and labelled test quantities are
also supplied.

## Predictive discrepancies and diagnostics

A `PredictiveDiscrepancy` has a stable `name`, configuration-specific `signature`, declared
reference tail, and `evaluate(values)` method returning one finite scalar. It receives one
observation vector at a time; grouping by subject or session belongs to the PPC runner so
the same discrepancy can be reused consistently.

New fit diagnostics should preserve raw evidence and stable issue codes. Avoid a plugin-
specific boolean “converged” field that discards restarts, boundary contact, effective
sample sizes, or undefined quantities.

## Literature recipes

An extension can contribute a recipe without contributing any model code. Follow the
[recipe standard](tutorials/recipe-contract.md), call only public APIs, provide a quick CI
path, and archive expensive deterministic outputs with source provenance. Figures must be
generated from those artifacts and registered as empirical or conceptual evidence.

## Compatibility tests

At minimum, an extension package should test:

1. stable model name and configuration signature;
2. correct `scored_columns` and prediction modes;
3. fit identity and training-row count;
4. prediction and pointwise-score length, finiteness, and source-row order;
5. session/subject reset semantics;
6. deterministic local seeds without global RNG leakage;
7. serialization of portable manifests or results;
8. simulation/parameter-name agreement when generative; and
9. end-to-end prospective evaluation on a small fixture.

Behavio should depend on the interface package only when a capability is broadly useful
and light enough for the core. Heavy solvers and domain-specific models should remain
optional downstream dependencies with their own release cycle.
