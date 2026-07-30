# Extend Behavio without forking it

Downstream packages should own domain-specific readers, models, solvers, and diagnostics.
Behavio owns the small contracts that let those components enter the same longitudinal
validation and evidence workflow. Compatibility is structural; subclassing an internal
base class is not required.

## Choose the smallest extension surface

Every protocol below lives in one place, `behavio.contracts`, alongside the dataclasses
those protocols declare structurally. Import them from there
([API reference](reference/contracts.md)); the modules they used to live in still
re-export them, so older imports keep working.

| You already have | Implement | What becomes available |
| --- | --- | --- |
| A task-specific table or API | a function returning `Study`, `TaskSpec`, and source provenance | common validation, clocks, splitters, and models |
| A file format, archive, or data service | `StudyAdapter` | declared identity and chronology policy, plus a runnable conformance harness |
| A fitted predictive model | `BehaviourEstimator` | prospective evaluation and matched comparison |
| A model that can also simulate | `GenerativeBehaviourModel` | parameter and model recovery |
| A model whose likelihood is a quadratically penalised linear predictor | `PenalisedLinearEstimator` | `smooth()`, `hierarchical()` and `mix()` apply to it, so the time-variation, population and mixture cells of your family exist without being written |
| A simpler process a model could be mixed with | `MixtureComponent` | `mix()` accepts it against any composable family, so one lapse or contaminant serves every model that scores the same observation |
| A natural/optimizer parameter description | `ParameterSpaceProvider` | portable transforms, bounds, priors, and backend adapters |
| An optimizer | `OptimizationBackend` | identical deterministic problems with complete attempt records |
| Posterior samples | `PosteriorResult` or ArviZ adapter | convergence audit, PPC, PSIS-LOO, SBC, and sensitivity |
| A sampler-backed model | `PosteriorBehaviourEstimator` | the sampled counterpart of the estimator contract, plus a declared point-summary projection |
| A behavioural summary | `PredictiveDiscrepancy` | grouped posterior-predictive checks |
| A training/test partition | `ValidationFold` | fold-fitted transforms, evaluation, comparison, and recovery |
| A complete public analysis | literature-recipe contract | documentation and evidence-bundle integration |

Three of these surfaces are optional widenings that only refine what a model already
does. Implement them when they say something true about your estimator, and nothing
changes if you do not:

| You already have | Implement | What becomes available |
| --- | --- | --- |
| A reporting coordinate that is not the estimated one | `NaturalParameterisation` | derived quantities with delta-method standard errors, and recovery reported in both coordinates without pooling them |
| Numbers a scientist would publish that the optimizer never sees | `DerivedQuantity` on `FitResult.derived` | those numbers reach comparison, evidence bundles, and `export_fit` instead of being dropped |
| A multistart optimizer whose restarts you retain | `MultistartFit` | an automatic `RestartAudit` in every fit audit |

`PenalisedLinearEstimator` is the widest of the optional surfaces and the one with the
largest payoff, because implementing it is what makes hierarchy, time-variation and
mixtures *derive* rather than be written three times more. Its five ingredients, and which member carries each, are
documented in [composing models](composing-models.md#the-contract). A family whose row
predicts several numbers rather than one -- a multinomial's per-category logits -- declares
them as `predictor_cells` and is composable on the same terms; a family with per-trial
support restrictions declares them as `-inf` in `predictor_offsets`.

It is genuinely restricted, in two ways rather than one. A likelihood that does not see the
study through a *linear* predictor cannot satisfy it -- a drift-diffusion density composes
its predictors through a Wiener first-passage time, not a link function. And a likelihood
whose row scores are not independent given the predictor cannot satisfy it either, however
its members are shaped: a mixture over latent states scores row *r* through a recursion
over every row before it. The second kind cannot be detected structurally, so a model in
that position declares `penalised_linear_refusal` and the combinators report the reason.

`MixtureComponent` is the smaller of the two composition surfaces and points the other way:
it is what a *process* implements so that any composable model can be mixed with it. It is
five members -- a per-row log density on the model's own outcome coordinate, a prediction of
the model's own shape, a simulator, an identity, and a refusal in a sentence -- and no
estimated parameters at all, because the only thing a mixture estimates is its weight. The
three components that ship are documented in
[composing models](composing-models.md#what-a-component-must-expose).

`required_task_columns` used to appear in that table, behind a separate
`TaskColumnEstimator` protocol. It is now a member of `BehaviourEstimator` itself and the
side protocol is gone. "What columns does this model need?" is the first question anyone
asks of an estimator and every estimator can answer it: a model whose likelihood reads
nothing but the column it scores returns `()`, which is an answer rather than a refusal to
answer. The declaration was optional only to avoid evicting models that had not yet
written it down.

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

## Data-source adapters

When a reader is more than one function -- a format with options, an archive with pinned
identity, a service with credentials -- declare it as a `StudyAdapter`. The adapter *is*
the immutable declaration of what to read; `read()` turns it into a `Study`:

```python
from dataclasses import dataclass
from pathlib import Path

from behavio.contracts.adapter import SessionOrderPolicy, SourceType
from behavio.study import Study


@dataclass(frozen=True, slots=True)
class MyBpodSource:
    path: Path
    session_order: int

    adapter_name = "mylab.bpod"
    adapter_version = "1"
    source_type = SourceType.LOCAL_FILE
    session_order_policy = SessionOrderPolicy.RECORDED

    def read(self) -> Study: ...
```

`session_order_policy` is the load-bearing declaration. `RECORDED` means chronology came
from the caller or from an explicit record in the source. `DERIVED` means the caller named
a rule -- a date column, an explicit ordering, file order -- and the adapter applied and
recorded it. There is no third option: an adapter with neither a record nor a named rule
must fail rather than number sessions by arrival. Behavio's own adapters follow this, which
is why generic NWB requires an explicit `session_order` and why the table reader requires
`session_order_from_column(...)`, `session_order_from_explicit(...)`, or
`session_order_from_appearance()` before it will read a table that lacks the column.

### Adapter conformance tests

`behavio.adapters.conformance` runs the adapter half of the compatibility list below
against your implementation:

```python
from behavio.adapters.conformance import assert_study_adapter_conforms

assert_study_adapter_conforms(
    MyBpodSource(fixture_path, session_order=0),
    expected_rows=[
        {"subject": "m1", "session": "day-1", "trial": 0, "choice": 1},
        {"subject": "m1", "session": "day-1", "trial": 1, "choice": 0},
    ],
    chronology_withheld=lambda: MyBpodSource(fixture_without_chronology),
    require_complete=True,
)
```

It checks that the adapter declares valid identity, returns a valid `Study`, preserves
source trial order and subject/session run boundaries against your fixture, refuses to
fabricate `session_order` when chronology is withheld, produces a chronology that actually
orders the study, and reads the same source twice identically. `check_study_adapter()`
returns the same run as data when you would rather inspect it than assert on it;
`require_complete=True` additionally rejects a run in which you did not supply
`expected_rows` or `chronology_withheld`, which is the mode an adapter's own suite should
use.

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

`EstimatorRegistry` is how a declared `implementation` string becomes an object. A frozen
protocol names each candidate by string, and turning that string into a model is the one
place where data becomes code — so it goes through an explicit allowlist rather than
through `importlib`. `builtin_estimator_registry()` returns the allowlist the `behavio`
command line resolves through; register into a copy of it to add your own:

```python
from behavio import builtin_estimator_registry

registry = builtin_estimator_registry()
registry.add(
    "mylab.models.ExternalModel",
    MyExternalModel,
    provider="my-behaviour-package",
    version="2.1.0",
    produces=MyExternalModel,
)

model = registry.create("mylab.models.ExternalModel", {"history_lags": 2})
manifest = registry.manifest()
```

The registration name is the string protocols declare; for the package's own models that is
their public import path (`behavio.models.BernoulliHistoryGLM`). Registries are
instance-scoped, reject replacement, and serialize provider/version metadata without
serializing executable factories. Extension packages should not mutate a process-global
registry at import time.

### Declare what your factory produces

`produces` is optional but worth supplying, because it is what lets
`verify_candidate_declarations` decide rather than shrug. Before a protocol run, Behavio
checks each supplied estimator against the frozen declaration; a registered implementation
is verified or contradicted by `isinstance`, while an unregistered one can only be compared
by class name against already-imported modules and is often recorded as *unverifiable*.
`model_name` pins the stable name the factory's output must report, which catches a factory
whose identity has drifted from the registration it was made under.

```python
from behavio.runner import run_protocol, verify_candidate_declarations

verification = verify_candidate_declarations(protocol, models, registry=registry)
assert all(item.verified for item in verification)
run = run_protocol(compiled, models)
```

### Combinators and composed candidates

A protocol candidate is one implementation name and a flat list of scalar settings, which
cannot spell a nested constructor call. It can spell a *reference*: `base` names another
registered implementation, and every setting prefixed `base.` configures it, recursively.
That is how a frozen protocol declares `hierarchical(smooth(BernoulliHistoryGLM(...)))`
without the registry growing one entry per composition:

```python
model = registry.create(
    "behavio.compose.hierarchical",
    {
        "base": "behavio.compose.smooth",
        "base.base": "behavio.models.BernoulliHistoryGLM",
        "base.base.covariates": ("stimulus",),
        "base.over": "session_order",
        "base.knots": (0.0, 4.0, 8.0),
        "over": "subject",
    },
)
```

Register a combinator of your own with `base_attribute="..."` naming the attribute your
object exposes its wrapped model under (both built-ins use `"model"`). The declaration check
then verifies `base.`-prefixed settings against the wrapped model rather than reporting them
as fields that do not exist.

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

A model that samples rather than optimizes can additionally implement
`PosteriorBehaviourEstimator`. It mirrors `BehaviourEstimator` exactly, with `fit`
replaced by `sample(study) -> PosteriorResult`, and adds one required method:

```python
from behavio.contracts import PosteriorBehaviourEstimator, posterior_point_summary

assert isinstance(external_model, PosteriorBehaviourEstimator)
posterior = external_model.sample(train_study)
fit = external_model.point_summary(posterior, converged=True)
```

`point_summary` is the explicit, lossy reduction that lets a sampled model enter the
frequentist machinery. `posterior_point_summary` is the reference implementation: it
returns the posterior mean (or median), the posterior standard deviation, and the full
posterior sample covariance, and it records nothing it did not measure. Optimizer-shaped
diagnostics -- iteration count, objective, gradient norm, Hessian condition, boundary
contact -- are set to `None`, meaning *inapplicable*. `FitResult.audit()` skips absent
diagnostics rather than warning about them; do not substitute zeros or NaNs, which would
be recorded as findings against your model.

`evaluate_splits`, `compare_models`, `nested_select_model`, `run_parameter_recovery` and
`run_model_recovery` accept a sampled estimator wherever they accept a frequentist one:

```python
from behavio import compare_models
from behavio.evaluation import PosteriorFoldPolicy

report = compare_models(
    {"sampled": external_model, "baseline": BiasOnly()},
    study,
    splits,
    posterior_policy=PosteriorFoldPolicy(),
)
```

Three rules follow from that:

- **Each fold is audited before it is projected.** Behavio calls `audit_posterior` on the
  fold's posterior and passes the verdict into `point_summary(..., converged=...)`. A
  failed audit therefore produces a failed `FitAudit`, which makes the candidate
  ineligible for `winner` and for model-recovery selection, exactly as a non-converged
  optimizer fit does. The fold's score is still reported so every candidate keeps
  identical aggregation units.
- **`predict` and `pointwise_log_prob` receive the posterior, not the projection.** Return
  the draw-averaged predictive probability and the log pointwise predictive density
  `log((1/S) sum_s p(y_i | theta_s))`; `posterior_log_predictive_density` computes the
  latter from a `(draw, observation)` matrix. Scoring at the posterior mean is a different
  quantity and must not be pooled with honest predictive densities.
- **Declare `posterior_parameter_labels` if you want parameter recovery.**
  `posterior_point_summary` names coordinates (`beta[coefficient='stimulus']`), while
  `simulate` takes scalar parameter names, so the two do not round-trip. The mapping is
  required, validated by `posterior_parameter_columns`, and never guessed. Recovery then
  reports coverage from the posterior quantile interval and labels it
  `posterior-quantile`, so it is never averaged with Wald coverage.

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

A data-source adapter has its own list, and
`behavio.adapters.conformance.assert_study_adapter_conforms` executes all six against a
small committed fixture:

1. stable adapter name, version, declared source type, and chronology policy;
2. `read()` returns a valid `Study`;
3. source trial order is preserved exactly;
4. subject and session run boundaries are preserved, not regrouped or merged;
5. `session_order` is refused rather than fabricated when no record and no named derivation
   are available; and
6. reading an unchanged source twice gives the same study.

Behavio should depend on the interface package only when a capability is broadly useful
and light enough for the core. Heavy solvers and domain-specific models should remain
optional downstream dependencies with their own release cycle.
