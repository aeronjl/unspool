# Compile and audit the evidence boundary

The compiler turns a frozen declaration and concrete prospective splits into a row-level
execution plan. This is the last review point before fitting. It answers a more precise
question than “is this train/test split disjoint?”: what may be fitted, what historical
context may be supplied at prediction time, what is scored, and what is deliberately
excluded?

## Materialize the declared cohort

```python
from behavio.protocol import materialize_protocol

materialized = materialize_protocol(frozen, source_study)
manifest = materialized.manifest

print(manifest.selected_subjects)
print(manifest.selected_sessions)
print(manifest.selected_observations)
print(manifest.fingerprint)
```

Materialization applies the closed cohort predicates, validates the expected denominator
and panel requirements, checks every declared observation column against its data
contract, resolves source-to-derived row identities, and fingerprints the resulting
canonical `Study`. It retains identities and hashes in the manifest, not raw trial values.

An expected denominator is a guardrail, not decorative metadata. If a release, adapter,
or eligibility rule produces a different number of animals, sessions, or observations,
materialization stops.

The declared `data_type` and `allowed_values` of each observation are guardrails in the
same sense. A protocol that declares `allowed_values=(0, 1)` cannot materialize a column
containing `7`; a column declared `continuous` cannot hold a label. The failure names the
column, the number of offending rows, and examples:

```python
from behavio import validate_observation_contract

for violation in validate_observation_contract(frozen, candidate_study):
    print(violation.rule.value, violation.n_violating_rows, violation.message)
```

Missing observations are not violations by default — behavioural data legitimately
contains omissions and aborted trials. They become violations only for a column whose
declared `allowed_values` set does not name `None`, mirroring how `ChoiceSpec` retains
omissions only when they were declared.

## Compile explicit row roles

```python
from behavio import compile_execution_plan
from behavio.models import model_capabilities

compiled = compile_execution_plan(
    materialized,
    splits,
    capabilities={name: model_capabilities(model) for name, model in models.items()},
)

for fold in compiled.plan.folds:
    print(
        fold.identifier,
        len(fold.fit_rows),
        len(fold.prediction_context_rows),
        len(fold.scored_rows),
        len(fold.excluded_rows),
    )
```

The four roles are distinct:

- `fit_rows` may affect fitted parameters and fold-fitted transforms;
- `prediction_context_rows` may establish allowed histories without being scored;
- `scored_rows` contribute pointwise predictions to the estimand;
- `excluded_rows` are intentionally invisible to both fitting and scoring.

This distinction matters for behavioural models with lags or filters. A prospective model
may need earlier choices to predict a future choice, but that does not authorize future
outcomes, smoothed posterior states, or an outcome-derived learning landmark.

## Read the audit before fitting

```python
for issue in compiled.plan.audit.issues:
    print(issue.level.value, issue.code, issue.fold, issue.message)

if not compiled.plan.audit.passed:
    raise RuntimeError("the declared analysis boundary is not executable")
```

The audit checks:

- temporal overlap and score-before-fit errors;
- subject and group leakage for held-out deployment targets;
- experimental-unit and score-unit compatibility;
- binary-outcome validity whenever inner or outer Brier scoring is declared;
- candidate support for unseen subjects or groups;
- filtered versus smoothed prediction information;
- training-only visibility for learned and outcome-derived transforms;
- nested inner rows remaining wholly inside the outer training study;
- row coverage and duplicate scoring.

Errors prevent the protocol from reaching `audited`. Warnings remain in the plan and
subsequent evidence bundle; they cannot be discarded just because a fit converged.

The audit checks that the runtime *capabilities* match the declaration. It does not see
the estimator objects themselves, so the identity and hyperparameters of each supplied
model are verified one step later, when `run_protocol` is handed the model registry. See
[The model that ran is the model that was declared](index.md#the-model-that-ran-is-the-model-that-was-declared).

## Nested selection

When `protocol.selection` is present, pass an inner splitter that accepts each outer
training `Study`. The compiler maps its local rows back to canonical global identities and
checks them against the untouched outer scoring set.

```python
def inner_splitter(outer_training, outer_fold_index):
    del outer_fold_index
    return cohort_forward_session_splits(
        outer_training,
        min_train_sessions=3,
        horizon=1,
    )


compiled = compile_execution_plan(
    materialized,
    outer_splits,
    capabilities=capabilities,
    inner_splitter=inner_splitter,
)
```

The selected candidate is refitted on the complete outer training study. Only that refit
is evaluated on the outer scoring rows. Candidate selection, tie-breaking, and inner seeds
are therefore part of the frozen scientific procedure rather than post-hoc analysis.

## Boundary failure is a result

A failed audit is not an inconvenience to suppress. It is evidence that the stated
question, source, splitter, and estimator capabilities do not currently form a valid
prospective study. Change the design through a pre-evidence amendment, or report the
failure as a limit of the proposed analysis.
