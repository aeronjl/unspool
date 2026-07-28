# Prospective model comparison

`compare_models` makes the comparison design—not only individual fits—a first-class
object. Every candidate is evaluated on the same prospective folds, pointwise scores are
aggregated under a declared unit, and the same bootstrap draws are reused for all models
and pairwise differences.

```python
from unspool import compare_models, cohort_forward_session_splits

splits = cohort_forward_session_splits(
    study,
    min_train_sessions=4,
    horizon=1,
)
report = compare_models(
    {
        "static": static_model,
        "shared_smooth": smooth_model,
    },
    study,
    splits,
    aggregation_column="subject",
    bootstrap_resamples=5_000,
    bootstrap_seed=2025,
)

for result in report.model_results:
    print(result.name, result.unit_balanced_log_loss, result.audit_status)
print(report.winner)
```

The point-estimate winner is descriptive. It is not a significance declaration and does
not override the paired interval. A candidate with a failed audit in any fold remains in
the report but is ineligible to win; warnings remain eligible. If every candidate fails,
the comparison winner is `None`. Exact score ties follow candidate insertion order so the
rule is deterministic and visible.

Every estimator declares `scored_columns`, the complete observed event represented by one
pointwise log probability. All candidates in a comparison must declare the same tuple.
This prevents, for example, ranking a choice-only Bernoulli likelihood directly against a
joint choice/response-time density. The binary `outcome_column` used for probabilities and
Brier scores must be one of those scored columns. Both declarations are retained in the
serialized report; see the [estimator contract](estimator-contract.md).

## Aggregation and uncertainty

For each candidate, the report retains:

- every `FoldEvaluation`, including its full fitted result and pointwise scores;
- one normalized fit audit per fold;
- log loss and Brier score for every aggregation unit;
- equal-unit and trial-pooled summaries;
- a percentile bootstrap interval over aggregation units; and
- paired unit-level log-loss differences for every unordered candidate pair.

With the default `aggregation_column="subject"`, trials are averaged within subject and
subjects are then weighted equally. Trials and repeated forecast targets remain weighted
equally *within* a subject. Changing the aggregation column changes the estimand: lab-
balanced and subject-balanced results are not interchangeable.

The bootstrap resamples whole aggregation units, not trials or folds. It therefore
preserves within-subject dependence across sessions and forecasting origins. Its validity
still requires the declared units to be a defensible independent sampling level. It does
not represent hyperparameter, landmark, cohort-selection, or model-class uncertainty.

`PairedModelComparison.left_minus_right` is positive when the right-hand candidate has
lower log loss. `bootstrap_probability_positive` is a descriptive fraction of bootstrap
draws, not a posterior model probability or a calibrated frequentist p-value.

## Serialization and fitted evidence

```python
import json

payload = report.to_dict()
rendered = json.dumps(payload, allow_nan=False)
```

The serialized record contains fold/session provenance, unit scores, intervals, pairwise
directions, and complete audit dictionaries. It deliberately omits potentially large fit
covariance arrays. The in-memory report retains the complete `FoldEvaluation` objects when
parameter estimates or predictions need inspection.

## Training-only nested selection

Candidate grids and hyperparameters must not be selected on the folds used for the final
claim. `nested_select_model` gives the splitter only the outer training study:

<figure class="doc-figure doc-figure--wide" data-figure-kind="Conceptual">
  <img src="../assets/nested-selection.svg" alt="Nested selection diagram in which outer training data are split into inner forecasts, candidates are compared, one procedure is selected and refitted, and the outer future test is opened only once.">
  <figcaption><strong>Nested selection boundary.</strong> Inner forecasts choose the candidate and hyperparameters using only outer-training data. The untouched outer test scores the complete selection procedure.</figcaption>
</figure>

```python
from unspool import nested_select_model


def inner_splitter(outer_training_study):
    return cohort_forward_session_splits(
        outer_training_study,
        min_train_sessions=3,
    )


nested = nested_select_model(
    {
        "static": static_model,
        "smoothness_3": smooth_model,
    },
    study,
    outer_splits=splits,
    inner_splitter=inner_splitter,
    bootstrap_seed=2025,
)
```

For each outer fold, the returned `NestedSelectionFold` retains the entire inner comparison,
the selected name, and the refit outer evaluation. Inner row positions are relative to
`study.take(outer_split.train_indices)`; the outer test rows are structurally absent from
the object passed to `inner_splitter`.

The outer report estimates the performance of the *selection procedure*. It should not be
reported as the performance of one fixed model when different candidates were selected
across folds. Likewise, inspecting outer results and then changing the candidate grid
invalidates the nesting and requires a new untouched evaluation layer or a new study.

The [replicated IBL nested-selection benchmark](https://github.com/aeronjl/unspool/tree/main/benchmarks/ibl2021_nested_selection)
applies this contract to 78 animals across nine labs. It nests both same-animal session
forecasting and unseen-lab future-session forecasting, and retains the exact inner targets,
selected candidate, outer fit audit, and subject-level scores for every outer fold.

## Recovery requirement

Nesting prevents direct test leakage; it does not guarantee reliable selection. The
[nested selection recovery benchmark](https://github.com/aeronjl/unspool/tree/main/benchmarks/nested_selection) tests the
whole procedure under stationary and shared-drift generators. It recovers strong drift in
40/40 outer folds and selects the static model in 37/40 stationary folds, retaining the
remaining resolution errors rather than treating nesting as a certificate.
