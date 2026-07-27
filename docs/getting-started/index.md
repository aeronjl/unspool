# Choose a workflow

Begin with the scientific boundary, not the model class.

For a complete analysis, encode that boundary in a frozen
[`StudyProtocol`](../protocols/index.md). The lower-level examples below remain useful for
interactive exploration, while a protocol adds source and cohort identity, fixed
candidates, recovery gates, bounded claims, and a portable evidence record.

<figure class="doc-figure">
  <img src="../assets/workflow-map.svg" alt="Four routes from a scientific question through a validation boundary to a bounded result: describing change, predicting later sessions, comparing explanations, and testing identifiability.">
  <figcaption><strong>Workflow map.</strong> The intended generalization target determines the split and evidence object before it determines the model family. This is a conceptual contract diagram.</figcaption>
</figure>

## I have a trial table

Map the four identity and chronology columns—`subject`, `session`, `trial`, and
`session_order`—into a [`Study`](../data-contract.md). Source order and additional columns
are retained. Unspool will not infer chronology from filenames or row order.

```python
from unspool import Study

study = Study.from_dataframe(
    trials,
    subject="mouse",
    session="session_id",
    trial="trial_index",
    session_order="training_day",
)
```

## I want to forecast later sessions

Use expanding or cohort-level forward-session splits. Learned clocks, scalers, landmarks,
and model choices must be fitted again inside each training fold.

```python
from unspool import cohort_forward_session_splits, evaluate_splits

splits = cohort_forward_session_splits(study, min_train_sessions=5, horizon=1)
report = evaluate_splits(model, study, splits)
```

Read [prospective validation](../validation.md) before interpreting the score.

## I want to compare explanations

Use [`compare_models`](../comparison.md) for a predeclared candidate set. Use
`nested_select_model` when candidates or hyperparameters are selected from data before the
final forecast.

## I have IBL or NWB data

Optional adapters preserve source identity and provenance without making ONE, PyNWB, or
DANDI core dependencies. See [data and interoperability](../interoperability.md).

## I want a complete example

The [worked studies](../tutorials/index.md) begin with published scientific questions and
carry them through cohort definition, modelling, validation, figures, and bounded
interpretation.

The [Cell 2025](../tutorials/cell2025-learning-trajectories.md) and
[IBL nested-selection](../tutorials/ibl2021-prospective-selection.md) studies are also
expressed as complete protocol migrations with exact numerical parity tests.
