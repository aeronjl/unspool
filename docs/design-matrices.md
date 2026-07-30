# Design matrices without hidden learning

`DesignSpec` turns declared source columns into a fixed, labelled numeric matrix. It is
the compositional layer between a [`TaskSpec`](task-contract.md) and models whose linear
predictors use common effects such as stimulus strength, condition, trial history, and
interactions.

```python
from behavio.design import CategoricalTerm, DesignSpec, HistoryTerm, InteractionTerm, NumericTerm

stimulus = NumericTerm("stimulus", center=0.0, scale=100.0)
condition = CategoricalTerm(
    "condition",
    levels=("training", "probe"),
    reference="training",
)
design = DesignSpec(
    terms=(
        stimulus,
        condition,
        HistoryTerm("choice", lags=(1, 2), coding="effect"),
        InteractionTerm(stimulus, condition),
    )
)

matrix = design.build(study)
print(matrix.names)
print(matrix.values.shape)
```

The same design can be written as a [formula](design-formulas.md), which desugars onto
exactly these terms and nothing else:

```python
design = DesignSpec.from_formula(
    "choice ~ numeric(stimulus, scale=100.0) * C(condition, ['training', 'probe'])"
    " + lag(choice, 1, 2)"
)
```

The output contains an immutable two-dimensional array, stable feature names, and the
complete design signature. The same specification can therefore be built on training and
test studies without discovering a different coordinate in each.

Every `DesignTerm` declares `feature_names` and `required_columns` before it sees a study.
`DesignSpec` verifies that `build()` returns exactly those names. Third-party terms
therefore compose with model parameter coordinates and task-role validation without
learning labels from test rows or changing coefficient meaning between datasets.

## Fixed terms

`NumericTerm` applies a declared centring and scaling. It never estimates either quantity.
`CategoricalTerm` requires the complete level set and an explicit or deterministic
reference level. A previously undeclared test category is an error, not a new column.
`InteractionTerm` forms all pairwise products of the named features produced by two terms.

When centring and scaling are not known in advance, fit them on training rows and retain
the returned fixed term:

```python
from behavio.design import DesignSpec, StandardizeTerm

stimulus = StandardizeTerm("stimulus").fit(training_study)
training_matrix = DesignSpec(terms=(stimulus,)).build(training_study)
test_matrix = DesignSpec(terms=(stimulus,)).build(test_study)
```

`StandardizeTerm.fit()` is intentionally separate from `DesignSpec.build()`. The latter
cannot inspect a test study to update the training mean or scale, and constant training
columns fail rather than producing an unstable coordinate.

These restrictions are deliberate. If a centre, scale, category vocabulary, spline basis,
or other transformation must be learned from data, it is a training-only transform. Fit it
inside each prospective training fold and freeze the resulting values before building the
test matrix. Behavio does not inspect the full study and quietly call that preprocessing.

## Histories and reset boundaries

```python
history = HistoryTerm(
    "choice",
    lags=(1, 2, 3),
    reset_by=("subject", "session"),
    coding="effect",
)
```

History is constructed in canonical chronological order and returned in source-row order.
The default reset boundary is `(subject, session)`, so session starts have zero-filled
history. Using `reset_by=("subject",)` explicitly permits the last observation of one
session to enter the first trial of the next. That may be scientifically appropriate, but
it is never inferred.

`coding="identity"` retains a numeric source value. `coding="effect"` requires binary
zero/one observations and maps them to minus/plus one. More elaborate kernels and learned
history representations are not learned implicitly.

`HistoryTerm` defaults to `coding="identity"`, because the term is written by someone who
has already decided what the column means. The [formula](design-formulas.md) spellings
`lag()` and `kernel()` default to `coding="effect"` instead, so that `lag(choice, 1)`
builds the same `choice_lag_1` column as the `choice_lags=` shorthand rather than a
differently scaled one under the same name.

`HistoryKernelTerm` contracts several explicit lags into one feature with fixed weights:

```python
from behavio.design import HistoryKernelTerm

trace = HistoryKernelTerm(
    "choice",
    lags=(1, 2, 3),
    weights=(0.6, 0.3, 0.1),
    reset_by=("subject", "session"),
    coding="effect",
)
```

The weights are part of the design signature. Estimating them from observations is a
model-fitting problem and must happen inside the training fold.

## Relationship to prospective validation

A `DesignSpec` is safe to reuse across folds only because every value that controls its
coordinate is already fixed. It does not make arbitrary preprocessing prospective. The
division is:

1. declare task semantics with `TaskSpec`;
2. fit any data-derived transformation on the training study only;
3. freeze its parameters into fixed design terms;
4. build aligned training and prediction-context matrices;
5. fit and score the model on the rows declared by the split.

This is the same information boundary enforced by Behavio's clocks, landmarks, comparison
procedures, and protocol compiler. Existing first-party models retain their current
model-specific design arguments unless a migration onto these terms has explicit numerical
parity tests.
