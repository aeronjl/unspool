# Multinomial and omission-aware choice

`MultinomialLogit` is Behavio's reference likelihood for tasks with more than two valid
actions. It also supplies the package's first explicit no-response likelihood: omissions
can be retained as an additional modeled category instead of being discarded, imputed, or
folded into an arbitrary action.

## Fit three actions on a fixed design

```python
from behavio import ChoiceSpec, DesignSpec, MultinomialLogit, NumericTerm

choice = ChoiceSpec(
    column="action",
    options=("left", "right", "wait"),
    available_options_column="available_actions",
)
model = MultinomialLogit(
    choice=choice,
    design=DesignSpec((NumericTerm("signed_contrast"),)),
    reference="left",
    l2=0.1,
)

fit = model.fit(study)
prediction = model.predict(study, fit)

print(prediction.categories)
print(prediction.probability.shape)  # trials × categories
print(fit.parameters)
```

For reference category (r), every other category (k) has

\[
\eta_{tk}=x_t^\top\beta_k,
\qquad
P(y_t=k)=\frac{\exp(\eta_{tk})}{\sum_j\exp(\eta_{tj})},
\qquad
\eta_{tr}=0.
\]

The parameter coordinate is explicit: names such as
`category['right']::signed_contrast` bind each coefficient to both a category and a
labelled design feature. Changing category order, reference category, design terms, or
regularization changes the model signature, so an old fit cannot be silently reused.

## Trial-specific choice sets

`ChoiceSpec.available_options_column` may contain a sequence of valid actions on every
trial. `MultinomialLogit` assigns unavailable actions probability zero by using a
negative-infinite logit before normalization. The observed action is validated against the
same mask during fitting and scoring.

The availability mask is also usable before outcomes exist:

```python
available = choice.availability(design_study)
simulated = model.simulate(design_study, truth, seed=42)
```

This separates the experiment's offered choice set from its eventual response. It also
prevents a simulator from treating an unavailable action as merely unlikely.

## Retain omissions in the likelihood

```python
choice = ChoiceSpec(
    column="action",
    options=("left", "right", "wait"),
    omission_values=("no_response",),
    available_options_column="available_actions",
)
model = MultinomialLogit(
    choice=choice,
    design=DesignSpec((NumericTerm("signed_contrast"),)),
    include_omission=True,
)
```

All omission representations declared by `ChoiceSpec` are pooled onto one additional
category. `omission_label` selects which declared value is generated during simulation;
by default it is the first declared omission value. The omission category is always
available because it represents failure to emit any valid action, not an offered action.

If omissions occur while `include_omission=False`, fitting and scoring fail with a
`ModelDataError`. They are never silently dropped. A missing value is considered an
omission only if the task explicitly sets `missing_is_omission=True`.

## Common evaluation and recovery

Categorical predictions participate in `evaluate_splits()`, `compare_models()`, the
audited protocol runner, parameter recovery, and model recovery. Fold artifacts retain the
full category coordinate, probability vector, and observed category index for every
scored row.

Behavio reports a multicategory Brier score as

\[
\frac{1}{2}\sum_k(p_{tk}-\mathbb{1}[y_t=k])^2.
\]

The factor of one half keeps the score in ([0,1]) and makes a two-category probability
vector exactly match Behavio's existing scalar binary Brier score. Log loss uses the
ordinary selected-category log probability.

## Interpretation and current boundary

This is a conditional choice baseline, not a learning model. Design terms may include
fixed numeric, categorical, history, kernel, and interaction features for fitting and
filtered prediction. Recursive simulation currently rejects terms that use the choice
outcome's own history, because precomputing those features from an observed choice column
would substitute recorded outcomes for generated ones.

The common protocol artifact currently marks scalar reliability calibration unavailable
for categorical predictions. Log loss, multicategory Brier score, prospective comparison,
and recovery are implemented; classwise and top-label calibration summaries remain a
separate future extension.
