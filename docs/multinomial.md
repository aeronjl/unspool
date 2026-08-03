# Multinomial and omission-aware choice

`MultinomialLogit` is Behavio's reference likelihood for tasks with more than two valid
actions. It also supplies the package's first explicit no-response likelihood: omissions
can be retained as an additional modeled category instead of being discarded, imputed, or
folded into an arbitrary action.

## Fit three actions on a fixed design

```python
from behavio import ChoiceSpec, MultinomialLogit
from behavio.design import DesignSpec, NumericTerm

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

## Drift and between-animal variation

`MultinomialLogit` satisfies
[`PenalisedLinearEstimator`](composing-models.md#the-contract), so the combinators
apply to it and there is no smooth, hierarchical or hierarchical-smooth multinomial *class*
to reach for:

```python
from behavio.compose import hierarchical, smooth

drifting = smooth(model, over="session_order", knots=(0.0, 4.0), smoothness=1.0)
pooled = hierarchical(model, over="subject", scale=0.4)
both = hierarchical(drifting, over="subject", scale=0.4)
```

What made this possible is one widening of the composition contract rather than a second
contract for categorical models. A multinomial's linear predictor is one number per
category instead of one per row, which it declares as `predictor_cells`; its design is
correspondingly `(trials, categories, parameters)`. Everything else -- the parameter
coordinate, the penalty, the group expansion -- is the same object the binary families use.

Parameter names compose mechanically, because the per-category structure was always in the
name:

| Model | A parameter |
| --- | --- |
| `model` | `category['right']::signed_contrast` |
| `smooth(model, over="session_order", knots=(0, 4))` | `category['right']::signed_contrast[session_order=0]` |
| `hierarchical(model, over="subject")` | `category['right']::signed_contrast` (the population coordinate, unchanged) |

Letting a single category vary by subject does not require writing `repr` quoting into a
string literal:

```python
hierarchical(model, over="subject", parameters=model.category_parameter_names("wait"))
```

### Availability and omissions under composition

Availability reaches the likelihood as a `-inf` offset on the unavailable category's cell,
and a combinator carries offsets through untouched. Two consequences worth stating, because
both are modelling answers and not broadcasting accidents:

* A trial that did not offer a category contributes **nothing** to that category's
  coefficients -- zero probability, zero gradient, zero curvature -- at population level and
  at group level alike.
* A subject who was never offered a category gets a deviation of exactly zero on that
  category's coefficients, with the prior standard deviation as its standard error. The
  population coefficients for that category remain identified by the subjects who *were*
  offered it.

The omission category is always available, so with `include_omission=True` a lapse rate is
an ordinary coefficient: it can vary by subject, or follow a path in session time, on the
same terms as a stimulus sensitivity.

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

This is a conditional choice baseline on its own, though `smooth()` makes its coefficients
non-stationary and `hierarchical()` makes them animal-specific. Design terms may include
fixed numeric, categorical, history, kernel, and interaction features for fitting and
filtered prediction. Recursive simulation currently rejects terms that use the choice
outcome's own history, because precomputing those features from an observed choice column
would substitute recorded outcomes for generated ones.

The common protocol artifact retains three categorical calibration estimands separately:

- confidence calibration pools the maximum predicted probability against correctness;
- top-label calibration conditions that curve on each predicted category; and
- classwise calibration compares each category's probability with its one-vs-rest outcome.

Every curve retains its populated equal-width reliability bins and descriptive ECE.
Multicategory Brier score remains the proper probability score. None of these calibration
notions implies the others, so the artifact does not collapse them into one binary curve.
