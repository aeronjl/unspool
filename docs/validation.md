# Longitudinal validation

Unspool's splitters operate both within subjects over time and across complete population
units. They answer different questions and retain the held-out unit in every split.

| Splitter | Training data | Test data | Prospective? | Primary use |
| --- | --- | --- | --- | --- |
| `forward_session_splits` | Expanding prefix of sessions | Next `horizon` sessions | Yes | Forecasting behaviour from the past available at that point |
| `cohort_forward_session_splits` | Same expanding session prefix across a cohort | Same future session rank for every eligible subject | Yes | Jointly fitting population or hierarchical models for within-subject forecasts |
| `within_session_rolling_splits` | Earlier sessions plus current-session prefix | Next `horizon` observed trials | Yes | Online, filtered prediction inside a session |
| `leave_one_session_out_splits` | Every other session for that subject | One complete session | No | Interpolation and sensitivity to a particular session |
| `leave_one_subject_out_splits` | Every other subject | All trials from one subject | Yes¹ | Generalization to an unseen animal |
| `leave_one_lab_out_splits` | Every other lab | All subjects and trials from one lab | Yes¹ | Generalization across acquisition sites |
| `leave_one_lab_out_session_forecast_splits` | Common session prefix from every other lab | Later common session horizon from one unseen lab | Yes² | Future-session transport to unseen sites and animals |

¹ Population folds exclude every observation from the held-out unit. Their `prospective`
flag therefore means leakage-safe generalization to an unseen subject or lab, not a
within-subject forecast through calendar time.

² The combined population-forecast fold protects both boundaries: the held-out lab is
absent from fitting, and later sessions from the training labs are excluded too.

## Forward-session prediction

```python
from unspool import forward_session_splits

for split in forward_session_splits(
    study,
    min_train_sessions=2,
    horizon=1,
):
    train = study.take(split.train_indices)
    test = study.take(split.test_indices)
    assert split.prospective
    assert max(split.train_session_orders) < min(split.test_session_orders)
```

These are expanding-history folds: later folds may train on sessions that were test data
at earlier forecasting origins. `step` controls the distance between origins and `horizon`
controls the number of consecutive future sessions tested at each origin. Subjects without
enough sessions simply produce no eligible fold.

## Cohort forward-session prediction

```python
from unspool import cohort_forward_session_splits, evaluate_splits

splits = cohort_forward_session_splits(
    study,
    min_train_sessions=5,
    horizon=1,
)
results = evaluate_splits(hierarchical_model, study, splits)
```

Each cohort fold contains all training and test rows for a shared session-rank origin, so
population and hierarchical models are fitted jointly rather than once per animal. Source
session identifiers and orders remain available in mappings keyed by subject. By default,
fold generation stops as soon as any subject lacks the requested future horizon; this
keeps the estimand's cohort fixed across origins. Set `require_all_subjects=False` only
when a transparently shrinking, follow-up-dependent cohort is intended.

Both forward-session splitters guarantee temporal ordering at session resolution. That
does not make a fitted pipeline prospective by itself. Any learned scaling, feature
selection, state alignment,
or behavioural landmark must also be fitted on `train_indices` only. Unspool's first
training-only landmark helper is described in the
[clock and transform guide](clocks-and-transforms.md).
Its uncertainty wrapper uses the same fold helper, so both point landmarks and bootstrap
draws are learned only from training rows. Frozen clock samples can then be applied to the
test side without reading held-out outcomes.

## Within-session rolling origins

```python
from unspool import within_session_rolling_splits

splits = within_session_rolling_splits(
    study,
    min_train_sessions=2,
    min_train_trials=100,
    horizon=10,
    step=25,
)
```

For each eligible session, training begins with every earlier complete session and the
first `min_train_trials` observed trials of the current session. Each origin tests the next
`horizon` observed trials; `step` advances the number of current-session trials available
to the next origin. Gaps in trial identifiers remain gaps: these arguments count observed
rows rather than manufacturing missing trials. `min_train_sessions=0` permits origins in a
subject's first session.

The current-session prefix appears in both `train_indices` and
`prediction_context_indices`. This is intentional. It is available when fitting occurs and
is replayed as observed context during filtered prediction, but it is not included in the
reported predictions or scores. Consequently, the first test trial can use the actual
pre-origin choice history instead of being treated as a new session.

For a test horizon greater than one, filtered evaluation is sequential: after each test
choice is observed, it can condition the one-step-ahead prediction for the next test trial.
This is not a joint open-loop forecast of all choices at the origin; such a forecast would
require model-specific integration over unobserved intermediate choices.

## Whole-session holdout

```python
from unspool import leave_one_session_out_splits

for split in leave_one_session_out_splits(study):
    assert not split.prospective
```

Leaving a complete session out protects session boundaries, but most folds train on data
collected after the held-out session. It must not be described as prospective prediction.
The method is still useful when the estimand is interpolation, robustness to session-level
perturbations, or the influence of individual sessions.

## Population holdout

```python
from unspool import leave_one_lab_out_splits, leave_one_subject_out_splits

for split in leave_one_subject_out_splits(study):
    assert set(split.train_subjects).isdisjoint(split.test_subjects)

for split in leave_one_lab_out_splits(study, lab_column="lab"):
    assert set(split.train_subjects).isdisjoint(split.test_subjects)
    assert set(split.train_groups).isdisjoint(split.test_groups)
```

Both splitters preserve source row positions and hold out complete subjects. Lab holdout
also requires every subject to map to exactly one non-missing lab; a subject appearing in
more than one lab is rejected instead of leaking across the fold. Studies with fewer than
two eligible subjects or labs produce no folds.

Population folds can be passed directly to `evaluate_splits`. That does not make every
model population-aware: the fitted model must define parameters that can be shared across
training subjects and applied to an unseen subject. For example, the static GLM supports
shared coefficients, while the smooth GLM requires an explicit `shared_trajectory=True`
choice before fitting multiple subjects. The hierarchical smooth GLM instead learns a
population trajectory and subject-deviation trajectories, then applies only the population
trajectory to a held-out subject.

## Held-out-lab future-session prediction

```python
from unspool import leave_one_lab_out_session_forecast_splits

splits = leave_one_lab_out_session_forecast_splits(
    aligned_study,
    train_session_count=5,
    horizon=1,
    lab_column="lab",
)
for split in splits:
    assert set(split.train_subjects).isdisjoint(split.test_subjects)
    assert set(split.train_groups).isdisjoint(split.test_groups)
    assert max(split.train_session_orders) < min(split.test_session_orders)
```

This is stricter than ordinary lab holdout. Training uses only the common aligned session
prefix in the other labs, while testing uses only the later common horizon in the held-out
lab. Every subject must share those explicit `session_order` coordinates; unequal raw
calendars must first be aligned by a scientifically declared, leakage-safe transform.
Subjects that do not reach the horizon cause an error rather than silently changing cohort
membership. Sessions before the horizon from test animals are withheld rather than used as
prediction context, so hierarchical models must apply their declared unseen-subject policy.

The [replicated IBL prospective benchmark](../benchmarks/ibl2021_prospective/README.md)
uses this splitter to distinguish future prediction for represented animals from future
prediction in an entirely unseen lab.

Its [nested-selection successor](../benchmarks/ibl2021_nested_selection/README.md) performs
candidate and smoothness selection on earlier inner forecasts within each outer training
study. The outer held-out lab and future session are absent during selection, so the final
score evaluates the complete training-only procedure.

Fold-fitted, subject-specific landmarks present a stricter boundary. A landmark learned
only for training subjects cannot be applied to a new test subject, and Unspool raises
rather than silently estimating it from held-out data. Population-transferable transforms
must define how new-subject values are obtained using training-fold information alone.

## Current boundary

The hierarchical Bernoulli GLMs can estimate static or smooth population and individual
effects through bounded partial pooling, using population plug-ins for held-out subjects.
Other model families do not yet have hierarchical extensions, and the library does not
align subject-specific latent states. Those remain model-aware contracts rather than
hidden assumptions in a generic splitter.
