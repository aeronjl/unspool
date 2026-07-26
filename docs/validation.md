# Longitudinal validation

Unspool's first splitters operate within subjects at complete-session and within-session
resolutions. They answer different questions and expose that difference through
`split.prospective`.

| Splitter | Training data | Test data | Prospective? | Primary use |
| --- | --- | --- | --- | --- |
| `forward_session_splits` | Expanding prefix of sessions | Next `horizon` sessions | Yes | Forecasting behaviour from the past available at that point |
| `within_session_rolling_splits` | Earlier sessions plus current-session prefix | Next `horizon` observed trials | Yes | Online, filtered prediction inside a session |
| `leave_one_session_out_splits` | Every other session for that subject | One complete session | No | Interpolation and sensitivity to a particular session |

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

The split guarantees temporal ordering at session resolution. It does not make a fitted
pipeline prospective by itself. Any learned scaling, feature selection, state alignment,
or behavioural landmark must also be fitted on `train_indices` only. Unspool's first
training-only landmark helper is described in the
[clock and transform guide](clocks-and-transforms.md).

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

## Current boundary

All splitters produce separate folds for each subject. They do not pool other animals into
the training indices, because the library does not yet have a model-aware contract for
hierarchical fitting. Leave-subject-out and leave-lab-out remain roadmap work rather than
hidden assumptions.
