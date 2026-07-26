# Longitudinal validation

The first Unspool splitters operate within subjects and hold out complete sessions. They
answer two different questions and expose that difference through `split.prospective`.

| Splitter | Training data | Test data | Prospective? | Primary use |
| --- | --- | --- | --- | --- |
| `forward_session_splits` | Expanding prefix of sessions | Next `horizon` sessions | Yes | Forecasting behaviour from the past available at that point |
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

Both splitters produce separate folds for each subject. They do not pool other animals into
the training indices, because the library does not yet have a model-aware contract for
hierarchical fitting. Leave-subject-out, leave-lab-out, and within-session rolling-origin
remain roadmap work rather than hidden assumptions.
