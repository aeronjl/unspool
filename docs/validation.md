# Longitudinal validation

Behavio's splitters operate both within subjects over time and across complete population
units. They answer different questions and retain the held-out unit in every split.

| Splitter | Training data | Test data | Prospective? | Primary use |
| --- | --- | --- | --- | --- |
| `forward_session_splits` | Expanding prefix of sessions | Next `horizon` sessions | Yes | Forecasting behaviour from the past available at that point |
| `cohort_forward_session_splits` | Same expanding session prefix across a cohort | Same future session rank for every eligible subject | Yes | Jointly fitting population or hierarchical models for within-subject forecasts |
| `historical_cohort_forecast_splits` | Complete aligned trajectories from reference animals plus an early prefix from forecast animals | Final aligned sessions from forecast animals | Yes³ | Forecasting a new animal after observing early behaviour, using a completed reference cohort |
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

³ Historical-cohort prospectivity depends on deployment order: reference animals must
have completed training before the forecast animals are observed. It is not an online
same-cohort split.

<figure class="doc-figure doc-figure--wide" data-figure-kind="Conceptual">
  <img src="../assets/validation-splits.svg" alt="A four-panel comparison of forward-session, whole-session, unseen-animal, and held-out-lab future-session validation geometries using blue training blocks, amber test blocks, and grey untargeted sessions.">
  <figcaption><strong>Four generalization targets.</strong> The colored blocks show which observations train and test each procedure. These geometries answer different scientific questions and their scores are not interchangeable.</figcaption>
</figure>

## Forward-session prediction

```python
from behavio import forward_session_splits

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
from behavio import cohort_forward_session_splits, evaluate_splits

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
or behavioural landmark must also be fitted on `train_indices` only. Behavio's first
training-only landmark helper is described in the
[clock and transform guide](clocks-and-transforms.md).
Its uncertainty wrapper uses the same fold helper, so both point landmarks and bootstrap
draws are learned only from training rows. Frozen clock samples can then be applied to the
test side without reading held-out outcomes.

## Historical-cohort forecasting

```python
from behavio import historical_cohort_forecast_splits

splits = historical_cohort_forecast_splits(
    aligned_study,
    context_session_count=8,
    horizon=5,
    n_folds=6,
)
for split in splits:
    assert set(split.reference_subjects).isdisjoint(split.forecast_subjects)
    assert set(split.prediction_context_indices).issubset(split.train_indices)
    assert max(split.context_session_orders) < min(split.test_session_orders)
```

Each deterministic round-robin fold treats one group of animals as the forecast cohort.
Their early aligned sessions are available both for fitting individual effects and as
prediction context; their final sessions are scored; any intervening sessions are absent.
All aligned sessions from the remaining reference animals are available because those
trajectories are assumed to have completed previously. The splitter requires an identical
aligned coordinate grid for every animal and records reference, context, and test session
identities explicitly.

This geometry answers a different question from `cohort_forward_session_splits`. It asks
how well a completed historical cohort and a new animal's observed prefix forecast that
animal's future. Its interpretation relies on exchangeability between historical and new
animals, so cohort drift, protocol changes, or batch effects require a transport analysis
rather than a stronger claim. The [Cell 2025 flagship study](tutorials/cell2025-learning-trajectories.md)
uses this contract to forecast each animal's final five sessions from its first eight
paper days.

## Within-session rolling origins

```python
from behavio import within_session_rolling_splits

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
from behavio import leave_one_session_out_splits

for split in leave_one_session_out_splits(study):
    assert not split.prospective
```

Leaving a complete session out protects session boundaries, but most folds train on data
collected after the held-out session. It must not be described as prospective prediction.
The method is still useful when the estimand is interpolation, robustness to session-level
perturbations, or the influence of individual sessions.

## Population holdout

```python
from behavio import leave_one_lab_out_splits, leave_one_subject_out_splits

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
from behavio import leave_one_lab_out_session_forecast_splits

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

The [replicated IBL prospective benchmark](https://github.com/aeronjl/behavio/tree/main/benchmarks/ibl2021_prospective)
uses this splitter to distinguish future prediction for represented animals from future
prediction in an entirely unseen lab.

Its [nested-selection successor](https://github.com/aeronjl/behavio/tree/main/benchmarks/ibl2021_nested_selection) performs
candidate and smoothness selection on earlier inner forecasts within each outer training
study. The outer held-out lab and future session are absent during selection, so the final
score evaluates the complete training-only procedure.

Fold-fitted, subject-specific landmarks present a stricter boundary. A landmark learned
only for training subjects cannot be applied to a new test subject, and Behavio raises
rather than silently estimating it from held-out data. Population-transferable transforms
must define how new-subject values are obtained using training-fold information alone.

## Every fold names itself

Each split declares an `identifier`. It is derived from the coordinates the fold already
carries, so it cannot disagree with them, and it never depends on the fold's position in
the returned tuple — filtering or reordering the splits renames nothing.

| Splitter | Identifier |
| --- | --- |
| `forward_session_splits` | `forward-session/subject=M1/forecast-sessions=4` |
| `cohort_forward_session_splits` | `cohort-forward-session/train-sessions=5` |
| `historical_cohort_forecast_splits` | `historical-cohort-session-forecast/fold=1-of-3` |
| `within_session_rolling_splits` | `within-session-rolling-origin/subject=M1/session=2/origin-trial=57` |
| `leave_one_session_out_splits` | `leave-one-session-out/subject=M1/held-out-session=2` |
| `leave_one_subject_out_splits` | `leave-one-subject-out/subject=M1` |
| `leave_one_lab_out_splits` | `leave-one-lab-out/lab=cortexlab` |
| `leave_one_lab_out_session_forecast_splits` | `leave-one-lab-out-session-forecast/lab=cortexlab` |

Each name begins with the scheme and then states the one coordinate that distinguishes the
fold from its siblings within a single splitter call: the held-out subject, the held-out
lab, the sessions being forecast, the trial the origin sits at. A cohort fold joins every
eligible subject, so what separates cohort folds is the expanding training prefix; a
historical-cohort fold selects its forecast animals deterministically from a sorted subject
list, so the round-robin position names the same animals on every run and is carried with
`n_folds`, because the same index means different animals under a different fold count.

The name is not decoration. `evaluate_splits` copies it onto
`FoldEvaluation.identifier`; a fold that fails under `FoldFailurePolicy.RETAIN` is recorded
under it; and an [evidence bundle](protocols/evidence-bundles.md) keys its prediction and
audit maps on it. Two folds sharing a name would not be an ambiguity to resolve later — one
of them would simply disappear from the record — so `evaluate_splits` refuses a split set
whose names collide.

`identifier` is a declared member of the
[`ValidationFold`](reference/contracts.md) protocol, which means a splitter you write
yourself must supply one. It used to be read with a `getattr` fallback that numbered
unnamed folds `fold-0000`; the library depended on a name it had never asked any fold for,
which is exactly the hidden name-based contract this package has been removing.

## Current boundary

The hierarchical Bernoulli GLMs can estimate static or smooth population and individual
effects through bounded partial pooling, using population plug-ins for held-out subjects.
Other model families do not yet have hierarchical extensions, and the library does not
align subject-specific latent states. Those remain model-aware contracts rather than
hidden assumptions in a generic splitter.
