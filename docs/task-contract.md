# The behavioural task contract

`Study` says which trial occurred when. `TaskSpec` says what was observed on that trial.
Keeping these contracts separate lets one longitudinal table support several scientifically
different models without hiding task semantics inside model-specific preprocessing.

## A first fit

```python
from unspool import (
    BernoulliHistoryGLM,
    ChoiceSpec,
    Study,
    TaskSpec,
    fit_model,
)

study = Study.from_dataframe(
    trials,
    subject="mouse",
    session="session_id",
    trial="trial_index",
    session_order="training_day",
)
task = TaskSpec(
    choice=ChoiceSpec(options=(0, 1)),
    predictors=("stimulus",),
)
model = BernoulliHistoryGLM(
    covariates=("stimulus",),
    choice_lags=1,
)

fitted = fit_model(model, study, task=task)
print(fitted.validation.n_trials)
print(fitted.result.parameters)
print(fitted.audit().status)
```

`fit_model()` is the small interactive golden path. It validates the task, confirms that
the estimator scores observations declared by that task, fits the model, checks result
identity and row counts, and returns the task denominators beside the raw `FitResult`.
Models may also declare `required_task_columns` for predictive context that they consume
but do not score—for example, reward in a win-stay/lose-shift model. Those columns must
have an explicit predictor, reward, response-time, block, episode, or availability role.
Prospective scientific claims should still use the fold-aware comparison or protocol
runner, which refits the model independently inside every training fold. Reusable numeric,
categorical, interaction, and history terms are described in the
[fixed design-matrix contract](design-matrices.md).

Use `export_fit(fitted, study)` when the interactive result needs a deterministic,
non-executable record with task, data, version, parameter, and diagnostic provenance. See
[fit artifacts and extension registries](fit-artifacts.md).

## Choices are a coordinate, not an incidental encoding

```python
from unspool import ChoiceSpec

choice = ChoiceSpec(
    column="action",
    options=("left", "right", "wait"),
    omission_values=("no_response",),
    missing_is_omission=False,
    available_options_column="available_actions",
)
encoded = choice.read(study)
```

`options` fixes one stable categorical coordinate. `encoded.codes` contains the declared
option position and uses `-1` only for explicit omissions. `encoded.available` is a
trial-by-option mask. An observed action that was unavailable on that trial is rejected.
The source choice column is never silently rewritten.

Missing values are not omissions by default. Setting `missing_is_omission=True` is an
explicit scientific assertion that the missing value represents a retained no-response
trial rather than data loss. A model must still advertise and implement the observation
semantics it can fit: declaring omissions does not make a binary choice model omission-aware.

## Rewards and response times

```python
from unspool import ResponseTimeSpec, RewardSpec, TaskSpec

task = TaskSpec(
    choice=choice,
    predictors=("contrast", "previous_outcome"),
    reward=RewardSpec(
        column="reward",
        minimum=-1.0,
        maximum=1.0,
        allow_missing=False,
    ),
    response_time=ResponseTimeSpec(
        column="response_time_ms",
        unit="milliseconds",
    ),
    block_column="block",
    episode_column="episode",
)
validation = task.validate(study)
```

Rewards have explicit numerical support and missingness. Response times reuse Unspool's
physical-unit contract and are converted to canonical seconds without changing the source
column. Block and episode identifiers remain ordinary source values: the task contract
declares their role but does not infer resets or reorder trials.

## Current boundary

The task layer introduced in 0.21 validates binary and multi-alternative observations, but
the current first-party choice models remain binary. Fixed history, kernel, design, and
training-only standardization components are available; migrating the model catalogue onto
them, richer learned transforms, multinomial likelihoods, and omission-aware mixtures
remain 0.22 package work. Until then, task validation failing successfully is preferable
to coercing a richer experiment through a binary estimand.
