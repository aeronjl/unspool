# The behavioural task contract

`Study` says which trial occurred when. `TaskSpec` says what was observed on that trial.
Keeping these contracts separate lets one longitudinal table support several scientifically
different models without hiding task semantics inside model-specific preprocessing.

## A first fit

```python
from behavio import BernoulliHistoryGLM, ChoiceSpec, Study, TaskSpec, fit_model

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
    predictors=("stimulus",),
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
Every model declares `required_task_columns` for predictive context that it consumes but
does not score—for example, reward in a win-stay/lose-shift model. The declaration is part
of the `BehaviourEstimator` contract rather than optional, and a model that reads nothing
but the column it scores declares `()`. Those columns must have an explicit predictor,
reward, response-time, block, episode, or availability role.
Prospective scientific claims should still use the fold-aware comparison or protocol
runner, which refits the model independently inside every training fold. Reusable numeric,
categorical, interaction, and history terms are described in the
[fixed design-matrix contract](design-matrices.md).

Use `export_fit(fitted, study)` when the interactive result needs a deterministic,
non-executable record with task, data, version, parameter, and diagnostic provenance. See
[fit artifacts and extension registries](fit-artifacts.md).

## Choices are a coordinate, not an incidental encoding

```python
from behavio import ChoiceSpec

choice = ChoiceSpec(
    column="action",
    options=("left", "right", "wait"),
    omission_values=("no_response",),
    missing_is_omission=False,
    available_options_column="available_actions",
)
encoded = choice.read(study)
offered = choice.availability(design_study)
```

`options` fixes one stable categorical coordinate. `encoded.codes` contains the declared
option position and uses `-1` only for explicit omissions. `encoded.available` is a
trial-by-option mask. An observed action that was unavailable on that trial is rejected.
The source choice column is never silently rewritten.

`availability()` validates and returns the same trial-by-option mask without requiring a
choice column. Generative models use it on a design study before outcomes exist.

Missing values are not omissions by default. Setting `missing_is_omission=True` is an
explicit scientific assertion that the missing value represents a retained no-response
trial rather than data loss. A model must still advertise and implement the observation
semantics it can fit: declaring omissions does not make a binary choice model omission-aware.
`MultinomialLogit` consumes this coordinate directly and can retain omissions as a modeled
category; see [multinomial and omission-aware choice](multinomial.md).

## Rewards and response times

```python
from behavio import ResponseTimeSpec, RewardSpec, TaskSpec

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

Rewards have explicit numerical support and missingness. Response times reuse Behavio's
physical-unit contract and are converted to canonical seconds without changing the source
column. Block and episode identifiers remain ordinary source values: the task contract
declares their role but does not infer resets or reorder trials.

## Current boundary

The task layer validates binary and multi-alternative observations, and the first
multinomial likelihood now uses that coordinate without re-encoding source labels. Binary
baselines and binary RL agents intentionally retain their narrower observation contract.
Choice-history simulation for a multinomial policy, richer learned transforms, and
task-specific motor or censoring models remain further catalogue work. Task validation
failing successfully is preferable to coercing a richer experiment through a different
estimand.
