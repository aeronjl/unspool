# Canonical binary-choice baselines

Specialized longitudinal models should compete against simple explanations with names,
assumptions, and complete generative contracts. Behavio therefore exposes the common
binary-choice baselines directly instead of requiring every lab to rediscover constructor
settings for a generic GLM.

| Model | Linear or mixture structure | Reset boundary |
| --- | --- | --- |
| `BiasOnly` | one stationary choice intercept | none |
| `Psychometric` | intercept plus one stimulus slope | none |
| `Perseveration` | intercept plus previous effect-coded choice | subject/session |
| `LapsePsychometric` | psychometric curve plus random-response mixture | none |
| `WinStayLoseShift` | previous choice after reward versus non-reward | subject/session |

All five models simulate, fit, predict, score pointwise, emit the common numerical audit,
and enter parameter or model recovery through the same public contract.

## Fit the nested baselines first

```python
from behavio import BiasOnly, Perseveration, Psychometric, compare_models

models = {
    "bias": BiasOnly(),
    "stimulus": Psychometric(stimulus="signed_contrast"),
    "choice_history": Perseveration(),
}
report = compare_models(models, study, splits)
```

`BiasOnly` asks whether a stationary response preference is sufficient. `Psychometric`
adds one declared stimulus column. `Perseveration` instead asks whether the immediately
previous observed choice predicts the next response, resetting at every subject/session
boundary. It is a behavioural dependency, not evidence for a particular learning
mechanism.

These are named views over the tested Bernoulli-GLM engine, but their identities remain
distinct in fits, audits, comparisons, recovery reports, and exported fit artifacts.

## Lapse mixtures

```python
from behavio import LapsePsychometric

model = LapsePsychometric(
    stimulus="signed_contrast",
    maximum_lapse=0.2,
    n_restarts=5,
)
truth = model.parameters_from_components(
    intercept=0.0,
    slope=1.5,
    lapse_rate=0.05,
)
```

The response probability is

\[
p(y=1\mid x)=\frac{\lambda}{2} + (1-\lambda)\,
\operatorname{logit}^{-1}(\beta_0 + \beta_1 x).
\]

`maximum_lapse` fixes the upper support before fitting. The optimizer estimates an
unconstrained `lapse_logit`; `parameter_components()` reports the natural lapse rate.
Every deterministic restart and the selected optimum remain on
`LapsePsychometricFitResult`, so a lapse parameter cannot silently absorb poor local
optimization.

A lapse mixture is still only one account of asymptotic errors. It should be compared with
stimulus nonlinearities, history, contaminants, state mixtures, and task-specific motor or
omission processes when those alternatives are scientifically plausible.

## Win-stay/lose-shift is an outcome-conditioned baseline

`WinStayLoseShift` uses two signed history features. After a rewarded trial, positive
`win_stay` favours repeating the previous choice. After a non-rewarded trial, positive
`lose_shift` favours the opposite choice. Both features are zero at every session start.
The observed reward is predictive context; the model's pointwise likelihood scores choice
only.

Simulation requires two explicitly named reward-probability columns, one for each action,
and generates both choice and reward recursively. This makes exact-design recovery
possible without implying that win-stay/lose-shift is a value-learning mechanism.

## Current boundary

These named baselines retain binary zero/one choices because their likelihoods state that
assumption. The separate [multinomial reference likelihood](multinomial.md) supports richer
action sets and retained omissions; it does not retroactively change the estimand of these
binary baselines. Standard input-driven or sticky GLM-HMM variants remain the next
catalogue layer.
