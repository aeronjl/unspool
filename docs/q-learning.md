# Reinforcement-learning agents

`BinaryQLearning` is Behavio's first reinforcement-learning reference agent. It is a compact
two-action Q-learning model intended to compete with history, smooth-drift, and latent-state
accounts—not a claim that all behavioural learning is model-free value learning.

`BinaryRLAgent` is the composable 0.22 successor. The original model remains available
unchanged for numerical parity with published Behavio analyses; new work can assemble a
learning rule, optional forgetting, optional choice kernel, policy, and reset boundary as
separate immutable components.

## Compose an agent

```python
from behavio import BinaryRLAgent
from behavio.models import (
    AsymmetricLearning,
    ChoiceKernel,
    ResetRule,
    SoftmaxPolicy,
    UnchosenForgetting,
)

model = BinaryRLAgent(
    learning=AsymmetricLearning(),
    forgetting=UnchosenForgetting(),
    choice_kernel=ChoiceKernel(),
    policy=SoftmaxPolicy(maximum_lapse=0.15),
    reset=ResetRule(("subject", "session")),
    n_restarts=5,
)
parameters = model.parameters_from_components(
    positive_learning_rate=0.35,
    negative_learning_rate=0.12,
    forgetting_rate=0.08,
    choice_kernel_rate=0.30,
    choice_kernel_weight=0.70,
    inverse_temperature=4.0,
    choice_bias=0.10,
    lapse_rate=0.03,
)
```

Every assembly has a configuration-specific signature and exact optimizer and natural
parameter coordinates. Components cannot silently introduce colliding parameter names.
`SymmetricLearning` supplies one delta-rule rate; `AsymmetricLearning` selects separate
rates for non-negative and negative reward-prediction errors.

`UnchosenForgetting` moves the unchosen action value toward the declared `initial_value`
after each outcome. `ChoiceKernel` updates a two-action trace toward the latest choice and
adds its weighted difference to the policy. These are distinct mechanisms: forgetting
changes expected reward, whereas a choice kernel changes the policy without changing
reward value.

`SoftmaxPolicy` always estimates inverse temperature, can omit the fixed action bias, and
can add a bounded random-response lapse mixture. `maximum_lapse` is fixed before fitting;
the natural `lapse_rate` must lie below it.

## Reset semantics are data semantics

`ResetRule(("subject", "session"))` is the default and creates a fresh value and kernel
state at each session. Other reset columns can represent explicit episodes or blocks.
`ResetRule(("subject",))` carries state across sessions within the supplied study, but it
must not be used with a validation split whose prediction context omits earlier sessions.
In that case the model would restart at the beginning of the test table even though its
scientific specification says to carry state. Session-reset agents remain the safe default
for ordinary future-session folds until evaluation exposes fitted terminal states as an
explicit prediction-context object.

## Generative model

Before trial $t$, the agent has action values $Q_t(0)$ and $Q_t(1)$. Choice follows:

$$
P(a_t=1) = \operatorname{logit}^{-1}\left[
\beta\{Q_t(1)-Q_t(0)\} + b + \rho h_{t-1}
\right],
$$

where $\beta>0$ is inverse temperature, $b$ is a fixed action bias, and $h_{t-1}$ is the
previous choice effect-coded as $-1$ or $+1$ (zero at session start). After the current
choice is scored and reward $r_t$ is observed, only the chosen value changes:

$$
Q_{t+1}(a_t) = Q_t(a_t) + \alpha\{r_t-Q_t(a_t)\},
$$

with $0<\alpha<1$. The unchosen value is unchanged. Rewards may be binary or continuous in
$[0,1]$ during fitting.

Values initialize to the fixed `initial_value` at every subject/session boundary, and
perseveration history also resets. This is a deliberate first-model restriction. Behavio's
current fold contract does not yet carry an inferred terminal value state across a session
boundary; allowing cross-session persistence without extending that contract would make
future-session evaluation incorrect.

## An explicit action-contingent environment

Simulation requires one reward-probability column per action. Defaults are
`reward_probability_0` and `reward_probability_1`:

```python
from behavio import BinaryQLearning

model = BinaryQLearning(n_restarts=5, random_seed=4)
parameters = model.parameters_from_components(
    learning_rate=0.25,
    inverse_temperature=5.0,
    choice_bias=0.1,
    perseveration=0.3,
)
study = model.simulate(design, parameters, seed=12)
```

On each simulated trial, the agent chooses first; reward is then sampled from the column
for that generated action. This prevents an incoherent simulation in which rewards were
fixed before the action they supposedly depend on. The generated `choice` and `reward`
columns overwrite any columns of those names in the design.

Fitting and prediction require only observed choices and rewards. Environment probability
columns are a generative-design contract, not privileged information supplied to the
fitted agent.

## Filtered prediction and value trajectories

Every reported choice probability uses earlier choices and rewards only. The current reward
updates values after the current choice has been predicted and scored. Smoothed prediction
is rejected rather than silently substituted.

```python
fit = model.fit(study)
prediction = model.predict(study, fit)
trajectory = model.value_trajectory(study, fit)

trajectory.pre_choice  # Q_t(0), Q_t(1)
trajectory.prediction_error  # r_t - Q_t(a_t)
trajectory.post_update  # values after observing r_t
```

The composable agent exposes the richer parallel record:

```python
fit = model.fit(study)
trajectory = model.trajectory(study, fit)

trajectory.pre_choice_values
trajectory.post_update_values
trajectory.pre_choice_kernel
trajectory.post_update_kernel
trajectory.prediction_error
trajectory.probability
```

Arrays retain the Study's source row order while recursion follows explicit within-subject
session chronology. Within-session rolling-origin evaluation replays the observed prefix,
so both values and perseveration reach the held-out block without scoring context trials.

## Fitting and diagnostics

The choice likelihood and its gradient are computed through the recursive value updates.
Learning rate is optimized on a logit scale and inverse temperature on a log scale. Each
fit runs deterministic L-BFGS-B restarts and retains every objective, convergence flag, and
optimizer message in `QLearningFitResult`.

The selected fit also exposes natural-scale learning rate and inverse temperature, a local
numerical-Hessian covariance, gradient and condition diagnostics, and a boundary flag. The
flag covers learning rates near zero or one, inverse temperatures near zero or above the
configured warning scale, and extreme bias or perseveration estimates.

`fit.audit()` normalizes those numerical flags and the retained restart outcomes into the
same status, issue-code, and `RestartAudit` contract used by the other reference models.
Raw objectives, convergence flags, and optimizer messages remain on `QLearningFitResult`.

`BinaryQLearning.parameter_space` now makes its natural and optimizer coordinates, bounds,
and transforms machine-readable through the shared
[parameter-space contract](parameter-spaces.md). Existing encoded names remain valid.
Portable fit artifacts record both coordinate identities and estimates, while covariance
and standard errors remain explicitly on the optimizer scale.

Fitting now runs through the common [deterministic inference backend](inference-backends.md).
`QLearningFitResult.optimization_run` retains every start, attempted estimate, objective,
status, message, iteration/evaluation count, and gradient norm. The established restart
arrays remain as checked compatibility views for diagnostics and existing analyses.

`BinaryRLAgent` uses the same deterministic multistart and fit-audit requirements, with a
numerical gradient because the recursion varies by component assembly. `RLFitResult`
retains every restart plus a labelled natural-parameter view. A single parameter-space
encoding for arbitrary component assemblies remains an explicit extension rather than a
hidden requirement of the stable fitting contract.

## Recovery and competing explanations

The model satisfies Behavio's generic parameter- and model-recovery contracts. Its tests
use an explicitly volatile two-armed environment, check the recursive analytic gradient
against finite differences, recover the generating agent, and require Q-learning to beat
both a static history GLM and a smooth session-time GLM on a future session.

Generic parameter-recovery tables operate on the stable optimizer coordinates listed in
`parameter_names`. Use `parameter_components()` to report learning rate and inverse
temperature on their natural scales.

This is still conditional evidence. Learning rate and temperature can trade off under weak
or stationary reward schedules; choice bias and perseveration can also absorb structure
that belongs to task covariates or latent strategies. Recovery must be rerun for the actual
reward schedule, session length, missingness, and candidate set.

## Drifting and pooled agents

Both agents compose. `smooth(agent, over="session_order", knots=..., parameters=...)` lets a
named parameter follow a path across training, and
`hierarchical(agent, over="subject", parameters=...)` gives each animal a shrunken deviation
on the *transformed* coordinate, so a per-animal learning rate stays inside \((0,1)\) by
construction:

```python
from behavio.compose import hierarchical, smooth

sharpening = smooth(
    agent, over="session_order", knots=(0.0, 5.0), parameters=("inverse_temperature_log",)
)
per_animal = hierarchical(agent, over="subject", parameters=("choice_bias",), scale=0.5)
```

Which parameters drift is a scientific choice and `parameters=` is how it is made: a
sharpening policy and a changing learning rate are different models. The clock is part of
that choice too, and it is restricted -- it must be constant within each reset block,
because a value trace written by a learning rate that changed mid-session cannot say which
of its values produced which part of the trace. Smoothing over a within-session counter
raises rather than averaging. See
[composing models](composing-models.md#models-whose-coordinate-is-bounded-not-linear).

`mix()` still refuses both agents, and that is a statement about the model: a lapse belongs
on the policy, inside the recursion, where it mixes the emitted action while leaving the
value update to see the action that was taken.

## Current boundary

The composable layer remains deliberately binary. It now covers symmetric or asymmetric
chosen-value learning, unchosen forgetting, an exponential choice kernel, bounded lapse,
explicit reset columns, session-level drift and partial pooling by any study column. It does
not yet cover counterfactual updates, Kalman or Bayesian learners, model-based planning,
stimulus-conditioned policies, within-session parameter drift, or multinomial action sets.
Component-rich fits can be weakly identified even when the
optimizer converges; exact-design parameter and model recovery remain mandatory evidence,
not a property conferred by composition.
