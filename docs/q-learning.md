# Session-reset binary Q-learning

`BinaryQLearning` is Unspool's first reinforcement-learning reference agent. It is a compact
two-action Q-learning model intended to compete with history, smooth-drift, and latent-state
accounts—not a claim that all behavioural learning is model-free value learning.

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
perseveration history also resets. This is a deliberate first-model restriction. Unspool's
current fold contract does not yet carry an inferred terminal value state across a session
boundary; allowing cross-session persistence without extending that contract would make
future-session evaluation incorrect.

## An explicit action-contingent environment

Simulation requires one reward-probability column per action. Defaults are
`reward_probability_0` and `reward_probability_1`:

```python
from unspool import BinaryQLearning

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

## Recovery and competing explanations

The model satisfies Unspool's generic parameter- and model-recovery contracts. Its tests
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

## Current boundary

This first agent has two actions, a single symmetric learning rate, fixed parameters, no
forgetting or counterfactual updates, and mandatory session resets. It does not yet provide
partial pooling, across-session value carry-over, asymmetric positive/negative learning,
model-based planning, or stimulus-conditioned policies. Those are extensions to earn with
design-specific recovery rather than options to accumulate speculatively.
