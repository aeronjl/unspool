# Fixed-transition Bernoulli GLM-HMM

`BernoulliGLMHMM` is Behavio's first discrete latent-state competitor. “Fixed transition”
means that one learned transition matrix is stationary across observed trials and sessions;
it does not mean that transition probabilities are supplied as known constants. An initial
state distribution is applied anew at every subject/session boundary.

The model is intentionally compact. Its purpose is to make state-switching explanations
executable, recoverable, and falsifiable alongside static and smoothly changing models—not
to assign cognitive meanings to latent labels.

## Generative model

For state $z_t \in \{1, \ldots, K\}$ and binary choice $y_t$:

$$
z_1 \sim \operatorname{Categorical}(\pi), \qquad
z_t \mid z_{t-1} \sim \operatorname{Categorical}(A_{z_{t-1}}),
$$

$$
y_t \mid z_t, x_t \sim
\operatorname{Bernoulli}\left(\operatorname{logit}^{-1}(x_t^\top \beta_{z_t})\right).
$$

Each state has its own intercept, task-covariate coefficients, and optional choice-history
coefficients. Choice history is bounded by sessions and updated recursively during
simulation. A transition occurs between consecutive *observed* trials. Gaps in trial
identifiers do not cause Behavio to invent unobserved transitions.

This is the standard input-driven GLM-HMM convention used in behavioural work: observed
task inputs enter the state-specific GLM emissions. They do not alter the transition
matrix. Covariate-dependent transitions are a different model and are not implied by the
word “input-driven” here.

```python
from behavio import BernoulliGLMHMM

model = BernoulliGLMHMM(
    covariates=("stimulus",),
    choice_lags=1,
    n_states=2,
    n_restarts=5,
    random_seed=7,
    l2=0.01,
)

parameters = model.parameters_from_components(
    initial_probabilities=[0.5, 0.5],
    transition_matrix=[[0.95, 0.05], [0.05, 0.95]],
    emissions={
        "intercept": [-1.0, 1.0],
        "stimulus": [0.4, 1.6],
        "choice_lag_1": [0.7, 0.1],
    },
)

study = model.simulate(design, parameters, seed=11)
fit = model.fit(study)
components = model.parameter_components(fit)
```

Natural probabilities are encoded internally as reference-category logits. The helper
validates strict positivity and row sums, canonicalizes labels, and returns the exact flat
mapping needed by simulation and the generic recovery API.

## Optional sticky transition prior

```python
sticky = BernoulliGLMHMM(
    covariates=("stimulus",),
    n_states=3,
    stickiness=2.0,
)
```

Positive `stickiness` performs MAP fitting under a sticky Dirichlet transition prior. It
adds the declared value as a pseudo-count to every self-transition and leaves off-diagonal
pseudo-counts at their flat baseline. Equivalently, the optimized objective adds

\[
-\kappa\sum_k \log A_{kk}.
\]

This regularizes state persistence; it does not fix dwell times, change the simulator, or
make the transition matrix trial-varying. `stickiness=0` is the original maximum-
likelihood model. The value is part of the model signature and must be selected inside
training data, preferably under nested prospective validation rather than by inspecting a
full-study state plot.

## Fitting and numerical diagnostics

The marginal likelihood is evaluated in log space with a forward recursion. Its analytic
gradient uses forward-backward state and transition expectations, and each fit runs
`n_restarts` deterministic L-BFGS-B optimizations. The converged restart with the best
objective is selected; if none converge, the best failed attempt remains visible rather
than being silently discarded.

`GLMHMMFitResult` retains:

- every restart objective, convergence flag, and optimizer message;
- the selected restart and raw-to-canonical state permutation;
- smoothed training-state occupancy used only as a fit diagnostic;
- minimum distance between state emission vectors;
- the ordering gap for the declared label coefficient;
- explicit low-occupancy and label-ambiguity flags; and
- the common numerical, boundary, gradient, and Hessian diagnostics.

`fit.audit()` exposes these through the shared `FitAudit`: restart disagreement,
nonconvergence, low occupancy, and label ambiguity receive the same stable issue codes used
in reports for other model families. The raw arrays and model-specific scalar diagnostics
remain on `GLMHMMFitResult`.

Standard errors use a local numerical Hessian of the penalized marginal likelihood. They
are local approximations around one mode, not a resolution of multimodality or label
uncertainty. With nonzero `l2`, they are also penalized-likelihood approximations.

## Labels are coordinates, not interpretations

State labels are non-identifiable under permutation. Behavio orders fitted states by
increasing values of `label_by` (the intercept by default), using the remaining emission
coefficients only as deterministic tie-breakers. The same ordering is applied when packing
generative components, which makes parameter recovery compare like with like.

This convention does not make “state 0” a disengaged, exploratory, or biased state. A small
`label_order_gap`, low emission separation, low occupancy, or materially different fits
across restarts should block such interpretations. Any cognitive claim still has to survive
observable-strategy, history, reinforcement-learning, static, and smooth-drift competitors.

Canonical ordering is not evidence that fitted labels match simulated labels. For
truth-aware recovery, `state_recovery()` ignores their names and finds the permutation that
maximizes posterior mass on the known simulated states, balanced equally across states:

```python
simulation = model.simulate_with_states(design, parameters, seed=11)
fit = model.fit(simulation.study)
recovery = model.state_recovery(simulation, fit)

recovery.reference_to_inferred
recovery.decoded_accuracy
recovery.posterior_accuracy
recovery.score_gap
recovery.ambiguous
```

The result retains the row-normalized soft confusion matrix, winning and runner-up
assignment scores, aligned probabilities and labels, reference-state counts, and both hard
and posterior accuracy. A missing reference state or a winning margin no larger than
`ambiguity_tolerance` is explicitly ambiguous. The default tolerance is `0.05` balanced
posterior-accuracy units; it is a declared recovery criterion, not a universal constant.

Alignment consumes latent simulation truth and therefore cannot be run as an ordinary fit
diagnostic on empirical data. It never changes model parameters or predictions. The
fit-internal `label_ambiguous` flag asks whether the chosen ordering coordinate is stable;
the recovery-level `ambiguous` flag asks whether fitted states can be uniquely matched to
known generative states. Either can fail without the other. Filtering excludes later
outcomes from each state update, but whether fitted parameters are prospective depends on
how the supplied `fit` was trained.

## Filtered prediction and state probabilities

`predict()` and `pointwise_log_prob()` are strictly one-step-ahead and filtered. Before
trial $t$, the predictive state distribution uses outcomes only through $t-1$. After
observing $y_t$, that distribution is updated and propagated through the transition matrix
for the next trial.

```python
state = model.state_probabilities(study, fit)
state.predictive  # p(z_t | y_1, ..., y_{t-1})
state.filtered  # p(z_t | y_1, ..., y_t)
```

Behavio does not silently substitute a smoothed state decoding. `PredictionMode.SMOOTHED`
is rejected. When a within-session rolling split is evaluated, its pre-origin prefix is
replayed as prediction context, so the latent filter and choice history both reach the
holdout boundary without scoring context trials.

## Simulation and recovery

Ordinary `simulate()` returns only the observed study. `simulate_with_states()` returns a
`GLMHMMSimulation` whose latent truth is stored separately, preventing a state column from
accidentally entering downstream model features. The model satisfies Behavio's generic
parameter- and model-recovery contracts.

Recovery remains design-specific. Transition probabilities near zero or one, rarely
occupied states, weakly separated emissions, short sessions, or insufficient state changes
can all make a nominally fitted model unrecoverable. The included tests use a deliberately
clear switching regime and also require it to outperform static and smooth session-time
GLMs on a future session. The repeated
[state-alignment benchmark](https://github.com/aeronjl/behavio/tree/main/benchmarks/state_alignment) additionally contrasts
clear and overlapping emissions and verifies exact invariance to inferred-label reversal.

## Current boundary

This implementation pools one parameter set across supplied subjects and resets state at
each session. It does not yet provide hierarchical partial pooling, state carry-over across
sessions, time-varying or covariate-dependent transitions, missing-outcome inference,
semi-Markov dwell times, or smoothed state reports. Those should be added only with targeted
recovery and competing-explanation tests.
