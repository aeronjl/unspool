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
    predictors=("stimulus",),
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
    predictors=("stimulus",),
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

## Partial pooling across subjects

`hierarchical()` works on this model, on the **emission coefficients** and on nothing else:

```python
from behavio import BernoulliGLMHMM
from behavio.compose import hierarchical

model = BernoulliGLMHMM(predictors=("stimulus",), n_states=2, l2=0.01)
pooled = hierarchical(model, over="subject", parameters=("intercept",), scale=0.5)

fit = pooled.fit(study)
fit.estimates  # the population coordinate, in canonical label order
fit.group_deviations  # (subjects, 2) deviations on state[0].intercept and state[1].intercept
fit.parameters_for("mouse-a")
```

Naming `"intercept"` varies `state[0].intercept` and `state[1].intercept` together, under one
prior scale. That is not a shorthand: it is the model. A GLM-HMM's coordinate is
*permutation-equivariant* — relabelling the states permutes the emission rows — so a
coefficient varies by group for every state at once or the fit stops being invariant under
relabelling, which is the invariance the canonical ordering canonicalises. Naming
`state[0].intercept` alone is refused.

### Why a subject-level hierarchy is expressible at all

A GLM-HMM's likelihood is a forward recursion, so there is no per-row score to profile a
group deviation against. There is a per-**session** score, and that is enough. The
bounded-coordinate contract asks for a negative log likelihood in one coordinate vector per
row *plus* [`row_blocks`](composing-models.md#models-whose-coordinate-is-bounded-not-linear),
the blocks the coordinate must be constant within; a GLM-HMM answers with its sessions, the
same answer a Q-learning agent gives. `over="subject"` is admissible because a session lies
inside a subject. A grouping column that cuts a session is refused rather than averaged:

```python
hierarchical(model, over="trial", parameters=("intercept",)).fit(study)
# ValueError: grouping by 'trial' splits a block this model's likelihood recurses over ...
```

### Why transitions stay pooled

A row of the transition matrix lives on a simplex. This model charts it with
reference-category logits, \(t_{rj} = \log A_{rj} - \log A_{r,K-1}\), and an isotropic
Gaussian on those coordinates is **not** a prior on the transition matrix — it is a prior on
the chart. For \(K = 2\) the two possible charts differ by a sign and a Gaussian survives it.
For \(K \ge 3\) they do not: independent normals on \((\log A_{r0}/A_{r2}, \log
A_{r1}/A_{r2})\) is a different distribution on the simplex from the one you get by making
state 0 the reference, so two users who ordered their states differently would be fitting
different models while reading the same declaration. And the reference state here is not
chosen by the user at all — it is whichever state canonicalisation puts last, which is a
function of the data.

So "this animal is stickier" is not a deviation this coordinate can carry honestly. It is a
statement about a contrast, it has no single named parameter for \(K > 2\), and the reference
state has no self-transition coordinate of its own. A per-animal transition model needs a
symmetric parameterisation — a centred or isometric log-ratio coordinate with a relabelling-
invariant penalty — and that is a different model rather than a wider declaration. Until it
exists:

- `stickiness=` is the declared, chart-free way to say that states persist, at population
  level;
- per-animal dynamics are answered by fitting subjects separately and comparing, not by a
  deviation the package cannot define.

`hierarchical(model, over="subject")` with the default `parameters=None` is therefore an
error, not a fit, and it names the alternative.

Composition also declines while `stickiness > 0`, because \(-\kappa \sum_k \log A_{kk}\) is
neither a per-row score nor a quadratic penalty and a combinator would have to apply it once
per session block instead of once per model.

### Labels under a joint fit

This is the part worth reading slowly, because a hierarchical latent-state model is exactly
where a fit can be well typed and meaningless.

**The likelihood cannot identify a subject's labels.** Relabelling one subject's states is
an exact symmetry of a GLM-HMM's likelihood whenever its dynamics are symmetric — the forward
recursion cannot tell the two apart at any sample size. Behavio's test suite asserts that
equality rather than assuming it.

**The group prior can, and is the only thing that does.** Subject \(g\)'s emissions are the
population's plus a deviation \(b_g \sim \mathcal N(0, \sigma^2)\). Relabelling subject \(g\)
alone leaves the likelihood where it was but replaces \(b_g\) with
\(\Pi(\beta + b_g) - \beta\), which for well-separated population states is a much larger
vector and pays a much larger price. Per-subject relabelling is therefore **not** a symmetry
of the joint objective, and the label-consistent solution is its global optimum. The one
symmetry that survives is the *global* one — relabelling the population and every deviation
together — which is the same symmetry the pooled fit has, and `fit_rows` resolves it the same
way, by sorting states along `label_by`. It can do that to the whole joint vector at once
because relabelling is a **linear** map on this coordinate (emissions permute; the two
simplexes permute and re-reference), so the covariance is carried through exactly rather than
recomputed. `model.relabelling_map(permutation)` is that matrix.

**"Global optimum" is not "the optimizer found it".** So the claim is checked rather than
asserted:

```python
agreement = model.group_label_agreement(fit)

agreement.aligned  # per group: is its closest match to the population the identity?
agreement.relabelled_groups  # the groups whose deviation is a relabelling, not a difference
agreement.margins  # how much worse the best *other* matching is
agreement.all_aligned
```

Each group's emission rows are Hungarian-matched to the population's on Euclidean distance. A
group that comes back permuted has a deviation that must not be read as "this animal is more
biased in state 1", because its state 1 is not the population's state 1. A group whose margin
is near zero has states too poorly separated for the anchor to have bitten, and its deviation
should be read with the same suspicion as a fit with a small `label_order_gap`.

This is a different question from `state_recovery()`. That aligns inferred state *posteriors*
against known simulated truth and so cannot be run on data; `group_label_agreement()` compares
two fitted parameter vectors and runs on anything.

The recovery test that opened this cell simulates three subjects of which one has its
intercepts reversed relative to the population — far enough that fitting that subject **alone**
puts its stimulus-sensitive state at index 0 while every other subject's sits at index 1. One
joint fit puts all three on the population's labelling, reports the reversal as a large
deviation rather than as a swap, recovers the population, and reports a visibly smaller
alignment margin for the subject whose labels were at risk.

### No path in clock time

`smooth()` is refused for this family, at any parameter selection:

```python
smooth(model, over="session_order", knots=(0.0, 4.0), parameters=("intercept",))
# TypeError: ... a GLM-HMM's latent labels are an ordering of one emission coefficient, and
# an ordering of coefficient *paths* is only a permutation where the paths do not cross ...
```

The ordering that names these states is an ordering of numbers. When `label_by` becomes a
path, states can be ordered one way early in training and the other way late, and no single
permutation canonicalises the fit: "state 0" would name different behaviour at the two ends of
the clock. The fit would converge and report knots, which is precisely the hazard — a drifting
GLM-HMM is a real and useful model, and it needs a labelling rule defined on paths, plus a
report of where paths cross, before anything it estimates can be read. Compare this model
against a [smooth GLM](composing-models.md) meanwhile; that is the competitor the comparison
was always for.

## Current boundary

This implementation resets state at each session and pools the transition matrix and the
initial distribution across subjects. Emission coefficients partially pool by any grouping
column that respects session boundaries. It does not provide per-group transition dynamics,
smooth (path-valued) parameters, state carry-over across sessions, covariate-dependent
transitions, missing-outcome inference, semi-Markov dwell times, or smoothed state reports.
Those should be added only with targeted recovery and competing-explanation tests, and two of
them need a labelling rule first.
