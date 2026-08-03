# Bernoulli GLM-HMM: stationary, covariate-dependent, and session-dynamic

`BernoulliGLMHMM` is Behavio's discrete latent-state choice competitor. With no transition
design, one learned transition matrix is stationary across observed trials and sessions;
it is estimated, not supplied as a known constant. With `transition_predictors=` or
`transition_design=`, the model is a non-homogeneous HMM: transition probabilities vary by
trial through a declared multinomial-logit regression. An initial state distribution is
applied anew at every subject/session boundary in both cases.

The model is intentionally compact. Its purpose is to make state-switching explanations
executable, recoverable, and falsifiable alongside static and smoothly changing models—not
to assign cognitive meanings to latent labels.

`SessionDynamicBernoulliGLMHMM` is a separate, single-subject research model. It gives
emission coefficients a stochastic session path and gives each session its own transition
matrix. That model is described below; it is not what `transition_predictors=` constructs.

## Generative model

For state $z_t \in \{1, \ldots, K\}$ and binary choice $y_t$:

$$
z_1 \sim \operatorname{Categorical}(\pi), \qquad
z_t \mid z_{t-1}, u_t \sim \operatorname{Categorical}(A_{z_{t-1},t}),
$$

$$
y_t \mid z_t, x_t \sim
\operatorname{Bernoulli}\left(\operatorname{logit}^{-1}(x_t^\top \beta_{z_t})\right).
$$

Each state has its own intercept, task-covariate coefficients, and optional choice-history
coefficients. Choice history is bounded by sessions and updated recursively during
simulation. A transition occurs between consecutive *observed* trials. Gaps in trial
identifiers do not cause Behavio to invent unobserved transitions.

The stationary default is the standard input-driven GLM-HMM convention used in behavioural
work: observed task inputs enter the state-specific GLM emissions but do not alter the
transition matrix. A declared transition design widens that model in the conventional
non-homogeneous-HMM way:

$$
\log A_{ij,t} = \log A_{ij,0} + u_t^\top \gamma_{ij}
 - \operatorname{logsumexp}_{m}
   \left(\log A_{im,0} + u_t^\top \gamma_{im}\right),
\qquad \sum_j \gamma_{ij}=0.
$$

Trial `t`'s transition row governs $P(z_t\mid z_{t-1})$. The first row of each session is
retained for source-row alignment but ignored by the likelihood because the chain resets to
$\pi$. Learned scaling or landmarks in a transition design must be fitted on training rows
before the model is constructed, just like learned emission preprocessing.

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

The stationary model encodes natural probabilities internally as reference-category logits
for backward compatibility. The helper validates strict positivity and row sums,
canonicalizes labels, and returns the exact flat mapping needed by simulation and recovery.

## Covariate-dependent transitions

```python
dynamic = BernoulliGLMHMM(
    predictors=("stimulus",),
    transition_predictors=("session_order",),
    choice_lags=1,
    n_states=2,
    transition_l2=0.5,
)

parameters = dynamic.parameters_from_components(
    initial_probabilities=[0.5, 0.5],
    transition_matrix=[[0.95, 0.05], [0.05, 0.95]],  # u_t = 0 baseline
    emissions={
        "intercept": [-1.0, 1.0],
        "stimulus": [0.4, 1.6],
        "choice_lag_1": [0.7, 0.1],
    },
    transition_coefficients={
        # One centred destination-logit effect matrix: rows sum to zero.
        "session_order": [[-0.15, 0.15], [0.10, -0.10]],
    },
)

study = dynamic.simulate(design, parameters, seed=11)
fit = dynamic.fit(study)
matrices = dynamic.transition_probabilities(study, fit)  # (trials, states, states)
```

`transition_design=` accepts the same fixed `DesignSpec` components as an emission design,
but must use `intercept=False`: `transition_matrix` is already the intercept at a zero-valued
design row. The model reports centred natural-scale destination effects through
`parameter_components()` and stores the optimized coordinate as Helmert-basis isometric
log-ratios (ILR). All ILR bases describe the same simplex geometry; a state permutation is
an orthogonal rotation, so Euclidean ridge and Gaussian group penalties do not depend on an
arbitrary reference state. `transition_l2` regularizes covariate effects, not the baseline
transition matrix.

This capability matches the standard covariate-dependent transition construction in
non-homogeneous HMMs and the fixed/random-effect transition coverage in
[hmmTMB](https://doi.org/10.18637/jss.v114.i05). It is not the session-random-walk model from
the recent [dynamic GLM-HMM learning study](https://pmc.ncbi.nlm.nih.gov/articles/PMC11623682/):
that model evolves emission parameters under a temporal prior and draws session transition
matrices around a global matrix rather than making either a deterministic regression on
observed covariates.

## Session-dynamic parameters

```python
from behavio import SessionDynamicBernoulliGLMHMM

dynamic = SessionDynamicBernoulliGLMHMM(
    predictors=("stimulus",),
    choice_lags=1,
    n_states=2,
    emission_step_scale=0.25,
    transition_concentration=20.0,
)

simulation = dynamic.simulate_with_trajectories(design, parameters, seed=11)
fit = dynamic.fit(simulation.study)

fit.session_emission_coefficients  # (sessions, states, coefficients)
fit.session_transition_matrices  # (sessions, states, states)
fit.label_crossings
recovery = dynamic.trajectory_recovery(simulation, fit)
```

For session $s$, state $k$, and transition source row $i$, the implemented priors are

\[
\beta^{(s)}_k \sim
\mathcal N\!\left(\beta^{(s-1)}_k,\sigma^2 I\right), \qquad
A^{(s)}_i \sim
\operatorname{Dirichlet}\!\left(\alpha \bar A_i + \mathbf 1\right).
\]

This is the distinction in the published model: emission weights have a Gaussian random
walk, but transition matrices are conditionally independent across sessions around the
global matrix $\bar A$. Because of the added-one parameterization, $\bar A$ is the prior
mode rather than its arithmetic mean. `transition_concentration` therefore controls
shrinkage toward that global transition mode; it is not a transition-path smoothness parameter. Both
hyperparameters are part of the model signature and must be selected only within training
data.

Fitting is the reference three-stage MAP-EM sequence: a stationary multistart GLM-HMM; an
emission-dynamic intermediate model with one re-estimated shared transition matrix; then the
fully dynamic model initialized from that partial fit. In both dynamic stages,
forward-backward supplies expected state and transition counts and one joint L-BFGS-B
emission M-step carries every adjacent-session Gaussian term. The full stage uses the exact
Dirichlet pseudo-count transition update around the partial stage's shared matrix $\bar A$.
Both objective histories and both convergence decisions are retained on the fit. The initial
distribution is estimated across session openings rather than fixed uniform. That is an
intentional Behavio convention and a difference from the cited implementation.

`n_states`, `emission_step_scale`, and `transition_concentration` remain candidate
specifications rather than parameters estimated by the EM loop. Supply their Cartesian grid
to `nested_select_model()` with forward-session outer and inner splits: the generic selector
fits every candidate using only the outer training study, and the untouched outer session is
used only after a specification has been chosen. The
[session-dynamic benchmark](../benchmarks/session_dynamic_glm_hmm/README.md) pins this
procedure against stationary, observed-transition, smooth-drift, and Q-learning competitors.
Its compact dynamic regime recovers the latent path but does **not** establish superior
future-session prediction over the stationary GLM-HMM, so retrospective path quality must
not be promoted into a deployment claim. The fit also reports no local covariance: standard
errors and covariance are `NaN`, `uncertainty_policy` is
`"not-estimated"`, and the fit audit keeps that limitation visible.

Labels are canonicalized once for the complete path, using the mean `label_by` coefficient.
They are never independently reordered by session. `label_crossings` reports sign changes
or near contacts in every pairwise label gap between adjacent sessions;
`label_path_ambiguous` is true when a crossing occurs or the smallest path gap is within
`label_tolerance`. `trajectory_recovery()` uses one truth-aware whole-path permutation and
reports state alignment plus emission- and transition-path RMSE.

The model fits one subject at a time. It refuses `hierarchical()` because a joint dynamic
population model needs an explicit distribution over subject paths, not pooled sessions.
It also refuses trial-level transition predictors and `stickiness`; those name different
transition models. For a fitted session it reuses that session's parameters. For a strictly
later unseen session, filtered prediction uses the random walk's conditional mean by
carrying the final emission weights forward and uses $\bar A$, the Dirichlet prior mode, as
the unseen transition matrix. Unseen earlier/interleaved sessions and unseen subjects are refused. This forecast
rule makes prospective scoring executable; it is not a claim that the original paper
validated unseen-session forecasting.

## Cross-subject dynamic hierarchy

`HierarchicalSessionDynamicBernoulliGLMHMM` is the dedicated population extension. It is not
`hierarchical(SessionDynamicBernoulliGLMHMM(...))`: the generic combinator has one fixed-width
coordinate per group, while this model has a population path plus a data-dependent number of
subject-session path points and needs one label permutation across all of them.

```python
from behavio import HierarchicalSessionDynamicBernoulliGLMHMM

population_dynamic = HierarchicalSessionDynamicBernoulliGLMHMM(
    predictors=("stimulus",),
    choice_lags=1,
    n_states=2,
    population_emission_step_scale=0.2,
    subject_emission_scale=0.4,
    emission_step_scale=0.15,
    transition_concentration=20.0,
)

simulation = population_dynamic.simulate_with_trajectories(design, parameters, seed=11)
fit = population_dynamic.fit(simulation.study)

fit.population_emission_coefficients  # (population orders, states, coefficients)
fit.session_emission_coefficients  # (subject-session blocks, states, coefficients)
fit.subject_deviations
recovery = population_dynamic.trajectory_recovery(simulation, fit)
```

For sorted observed population session order $r$, subject $m$, and that subject's observed
session position $s$, the emission hierarchy is

\[
M_r\sim\mathcal N(M_{r-1},\sigma_{\mathrm{pop}}^2I),\qquad
D_{m,0}\sim\mathcal N(0,\tau^2I),\qquad
D_{m,s}\sim\mathcal N(D_{m,s-1},\sigma_{\mathrm{subj}}^2I),
\]

\[
W_{m,s}=M_{r(m,s)}+D_{m,s}.
\]

`population_emission_step_scale` is $\sigma_{\mathrm{pop}}$,
`subject_emission_scale` is $\tau$, and the inherited `emission_step_scale` is
$\sigma_{\mathrm{subj}}$. “Adjacent” means adjacent observed population orders or adjacent
observed sessions for that subject; gaps are not silently converted to elapsed time.

Every subject-session transition row keeps the original direct population distribution,

\[
A^{(m,s)}_i\sim
\operatorname{Dirichlet}(\alpha\bar A_i+\mathbf 1).
\]

There is no persistent subject-specific transition centre and no temporal transition path.
Adding either would be another model. The population class uses the same stationary,
emission-dynamic/static-transition, and fully dynamic MAP-EM stages as the one-subject class.
The joint emission M-step optimizes $M$ and every $W_m$ together. The pooled stationary fit
initializes one shared state coordinate; the mean population path canonicalizes it once; and
population and within-subject crossing evidence remains separate on the fit.

Prediction distinguishes two deployment cases. For a strictly later session of a fitted
subject, the model evaluates the population path at that order and carries the subject's
last deviation forward. For an unseen subject, it uses the zero-deviation population
plug-in. Past or interleaved missing sessions of fitted subjects are refused. Beyond the last
population order, the random walk's conditional mean carries the population path forward.
Every forecast uses $\bar A$, resets to the pooled initial distribution, and remains
filtered.

The Gaussian hierarchy follows the standard random-effects principle used in mixed HMMs,
including [hmmTMB](https://doi.org/10.18637/jss.v114.i05), but this exact population path was
not fitted by the [dynamic GLM-HMM study](https://pmc.ncbi.nlm.nih.gov/articles/PMC11623682/).
It is a Behavio model specified in
[SDR-0066](decisions/0066-fit-a-population-session-dynamic-glm-hmm-with-subject-deviation-paths.md).
The three scales are fixed model specifications. Standard errors and covariance are `NaN`;
label-aware path and hyperparameter uncertainty remains the next capability.

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

`stickiness > 0` is refused with a transition design. A Dirichlet pseudo-count on one
stationary matrix is not a prior on the set of trial-varying matrices generated by a
multinomial regression.

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
observing $y_t$, that distribution is updated and propagated through the stationary matrix
or through the dynamic matrix evaluated from trial $t+1$'s transition covariates.

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

`hierarchical()` works on stationary-model emission coefficients. A covariate-dependent
model additionally admits the complete transition regression as one chart-free block:

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

For a dynamic transition model:

```python
dynamic = BernoulliGLMHMM(
    predictors=("stimulus",),
    transition_predictors=("session_order",),
    n_states=3,
)
pooled_dynamics = hierarchical(
    dynamic,
    over="subject",
    parameters=("transition",),
    scale=0.3,
)
fit = pooled_dynamics.fit(study)
```

`"transition"` expands to every source-state ILR intercept and every transition-covariate
ILR coefficient. One shared scale gives an isotropic Gaussian random effect in the
orthonormal transition coordinate. Naming one source, destination, or contrast is refused:
that subset is not closed under latent-state relabelling and would make the fitted prior
depend on state order. Emission and transition blocks may be named together, with separate
`parameter_scales` if their heterogeneity is not commensurable.

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

### Why stationary transitions stay pooled, and dynamic transitions need ILR

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

So "this animal is stickier" is not a deviation the stationary coordinate can carry
honestly. The dynamic transition model supplies the missing symmetric parameterisation:
each transition row is represented by a centred log composition and projected onto an
orthonormal Helmert basis. Relabelling destination states rotates that coordinate
orthogonally; relabelling source states permutes complete blocks. Consequently an isotropic
Gaussian on the complete `"transition"` block has the same density after any global state
permutation. This follows the ILR construction of
[Egozcue et al. (2003)](https://doi.org/10.1023/A:1023818214614).

The random effects are conditional MAP estimates under declared Gaussian scales. They are
not a fully Bayesian multilevel HMM, and an apparent between-subject transition difference
still needs enough state changes, prospective validation, and design-specific recovery.
Mixed and multilevel HMM literature commonly models transition probabilities with
multinomial-logit fixed and random effects; Behavio narrows that pattern to a
relabeling-invariant coordinate and a session-blocked behavioural likelihood.

`hierarchical(model, over="subject")` with the default `parameters=None` remains an error:
the initial distribution is still reference-coded, and a complete model-wide random effect
would include it. Name emission coefficients and/or `"transition"` explicitly.

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

**The group prior can, and is the only thing that does.** Subject \(g\)'s selected emission
and/or dynamic-transition block is the population's plus a deviation
\(b_g \sim \mathcal N(0, \sigma^2)\). Relabelling subject \(g\) alone leaves the likelihood
where it was but replaces \(b_g\) with
\(\Pi(\beta + b_g) - \beta\), which for well-separated population states is a much larger
vector and pays a much larger price. Per-subject relabelling is therefore **not** a symmetry
of the joint objective, and the label-consistent solution is its global optimum. The one
symmetry that survives is the *global* one — relabelling the population and every deviation
together — which is the same symmetry the pooled fit has, and `fit_rows` resolves it the same
way, by sorting states along `label_by`. It can do that to the whole joint vector at once
because relabelling is a **linear** map on this coordinate (emissions permute; the two
stationary simplexes permute and re-reference; dynamic ILR blocks permute and rotate), so the
covariance is carried through exactly rather than recomputed.
`model.relabelling_map(permutation)` is that matrix.

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

### Why generic `smooth()` still does not create a GLM-HMM path

`smooth()` is refused for this family, at any parameter selection:

```python
smooth(model, over="session_order", knots=(0.0, 4.0), parameters=("intercept",))
# TypeError: ... a GLM-HMM's latent labels are an ordering of one emission coefficient, and
# an ordering of coefficient *paths* is only a permutation where the paths do not cross ...
```

The ordering that names these states is an ordering of numbers. When `label_by` becomes a
path, states can be ordered one way early in training and the other way late, and no single
per-session ordering canonicalises the fit: "state 0" would name different behaviour at the
two ends of the clock. A path model therefore needs its own whole-path identity and crossing
evidence. `SessionDynamicBernoulliGLMHMM` supplies those under its specific Gaussian random-
walk/Dirichlet model; that does not make an arbitrary spline-composed GLM-HMM identified.
Keep the [smooth GLM](composing-models.md) as an observable competitor.

## Current boundary

Every implementation resets state at each session. The base class supports stationary or
exogenous covariate-dependent multinomial-logit transitions and Gaussian partial pooling of
complete dynamic transition regressions over grouping columns that respect session
boundaries. The one-subject session-dynamic class supports Gaussian-random-walk emissions
and independently Dirichlet-shrunken session transition matrices. The dedicated population
class adds a Gaussian population path, evolving subject deviations, and unseen-subject
plug-in prediction. Neither dynamic class estimates local covariance or its path scales.

The family still does not provide arbitrary smooth path-valued emissions, temporally smooth
transition paths, state carry-over across sessions, missing-outcome inference, semi-Markov
dwell times, or smoothed state reports. Do not approximate stochastic drift by putting
`session_order` into a transition regression: one is a deterministic conditional effect,
the other is a latent session path.
