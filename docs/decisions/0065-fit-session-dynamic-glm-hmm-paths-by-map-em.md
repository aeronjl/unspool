# SDR-0065: Fit session-dynamic GLM-HMM paths by MAP EM

- **Status:** Accepted and implemented
- **Date:** 2026-08-03
- **Related guide:** [Bernoulli GLM-HMM](../glm-hmm.md)

> **Subsequent decision:** [SDR-0067](0067-keep-dynamic-glm-hmm-uncertainty-conditional-on-one-label-mode.md)
> supersedes this record's initial decision to omit local
> covariance and keep both dynamic hyperparameters outside every fit. The path model,
> three-stage initialization, transition prior, label rule, and forecast policy below remain
> unchanged.

## Context

SDR-0064 separated observed covariate-dependent transition regression from latent
session-to-session parameter variation. The remaining model could not be obtained by
passing `session_order` to that regression or by applying the generic `smooth()`
combinator. The published dynamic GLM-HMM gives state-specific emission weights a Gaussian
random walk, but gives each session transition row an independent Dirichlet distribution
around a global transition row. Calling both paths “smooth” would change the model.

A session path also makes the stationary label convention insufficient. Ordering each
session independently can splice one latent state onto another whenever emission paths
cross. A prospective library additionally needs to say what an unseen later session means;
the source study selected hyperparameters using held-out blocks within observed sessions and
did not define that deployment rule.

## Decision

Add `SessionDynamicBernoulliGLMHMM` as a distinct single-subject estimator. For session
`s`, fit

\[
\beta_k^{(s)}\sim\mathcal N(\beta_k^{(s-1)},\sigma^2 I),\qquad
A_i^{(s)}\sim\operatorname{Dirichlet}(\alpha\bar A_i+\mathbf 1).
\]

Use the published three-stage initialization: fit the existing stationary multistart
GLM-HMM; fit an emission-dynamic intermediate model with one re-estimated shared transition
matrix; then initialize the fully dynamic model from that partial fit. Both dynamic stages
use exact forward-backward expectations and a joint analytic-gradient L-BFGS-B emission
M-step carrying every adjacent-session Gaussian term. The full stage uses the closed-form
Dirichlet pseudo-count transition update around the partial stage's shared matrix `bar A`.
Retain both objective histories and both convergence decisions. Estimate one pooled initial
distribution across session openings. This differs from the cited implementation's uniform
first-session state prior.

Keep `K`, `sigma`, and `alpha` outside the EM coordinate. Select their declared candidate
grid through the existing nested prospective comparison, whose inner selector sees only the
outer training study. This is stricter than selecting from held-out blocks and then making a
future-session claim from the same observed sessions.

Canonicalize states once for the complete path by the mean declared label coefficient.
Never reorder states independently by session. Retain every adjacent-session pairwise label
crossing or near contact, the minimum label-path gap, and a path-ambiguity flag. Simulation
retains state, emission-path, transition-path, and global-transition truth outside the
observed study. Recovery uses one whole-path truth alignment before reporting state accuracy
and emission/transition path RMSE.

For a strictly later unseen session of the fitted subject, carry the final emission weights
forward—the conditional mean of the random walk—and use the global transition matrix—the
mode of the added-one Dirichlet prior. Reset to the fitted pooled initial distribution. Refuse
unseen earlier or interleaved sessions and unseen subjects. This is a Behavio forecast
policy, not a result attributed to the source paper.

Do not report a local covariance for this first implementation. Store `NaN` standard errors
and covariance with `uncertainty_policy="not-estimated"`, so the normal fit audit exposes the
limitation. Refuse generic smooth or hierarchical composition, transition covariates, and a
stationary sticky prior. The separate cross-subject population model is specified by
[SDR-0066](0066-fit-a-population-session-dynamic-glm-hmm-with-subject-deviation-paths.md).

## Consequences

- Session-specific emission and transition parameters are immutable, directly reportable,
  and used by the same filtered recursion for prediction and pointwise scoring.
- `emission_step_scale` and `transition_concentration` are scientific hyperparameters in the
  model signature. They must be selected inside training data.
- Transition concentration is shrinkage toward a global matrix, not temporal transition
  smoothness. No transition random walk is claimed.
- The path keeps latent identity through crossings instead of manufacturing ordered states
  at every session. Crossing flags warn that `state 0` cannot be read as a stable ordinal
  emission category over learning.
- One-subject fitting preserves subject/session boundaries rather than pooling unrelated
  paths. It does not support unseen-subject population prediction.
- A converged point fit remains only a candidate explanation. Observable history, smooth
  drift, learning, stationary GLM-HMM, and covariate-transition competitors remain required.
- The first matched training-only benchmark recovers two-state dynamic paths strongly but
  does not show lower future-session log loss than the stationary GLM-HMM. That negative
  prospective result remains visible rather than being replaced by descriptive recovery.

## Alternatives considered

**Apply `smooth()` to a stationary GLM-HMM.** Rejected because independent spline paths can
cross without a path-label rule, and the generic bounded-coordinate contract cannot carry
the nonquadratic Dirichlet session priors or a session-dependent coordinate dimension.

**Regress transitions and emissions on session order.** Rejected because a deterministic
trend is not a stochastic random walk or a set of session deviations.

**Random-walk transition logits.** Not selected because it is not the transition prior in
the target dynamic GLM-HMM. It remains a legitimate separate research model.

**Independently canonicalize every session.** Rejected because it can swap latent identity
exactly where path crossings are scientifically important.

**Fit all subjects as one sequence collection.** Rejected because that implies neither a
population distribution nor independent subject paths and would silently overstate the
estimand.

**Reuse the last fitted session transition matrix for forecasting.** Rejected because
published session matrices are conditionally independent around the global matrix. The
global centre, not the last deviation, is the coherent new-session plug-in.

## Evidence available at acceptance

- The analytic joint emission M-step gradient matches finite differences with both
  random-walk and ridge terms.
- The session transition M-step exactly matches expected transition counts plus the
  declared Dirichlet pseudo-counts.
- Simulated trajectories, choices, and latent states are reproducible and retained
  separately from observed columns.
- Fitted pointwise scores sum to the session-parameter forward likelihood.
- Clear-state simulations recover latent states and produce finite truth-aligned emission
  and transition trajectory errors.
- The intermediate stage retains a finite objective history and fits one shared transition
  matrix before session transition deviations are admitted.
- Future-session tests pin the carried emission and global-transition policy, while unseen
  past sessions, subjects, and undeclared hybrid transition models are refused.
- Nested forward-session selection jointly covers state count, emission step scale, and
  transition concentration without exposing outer-test choices. In the four-repetition
  matched benchmark's dynamic-truth regime, path recovery was strong, but the stationary
  GLM-HMM retained the lower mean future-session log loss. The stationary-truth regime also
  exposed one selected three-state specification whose later full-path partial stage did not
  converge.

The model equations and fitting structure follow:

- [Internal states emerge early during learning and affect the dynamics of future behavioural choices](https://pmc.ncbi.nlm.nih.gov/articles/PMC11623682/)
- [Reference implementation](https://github.com/lenca56/dynamic_glmhmm)
- [Ashwood et al. input-driven GLM-HMM](https://doi.org/10.1038/s41593-021-01007-z)

## Revisit trigger

The original selection/competitor trigger is now met as an implementation and falsification
contract, not as positive prospective evidence. Revisit the forecast claim only with a
preregistered, higher-information design or public longitudinal panel that separates
stationary and session-dynamic predictions prospectively. Create a separate decision record
before adding temporal transition smoothing, state carry-over between sessions, or
uncertainty claims beyond the retained MAP point fit. Cross-subject hierarchy now has its
own record and estimator under SDR-0066.
