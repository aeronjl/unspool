# SDR-0066: Fit a population session-dynamic GLM-HMM with subject deviation paths

- **Status:** Accepted and implemented
- **Date:** 2026-08-03
- **Related guide:** [Bernoulli GLM-HMM](../glm-hmm.md)

> **Subsequent decision:** [SDR-0067](0067-keep-dynamic-glm-hmm-uncertainty-conditional-on-one-label-mode.md)
> adds canonical-mode path covariance, opt-in hierarchy
> hyperparameter estimation, and coherent unseen-subject Monte Carlo integration. The
> population/subject hierarchy and deterministic plug-in policies specified below remain
> unchanged.

## Context

SDR-0065 deliberately limited the published session-dynamic GLM-HMM to one subject. Fitting
that estimator separately to several subjects neither defines a population estimand nor
supports an unseen subject. Pooling all sessions into one path is worse: it treats a change
of animal as a temporal continuation and destroys the subject-specific history the emission
random walk is meant to represent.

The generic `hierarchical()` combinator cannot repair this. Its coordinate has fixed width
per group, while a session-dynamic GLM-HMM has one emission vector and transition matrix per
observed session. It also needs one population-level label permutation for the complete
latent path. Mixed-effects HMMs establish Gaussian subject effects as a standard way to
represent between-subject heterogeneity, but the dynamic GLM-HMM source paper did not fit the
specific population path introduced here. The extension therefore needs its own equations
and forecast policy.

## Decision

Add `HierarchicalSessionDynamicBernoulliGLMHMM` as a dedicated estimator. At sorted observed
population session order `r`, let the population emission path be

\[
M_r \sim \mathcal N(M_{r-1}, \sigma_{\mathrm{pop}}^2 I).
\]

For subject `m`, define the subject deviation at its first observed session and its later
increments by

\[
D_{m,0} \sim \mathcal N(0, \tau^2 I), \qquad
D_{m,s} \sim \mathcal N(D_{m,s-1}, \sigma_{\mathrm{subj}}^2 I),
\]

and set the subject-session emission weights to

\[
W_{m,s}=M_{r(m,s)}+D_{m,s}.
\]

The three declared scales are `population_emission_step_scale`, `subject_emission_scale`,
and the inherited `emission_step_scale`. Adjacent means adjacent observed population orders
or adjacent observed sessions for that subject; no elapsed-time interpretation is implied.

Retain the source transition distribution directly across all subject-session blocks,

\[
P_{m,s,i}\sim
\operatorname{Dirichlet}(\alpha\bar A_i+\mathbf 1).
\]

Do not add an undeclared subject-transition centre or temporal transition random walk. One
pooled initial-state distribution applies at every session opening.

Fit the model by the same three stages as SDR-0065: a pooled stationary multistart GLM-HMM;
an intermediate population/subject emission-path fit with one shared transition matrix; and
the full session-transition model. The joint emission M-step optimizes population and
subject paths together under the complete Gaussian penalty. Canonicalize labels once from
the mean population path and apply that permutation to every subject. Retain population and
subject crossing diagnostics separately.

For a fitted subject in a strictly later session, evaluate the population path at that order
and add the subject's last fitted deviation—the conditional mean of its deviation random
walk. For an unseen subject, use the population path itself, the zero-mean plug-in under the
subject distribution. Beyond the last fitted population order, carry the final population
weights forward. Both cases use the global transition prior mode and reset to the pooled
initial distribution. Refuse unseen earlier/interleaved sessions for a fitted subject.

Keep all three hierarchy scales fixed and declared in this first implementation. Report no
local covariance: standard errors and covariance remain `NaN` with
`uncertainty_policy="not-estimated"`. Scale estimation and label-aware uncertainty are a
separate capability rather than an implicit by-product of the MAP path.

## Consequences

- Simulation exposes population emissions, subject emissions, session transitions, and
  latent states as separate immutable truth.
- A fitted result retains both dynamic objective histories, population and subject label
  crossings, subject deviations, restart evidence, and common occupancy/boundary audits.
- Seen- and unseen-subject forecasts have different explicit policies and are prospectively
  scoreable through the normal estimator contract.
- Population and subject trajectory recovery use one truth-aware permutation; labels are
  never independently sorted by subject or session.
- Direct Dirichlet pooling of transition matrices supports unseen subjects but does not
  claim persistent subject-specific transition styles.
- `hierarchical()` and `smooth()` remain refused around this already hierarchical,
  data-dependent coordinate.

## Alternatives considered

**Fit one dynamic model per subject and average parameters.** Rejected because post-hoc
averaging does not shrink weak subjects, align population labels during fitting, or define
an unseen-subject distribution.

**Pool all subjects into one session path.** Rejected because it creates temporal links
across different animals and has no population estimand.

**Use independent subject deviations around each population order.** Rejected because it
would discard the within-subject temporal continuity that motivates the dynamic model.

**Give each subject another latent transition centre.** Deferred. A two-level Dirichlet
hierarchy requires its normalization terms when the centre is estimated and adds a distinct
scientific claim about stable subject transition styles. Direct population-to-session
pooling is the smallest coherent extension.

**Integrate unseen-subject deviations immediately.** Deferred to label-aware uncertainty.
The current zero-deviation population plug-in is explicit and matches the existing
hierarchical point-prediction convention.

## Evidence available at acceptance

- The joint population/subject Gaussian-path gradient matches finite differences.
- Multi-subject simulation is deterministic and retains read-only population, subject,
  transition, and latent-state truth.
- Clear-state simulations converge through both dynamic stages, reproduce their filtered
  likelihood through pointwise scores, and recover population and subject paths after one
  global alignment.
- Seen-future prediction carries the last subject deviation; unseen-subject prediction uses
  the population path; both use the global transition centre.
- One-subject fitting, unseen past sessions for fitted subjects, and another generic
  hierarchy are explicitly refused.

The statistical boundaries are informed by:

- [Dynamic GLM-HMM learning study](https://pmc.ncbi.nlm.nih.gov/articles/PMC11623682/)
- [Reference dynamic GLM-HMM implementation](https://github.com/lenca56/dynamic_glmhmm)
- [hmmTMB mixed-effects HMM methodology](https://doi.org/10.18637/jss.v114.i05)
- [Mixed HMMs for multiple behavioural series](https://pmc.ncbi.nlm.nih.gov/articles/PMC3293199/)

## Revisit trigger

Add label-aware path and hyperparameter uncertainty before interpreting population or
subject differences inferentially. Create another decision record before estimating the
three path scales, adding stable subject transition centres, nesting subjects within labs,
or integrating unseen-subject random effects rather than using the declared plug-in.
