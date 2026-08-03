# SDR-0064: Model dynamic transitions with multinomial logits and ILR group effects

- **Status:** Accepted and implemented
- **Date:** 2026-08-03
- **Related guide:** [Bernoulli GLM-HMM](../glm-hmm.md)

## Context

The stationary `BernoulliGLMHMM` learned one transition matrix and used
reference-category logits as its optimizer coordinate. That is a valid chart for fitting a
pooled simplex, but not for an exchangeable Gaussian transition effect: for three or more
states, changing the reference state is not an orthogonal transformation, so the same
isotropic normal declaration induces a different distribution on transition probabilities.
The package therefore correctly refused transition hierarchy.

Two distinct extensions were being called “dynamic”. Non-homogeneous HMMs make transition
probabilities deterministic functions of observed covariates through multinomial logits.
The dynamic GLM-HMM learning literature instead lets emission and transition parameters
evolve between sessions under temporal priors. They answer different questions and require
different recovery evidence.

## Decision

Implement observed transition non-homogeneity first, using one no-intercept exogenous design
and one multinomial-logit regression per source state. Trial `t`'s design row governs the
transition into trial `t`; session-opening rows are ignored because the chain resets to the
initial distribution. The stationary transition matrix is the baseline at a zero-valued
design row.

Store dynamic transition intercepts and slopes in an isometric log-ratio coordinate built
from an orthonormal Helmert basis. Report covariate effects naturally as centred destination
logits whose rows sum to zero, and report the complete transition matrix on every study row.
Ridge only the covariate effects, not the baseline matrix.

Allow `hierarchical(..., parameters=("transition",))` only for the complete dynamic
transition regression. The alias expands over every source state, ILR contrast, and
transition predictor under one declared scale. Refuse individual source, destination, or
contrast coordinates because such a subset is not closed under latent-state relabelling.
Continue to pool the reference-coded initial distribution, and continue to refuse
`parameters=None` because it would include that non-exchangeable chart.

Do not call an observed `session_order` transition effect a latent learning trajectory.
Session-varying latent parameters remain a separate model requiring their published priors,
path-level state labelling, path-crossing diagnostics, and targeted joint state/trajectory
recovery. That model is now implemented under the boundary in [SDR-0065](0065-fit-session-dynamic-glm-hmm-paths-by-map-em.md).

## Consequences

- Stationary models and their parameter names remain backward compatible.
- Dynamic models have exact simulation, forward-backward likelihoods and gradients,
  filtered prediction, pointwise scoring, natural transition reports, and session-blocked
  hierarchical fitting.
- A sticky Dirichlet pseudo-count is refused for dynamic transitions: a prior on one
  stationary matrix is not a prior on a regression-generated family of matrices.
- State-count selection, occupancy, restart evidence, label ambiguity, prospective scoring,
  and design-specific recovery remain required. The new coordinate makes a group prior
  invariant; it does not make latent states psychologically interpretable.
- Learned transition preprocessing must be fitted within training folds through a frozen
  `DesignSpec`, as for emission preprocessing.

## Alternatives considered

**Reference-category transition random effects.** Rejected because the prior changes with
the arbitrary reference state for `K >= 3`.

**Independent normals on centred destination logits.** Rejected as an overcomplete singular
coordinate. Projecting onto an orthonormal contrast basis gives the same centred natural
effects with an identified Euclidean parameter.

**Per-subject transition matrices fitted separately.** Retained as a descriptive fallback,
not partial pooling; it discards the population model and is weak when state changes are
rare.

**Treat session order as the dynamic GLM-HMM.** Rejected because a deterministic covariate
effect and stochastic parameter evolution are different estimands.

## Evidence available at acceptance

- Analytic dynamic-transition gradients match finite differences.
- Filtered pointwise scores sum to the dynamic forward likelihood.
- Simulation and natural transition reports use the same incoming-trial convention and
  reset at session boundaries.
- Three-state relabelling preserves transition probabilities, effects, and the Euclidean
  norm of the complete ILR block.
- A complete transition group block constructs, simulates, fits, and predicts through the
  existing session-blocked hierarchy; an individual coordinate is refused.
- The complete repository suite passes.

The statistical shape follows covariate-dependent and mixed HMM practice:

- [Bayesian variable selection in non-homogeneous HMMs](https://doi.org/10.1016/j.csda.2019.106840)
- [hmmTMB: HMMs with flexible covariate and random effects](https://doi.org/10.18637/jss.v114.i05)
- [A mixed non-homogeneous HMM](https://pubmed.ncbi.nlm.nih.gov/22302505/)
- [Isometric log-ratio transformations](https://doi.org/10.1023/A:1023818214614)
- [Dynamic GLM-HMM learning study](https://pmc.ncbi.nlm.nih.gov/articles/PMC11623682/)

## Revisit trigger

Revisit the complete-block scale structure when recovery studies can distinguish
source-specific or predictor-specific heterogeneity without losing relabelling invariance.
Revisit alongside SDR-0065 before adding alternative state-identification rules, temporal
transition smoothing, or state carry-over across sessions.
