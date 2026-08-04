# SDR-0069: Sample GLM-HMM parameters after marginalizing the discrete state path

- **Status:** Accepted
- **Date:** 2026-08-03
- **Related decisions:** [SDR-0065](0065-fit-session-dynamic-glm-hmm-paths-by-map-em.md), [SDR-0067](0067-keep-dynamic-glm-hmm-uncertainty-conditional-on-one-label-mode.md), [SDR-0068](0068-model-laboratories-as-an-exchangeable-level-above-subjects.md)
- **Related guide:** [Bernoulli GLM-HMM](../glm-hmm.md)

## Context

The optimized GLM-HMM family reports observed-likelihood local curvature around one MAP
mode. That is useful numerical uncertainty, but it does not integrate emission paths,
transition matrices, hierarchy scales, or transition concentration over their joint
posterior. Calling those intervals “full Bayesian” would conceal both the plug-in layers
and latent-label symmetry.

NUTS cannot sample a discrete state at every trial. A finite HMM does not require it: the
forward recursion sums the state sequence out exactly and leaves a differentiable marginal
likelihood for continuous parameters. Symmetric state priors then make the posterior
invariant to complete permutations, so state-specific summaries require an explicit
post-sampling label policy.

## Decision

`PyMCBernoulliGLMHMM` is a separate proper-prior Bayesian wrapper over the stationary,
session-dynamic, population/subject, and population/lab/subject Bernoulli GLM-HMMs.

- Initial probabilities and transition rows receive normalized Dirichlet priors.
- Emission origins receive proper Normal priors. Gaussian path and hierarchy scales receive
  half-Normal hyperpriors, and session-transition concentration receives an exponential
  hyperprior.
- Population, laboratory, subject, and session paths use non-centred innovations. No fitted
  MAP path, transition matrix, or variance component enters the sampled likelihood.
- The discrete state sequence is marginalized by one scaled forward recursion per session.
  Its incremental normalizers are retained as filtered pointwise log likelihood.
- Each complete draw is relabelled after sampling by increasing `label_by`. Initial states,
  transition rows and columns, emissions, hierarchy deviations, and filtered state
  probabilities all receive the same permutation. Dynamic labels use one mean whole-path
  order; a crossing at any fitted session marks the draw ambiguous rather than triggering a
  different permutation at that session.
- Posterior prediction integrates one-step filtered choice probabilities over draws and
  replays only earlier observed choices. Posterior predictive choices use the same
  conditional history.
- Prior-joint simulation retains array-valued truth suitable for explicit SBC quantities.

The wrapper deliberately has a new model signature containing its normalized priors. The
optimized model's ridge terms, bounds, and empirical-Bayes estimates are not reinterpreted
as if they were this posterior model.

## Consequences

The first-party GLM-HMM line now has real joint posterior propagation through the deepest
nested dynamic hierarchy, including laboratory and subject paths and all supported variance
components. Static and fitted-session filtered scores satisfy the common sampled-estimator
and labelled `PosteriorResult` contracts.

This does not make label-specific summaries invariant scientific facts. The result retains
the permutation, minimum label gap, whole-path crossing flag, and ambiguity flag per draw.
Convergence must be audited after relabelling and recovery remains design-specific.

Dynamic extrapolation is intentionally narrower than the optimized model's Monte Carlo
predictors. The sampled wrapper currently predicts only the subject-session blocks present
in its posterior. Unseen sessions, subjects, and laboratories require posterior predictive
path propagation from every retained draw; until that is implemented they are refused, not
replaced by a MAP or posterior-mean plug-in.

## Alternatives considered

**Sample the discrete state sequence.** Rejected for NUTS because Hamiltonian methods require
continuous differentiable coordinates. The forward recursion computes the exact marginal
likelihood for this finite-state model.

**Order one emission coefficient inside the sampler.** Rejected as the only label treatment.
An artificial constraint can truncate the symmetric posterior and is especially misleading
when dynamic paths cross. Complete-draw post-processing preserves the symmetric model and
lets ambiguity remain observable.

**Draw Gaussian perturbations around the MAP result.** Rejected because local Laplace draws
would still condition on fitted transition and hyperparameter layers and would duplicate the
existing explicitly local uncertainty under a stronger name.

**Reuse flat coefficients from the fixed-scale PyMC GLM adapter.** Rejected because an
improper prior does not define a prior predictive joint or SBC target. The GLM-HMM wrapper
uses proper priors throughout.

## Revisit trigger

Revisit when posterior-draw path propagation can support future sessions, unseen subjects,
and unseen laboratories with group-joint scores and Monte Carlo diagnostics; when repeated
SBC establishes calibration across weak separation and path-crossing regimes; or when a
relabeling loss demonstrably outperforms the declared whole-path order for the intended
state-specific estimand.

## Primary methodological basis

- [Stan hidden Markov model functions and exact state marginalization](https://mc-stan.org/docs/functions-reference/hidden_markov_models.html)
- [Stephens (2000), label switching and posterior relabeling](https://doi.org/10.1111/1467-9868.00265)
- [Papaspiliopoulos, Roberts & Sköld (2007), centred and non-centred hierarchical parameterizations](https://doi.org/10.1214/088342307000000014)
- [Hoffman & Gelman (2014), NUTS](https://www.jmlr.org/papers/v15/hoffman14a.html)
- [Talts et al. (2018), simulation-based calibration](https://arxiv.org/abs/1804.06788)
