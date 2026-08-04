# SDR-0067: Keep dynamic GLM-HMM uncertainty conditional on one whole-path label mode

- **Status:** Accepted and implemented
- **Date:** 2026-08-03
- **Related guide:** [Bernoulli GLM-HMM](../glm-hmm.md)

## Context

SDR-0065 and SDR-0066 introduced single-subject and population session-dynamic GLM-HMMs
without path covariance. Their Gaussian scales and transition concentration were fixed model
specifications, unseen-subject prediction used the population mean, and the fit explicitly
reported `uncertainty_policy="not-estimated"`.

Three distinct missing-information problems must not be collapsed. Latent HMM states make
the fixed-responsibility EM M-step curvature too precise; Gaussian path and hierarchy scales
are variance components rather than ordinary regression coefficients; and the likelihood is
unchanged by a global permutation of latent-state labels. A covariance calculated after
sorting each session independently would additionally splice different latent paths together.

## Decision

Compute local path covariance from the Hessian of the observed, state-marginalized negative
log posterior. The gradient recomputes forward-backward probabilities at every displaced
path point, so it includes latent-state missing information rather than differentiating the
fixed-responsibility EM objective. Hold the fitted transition layer and hyperparameters fixed
and state that conditioning in `uncertainty_policy`.

Apply the same one-time whole-path canonical permutation used by the point fit before
computing covariance. Report intervals as conditional on that canonical modal region. Never
average state-labelled coefficients over permutations and never independently relabel a
session. Retain `label_path_ambiguous`, all crossing arrays, the minimum gaps, and
`uncertainty_label_policy="conditional-on-one-whole-path-canonical-mode"` beside every
covariance. Thus an interval does not turn a weak label coordinate into identified state
meaning.

Add opt-in `estimate_hyperparameters=True`. Estimate each Gaussian path scale strictly from
the supplied training study by Laplace EM: the normalized Gaussian-prior M-step uses the path
mode plus its conditional covariance. Report log-scale observed information using Louis'
complete-minus-missing identity. If that information is not positive definite, difference
the fitted EM map and apply supplemented EM. Retain its rate matrix and spectral radius, and
leave the affected scale errors as `NaN` when the rate is unstable rather than clipping the
information matrix. Bounds, iterations, convergence, and boundary flags remain on the fit.

Estimate `transition_concentration` separately by bounded optimization of the conditional
Dirichlet-multinomial evidence formed from smoothed per-session transition counts. Report its
local log-scale curvature and conditional Dirichlet standard errors for every session row.
The Gaussian-scale and transition-concentration covariance blocks are reported as a declared
block-diagonal approximation; no cross-family covariance is inferred.

Keep ordinary `predict()` deterministic for prospective model comparison. Add
`predict_new_subjects()` for an entirely unseen population: each Monte Carlo draw contains
one shared draw of the fitted population path, future population random-walk increments, one
coherent evolving deviation per unseen subject, and one Dirichlet transition matrix per
subject-session. Retain draws, pointwise marginal scores, subject-joint log probabilities,
effective draws, and Monte Carlo errors. The population covariance draw is conditional on
the canonical label mode; pooled initial probabilities, the global transition centre, and
fitted hyperparameters remain fixed.

## Consequences

- Population, realized subject, and subject-deviation path errors are separate reportable
  arrays; none is reconstructed from marginal standard errors alone.
- A positive-definite path Hessian is required before hyperparameter estimation or
  population-path integration. Failure remains visible as unavailable uncertainty.
- Variance-component uncertainty can legitimately be absent in a small or weakly replicated
  panel even when the point fit converges. A spectral radius at or above one is evidence of
  instability, not an invitation to regularize the reported covariance silently.
- Hyperparameter estimation is empirical Bayes and local Laplace inference, not full
  Bayesian propagation. It remains a training-only alternative to nested prospective grid
  selection, not permission to tune on a held-out test session.
- Pointwise marginal Monte Carlo scores do not add to the subject-joint score. Deployment
  claims use the coherent subject-joint integral and retain its effective sample size and
  Monte Carlo error.

## Alternatives considered

**Invert the EM M-step Hessian.** Rejected because fixed responsibilities omit latent-state
missing information and systematically overstate precision.

**Order every covariance draw or session independently.** Rejected because ordering is not
identification and session-wise ordering can splice paths at scientifically important
crossings.

**Average over every label permutation.** Rejected for state-labelled coefficients because
the resulting means can describe no posterior mode. Permutation-invariant summaries remain
valid, while labelled intervals are explicitly mode-conditional.

**Clip a non-positive scale information matrix.** Rejected because it converts a weak or
unstable variance component into apparently finite evidence. Louis and supplemented-EM
failures remain part of the result.

**Replace deterministic prediction with Monte Carlo everywhere.** Rejected because it would
make ordinary pointwise comparison depend on simulation precision. Integrated unseen-subject
prediction is an explicit API with its own coherent joint score and numerical diagnostics.

**Call the approximation fully Bayesian.** Rejected. The fit does not integrate global
transition-centre, initial-distribution, or hyperparameter uncertainty and it conditions on
one label mode.

## Evidence

Louis' observed-information identity motivates subtracting missing information from
complete-data curvature ([Louis, 1982](https://doi.org/10.1111/j.2517-6161.1982.tb01203.x)).
The EM-rate correction follows the supplemented-EM construction
([Meng and Rubin, 1991](https://doi.org/10.1080/01621459.1991.10475130)). The conditional
transition calculation uses the normalized Dirichlet and Dirichlet-multinomial likelihood;
their concentration estimation is reviewed by
[Minka](https://tminka.github.io/papers/dirichlet/). The insistence on a declared relabelling
coordinate follows the known permutation invariance of mixture and HMM posteriors
([Papastamoulis, 2016](https://doi.org/10.18637/jss.v069.c01)).

Focused tests verify observed path covariance shapes and finiteness, interval construction,
hyperparameter estimation and boundary evidence, unstable supplemented-EM refusal, coherent
unseen-subject draws, subject-joint scoring diagnostics, deterministic seeding, and retention
of the fitted label-ambiguity flag.

## Revisit trigger

Revisit when a full posterior backend can sample the complete dynamic hierarchy with
permutation-aware draw alignment, or when repeated-design calibration shows that the local
path or variance-component intervals fail their declared conditional estimand. Such a
backend must still retain unresolved label modes rather than summarize them away.
