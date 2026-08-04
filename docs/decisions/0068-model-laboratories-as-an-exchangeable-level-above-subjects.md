# SDR-0068 — Model laboratories as an exchangeable level above subjects

- **Status:** Accepted
- **Date:** 2026-08-03

## Context

Behavio already treated laboratory as a protected validation unit and could compare fixed
empirical laboratory trajectories. Neither operation defines a population of laboratories.
The cross-subject session-dynamic GLM-HMM likewise pooled subjects around one population
path, so it could not separate a laboratory-shared trajectory from subject heterogeneity.

Mixed hidden Markov models use explicit continuous random effects to represent differences
between repeated processes ([Altman 2007](https://doi.org/10.1198/016214506000001086)).
Multilevel mixed HMMs have also separated subject and higher cluster levels
([Zhang et al. 2014](https://doi.org/10.1002/sim.6039)), and `hmmTMB` estimates variance
parameters and predicts random effects through a Laplace approximation
([Michelot 2025](https://doi.org/10.18637/jss.v114.i05)). These precedents establish the
random-effects pattern, not Behavio's exact dynamic hierarchy. Multi-laboratory behavioural
data make the higher level scientifically material: the IBL standardized task found
variation in learning speed across both mice and laboratories even though trained behaviour
was reproducible ([IBL 2021](https://doi.org/10.7554/eLife.63711)), while prospective
multi-lab work explicitly uses a random-laboratory model to quantify between-lab
replicability ([Jaljuli et al. 2023](https://doi.org/10.1371/journal.pbio.3002082)).

## Decision

Add a separate `LabHierarchicalSessionDynamicBernoulliGLMHMM`; do not make a `lab` column
silently change the existing cross-subject model. For population session order (r),
laboratory (l), and subject (m) nested within that laboratory, define

\[
M_r\sim\mathcal N(M_{r-1},\sigma_{\mathrm{pop}}^2I),\qquad
L_{l,0}\sim\mathcal N(0,\tau_{\mathrm{lab}}^2I),\qquad
L_{l,s}\sim\mathcal N(L_{l,s-1},\sigma_{\mathrm{lab}}^2I),
\]

\[
D_{m,0}\sim\mathcal N(0,\tau_{\mathrm{subj}}^2I),\qquad
D_{m,s}\sim\mathcal N(D_{m,s-1},\sigma_{\mathrm{subj}}^2I),\qquad
W_{m,s}=M_{r(m,s)}+L_{l(m),s}+D_{m,s}.
\]

The five Gaussian scales are either fixed or estimated inside training data by the same
bounded Laplace-EM procedure used by the two-level model. The fit exposes population,
laboratory-deviation, realized laboratory, realized subject, and subject-deviation path
uncertainty from the observed state-marginalized objective. Louis or supplemented-EM
variance-component uncertainty, conditional transition-concentration uncertainty, boundary
estimates, and failed information corrections remain visible. One whole-path permutation
canonicalizes population, lab, and subject coordinates together; intervals remain
conditional on that label mode.

Subjects must occur in exactly one laboratory. Fitting and simulation require at least two
laboratories and at least two independent subjects in every laboratory. These are
identifiability safeguards, not an assertion that two laboratories provide precise
population-of-laboratories inference. Strong generalization claims require a substantively
defined exchangeable lab population, more independent laboratories, design-specific
recovery, and complete-lab prospective validation.

Prediction distinguishes four targets:

1. fitted subject-sessions reuse fitted paths;
2. later sessions of a fitted subject carry its final deviation around the relevant
   population-plus-lab path;
3. a new subject in a fitted lab has a zero-deviation plug-in or an integrated subject-path
   prediction whose coherent score is subject-joint; and
4. a wholly new lab has a zero-lab/zero-subject plug-in or `predict_new_labs()`, which draws
   one lab path shared by all of its subjects and reports a lab-joint predictive score.

Earlier or interleaved missing population/lab orders are refused. The transition layer
remains independent subject-session rows drawn around one global matrix; no laboratory or
subject-specific persistent transition style is inferred.

## Consequences

- Laboratory becomes a modeled sampling level, not a synonym for a holdout label or fixed
  set of curves.
- Shared laboratory draws preserve dependence among subjects during unseen-lab prediction;
  summing pointwise marginal scores is explicitly not the same estimand.
- Complete-lab holdouts exercise the deployment claim without leaking animals or future
  laboratory observations into fitting.
- The model assumes exchangeable Gaussian laboratory paths. It does not identify which
  environmental mechanism caused a lab difference and does not make observed labs
  representative by construction.
- The additional path and five variance components increase curvature cost and make sparse
  designs visibly unstable rather than silently collapsing levels.

## Alternatives considered

- **Treat lab as a fixed covariate.** This describes the observed labs but supplies no
  distribution or predictive target for an unseen lab.
- **Reuse the existing subject hierarchy with lab labels as subjects.** This removes the
  independent animal level and confounds within-lab subject variation with lab variation.
- **Fit labs separately and meta-analyse paths.** This loses joint label anchoring and does
  not propagate the nested likelihood into subject or unseen-lab prediction.
- **Add laboratory-specific transition centres now.** This is a scientifically distinct,
  substantially less identified model and is deferred until designs can recover it.
- **Require a full Bayesian sampler first.** Full posterior propagation remains desirable,
  but it is not required to make the hierarchy, prediction unit, local uncertainty, and
  instability evidence explicit.

## Revisit trigger

Revisit when design-specific recovery supports persistent laboratory transition styles,
cross-classified subjects or apparatuses require a non-nested model, elapsed-time rather
than observed-order lab dynamics are scientifically justified, or a full posterior backend
can integrate labels and all hierarchy scales without weakening the current diagnostics.
