# Roadmap

The roadmap is organized by scientific contracts rather than by model count.

## 0.1 — The longitudinal contract

- **Implemented:** define and validate the canonical study schema.
- **Implemented:** represent session-order, cumulative-trial, elapsed-time, task-phase,
  and landmark-relative clocks without silently aligning them. Threshold landmarks use a
  fold-fitted transform contract with immutable training provenance; richer landmark
  definitions remain later work.
- **Implemented:** establish the generative model, prediction, scoring, diagnostics, and
  fit-result protocols.
- **Implemented:** add static and smoothly time-varying Bernoulli history GLM baselines,
  with an explicit clock, fixed knots, and inspectable coefficient trajectories.
- **Implemented:** add expanding forward-session, within-session rolling-origin, and
  deliberately non-prospective leave-one-session-out validation. Within-session evaluation
  preserves observed pre-origin history while scoring only future trials.
- **Implemented:** produce an end-to-end synthetic parameter-recovery report for the
  static GLM. Cross-model recovery remains part of 0.2.

## 0.2 — Competing explanations

- **Implemented:** add a fixed-transition Bernoulli GLM-HMM and a compact session-reset
  binary Q-learning agent, both with restart, recovery, and prospective competing-
  explanation diagnostics.
- **Implemented:** add parameter- and model-recovery grids with design-specific engines,
  a static-versus-smooth example, named-design and scenario-level matrices, fit-audit
  propagation, a matched four-family benchmark, and repeated boundary-near regimes with
  finite-simulation uncertainty intervals.
- **Implemented:** reproduce the Cell 2025 Figure 1 early-strategy/late-strategy result
  from a checksum-pinned public input while preserving a source session-identity collision.
- **Implemented:** validate the canonical longitudinal schema on a checksum-pinned IBL
  public-learning panel spanning nine labs, with trial-outcome-blind cohort selection and
  disjoint transition-anchored early/late-training windows.
- **Implemented:** make optimization, boundary, restart, occupancy, and label-ambiguity
  evidence first-class through a common fit-audit status and stable issue codes, without
  discarding model-specific diagnostics.

## 0.3 — Population structure

- **Implemented:** add complete-subject and complete-lab holdout folds with explicit group
  provenance, cross-lab subject-leakage rejection, and a pinned IBL coverage contract.
- **Implemented:** introduce a first constrained partial-pooling Bernoulli GLM with a
  fixed Gaussian subject scale, explicit seen/unseen-subject prediction, retained subject
  effects, and a matched recovery benchmark.
- **Implemented:** estimate one bounded shared subject scale with a Laplace marginal-
  likelihood approximation, local-Hessian uncertainty, boundary diagnostics, and a
  sample-size recovery benchmark. Multiple variance components and posterior predictive
  uncertainty remain later work.
- **Implemented:** add fixed-knot population trajectories with shrunken smooth subject-
  deviation paths, explicit unseen-subject prediction, and a factorial benchmark that
  distinguishes stationarity, stable heterogeneity, shared drift, and individual drift.
- **Implemented:** test latent-state recovery with a balanced permutation-invariant
  assignment, retained winning and runner-up scores, explicit missing-state and near-tie
  ambiguity, and a repeated clear-versus-overlapping emission benchmark.
- **Implemented:** quantify threshold-landmark and relative-clock uncertainty with a
  fold-safe plug-in Bernoulli bootstrap, explicit unresolved draws, frozen clock samples,
  and a repeated decisive-versus-marginal learning benchmark.
- **Implemented:** add cohort-level prospective session folds and use them in a matched
  six-session Cell/IBL study comparing four population/trajectory structures with
  subject-balanced scores, paired subject-bootstrap uncertainty, fit audits, and retained
  individual trajectories.

## 0.4 — Training-only comparison procedures

- **Implemented:** compare arbitrary behavioural-model candidates over common prospective
  folds with declared aggregation units, equal-unit and pooled log-loss/Brier summaries,
  paired unit-bootstrap intervals, complete fit audits, and JSON-safe fold provenance.
- **Implemented:** select candidates independently inside every outer training study,
  retain the complete inner comparison and selected outer fit, and aggregate the resulting
  outer performance as a selection procedure rather than relabelling it as one model.
- **Implemented:** migrate the flagship Cell/IBL study onto the reusable report and test
  the nested selector under stationary and strong shared-drift generators, including a
  direct regression that changes outer-test outcomes without changing inner selection.

## 0.5 — Extensible estimator and recovery contract

- **Implemented:** separate the prospective `BehaviourEstimator` contract from the
  simulation-capable `GenerativeBehaviourModel` contract while retaining `BehaviourModel`
  as the compatible full-model name.
- **Implemented:** declare and validate the complete observed columns scored by every
  pointwise likelihood, rejecting rankings between choice-only and joint-observation
  estimators.
- **Implemented:** validate plugin fit identity and row alignment at evaluation boundaries,
  and make complete fit audits, audit eligibility, finite-uncertainty denominators, and
  standards-compliant JSON serialization first-class in parameter recovery.

## 0.6 — Joint choice and response time

- **Implemented:** add an explicit positive response-time schema with seconds and
  milliseconds as typed physical units and canonical internal conversion.
- **Implemented:** add a fixed-parameter Wiener drift-diffusion family with covariate-
  dependent drift, joint choice/response-time scoring, analytic choice probabilities,
  paired first-passage expansions, and discretized generative simulation.
- **Implemented:** retain deterministic restart evidence, local-Hessian uncertainty,
  boundary warnings, minimum observed response time, and likelihood-floor counts.
- **Implemented:** validate density normalization and natural-scale parameter recovery,
  including a repeated 400-versus-1,200-trial benchmark in which every parameter's RMSE
  decreases with sample size.

## 0.7 — Explicit response contaminants

- **Implemented:** add a fixed-support uniform joint choice/response-time contaminant
  component with a fitted mixture probability and explicit physical-time units.
- **Implemented:** require contaminant support and non-decision-time search bounds to be
  fixed in model configuration, so held-out response times cannot define their own density.
- **Implemented:** retain simulated contaminant truth separately from observed studies and
  expose posterior trial responsibilities without converting them into hard exclusions.
- **Implemented:** compare contaminant-aware and naive Wiener fits on 20 paired designs;
  the robust model lowers every shared parameter RMSE and future-session joint log loss in
  all 20 repetitions.

## Later

- Session-varying GLM-HMM parameters
- Non-stationary and hierarchical response-time families
- Richer mixtures and model plugins
- NWB/DANDI streaming workflows
- Cross-lab trajectory-shape comparisons

Each expansion should be justified by a benchmark or user need and should add recovery
tests before adding interpretive claims.
