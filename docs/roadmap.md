# Roadmap

The roadmap is organized by scientific contracts rather than by model count.

## 0.1 — The longitudinal contract

- **Implemented:** define and validate the canonical study schema.
- **Implemented:** represent session-order, cumulative-trial, elapsed-time, task-phase,
  and landmark-relative clocks without silently aligning them. Threshold landmarks use a
  fold-fitted transform contract with immutable training provenance; richer landmark
  definitions and uncertainty remain later work.
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
- Test latent-state alignment rather than assuming it.
- Quantify alignment and landmark uncertainty.

## Later

- Session-varying GLM-HMM parameters
- Reaction-time and drift-diffusion families
- Richer mixtures and model plugins
- NWB/DANDI streaming workflows
- Cross-lab trajectory-shape comparisons

Each expansion should be justified by a benchmark or user need and should add recovery
tests before adding interpretive claims.
