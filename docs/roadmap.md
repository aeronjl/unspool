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
- **In progress:** expand parameter- and model-recovery grids. The design-specific engines,
  static-versus-smooth example, named-design grid contract, audit propagation, and first
  matched four-family benchmark are implemented; repeated weak-signal grids remain.
- **Implemented:** reproduce the Cell 2025 Figure 1 early-strategy/late-strategy result
  from a checksum-pinned public input while preserving a source session-identity collision.
- **Implemented:** validate the canonical longitudinal schema on a checksum-pinned IBL
  public-learning panel spanning nine labs, with trial-outcome-blind cohort selection and
  disjoint transition-anchored early/late-training windows.
- **Implemented:** make optimization, boundary, restart, occupancy, and label-ambiguity
  evidence first-class through a common fit-audit status and stable issue codes, without
  discarding model-specific diagnostics.

## 0.3 — Population structure

- Introduce constrained partial pooling across subjects.
- Test latent-state alignment rather than assuming it.
- Add leave-subject-out and leave-lab-out evaluation.
- Quantify alignment and landmark uncertainty.

## Later

- Session-varying GLM-HMM parameters
- Reaction-time and drift-diffusion families
- Richer mixtures and model plugins
- NWB/DANDI streaming workflows
- Cross-lab trajectory-shape comparisons

Each expansion should be justified by a benchmark or user need and should add recovery
tests before adding interpretive claims.
