# Roadmap

The roadmap is organized by scientific contracts rather than by model count.

## 0.1 — The longitudinal contract

- **Implemented:** define and validate the canonical study schema.
- Represent multiple learning clocks without silently aligning them.
- **Implemented:** establish the generative model, prediction, scoring, diagnostics, and
  fit-result protocols.
- **In progress:** add static GLM and smooth-drift baselines. The static Bernoulli history
  GLM is implemented; smooth drift remains its first nonstationary competitor.
- **In progress:** add session and rolling-origin validation. Expanding forward-session and
  leave-one-session-out folds are implemented; within-session origins remain.
- **Implemented:** produce an end-to-end synthetic parameter-recovery report for the
  static GLM. Cross-model recovery remains part of 0.2.

## 0.2 — Competing explanations

- Add fixed GLM-HMM and compact reinforcement-learning reference models.
- Add parameter- and model-recovery grids.
- Make label ambiguity and optimization diagnostics first-class outputs.
- Reproduce a bounded Cell 2025 behavioural result.
- Add an IBL public-learning benchmark.

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
