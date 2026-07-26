# Scientific scope

## Objective

Unspool will provide a common, tested contract for fitting and comparing generative
models of trial-level behaviour across subjects and sessions. Its differentiator is the
evaluation layer around the models: temporal validation, recovery, diagnostics, and
explicit comparisons among alternative accounts of nonstationarity.

## Proposed v0.1

The first useful release should contain:

1. A canonical study schema with subject, session, trial, explicit session chronology, and
   source-specific columns. Outcomes, timing, and task covariates remain unopinionated
   source fields until model protocols define their roles.
2. Explicit clocks for session order, cumulative trials, elapsed time, task phase, and
   data-derived landmarks.
3. A small model protocol covering simulation, fitting, pointwise log probability,
   prediction, and diagnostics.
4. Four reference families: a static psychometric/history GLM, a smooth dynamic GLM,
   a fixed-transition GLM-HMM, and a compact reinforcement-learning model.
5. Whole-session, rolling-origin, leave-subject-out, and leave-lab-out splitters.
6. Parameter-recovery and model-recovery reports tied to an experimental design.
7. Optional adapters for tabular data, IBL ONE, and NWB; the core representation should
   remain lightweight and format-independent.
8. Reproducible benchmarks against the Cell 2025 study and public IBL learning data.

The current executable slice covers the canonical study contract; typed design and
landmark-relative clocks; fold-fitted threshold landmarks; complete-session and
within-session rolling-origin validation; common model outputs; static and smoothly
time-varying Bernoulli history GLMs; a fixed-transition Bernoulli GLM-HMM; and design-specific
parameter and model recovery. See the
[clock and transform guide](clocks-and-transforms.md), [modelling guide](modelling.md),
[smooth-drift guide](smooth-drift.md), [GLM-HMM guide](glm-hmm.md), and
[model-recovery guide](model-recovery.md) for their assumptions and current boundaries.

## Non-goals for v0.1

- A catalogue of every cognitive model
- Full hierarchical Bayesian inference
- Drift-diffusion and arbitrary mixture families
- A universal learning landmark
- Automatic cognitive interpretation of latent states
- A graphical interface or hosted analysis service
- GPU acceleration as a baseline requirement

These may become later extensions after the shared data, model, and evaluation contracts
have survived real benchmarks.

## Claim discipline

Recovery is conditional on the simulated design, parameter distribution, missingness,
and sample size. Passing one recovery experiment does not establish global identifiability.
Likewise, held-out predictive performance does not by itself identify a cognitive
mechanism. Unspool should report those distinctions rather than collapse them into a
single model ranking.
