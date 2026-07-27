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
  sample-size recovery benchmark. Parameter-specific longitudinal Wiener components and
  empirical-Bayes unseen-subject prediction are now implemented in 0.10–0.11; full
  Bayesian propagation remains later work.
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

## 0.8 — Longitudinal decision-process trajectories

- **Implemented:** represent Wiener drift coefficients, boundary separation, and starting
  bias as natural-scale fixed-knot paths over an explicit external study clock, while
  distinguishing across-trial change from within-decision dynamics.
- **Implemented:** retain non-decision time as stationary, require explicit shared-path
  opt-in for multi-animal data, and carry unsupported future knots forward through a
  time-scaled first-difference penalty.
- **Implemented:** expose read-only named parameter trajectories and support simulation,
  prospective scoring, generic parameter recovery, deterministic restarts, local-Hessian
  uncertainty, and complete fit audits.
- **Implemented:** compare static and smooth Wiener accounts under 20 stationary and 20
  changing matched designs; the scientifically matched family wins training-path recovery
  and held-out final-session joint log loss in both regimes.

## 0.9 — Partially pooled decision-process trajectories

- **Implemented:** jointly estimate smooth population Wiener paths and additive,
  Gaussian-shrunken subject-deviation paths for explicitly selected varying parameters.
- **Implemented:** expose immutable population and subject trajectories, retain random-
  effect truth outside observed studies, enforce effective natural-scale bounds, and use a
  declared population-trajectory plug-in for unseen animals.
- **Implemented:** derive local population and subject uncertainty from the numerical
  arrowhead Hessian and Schur complement while retaining restart and fit-audit evidence.
- **Implemented:** compare complete pooling, shared smooth, independent smooth, and
  hierarchical smooth fits across 20 stationary-identical, shared-change, and individual-
  change panels; the matched structure wins subject-path RMSE and future-session joint log
  loss in every regime, with all 480 fits converged.

## 0.10 — Parameter-specific decision-process heterogeneity

- **Implemented:** replace the single longitudinal Wiener subject scale with named,
  natural-scale components for each selected drift, boundary, or bias trajectory while
  retaining the common scalar as a backward-compatible fixed fallback.
- **Implemented:** estimate those components strictly inside each training fit with bounded
  Laplace-EM updates, retained iteration convergence, local-curvature uncertainty, and
  parameter-specific bound diagnostics.
- **Implemented:** pin a prospective recovery benchmark with unequal drift and boundary
  heterogeneity. Doubling animals from 6 to 12 lowers joint scale RMSE, all 16 fits
  converge, oracle predictive loss remains close, and poor local-interval coverage stays
  visible as a calibration limit.

## 0.11 — Variance-component and unseen-animal uncertainty

- **Implemented:** add an opt-in supplemented EM covariance for longitudinal Wiener
  subject scales by differentiating the fitted update in log-scale coordinates and
  accounting for missing information omitted by the local expected-prior curvature.
- **Implemented:** reject non-converged or locally unstable supplemented covariances rather
  than clipping an information matrix into apparent validity; retain the local standard
  errors, selected covariance, EM rate matrix, spectral radius, and declared interval
  bounds on immutable fit results.
- **Implemented:** integrate one coherent fitted random-effect trajectory per unseen animal
  and Monte Carlo draw, returning marginal choice probabilities, pointwise marginal joint
  densities, subject-joint log probabilities, effective draws, and log-score Monte Carlo
  standard errors.
- **Implemented:** pin a 20-panel benchmark in which supplemented intervals are finite in
  18 panels and improve coarse scale coverage, while random-effect integration improves
  the mean joint score across 80 unseen animals. The two stability failures and all Monte
  Carlo precision diagnostics remain part of the result.

## 0.12 — NWB and DANDI interoperability

- **Implemented:** ingest dataframe-like trial tables without treating their index as
  identity or chronology, preserving explicit source column and row order.
- **Implemented:** read and write scalar NWB `trials` tables behind optional dependencies,
  with lossless native canonical IDs, one-session export safeguards, explicit column
  mapping, PyNWB schema validation, and multi-file assembly that never infers session
  order from paths or timestamps.
- **Implemented:** resolve exact published DANDI versions and blob-backed NWB asset paths,
  stream only selected HDF5 datasets, and retain the asset path, ID, byte size, SHA-256,
  version, and NWB identifier as trial-addressable provenance.
- **Implemented:** pin a public 200-trial DANDI benchmark that preserves source IDs, row
  order, balanced task phases and categories, valid time intervals, and uninterpreted
  response semantics without redistributing the 72.6 MB source file.

## 0.13 — Cross-lab trajectory geometry

- **Implemented:** represent one explicitly aligned trajectory per independent subject on
  a fixed common clock, without implicit interpolation, landmark fitting, or time warping.
- **Implemented:** audit group replication before comparison and reject inferential lab
  summaries when any lab has fewer than the declared minimum number of animals.
- **Implemented:** decompose group-mean trajectories into overall level, centered change,
  amplitude, and scale-free shape, with trapezoid-weighted distances and explicit handling
  of flat trajectories whose shape is undefined.
- **Implemented:** quantify uncertainty by resampling subjects within fixed labs while
  stating that these intervals do not generalize to a population of laboratories.
- **Implemented:** pin a matched four-lab recovery benchmark and apply the design audit to
  the public IBL panel, preserving its one-animal-per-lab confounding as a failed readiness
  check rather than a lab-effect result.

## 0.14 — Exact IBL ONE interoperability and replicated public cohort

- **Implemented:** import exact IBL trial-table UUIDs through the optional ONE client,
  verify declared relative path, byte size, and MD5, and retain release, session, dataset,
  and Alyx identity as trial-addressable provenance.
- **Implemented:** preserve IBL source choice coding without silently converting `-1`, `0`,
  and `+1` into a binary modelling response.
- **Implemented:** build an outcome-blind manifest that retains all 78 eligible animals in
  nine labs, with at least four animals per lab and the first and final three pre-transition
  training sessions per animal.
- **Implemented:** pin 468 source datasets and a 260,833-trial public-data regression result,
  with replicated-lab trajectory geometry and nine complete leave-one-lab-out folds.
- **Bounded claim:** selection is conditioned on the training-policy transition, endpoint
  windows are ordinal rather than uniform elapsed time, and fixed-lab bootstrap intervals
  do not generalize to a population of laboratories.

## Later

- Session-varying GLM-HMM parameters
- Full propagation of population-parameter and scale uncertainty into predictions
- Richer mixtures and model plugins
- NWB-Zarr adapter
- Population-of-labs uncertainty and hierarchical lab effects

Each expansion should be justified by a benchmark or user need and should add recovery
tests before adding interpretive claims.
