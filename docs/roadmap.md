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

## 0.15 — Prospective modelling on the replicated IBL cohort

- **Implemented:** add a combined held-out-lab/future-session splitter that requires common
  aligned session coordinates, excludes every held-out animal from fitting, and excludes
  later training-lab sessions beyond the declared origin.
- **Implemented:** compare static partial pooling with hierarchical smooth drift on a
  predeclared 46,152-trial panel, with a source-row cap applied before choice eligibility
  and no outcome-fitted preprocessing or hyperparameters.
- **Implemented:** separate same-animal future prediction from unseen-lab future transport,
  retaining subject-balanced and lab-balanced paired uncertainty and all 20 fit audits.
- **Evidence:** smooth drift improves same-animal final-session log loss by `0.0851` with a
  paired interval above zero; its smaller held-out-lab advantage remains unresolved under
  both subject- and lab-balanced estimands.
- **Bounded claim:** endpoint windows are ordinal, cohort entry is transition-conditioned,
  and nine empirical labs do not support population-of-laboratories inference.

## 0.16 — Nested model and hyperparameter selection on replicated IBL

- **Implemented:** compare static partial pooling with three declared smoothness levels
  entirely inside the outer training data, with deterministic candidate ordering and
  training-only tie breaking.
- **Implemented:** match inner validation to each outer estimand: earlier cohort-session
  forecasts for represented animals and inner held-out-lab position-4 forecasts for
  unseen-lab transfer.
- **Implemented:** pin all inner targets, candidate scores, selections, outer fit audits,
  subject scores, and paired comparisons with the previously fixed candidates.
- **Evidence:** smoothness 9 is selected in the one same-animal fold and all nine lab folds;
  untouched outer log loss improves on fixed smoothness 3 by `0.00768` and `0.00777`, with
  both paired subject-bootstrap intervals below zero.
- **Bounded claim:** smoothness 9 is the upper grid boundary, one same-animal outer fold is
  not a selection-stability sample, and the selected procedure's unseen-lab advantage over
  static remains unresolved.

## 0.17 — Scientist-facing documentation and worked evidence

- **Implemented:** publish the existing scientific contracts as a searchable, strictly
  validated documentation site organized by question rather than Python module.
- **Implemented:** distinguish supported, experimental, planned, and out-of-scope
  capabilities in one scientist-facing matrix.
- **Implemented:** turn the Cell 2025 reproduction, replicated IBL trajectories,
  prospective nested selection, and four-family recovery grid into worked literature
  studies with explicit denominators, estimands, limitations, and reproduction commands.
- **Implemented:** generate twenty-two versioned figures: fourteen empirical or simulation-based
  evidence displays from checksum-pinned public inputs and committed benchmark results,
  plus eight clearly labelled conceptual maps of workflow, validation, model structure, and
  diagnostics. Every figure has descriptive alternative text, an interpretive caption,
  and an auditable entry in the figure provenance register.
- **Implemented:** build documentation on every pull request and deploy the exact successful
  main-branch artifact to GitHub Pages.
- **Bounded claim:** the site makes current evidence legible; it does not expand the
  scientific coverage declared by the capability matrix.

## 0.18 — Public joint-outcome and latent-state studies

- **Implemented:** extend the frozen IBL cohort into a one-animal joint
  choice/movement-onset-response-time study with a fixed eligibility window, physical
  units, an untouched future session, and naive-versus-contaminant Wiener evidence.
- **Implemented:** retain the negative robust-versus-naive result, model-dependent trial
  responsibilities, a posterior-predictive response-time draw, and the boundary warning
  rather than presenting robustness as automatic improvement.
- **Implemented:** translate the question in Ashwood et al. into a structural analogue
  with training-only 2/3/4-state selection, a static history-GLM comparator, and
  one-step-ahead scoring in an untouched session.
- **Implemented:** expose the near-tied inner selection, improved outer score, fitted
  state probabilities, and warning-level restart, boundary, and curvature diagnostics.
- **Bounded claim:** both examples concern one outcome-blindly selected animal and one
  future session; they demonstrate complete software workflows, not population effects,
  mechanistic identification, or a reproduction of Ashwood et al.

## 0.19 — Cell 2025 behavioural flagship

- **Implemented:** refactor the public behavioural analysis from Liebana, Laffere et al.
  into two explicit layers: independently reproduced retrospective results and a new
  historical-cohort-calibrated prospective estimand.
- **Implemented:** preserve 391 source sessions while deriving the paper's 390 modeling
  days, align days 1–8 and each animal's final five sessions, and exclude intervening
  forecast-animal outcomes from every fit and transform.
- **Implemented:** add the reusable `historical_cohort_forecast_splits` contract and
  compare six frozen candidates in six animal-level folds with subject-balanced scores,
  paired animal-bootstrap intervals, and retained numerical audits.
- **Evidence:** early bias has the lowest mean log loss (`0.58109`) and improves on pooled
  psychometric prediction by `0.04219` (`95% CI 0.01818–0.06425`), while its incremental
  value over a late-phase control and hierarchical smooth drift remains unresolved.
- **Implemented:** reproduce the released Gaussian-process/soft-DTW membership exactly,
  safely summarize the released Q-value artifact, and independently reproduce response-
  time changes without presenting clusters as natural kinds or released fits as new
  optimization.
- **Implemented:** run structural, hierarchical-path, outcome-derived-feature, and reward-
  history competing-explanation recovery on the exact 73,042-trial design, retaining the
  complete-pooling ambiguity and null-world false selections.
- **Implemented:** publish a paper-style worked chapter and four new evidence figures with
  explicit denominators, provenance, deployment assumptions, and claim boundaries.
- **Bounded claim:** the study establishes internally validated behavioural forecastability
  in one cohort. It does not uniquely identify a latent mechanism, establish causality,
  reproduce multimodal/neural analyses, or demonstrate transport to another laboratory.

## 0.20 — Reproducible study protocols

- **Implemented:** make a typed, immutable, versioned `StudyProtocol` the scientific
  boundary for source provenance, outcome-blind cohorts, units, observations, clocks,
  panels, estimands, transforms, deployment geometry, fixed candidates, uncertainty,
  nested selection, exact-design recovery, and bounded reporting.
- **Implemented:** give each declaration canonical JSON, a stable scientific fingerprint,
  explicit pre-evidence amendments, and an evidence-backed lifecycle from draft through
  reported.
- **Implemented:** compile canonical studies into exact fit, prediction-context, score,
  and exclusion rows; audit temporal, animal, laboratory, experimental-unit, capability,
  transform-visibility, and nested-selection boundaries before fitting.
- **Implemented:** execute candidates through one common runner that retains fits,
  pointwise predictions, equal-unit and pooled scores, calibration, paired uncertainty,
  numerical audits, failures, and unresolved decisions.
- **Implemented:** make model, parameter, and outcome-derived-feature recovery first-class
  claim gates executed through the identical compiled design.
- **Implemented:** produce deterministic content-addressed evidence bundles with protocol,
  amendment, environment, source, cohort, plan, fold, audit, comparison, prediction,
  recovery, figure, report, replay, and bundle-comparison evidence, without raw-data
  redistribution or executable serialization.
- **Implemented:** expose a closed, lean command line for validation, execution,
  inspection, comparison, and report extraction rather than a general scheduler or
  arbitrary configuration language.
- **Implemented:** prove generality by migrating the Cell 2025 flagship and the public IBL
  same-animal/held-out-laboratory nested-selection study with exact denominator, fold,
  score, interval, audit, selection, and recovery parity.
- **Bounded claim:** a protocol and internally consistent evidence bundle make an analysis
  reviewable and reproducible against its identified source. They do not make the source
  unbiased, the model identifiable, or the estimand transport beyond its declared
  population.

## 0.21 — Dependable golden path and extension contracts

- **Implemented:** map arbitrary dataframe identity columns into the canonical longitudinal
  names through the documented `Study.from_dataframe()` interface.
- **Implemented:** introduce a model-independent `TaskSpec` for categorical choices,
  explicit omissions, trial-specific option availability, bounded scalar rewards, physical
  response-time units, predictors, blocks, and episodes, with retained denominator audits.
- **Implemented:** add `fit_model()` as the first task-validated interactive path, retaining
  the complete model-specific fit and common numerical audit.
- **Implemented:** define fixed reusable numeric, categorical, interaction, and lagged-
  history design components with stable names and explicit reset boundaries.
- **Implemented:** add fixed history kernels and an explicit training-only standardization
  transform that freezes into the same design-term contract without seeing prediction rows.
- **Implemented:** stabilize a context-bound common fit-artifact schema and an instance-
  scoped public estimator registry through which external models can be added without
  editing Unspool core or serializing executable objects.
- **Release boundary:** load, specify, fit, diagnose, compare, simulate/recover, and report
  must form one coherent public workflow, with every result retaining model, task, data,
  version, and numerical provenance.

## 0.22 — Canonical behavioural model catalogue

- **Implemented:** add named bias-only, psychometric, lapse-mixture, perseveration, and
  outcome-conditioned win-stay/lose-shift baselines with complete simulation, fit,
  prediction, score, audit, recovery, reward semantics, and reset boundaries.
- **Implemented:** make symmetric or asymmetric value updating, unchosen forgetting,
  exponential choice kernels, bias/lapse softmax policies, and reset columns composable in
  a binary RL agent while retaining the original validated Q-learning model.
- **Implemented:** support multinomial and omission-aware choice likelihoods on the common
  task coordinate, including finite JSON-scalar labels, trial availability, categorical
  predictions, prospective comparison, protocol artifacts, and recovery.
- **Implemented:** expose the standard input-driven GLM-HMM convention through
  state-specific task-input emissions and add an explicit sticky Dirichlet self-transition
  prior, retaining state-count, occupancy, alignment, and prospective-selection
  diagnostics. Covariate-dependent transitions remain a distinct later extension.
- **Planned:** cover standard DDM regressions, collapsing bounds, and race/LBA models through
  first-party reference components or adapters to mature packages rather than duplicating
  validated solvers.
- **Release boundary:** every catalogue model must simulate, fit, predict, score pointwise,
  diagnose, document its assumptions, and participate in design-specific recovery.

## 0.23 — Fitting and diagnostic interoperability

- **Implemented:** define a portable, content-addressed parameter space with stable natural
  and optimizer coordinates, transforms, scientific and plausible bounds, numerical fit
  bounds, fixed/free roles, normalized priors, density Jacobians, strict round trips, and
  fit-artifact metadata; integrate it through the reference Q-learning optimizer as the
  first backwards-compatible consumer.
- **Implemented:** expose backend-neutral deterministic MLE/MAP problems and complete run
  records, including explicit natural-versus-optimizer MAP measure, analytic prior and
  Jacobian gradients, immutable backend configuration, and every attempted optimum; route
  reference Q-learning through the first SciPy L-BFGS-B multistart backend without breaking
  its existing restart diagnostics.
- **Implemented:** add an optional, independently seeded PyBADS multistart implementation
  of the identical backend contract, requiring declared finite plausible bounds, restoring
  upstream global RNG state, retaining backend failures, and conservatively distinguishing
  limit termination from convergence without changing model or task semantics.
- **Implemented:** integrate established PyMC NUTS sampling for the existing fixed-scale
  hierarchical Bernoulli history GLM, reusing task validation and filtered-history design,
  preserving its flat/L2/Gaussian prior semantics, retaining likelihood and predictive
  groups, and rejecting empirical-Bayes scale estimation until a full-posterior scale prior
  is explicitly declared; test real PyMC 5 and 6 paths under dedicated CI.
- **Implemented:** retain immutable, fully labelled posterior, predictive, observed-data,
  constant-data, pointwise log-likelihood, and per-draw diagnostic groups in a NumPy-native
  result with explicit model, backend, and parameter-space provenance; export and import
  the same contract through optional ArviZ `InferenceData` and current xarray `DataTree`
  conventions, with both supported Python-version paths under dedicated CI.
- **Planned:** add posterior predictive checks, simulation-based calibration, PSIS-LOO,
  sensitivity, and test-retest reliability alongside Unspool's prospective validation and
  exact-design recovery.
- **Release boundary:** changing an inference backend must not change task semantics,
  prediction information, split geometry, scored observations, or result interpretation.

## 0.24 — Literature recipes and ecosystem documentation

- **Planned:** publish end-to-end public-data recipes for psychometric/history models,
  bandit RL, GLM-HMMs, perceptual DDMs, and longitudinal learning comparisons.
- **Planned:** make every recipe produce publication-quality figures through public APIs,
  with data provenance, expected runtime, diagnostics, recovery evidence, and explicit
  claim limits.
- **Planned:** add task and model decision guides, model cards, and migration guides from
  hand-written SciPy, `ssm`, hBayesDM, HDDM, and PyDDM workflows.
- **Planned:** document how downstream libraries contribute task adapters, model
  components, inference backends, diagnostics, and recipes without adopting Unspool's
  internal implementations.
- **Release boundary:** a new user can reproduce a canonical analysis from the literature,
  understand why each step is present, replace one component, and obtain the same standard
  evidence objects.

## Deferred research extensions

Session-varying GLM-HMM research, population-of-laboratories inference, inverse RL,
agent-discovery models, novel trajectory-shape claims, and further Cell-paper analyses are
outside the 0.21–0.24 critical path. They may later demonstrate the package, but they will
not determine its basic architecture.
