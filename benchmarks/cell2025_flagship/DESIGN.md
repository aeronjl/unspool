# Frozen design: Cell 2025 behavioural flagship

Status: frozen before fitting<br>
Primary source: Liebana, Laffere et al., “Dopamine encodes deep network teaching
signals for individual learning trajectories,” *Cell* 188 (2025), 3789–3838,
[doi:10.1016/j.cell.2025.05.025](https://doi.org/10.1016/j.cell.2025.05.025)<br>
Data release: [doi:10.6084/m9.figshare.28877912.v1](https://doi.org/10.6084/m9.figshare.28877912.v1)<br>
Analysis release: [doi:10.6084/m9.figshare.28877942.v1](https://doi.org/10.6084/m9.figshare.28877942.v1)

This document freezes the scientific contract before any flagship models are fitted. It
separates reproduction of the published analysis from new estimands introduced by
Behavio. A successful reproduction does not validate a new forecast, and a successful
forecast does not make the paper's descriptive trajectory clusters prospective.

## Questions and claim boundaries

### Layer A: published-result reproduction

Layer A independently reproduces the behavioural analyses for which the public trial
table contains the required observables:

1. Figure 1G: correlation between mean bias on paper days 4–8 and mean bias over the
   final five paper sessions.
2. Figure 1I: correlation between mean bias on days 4–8 and the final-five-session
   right-minus-left psychometric-slope difference.
3. Accuracy across learning, including the first- versus last-session means.
4. Smoothed left-versus-right psychometric-slope trajectories and the released
   three-cluster visualization.
5. Response-time/chronometric summaries where the released table supplies response
   times.
6. The first-five-day single-state Q-value model comparison, including innate,
   day-specific, and reward-history components, when the released implementation can be
   matched exactly.

These are descriptive or retrospective analyses. The three trajectory clusters are a
visualization of continuous diversity, not three natural kinds, and are not used as
prospective labels.

### Layer B: new prospective estimand

The primary new estimand is the improvement in held-out future-choice log loss when a
model forecasts the final five sessions of a new animal after observing only its first
eight paper days, relative to simpler behavioural competitors. Completed trajectories
from other animals are treated as a historical reference cohort.

This is a **historical-cohort-calibrated individual forecast**. Its intended deployment
condition is: a previous cohort has completed training; a new animal has completed eight
days; its later behaviour has not yet occurred. It is not a same-cohort online forecast,
and it relies on exchangeability between reference and forecast animals.

The scientific comparisons are:

- Does the paper's early-bias summary improve final-five-session forecasts beyond a
  pooled psychometric curve?
- Do stable animal effects improve forecasts beyond complete pooling?
- Does a shared population trajectory improve forecasts beyond a stationary model?
- Do individual smooth trajectories improve forecasts beyond stable heterogeneity and
  shared drift?

No neural variables are used. Layer B therefore tests behavioural predictability, not a
dopaminergic mechanism or a causal teaching signal.

## Experimental units and source boundaries

- Sampling unit for uncertainty and fold assignment: animal.
- Observation scored by the likelihood: trial-level binary left/right choice.
- Repeated-measures unit: paper day nested within animal.
- Source session identifiers are retained for provenance.
- The modeling session is the derived `(animal, paper day)` unit used by the published
  behavioural analysis. The two DAP021 source sessions both marked paper day 1 are
  combined only in this derived layer; their source identifiers and trial numbers remain
  available.
- Animals are assigned to folds by sorted stable subject identifier, before any outcome
  is examined.
- No-go trials are excluded rather than modeled as a third response.

The public behaviour table does not contain every signal needed for the paper's complete
multimodal analysis. Video, pupil, wheel, lick, photometry, dopamine, and network-model
analyses are out of scope unless a separately checksum-pinned source is added. They must
not be described as reproduced by this benchmark.

## Trial exclusions and preprocessing

The released order of operations is part of the contract:

1. Remove no-go trials for the binary-choice analysis.
2. Compute response time and its z-score within `(animal, paper day)` using the no-go-
   filtered rows.
3. Exclude trials whose response-time z-score is not below 2.
4. Retain `repeatNumber == 1`.
5. Exclude shaped animals.
6. Retain expert/learner animals and then require observation from the first two paper
   days, matching the released Figure 1 analysis.
7. Map stimulus contrast to signed contrast (`right - left`) and retain stimulus side.

No learned transform may inspect a forecast animal's middle or final sessions. The
early-bias feature is calculated only from that animal's eligible training rows on paper
days 4–8. Scaling or basis choices that depend on data are fitted inside each fold.

## Frozen clocks, panel, and landmarks

The prospective panel contains 13 aligned coordinates per animal:

- coordinates 0–7: paper days 1–8, the observable context;
- coordinates 8–12: the animal's final five paper sessions, the forecast horizon.

Rows between paper day 8 and the final five sessions are neither fitted nor scored for a
forecast animal. For reference animals all 13 selected coordinates are available.
`paper_session_order` retains the source day number; `session_order` is the aligned
0–12 coordinate used by trajectory models.

The only outcome-derived landmark is early bias, frozen as mean zero-contrast rightward
choice minus 0.5 over paper days 4–8, using the paper's sparse-session carry-forward
rules. It is fitted independently inside each fold. The 70% learner threshold is part of
the released cohort definition and is not re-estimated as a new prospective landmark.

## Validation geometry

Thirty eligible animals are partitioned into six deterministic folds of five forecast
animals by round-robin assignment after sorting subject identifiers. In fold `k`:

- reference animals: the other 25 animals, with all 13 selected coordinates in training;
- forecast animals: their first eight coordinates in training as individual context;
- excluded for forecast animals: all unselected middle sessions;
- test: only their final five coordinates;
- unit of equal weighting and resampling: forecast animal.

The split object records reference subjects, forecast subjects, context orders, test
orders, and source-row indices. It rejects overlapping rows, incomplete aligned panels,
or test orders that do not strictly follow context orders. Prediction is prospective
under the declared historical-cohort ordering even though reference animals contribute
later aligned coordinates.

The primary score is mean forecast-animal log loss. Secondary scores are Brier score and
calibration summaries. Pairwise uncertainty uses a paired bootstrap over forecast
animals with a fixed seed and a two-sided 95% percentile interval. Trial-weighted scores
are reported only as diagnostics because they would give animals with more trials more
influence.

## Frozen candidate set

All primary candidates use binary right-choice likelihoods, separate left and right
contrast terms, no choice-history lags, and identical folds:

1. `pooled_psychometric`: stationary complete-pooling GLM.
2. `late_phase_psychometric`: pooled GLM plus a frozen late-phase indicator and its
   left/right contrast interactions. This controls for general learning between context
   and forecast sessions without using animal-specific early bias.
3. `early_bias_forecast`: the late-phase control plus the fold-fitted early-bias feature,
   its late-phase interaction, and its late-phase interactions with left/right contrast.
4. `static_partial_pooling`: hierarchical stationary GLM with animal effects.
5. `shared_smooth_trajectory`: smooth population GLM over aligned coordinate with fixed
   knots `(0, 3, 7, 9, 12)`.
6. `hierarchical_smooth_trajectory`: smooth population trajectory plus partially pooled
   animal trajectories with the same knots.

The frozen regularization values are `l2 = 0.02`, population-path `smoothness = 3.0`,
`subject_scale = 0.4`, and subject-path `subject_smoothness = 3.0`. These values are shared
with Behavio's preceding matched longitudinal benchmark and are not selected on the Cell
forecast outcomes.

### Design amendment after simulation falsification

Before final results were pinned, a three-repeat exact-design smoke recovery showed that
the initially proposed static early-bias interactions could not recover a world in which
days 4–8 truly determined late slope asymmetry (0/3 selections). The same coefficients
were being forced to describe both context and forecast rows, which did not represent the
declared estimand. The candidate set above was therefore amended to add the explicit
`late_phase_psychometric` control and restrict the predictive early-bias interactions to
the forecast phase. This amendment, its failed precursor, and the subsequent recovery are
retained as part of the design history; no final benchmark result preceded it.

Choice-history terms are excluded because combining DAP021's duplicated source day into
the published paper-day unit does not define a defensible cross-source-session trial
history. The benchmark may later add a source-session analysis as a separately named
estimand; it cannot silently alter this contract.

Behavio's current `BinaryQLearning` is not a reproduction of the paper's single-state
Q-value model: it resets values by session, uses one learning rate, and includes different
choice-history terms. It is therefore excluded from the primary candidate set. The exact
published first-five-day model comparison is reproduced separately in Layer A; any
generalized learning-agent competitor must be named as a new model and pass design-
matched recovery first.

## Recovery and falsification

Model recovery uses the exact 13-coordinate trial-count, contrast, subject, and split
geometry. Each structural candidate—pooled stationary, static partial pooling, shared
smooth drift, and hierarchical smooth drift—generates data and competes against the same
set. The outcome-derived `early_bias_forecast` is excluded from the primary recovery
matrix because its feature must be recomputed from simulated choices; it is instead
audited in a dedicated simulation that performs that recomputation.

Parameter recovery is required for the winning structural model on the same design. It
reports population-trajectory error and, where identifiable, animal-trajectory error,
with optimization failures and boundary estimates retained.

Competing-explanation tests must include at least:

- stationary but heterogeneous animals;
- shared learning drift without stable strategy differences;
- individual smooth drift;
- a reward-history process matched to the early learning design.

The reward-history world uses a symmetric two-action delta rule with one shared learning
rate and inverse temperature, action values carried across the retained panel, stochastic
reward at zero contrast, and task-contingent reward otherwise. It contains no stable
animal-specific preference. Its purpose is to measure false selection of the early-bias
summary under reinforcement history, not to reproduce the paper's fitted Q-value model.

A flagship conclusion is weakened or withheld if the design cannot recover its claimed
structure, if the winning model is numerically unreliable, or if a simpler competing
explanation is indistinguishable at the frozen sample size.

## Reproducibility contract

- Raw data stay out of Git and are verified against the published member SHA-256.
- Small derived tables, fold manifests, model/recovery summaries, and figures are
  committed with generation metadata and deterministic seeds.
- Released analysis versions are recorded; exact clustering is derived in an isolated
  environment matching NumPy 1.26.4, scikit-learn 1.5.1, and tslearn 0.6.3 rather than
  becoming a core runtime dependency.
- Fits must expose convergence and audit status; failed candidates are never silently
  dropped.
- Repository verification is `ruff check`, `ruff format --check`, `pytest`, `uv build`,
  and a strict documentation build.

## Frozen interpretation

Evidence that early behaviour predicts later choices supports forecastability under the
historical-cohort estimand. It does not show that early bias causes later strategy, that
trajectory clusters are discrete, that dopamine is necessary or sufficient, or that the
result generalizes to another task, laboratory, species, or training protocol. Those are
separate estimands requiring external cohorts or interventions.
