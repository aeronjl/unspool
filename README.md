# Behavio

[![CI](https://github.com/aeronjl/behavio/actions/workflows/ci.yml/badge.svg)](https://github.com/aeronjl/behavio/actions/workflows/ci.yml)
[![Documentation](https://github.com/aeronjl/behavio/actions/workflows/docs.yml/badge.svg)](https://github.com/aeronjl/behavio/actions/workflows/docs.yml)

**[Read the documentation](https://aeronjl.github.io/behavio/)** to choose a
longitudinal workflow, browse supported and experimental methods, and run worked
Cell, Chen restless-bandit, and IBL studies.

New analyses should start with **[Choose a model by the claim](https://aeronjl.github.io/behavio/model-choice-guide/)**
and the common-format **[model cards](https://aeronjl.github.io/behavio/model-cards/)**,
then follow the **[literature-recipe standard](https://aeronjl.github.io/behavio/tutorials/recipe-contract/)**.

> “No two moments are identical in a conscious being.”
> — Henri Bergson, *The Creative Mind*

**Fit and falsify trial-level models of behaviour — psychometric curves, GLM-HMMs,
drift-diffusion models of choice and response time, reinforcement-learning agents, and
hierarchical learning trajectories — with simulation, parameter and model recovery, and
time-aware validation built into the fitting interface.**

Behavio is an emerging Python library for trial-level behavioural modelling within and
across sessions and subjects. It will make simulation, parameter recovery, model recovery,
and time-aware validation part of the modelling interface rather than analyses added
after a model has been selected.

The project begins from a simple claim: **behaviour is not a sequence of independent
nows**. A trial inherits a history, changes the learner that encounters the next trial,
and sits within several non-equivalent clocks—trials, sessions, calendar time, task
stages, and inferred learning landmarks.

> [!IMPORTANT]
> Behavio is pre-alpha. Its longitudinal data, clock, fold-fitted transform, validation,
> first modelling, and parameter- and model-recovery contracts are executable, but the API
> is not yet stable. The model catalogue currently contains static and smoothly time-
> varying Bernoulli GLMs, static and smooth hierarchical Bernoulli GLMs, a fixed-transition
> GLM-HMM, compact and composable binary reinforcement-learning agents, and a joint
> choice/response-time Wiener
> drift-diffusion family with stationary, smooth session-varying, or partially pooled
> animal-specific trajectories and an optional explicit contaminant mixture for the
> stationary model.

## Why “Behavio”?

`behavio` is the exact stem that “behaviour” and “behavior” share, so the name takes no
side on spelling. It also yields the alias most people will actually type:

```python
import behavio as behav
```

The name is deliberately unmetaphorical. The package was previously called `unspool`,
after a Bergsonian image of duration, which argued for the package in longitudinal
analyses and against it in every other one — single-session psychometrics, within-session
choice and response time, ethograms, foraging. The commitments that name stood for are
unchanged: several non-equivalent clocks, ordering that is never shuffled away, and
learning histories that are not silently made interchangeable by alignment. They are
developed in [Philosophy of Behavio](docs/philosophy.md), alongside Heideggerian
temporality, Husserlian retention and protention, Simondonian individuation, and
scientific underdetermination.

## Intended guarantees

Behavio is being designed so that:

- sequential data are not shuffled into invalid trial-wise folds by default;
- subject- and lab-held-out folds exclude complete population units;
- data-derived landmarks are learned inside training folds;
- unresolved landmark-uncertainty draws remain visible in relative-clock distributions;
- every fitted model can be paired with a generative simulation;
- convergence failures, boundary estimates, and label ambiguity remain visible;
- the declared proper score controls selection, comparison, uncertainty, and ranking;
- recovery is reported for a particular design and sample size, not awarded as a
  universal certificate;
- individual trajectories remain inspectable when population information is pooled;
- cross-lab trajectory comparisons require independent animals within every lab and keep
  level, amplitude, and scale-free shape distinct;
- external tables retain explicit identity, chronology, source semantics, and provenance;
- discrete states must compete against smooth drift, learning, history, and observable
  behavioural alternatives.

See the [scientific scope](docs/scientific-scope.md) and [roadmap](docs/roadmap.md) for
the proposed first release.

## A task before a model

Behavio separates longitudinal identity from task semantics. `Study` records which trial
occurred when; `TaskSpec` declares choices, omissions, available actions, rewards, response
times, predictors, blocks, and episodes before any model is fitted.

```python
from behavio import BernoulliHistoryGLM, ChoiceSpec, TaskSpec, fit_model

task = TaskSpec(
    choice=ChoiceSpec(options=(0, 1)),
    predictors=("stimulus",),
)
model = BernoulliHistoryGLM(covariates=("stimulus",), choice_lags=1)
fitted = fit_model(model, study, task=task)

print(fitted.validation.n_trials)
print(fitted.result.parameters)
print(fitted.audit().status)
```

This is the first 0.21 golden path. Prospective claims still belong in fold-aware
evaluation or a frozen study protocol. Common numeric, categorical, interaction, and
explicit-reset history terms are available through `DesignSpec`. Read the
[task contract](docs/task-contract.md) and [fixed design matrices](docs/design-matrices.md).
Portable `FitArtifact` records then bind a fit to task, complete-data identity, package
version, labelled parameters, and numerical audits without serializing executable model
objects; external packages can contribute conforming factories through an explicit local
`EstimatorRegistry`.

## Reproducible study protocols

Complete analyses can now be frozen as typed, immutable `StudyProtocol` declarations.
A protocol fixes source provenance, outcome-blind cohort rules, units, clocks, estimands,
training-only transforms, deployment geometry, candidates, uncertainty, recovery gates,
limitations, and prohibited claims before fitting. The compiler then materializes exact
denominators and audits fit, prediction-context, scored, and excluded rows before the
common runner can execute.

```python
materialized = materialize_protocol(protocol.freeze(), source_study)
compiled = compile_execution_plan(
    materialized,
    splits,
    capabilities=capabilities,
)
evaluation = run_protocol(compiled, models)
```

The final reported analysis can be written as a deterministic, content-addressed evidence
bundle containing the protocol and amendment history, source and cohort identities,
execution plan, predictions, numerical audits, comparisons, recovery, figures, and
bounded report—without pickled model objects or redistributed raw trials.

Read the [study-protocol workflow](docs/protocols/index.md), or inspect the command line:

```bash
uv run behavio --help
```

## First executable contract

```python
from behavio import Study, forward_session_splits

study = Study(
    {
        "subject": ["mouse-1", "mouse-1", "mouse-1"],
        "session": ["day-1", "day-1", "day-2"],
        "trial": [0, 1, 0],
        "session_order": [0, 0, 1],
        "choice": [1, 0, 1],
    }
)

split = forward_session_splits(study)[0]
train = study.take(split.train_indices)
test = study.take(split.test_indices)
```

The required schema makes chronology explicit while preserving arbitrary source fields.
Forward-session folds train only on earlier complete sessions. Within-session rolling
origins carry the observed pre-origin prefix into filtered prediction while scoring only
future trials. A separate leave-one-session-out splitter is intentionally marked non-
prospective because it can train on the held-out session's future. Typed session,
cumulative-trial, elapsed-time, task-phase, and landmark-relative clocks prevent unlike
temporal coordinates from becoming anonymous columns. Behavioural landmarks are fitted
independently inside each training fold with immutable provenance. See the
[data contract](docs/data-contract.md),
[NWB/DANDI interoperability guide](docs/interoperability.md),
[clock and transform guide](docs/clocks-and-transforms.md), and
[validation guide](docs/validation.md).

Leave-subject-out and leave-lab-out folds train on complete disjoint population units.
Lab holdout rejects any subject assigned to more than one lab rather than permitting
cross-fold leakage.

The executable catalogue begins with named bias-only, psychometric, lapse-psychometric,
perseveration, and win-stay/lose-shift baselines. It also includes a static Bernoulli GLM,
a smoothly time-varying
competitor with fixed temporal knots, a static partial-pooling Bernoulli GLM, a partially
pooled smooth trajectory model, a fixed-transition Bernoulli GLM-HMM, and a compact
session-reset binary Q-learning agent plus a composable successor with asymmetric
learning, forgetting, choice kernels, lapse-softmax policies, and explicit reset columns,
plus a fixed-parameter Wiener drift-diffusion model
that jointly scores choice and response time and can include a fixed-support contaminant
component, and a smooth longitudinal Wiener model for drift, boundary, and starting-bias
paths, plus a hierarchical Wiener model with shrunken animal-specific trajectories. They
expose recursive simulation, fitting,
filtered prediction, pointwise scoring, numerical diagnostics, prospective fold evaluation,
and design-specific recovery through one common contract. Every fit also produces a
normalized audit without discarding its model-specific evidence. See the
[modelling guide](docs/modelling.md), [fit-audit guide](docs/diagnostics.md),
[estimator and plugin contract](docs/estimator-contract.md),
[prospective comparison guide](docs/comparison.md),
[cross-lab trajectory-shape guide](docs/trajectory-shapes.md),
[smooth-drift guide](docs/smooth-drift.md),
[partial-pooling guide](docs/hierarchical-glm.md),
[partially pooled trajectory guide](docs/hierarchical-smooth-glm.md),
[GLM-HMM guide](docs/glm-hmm.md), [reinforcement-learning guide](docs/q-learning.md),
[drift-diffusion guide](docs/drift-diffusion.md),
[session-varying drift-diffusion guide](docs/smooth-ddm.md),
[hierarchical drift-diffusion guide](docs/hierarchical-smooth-ddm.md),
and the [model-recovery guide](docs/model-recovery.md),
plus the [canonical baseline guide](docs/baselines.md),
or run:

```bash
uv run python examples/static_glm.py
uv run python examples/smooth_glm.py
uv run python examples/model_recovery.py
uv run python examples/temporal_transforms.py
uv run python examples/within_session_validation.py
uv run python examples/glm_hmm.py
uv run python examples/q_learning.py
uv run python examples/drift_diffusion.py
uv run python examples/smooth_drift_diffusion.py
uv run python examples/hierarchical_smooth_drift_diffusion.py
uv run python examples/contaminant_ddm.py
uv run python examples/population_validation.py
uv run python examples/hierarchical_glm.py
uv run python examples/hierarchical_smooth_glm.py
uv run python examples/prospective_comparison.py
```

## Synthetic recovery benchmark

The first matched recovery grid makes all four reference families compete under the same
prospective folds. A nested 150-trial design recovers two of four generating families,
while its 300-trial counterpart recovers all four for the exact fixed parameter regimes.
Fit-audit warnings remain visible and unresolved outcomes remain part of every confusion
matrix. See the [four-family recovery benchmark](benchmarks/recovery_grid/README.md).

A repeated follow-up holds the 300-trial design fixed while moving each family toward a
limiting case in which competitors can imitate it. Recovery falls from 70.0% across 40
stronger-reference runs to 32.5% across 40 boundary-near runs, with scenario-level
confusion and Wilson intervals retained. See the
[weak-signal recovery benchmark](benchmarks/weak_signal_recovery/README.md).

The first population benchmark compares complete pooling, independent subject fits, and
fixed-scale partial pooling over low, moderate, and high subject heterogeneity. Across 20
matched repetitions per regime, partial pooling has the lowest individual-coefficient RMSE
and future-session log loss in all three. The scale is supplied from the generator, so this
validates shrinkage mechanics rather than variance-component estimation. See the
[hierarchical GLM benchmark](benchmarks/hierarchical_glm/README.md).

A follow-up estimates the shared subject scale from a neutral starting value using bounded
Laplace marginal likelihood. Across 120 fits, all optimizations converge; moving from 8 to
24 subjects reduces scale RMSE at every tested heterogeneity level, while approximate 95%
interval coverage ranges from 95% to 100%. Weak heterogeneity still reaches the lower bound
in 40% of 8-subject runs, preserving an important resolution limit. See the
[subject-scale recovery benchmark](benchmarks/subject_scale_recovery/README.md).

The factorial trajectory benchmark then makes five models compete across stationary
identical animals, stable individual differences, shared drift, and individual drift. The
scientifically matched account wins every regime under both realized subject-trajectory
RMSE and held-out final-session log loss; the hierarchical smooth model wins only when
animals genuinely change differently. See the
[trajectory-recovery benchmark](benchmarks/trajectory_recovery/README.md).

The cross-lab trajectory benchmark separates a constant level shift, doubled centered
amplitude, and a genuinely different path across four replicated synthetic labs. All 20
matched repetitions recover the generating component structure, while a nine-lab
singleton design is rejected as inferentially unready. See the
[trajectory-shape benchmark](benchmarks/trajectory_shapes/README.md).

The longitudinal Wiener hierarchy now supports named drift and boundary heterogeneity and
can estimate each scale inside the training fold with bounded Laplace-EM updates. In a
prospective 6-versus-12-animal benchmark, joint scale RMSE falls from `0.06144` to
`0.04806`; all 16 fits converge and mean future-session log loss stays within `0.00081` of
an oracle. Scale intervals now default to the Louis observed-information correction rather
than the complete-data curvature, and coverage across the four parameter-by-cohort cells
is 87.5–100% against a nominal 95%. Those intervals are conservative, not exact: their
standard errors run up to `2.07x` the Monte Carlo sampling spread of the estimates
themselves. The drift-scale point estimate also remains biased low by 21–28%, an EM/Laplace
shrinkage the wider interval partly absorbs rather than removes. See the
[parameter-specific DDM scale benchmark](benchmarks/ddm_subject_scale_recovery/README.md).

An opt-in supplemented EM correction accounts for the same omitted uncertainty by a
different route, while refusing unstable covariance estimates. Across 20 eight-animal
panels it now returns 20 finite intervals: conditional coverage is 100% for both drift and
boundary scale, versus 50% and 85% under the uncorrected local curvature. Empirical-Bayes
integration over random-effect paths improves mean joint log probability by `0.98299`
across 80 entirely unseen animals and wins for 68.75% of them, with Monte Carlo precision
retained per animal.
See the [DDM predictive-uncertainty benchmark](benchmarks/ddm_predictive_uncertainty/README.md).

A latent-state recovery benchmark separates arbitrary HMM label names from recovered state
identity. Across 20 clear-state and 20 overlapping-state fits, aligned metrics are invariant
to complete label reversal. Clear states reach 91.85% decoded accuracy with no ambiguous
assignments; overlapping states fall to 56.53%, with 35% explicitly ambiguous. See the
[state-alignment benchmark](benchmarks/state_alignment/README.md).

The first landmark-uncertainty benchmark compares decisive and marginal learning under one
fixed threshold definition. Decisive transitions resolve in every point estimate and
bootstrap draw, whereas marginal learning resolves in 83.33% of datasets and 82.05% of
draws; its conditional intervals are more than three times wider. See the
[landmark-uncertainty benchmark](benchmarks/landmark_uncertainty/README.md).

The nested prospective-selection benchmark validates the complete selection procedure,
not only its component models. Across 20 stationary datasets it chooses the static model
in 37/40 outer folds; across 20 strong shared-drift datasets it chooses the smooth model in
40/40. All selected outer fits pass audit, and the remaining stationary errors stay visible
as a design-specific resolution limit. See the
[nested selection benchmark](benchmarks/nested_selection/README.md).

The first joint choice/response-time benchmark fits a fixed-parameter Wiener
drift-diffusion model across 20 repetitions at both 400 and 1,200 trials. All 40 fits pass
audit, and increasing the design size reduces RMSE for drift intercept, stimulus drift,
boundary, starting bias, and non-decision time. See the
[drift-diffusion recovery benchmark](benchmarks/ddm_recovery/README.md).

A paired contaminant benchmark then injects five-percent independent uniform response-time
contamination into 20 five-session designs. The explicit-mixture Wiener fit lowers RMSE for
every shared parameter and beats the naive fit on held-out fifth-session joint log loss in
all 20 repetitions. Posterior trial responsibilities remain soft and the support is fixed
before fitting. See the
[contaminant-aware DDM benchmark](benchmarks/ddm_contaminants/README.md).

A longitudinal Wiener benchmark then compares static and smooth fits under stationary and
changing truth. Across 20 matched repetitions in each regime, the static model has lower
training-path RMSE and final-session joint log loss under stationarity; the smooth model
wins both metrics under the specified changing drift and boundary paths, including all 20
future-session comparisons. See the
[session-varying DDM benchmark](benchmarks/smooth_ddm/README.md).

The hierarchical Wiener benchmark then makes complete pooling, shared smooth, independent
smooth, and partially pooled smooth trajectories compete under stationary identical
animals, shared change, and individual change. Across 20 repetitions per regime, the
scientifically matched structure wins both subject-path RMSE and held-out fifth-session
joint log loss; all 480 fits converge. See the
[hierarchical DDM benchmark](benchmarks/hierarchical_smooth_ddm/README.md).

## Published-data benchmarks

An interoperability benchmark streams a version-pinned 72.6 MB NWB asset from published
Dandiset `000004` without downloading it wholesale. All 200 source trial IDs, row order,
time intervals, task phases, stimulus categories, field semantics, and trial-addressable
asset provenance satisfy the canonical contract. See the
[NWB/DANDI interoperability benchmark](benchmarks/nwb_dandi_interoperability/README.md).

Four papers are held to the published-parity contract: Liebana, Laffere et al. (2025),
International Brain Laboratory et al. (2021), Ashwood et al. (2022), and Chen et al.
(2021). Each declares the values printed in its paper, the tolerance those values are
checked against, and the tolerance's justification, in a `published_claims.json` that
[`tests/test_published_parity.py`](tests/test_published_parity.py) revalidates offline.
Claims that do not reproduce are recorded as `failed-parity` and retained rather than
tuned away: the IBL cohort size and six of Ashwood's model-derived quantities currently
fail, each for a documented reason.

One of those benchmarks reproduces two central longitudinal-behaviour panels from
Liebana, Laffere et al. (2025): bias during days 4–8 is negatively associated with bias in
the final-five-paper-day window (`r = -0.52764`) and positively associated with its
right-minus-left psychometric-slope asymmetry across 30 mice (`r = 0.69479`,
`p = 2.04e-05`). The workflow fetches only the required member of the versioned public
Figshare archive, verifies its checksum, maps trials to `Study`, and enforces a numerical
regression contract. Both correlations are additionally checked against the values printed
in the paper (`-0.53` and `0.69`) by
[`published_claims.json`](benchmarks/cell2025/published_claims.json), which declares the
two-decimal rounding tolerance and is revalidated offline by
[`tests/test_published_parity.py`](tests/test_published_parity.py). See the
[Cell 2025 benchmark](benchmarks/cell2025/README.md).

The [Cell behavioural flagship](benchmarks/cell2025_flagship/README.md) retains that
reproduction and freezes a separate historical-cohort forecast: 25 completed reference
animals plus days 1–8 from five forecast animals predict those animals' final five
sessions in each of six folds. Early bias has the lowest animal-balanced log loss and
clearly improves on complete pooling (`+0.04219`, 95% interval `+0.01818` to `+0.06425`),
while its advantage over a late-phase control and hierarchical smooth trajectories remains
unresolved. Exact-design recovery, released trajectory/Q-value compatibility, response-
time summaries, numerical audits, and the unresolved comparisons remain committed as one
paper-style evidence artifact.

The second benchmark maps 28,400 public IBL trials into the same `Study` contract. A
trial-outcome-blind rule selects one transition-anchored trajectory from each of nine labs,
and the benchmark compares three disjoint early sessions with the final three training
sessions before the first biased-task transition. All nine selected animals improve on
easy trials (`+0.42943` mean accuracy), providing a checksum-pinned positive control for
public-data retrieval, session chronology, and landmark-relative phase construction. See
the [IBL 2021 benchmark](benchmarks/ibl2021/README.md). Its population-validation contract
also confirms complete trial coverage for nine subject and nine lab holdouts while making
their one-subject-per-lab equivalence explicit.

The replicated successor uses the general optional ONE adapter to address that confounding
directly. Its outcome-blind manifest retains all 78 eligible animals across the same nine
labs (at least four per lab), pins 468 trial-table UUIDs and checksums, and maps 260,833
trials into trial-addressable source provenance. All 78 animals improve descriptively from
their first-three to final-three pre-transition training windows (`+0.42281` subject-weighted
mean easy-trial accuracy). This is a strong retrieval and chronology positive control, not
an unbiased learning estimate: selection is conditioned on the protocol transition, and
the six endpoint positions are not uniform elapsed time. See the
[replicated IBL 2021 benchmark](benchmarks/ibl2021_replicated/README.md).

Its prospective successor then compares static partial pooling with hierarchical smooth
drift under two leakage-safe boundaries. When positions 0–4 predict position 5 for the same
78 animals, drift lowers subject-balanced log loss from `0.6400` to `0.5549` (paired
improvement `+0.0851`, 95% interval `+0.0162` to `+0.1460`). When both the future session
and an entire lab are held out, drift retains the lower point estimate but the interval
spans zero; the represented sites do not resolve a transport advantage. See the
[replicated IBL prospective comparison](benchmarks/ibl2021_prospective/README.md).

The nested successor moves model structure and smoothness selection entirely inside those
training boundaries. A four-candidate grid selects the smoothness-9 drifting model in the
represented-animal fold and in all nine held-out-lab folds. On untouched position-5 data,
the selected procedure lowers subject-balanced log loss by `0.00768` relative to the fixed
smoothness-3 drift model for represented animals and by `0.00777` under lab transfer; both
paired intervals exclude zero. Smoothness 9 is the declared grid boundary, so this is
evidence for stronger regularization under this design, not an estimate of an optimal
continuous smoothness. See the
[replicated IBL nested-selection benchmark](benchmarks/ibl2021_nested_selection/README.md).

The first end-to-end [prospective longitudinal study](benchmarks/flagship_longitudinal/README.md)
then aligns six sessions per animal in both public sources and forecasts the sixth from the
first five. Complete pooling, static partial pooling, shared smooth drift, and hierarchical
smooth trajectories compete under one cohort-level fold, with subject-balanced scoring,
paired subject-bootstrap intervals, fit audits, and individual paths retained. Static
partial pooling has the lowest point estimate in the 30-mouse Cell panel; shared drift has
the lowest in the nine-mouse IBL panel, where uncertainty leaves the structural ranking
unresolved.

## Development

Behavio requires Python 3.11 or newer. The development interpreter is pinned to Python
3.12 and dependencies are managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync --locked --all-groups
uv run pytest
uv run python examples/static_glm.py
uv run python examples/smooth_glm.py
uv run python examples/model_recovery.py
uv run python examples/temporal_transforms.py
uv run python examples/within_session_validation.py
uv run python examples/glm_hmm.py
uv run python examples/q_learning.py
uv run python examples/drift_diffusion.py
uv run python examples/smooth_drift_diffusion.py
uv run python examples/hierarchical_smooth_drift_diffusion.py
uv run python examples/contaminant_ddm.py
uv run python examples/population_validation.py
uv run python examples/hierarchical_glm.py
uv run python examples/hierarchical_smooth_glm.py
uv run python examples/prospective_comparison.py
uv run python -m benchmarks.recovery_grid.benchmark
uv run python -m benchmarks.weak_signal_recovery.benchmark
uv run python -m benchmarks.hierarchical_glm.benchmark
uv run python -m benchmarks.subject_scale_recovery.benchmark
uv run python -m benchmarks.trajectory_recovery.benchmark
uv run python -m benchmarks.state_alignment.benchmark
uv run python -m benchmarks.landmark_uncertainty.benchmark
uv run python -m benchmarks.nested_selection.benchmark
uv run python -m benchmarks.ddm_recovery.benchmark
uv run python -m benchmarks.ddm_contaminants.benchmark
uv run python -m benchmarks.smooth_ddm.benchmark
uv run python -m benchmarks.hierarchical_smooth_ddm.benchmark
uv run python -m benchmarks.ddm_predictive_uncertainty.benchmark
uv run --extra dandi python -m benchmarks.nwb_dandi_interoperability.benchmark
uv run python -m benchmarks.cell2025.fetch_data
uv run python -m benchmarks.cell2025.benchmark \
  benchmarks/cell2025/data/long_term_learning_dataset_preprocessed_behaviour_all.csv
uv run python -m benchmarks.cell2025_flagship.fetch_released_artifacts
uv run python -m benchmarks.cell2025_flagship.benchmark
uv run python -m benchmarks.ibl2021.fetch_data
uv run --with pyarrow python -m benchmarks.ibl2021.benchmark
uv run --extra ibl python -m benchmarks.ibl2021_replicated.benchmark
uv run --extra ibl python -m benchmarks.ibl2021_prospective.benchmark
uv run --extra ibl python -m benchmarks.ibl2021_nested_selection.benchmark
uv run --with pyarrow python -m benchmarks.flagship_longitudinal.benchmark
uv run ruff check .
uv run ruff format --check .
uv build
```

Build or preview the scientist-facing documentation with:

```bash
uv sync --group docs --locked
uv run --group docs mkdocs build --strict
uv run --group docs mkdocs serve
```

Regenerate the versioned documentation figures with:

```bash
uv run --group docs python -m scripts.plot_documentation_figures
```

Use `--skip-cell` when the checksum-pinned Cell source table has not been downloaded yet.
The [figure provenance register](docs/reference/figure-provenance.md) distinguishes
conceptual diagrams from displays generated from public data or committed benchmarks.

To apply formatting locally:

```bash
uv run ruff format .
```

The same non-mutating checks run in continuous integration. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the development contract.

## Intellectual and software provenance

Behavio grows from analyses associated with Liebana, Laffere et al. (2025), subsequent
IBL modelling work, and earlier exploratory repositories. Reusable ideas will be
reimplemented behind a coherent public API with tests, attribution, and explicit
scientific boundaries. See [provenance](docs/provenance.md).

## License

Behavio is licensed under the [MIT License](LICENSE).
