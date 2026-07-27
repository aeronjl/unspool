# Unspool

[![CI](https://github.com/aeronjl/unspool/actions/workflows/ci.yml/badge.svg)](https://github.com/aeronjl/unspool/actions/workflows/ci.yml)

> “No two moments are identical in a conscious being.”
> — Henri Bergson, *The Creative Mind*

**A process-first framework for fitting and falsifying behavioural models across learning.**

Unspool is an emerging Python library for trial-level behavioural modelling across
subjects and sessions. It will make simulation, parameter recovery, model recovery,
and time-aware validation part of the modelling interface rather than analyses added
after a model has been selected.

The project begins from a simple claim: **behaviour is not a sequence of independent
nows**. A trial inherits a history, changes the learner that encounters the next trial,
and sits within several non-equivalent clocks—trials, sessions, calendar time, task
stages, and inferred learning landmarks.

> [!IMPORTANT]
> Unspool is pre-alpha. Its longitudinal data, clock, fold-fitted transform, validation,
> first modelling, and parameter- and model-recovery contracts are executable, but the API
> is not yet stable. The model catalogue currently contains static and smoothly time-
> varying Bernoulli GLMs, static and smooth hierarchical Bernoulli GLMs, a fixed-transition
> GLM-HMM, a compact binary Q-learning agent, and a joint choice/response-time Wiener
> drift-diffusion family with stationary, smooth session-varying, or partially pooled
> animal-specific trajectories and an optional explicit contaminant mixture for the
> stationary model.

## Why “Unspool”?

Henri Bergson used two spools joined by a tape as an image of duration: one unwinds as
the future contracts, while the other gathers an accumulating past. He also warned that
the image can make time appear falsely homogeneous and reversible. That tension is
methodologically productive for longitudinal behavioural science. We need common
coordinates without pretending that learning histories are interchangeable.

The philosophical background is developed in [Philosophy of Unspool](docs/philosophy.md),
where Bergsonian duration, Heideggerian temporality, Husserlian retention and protention,
Simondonian individuation, and scientific underdetermination are connected to concrete
software commitments.

## Intended guarantees

Unspool is being designed so that:

- sequential data are not shuffled into invalid trial-wise folds by default;
- subject- and lab-held-out folds exclude complete population units;
- data-derived landmarks are learned inside training folds;
- unresolved landmark-uncertainty draws remain visible in relative-clock distributions;
- every fitted model can be paired with a generative simulation;
- convergence failures, boundary estimates, and label ambiguity remain visible;
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

## First executable contract

```python
from unspool import Study, forward_session_splits

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

Nine reference models are executable: a static Bernoulli GLM, a smoothly time-varying
competitor with fixed temporal knots, a static partial-pooling Bernoulli GLM, a partially
pooled smooth trajectory model, a fixed-transition Bernoulli GLM-HMM, and a compact
session-reset binary Q-learning agent, plus a fixed-parameter Wiener drift-diffusion model
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
[GLM-HMM guide](docs/glm-hmm.md), [Q-learning guide](docs/q-learning.md),
[drift-diffusion guide](docs/drift-diffusion.md),
[session-varying drift-diffusion guide](docs/smooth-ddm.md),
[hierarchical drift-diffusion guide](docs/hierarchical-smooth-ddm.md),
and the [model-recovery guide](docs/model-recovery.md),
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
prospective 6-versus-12-animal benchmark, joint scale RMSE falls from `0.09178` to
`0.05138`; all 16 fits converge and mean future-session log loss stays within `0.00233` of
an oracle. Local interval coverage is only 50–62.5%, so Unspool reports those intervals as
diagnostics rather than calibrated uncertainty. See the
[parameter-specific DDM scale benchmark](benchmarks/ddm_subject_scale_recovery/README.md).

An opt-in supplemented EM correction now accounts for uncertainty omitted by that local
scale curvature, while refusing unstable covariance estimates. Across 20 eight-animal
panels it returns 18 finite intervals: conditional coverage is 100% for drift scale and
88.9% for boundary scale, versus 70% and 65% locally. Empirical-Bayes integration over
random-effect paths improves mean joint log probability by `0.79135` across 80 entirely
unseen animals and wins for 70% of them, with Monte Carlo precision retained per animal.
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

The first external benchmark reproduces the central longitudinal-behaviour result from
Liebana, Laffere et al. (2025): bias during days 4–8 predicts the final-five-session
right-minus-left psychometric-slope asymmetry across 30 mice (`r = 0.69479`,
`p = 2.04e-05`). The workflow fetches only the required member of the versioned public
Figshare archive, verifies its checksum, maps trials to `Study`, and enforces a numerical
regression contract. See the [Cell 2025 benchmark](benchmarks/cell2025/README.md).

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

The first end-to-end [prospective longitudinal study](benchmarks/flagship_longitudinal/README.md)
then aligns six sessions per animal in both public sources and forecasts the sixth from the
first five. Complete pooling, static partial pooling, shared smooth drift, and hierarchical
smooth trajectories compete under one cohort-level fold, with subject-balanced scoring,
paired subject-bootstrap intervals, fit audits, and individual paths retained. Static
partial pooling has the lowest point estimate in the 30-mouse Cell panel; shared drift has
the lowest in the nine-mouse IBL panel, where uncertainty leaves the structural ranking
unresolved.

## Development

Unspool requires Python 3.11 or newer. The development interpreter is pinned to Python
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
uv run python -m benchmarks.ibl2021.fetch_data
uv run --with pyarrow python -m benchmarks.ibl2021.benchmark
uv run --extra ibl python -m benchmarks.ibl2021_replicated.benchmark
uv run --extra ibl python -m benchmarks.ibl2021_prospective.benchmark
uv run --with pyarrow python -m benchmarks.flagship_longitudinal.benchmark
uv run ruff check .
uv run ruff format --check .
uv build
```

To apply formatting locally:

```bash
uv run ruff format .
```

The same non-mutating checks run in continuous integration. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the development contract.

## Intellectual and software provenance

Unspool grows from analyses associated with Liebana, Laffere et al. (2025), subsequent
IBL modelling work, and earlier exploratory repositories. Reusable ideas will be
reimplemented behind a coherent public API with tests, attribution, and explicit
scientific boundaries. See [provenance](docs/provenance.md).

## License

Unspool is licensed under the [MIT License](LICENSE).
