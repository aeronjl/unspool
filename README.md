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
> is not yet stable and the model catalogue currently contains static and smoothly time-
> varying Bernoulli GLMs, a fixed-scale hierarchical Bernoulli GLM, a fixed-transition
> GLM-HMM, and a compact binary Q-learning agent.

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
- every fitted model can be paired with a generative simulation;
- convergence failures, boundary estimates, and label ambiguity remain visible;
- recovery is reported for a particular design and sample size, not awarded as a
  universal certificate;
- individual trajectories remain inspectable when population information is pooled;
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
[clock and transform guide](docs/clocks-and-transforms.md), and
[validation guide](docs/validation.md).

Leave-subject-out and leave-lab-out folds train on complete disjoint population units.
Lab holdout rejects any subject assigned to more than one lab rather than permitting
cross-fold leakage.

Five reference models are executable: a static Bernoulli GLM, a smoothly time-varying
competitor with fixed temporal knots, a fixed-scale partial-pooling Bernoulli GLM, a fixed-
transition Bernoulli GLM-HMM, and a compact session-reset binary Q-learning agent. They
expose recursive simulation, fitting, filtered
prediction, pointwise scoring, numerical diagnostics, prospective fold evaluation, and
design-specific recovery through one common contract. Every fit also produces a normalized
audit without discarding its model-specific evidence. See the
[modelling guide](docs/modelling.md), [fit-audit guide](docs/diagnostics.md),
[smooth-drift guide](docs/smooth-drift.md),
[partial-pooling guide](docs/hierarchical-glm.md),
[GLM-HMM guide](docs/glm-hmm.md), [Q-learning guide](docs/q-learning.md), and
the [model-recovery guide](docs/model-recovery.md),
or run:

```bash
uv run python examples/static_glm.py
uv run python examples/smooth_glm.py
uv run python examples/model_recovery.py
uv run python examples/temporal_transforms.py
uv run python examples/within_session_validation.py
uv run python examples/glm_hmm.py
uv run python examples/q_learning.py
uv run python examples/population_validation.py
uv run python examples/hierarchical_glm.py
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

## Published-data benchmarks

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
uv run python examples/population_validation.py
uv run python examples/hierarchical_glm.py
uv run python -m benchmarks.recovery_grid.benchmark
uv run python -m benchmarks.weak_signal_recovery.benchmark
uv run python -m benchmarks.hierarchical_glm.benchmark
uv run python -m benchmarks.cell2025.fetch_data
uv run python -m benchmarks.cell2025.benchmark \
  benchmarks/cell2025/data/long_term_learning_dataset_preprocessed_behaviour_all.csv
uv run python -m benchmarks.ibl2021.fetch_data
uv run --with pyarrow python -m benchmarks.ibl2021.benchmark
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
