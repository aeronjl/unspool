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
> Unspool is pre-alpha. Its longitudinal data, validation, first modelling, and parameter-
> recovery contracts are executable, but the API is not yet stable and only one static
> reference model is implemented.

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
Forward-session folds train only on earlier complete sessions. A separate leave-one-session-
out splitter is intentionally marked non-prospective because it can train on the held-out
session's future. See the [data contract](docs/data-contract.md) and
[validation guide](docs/validation.md).

The first reference model is also executable: a static Bernoulli GLM with recursively
generated, session-bounded choice history. It exposes simulation, fitting, filtered
prediction, pointwise scoring, numerical diagnostics, prospective fold evaluation, and
design-specific parameter recovery through one common contract. See the
[modelling guide](docs/modelling.md) or run:

```bash
uv run python examples/static_glm.py
```

## Development

Unspool requires Python 3.11 or newer. The development interpreter is pinned to Python
3.12 and dependencies are managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync --locked --all-groups
uv run pytest
uv run python examples/static_glm.py
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
