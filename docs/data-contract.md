# The longitudinal study contract

`Study` is Behavio's canonical, format-independent representation of observed trials. It
is intentionally smaller than a dataframe standard: source columns remain available, but
the fields needed to reason safely about longitudinal order have precise meanings.

## Required columns

| Column | Contract |
| --- | --- |
| `subject` | Non-missing, hashable identifier. Stable for one subject throughout the study. |
| `session` | Non-missing, hashable identifier. Stable within a subject; it need not be globally unique. |
| `trial` | Non-negative integer position within a subject/session. Gaps are allowed, duplicates are not. |
| `session_order` | Non-negative integer chronology within a subject. It must map one-to-one to sessions within that subject. |

The composite `(subject, session, trial)` key must be unique. `session_order` may have gaps
and does not claim that order 3 for one animal is commensurate with order 3 for another.
It exists so that chronology is explicit rather than inferred from filenames, session
labels, or the accidental order of input rows.

All additional one-dimensional columns are copied and retained. They may contain choices,
rewards, reaction times, stimulus values, laboratory identifiers, calendar timestamps,
task versions, or source-specific metadata. Behavio does not silently rename, aggregate,
impute, or sort them.

`Study` itself declares nothing about what those columns may contain. Two contracts do,
and both are enforced rather than described: the
[behavioural task contract](task-contract.md) validates observed choices, omissions, and
availability, and a [study protocol](protocols/index.md) validates each declared
`ObservationSpec` — its measurement type and its permitted values — against the
materialized cohort. Missing values remain representable under both: `ChoiceSpec` retains
them as omissions when `missing_is_omission` is set, and `ObservationSpec` permits them
unless a declared `allowed_values` set omits `None`.

## Construction

```python
from behavio import Study

study = Study.from_columns(
    {
        "subject": ["mouse-1", "mouse-1", "mouse-1"],
        "session": ["day-2", "day-1", "day-1"],
        "trial": [0, 1, 0],
        "session_order": [1, 0, 0],
        "choice": [1, 0, 1],
        "reward": [1, 0, 1],
    }
)

# Source order is preserved.
assert study["session"].tolist() == ["day-2", "day-1", "day-1"]

# Chronological order is requested rather than imposed.
assert study.chronological_indices().tolist() == [2, 1, 0]
```

Inputs are copied and exposed as read-only NumPy arrays. Subsets created with
`study.take(indices)` pass through the same validation contract.

A trial table on disk is read with `read_table()`, which needs no optional dependencies for
CSV or TSV:

```python
from behavio.adapters.table import read_table

study = read_table("trials.csv")
```

It is a reader rather than a `Study` constructor on purpose: `behavio.study` is the leaf
module every other module imports, and file formats, optional readers, and chronology
derivations do not belong inside the contract they produce. What the reader will and will
not infer -- above all that it never invents `session_order` -- is documented in
[Tabular, NWB, and DANDI interoperability](interoperability.md).

Pandas-like objects can use `Study.from_dataframe()`. Its index is deliberately ignored;
the required identity and chronology must remain explicit columns. Source column names
can be mapped without first mutating the dataframe:

```python
study = Study.from_dataframe(
    trials,
    subject="mouse",
    session="session_id",
    trial="trial_index",
    session_order="training_day",
)
```

Mapped columns are renamed in their source position and all other columns are retained.
Their task roles can then be declared with the [behavioural task contract](task-contract.md).
Local NWB round trips and version-pinned DANDI streaming use the same study contract and
are documented in
[Tabular, NWB, and DANDI interoperability](interoperability.md).

## Signed zeros are canonicalised on ingest

A study replaces every `-0.0` with `+0.0` as columns are copied in. IEEE 754 keeps two
zeros and they compare equal, so nothing downstream of a study — no fit, score, prediction
or design matrix — can tell them apart.

Provenance can. Behavio hashes a study's values into
[`FitArtifact`](fit-artifacts.md) content addresses through `json.dumps`, which writes
`-0.0` and `0.0` as different text, so the *same* column used to land on two different
digests depending on which arithmetically equivalent expression produced it: `np.einsum`
normalises the sign of a zero and elementwise `*` does not, `0.0 * -1` is `-0.0` and
`0.0 * 1` is not. Three separate defects of exactly that shape were fixed one at a time
before the rule was moved here. A column's content address must depend on the column's
values and on nothing else.

Only the sign of a zero is touched. Negative subnormals, negative infinity, `NaN` and every
other value survive bit for bit, and the rule applies to float, complex and object columns
alike. It is an *ingest* boundary and only that: intermediate arrays such as a
`FeatureBlock` or a design matrix are not canonicalised, because they are not hashed, so
`InteractionTerm` still multiplies exactly as IEEE says it should and its `-0.0` becomes
`0.0` only if and when it is stored in a study.

```python
import numpy as np
from behavio import Study
from behavio.interchange import _study_record

identity = {"subject": ["m", "m"], "session": ["d", "d"], "trial": [0, 1], "session_order": [0, 0]}
stimulus = np.asarray([-2.0, 0.0])
gate = np.asarray([0.0, -1.0])

# Two equivalent constructions disagree on the sign of a zero...
assert np.array_equal(stimulus * gate, np.einsum("i,i->i", stimulus, gate))
assert np.signbit(stimulus * gate).any()
assert not np.signbit(np.einsum("i,i->i", stimulus, gate)).any()

# ...but the two studies agree on one content address.
multiplied = Study({**identity, "interaction": stimulus * gate})
contracted = Study({**identity, "interaction": np.einsum("i,i->i", stimulus, gate)})
assert not np.signbit(multiplied["interaction"]).any()
assert _study_record(multiplied)["sha256"] == _study_record(contracted)["sha256"]
```

## Planned designs

A simulation, a recovery study, or a worked example does not start from a file. It starts
from a *planned* design: every subject runs every session, and every session runs the same
number of trials. `Study.factorial()` builds that crossed grid, so the three nested
comprehensions it replaces cannot get `session_order` subtly wrong:

```python
design = Study.factorial(
    trials=120,
    subjects=("mouse-a", "mouse-b", "mouse-c"),
    sessions=5,
    columns={"stimulus": lambda rng, n_rows: rng.normal(size=n_rows)},
    seed=2025,
)
```

Rows are emitted subject-major, then in session order, then in trial order, so
`chronological_indices()` returns `0, 1, 2, ...`. `session_order` is the zero-based
position of a session within its subject, which is exactly the pair of invariants above:
constant inside a `(subject, session)` pair, and one-to-one within a subject.

`subjects` and `sessions` are a count, a single label, or an explicit sequence of labels.
When each subject's sessions carry the subject's own name, `session_label(subject, order)`
supplies them; its results must stay unique within a subject.

`columns` adds per-trial columns and accepts three kinds of value:

| Value | Meaning |
| --- | --- |
| A constant, such as `"lab-1"` or `0.0` | Broadcast to every row. |
| A sequence of exactly one value per row | Used in row order. |
| A *draw*, `draw(generator, n_rows)` | Called with a seeded `numpy.random.Generator`. |

A draw requires `seed`, and every draw consumes that one generator in `columns` order.
Randomness in a planned design is therefore reproducible from the arguments alone: there
is no way to reach an unseeded global stream through this constructor, and a supplied seed
that no column uses is an error rather than a silent no-op.

## Multiple clocks

`session_order` is only the minimum clock required for session-aware validation. Calendar
time, cumulative exposure, protocol stage, physiological age, and distance from a learned
landmark are not interchangeable with it. They are stored as distinct columns and described
by typed `ClockSpec` metadata. Cumulative-trial and elapsed-time builders, categorical task
phase declarations, and training-fold threshold landmarks are documented in
[Clocks and fold-fitted temporal transforms](clocks-and-transforms.md).

The contract deliberately does not guess a clock or align a landmark. In particular, a
landmark estimated from behaviour is fitted using training observations only before it is
used to transform a validation fold.

## Why row order is not chronology

Preserving source order makes transformations auditable and lets returned split indices
address the original data directly. Requiring an explicit chronology prevents a shuffled
table from changing the scientific question. This is one implementation of Behavio's
broader commitment: useful common coordinates must not manufacture a homogeneous history.
