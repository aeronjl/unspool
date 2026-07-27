# The longitudinal study contract

`Study` is Unspool's canonical, format-independent representation of observed trials. It
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
task versions, or source-specific metadata. Unspool does not silently rename, aggregate,
impute, or sort them.

## Construction

```python
from unspool import Study

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

Pandas-like objects can use `Study.from_dataframe()`. Its index is deliberately ignored;
the required identity and chronology must remain explicit columns. Local NWB round trips
and version-pinned DANDI streaming use the same contract and are documented in
[Tabular, NWB, and DANDI interoperability](interoperability.md).

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
table from changing the scientific question. This is one implementation of Unspool's
broader commitment: useful common coordinates must not manufacture a homogeneous history.
