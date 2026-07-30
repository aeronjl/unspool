# Tabular, IBL ONE, NWB, and DANDI interoperability

Behavio keeps `Study` small and format-independent. Interoperability code converts source
tables into that contract; it does not let file names, dataframe indices, or archive order
become hidden scientific assumptions.

!!! tip "Looking for pose, ethograms, or bouts?"
    This page is about *trial tables*. Continuously sampled behaviour - keypoint
    trajectories, discovered state bouts, human ethograms, and the clock
    transforms that put them on one time coordinate - has its own boundary:
    [Observed behaviour](observed-behaviour.md).

<figure class="doc-figure doc-figure--wide" data-figure-kind="Conceptual">
  <img src="../assets/interoperability-pipeline.svg" alt="Dataframes, IBL ONE, NWB, and DANDI converge on one Study contract, pass through explicit semantic and provenance declarations, and enter the same prospective analysis boundary.">
  <figcaption><strong>One scientific boundary across formats.</strong> Adapters preserve source identity and translate structure into the Study contract; they never invent chronology, units, or behavioural semantics.</figcaption>
</figure>

## CSV, TSV, and Parquet files

`read_table()` is the shortest supported path from a file on disk to a validated study:

```python
from behavio.adapters.table import read_table

study = read_table("trials.csv")
```

CSV and TSV need **no optional dependencies at all**. A trial table is the baseline way
data enters Behavio rather than an optional data source, so the delimited reader uses only
the standard library and NumPy. Parquet is the one tabular format behind an extra, because
a binary container genuinely needs a reader:

```bash
uv sync --extra parquet
```

The format follows the suffix (`.csv`, `.tsv`, `.tab`, `.parquet`, `.pqt`) and can be
declared with `format=` when a file is named something else. A UTF-8 byte-order mark, as
written by spreadsheet exports, is handled by default.

### What the reader will not guess

Three source facts are never inferred, because each is a scientific claim rather than a
parsing detail.

**Session chronology.** A table that carries no `session_order` column cannot be read until
the caller names a derivation. The refusal is the same one the NWB adapter makes, and it
says how to proceed:

```text
trials.csv has no column 'session_order'. Behavio never infers session chronology from row,
file, or filename order. Either add the column, name it with session_order_column=..., or
record an explicit derivation with session_order=session_order_from_column('date'),
session_order_from_explicit([...]), or session_order_from_appearance().
Available columns: ['subject', 'session', 'session_date', 'stimulus', 'choice'].
```

Each derivation is a *recorded* choice, written to a `source_session_order_rule` column on
every trial, so a derived chronology can never be mistaken for one the source declared:

```python
from behavio.adapters.table import (
    read_table,
    session_order_from_appearance,
    session_order_from_column,
    session_order_from_explicit,
)

# Rank each subject's sessions by a column that carries time.
study = read_table("trials.csv", session_order=session_order_from_column("session_date"))

# Or state the ordering of session labels yourself.
study = read_table(
    "trials.csv", session_order=session_order_from_explicit(["baseline", "week-1", "week-4"])
)

# Or claim, explicitly, that the rows were written in chronological order.
study = read_table("trials.csv", session_order=session_order_from_appearance())
```

`read_tables([...])` reads several identically declared files in the given order, so one
file per session plus `session_order_from_appearance()` derives chronology from file order.

**Trial numbering.** A table with no `trial` column is not silently numbered by row
position; `number_trials_by_row_order=True` makes that choice explicit and records it in a
`source_trial_rule` column.

**Column types.** Types are inferred by a published ladder: a column whose non-missing
cells all parse as integers becomes `int64`, then floats become `float64`, and anything
else stays text. Zero-padded digits such as `007` are treated as identifiers and stay text,
so a participant code is never flattened into a number. Any column can be declared instead:

```python
study = read_table("trials.csv", dtypes={"rt": "float", "participant_code": "str"})
```

A declared type that a cell cannot satisfy names the column, the row, the file, the
offending value, and both ways out:

```text
could not convert column 'rt' to float: data row 412 (line 413) of trials.csv contains 'n.a.'.
Add it to missing_values if it means 'missing', or read the column as text with
dtypes={'rt': 'str'}.
```

### Missing data

Empty cells and the sentinels `NA`, `N/A`, `na`, `n/a`, `NaN`, `NAN`, `nan`, `NULL`,
`null`, and `None` are treated as missing after surrounding whitespace is stripped; pass
`missing_values=(...)` to replace that list exactly. A missing cell becomes `NaN` in a
numeric column and `None` in a text column, so an absent label stays distinguishable from
an empty one. Missing `subject` or `session` identity is an error naming the row rather
than a blank identifier, and missing `trial` or `session_order` is an error too, since
neither may be imputed.

### Provenance and reproducibility

Every trial records its absolute source file in `source_table_path`, matching the NWB
adapter. That path is machine-specific and therefore enters the protocol fingerprint; pass
`record_source_path=False` (or `--omit-source-path` on the command line) when a
fingerprint that is identical across machines matters more.

## Dataframes

`Study.from_dataframe(frame)` preserves dataframe column order and row order while
deliberately ignoring the index. Subject identity, session identity, within-session trial
number, and session chronology must be ordinary explicit columns:

```python
from behavio import Study

study = Study.from_dataframe(trials_dataframe)
```

The method is dataframe-like rather than pandas-specific and does not add a core pandas
dependency. It stays the right entry point for a frame you already have in memory;
`read_table()` is the entry point for a file, and unlike `from_dataframe` it can type
columns, name a chronology derivation, and report failures against source line numbers.

## Reading exact IBL ONE trial tables

The optional ONE adapter makes release identity and dataset identity explicit instead of
searching for whichever table currently matches a filename:

```bash
uv sync --extra ibl
```

```python
from behavio.adapters import IBLONETrialSource, study_from_ibl_one

study = study_from_ibl_one(
    IBLONETrialSource(
        session_id="13572468-1234-4abc-8def-0123456789ab",
        dataset_id="24681357-1234-4abc-8def-0123456789ab",
        dataset_path="alf/_ibl_trials.table.pqt",
        file_size=12_345,
        md5="0123456789abcdef0123456789abcdef",
        release_tag="2021_Q1_IBL_et_al_Behaviour",
        subject="mouse-1",
        session_order=4,
        columns=("contrastLeft", "contrastRight", "feedbackType", "choice"),
        column_map={"feedbackType": "source_feedback", "choice": "source_choice"},
    )
)
```

The adapter asks ONE to load the exact dataset UUID with hash checking, then verifies its
relative path, byte size, and MD5 against the declaration. Session UUID, dataset UUID,
release tag, path, size, checksum, and Alyx origin remain addressable on every trial. A
multi-session reader preserves declared input order while `Study.chronological_indices()`
uses the explicit `session_order`.

IBL's source `choice` coding is not Behavio's binary Bernoulli coding. The adapter therefore
does not silently reinterpret `-1`, `0`, and `+1`; callers must give the source field an
honest name such as `source_choice` and perform any model-specific recoding explicitly.
The [replicated IBL benchmark](https://github.com/aeronjl/behavio/tree/main/benchmarks/ibl2021_replicated) exercises this
contract against 468 checksum-pinned public tables.

## Reading local NWB sessions

NWB stores trials in a `TimeIntervals` dynamic table and normally uses one file for one
experimental session. Install the optional adapter dependencies and declare chronology:

```bash
uv sync --extra nwb
```

```python
from behavio.adapters import NWBSessionSource, read_nwb

study = read_nwb(
    NWBSessionSource(
        "sub-mouse1_ses-day3_behavior.nwb",
        session_order=2,
        columns=("start_time", "stop_time", "response", "stimulus"),
        column_map={"response": "choice"},
    )
)
```

Subject identity comes from `Subject.subject_id` and session identity from `session_id`
unless supplied explicitly. `session_order` is different: generic NWB has no standard
cross-session ordinal, so Behavio refuses to infer it from timestamps, paths, or the order
of a list. An Behavio-authored NWB file embeds it losslessly; an external file requires the
caller to supply it.

By default, all scalar trial columns are copied. Ragged arrays and object-valued cells are
rejected because silently stringifying them would change their meaning. Use `columns` to
select the scalar behavioral fields needed for a study. `column_map` renames source fields
but never allows one to replace the four canonical identity columns accidentally.
Source names alone do not establish units or event semantics: for example, an NWB
`response_time` may be an absolute event timestamp rather than a decision duration. Map
such fields to an unambiguous name before configuring a model that expects response times.

Several sessions can be assembled with `read_nwb_sessions()`. Every
`NWBSessionSource` retains its own explicit mapping, and all resulting sessions must have
the same columns. Input file order is preserved; chronological order remains available
separately through `Study.chronological_indices()`.

## Writing NWB

NWB export requires exactly one subject and one session, matching NWB's session-level
file model. It also requires explicit finite `start_time` and `stop_time` columns:

```python
from datetime import UTC, datetime
from behavio.adapters import write_nwb

write_nwb(
    one_session,
    "sub-mouse1_ses-day3_behavior.nwb",
    session_description="Probabilistic choice learning.",
    identifier="mouse1-day3",
    session_start_time=datetime(2026, 7, 27, tzinfo=UTC),
)
```

The writer embeds native subject, session, trial, and session-order values in custom trial
columns as well as populating standard NWB subject/session metadata. This makes an
Behavio–NWB–Behavio round trip lossless even when canonical trial IDs are not the NWB row
numbers. Existing files are never overwritten unless `overwrite=True` is explicit.

PyNWB's structural validator is exercised by the adapter test suite. The NWB project
recommends NWB Inspector as an additional best-practice check before archival release;
see the [official validation guidance](https://pynwb.readthedocs.io/en/stable/validation.html).

## Streaming from DANDI

The DANDI adapter accepts only an exact published version and exact NWB asset path. It
resolves the asset through the public REST API, retains its asset ID, byte size, SHA-256,
and content-addressed S3 URL, then uses `remfile` to fetch only the HDF5 regions needed for
the selected trial columns:

```bash
uv sync --extra dandi
```

```python
from behavio.adapters import DANDINWBSource, study_from_dandi

study = study_from_dandi(
    DANDINWBSource(
        dandiset_id="000004",
        version="0.220126.1852",
        asset_path="sub-P11HMH/sub-P11HMH_ses-20061101_ecephys+image.nwb",
        session_order=0,
        session="2006-11-01",
        columns=("start_time", "stop_time", "stim_phase", "response_value"),
    )
)
```

Draft versions are rejected because their contents can change. Public reads need no API
key. Every trial carries the Dandiset ID, version, path, asset ID, SHA-256, and NWB
identifier, so provenance survives subsetting and validation folds. The implementation
follows the official [DANDI REST API](https://docs.dandiarchive.org/api/rest-api/) and
[PyNWB streaming guidance](https://pynwb.readthedocs.io/en/stable/tutorials/advanced_io/streaming.html).

## The adapter contract

Every source above is also a `StudyAdapter`: it declares a stable `adapter_name` and
`adapter_version`, the `source_type` it reads from, the `session_order_policy` it follows,
and a `read()` method returning a `Study`.

```python
from behavio.adapters.table import TableSource
from behavio.contracts.adapter import adapter_capabilities

source = TableSource("trials.csv")
adapter_capabilities(source).to_dict()
# {'adapter_name': 'behavio.table', 'adapter_version': '1',
#  'source_type': 'local-file', 'session_order_policy': 'recorded'}
```

`session_order_policy` is the honest half of the declaration: `recorded` means the
chronology came from the caller or from an explicit record in the source, and `derived`
means a named rule produced it. Writing your own adapter is documented in
[Extend Behavio](extensions.md), including the runnable conformance harness.

## Current boundary

- BIDS `_beh.tsv`/`_events.tsv`, PsychoPy, jsPsych, Bpod, and pyControl have no dedicated
  readers yet; their tables can be read today with `read_table()` plus explicit column
  names and a named chronology derivation.
- HDF5 blob-backed NWB assets are supported; NWB-Zarr is not yet supported.
- The adapter imports trial tables, not arbitrary neural time series or processing modules.
- Ragged trial fields require source-specific preprocessing outside the generic adapter.
- DANDI upload, authentication, and draft mutation are deliberately out of scope.
- The ONE adapter is a read-only exact-dataset importer; release discovery, remote mutation,
  and implicit selection by dataset name remain outside its contract.

The [public interoperability benchmark](https://github.com/aeronjl/behavio/tree/main/benchmarks/nwb_dandi_interoperability)
pins a real DANDI asset and verifies the complete identity, chronology, source-semantics,
and provenance contract.
