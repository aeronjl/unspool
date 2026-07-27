# Tabular, IBL ONE, NWB, and DANDI interoperability

Unspool keeps `Study` small and format-independent. Interoperability code converts source
tables into that contract; it does not let file names, dataframe indices, or archive order
become hidden scientific assumptions.

## Dataframes

`Study.from_dataframe(frame)` preserves dataframe column order and row order while
deliberately ignoring the index. Subject identity, session identity, within-session trial
number, and session chronology must be ordinary explicit columns:

```python
from unspool import Study

study = Study.from_dataframe(trials_dataframe)
```

The method is dataframe-like rather than pandas-specific and does not add a core pandas
dependency.

## Reading exact IBL ONE trial tables

The optional ONE adapter makes release identity and dataset identity explicit instead of
searching for whichever table currently matches a filename:

```bash
uv sync --extra ibl
```

```python
from unspool import IBLONETrialSource, study_from_ibl_one

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

IBL's source `choice` coding is not Unspool's binary Bernoulli coding. The adapter therefore
does not silently reinterpret `-1`, `0`, and `+1`; callers must give the source field an
honest name such as `source_choice` and perform any model-specific recoding explicitly.
The [replicated IBL benchmark](../benchmarks/ibl2021_replicated/README.md) exercises this
contract against 468 checksum-pinned public tables.

## Reading local NWB sessions

NWB stores trials in a `TimeIntervals` dynamic table and normally uses one file for one
experimental session. Install the optional adapter dependencies and declare chronology:

```bash
uv sync --extra nwb
```

```python
from unspool import NWBSessionSource, read_nwb

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
cross-session ordinal, so Unspool refuses to infer it from timestamps, paths, or the order
of a list. An Unspool-authored NWB file embeds it losslessly; an external file requires the
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
from unspool import write_nwb

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
Unspool–NWB–Unspool round trip lossless even when canonical trial IDs are not the NWB row
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
from unspool import DANDINWBSource, study_from_dandi

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

## Current boundary

- HDF5 blob-backed NWB assets are supported; NWB-Zarr is not yet supported.
- The adapter imports trial tables, not arbitrary neural time series or processing modules.
- Ragged trial fields require source-specific preprocessing outside the generic adapter.
- DANDI upload, authentication, and draft mutation are deliberately out of scope.
- The ONE adapter is a read-only exact-dataset importer; release discovery, remote mutation,
  and implicit selection by dataset name remain outside its contract.

The [public interoperability benchmark](../benchmarks/nwb_dandi_interoperability/README.md)
pins a real DANDI asset and verifies the complete identity, chronology, source-semantics,
and provenance contract.
