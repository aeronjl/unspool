# `behavio.adapters` API

Adapters retain source identities while converting trials into the same canonical study
contract. Every source dataclass here satisfies the
[`StudyAdapter` contract](contracts.md#data-source-adapters).

Third-party dependencies are optional everywhere except delimited tables: CSV and TSV are
read with the standard library and NumPy alone, and only Parquet requires an extra.

The last three sections are not data readers. They are the tooling an author of a *model*
wrapper needs — the sequence/row helper, the continuous-outcome prediction type, and the
estimator conformance harness — and none of them names a third-party package. Concrete
wrappers live in [`behavio.foreign`](foreign-models.md).

## Tables: CSV, TSV, and Parquet

::: behavio.adapters.table
    options:
      members_order: source
      show_root_heading: false
      show_source: false

## Adapter conformance

::: behavio.adapters.conformance
    options:
      members_order: source
      show_root_heading: false
      show_source: false

## NWB

::: behavio.adapters.nwb
    options:
      members_order: source
      show_root_heading: false
      show_source: false

## DANDI

::: behavio.adapters.dandi
    options:
      members_order: source
      show_root_heading: false
      show_source: false

## IBL ONE

::: behavio.adapters.ibl_one
    options:
      members_order: source
      show_root_heading: false
      show_source: false

## Trial sequences and source row order

::: behavio.adapters.sequences
    options:
      members_order: source
      show_root_heading: false
      show_source: false

## Continuous-outcome predictions

::: behavio.adapters.prediction
    options:
      members_order: source
      show_root_heading: false
      show_source: false

## Estimator conformance

::: behavio.adapters.estimator_conformance
    options:
      members_order: source
      show_root_heading: false
      show_source: false
