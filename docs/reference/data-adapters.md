# `behavio.adapters` API

Adapters retain source identities while converting trials into the same canonical study
contract. Every source dataclass here satisfies the
[`StudyAdapter` contract](contracts.md#data-source-adapters).

Third-party dependencies are optional everywhere except delimited tables: CSV and TSV are
read with the standard library and NumPy alone, and only Parquet requires an extra.

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
