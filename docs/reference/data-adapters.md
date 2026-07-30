# `behavio.adapters` API

Adapters retain source identities while converting trials into the same canonical study
contract. Every source dataclass here satisfies the
[`StudyAdapter` contract](contracts.md#data-source-adapters).

Third-party dependencies are optional everywhere except delimited tables: CSV and TSV are
read with the standard library and NumPy alone, and only Parquet requires an extra.

The last section is not a data reader. It is the estimator conformance harness a *model*
wrapper author runs, and it names no third-party package. The other two things a wrapper
author needs live where what they describe lives: `SequenceLayout` beside
[`Study`](study-and-task.md), and `DensityPrediction` beside the other predictions in
[`behavio.contracts`](contracts.md). Concrete wrappers live in
[`behavio.foreign`](foreign-models.md).

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

## Estimator conformance

::: behavio.adapters.estimator_conformance
    options:
      members_order: source
      show_root_heading: false
      show_source: false
