# Tabular ingest fixtures

Small hand-authored trial tables that exercise the CSV/TSV/Parquet reader
(`behavio.adapters.table`). They are written here rather than downloaded, so no checksum
manifest is needed; `trials-clean.parquet` is generated from `trials-clean.csv` by
`make_parquet_fixture.py` in this directory and committed so that Parquet coverage does not
depend on a writer being installed.

| File | Exercises |
| --- | --- |
| `trials-clean.csv` | a complete table: all four canonical columns, source row order that is not chronological order |
| `trials-clean.parquet` | the same trials in a typed binary container |
| `trials-no-chronology.csv` | no `session_order` and no `trial`; carries a `session_date` column so a derivation can be named |
| `trials-missing.tsv` | tab-separated, with `NA`, `N/A`, and empty-cell missing sentinels in float, integer, and text columns |
| `trials-bad-types.csv` | an `rt` column containing `n.a.`, which is not a default sentinel, so a declared float type must fail with an actionable message |
| `sessions-first.csv`, `sessions-second.csv` | one session per file, read together so chronology can be derived from file order |
