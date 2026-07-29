"""Regenerate `trials-clean.parquet` from `trials-clean.csv`.

Run with `uv run python tests/fixtures/tables/make_parquet_fixture.py`. The output is
committed so that Parquet coverage does not require a writer at test time.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as parquet

HERE = Path(__file__).parent
TYPES = {
    "subject": pa.string(),
    "session": pa.string(),
    "trial": pa.int64(),
    "session_order": pa.int64(),
    "stimulus": pa.float64(),
    "choice": pa.int64(),
    "rt": pa.float64(),
}


def main() -> None:
    with (HERE / "trials-clean.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    schema = pa.schema([(name, kind) for name, kind in TYPES.items()])
    columns = {name: [_convert(row[name], kind) for row in rows] for name, kind in TYPES.items()}
    parquet.write_table(pa.table(columns, schema=schema), HERE / "trials-clean.parquet")


def _convert(text: str, kind: pa.DataType) -> object:
    if kind == pa.int64():
        return int(text)
    if kind == pa.float64():
        return float(text)
    return text


if __name__ == "__main__":
    main()
