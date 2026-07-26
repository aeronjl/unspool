"""Regenerate the pinned IBL benchmark manifest from the fixed public release tag."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarks.ibl2021.selection import RELEASE_TAG, manifest_digest, select_learning_panel

PUBLIC_ALYX_URL = "https://openalyx.internationalbrainlab.org"
PUBLIC_PASSWORD = "international"
DEFAULT_MANIFEST = Path(__file__).with_name("manifest.json")


def build_manifest() -> dict[str, Any]:
    """Query OpenAlyx and construct the exact public-data manifest."""

    try:
        from one.api import ONE
    except ImportError as error:
        raise RuntimeError(
            "refreshing the manifest requires ONE; run with `uv run --with one-api`"
        ) from error

    one = ONE(base_url=PUBLIC_ALYX_URL, password=PUBLIC_PASSWORD, silent=True)
    sessions = one.alyx.rest("sessions", "list", tag=RELEASE_TAG)
    datasets = one.alyx.rest(
        "datasets",
        "list",
        tag=RELEASE_TAG,
        dataset_type="trials.table",
    )
    datasets_by_session = {_session_identifier(dataset): dataset for dataset in datasets}
    selected = select_learning_panel(sessions, set(datasets_by_session))
    rows = [_attach_dataset(row, datasets_by_session[row["session"]]) for row in selected]
    return {
        "release_tag": RELEASE_TAG,
        "public_alyx_url": PUBLIC_ALYX_URL,
        "selection_policy": (
            "Per lab, maximise the number of trials.table trainingChoiceWorld sessions "
            "before the first biasedChoiceWorld session; break ties by total eligible "
            "task-session coverage then subject identifier; retain the first and final "
            "three pre-transition training sessions."
        ),
        "sessions_per_phase": 3,
        "sessions": rows,
        "sessions_sha256": manifest_digest(rows),
    }


def _session_identifier(dataset: dict[str, Any]) -> str:
    return str(dataset["session"]).rstrip("/").rsplit("/", 1)[-1]


def _attach_dataset(row: dict[str, Any], dataset: dict[str, Any]) -> dict[str, Any]:
    file_records = [record for record in dataset.get("file_records", []) if record.get("exists")]
    aws_records = [
        record
        for record in file_records
        if "ibl-brain-wide-map-public.s3.amazonaws.com" in str(record.get("data_url") or "")
    ]
    if not aws_records:
        raise RuntimeError(f"no public AWS record for session {row['session']}")
    dataset_id = str(dataset["url"]).rstrip("/").rsplit("/", 1)[-1]
    return {
        **row,
        "dataset_id": dataset_id,
        "file_size": int(dataset["file_size"]),
        "md5": str(dataset["hash"]),
        "data_url": str(aws_records[0]["data_url"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    manifest = build_manifest()
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{args.output}: {manifest['sessions_sha256']}")


if __name__ == "__main__":
    main()
