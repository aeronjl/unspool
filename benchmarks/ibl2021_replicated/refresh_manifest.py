"""Build the outcome-blind replicated-lab manifest from the fixed IBL release."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from benchmarks.ibl2021.refresh_manifest import PUBLIC_ALYX_URL, PUBLIC_PASSWORD
from benchmarks.ibl2021.selection import (
    RELEASE_TAG,
    manifest_digest,
    select_replicated_learning_panel,
)

DEFAULT_MANIFEST = Path(__file__).with_name("manifest.json")
MINIMUM_SUBJECTS_PER_LAB = 4


def build_manifest() -> dict[str, Any]:
    """Query OpenAlyx and retain all eligible replicated learning windows."""

    try:
        from one.api import ONE
    except ImportError as error:
        raise RuntimeError(
            "refreshing the manifest requires ONE; run with `uv run --extra ibl`"
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
    selected = select_replicated_learning_panel(
        sessions,
        set(datasets_by_session),
        minimum_subjects_per_lab=MINIMUM_SUBJECTS_PER_LAB,
    )
    rows = [_attach_dataset(row, datasets_by_session[row["session"]]) for row in selected]
    subjects_per_lab = Counter((row["lab"], row["subject"]) for row in rows)
    lab_counts = Counter(lab for lab, _subject in subjects_per_lab)
    return {
        "release_tag": RELEASE_TAG,
        "public_alyx_url": PUBLIC_ALYX_URL,
        "selection_policy": (
            "Retain every subject with a first biasedChoiceWorld transition and at least "
            "six preceding trainingChoiceWorld sessions that have trials.table; require "
            "at least four eligible subjects in every observed lab; retain the first and "
            "final three pre-transition training sessions. Selection never reads trial "
            "choices, feedback, rewards, or accuracy."
        ),
        "clock_boundary": (
            "window_position is an ordinal over retained endpoint windows, not uniform "
            "elapsed training time; source session_order is retained separately"
        ),
        "sessions_per_phase": 3,
        "minimum_subjects_per_lab": MINIMUM_SUBJECTS_PER_LAB,
        "n_labs": len(lab_counts),
        "n_subjects": len(subjects_per_lab),
        "n_sessions": len(rows),
        "total_file_size": sum(int(row["file_size"]) for row in rows),
        "subjects_per_lab": dict(sorted(lab_counts.items())),
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
    if len(aws_records) != 1:
        raise RuntimeError(f"expected one public AWS record for session {row['session']}")
    dataset_id = str(dataset["url"]).rstrip("/").rsplit("/", 1)[-1]
    collection = str(dataset.get("collection") or "")
    name = str(dataset.get("name") or "")
    dataset_path = f"{collection}/{name}" if collection else name
    return {
        **row,
        "dataset_id": dataset_id,
        "dataset_path": dataset_path,
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
    print(
        f"{args.output}: {manifest['sessions_sha256']} "
        f"({manifest['n_subjects']} subjects, {manifest['n_sessions']} sessions)"
    )


if __name__ == "__main__":
    main()
