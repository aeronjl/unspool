"""Pinned replicated-lab IBL manifest and adapter source construction."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from behavio import IBLONETrialSource
from benchmarks.ibl2021.selection import manifest_digest

MANIFEST_PATH = Path(__file__).with_name("manifest.json")
EXPECTED_MANIFEST_SHA256 = "c0a45addbc14b936f6b3aaac0c06b6a4d7108725d82dcc3df8f1501f4b1aec0b"
EXPECTED_LABS = 9
EXPECTED_SUBJECTS = 78
EXPECTED_SESSIONS = 468
EXPECTED_TOTAL_FILE_SIZE = 21_465_187
EXPECTED_SUBJECTS_PER_LAB = {
    "angelakilab": 4,
    "churchlandlab": 17,
    "cortexlab": 11,
    "danlab": 6,
    "hoferlab": 6,
    "mainenlab": 9,
    "mrsicflogellab": 7,
    "wittenlab": 8,
    "zadorlab": 10,
}


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    """Load and validate the exact outcome-blind cohort manifest."""

    manifest = json.loads(path.read_text(encoding="utf-8"))
    rows = manifest.get("sessions")
    if not isinstance(rows, list):
        raise ValueError("manifest sessions must be a list")
    observed_digest = manifest_digest(rows)
    if (
        observed_digest != EXPECTED_MANIFEST_SHA256
        or manifest.get("sessions_sha256") != EXPECTED_MANIFEST_SHA256
    ):
        raise ValueError("replicated IBL manifest checksum mismatch")
    subject_sessions = Counter((row["lab"], row["subject"]) for row in rows)
    lab_subjects = Counter(lab for lab, _subject in subject_sessions)
    observed = {
        "n_labs": len(lab_subjects),
        "n_subjects": len(subject_sessions),
        "n_sessions": len(rows),
        "total_file_size": sum(int(row["file_size"]) for row in rows),
        "subjects_per_lab": dict(sorted(lab_subjects.items())),
    }
    expected = {
        "n_labs": EXPECTED_LABS,
        "n_subjects": EXPECTED_SUBJECTS,
        "n_sessions": EXPECTED_SESSIONS,
        "total_file_size": EXPECTED_TOTAL_FILE_SIZE,
        "subjects_per_lab": EXPECTED_SUBJECTS_PER_LAB,
    }
    if observed != expected:
        raise ValueError(f"replicated IBL manifest summary mismatch: {observed}")
    if any(count != 6 for count in subject_sessions.values()):
        raise ValueError("every replicated IBL subject must contribute exactly six sessions")
    return manifest


def sources_from_manifest(manifest: dict[str, Any] | None = None) -> tuple[IBLONETrialSource, ...]:
    """Construct exact ONE adapter sources in committed manifest order."""

    declared = load_manifest() if manifest is None else manifest
    release_tag = str(declared["release_tag"])
    alyx_url = str(declared["public_alyx_url"])
    return tuple(
        IBLONETrialSource(
            session_id=str(row["session"]),
            dataset_id=str(row["dataset_id"]),
            dataset_path=str(row["dataset_path"]),
            file_size=int(row["file_size"]),
            md5=str(row["md5"]),
            release_tag=release_tag,
            session_order=int(row["session_order"]),
            subject=str(row["subject"]),
            session=str(row["session"]),
            lab=str(row["lab"]),
            columns=("contrastLeft", "contrastRight", "feedbackType", "choice"),
            column_map={
                "feedbackType": "source_feedback",
                "choice": "source_choice",
            },
            source_columns={
                "phase": str(row["phase"]),
                "window_position": int(row["window_position"]),
                "task_protocol": str(row["task_protocol"]),
                "n_training_sessions_before_transition": int(
                    row["n_training_sessions_before_transition"]
                ),
                "transition_session": str(row["transition_session"]),
            },
            alyx_url=alyx_url,
        )
        for row in declared["sessions"]
    )
