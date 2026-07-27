from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from unspool import (
    IBLONEAdapterError,
    IBLONETrialSource,
    read_ibl_one_sessions,
    study_from_ibl_one,
)

SESSION_ID = "13572468-1234-4abc-8def-0123456789ab"
DATASET_ID = "24681357-1234-4abc-8def-0123456789ab"


class Frame:
    def __init__(self, columns: dict[str, list[object]]) -> None:
        self.columns = tuple(columns)
        self._columns = columns

    def __getitem__(self, name: str) -> list[object]:
        return self._columns[name]


class Client:
    def __init__(self, *, fail_first: bool = False, details: dict | None = None) -> None:
        self.fail_first = fail_first
        self.load_calls = 0
        self.list_calls: list[tuple[str, bool, str]] = []
        self.details = details or {
            "rel_path": "alf/_ibl_trials.table.pqt",
            "file_size": 1234,
            "hash": "a" * 32,
        }

    def load_dataset_from_id(self, dataset_id: str, *, details: bool, check_hash: bool):
        self.load_calls += 1
        assert dataset_id == DATASET_ID
        assert details is True
        assert check_hash is True
        if self.fail_first and self.load_calls == 1:
            raise LookupError("dataset is not in the local ONE cache")
        return (
            Frame(
                {
                    "choice": [-1, 1, 0],
                    "feedbackType": [1, -1, -1],
                    "contrastLeft": [0.5, np.nan, 0.0],
                }
            ),
            self.details,
        )

    def list_datasets(self, session_id: str, *, details: bool, query_type: str):
        self.list_calls.append((session_id, details, query_type))
        return None


def source(**changes: object) -> IBLONETrialSource:
    values = {
        "session_id": SESSION_ID,
        "dataset_id": DATASET_ID,
        "dataset_path": "alf/_ibl_trials.table.pqt",
        "file_size": 1234,
        "md5": "a" * 32,
        "release_tag": "2021_Q1_IBL_et_al_Behaviour",
        "session_order": 4,
        "subject": "mouse-1",
        "session": "training-5",
        "lab": "lab-a",
        "columns": ("choice", "feedbackType"),
        "column_map": {"feedbackType": "source_feedback"},
        "source_columns": {"task_phase": "late_training"},
    }
    values.update(changes)
    return IBLONETrialSource(**values)


def test_one_adapter_preserves_exact_identity_provenance_and_source_semantics() -> None:
    client = Client()

    study = study_from_ibl_one(source(), client=client)

    assert study.subjects == ("mouse-1",)
    assert study.columns[:4] == ("subject", "session", "trial", "session_order")
    assert list(study["trial"]) == [0, 1, 2]
    assert set(study["session_order"]) == {4}
    assert list(study["choice"]) == [-1, 1, 0]
    assert "source_feedback" in study.columns
    assert set(study["lab"]) == {"lab-a"}
    assert set(study["task_phase"]) == {"late_training"}
    assert set(study["source_ibl_dataset_id"]) == {DATASET_ID}
    assert set(study["source_ibl_dataset_md5"]) == {"a" * 32}
    assert client.list_calls == []


def test_one_adapter_primes_remote_session_metadata_when_id_is_not_cached() -> None:
    client = Client(fail_first=True)

    study = study_from_ibl_one(source(), client=client)

    assert len(study) == 3
    assert client.load_calls == 2
    assert client.list_calls == [(SESSION_ID, True, "remote")]


def test_one_adapter_rejects_changed_dataset_provenance() -> None:
    client = Client(
        details={
            "rel_path": "alf/_ibl_trials.table.pqt",
            "file_size": 1234,
            "hash": "b" * 32,
        }
    )

    with pytest.raises(IBLONEAdapterError, match="provenance mismatch"):
        study_from_ibl_one(source(), client=client)


def test_multiple_one_sessions_preserve_input_order_and_require_common_columns() -> None:
    client = Client()
    second = replace(
        source(),
        session_id="aaaaaaaa-1234-4abc-8def-0123456789ab",
        dataset_id=DATASET_ID,
        session_order=5,
        session="training-6",
    )

    study = read_ibl_one_sessions((source(), second), client=client)

    assert len(study) == 6
    assert list(study["session"]) == ["training-5"] * 3 + ["training-6"] * 3
    assert list(study["session_order"]) == [4] * 3 + [5] * 3

    incompatible = replace(second, columns=("choice",), column_map={})
    with pytest.raises(IBLONEAdapterError, match="different columns"):
        read_ibl_one_sessions((source(), incompatible), client=client)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"session_id": "not-a-uuid"}, "UUID"),
        ({"dataset_path": "/absolute/table.pqt"}, "normalized relative"),
        ({"file_size": 0}, "positive integer"),
        ({"md5": "abc"}, "32 lowercase"),
        ({"release_tag": ""}, "non-empty"),
        ({"session_order": -1}, "non-negative"),
        ({"column_map": {"choice": "subject"}}, "canonical identity"),
        ({"source_columns": {"lab": "other"}}, "reserved"),
        ({"alyx_url": "http://example.com"}, "HTTPS"),
    ],
)
def test_one_source_rejects_unpinned_or_ambiguous_identity(changes, message) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        source(**changes)


def test_one_adapter_rejects_missing_selected_columns() -> None:
    with pytest.raises(IBLONEAdapterError, match="missing selected"):
        study_from_ibl_one(source(columns=("choice", "missing"), column_map={}), client=Client())
