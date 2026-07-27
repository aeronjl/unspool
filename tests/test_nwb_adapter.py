from datetime import UTC, datetime

import numpy as np
import pytest

from unspool import (
    NWBAdapterError,
    NWBSessionSource,
    Study,
    add_study_trials,
    read_nwb,
    read_nwb_sessions,
    study_from_nwbfile,
    write_nwb,
)

pynwb = pytest.importorskip("pynwb")
NWBFile = pynwb.NWBFile
Subject = pynwb.file.Subject
validate = pynwb.validate


def make_study(*, session: str = "day-2", session_order: int = 1) -> Study:
    return Study(
        {
            "subject": ["mouse-1"] * 3,
            "session": [session] * 3,
            "trial": [4, 2, 3],
            "session_order": [session_order] * 3,
            "start_time": [10.0, 2.0, 6.0],
            "stop_time": [11.0, 3.0, 7.0],
            "choice": [1, 0, 1],
            "stimulus": [0.5, -0.25, 0.75],
            "label": ["right", "left", "right"],
        }
    )


def test_nwb_round_trip_preserves_trial_rows_identity_and_source_columns(tmp_path) -> None:
    original = make_study()
    path = tmp_path / "session.nwb"

    written = write_nwb(
        original,
        path,
        session_description="A behavioral learning session.",
        identifier="unspool-round-trip",
        session_start_time=datetime(2025, 1, 2, tzinfo=UTC),
    )
    restored = read_nwb(written)

    assert validate(path=str(path)) == []
    for name in original.columns:
        np.testing.assert_array_equal(restored[name], original[name])
    assert restored["source_nwb_identifier"].tolist() == ["unspool-round-trip"] * 3
    assert restored["source_nwb_path"].tolist() == [str(path.resolve())] * 3
    assert restored.chronological_indices().tolist() == [1, 2, 0]


def test_external_nwb_requires_explicit_chronology_and_supports_column_mapping() -> None:
    nwbfile = NWBFile(
        session_description="External task",
        identifier="external",
        session_start_time=datetime(2024, 4, 5, tzinfo=UTC),
        session_id="session-a",
        subject=Subject(subject_id="subject-a"),
    )
    nwbfile.add_trial_column("response", "binary response")
    nwbfile.add_trial(start_time=0.0, stop_time=1.0, response=1)
    nwbfile.add_trial(start_time=2.0, stop_time=3.0, response=0)

    with pytest.raises(NWBAdapterError, match="cannot be inferred"):
        study_from_nwbfile(nwbfile)

    study = study_from_nwbfile(
        nwbfile,
        session_order=3,
        columns=("start_time", "stop_time", "response"),
        column_map={"response": "choice"},
    )

    assert study["subject"].tolist() == ["subject-a", "subject-a"]
    assert study["session"].tolist() == ["session-a", "session-a"]
    assert study["trial"].tolist() == [0, 1]
    assert study["session_order"].tolist() == [3, 3]
    assert study["choice"].tolist() == [1, 0]


def test_nwb_import_rejects_non_scalar_trials_unless_they_are_not_selected() -> None:
    nwbfile = NWBFile(
        session_description="Ragged task",
        identifier="ragged",
        session_start_time=datetime(2024, 4, 5, tzinfo=UTC),
        session_id="session-a",
        subject=Subject(subject_id="subject-a"),
    )
    nwbfile.add_trial_column("events", "event labels", index=True)
    nwbfile.add_trial(start_time=0.0, stop_time=1.0, events=[1, 2])
    nwbfile.add_trial(start_time=2.0, stop_time=3.0, events=[3])

    with pytest.raises(NWBAdapterError, match="non-scalar"):
        study_from_nwbfile(nwbfile, session_order=0)

    study = study_from_nwbfile(
        nwbfile,
        session_order=0,
        columns=("start_time", "stop_time"),
    )
    assert study.columns[:6] == (
        "subject",
        "session",
        "trial",
        "session_order",
        "start_time",
        "stop_time",
    )


def test_nwb_export_rejects_ambiguous_or_incomplete_session_data(tmp_path) -> None:
    missing_times = Study(
        {
            "subject": ["mouse"],
            "session": ["one"],
            "trial": [0],
            "session_order": [0],
        }
    )
    with pytest.raises(NWBAdapterError, match="start_time"):
        write_nwb(
            missing_times,
            tmp_path / "missing.nwb",
            session_description="missing",
            identifier="missing",
            session_start_time=datetime.now(UTC),
        )

    multi_session = make_study()
    columns = {name: multi_session[name].copy() for name in multi_session.columns}
    columns["session"][0] = "another"
    columns["session_order"][0] = 2
    with pytest.raises(NWBAdapterError, match="exactly one"):
        add_study_trials(
            NWBFile(
                session_description="multi",
                identifier="multi",
                session_start_time=datetime.now(UTC),
                session_id="day-2",
                subject=Subject(subject_id="mouse-1"),
            ),
            Study(columns),
        )


def test_multiple_nwb_sessions_preserve_source_order_and_explicit_chronology(tmp_path) -> None:
    late_path = tmp_path / "late.nwb"
    early_path = tmp_path / "early.nwb"
    common = {
        "session_description": "learning",
        "session_start_time": datetime(2025, 2, 3, tzinfo=UTC),
    }
    write_nwb(make_study(session="late", session_order=1), late_path, identifier="late", **common)
    write_nwb(
        make_study(session="early", session_order=0), early_path, identifier="early", **common
    )

    study = read_nwb_sessions(
        [
            NWBSessionSource(late_path),
            NWBSessionSource(early_path),
        ]
    )

    assert study["session"][:3].tolist() == ["late"] * 3
    assert study["session"][-3:].tolist() == ["early"] * 3
    assert study.chronological_indices().tolist() == [4, 5, 3, 1, 2, 0]
