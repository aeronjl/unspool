"""The study-adapter contract and the conformance harness that makes it executable."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from behavio.adapters.conformance import (
    AdapterConformanceError,
    CheckStatus,
    assert_study_adapter_conforms,
    check_study_adapter,
)
from behavio.adapters.dandi import DANDINWBSource
from behavio.adapters.ibl_one import IBLONETrialSource
from behavio.adapters.nwb import NWBSessionSource
from behavio.adapters.table import (
    TableSource,
    session_order_from_appearance,
    session_order_from_column,
)
from behavio.contracts.adapter import (
    AdapterCapabilities,
    SessionOrderPolicy,
    SourceType,
    StudyAdapter,
    adapter_capabilities,
)
from behavio.trials import Study

FIXTURES = Path(__file__).parent / "fixtures" / "tables"

CLEAN_ROWS = (
    {"subject": "m1", "session": "day-2", "trial": 0, "choice": 1, "rt": 0.412},
    {"subject": "m1", "session": "day-2", "trial": 1, "choice": 0, "rt": 0.523},
    {"subject": "m1", "session": "day-1", "trial": 0, "choice": 1, "rt": 0.301},
    {"subject": "m1", "session": "day-1", "trial": 1, "choice": 0, "rt": 0.664},
    {"subject": "m2", "session": "day-1", "trial": 0, "choice": 1, "rt": 0.288},
    {"subject": "m2", "session": "day-1", "trial": 1, "choice": 0, "rt": 0.712},
)


def test_every_shipped_source_satisfies_the_study_adapter_protocol() -> None:
    sources = (
        TableSource(FIXTURES / "trials-clean.csv"),
        NWBSessionSource("session.nwb", session_order=0),
        DANDINWBSource(
            dandiset_id="000004",
            version="0.220126.1852",
            asset_path="sub-P11HMH/sub-P11HMH_ses-20061101_ecephys+image.nwb",
            session_order=0,
        ),
        IBLONETrialSource(
            session_id="13572468-1234-4abc-8def-0123456789ab",
            dataset_id="24681357-1234-4abc-8def-0123456789ab",
            dataset_path="alf/_ibl_trials.table.pqt",
            file_size=12_345,
            md5="0123456789abcdef0123456789abcdef",
            release_tag="2021_Q1_IBL_et_al_Behaviour",
            subject="mouse-1",
            session_order=4,
        ),
    )

    for source in sources:
        assert isinstance(source, StudyAdapter)
        capabilities = adapter_capabilities(source)
        assert capabilities.adapter_name.startswith("behavio.")
        assert capabilities.adapter_version
        assert isinstance(capabilities.source_type, SourceType)


def test_declared_source_types_and_chronology_policies_are_honest() -> None:
    """Network adapters say so, and none of the archive adapters derive chronology."""

    assert TableSource(FIXTURES / "trials-clean.csv").source_type is SourceType.LOCAL_FILE
    assert NWBSessionSource("s.nwb").source_type is SourceType.LOCAL_FILE
    assert (
        DANDINWBSource(
            dandiset_id="000004",
            version="0.220126.1852",
            asset_path="a.nwb",
            session_order=0,
        ).source_type
        is SourceType.REMOTE_ARCHIVE
    )
    assert NWBSessionSource("s.nwb").session_order_policy is SessionOrderPolicy.RECORDED
    assert (
        TableSource(
            FIXTURES / "trials-no-chronology.csv",
            session_order=session_order_from_appearance(),
            number_trials_by_row_order=True,
        ).session_order_policy
        is SessionOrderPolicy.DERIVED
    )


def test_capabilities_serialize_for_a_provenance_record() -> None:
    capabilities = adapter_capabilities(TableSource(FIXTURES / "trials-clean.csv"))

    assert capabilities.to_dict() == {
        "adapter_name": "behavio.table",
        "adapter_version": "1",
        "source_type": "local-file",
        "session_order_policy": "recorded",
    }


def test_capabilities_reject_an_object_that_is_not_an_adapter() -> None:
    with pytest.raises(TypeError, match="StudyAdapter contract"):
        adapter_capabilities(object())


def test_capabilities_reject_empty_identity() -> None:
    with pytest.raises(ValueError, match="adapter_name"):
        AdapterCapabilities(
            adapter_name="",
            adapter_version="1",
            source_type=SourceType.LOCAL_FILE,
            session_order_policy=SessionOrderPolicy.RECORDED,
        )


def test_the_table_adapter_passes_every_conformance_check() -> None:
    report = assert_study_adapter_conforms(
        TableSource(FIXTURES / "trials-clean.csv"),
        expected_rows=CLEAN_ROWS,
        chronology_withheld=lambda: TableSource(FIXTURES / "trials-no-chronology.csv"),
        require_complete=True,
    )

    assert report.passed
    assert not report.skipped
    assert {check.status for check in report} == {CheckStatus.PASSED}
    assert report.capabilities is not None
    assert report.capabilities.adapter_name == "behavio.table"


def test_a_derived_chronology_still_has_to_be_asked_for() -> None:
    """A derivation is a named choice: remove the name and the adapter must refuse."""

    assert_study_adapter_conforms(
        TableSource(
            FIXTURES / "trials-no-chronology.csv",
            session_order=session_order_from_column("session_date"),
            number_trials_by_row_order=True,
        ),
        expected_rows=(
            {"subject": "p01", "session": "visit-b", "choice": 1},
            {"subject": "p01", "session": "visit-b", "choice": 0},
            {"subject": "p01", "session": "visit-a", "choice": 1},
            {"subject": "p02", "session": "visit-a", "choice": 0},
            {"subject": "p02", "session": "visit-c", "choice": 1},
        ),
        chronology_withheld=lambda: TableSource(
            FIXTURES / "trials-no-chronology.csv", number_trials_by_row_order=True
        ),
        require_complete=True,
    )


def test_the_harness_catches_an_adapter_that_invents_session_order() -> None:
    report = check_study_adapter(
        _FabricatingAdapter(),
        chronology_withheld=_FabricatingAdapter,
    )

    assert not report.passed
    failure = report.failures[0]
    assert failure.name == "refuses-to-fabricate-session-order"
    assert "invented session_order" in failure.detail
    assert "claim about time" in failure.detail
    with pytest.raises(AdapterConformanceError, match="refuses-to-fabricate-session-order"):
        assert_study_adapter_conforms(
            _FabricatingAdapter(), chronology_withheld=_FabricatingAdapter
        )


def test_the_harness_catches_an_adapter_that_sorts_source_rows() -> None:
    report = check_study_adapter(_SortingAdapter(), expected_rows=CLEAN_ROWS)

    failures = {check.name for check in report.failures}
    assert "preserves-source-trial-order" in failures
    assert "preserves-subject-and-session-boundaries" in failures
    detail = next(
        check.detail for check in report.failures if check.name.startswith("preserves-source")
    )
    assert "reordered or altered source rows" in detail


def test_the_harness_catches_an_adapter_that_merges_sessions() -> None:
    report = check_study_adapter(_MergingAdapter(), expected_rows=CLEAN_ROWS)

    failures = {check.name for check in report.failures}
    assert "preserves-subject-and-session-boundaries" in failures


def test_the_harness_reports_what_it_could_not_check() -> None:
    report = check_study_adapter(TableSource(FIXTURES / "trials-clean.csv"))

    assert report.passed
    skipped = {check.name for check in report.skipped}
    assert skipped == {
        "preserves-source-trial-order",
        "preserves-subject-and-session-boundaries",
        "refuses-to-fabricate-session-order",
    }
    with pytest.raises(AdapterConformanceError, match="skipped"):
        assert_study_adapter_conforms(
            TableSource(FIXTURES / "trials-clean.csv"), require_complete=True
        )


def test_the_harness_reports_an_adapter_that_cannot_read_at_all() -> None:
    report = check_study_adapter(_FailingAdapter())

    assert not report.passed
    assert report.failures[0].name == "reads-a-valid-study"
    assert "RuntimeError" in report.failures[0].detail


def test_the_harness_rejects_an_object_that_is_not_an_adapter() -> None:
    report = check_study_adapter(object())

    assert report.capabilities is None
    assert report.failures[0].name == "declares-adapter-identity"


def test_the_nwb_adapter_passes_the_same_harness(tmp_path: Path) -> None:
    pytest.importorskip("pynwb")
    from behavio.adapters.nwb import write_nwb

    original = Study(
        {
            "subject": ["mouse-1"] * 3,
            "session": ["day-3"] * 3,
            "trial": [0, 1, 2],
            "session_order": [2] * 3,
            "start_time": [1.0, 2.0, 3.0],
            "stop_time": [1.5, 2.5, 3.5],
            "choice": [1, 0, 1],
        }
    )
    path = write_nwb(
        original,
        tmp_path / "session.nwb",
        session_description="A behavioural session.",
        identifier="conformance",
        session_start_time=datetime(2025, 1, 2, tzinfo=UTC),
    )
    stripped = _nwb_without_embedded_chronology(tmp_path, original)

    report = assert_study_adapter_conforms(
        NWBSessionSource(path),
        expected_rows=(
            {"subject": "mouse-1", "session": "day-3", "trial": 0, "choice": 1},
            {"subject": "mouse-1", "session": "day-3", "trial": 1, "choice": 0},
            {"subject": "mouse-1", "session": "day-3", "trial": 2, "choice": 1},
        ),
        chronology_withheld=lambda: NWBSessionSource(stripped),
        require_complete=True,
    )

    assert report.capabilities is not None
    assert report.capabilities.adapter_name == "behavio.nwb"


def _nwb_without_embedded_chronology(tmp_path: Path, study: Study) -> Path:
    """Write a plain external NWB file: real trials, no Behavio chronology column."""

    from pynwb import NWBHDF5IO, NWBFile
    from pynwb.file import Subject

    nwbfile = NWBFile(
        session_description="An external session.",
        identifier="external",
        session_start_time=datetime(2025, 1, 2, tzinfo=UTC),
        session_id="day-3",
        subject=Subject(subject_id="mouse-1"),
    )
    nwbfile.add_trial_column(name="choice", description="Observed choice.")
    for row in range(len(study)):
        nwbfile.add_trial(
            start_time=float(study["start_time"][row]),
            stop_time=float(study["stop_time"][row]),
            choice=int(study["choice"][row]),
        )
    path = tmp_path / "external.nwb"
    with NWBHDF5IO(path, mode="w") as io:
        io.write(nwbfile)
    return path


class _StubAdapter:
    """A minimal in-memory adapter used to prove the harness can fail an implementation."""

    adapter_name = "test.stub"
    adapter_version = "0"
    source_type = SourceType.IN_MEMORY
    session_order_policy = SessionOrderPolicy.RECORDED

    def _columns(self) -> dict[str, Any]:
        return {
            "subject": [row["subject"] for row in CLEAN_ROWS],
            "session": [row["session"] for row in CLEAN_ROWS],
            "trial": [row["trial"] for row in CLEAN_ROWS],
            "session_order": [1, 1, 0, 0, 0, 0],
            "choice": [row["choice"] for row in CLEAN_ROWS],
            "rt": [row["rt"] for row in CLEAN_ROWS],
        }

    def read(self) -> Study:
        return Study(self._columns())


class _FabricatingAdapter(_StubAdapter):
    """Numbers sessions by arrival with no record and no named rule."""

    def read(self) -> Study:
        columns = self._columns()
        seen: dict[Any, dict[Any, int]] = {}
        orders = []
        for subject, session in zip(columns["subject"], columns["session"], strict=True):
            ranks = seen.setdefault(subject, {})
            ranks.setdefault(session, len(ranks))
            orders.append(ranks[session])
        columns["session_order"] = orders
        return Study(columns)


class _SortingAdapter(_StubAdapter):
    """Silently returns trials in chronological order instead of source order."""

    def read(self) -> Study:
        study = Study(self._columns())
        return study.take(study.chronological_indices())


class _MergingAdapter(_StubAdapter):
    """Keeps row order but collapses two sessions of one subject into one."""

    def read(self) -> Study:
        columns = self._columns()
        columns["session"] = ["day-1"] * 6
        columns["session_order"] = [0] * 6
        columns["trial"] = [0, 1, 2, 3, 0, 1]
        return Study(columns)


class _FailingAdapter(_StubAdapter):
    def read(self) -> Study:
        raise RuntimeError("the source is unavailable")


def test_the_stub_adapters_are_themselves_valid_studies() -> None:
    """Guard the negative tests: they must fail on order, not on an invalid study."""

    for adapter in (_SortingAdapter(), _MergingAdapter(), _FabricatingAdapter()):
        study = adapter.read()
        assert isinstance(study, Study)
        assert len(study) == 6
        assert np.asarray(study["trial"]).dtype.kind == "i"
