"""Tabular ingest: the dependency-free CSV/TSV path and the optional Parquet path."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from behavio.adapters.table import (
    ColumnType,
    TableFormat,
    TableReadError,
    TableSource,
    read_table,
    read_tables,
    session_order_from_appearance,
    session_order_from_column,
    session_order_from_explicit,
)
from behavio.contracts.adapter import SessionOrderPolicy, SourceType
from behavio.trials import Study

FIXTURES = Path(__file__).parent / "fixtures" / "tables"


@pytest.fixture
def write_table(tmp_path: Path) -> Callable[[str, str], Path]:
    """Write a one-off table for an edge case that does not deserve a committed fixture."""

    def write(name: str, content: str) -> Path:
        path = tmp_path / name
        path.write_text(content, encoding="utf-8")
        return path

    return write


def test_clean_csv_reads_without_any_optional_dependency() -> None:
    """The headline path: one call, one file, no extras, no configuration."""

    study = read_table(FIXTURES / "trials-clean.csv")

    assert isinstance(study, Study)
    assert len(study) == 6
    assert study.subjects == ("m1", "m2")
    assert study.columns[:4] == ("subject", "session", "trial", "session_order")
    assert study["subject"].tolist() == ["m1", "m1", "m1", "m1", "m2", "m2"]
    assert study["session"].tolist() == ["day-2", "day-2", "day-1", "day-1", "day-1", "day-1"]
    assert study["session_order"].tolist() == [1, 1, 0, 0, 0, 0]
    assert study["choice"].tolist() == [1, 0, 1, 0, 1, 0]
    assert study["rt"].tolist() == [0.412, 0.523, 0.301, 0.664, 0.288, 0.712]


def test_reading_preserves_source_row_order_and_keeps_chronology_separate() -> None:
    """Source order is never sorted away; chronology is requested, not imposed."""

    study = read_table(FIXTURES / "trials-clean.csv")

    assert study["session"].tolist()[0] == "day-2"
    assert study.chronological_indices().tolist() == [2, 3, 0, 1, 4, 5]


def test_inference_types_integers_floats_and_text_separately() -> None:
    study = read_table(FIXTURES / "trials-clean.csv")

    assert study["trial"].dtype == np.int64
    assert study["session_order"].dtype == np.int64
    assert study["choice"].dtype == np.int64
    assert study["rt"].dtype == np.float64
    assert study["subject"].dtype.kind == "U"


def test_identifiers_with_leading_zeros_stay_text(write_table) -> None:
    """`007` is an identifier, not the number seven; inference must not flatten it."""

    path = write_table(
        "leading-zero.csv",
        "subject,session,trial,session_order,code\n007,001,0,0,010\n007,001,1,0,020\n",
    )

    study = read_table(path)

    assert study["subject"].tolist() == ["007", "007"]
    assert study["session"].tolist() == ["001", "001"]
    assert study["code"].tolist() == ["010", "020"]


def test_declared_types_override_inference() -> None:
    study = read_table(
        FIXTURES / "trials-clean.csv",
        dtypes={"choice": ColumnType.FLOAT, "trial": "int", "stimulus": "str"},
    )

    assert study["choice"].dtype == np.float64
    assert study["trial"].dtype == np.int64
    assert study["stimulus"].tolist() == ["0.25", "-0.5", "0.75", "-0.25", "0.5", "-0.75"]


def test_a_table_without_session_order_refuses_to_guess_one() -> None:
    """The scientific centre of the reader: chronology is never inferred."""

    with pytest.raises(TableReadError) as failure:
        read_table(FIXTURES / "trials-no-chronology.csv")

    message = str(failure.value)
    assert "no column 'session_order'" in message
    assert "never infers session chronology" in message
    assert "session_order_from_column('date')" in message
    assert "session_order_from_appearance()" in message
    assert "'session_date'" in message


def test_session_order_derived_from_a_date_column_is_recorded_on_every_trial() -> None:
    study = read_table(
        FIXTURES / "trials-no-chronology.csv",
        session_order=session_order_from_column("session_date"),
        number_trials_by_row_order=True,
    )

    assert study["session_order"].tolist() == [1, 1, 0, 0, 1]
    assert study["trial"].tolist() == [0, 1, 0, 0, 0]
    assert study["source_session_order_rule"].tolist() == ["column:session_date"] * 5
    assert study["source_trial_rule"].tolist() == ["row-order"] * 5


def test_session_order_derived_from_appearance_differs_and_says_so() -> None:
    """Appearance order is a different scientific claim, and the record shows which was made."""

    by_date = read_table(
        FIXTURES / "trials-no-chronology.csv",
        session_order=session_order_from_column("session_date"),
        number_trials_by_row_order=True,
    )
    by_appearance = read_table(
        FIXTURES / "trials-no-chronology.csv",
        session_order=session_order_from_appearance(),
        number_trials_by_row_order=True,
    )

    assert by_date["session_order"].tolist() == [1, 1, 0, 0, 1]
    assert by_appearance["session_order"].tolist() == [0, 0, 1, 0, 1]
    assert by_appearance["source_session_order_rule"].tolist() == ["appearance"] * 5


def test_explicit_session_ordering_is_accepted_and_unknown_sessions_are_rejected() -> None:
    study = read_table(
        FIXTURES / "trials-no-chronology.csv",
        session_order=session_order_from_explicit(["visit-a", "visit-b", "visit-c"]),
        number_trials_by_row_order=True,
    )

    assert study["session_order"].tolist() == [1, 1, 0, 0, 2]

    with pytest.raises(TableReadError, match="does not mention"):
        read_table(
            FIXTURES / "trials-no-chronology.csv",
            session_order=session_order_from_explicit(["visit-a"]),
            number_trials_by_row_order=True,
        )


def test_a_derivation_column_that_varies_within_a_session_is_rejected(write_table) -> None:
    path = write_table(
        "varying-date.csv",
        "subject,session,trial,date\nm1,day-1,0,2025-01-01\nm1,day-1,1,2025-01-02\n",
    )

    with pytest.raises(TableReadError) as failure:
        read_table(path, session_order=session_order_from_column("date"))

    message = str(failure.value)
    assert "not constant within subject 'm1' session 'day-1'" in message
    assert "property of the session" in message


def test_deriving_chronology_for_a_table_that_already_has_it_is_rejected() -> None:
    with pytest.raises(TableReadError, match="already has a 'session_order' column"):
        read_table(FIXTURES / "trials-clean.csv", session_order=session_order_from_appearance())


def test_missing_trial_column_requires_an_explicit_row_order_choice() -> None:
    with pytest.raises(TableReadError) as failure:
        read_table(
            FIXTURES / "trials-no-chronology.csv",
            session_order=session_order_from_appearance(),
        )

    assert "does not number trials from row position unless asked to" in str(failure.value)


def test_missing_data_sentinels_become_nan_or_none_by_column_type() -> None:
    study = read_table(FIXTURES / "trials-missing.tsv")

    assert study["rt"].dtype == np.float64
    assert np.isnan(study["rt"][1])
    assert study["rt"].tolist()[0] == 0.412
    assert study["score"].tolist()[0] == 3.0
    assert np.isnan(study["score"][1])
    assert study["label"].tolist() == ["left", "right", None]


def test_missing_values_are_configurable() -> None:
    study = read_table(FIXTURES / "trials-missing.tsv", missing_values=("",))

    assert study["rt"].tolist() == ["0.412", "NA", "0.5"]
    assert study["label"].tolist() == ["left", "right", "N/A"]


def test_missing_subject_identity_is_an_error_rather_than_an_empty_label(write_table) -> None:
    path = write_table(
        "missing-subject.csv",
        "subject,session,trial,session_order\nm1,day-1,0,0\n,day-1,1,0\n",
    )

    with pytest.raises(TableReadError) as failure:
        read_table(path)

    message = str(failure.value)
    assert "'subject' has no value at data row 2 (line 3)" in message
    assert "identity may not be missing" in message


def test_a_bad_cell_names_the_column_the_row_and_both_ways_to_fix_it() -> None:
    with pytest.raises(TableReadError) as failure:
        read_table(FIXTURES / "trials-bad-types.csv", dtypes={"rt": "float"})

    message = str(failure.value)
    assert "could not convert column 'rt' to float" in message
    assert "data row 3 (line 4)" in message
    assert "trials-bad-types.csv" in message
    assert "contains 'n.a.'" in message
    assert "missing_values" in message
    assert "dtypes={'rt': 'str'}" in message


def test_non_integer_trial_numbers_report_the_offending_row(write_table) -> None:
    path = write_table(
        "fractional-trial.csv",
        "subject,session,trial,session_order\nm1,day-1,0,0\nm1,day-1,1.5,0\n",
    )

    with pytest.raises(TableReadError) as failure:
        read_table(path)

    message = str(failure.value)
    assert "column 'trial' must contain non-negative integers" in message
    assert "data row 2 (line 3)" in message


def test_a_ragged_row_reports_its_line_and_both_field_counts(write_table) -> None:
    path = write_table(
        "ragged.csv",
        "subject,session,trial,session_order\nm1,day-1,0,0\nm1,day-1,1\n",
    )

    with pytest.raises(TableReadError, match=r"line 3 has 3 fields but the header declares 4"):
        read_table(path)


def test_unknown_declared_type_lists_the_supported_names() -> None:
    with pytest.raises(ValueError, match="unknown declared type 'float64'"):
        read_table(FIXTURES / "trials-clean.csv", dtypes={"rt": "float64"})


def test_columns_can_be_selected_and_renamed_without_touching_identity() -> None:
    study = read_table(
        FIXTURES / "trials-clean.csv",
        columns=("subject", "session", "trial", "session_order", "choice"),
        column_map={"choice": "source_choice"},
    )

    assert "stimulus" not in study.columns
    assert "rt" not in study.columns
    assert study["source_choice"].tolist() == [1, 0, 1, 0, 1, 0]


def test_a_source_column_may_not_silently_replace_a_canonical_column() -> None:
    with pytest.raises(TableReadError, match="would replace the canonical 'trial' column"):
        read_table(FIXTURES / "trials-clean.csv", column_map={"choice": "trial"})


def test_identity_columns_can_be_named_when_they_use_other_words(write_table) -> None:
    path = write_table(
        "renamed.csv",
        "participant,visit,index,visit_order,rt\np01,v1,0,0,0.4\np01,v1,1,0,0.5\n",
    )

    study = read_table(
        path,
        subject_column="participant",
        session_column="visit",
        trial_column="index",
        session_order_column="visit_order",
    )

    assert study["subject"].tolist() == ["p01", "p01"]
    assert study["session"].tolist() == ["v1", "v1"]
    assert study["trial"].tolist() == [0, 1]


def test_a_missing_identity_column_names_the_keyword_that_fixes_it(write_table) -> None:
    path = write_table("no-subject.csv", "participant,session,trial,session_order\np,s,0,0\n")

    with pytest.raises(TableReadError) as failure:
        read_table(path)

    message = str(failure.value)
    assert "no column 'subject'" in message
    assert "subject_column=..." in message
    assert "'participant'" in message


def test_tsv_is_read_by_suffix_and_the_format_can_be_declared_explicitly() -> None:
    by_suffix = read_table(FIXTURES / "trials-missing.tsv")
    declared = read_table(FIXTURES / "trials-missing.tsv", format=TableFormat.TSV)

    assert by_suffix.columns == declared.columns
    assert len(by_suffix) == len(declared) == 3


def test_an_unrecognized_suffix_asks_for_an_explicit_format(write_table) -> None:
    path = write_table("trials.dat", "subject,session,trial,session_order\nm1,day-1,0,0\n")

    with pytest.raises(TableReadError, match="cannot tell the table format"):
        read_table(path)

    assert len(read_table(path, format="csv")) == 1


def test_several_single_session_files_derive_chronology_from_file_order() -> None:
    sources = [
        TableSource(
            FIXTURES / name,
            session_order=session_order_from_appearance(),
            number_trials_by_row_order=True,
        )
        for name in ("sessions-first.csv", "sessions-second.csv")
    ]

    study = read_tables(sources)

    assert study["session"].tolist() == ["pre", "pre", "pre", "post", "post", "post"]
    assert study["session_order"].tolist() == [0, 0, 0, 1, 1, 1]
    assert study["trial"].tolist() == [0, 1, 0, 0, 0, 1]
    assert {Path(path).name for path in study["source_table_path"].tolist()} == {
        "sessions-first.csv",
        "sessions-second.csv",
    }


def test_reading_several_files_requires_the_same_declared_options() -> None:
    first = TableSource(FIXTURES / "sessions-first.csv", number_trials_by_row_order=True)
    second = TableSource(FIXTURES / "sessions-second.csv", number_trials_by_row_order=False)

    with pytest.raises(TableReadError, match="differs in \\['number_trials_by_row_order'\\]"):
        read_tables([first, second])


def test_source_provenance_records_the_resolved_file_path() -> None:
    study = read_table(FIXTURES / "trials-clean.csv")

    resolved = str((FIXTURES / "trials-clean.csv").resolve())
    assert study["source_table_path"].tolist() == [resolved] * 6


def test_a_source_column_may_not_take_a_provenance_name(write_table) -> None:
    path = write_table(
        "collide.csv",
        "subject,session,trial,session_order,source_table_path\nm1,day-1,0,0,x\n",
    )

    with pytest.raises(TableReadError, match="reserved for adapter provenance"):
        read_table(path)


def test_an_empty_or_header_only_file_says_which_problem_it_has(write_table) -> None:
    empty = write_table("empty.csv", "")
    header_only = write_table("header-only.csv", "subject,session,trial,session_order\n")

    with pytest.raises(TableReadError, match="is empty"):
        read_table(empty)
    with pytest.raises(TableReadError, match="no data rows"):
        read_table(header_only)


def test_duplicate_headers_are_rejected_before_any_typing(write_table) -> None:
    path = write_table("duplicate.csv", "subject,session,trial,session_order,rt,rt\nm1,d,0,0,1,2\n")

    with pytest.raises(TableReadError, match=r"duplicate header names \['rt'\]"):
        read_table(path)


def test_a_utf8_bom_from_a_spreadsheet_export_does_not_corrupt_the_first_column(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bom.csv"
    path.write_bytes(b"\xef\xbb\xbfsubject,session,trial,session_order\nm1,day-1,0,0\n")

    study = read_table(path)

    assert study.columns[0] == "subject"
    assert study["subject"].tolist() == ["m1"]


def test_table_source_declares_the_adapter_contract() -> None:
    recorded = TableSource(FIXTURES / "trials-clean.csv")
    derived = TableSource(
        FIXTURES / "trials-no-chronology.csv",
        session_order=session_order_from_appearance(),
        number_trials_by_row_order=True,
    )

    assert recorded.adapter_name == "behavio.table"
    assert recorded.source_type is SourceType.LOCAL_FILE
    assert recorded.session_order_policy is SessionOrderPolicy.RECORDED
    assert derived.session_order_policy is SessionOrderPolicy.DERIVED
    assert len(recorded.read()) == 6


def test_a_prepared_source_cannot_be_combined_with_loose_options() -> None:
    with pytest.raises(TypeError, match="cannot be combined"):
        read_table(TableSource(FIXTURES / "trials-clean.csv"), number_trials_by_row_order=True)


def test_a_missing_file_raises_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        read_table(FIXTURES / "absent.csv")


def test_parquet_and_csv_fixtures_describe_the_same_study() -> None:
    pytest.importorskip("pyarrow")

    from_csv = read_table(FIXTURES / "trials-clean.csv")
    from_parquet = read_table(FIXTURES / "trials-clean.parquet")

    assert from_parquet.columns == from_csv.columns
    for name in from_csv.columns:
        if name == "source_table_path":
            continue
        np.testing.assert_array_equal(from_parquet[name], from_csv[name])


def test_parquet_honours_derivations_and_declared_types() -> None:
    pytest.importorskip("pyarrow")

    study = read_table(
        FIXTURES / "trials-clean.parquet",
        dtypes={"choice": "float"},
    )

    assert study["choice"].dtype == np.float64
    assert study["session_order"].tolist() == [1, 1, 0, 0, 0, 0]
