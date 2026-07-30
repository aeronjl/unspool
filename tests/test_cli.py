"""Tests for the intentionally narrow protocol and evidence command line."""

import csv
import json
from dataclasses import replace

from test_compiler import source_study
from test_evidence_bundles import bundle
from test_protocol import example_protocol

from behavio.cli import main
from behavio.protocol.schema import CandidateSpec, Setting
from behavio.report.evidence_bundles import write_evidence_bundle


def executable_protocol():
    protocol = example_protocol(with_recovery=False)
    candidates = (
        CandidateSpec(
            name="static",
            implementation="behavio.models.BernoulliHistoryGLM",
            hyperparameters=(
                Setting("predictors", ("stimulus",)),
                Setting("choice_lags", 0),
                Setting("l2", 0.1),
            ),
            scored_columns=("choice",),
        ),
        CandidateSpec(
            name="smooth",
            implementation="behavio.models.BernoulliHistoryGLM",
            hyperparameters=(
                Setting("predictors", ("stimulus",)),
                Setting("choice_lags", 0),
                Setting("l2", 1.0),
            ),
            scored_columns=("choice",),
        ),
    )
    return replace(
        protocol,
        cohort=replace(
            protocol.cohort,
            expected_subjects=2,
            expected_sessions=6,
            expected_observations=12,
        ),
        panel=replace(protocol.panel, minimum_sessions=3),
        validation=replace(
            protocol.validation,
            settings=(Setting("min_train_sessions", 2), Setting("horizon", 1)),
        ),
        candidates=candidates,
        comparison=replace(protocol.comparison, bootstrap_repetitions=50),
    )


def write_protocol_and_study(tmp_path):
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(executable_protocol().canonical_json(), encoding="utf-8")
    study = source_study()
    study_path = tmp_path / "study.json"
    study_path.write_text(
        json.dumps({"columns": {column: study[column].tolist() for column in study.columns}}),
        encoding="utf-8",
    )
    return protocol_path, study_path


def write_study_table(path, *, drop=(), extra=None, delimiter=","):
    """Write the shared fixture study as a delimited trial table."""

    study = source_study()
    names = [name for name in study.columns if name not in drop]
    rows = []
    for row in range(len(study)):
        record = {name: study[name][row] for name in names}
        if extra is not None:
            record.update({name: values[row] for name, values in extra.items()})
        rows.append(record)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_protocol_validate_can_emit_frozen_canonical_json(tmp_path, capsys) -> None:
    protocol_path, _study_path = write_protocol_and_study(tmp_path)
    frozen_path = tmp_path / "frozen.json"

    status = main(["protocol-validate", str(protocol_path), "--freeze-out", str(frozen_path)])
    response = json.loads(capsys.readouterr().out)

    assert status == 0
    assert response["valid"]
    assert response["state"] == "draft"
    assert response["frozen_output"] == str(frozen_path)
    assert json.loads(frozen_path.read_text())["state"] == "frozen"


def test_execute_writes_reviewable_non_executable_snapshot(tmp_path, capsys) -> None:
    protocol_path, study_path = write_protocol_and_study(tmp_path)
    output = tmp_path / "run"

    status = main(["execute", str(protocol_path), str(study_path), str(output)])
    response = json.loads(capsys.readouterr().out)

    assert status == 0
    assert response["state"] == "evaluated"
    assert not response["recovery_pending"]
    assert {path.name for path in output.iterdir()} == {
        "protocol.json",
        "cohort.json",
        "plan.json",
        "evaluation.json",
        "snapshot.json",
    }
    snapshot = json.loads((output / "snapshot.json").read_text())
    assert snapshot["schema_version"] == "behavio.evaluate.folds-snapshot/1"
    assert snapshot["state"] == "evaluated"
    assert all(not path.name.endswith((".pkl", ".pickle")) for path in output.iterdir())


def test_execute_rejects_output_overwrite(tmp_path, capsys) -> None:
    protocol_path, study_path = write_protocol_and_study(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()

    status = main(["execute", str(protocol_path), str(study_path), str(output)])

    assert status == 2
    assert "output path already exists" in capsys.readouterr().err


def test_inspect_compare_and_report_verify_bundles(tmp_path, capsys) -> None:
    left = write_evidence_bundle(bundle(marker="a"), tmp_path / "left.zip")
    right = write_evidence_bundle(bundle(marker="b"), tmp_path / "right.zip")

    assert main(["inspect", str(left)]) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["protocol_state"] == "reported"
    assert inspected["cohort_observations"] == 12

    assert main(["bundle-compare", str(left), str(right)]) == 0
    compared = json.loads(capsys.readouterr().out)
    assert compared["same_protocol"]
    assert compared["changed_paths"] == ["environment/environment.json"]

    report_path = tmp_path / "report.md"
    assert main(["report", str(left), "--output", str(report_path)]) == 0
    response = json.loads(capsys.readouterr().out)
    assert response["report"] == str(report_path)
    assert report_path.read_text().startswith("# Learning trajectory forecast")


def test_execute_reads_a_csv_study_and_matches_the_json_path(tmp_path, capsys) -> None:
    """The adoption case: a plain CSV reaches the same evaluation as a JSON columns blob."""

    protocol_path, study_path = write_protocol_and_study(tmp_path)
    csv_path = write_study_table(tmp_path / "trials.csv")

    assert main(["execute", str(protocol_path), str(study_path), str(tmp_path / "json")]) == 0
    from_json = json.loads(capsys.readouterr().out)
    assert main(["execute", str(protocol_path), str(csv_path), str(tmp_path / "csv")]) == 0
    from_csv = json.loads(capsys.readouterr().out)

    assert from_csv["state"] == "evaluated"
    assert from_csv["winner"] == from_json["winner"]
    assert from_csv["ranking_status"] == from_json["ranking_status"]
    from_csv_report = json.loads((tmp_path / "csv" / "evaluation.json").read_text())
    from_json_report = json.loads((tmp_path / "json" / "evaluation.json").read_text())
    # The CSV read records provenance the JSON blob does not carry, so the plan fingerprint
    # legitimately differs; every scientific result must not.
    assert (
        from_csv_report["execution_plan_fingerprint"]
        != from_json_report["execution_plan_fingerprint"]
    )
    del from_csv_report["execution_plan_fingerprint"]
    del from_json_report["execution_plan_fingerprint"]
    assert from_csv_report == from_json_report


def test_omitting_the_source_path_reproduces_the_json_fingerprint(tmp_path, capsys) -> None:
    """A machine-independent fingerprint is available when provenance is not wanted."""

    protocol_path, study_path = write_protocol_and_study(tmp_path)
    csv_path = write_study_table(tmp_path / "trials.csv")

    assert main(["execute", str(protocol_path), str(study_path), str(tmp_path / "json")]) == 0
    capsys.readouterr()
    status = main(
        [
            "execute",
            str(protocol_path),
            str(csv_path),
            str(tmp_path / "csv"),
            "--omit-source-path",
        ]
    )
    capsys.readouterr()

    assert status == 0
    assert json.loads((tmp_path / "csv" / "evaluation.json").read_text()) == json.loads(
        (tmp_path / "json" / "evaluation.json").read_text()
    )


def test_execute_reads_a_tsv_study(tmp_path, capsys) -> None:
    protocol_path, _ = write_protocol_and_study(tmp_path)
    tsv_path = write_study_table(tmp_path / "trials.tsv", delimiter="\t")

    status = main(["execute", str(protocol_path), str(tsv_path), str(tmp_path / "run")])

    assert status == 0
    assert json.loads(capsys.readouterr().out)["state"] == "evaluated"


def test_execute_refuses_a_csv_that_carries_no_chronology(tmp_path, capsys) -> None:
    protocol_path, _ = write_protocol_and_study(tmp_path)
    csv_path = write_study_table(tmp_path / "trials.csv", drop=("session_order",))

    status = main(["execute", str(protocol_path), str(csv_path), str(tmp_path / "run")])

    error = capsys.readouterr().err
    assert status == 2
    assert "never infers session chronology" in error
    assert "--session-order-from-column" in error
    assert "--session-order-from-appearance" in error


def test_execute_derives_chronology_only_when_the_caller_names_a_rule(tmp_path, capsys) -> None:
    protocol_path, _ = write_protocol_and_study(tmp_path)
    study = source_study()
    dates = [f"2025-01-{int(order) + 1:02d}" for order in study["session_order"]]
    csv_path = write_study_table(
        tmp_path / "trials.csv",
        drop=("session_order",),
        extra={"session_date": dates},
    )

    status = main(
        [
            "execute",
            str(protocol_path),
            str(csv_path),
            str(tmp_path / "run"),
            "--session-order-from-column",
            "session_date",
        ]
    )

    assert status == 0
    assert json.loads(capsys.readouterr().out)["state"] == "evaluated"


def test_execute_honours_an_explicit_study_format_override(tmp_path, capsys) -> None:
    protocol_path, _ = write_protocol_and_study(tmp_path)
    odd_suffix = write_study_table(tmp_path / "trials.data")

    assert main(["execute", str(protocol_path), str(odd_suffix), str(tmp_path / "guess")]) == 2
    assert "cannot tell the table format" in capsys.readouterr().err

    status = main(
        [
            "execute",
            str(protocol_path),
            str(odd_suffix),
            str(tmp_path / "declared"),
            "--study-format",
            "csv",
        ]
    )

    assert status == 0
    assert json.loads(capsys.readouterr().out)["state"] == "evaluated"


def test_execute_maps_source_identity_column_names(tmp_path, capsys) -> None:
    protocol_path, _ = write_protocol_and_study(tmp_path)
    study = source_study()
    rows = [
        {
            "participant": study["subject"][row],
            "visit": study["session"][row],
            "trial": int(study["trial"][row]),
            "visit_order": int(study["session_order"][row]),
            "choice": int(study["choice"][row]),
            "stimulus": float(study["stimulus"][row]),
            "species": study["species"][row],
            "source_asset": study["source_asset"][row],
            "source_row": int(study["source_row"][row]),
        }
        for row in range(len(study))
    ]
    csv_path = tmp_path / "renamed.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    status = main(
        [
            "execute",
            str(protocol_path),
            str(csv_path),
            str(tmp_path / "run"),
            "--subject-column",
            "participant",
            "--session-column",
            "visit",
            "--session-order-column",
            "visit_order",
        ]
    )

    assert status == 0
    assert json.loads(capsys.readouterr().out)["state"] == "evaluated"


def test_invalid_protocol_returns_clean_error(tmp_path, capsys) -> None:
    path = tmp_path / "invalid.json"
    path.write_text("{}", encoding="utf-8")

    status = main(["protocol-validate", str(path)])

    assert status == 2
    assert "invalid protocol" in capsys.readouterr().err


def composed_protocol():
    """A protocol whose candidates are combinator expressions rather than plain classes.

    Before the estimator registry was wired into declaration verification, this protocol
    could be *instantiated* by the command line but could not be *run*: the verifier
    resolved ``behavio.compose.hierarchical`` to the combinator function, saw a
    ``HierarchicalModel`` instance, and reported a contradiction that refused the run.
    """

    protocol = executable_protocol()
    candidates = (
        CandidateSpec(
            name="static",
            implementation="behavio.models.BernoulliHistoryGLM",
            hyperparameters=(
                Setting("predictors", ("stimulus",)),
                Setting("choice_lags", 0),
                Setting("l2", 0.1),
            ),
            scored_columns=("choice",),
        ),
        CandidateSpec(
            name="smooth",
            implementation="behavio.compose.hierarchical",
            hyperparameters=(
                Setting("base", "behavio.models.BernoulliHistoryGLM"),
                Setting("base.predictors", ("stimulus",)),
                Setting("base.choice_lags", 0),
                Setting("base.l2", 1.0),
                Setting("over", "subject"),
                Setting("scale", 0.5),
            ),
            scored_columns=("choice",),
        ),
    )
    return replace(protocol, candidates=candidates)


def test_execute_runs_a_composed_candidate_from_flat_settings(tmp_path, capsys) -> None:
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(composed_protocol().canonical_json(), encoding="utf-8")
    study = source_study()
    study_path = tmp_path / "study.json"
    study_path.write_text(
        json.dumps({"columns": {column: study[column].tolist() for column in study.columns}}),
        encoding="utf-8",
    )
    output = tmp_path / "snapshot"

    assert main(["execute", str(protocol_path), str(study_path), str(output)]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["state"] == "evaluated"
    evaluation = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
    assert set(evaluation["candidates"]) == {"static", "smooth"}
    assert evaluation["candidates"]["smooth"]["model_signature"].startswith("hierarchical[")


def test_execute_names_the_registry_when_an_implementation_is_unknown(tmp_path, capsys) -> None:
    protocol = replace(
        executable_protocol(),
        candidates=(
            CandidateSpec(
                name="static",
                implementation="not.a.registered.Model",
                hyperparameters=(),
                scored_columns=("choice",),
            ),
            CandidateSpec(
                name="smooth",
                implementation="behavio.models.BernoulliHistoryGLM",
                hyperparameters=(Setting("predictors", ("stimulus",)),),
                scored_columns=("choice",),
            ),
        ),
    )
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(protocol.canonical_json(), encoding="utf-8")
    study = source_study()
    study_path = tmp_path / "study.json"
    study_path.write_text(
        json.dumps({"columns": {column: study[column].tolist() for column in study.columns}}),
        encoding="utf-8",
    )

    assert main(["execute", str(protocol_path), str(study_path), str(tmp_path / "out")]) == 2

    error = capsys.readouterr().err
    assert "unknown estimator 'not.a.registered.Model'" in error
    assert "behavio.models.BernoulliHistoryGLM" in error
