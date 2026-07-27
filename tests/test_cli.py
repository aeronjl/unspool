"""Tests for the intentionally narrow protocol and evidence command line."""

import json
from dataclasses import replace

from test_compiler import source_study
from test_evidence import bundle
from test_protocol import example_protocol

from unspool.cli import main
from unspool.evidence import write_evidence_bundle
from unspool.protocol import CandidateSpec, Setting


def executable_protocol():
    protocol = example_protocol(with_recovery=False)
    candidates = (
        CandidateSpec(
            name="static",
            implementation="unspool.models.BernoulliHistoryGLM",
            hyperparameters=(
                Setting("covariates", ("stimulus",)),
                Setting("choice_lags", 0),
                Setting("l2", 0.1),
            ),
            scored_columns=("choice",),
        ),
        CandidateSpec(
            name="smooth",
            implementation="unspool.models.BernoulliHistoryGLM",
            hyperparameters=(
                Setting("covariates", ("stimulus",)),
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
    assert snapshot["schema_version"] == "unspool.evaluation-snapshot/1"
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


def test_invalid_protocol_returns_clean_error(tmp_path, capsys) -> None:
    path = tmp_path / "invalid.json"
    path.write_text("{}", encoding="utf-8")

    status = main(["protocol-validate", str(path)])

    assert status == 2
    assert "invalid protocol" in capsys.readouterr().err
