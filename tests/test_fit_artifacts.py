import json

import numpy as np
import pytest

from behavio import BernoulliHistoryGLM, ChoiceSpec, Study, TaskSpec, export_fit, fit_model
from behavio.report import (
    FIT_ARTIFACT_SCHEMA,
    FitArtifactError,
    fit_artifact_from_dict,
    fit_artifact_from_json,
)


def fitted_example():
    design = Study(
        {
            "subject": ["a"] * 60,
            "session": ["s"] * 60,
            "trial": list(range(60)),
            "session_order": [0] * 60,
            "stimulus": np.linspace(-2.0, 2.0, 60),
        }
    )
    model = BernoulliHistoryGLM(predictors=("stimulus",), choice_lags=0)
    study = model.simulate(design, {"intercept": -0.2, "stimulus": 1.0}, seed=14)
    task = TaskSpec(choice=ChoiceSpec(options=(0, 1)), predictors=("stimulus",))
    return fit_model(model, study, task=task), study


def test_fit_artifact_round_trips_common_context_without_executable_objects() -> None:
    fitted, study = fitted_example()

    artifact = export_fit(fitted, study)
    restored = fit_artifact_from_json(artifact.canonical_json())

    assert artifact.schema_version == FIT_ARTIFACT_SCHEMA
    assert restored.to_dict() == artifact.to_dict()
    assert restored.fingerprint == artifact.fingerprint
    assert artifact.data["n_trials"] == 60
    assert len(artifact.data["sha256"]) == 64
    assert artifact.task["choice"]["options"] == (0, 1)
    assert artifact.parameters[0]["name"] == "intercept"
    assert artifact.audit["model_signature"] == fitted.model.signature
    json.dumps(artifact.to_dict(), allow_nan=False)


def test_fit_artifact_fingerprint_changes_with_source_data() -> None:
    fitted, study = fitted_example()
    changed = Study(
        {
            **{name: study[name] for name in study.columns if name != "stimulus"},
            "stimulus": np.asarray(study["stimulus"]) + 0.01,
        }
    )

    original = export_fit(fitted, study)
    modified = export_fit(fitted, changed)

    assert original.data["sha256"] != modified.data["sha256"]
    assert original.fingerprint != modified.fingerprint


def test_fit_artifact_reader_rejects_schema_drift() -> None:
    fitted, study = fitted_example()
    payload = export_fit(fitted, study).to_dict()
    payload["unexpected"] = True

    with pytest.raises(FitArtifactError, match="fields differ"):
        fit_artifact_from_dict(payload)


def test_fit_artifact_rejects_nonportable_study_values() -> None:
    fitted, study = fitted_example()
    opaque = Study(
        {**{name: study[name] for name in study.columns}, "opaque": [object()] * len(study)}
    )

    with pytest.raises(FitArtifactError, match="not portable JSON data"):
        export_fit(fitted, opaque)


def psychometric_example():
    from behavio.models import PsychometricFunction

    levels = np.linspace(-2.0, 2.0, 9)
    stimulus = np.repeat(levels, 8)
    rows = stimulus.size
    design = Study(
        {
            "subject": ["a"] * rows,
            "session": ["s"] * rows,
            "trial": list(range(rows)),
            "session_order": [0] * rows,
            "stimulus": stimulus,
        }
    )
    model = PsychometricFunction(stimulus="stimulus", outcome="choice", n_restarts=2)
    truth = model.parameters_from_components(
        threshold=0.2, width=0.6, guess_rate=0.05, lapse_rate=0.05
    )
    study = model.simulate(design, dict(truth), seed=3)
    task = TaskSpec(choice=ChoiceSpec(options=(0, 1)), predictors=("stimulus",))
    return fit_model(model, study, task=task), study


def test_derived_quantities_reach_the_exported_artifact() -> None:
    fitted, study = psychometric_example()

    artifact = export_fit(fitted, study)
    payload = artifact.to_dict()
    by_name = {record["name"]: record for record in payload["derived"]}

    # The estimated coordinate is still ``log_width``; the published width is beside it.
    assert [record["name"] for record in payload["parameters"]][1] == "log_width"
    assert by_name["width"]["interval_level"] == 0.95
    assert by_name["width"]["standard_error"] is not None
    assert len(by_name["width"]["interval"]) == 2
    assert by_name["slope_at_threshold"]["standard_error"] is None
    restored = fit_artifact_from_json(artifact.canonical_json())
    assert restored.to_dict() == payload
    assert restored.fingerprint == artifact.fingerprint
    json.dumps(payload, allow_nan=False)


def test_an_artifact_without_derived_quantities_is_byte_identical() -> None:
    fitted, study = fitted_example()

    artifact = export_fit(fitted, study)
    payload = artifact.to_dict()

    assert artifact.derived == ()
    assert "derived" not in payload
    # An artifact written before derived quantities existed still decodes unchanged.
    assert fit_artifact_from_dict(payload).fingerprint == artifact.fingerprint


def test_the_artifact_reader_rejects_a_malformed_derived_block() -> None:
    fitted, study = psychometric_example()
    payload = export_fit(fitted, study).to_dict()

    with pytest.raises(FitArtifactError, match="derived must be an array"):
        fit_artifact_from_dict({**payload, "derived": "width"})
    with pytest.raises(FitArtifactError, match="derived quantity names must be unique"):
        fit_artifact_from_dict({**payload, "derived": [{"name": "w"}, {"name": "w"}]})
