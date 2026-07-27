import json

import numpy as np
import pytest

from unspool import (
    FIT_ARTIFACT_SCHEMA,
    BernoulliHistoryGLM,
    ChoiceSpec,
    FitArtifactError,
    Study,
    TaskSpec,
    export_fit,
    fit_artifact_from_dict,
    fit_artifact_from_json,
    fit_model,
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
    model = BernoulliHistoryGLM(covariates=("stimulus",), choice_lags=0)
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
