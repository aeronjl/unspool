import importlib

import numpy as np
import pytest

from unspool import (
    ArviZUnavailableError,
    PosteriorError,
    PosteriorGroup,
    PosteriorResult,
    PosteriorVariable,
    posterior_result_from_arviz,
)


def variable(name, values, dims, **coords):
    return PosteriorVariable(name=name, values=np.asarray(values), dims=dims, coords=coords)


def complete_result() -> PosteriorResult:
    chain = np.asarray([0, 1])
    draw = np.asarray([10, 11, 12])
    subject = np.asarray(["mouse-a", "mouse-b"])
    trial = np.asarray(["s1:0", "s1:1", "s2:0", "s2:1"])
    prediction_trial = np.asarray(["s3:0", "s3:1"])
    posterior = PosteriorGroup(
        "posterior",
        (
            variable(
                "learning_rate",
                np.full((2, 3), 0.25),
                ("chain", "draw"),
                chain=chain,
                draw=draw,
            ),
            variable(
                "bias",
                np.arange(12).reshape(2, 3, 2) / 10,
                ("chain", "draw", "subject"),
                chain=chain,
                draw=draw,
                subject=subject,
            ),
        ),
        attrs={"sampler": "nuts"},
    )
    sample_stats = PosteriorGroup(
        "sample_stats",
        (
            variable(
                "diverging",
                np.zeros((2, 3), dtype=bool),
                ("chain", "draw"),
                chain=chain,
                draw=draw,
            ),
        ),
    )
    posterior_predictive = PosteriorGroup(
        "posterior_predictive",
        (
            variable(
                "choice",
                np.zeros((2, 3, 4), dtype=np.int64),
                ("chain", "draw", "trial"),
                chain=chain,
                draw=draw,
                trial=trial,
            ),
        ),
    )
    log_likelihood = PosteriorGroup(
        "log_likelihood",
        (
            variable(
                "choice",
                -np.ones((2, 3, 4)),
                ("chain", "draw", "trial"),
                chain=chain,
                draw=draw,
                trial=trial,
            ),
        ),
    )
    observed_data = PosteriorGroup(
        "observed_data",
        (variable("choice", [0, 1, 1, 0], ("trial",), trial=trial),),
    )
    constant_data = PosteriorGroup(
        "constant_data",
        (variable("stimulus", [-1.0, 1.0, 0.5, -0.5], ("trial",), trial=trial),),
    )
    predictions = PosteriorGroup(
        "predictions",
        (
            variable(
                "choice",
                np.ones((2, 3, 2), dtype=np.int64),
                ("chain", "draw", "trial"),
                chain=chain,
                draw=draw,
                trial=prediction_trial,
            ),
        ),
    )
    predictions_constant_data = PosteriorGroup(
        "predictions_constant_data",
        (
            variable(
                "stimulus",
                [0.75, -0.75],
                ("trial",),
                trial=prediction_trial,
            ),
        ),
    )
    return PosteriorResult(
        model_name="binary-q-learning",
        model_signature="sha256:model",
        inference_library="PyMC",
        inference_library_version="5.25.0",
        parameter_names=("learning_rate", "bias"),
        groups=(
            posterior,
            sample_stats,
            log_likelihood,
            posterior_predictive,
            observed_data,
            constant_data,
            predictions,
            predictions_constant_data,
        ),
        parameter_space_fingerprint="sha256:parameters",
        attrs={"task": {"outcome": "choice"}},
    )


def test_labelled_result_is_immutable_and_retains_natural_parameter_identity() -> None:
    result = complete_result()

    assert result.group_names == (
        "posterior",
        "sample_stats",
        "log_likelihood",
        "posterior_predictive",
        "observed_data",
        "constant_data",
        "predictions",
        "predictions_constant_data",
    )
    assert result.n_chains == 2
    assert result.n_draws == 3
    assert result["posterior"].variable_names == ("learning_rate", "bias")
    assert result["posterior"]["bias"].intrinsic_dims == ("subject",)
    assert not result["posterior"]["bias"].values.flags.writeable
    assert not result["posterior"]["bias"].coords["subject"].flags.writeable
    with pytest.raises(ValueError):
        result["posterior"]["bias"].values[0, 0, 0] = 4.0
    with pytest.raises(TypeError):
        result.attrs["new"] = "value"


def test_variable_and_result_reject_unlabelled_or_misaligned_evidence() -> None:
    with pytest.raises(PosteriorError, match="one label"):
        variable("alpha", [0.1, 0.2], ("draw",), draw=[0])
    with pytest.raises(PosteriorError, match="missing declared"):
        PosteriorResult(
            model_name="model",
            model_signature="signature",
            inference_library="backend",
            inference_library_version="1",
            parameter_names=("missing",),
            groups=(
                PosteriorGroup(
                    "posterior",
                    (
                        variable(
                            "alpha",
                            [[0.1, 0.2]],
                            ("chain", "draw"),
                            chain=[0],
                            draw=[0, 1],
                        ),
                    ),
                ),
            ),
        )


def test_sampled_groups_must_match_posterior_chain_and_draw_coordinates() -> None:
    result = complete_result()
    bad_stats = PosteriorGroup(
        "sample_stats",
        (
            variable(
                "diverging",
                np.zeros((2, 3), dtype=bool),
                ("chain", "draw"),
                chain=[0, 2],
                draw=[10, 11, 12],
            ),
        ),
    )

    with pytest.raises(PosteriorError, match=r"conflicting coordinates|does not match"):
        PosteriorResult(
            model_name=result.model_name,
            model_signature=result.model_signature,
            inference_library=result.inference_library,
            inference_library_version=result.inference_library_version,
            parameter_names=result.parameter_names,
            groups=(result["posterior"], bad_stats),
        )


def test_arviz_roundtrip_preserves_groups_dimensions_coordinates_and_provenance() -> None:
    pytest.importorskip("arviz")
    result = complete_result()

    arviz_data = result.to_arviz()
    imported = posterior_result_from_arviz(
        arviz_data,
        model_name=result.model_name,
        model_signature=result.model_signature,
        inference_library=result.inference_library,
        inference_library_version=result.inference_library_version,
        parameter_names=result.parameter_names,
        parameter_space_fingerprint=result.parameter_space_fingerprint,
    )

    assert set(imported.group_names) == set(result.group_names)
    assert imported.parameter_names == result.parameter_names
    assert imported.parameter_space_fingerprint == result.parameter_space_fingerprint
    for group in result.groups:
        for original in group.variables:
            restored = imported[group.name][original.name]
            assert restored.dims == original.dims
            np.testing.assert_array_equal(restored.values, original.values)
            for dim in original.dims:
                np.testing.assert_array_equal(restored.coords[dim], original.coords[dim])


def test_arviz_remains_an_optional_dependency(monkeypatch) -> None:
    posterior_module = importlib.import_module("unspool.posterior")
    real_import = posterior_module.importlib.import_module

    def unavailable(name):
        if name == "arviz":
            raise ImportError("not installed")
        return real_import(name)

    monkeypatch.setattr(posterior_module.importlib, "import_module", unavailable)

    with pytest.raises(ArviZUnavailableError, match=r"unspool\[probabilistic\]"):
        complete_result().to_arviz()
