"""``behavio.foreign`` must be importable, describable, and honest without any extra.

Deliberately *not* gated on ``pytest.importorskip``. The point of these tests is the
behaviour a machine without the extra sees, so they block the real ``pyddm`` import
regardless of whether it happens to be installed, which is the only way the missing-extra
path is ever executed in a checkout that has it.
"""

from __future__ import annotations

import sys
from types import ModuleType

import numpy as np
import pytest

from behavio import Study
from behavio.contracts import BehaviourEstimator, GenerativeBehaviourModel
from behavio.foreign import PYDDM_EXTRA, PYDDM_SERIES, ForeignPackageUnavailableError
from behavio.foreign.pyddm import PyDDMDriftDiffusion


@pytest.fixture
def without_pyddm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``import pyddm`` fail the way it does on a machine without the extra."""

    monkeypatch.setitem(sys.modules, "pyddm", None)


@pytest.fixture
def wrong_pyddm_series(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ModuleType("pyddm")
    module.__version__ = "0.8.1"
    monkeypatch.setitem(sys.modules, "pyddm", module)


def study() -> Study:
    return Study(
        {
            "subject": ["m1"] * 4,
            "session": ["d1"] * 4,
            "trial": [0, 1, 2, 3],
            "session_order": [0] * 4,
            "stimulus": [-1.0, 1.0, -1.0, 1.0],
            "choice": [0, 1, 1, 1],
            "response_time": [0.4, 0.5, 0.6, 0.45],
        }
    )


def test_the_wrapper_module_imports_with_no_third_party_package_present(
    without_pyddm: None,
) -> None:
    """``import behavio`` and every numerical API must work without a wrapped solver."""

    model = PyDDMDriftDiffusion(predictors=("stimulus",))

    assert isinstance(model, BehaviourEstimator)
    assert isinstance(model, GenerativeBehaviourModel)
    assert model.model_name == "pyddm-drift-diffusion"


def test_the_signature_is_computable_without_the_extra(without_pyddm: None) -> None:
    """A fingerprint that needed the dependency could not name a fit made elsewhere."""

    signature = PyDDMDriftDiffusion(predictors=("stimulus",)).signature

    assert f"backend=pyddm:{PYDDM_SERIES}" in signature


def test_describe_answers_without_the_extra(without_pyddm: None) -> None:
    description = PyDDMDriftDiffusion(predictors=("stimulus",)).describe(study())

    assert description.parameter_names[-3:] == ("boundary", "starting_bias", "nondecision_time")
    assert not description.errors


@pytest.mark.parametrize("method", ["fit", "predict_density", "pointwise_log_prob"])
def test_every_numerical_entry_point_names_the_extra_it_needs(
    without_pyddm: None, method: str
) -> None:
    model = PyDDMDriftDiffusion(predictors=("stimulus",))
    arguments = (study(),) if method == "fit" else (study(), _fit_stub(model))

    with pytest.raises(ForeignPackageUnavailableError) as error:
        getattr(model, method)(*arguments)

    assert f"behavio[{PYDDM_EXTRA}]" in str(error.value)


def test_simulation_names_the_extra_it_needs(without_pyddm: None) -> None:
    model = PyDDMDriftDiffusion(predictors=("stimulus",))
    values = model.parameters_from_components(
        drift={"drift.intercept": 0.0, "drift.stimulus": 1.0},
        boundary=1.0,
        starting_bias=0.5,
        nondecision_time=0.1,
    )

    with pytest.raises(ForeignPackageUnavailableError, match=r"behavio\[pyddm\]"):
        model.simulate(study(), values, seed=0)


def test_an_unsupported_pyddm_series_is_refused_rather_than_used(
    wrong_pyddm_series: None,
) -> None:
    """A solver whose numerics may differ is not the same model under the same parameters."""

    model = PyDDMDriftDiffusion(predictors=("stimulus",))

    with pytest.raises(ForeignPackageUnavailableError) as error:
        model.fit(study())

    assert "0.8.1" in str(error.value)
    assert PYDDM_SERIES in str(error.value)


def _fit_stub(model: PyDDMDriftDiffusion):
    from behavio.contracts import FitDiagnostics, FitResult

    size = len(model.parameter_names)
    return FitResult(
        model_name=model.model_name,
        model_signature=model.signature,
        parameter_names=model.parameter_names,
        estimates=np.asarray([0.0, 1.0, 1.0, 0.5, 0.1]),
        standard_errors=np.zeros(size),
        covariance=np.zeros((size, size)),
        n_observations=4,
        diagnostics=FitDiagnostics.closed_form(procedure="stub", message="stub"),
    )
