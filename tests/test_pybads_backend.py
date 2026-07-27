import importlib
from typing import ClassVar

import numpy as np
import pytest

import unspool.inference as inference
from unspool import (
    InferenceError,
    OptimizationBackend,
    OptimizationProblem,
    ParameterSpace,
    ParameterSpec,
    PyBADSMultistart,
    PyBADSUnavailableError,
)

pytestmark = pytest.mark.filterwarnings("ignore:Bitwise inversion.*:DeprecationWarning")


def problem(*, plausible_bounds=(-5.0, 5.0), starts=((-2.0,), (2.0,))):
    space = ParameterSpace(
        (
            ParameterSpec(
                "x",
                bounds=(-10.0, 10.0),
                plausible_bounds=plausible_bounds,
            ),
        )
    )
    return OptimizationProblem(
        parameter_space=space,
        objective=lambda vector: float((vector[0] - 1.0) ** 2),
        starts=starts,
        has_gradient=False,
    )


class FakeBADS:
    calls: ClassVar[list[dict]] = []

    def __init__(
        self,
        fun,
        x0,
        lower_bounds,
        upper_bounds,
        plausible_lower_bounds,
        plausible_upper_bounds,
        options,
    ):
        self.fun = fun
        self.x0 = np.asarray(x0)
        self.options = options
        self.function_logger = type("Logger", (), {"func_count": 3})()
        type(self).calls.append(
            {
                "x0": self.x0.copy(),
                "lower": np.asarray(lower_bounds).copy(),
                "upper": np.asarray(upper_bounds).copy(),
                "plausible_lower": np.asarray(plausible_lower_bounds).copy(),
                "plausible_upper": np.asarray(plausible_upper_bounds).copy(),
                "options": dict(options),
            }
        )

    def optimize(self):
        seed = self.options["random_seed"]
        estimate = np.asarray([0.0 if seed % 2 else 1.0])
        return {
            "x": estimate,
            "fval": self.fun(estimate),
            "success": True,
            "message": (
                "Optimization terminated: change in function value"
                if seed % 2
                else "Optimization terminated: reached maximum number of function evaluations"
            ),
            "iterations": 3,
            "func_count": 7,
        }


def test_pybads_maps_bounds_seeds_and_attempts_into_common_run(monkeypatch) -> None:
    FakeBADS.calls = []
    monkeypatch.setattr(inference, "_load_pybads", lambda: (FakeBADS, "1.0.6"))
    backend = PyBADSMultistart(
        random_seed=11,
        max_iterations=20,
        max_function_evaluations=80,
        function_tolerance=1e-5,
    )
    random_state = np.random.get_state()

    run = backend.run(problem())

    assert isinstance(backend, OptimizationBackend)
    assert run.backend == "pybads/BADS"
    assert run.backend_config == {
        "version": "1.0.6",
        "random_seed": 11,
        "max_iterations": 20,
        "max_function_evaluations": 80,
        "function_tolerance": 1e-5,
        "uncertainty_handling": False,
    }
    assert [call["options"]["random_seed"] for call in FakeBADS.calls] == [11, 12]
    assert all(call["options"]["display"] == "off" for call in FakeBADS.calls)
    np.testing.assert_array_equal(FakeBADS.calls[0]["lower"], [-10.0])
    np.testing.assert_array_equal(FakeBADS.calls[0]["upper"], [10.0])
    np.testing.assert_array_equal(FakeBADS.calls[0]["plausible_lower"], [-5.0])
    np.testing.assert_array_equal(FakeBADS.calls[0]["plausible_upper"], [5.0])
    assert run.attempts[0].converged
    assert not run.attempts[1].converged
    assert run.attempts[1].objective < run.attempts[0].objective
    assert run.selected_index == 0
    assert all(attempt.n_gradient_evaluations == 0 for attempt in run.attempts)
    assert all(np.isinf(attempt.gradient_norm) for attempt in run.attempts)
    restored_state = np.random.get_state()
    assert restored_state[0] == random_state[0]
    np.testing.assert_array_equal(restored_state[1], random_state[1])
    assert restored_state[2:] == random_state[2:]


def test_pybads_retains_backend_exceptions_as_failed_attempts(monkeypatch) -> None:
    class FailingBADS(FakeBADS):
        def optimize(self):
            if self.options["random_seed"] == 3:
                raise FloatingPointError("surrogate failed")
            return super().optimize()

    FailingBADS.calls = []
    monkeypatch.setattr(inference, "_load_pybads", lambda: (FailingBADS, "1.0.6"))

    run = PyBADSMultistart(random_seed=3).run(problem())

    assert not run.attempts[0].finite
    assert run.attempts[0].status == -1
    assert "FloatingPointError: surrogate failed" in run.attempts[0].message
    assert run.attempts[0].n_evaluations == 3
    assert run.selected_index == 1


def test_pybads_requires_explicit_plausible_bounds_and_eligible_starts(monkeypatch) -> None:
    monkeypatch.setattr(inference, "_load_pybads", lambda: (FakeBADS, "1.0.6"))
    with pytest.raises(InferenceError, match="finite plausible bounds"):
        PyBADSMultistart().run(problem(plausible_bounds=None))
    with pytest.raises(InferenceError, match="inside plausible bounds"):
        PyBADSMultistart().run(problem(starts=((-6.0,),)))


def test_pybads_missing_extra_has_actionable_error(monkeypatch) -> None:
    def missing(name):
        if name == "pybads":
            raise ImportError("missing")
        return importlib.import_module(name)

    monkeypatch.setattr(inference.importlib, "import_module", missing)

    with pytest.raises(PyBADSUnavailableError, match=r"unspool\[optimization\]"):
        PyBADSMultistart().run(problem())


def test_real_pybads_optimizes_the_same_problem_when_extra_is_installed() -> None:
    pytest.importorskip("pybads")
    backend = PyBADSMultistart(
        random_seed=7,
        max_iterations=12,
        max_function_evaluations=60,
        function_tolerance=1e-4,
    )
    one_start = problem(starts=((0.0,),))

    run = backend.run(one_start)

    assert run.selected is not None
    assert run.selected.estimate[0] == pytest.approx(1.0, abs=0.15)
    assert run.selected.n_evaluations > 0
    assert run.backend_config["version"] == "1.0.6"
