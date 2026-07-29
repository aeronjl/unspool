"""Seeds emitted by the evidence runners must be accepted by the inference backends.

The recovery, model-recovery and SBC runners emit 64-bit child seeds. The PyMC and PyBADS
backends used to reject anything at or above ``2**32``, so feeding a real emitted seed
straight into either backend raised -- the two halves of the package had never been
composed end to end. The backends now treat their seed as entropy for a
``numpy.random.SeedSequence`` and derive whatever narrow word they actually need.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_model_recovery import competing_models, recovery_design, recovery_scenarios

from behavio.inference import InferenceError, PyBADSMultistart
from behavio.model_recovery import run_model_recovery, run_model_recovery_grid
from behavio.models import BernoulliHistoryGLM
from behavio.pymc_backend import PyMCBackendError, PyMCHierarchicalGLMBackend
from behavio.recovery import run_parameter_recovery
from behavio.sbc import (
    PosteriorParameterQuantity,
    SBCSimulation,
    run_simulation_based_calibration,
)
from behavio.study import Study


def parameter_recovery_seeds() -> tuple[int, ...]:
    """Seeds emitted by ``behavio.recovery.run_parameter_recovery``."""

    model = BernoulliHistoryGLM(covariates=("stimulus",), choice_lags=0, l2=0.1)
    generator = np.random.default_rng(3)
    n_trials = 60
    design = Study(
        {
            "subject": ["a"] * n_trials,
            "session": ["s"] * n_trials,
            "trial": list(range(n_trials)),
            "session_order": [0] * n_trials,
            "stimulus": generator.normal(size=n_trials),
        }
    )
    report = run_parameter_recovery(
        model,
        design,
        [{"intercept": 0.1, "stimulus": 0.8}],
        repeats=2,
        seed=11,
    )
    return tuple(int(value) for value in report.seeds)


def model_recovery_seeds() -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Seeds emitted by the model-recovery runner and by its design grid."""

    static, smooth = competing_models(n_sessions=4)
    scenarios = recovery_scenarios(static, smooth)[:1]
    candidates = {"static": static, "smooth": smooth}
    design = recovery_design(n_sessions=4, n_trials=40)
    report = run_model_recovery(
        design,
        scenarios,
        candidates,
        repeats=1,
        seed=13,
        min_train_sessions=2,
    )
    grid = run_model_recovery_grid(
        {"small": design},
        scenarios,
        candidates,
        repeats=1,
        seed=17,
        min_train_sessions=2,
    )
    return (
        tuple(int(value) for value in report.seeds),
        tuple(int(value) for value in grid.seeds),
    )


def sbc_seeds() -> tuple[int, ...]:
    """Seeds emitted by ``behavio.sbc.run_simulation_based_calibration``."""

    def simulator(seed: int) -> SBCSimulation:
        study = Study({"subject": ["a"], "session": ["s"], "trial": [0], "session_order": [0]})
        return SBCSimulation(study, {"probability": 0.5})

    def failing_inference(study, seed):
        raise RuntimeError("stop before inference so the seeds are retained")

    report = run_simulation_based_calibration(
        simulator,
        failing_inference,
        (PosteriorParameterQuantity("probability"),),
        repeats=3,
        seed=19,
        simulation_signature="seed-composition[v1]",
        inference_signature="seed-composition[v1]",
    )
    return tuple(
        seed
        for failure in report.failures
        for seed in (failure.simulation_seed, failure.inference_seed)
    )


def emitted_seeds() -> dict[str, tuple[int, ...]]:
    recovery_seeds, grid_seeds = model_recovery_seeds()
    return {
        "run_parameter_recovery": parameter_recovery_seeds(),
        "run_model_recovery": recovery_seeds,
        "run_model_recovery_grid": grid_seeds,
        "run_simulation_based_calibration": sbc_seeds(),
    }


def test_emitted_seeds_are_accepted_by_every_inference_backend() -> None:
    emitters = emitted_seeds()

    assert emitters
    for emitter, seeds in emitters.items():
        assert seeds, emitter
        for seed in seeds:
            assert seed >= 0, emitter
            pymc = PyMCHierarchicalGLMBackend(seed=seed)
            bads = PyBADSMultistart(random_seed=seed)

            assert pymc.backend_config["seed"] == seed
            assert bads.backend_name == "pybads/BADS"


def test_emitters_still_exercise_the_range_the_backends_used_to_reject() -> None:
    """Guard the regression: these emitters really do produce 64-bit seeds."""

    emitters = emitted_seeds()
    wide = [
        seed for seeds in emitters.values() for seed in seeds if seed >= np.iinfo(np.uint32).max
    ]

    assert wide


def test_backends_still_reject_seeds_that_are_not_non_negative_integers() -> None:
    with pytest.raises(PyMCBackendError, match="non-negative integer"):
        PyMCHierarchicalGLMBackend(seed=-1)
    with pytest.raises(InferenceError, match="non-negative integer"):
        PyBADSMultistart(random_seed=-1)
    with pytest.raises(InferenceError, match="non-negative integer"):
        PyBADSMultistart(random_seed=True)
