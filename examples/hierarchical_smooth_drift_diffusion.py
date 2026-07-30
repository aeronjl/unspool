"""Fit population and animal-specific Wiener parameter trajectories."""

import numpy as np

from behavio import Study, WienerDriftDiffusion
from behavio.compose import hierarchical, smooth

design = Study.factorial(
    trials=50,
    subjects=tuple(f"mouse-{index}" for index in range(3)),
    sessions=3,
    columns={"stimulus": lambda rng, n_rows: rng.normal(size=n_rows)},
    seed=811,
)
model = hierarchical(
    smooth(
        WienerDriftDiffusion(
            covariates=("stimulus",),
            n_restarts=2,
            max_iterations=300,
            simulation_time_step=0.001,
        ),
        over="session_order",
        knots=(0.0, 2.0),
        parameters=("drift.stimulus", "boundary"),
        smoothness=8.0,
        group_smoothness=8.0,
    ),
    over="subject",
    parameters=("drift.stimulus", "boundary"),
    scale=0.2,
)
truth = model.parameters_from_paths(
    {
        "drift.intercept": 0.1,
        "drift.stimulus": [0.6, 1.4],
        "boundary": [1.45, 1.1],
        "starting_bias": 0.48,
        "nondecision_time": 0.25,
    }
)
simulation = model.simulate_with_effects(design, truth, seed=812)
fit = model.fit(simulation.study)
population = model.coefficient_trajectory(fit)
mouse = model.group_trajectory(fit, "mouse-0")
unseen = model.group_trajectory(fit, "new-mouse")

print("Partially pooled drift-diffusion trajectories")
print(f"fit audit: {fit.audit().status.value} {list(fit.audit().issue_codes)}")
print(f"unseen-group policy: {fit.unseen_group_policy}")
print("population stimulus:", np.round(population.path("drift.stimulus"), 3).tolist())
print("mouse-0 stimulus:   ", np.round(mouse.path("drift.stimulus"), 3).tolist())
print("new-mouse stimulus: ", np.round(unseen.path("drift.stimulus"), 3).tolist())
