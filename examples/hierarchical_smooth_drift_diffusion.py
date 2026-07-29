"""Fit population and animal-specific Wiener parameter trajectories."""

import numpy as np

from behavio import HierarchicalSmoothWienerDriftDiffusion, Study

design = Study.factorial(
    trials=50,
    subjects=tuple(f"mouse-{index}" for index in range(3)),
    sessions=3,
    columns={"stimulus": lambda rng, n_rows: rng.normal(size=n_rows)},
    seed=811,
)
model = HierarchicalSmoothWienerDriftDiffusion(
    covariates=("stimulus",),
    knots=(0.0, 2.0),
    varying_parameters=("drift.stimulus", "boundary"),
    subject_parameters=("drift.stimulus", "boundary"),
    subject_scale=0.2,
    subject_smoothness=8.0,
    smoothness=8.0,
    n_restarts=2,
    max_iterations=300,
    simulation_time_step=0.001,
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
population = model.population_trajectory(fit)
mouse = model.subject_trajectory(fit, "mouse-0")
unseen = model.subject_trajectory(fit, "new-mouse")

print("Partially pooled drift-diffusion trajectories")
print(f"fit audit: {fit.audit().status.value} {list(fit.audit().issue_codes)}")
print(f"unseen-subject policy: {fit.unseen_subject_policy}")
print("population stimulus:", np.round(population.path("drift.stimulus"), 3).tolist())
print("mouse-0 stimulus:   ", np.round(mouse.path("drift.stimulus"), 3).tolist())
print("new-mouse stimulus: ", np.round(unseen.path("drift.stimulus"), 3).tolist())
