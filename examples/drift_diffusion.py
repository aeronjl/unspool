"""Fit joint binary choices and response times with a simple Wiener model."""

import numpy as np

from unspool import Study, WienerDriftDiffusion

generator = np.random.default_rng(104)
n_trials = 600
design = Study(
    {
        "subject": ["synthetic-subject"] * n_trials,
        "session": ["session-1"] * n_trials,
        "trial": list(range(n_trials)),
        "session_order": [0] * n_trials,
        "stimulus": generator.normal(size=n_trials),
    }
)
model = WienerDriftDiffusion(
    covariates=("stimulus",),
    n_restarts=3,
    simulation_time_step=0.0005,
)
truth = model.parameters_from_components(
    drift={"drift.intercept": 0.2, "drift.stimulus": 1.2},
    boundary=1.2,
    starting_bias=0.45,
    nondecision_time=0.25,
)
study = model.simulate(design, truth, seed=105)
fit = model.fit(study)

print("Joint choice/response-time drift-diffusion model")
print(f"scored columns: {model.scored_columns}")
print(f"fit audit: {fit.audit().status.value} {list(fit.audit().issue_codes)}")
for name, true_value, estimate in zip(
    model.parameter_names,
    truth.values(),
    fit.estimates,
    strict=True,
):
    print(f"{name:20s} truth={true_value:7.3f} estimate={estimate:7.3f}")
