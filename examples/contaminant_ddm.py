"""Fit a contaminant-aware joint choice/response-time Wiener model."""

import numpy as np

from unspool import Study, UniformResponseTimeContaminant, WienerDriftDiffusion

generator = np.random.default_rng(204)
n_trials = 800
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
    contaminant=UniformResponseTimeContaminant(time_bounds=(0.05, 3.0)),
    nondecision_time_bounds=(0.1, 0.6),
    n_restarts=3,
)
truth = model.parameters_from_components(
    drift={"drift.intercept": 0.2, "drift.stimulus": 1.2},
    boundary=1.2,
    starting_bias=0.45,
    nondecision_time=0.25,
    contaminant_probability=0.05,
)
simulation = model.simulate_with_contaminants(design, truth, seed=205)
fit = model.fit(simulation.study)

print("Contaminant-aware joint choice/response-time model")
print(f"fit audit: {fit.audit().status.value} {list(fit.audit().issue_codes)}")
print(f"generated contaminants: {int(np.sum(simulation.contaminants))}")
print(f"posterior expected count: {fit.expected_contaminant_count:.2f}")
for name, true_value, estimate in zip(
    model.parameter_names,
    truth.values(),
    fit.estimates,
    strict=True,
):
    print(f"{name:25s} truth={true_value:7.3f} estimate={estimate:7.3f}")
