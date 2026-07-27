# Partially pooled Wiener trajectories

`HierarchicalSmoothWienerDriftDiffusion` estimates a smooth population trajectory together
with shrunken animal-specific deviations. It occupies the middle ground between treating
all animals as copies of one average animal and fitting every animal independently.

For selected parameter \(p\), animal \(s\), and knot \(k\),

\[
\theta_{spk}=\mu_{pk}+\delta_{spk}.
\]

The population path \(\mu_p\) has the same time-scaled first-difference penalty as
`SmoothWienerDriftDiffusion`. Subject deviations use

\[
\frac{1}{2\sigma_p^2}\sum_k\delta_{spk}^2
+\frac{\lambda_s}{2}\sum_{k=2}^{K}
\frac{(\delta_{spk}-\delta_{sp,k-1})^2}{u_k-u_{k-1}}.
\]

Each selected parameter may have its own natural-scale deviation size \(\sigma_p\), while
`subject_smoothness` is the shared \(\lambda_s\). The model performs a joint penalized
maximum-a-posteriori fit. It is hierarchical partial pooling, but it is not a full Bayesian
posterior sampler.

## Declaring the hierarchy

Only parameters already listed in `varying_parameters` may receive subject paths. A narrow
hypothesis keeps the optimization and interpretation tractable:

```python
from unspool import HierarchicalSmoothWienerDriftDiffusion

model = HierarchicalSmoothWienerDriftDiffusion(
    covariates=("stimulus",),
    time="session_order",
    knots=(0.0, 2.0, 4.0),
    varying_parameters=("drift.stimulus", "boundary"),
    subject_parameters=("drift.stimulus", "boundary"),
    smoothness=8.0,
    subject_parameter_scales={"drift.stimulus": 0.2, "boundary": 0.08},
    subject_smoothness=8.0,
)
```

`subject_scale` remains a backward-compatible common fallback when
`subject_parameter_scales` is omitted. A mapping must name every selected subject
parameter exactly; its values are ordered internally by `subject_parameters`, never by
mapping insertion order.

## Estimating heterogeneity from training data

Fixed scales remain the default. To estimate separate components, declare bounds and a
neutral starting value for every parameter:

```python
model = HierarchicalSmoothWienerDriftDiffusion(
    covariates=("stimulus",),
    time="session_order",
    knots=(0.0, 2.0, 4.0),
    varying_parameters=("drift.stimulus", "boundary"),
    subject_parameters=("drift.stimulus", "boundary"),
    subject_parameter_scales={"drift.stimulus": 0.15, "boundary": 0.15},
    estimate_subject_scales=True,
    subject_scale_bounds=(0.03, 0.5),
    subject_scale_uncertainty="supplemented",
)

fit = model.fit(training_study)
print(fit.subject_scale_map)
print(fit.subject_scale_standard_error_map)
print(fit.subject_scale_at_boundary_map)
```

The estimator alternates a joint path-MAP step with bounded variance-component updates.
Each update minimizes the expected normalized Gaussian-prior loss under a local
conditional Laplace approximation. This avoids treating scales as raw joint-MAP
coordinates, which would reward collapsing scales and deviations together. It is an
approximate Laplace-EM procedure, not exact marginal likelihood.

Only rows passed to `fit()` participate. Consequently, prospective split evaluation
estimates scales from each training study before scoring its held-out sessions or animals.
`scale_estimation_iterations`, `scale_estimation_converged`, and named bound flags remain
on the fit result. A bound hit means the design did not resolve heterogeneity beyond the
declared range; it is not evidence that the true variance equals the bound.

The default `subject_scale_uncertainty="local"` uses final expected-prior curvature in
log-scale coordinates. The parameter-specific recovery benchmark shows that these local
intervals are too narrow in its finite design, so they are optimization diagnostics rather
than calibrated posterior intervals.

The opt-in `"supplemented"` mode differentiates one forced EM update around the fitted
log scales and uses its rate matrix to correct the complete-data information for missing
information. This follows the supplemented EM construction of
[Meng and Rubin (1991)](https://doi.org/10.1080/01621459.1991.10475130), applied to
Unspool's approximate Laplace-EM map. The fit retains both
`subject_scale_local_standard_errors` and the selected
`subject_scale_standard_errors`, plus `subject_scale_covariance`,
`subject_scale_em_rate_matrix`, and `subject_scale_em_spectral_radius`. Reported 95%
intervals are transformed on the log scale and clipped only to the declared scale bounds.

Supplementation requires a converged scale procedure, an EM spectral radius below one,
and positive observed information. A failed condition raises `ModelDataError`; Unspool
does not manufacture a covariance by clipping eigenvalues. The pinned benchmark resolves
18/20 panels and leaves two stability failures visible, so this is a guarded finite-design
improvement rather than a universal calibration guarantee.

Population simulation parameters retain the smooth Wiener's stable natural-scale
coordinates. `simulate_with_effects()` either draws deviation paths from the configured
Gaussian precision or accepts explicit paths for recovery experiments. Realized random
effects are returned in `HierarchicalSmoothDriftDiffusionSimulation`; they are never added
to observed `Study` columns.

```python
simulation = model.simulate_with_effects(design, population_truth, seed=31)
fit = model.fit(simulation.study)

population = model.population_trajectory(fit)
mouse_path = model.subject_trajectory(fit, "mouse-03")
```

`HierarchicalSmoothDriftDiffusionFitResult` retains population estimates, every subject
deviation and local standard error, restart evidence, the common fit audit, and the
declared population policy. Arrays are read-only.

## Natural-scale constraints

Deviations are additive on the public natural scale. Effective drift, boundary, and bias
paths must therefore remain within their configured bounds. Explicit simulation paths are
rejected if they violate those bounds. Random simulation draws use bounded rejection, and
the joint optimizer uses a continuous quadratic constraint penalty while evaluating the
likelihood at the nearest admissible path. A fitted optimum outside tolerance is rejected
rather than silently clipped.

The local Hessian has an arrowhead structure: all animals couple to the population block,
but one animal's deviation block does not couple directly to another's. Unspool evaluates
the population, subject, and population–subject curvature blocks numerically and inverts
them with the Schur complement. This retains population–subject uncertainty coupling while
avoiding evaluations of known zero cross-animal blocks. It remains a local Gaussian
approximation conditional on the fitted penalties.

## Seen and unseen animals

For an animal present during fitting, prediction uses its fitted population-plus-deviation
trajectory. A completely unseen animal uses the population trajectory plug-in, recorded as
`unseen_subject_policy="population-trajectory-plugin"`. This remains the deterministic
`predict()` behavior, making generic prospective evaluation reproducible and cheap.

For a predictive distribution over new heterogeneity, use the explicit Monte Carlo API:

```python
predictive = model.predict_new_subjects(
    held_out_animals,
    fit,
    n_draws=4096,
    seed=812,
)

print(predictive.prediction.probability)
print(predictive.subject_joint_log_probability_map)
print(predictive.subject_effective_draws)
print(predictive.subject_log_probability_mcse)
```

Every draw samples one smooth deviation path per unseen animal and reuses it across all of
that animal's rows. This preserves within-animal dependence. The result distinguishes
pointwise marginal joint densities from the scientifically appropriate subject-joint
score, which takes the log only after multiplying each draw's trial densities. It also
retains marginal choice probabilities, the random-effect draws, effective draw counts,
and delta-method log-score Monte Carlo standard errors. The method rejects any animal that
appeared in the fit, preventing accidental replacement of fitted individual trajectories.

This distribution conditions on the fitted population trajectories and scale estimates;
it does not integrate their uncertainty. It is therefore empirical-Bayes random-effect
prediction, not full Bayesian posterior prediction.

Use complete-subject holdouts to test the population policy and cohort-forward session
splits to test future sessions of represented animals. These are different generalization
questions and should not be pooled under one generic cross-validation score.

## Interpretation boundary

Hierarchical DDMs are valuable because they estimate group and individual parameters
simultaneously rather than imposing either complete pooling or fully independent fits.
The original HDDM recovery experiments found the greatest benefit when individual trial
counts were small. Unspool adopts that partial-pooling motivation, not HDDM's MCMC engine;
see [Wiecki, Sofer, and Frank (2013)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3731670/).

Current limitations are explicit:

- scale estimation is empirical-Bayes Laplace-EM rather than full posterior inference;
- supplemented scale intervals remain a local numerical approximation and can be
  unresolved when the fitted EM map is unstable;
- all parameter-specific components share one path-smoothness value;
- subject deviations for stationary non-decision time are not supported;
- contaminant mixtures and within-decision time-varying dynamics are not supported;
- unseen-animal random-effect prediction does not propagate population or scale
  uncertainty;
- lab-level random effects remain future work; aligned fitted trajectories can be passed
  to the separate [cross-lab trajectory-shape contract](trajectory-shapes.md), which first
  audits independent animals per lab.

## Recovery evidence

The [hierarchical Wiener benchmark](../benchmarks/hierarchical_smooth_ddm/README.md) makes
complete pooling, shared smooth, independent smooth, and hierarchical smooth fits compete
across three regimes. Across 20 repetitions per regime, the scientifically matched model
wins both subject-path RMSE and fifth-session joint log loss: complete pooling for
stationary identical animals, shared smooth for shared change, and hierarchical smooth for
individual change. All 480 fits converge.

The [parameter-specific scale benchmark](../benchmarks/ddm_subject_scale_recovery/README.md)
starts drift and boundary components at the same value, estimates them from three training
sessions, and scores a held-out fourth session against an oracle given the true scales.
Doubling the cohort from 6 to 12 animals reduces joint scale RMSE from `0.09178` to
`0.05138`; all 16 variance procedures and final fits converge. Mean excess future-session
log loss is `0.00232` and `0.00080`, respectively. Local interval coverage is only
50–62.5%, preserving the approximation's calibration limit rather than hiding it.

The [predictive-uncertainty benchmark](../benchmarks/ddm_predictive_uncertainty/README.md)
then compares local and supplemented scale intervals over 20 eight-animal panels. Local
coverage is 70% for drift scale and 65% for boundary scale. Supplementation is stable in
18/20 panels and reaches conditional coverage of 100% and 88.9%, respectively. Across 80
entirely new animals, integrating fitted random effects improves mean subject-joint log
probability by `0.79135` and wins for 70%; effective draws and score Monte Carlo errors
remain attached to every subject.

Run the example and benchmark with:

```bash
uv run python examples/hierarchical_smooth_drift_diffusion.py
uv run python -m benchmarks.hierarchical_smooth_ddm.benchmark
uv run python -m benchmarks.ddm_subject_scale_recovery.benchmark
uv run python -m benchmarks.ddm_predictive_uncertainty.benchmark
```
