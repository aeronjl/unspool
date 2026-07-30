# Partially pooled drift diffusion

A study with several animals offers two bad answers and one good one. Fitting every animal
together treats them as copies of one average animal and hides the individual differences
the experiment was run to measure. Fitting every animal separately throws away the fact
that they performed the same task, and gives the animals with the fewest trials the noisiest
parameters. Partial pooling is the middle ground: a population estimate and a shrunken
per-animal deviation from it, estimated together, with the amount of shrinkage governed by
a declared prior width rather than by a choice between the two extremes.

In Behavio this is `hierarchical()` applied to a Wiener model. It is not a class of its
own — [Composing models](composing-models.md) covers the combinator, and this page covers
what the resulting model means for a cohort of animals.

## The two shapes

```python
from behavio import WienerDriftDiffusion
from behavio.compose import hierarchical, smooth

base = WienerDriftDiffusion(predictors=("stimulus",))

# Animals differ, but nothing changes across sessions.
pooled = hierarchical(base, over="subject", parameters=("drift.stimulus", "boundary"), scale=0.2)

# Animals differ *and* the population follows a path across sessions.
paths = smooth(
    base,
    over="session_order",
    knots=(0.0, 2.0, 4.0),
    parameters=("drift.stimulus", "boundary"),
    smoothness=8.0,
    group_smoothness=8.0,
)
pooled_paths = hierarchical(
    paths, over="subject", parameters=("drift.stimulus", "boundary"), scale=0.2
)
```

The first of those never existed before the combinators: there was a hierarchical *smooth*
drift-diffusion class but no hierarchical static one, so a cohort with no longitudinal
hypothesis had to declare knots it did not believe in. It is now the ordinary case, and it
is the shape most published hierarchical DDM analyses actually have.

**Hierarchy is the outer combinator.** `hierarchical(smooth(model))` is the working order
and `smooth(hierarchical(model))` raises `TypeError`: a hierarchical estimator reports the
population coordinate while fitting a joint one whose width depends on how many animals the
study contains, and nothing outside it can expand a coordinate of unknown width.

## Shrinkage

For selected parameter \(p\), animal \(s\), and knot \(k\),

\[
\theta_{spk}=\mu_{pk}+\delta_{spk}.
\]

The population path \(\mu_p\) carries the same time-scaled first-difference penalty as any
smooth model. Animal deviations use

\[
\frac{1}{2\sigma_p^2}\sum_k\delta_{spk}^2
+\frac{\lambda_s}{2}\sum_{k=2}^{K}
\frac{(\delta_{spk}-\delta_{sp,k-1})^2}{u_k-u_{k-1}}.
\]

The first term is what shrinks: as \(\sigma_p \to 0\) every animal collapses onto the
population estimate, which is complete pooling, and as \(\sigma_p \to \infty\) the animals
separate into independent fits. Each selected parameter has its own natural-scale deviation
size \(\sigma_p\), declared with `scale=` or per parameter with `parameter_scales=`.

The second term exists because **a deviation from a path is itself a path**. \(\lambda_s\)
is the `group_smoothness` declared on the inner `smooth()` call, and it defaults to
`smoothness`. Without it, an animal's deviation would get a plain isotropic ridge — one
independent Gaussian per knot — and every animal would be free to jump between adjacent
knots at no cost, so the fitted "individual trajectories" would be noise wearing a
trajectory's clothes. A parameter that was never smoothed has a deviation that is a number
rather than a path, and gets the ordinary ridge.

The model performs a joint penalized maximum-a-posteriori fit. It is hierarchical partial
pooling, but it is not a full Bayesian posterior sampler.

## Declaring the hierarchy

Any parameter of the wrapped model may vary by animal, whether or not it follows a path. A
smooth parameter varies as a *whole path*: naming `"boundary"` names every knot of the
boundary path and gives the whole path one scale, because a partial path has no roughness
prior. A narrow hypothesis keeps the optimization and interpretation tractable:

```python
model = hierarchical(
    smooth(
        WienerDriftDiffusion(predictors=("stimulus",)),
        over="session_order",
        knots=(0.0, 2.0, 4.0),
        parameters=("drift.stimulus", "boundary"),
        smoothness=8.0,
        group_smoothness=8.0,
    ),
    over="subject",
    parameters=("drift.stimulus", "boundary"),
    parameter_scales={"drift.stimulus": 0.2, "boundary": 0.08},
)
```

`scale=` is the common fallback when `parameter_scales=` omits a parameter. `over=` is any
study column, so `over="lab"` declares a lab-level model with the same machinery — though
whether a handful of labs supports population-of-labs inference is a design question, not a
mechanical one.

## Estimating heterogeneity from training data

Fixed scales remain the default. `estimate_scale=True` with
`scale_estimator="laplace-em"` estimates one scale per named parameter from the training
rows, starting at the declared values:

```python
model = hierarchical(
    smooth(
        WienerDriftDiffusion(predictors=("stimulus",)),
        over="session_order",
        knots=(0.0, 2.0, 4.0),
        parameters=("drift.stimulus", "boundary"),
    ),
    over="subject",
    parameters=("drift.stimulus", "boundary"),
    parameter_scales={"drift.stimulus": 0.15, "boundary": 0.15},
    estimate_scale=True,
    scale_estimator="laplace-em",
    scale_bounds=(0.03, 0.5),
)

fit = model.fit(training_study)
print(fit.scale_map)
print(fit.scale_standard_error_map)
print(fit.scale_at_boundary_map)
```

The other estimator, `scale_estimator="laplace-profile"` (the default), optimises one
common multiplier on the declared scales instead. Use `"laplace-em"` when the parameters
that vary are not commensurable — a drift coefficient and a boundary separation are not.

The EM estimator alternates a joint path-MAP step with bounded variance-component updates.
Each update minimizes the expected normalized Gaussian-prior loss under a local
conditional Laplace approximation. This avoids treating scales as raw joint-MAP
coordinates, which would reward collapsing scales and deviations together. It is an
approximate Laplace-EM procedure, not exact marginal likelihood.

Only rows passed to `fit()` participate. Consequently, prospective split evaluation
estimates scales from each training study before scoring its held-out sessions or animals.
`fit.scale_estimation_iterations`, `fit.scale_estimation_converged`, and
`fit.scale_at_boundary_map` remain on the fit result. A bound hit means the design did not
resolve heterogeneity beyond the declared range; it is not evidence that the true variance
equals the bound.

The opt-in `scale_uncertainty="local"` mode uses final expected-prior curvature in
log-scale coordinates. That curvature is complete-data information, which is never smaller
than the information the marginal likelihood actually carries, so local intervals are
systematically too narrow. They are optimization diagnostics rather than calibrated
posterior intervals and are retained only for comparison.

The default `"observed"` mode subtracts the conditional variance of the complete-data
score from that curvature, which is the observed-information identity of
[Louis (1982)](https://doi.org/10.1111/j.2517-6161.1982.tb01203.x). For a Gaussian
deviation prior every term is closed form in the conditional means and covariances the
E-step already computes, so the correction adds no optimization and cannot fail on a
stability condition. A non-positive-definite corrected information raises
`ModelDataError` rather than being clipped.

The opt-in `scale_uncertainty="supplemented"` mode instead differentiates one forced EM update around the
fitted log scales and uses its rate matrix to correct the complete-data information for
missing information. This follows the supplemented EM construction of
[Meng and Rubin (1991)](https://doi.org/10.1080/01621459.1991.10475130), applied to
Behavio's approximate Laplace-EM map. The fit retains both
`fit.scale_local_standard_errors` and the selected `fit.scale_standard_errors`, plus
`fit.scale_covariance`, `fit.scale_em_rate_matrix`, and `fit.scale_em_spectral_radius`.
Reported 95% intervals are transformed on the log scale and clipped only to the declared
scale bounds.

Supplementation requires a converged scale procedure, an EM spectral radius below one,
and positive observed information. A failed condition raises `ModelDataError`; Behavio
does not manufacture a covariance by clipping eigenvalues. Because it can refuse, it is
requested rather than assumed — `hierarchical(..., scale_uncertainty="supplemented")` is
also rejected at construction unless `estimate_scale=True` and
`scale_estimator="laplace-em"` are both set, so the refusal cannot be discovered halfway
through a study. The pinned benchmark now
resolves 20/20 panels, with a maximum spectral radius of `0.89677`; the refusal path is
retained and still fires on an unstable map, but this pinned design no longer exercises
it. That is a guarded finite-design improvement rather than a universal calibration
guarantee.

## Simulation and reading a fit

Population simulation parameters are the wrapped model's own stable natural-scale
coordinates, so a recovery study compares fitted population estimates against the same
named truth it simulated from. `simulate_with_effects()` either draws deviation paths from
the configured Gaussian precision or accepts explicit `group_deviations` for recovery
experiments. Realized random effects are returned on the `HierarchicalSimulation`; they are
never added to observed `Study` columns.

```python
simulation = model.simulate_with_effects(design, population_truth, seed=31)
fit = model.fit(simulation.study)

population = model.coefficient_trajectory(fit)
mouse_path = model.group_trajectory(fit, "mouse-03")
```

`HierarchicalFitResult` retains population estimates, every animal deviation and local
standard error, the fitted scales, restart evidence, the common fit audit, and the declared
unseen-group policy. `fit.group_deviations` is `(groups, varying)`,
`fit.parameters_for("mouse-03")` is one animal's full parameter vector, and
`fit.group_was_fitted(label)` says whether an animal was in training. Arrays are read-only.
Without an inner `smooth()` there are no trajectories to read, and
`coefficient_trajectory` says so rather than inventing a clock.

## Natural-scale constraints

Deviations are additive on the public natural scale, with no link function between them and
the likelihood. Effective drift, boundary, bias and non-decision values must therefore
remain within the configured natural bounds, and that requirement couples two coordinates
rather than bounding either one: a deviation is boxed only by the *width* of its parameter's
admissible range, because anything tighter would be a prior smuggled in as a constraint.

The fit is where it is enforced. The joint optimizer evaluates the likelihood at the nearest
admissible population-plus-deviation value and prices the excursion with a continuous
quadratic penalty, so the search is pushed back inside rather than walking off the natural
scale; a fitted optimum still outside tolerance raises `ModelDataError` instead of being
silently clipped, and the boundary diagnostic inspects population-plus-deviation rather than
only the coordinates the optimizer returned.

Simulation does not enforce it. Neither a Gaussian deviation draw nor an explicit
`group_deviations` mapping is rejected for leaving the natural range, so a `scale` large
relative to a parameter's range can generate an animal whose effective boundary is
inadmissible and whose trials are degenerate. Declared scales are a modelling statement
about a cohort; check them against the parameter's bounds before simulating from them.

The local Hessian has an arrowhead structure: all animals couple to the population block,
but one animal's deviation block does not couple directly to another's. Behavio evaluates
the population, subject, and population–subject curvature blocks numerically and inverts
them with the Schur complement. This retains population–subject uncertainty coupling while
avoiding evaluations of known zero cross-animal blocks. It remains a local Gaussian
approximation conditional on the fitted penalties.

## Seen and unseen animals

For an animal present during fitting, prediction uses its fitted population-plus-deviation
parameters. A completely unseen animal uses the population plug-in, recorded as
`unseen_group_policy="population-plugin"`. This remains the deterministic `predict()`
behavior, making generic prospective evaluation reproducible and cheap. It is the *mode* of
a new animal's prior, which is not that animal's predictive distribution: a new animal is
not an average animal.

For a predictive distribution over new heterogeneity, use the explicit Monte Carlo API:

```python
predictive = model.predict_new_groups(
    held_out_animals,
    fit,
    n_draws=4096,
    seed=812,
)

print(predictive.probability)
print(predictive.group_joint_log_probability_map)
print(predictive.group_effective_draws)
print(predictive.group_log_probability_mcse)
```

Every draw samples one deviation per unseen animal — one smooth path per animal when the
wrapped model is smooth — and reuses it across all of that animal's rows. This preserves
within-animal dependence. The result distinguishes pointwise marginal joint densities from
the scientifically appropriate subject-joint score, which takes the log only after
multiplying each draw's trial densities. It also retains marginal choice probabilities, the
random-effect draws, effective draw counts, and delta-method log-score Monte Carlo standard
errors. The method rejects any animal that appeared in the fit, preventing accidental
replacement of fitted individual trajectories.

This distribution conditions on the fitted population parameters and scale estimates; it
does not integrate their uncertainty. It is therefore empirical-Bayes random-effect
prediction, not full Bayesian posterior prediction.

Use complete-subject holdouts to test the population policy and cohort-forward session
splits to test future sessions of represented animals. These are different generalization
questions and should not be pooled under one generic cross-validation score.

## Interpretation boundary

Hierarchical DDMs are valuable because they estimate group and individual parameters
simultaneously rather than imposing either complete pooling or fully independent fits.
The original HDDM recovery experiments found the greatest benefit when individual trial
counts were small. Behavio adopts that partial-pooling motivation, not HDDM's MCMC engine;
see [Wiecki, Sofer, and Frank (2013)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3731670/).

Current limitations are explicit:

- scale estimation is empirical-Bayes Laplace-EM rather than full posterior inference;
- supplemented scale intervals remain a local numerical approximation and can be
  unresolved when the fitted EM map is unstable;
- one `group_smoothness` value governs every deviation path;
- deviations are independent across parameters: there is no correlated random-effect
  covariance;
- within-decision time-varying dynamics are not supported, here or anywhere in the family;
- unseen-animal random-effect prediction does not propagate population or scale
  uncertainty;
- fitting uses a dense joint design and Hessian, so it targets moderate cohorts rather
  than thousands of animals;
- `over="lab"` is mechanically available, but cross-lab claims should still go through the
  separate [cross-lab trajectory-shape contract](trajectory-shapes.md), which first audits
  independent animals per lab.

Two limitations the hand-written class carried have gone. A stationary parameter such as
non-decision time can now carry an animal deviation, because `parameters=` is a free
declaration over the wrapped coordinate rather than a subset of the smoothed parameters.
And a contaminant weight composes like any other parameter, so a per-animal lapse rate is
an ordinary use of `hierarchical()`.

## Recovery evidence

The [hierarchical Wiener benchmark](https://github.com/aeronjl/behavio/tree/main/benchmarks/hierarchical_smooth_ddm) makes
complete pooling, shared smooth, independent smooth, and hierarchical smooth fits compete
across three regimes. Across 20 repetitions per regime, the scientifically matched model
wins both subject-path RMSE and fifth-session joint log loss: complete pooling for
stationary identical animals, shared smooth for shared change, and hierarchical smooth for
individual change. All 480 fits converge.

The [parameter-specific scale benchmark](https://github.com/aeronjl/behavio/tree/main/benchmarks/ddm_subject_scale_recovery)
starts drift and boundary components at the same value, estimates them from three training
sessions, and scores a held-out fourth session against an oracle given the true scales.
Doubling the cohort from 6 to 12 animals reduces joint scale RMSE from `0.06144` to
`0.04806`; all 16 variance procedures and final fits converge. Mean excess future-session
log loss is `0.00081` and `0.00070`, respectively. Under the default Louis
observed-information interval, coverage over the four parameter-by-cohort cells is
100%, 100%, 100%, and 87.5% against a nominal 95%.

Two limits survive that improvement and must be read alongside it. First, the corrected
interval is conservative rather than exact: its standard error runs from `0.92x` to `2.07x`
the Monte Carlo sampling spread of the estimates depending on the cell, and over-covers in
three of four cells. Second, the drift-scale point estimate is biased low — mean `0.15827` at six
animals and `0.17412` at twelve, against a truth of `0.22`, or 21–28% low in both cohorts.
That is EM/Laplace shrinkage in the point estimate, not a simulation artefact, and the
wide interval partly absorbs it. It is an open gap, not a resolved one.

The [predictive-uncertainty benchmark](https://github.com/aeronjl/behavio/tree/main/benchmarks/ddm_predictive_uncertainty)
then compares local and supplemented scale intervals over 20 eight-animal panels. Local
coverage is 50% for drift scale and 85% for boundary scale. Supplementation is stable in
all 20 panels and reaches conditional coverage of 100% for both. Across 80 entirely new
animals, integrating fitted random effects improves mean subject-joint log probability by
`0.98299` and wins for 68.75%; effective draws and score Monte Carlo errors remain
attached to every subject.

`tests/test_compose_ddm.py` additionally replays a stored reference produced by the deleted
`HierarchicalSmoothWienerDriftDiffusion` before it was removed. Simulated data and drawn
random effects are bit-for-bit equal and the joint objective agrees to one unit in the last
place at the deleted class's own optimum; the fitted estimates agree to about `1e-3`,
which is the optimizer's amplification of that last-place difference rather than a
modelling change.

Run the example and benchmark with:

```bash
uv run python examples/hierarchical_smooth_drift_diffusion.py
uv run python -m benchmarks.hierarchical_smooth_ddm.benchmark
uv run python -m benchmarks.ddm_subject_scale_recovery.benchmark
uv run python -m benchmarks.ddm_predictive_uncertainty.benchmark
```
