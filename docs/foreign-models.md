# Wrapped models: compatibility and licences

Behavio's contribution is falsification, not likelihoods. Prospective splitters that block
correctly, exact-design parameter and model recovery, simulation-based calibration,
sensitivity, test-retest reliability, recovery gates that block a named claim, and
content-addressed replayable evidence — none of the mature modelling packages in this
ecosystem ship those. Several of them compute a density better than Behavio can hand-roll
one. So where a maintained package already does the arithmetic, Behavio wraps it rather
than reimplementing it, and puts the falsification machinery around the result.

`behavio.foreign` is where those wrappers live. Each is a real
[`BehaviourEstimator`](reference/contracts.md) — or, when the package fits by sampling, a
real [`PosteriorBehaviourEstimator`](reference/contracts.md): it flows through
`evaluate_splits`, `run_parameter_recovery` and `describe()` unchanged, keeps its dependency
behind its own extra, and states its licence here before you install it. Three are shipped:
**PyDDM** for the first-passage density, **Bambi** for mixed-effects regression behind the
sampled contract, and **dynamax** for latent-state models of a continuous measurement. Each
reported something the contract could not yet express, and those reports are kept on this
page rather than smoothed over.

## Compatibility matrix

| Package | Extra | Status | Licence | Commercial use | Notes |
| --- | --- | --- | --- | --- | --- |
| [PyDDM](https://pyddm.readthedocs.io) 0.9.x | `pyddm` | **Shipped** — `behavio.foreign.pyddm.PyDDMDriftDiffusion` | MIT | Permitted | Adaptive Navarro–Fuss first-passage density; MIT like Behavio, so installing the extra changes nothing about your obligations |
| [Bambi](https://bambinos.github.io/bambi/) 0.17.x / 0.19.x | `bambi` | **Shipped** — `behavio.foreign.bambi.BambiRegression` | MIT | Permitted | PyMC-backed mixed-effects regression, behind the *sampled* contract. MIT, and so is every package in its transitive closure. Two series because Bambi 0.18 raised its Python floor to 3.12 — see [the Python floor](#the-python-floor) |
| [HSSM](https://lnccbrown.github.io/HSSM/) 0.4 | — | Not wrapped | **Brown University, all rights reserved**; use granted "for any purpose other than its incorporation into a commercial product or service" | **Prohibited** | Also wants a modern jax (its CUDA extras pin `jax>=0.7.0`); see the conflict below |
| [hBayesDM](https://github.com/CCS-Lab/hBayesDM) 1.1 | — | Not wrapped | **GPL-3.0** | Permitted, but copyleft | Linking a GPL library into a distributed work makes that work GPL |
| [cpm-toolbox](https://github.com/DevComPsy/cpm) 0.25 | — | Not wrapped | **AGPL-3.0** | Permitted, but network copyleft | AGPL reaches hosted services, not only distributed binaries. The unrelated PyPI package named `cpm` is MIT and is not this |
| [keypoint-moseq](https://keypoint-moseq.readthedocs.io) 0.6 | — | Not wrapped (its *outputs* are read by `behavio.observed`) | **Harvard academic-use-only**; no licence metadata is published on PyPI, so read the repository `LICENSE.md` | **Prohibited** | Behavio reads Keypoint-MoSeq result files; it never imports the package |
| [psignifit](https://github.com/wichmann-lab/python-psignifit) 4.3 | — | Not wrapped | **GPL-3.0** | Permitted, but copyleft | Behavio's own `Psychometric` covers the same ground under MIT |
| [metadpy](https://github.com/LegrandNico/metadpy) 0.1 | — | Not wrapped | **GPL-3.0** | Permitted, but copyleft | Behavio's own `MetaSDT` covers the same ground under MIT |
| [pyhgf](https://github.com/ilabcode/pyhgf) 0.3 | — | Not wrapped | **GPL-3.0** | Permitted, but copyleft | Pins `jax>=0.4.26,<0.4.32`; see the conflict below |
| [dynamax](https://github.com/probml/dynamax) 1.0.x | `dynamax` | **Shipped** — `behavio.foreign.dynamax.DynamaxSwitchingAutoregression` | MIT | Permitted | Switching linear autoregression and Gaussian-emission HMM. MIT, and permissive throughout its closure — but it depends on `tfp-nightly` rather than a released TensorFlow Probability, and it brings a modern jax. See [the jax conflict](#the-jax-conflict) and [the nightly dependency](#the-nightly-dependency) |
| [ssm](https://github.com/lindermanlab/ssm) | — | Not wrapped | MIT | Permitted | Installed from GitHub; the PyPI name `ssm` is an unrelated placeholder. A smoother by default |

"Not wrapped" means exactly that: no import, no extra, no dependency. The rows are here so
that the licence position is stated for the packages a user is most likely to reach for
next, not because anything is pending. Licences were read from each project's published
metadata; a licence can change between releases, so the version each row was checked at is
named and the project's own `LICENSE` is authoritative.

## Licence policy

Behavio is MIT. A user who runs `pip install behavio[...]` should not discover afterwards
that their environment now contains a non-commercial or copyleft library, so:

1. **Every extra states its licence.** Where an extra is not MIT, the entry in
   `pyproject.toml` carries a licence notice beside it and this page repeats it. Today every
   optional dependency Behavio declares is permissively licensed — MIT, BSD, Apache-2.0 or
   HDF-Group-style — and the notice block in `pyproject.toml` says so extra by extra, so
   that stays a checked fact rather than an assumption.
2. **A non-MIT package is not wrapped without an explicit notice.** A wrapper whose
   dependency is GPL, AGPL, non-commercial or academic-only must name the licence in its
   module docstring, in its extra, and in this table, and installing that extra must be a
   separate decision from installing Behavio.
3. **Reading a file format is not using a package.** Behavio reads DeepLabCut, SLEAP,
   Keypoint-MoSeq and BORIS *outputs* with its own code. A file format carries no licence
   obligation; importing the tool that wrote it would.
4. **Nothing non-MIT is ever a core dependency.** The core is NumPy and SciPy.

## The jax conflict

`pyhgf` 0.3 pins `jax>=0.4.26,<0.4.32`. HSSM 0.4 wants a modern jax — its accelerator
extras pin `jax>=0.7.0` and its ONNX runtime dependency follows suit. Those ranges do not
intersect, so the two cannot be installed into one environment at any version of either, and
no resolver setting will fix it. Anything else in the ecosystem that pins jax joins the same
fight.

**`behavio[dynamax]` is now on the modern side of that line.** dynamax declares an unpinned
`jax`, which resolves to a current release (0.11 at the time of writing), so
`behavio[dynamax]` and `pyhgf` cannot share an environment. dynamax and HSSM want the same
side of the split and would resolve together; neither can be installed beside `pyhgf`. The
extras are not declared mutually exclusive in `pyproject.toml`, because Behavio does not
depend on `pyhgf` and cannot: this is a conflict between an extra of Behavio's and a package
a *user* may also want, and the honest form of it is a documented statement rather than a
resolver constraint.

One further consequence of the extra is a **process-wide side effect**, and it is stated
because it cannot be scoped: `behavio.foreign.dynamax` switches jax's 64-bit mode on the
first time it is used. jax defaults to 32-bit floats, a forward–backward pass accumulates
log probabilities badly in that precision over a few hundred trials, and the
observed-information Hessian the wrapper differentiates out of the marginal likelihood is not
meaningfully computable in it. jax 0.11 removed the `enable_x64` context manager, so there is
no scoped form of the switch left. Nothing else in Behavio uses jax, so the only code
affected is the caller's own.

Behavio's answer to the general problem is structural rather than diplomatic: **no wrapper's
dependency is ever a Behavio dependency**, and each lives behind its own extra. A user who
needs two conflicting packages runs two environments and moves `Study` tables and evidence
bundles between them, which they can, because both are plain data. A user who needs neither
pays nothing. This is the same rule that keeps NWB, DANDI and ONE optional, applied to
models.

If Behavio ever adds a wrapper whose extra conflicts with an existing one, the conflict goes
in the table above and the two extras are documented as mutually exclusive. There is no
version of this problem that a library can solve for its users by resolving harder.

## The nightly dependency

dynamax 1.0 depends on **`tfp-nightly`**, not on a released `tensorflow-probability`. That
is upstream's choice and it is not a licence problem — `tfp-nightly` is Apache-2.0 like the
release — but it is a reproducibility problem worth knowing about before installing the
extra:

- a nightly is a dated build. `uv.lock` pins the exact one, so a locked checkout is
  reproducible; an unlocked `pip install 'behavio[dynamax]'` resolves to whatever nightly is
  current that day.
- nightlies are not kept forever. A lock file old enough may reference a build that PyPI no
  longer serves, and relocking is then the only option.

The rest of the closure is ordinary and permissive: jax and jaxlib Apache-2.0, optax
Apache-2.0, scikit-learn BSD-3-Clause, `jaxtyping` MIT, `fastprogress` Apache-2.0. One
transitive edge is surprising and worth naming so nobody has to discover it in a lock diff:
`fastprogress` pulls in `python-fasthtml`, and with it `starlette`, `uvicorn` and a small web
stack. None of it is imported by anything Behavio calls.

## The Python floor

Behavio supports Python 3.11. Parts of this ecosystem no longer do: **Bambi 0.18 raised its
floor to 3.12 and moved to PyMC 6**, and PyMC 6 itself requires 3.12.

A marker that installs nothing below 3.12 would make `pip install 'behavio[bambi]'` a silent
no-op on the interpreter floor this package still claims, so the `bambi` extra splits the way
the `bayesian` and `probabilistic` extras already do:

| Python | Bambi | PyMC | ArviZ |
| --- | --- | --- | --- |
| 3.11 | 0.17.2–0.17.x | 5.27–5.x | 0.22–1.1 |
| ≥ 3.12 | 0.19.x | 6.x | ≥ 1.2 |

The wrapper is written against both series and the accepted set is enforced on import.
Only the 3.12 row is exercised by the default test run, because that is the interpreter this
repository pins; a 3.11 checkout runs the same suite against the older pair.

## PyDDM

```bash
uv sync --extra pyddm          # or: python -m pip install -e ".[pyddm]"
```

```python
from behavio import Study, compare_models, forward_session_splits
from behavio.foreign.pyddm import PyDDMDriftDiffusion
from behavio.models.ddm import WienerDriftDiffusion

model = PyDDMDriftDiffusion(predictors=("stimulus",))
splits = forward_session_splits(study)

report = compare_models(
    {"pyddm": model, "behavio": WienerDriftDiffusion(predictors=("stimulus",))},
    study,
    splits,
)
```

Both models take the same `predictors` or `design`, score the same `(choice,
response_time)` observation, and use the **same parameter names** — `drift.intercept`,
`drift.<predictor>`, `boundary`, `starting_bias`, `nondecision_time`. One parameter set can
be handed to either simulator, which is what makes the comparison above a comparison of
arithmetic rather than of bookkeeping.

### Why wrap it

Behavio's own Wiener density is a fixed twelve-term series with a hardcoded switch between
the small-time and large-time expansions at `t < 0.15`. PyDDM uses adaptive Navarro–Fuss
term selection with published accuracy bounds. Over the body of the distribution the two
agree to six significant figures, which the test suite asserts trial by trial; PyDDM is the
one you want when the tails or the term count start to matter.

### What the wrapper adds

PyDDM's `fit(sample) → Model` is a point estimate and a loss value. Four things the Behavio
contract needs are not in it:

- **A parameter map.** PyDDM names parameters after the objects holding them. Boundary
  separation is *twice* PyDDM's `B`; Behavio's relative start `w` is `(x0 + 1) / 2`. The
  correspondence is published as `PARAMETER_CORRESPONDENCE` and checked against Behavio's
  own density, because recovery that simulates under one meaning and refits under another
  reports a clean diagonal for the wrong quantity.
- **Interpolated per-trial densities.** PyDDM tabulates the PDF on a time grid and its own
  loss reads it by rounding each response time to the nearest grid index, which makes a
  per-trial likelihood a function of the step size. `pointwise_log_prob` interpolates
  through a [`DensityPrediction`](reference/contracts.md) instead, which is also what
  `predict()` returns: a diffusion predicts a joint distribution over which boundary and
  when, and reporting the choice probability alone would discard half of it at the point
  where a fold picks the prediction up.
- **A covariance.** The wrapper differences PyDDM's own loss at the optimum for an
  observed-information matrix, and reports `NaN` rather than a number it does not believe
  when that matrix is not positive definite.
- **A verified convergence claim.** See below.

### Where PyDDM strains the contract

These are findings, not defects in the wrapper. Each is reported by the fit rather than
smoothed over.

**PyDDM solves a condition grid, not a trial.** A `pyddm.Sample` groups trials into unique
combinations of its condition columns and solves one first-passage grid per combination.
That is ideal for a handful of contrast levels and wrong for a continuous per-trial
covariate, where the solve count is the trial count. `describe()` reports the number of
distinct design rows and `fit()` refuses above `max_conditions` (default 64) with a message
naming the count. Bin a continuous covariate, or raise the threshold and accept the cost.

**Two of the five parameters enter PyDDM as grid indices.** `OverlayNonDecision` shifts the
distribution by `int(nondecision_time / dt)` bins and `ICPointRatio` places the start on the
`dx` lattice — both truncating rather than rounding. The likelihood is therefore a *step
function* of `nondecision_time` and of `starting_bias`, with three consequences:

- `nondecision_time` is identified only to within one `time_step`. The fit declares that as
  a derived quantity, `nondecision_time_grid_quantum`.
- Truncation is a biased quantiser, so the fitted `nondecision_time` sits high in its cell.
  Behavio's own `run_parameter_recovery` is what makes that visible; the test suite bounds
  the bias at three solver steps and it shrinks with `time_step`.
- A numerical derivative taken *inside* one cell measures exactly zero, so the wrapper's
  finite differences deliberately span two cells of whichever lattice a coordinate lives on.

**PyDDM 0.9 reports no convergence flag.** `fit_adjust_model` reads SciPy's `message` off
`result.__dict__`, which an `OptimizeResult` does not populate, so the field is always
empty for differential evolution and `success` is discarded. The wrapper therefore verifies
convergence itself, as **coordinate-wise local optimality on the solver lattice**: no
single-parameter move of one difference step improves PyDDM's loss by more than
`convergence_tolerance` nats. That is weaker than "the optimizer's stopping rule fired" and
is the strongest claim available. An underpowered search fails it and the fit's audit fails
with it.

**The far tail of the analytic solution underflows to zero.** Past a few seconds PyDDM's
grid holds exact zeros where Behavio's series still returns `6e-10`. One such trial would
make a fold's score `-inf`, so scores are floored at the same constant Behavio's own Wiener
density uses and the count of floored rows is retained as `likelihood_floor_count`.

**The scoring objective is not the fitting objective.** `pointwise_log_prob` interpolates;
PyDDM's loss rounds to a grid bin. The two are different functions of the same density, so
they disagree by a fraction of a nat over a few hundred trials. The gap is reported as
`interpolation_gap` on the fit and shrinks when `time_step` does. Read it before treating
the pointwise scores as the objective that produced the estimate.

### Determinism

Differential evolution is stochastic. `seed` is a declared field, it is passed to SciPy, and
it is part of the model's `signature` — two fits that differ only in seed are two different
procedures and do not share a fingerprint. Fitting the same study twice with the same
configuration gives bit-identical estimates.

## Bambi

```bash
uv sync --extra bambi          # or: python -m pip install -e ".[bambi]"
```

```python
from behavio import BiasOnly, compare_models, forward_session_splits
from behavio.foreign.bambi import BambiRegression

model = BambiRegression("choice ~ stimulus + (1|subject)", chains=4, draws=1000, seed=0)

report = compare_models(
    {"bambi": model, "bias": BiasOnly()},
    study,
    forward_session_splits(study),
)
```

`BambiRegression` is a
[`PosteriorBehaviourEstimator`](reference/contracts.md), not a `BehaviourEstimator`: it has
`sample` instead of `fit`, returns a `PosteriorResult`, and declares the `point_summary`
projection that lets a sampled model enter the frequentist machinery. Every fold is sampled,
audited by `audit_posterior`, and projected; a fold whose posterior fails its convergence
audit makes the whole candidate ineligible, exactly as a non-convergent optimizer does.

### Why wrap it

Not for the formula — Behavio has [its own](design-formulas.md). For three things Behavio does not
have and is not going to grow:

- **Crossed and nested random effects.** `behavio.compose.hierarchical` pools over exactly
  one grouping. `(1|subject) + (1|stimulus_id)` and `(1|subject/session)` are Bambi's.
- **Splines and other stateful basis transforms.** `bs(x, df=5)`, `scale(x)`, `center(x)`
  are fitted on the training frame and *reused* on new rows from stored transform state,
  which is the fold boundary `AGENTS.md` demands. The test suite verifies that
  behaviourally: a test frame whose covariate is on a wildly different scale is not
  recentred onto itself.
- **A proper multi-alternative choice family.** `family="categorical"` scores a simplex over
  declared levels, through `CategoricalPrediction`.

### The formula collision

`BambiRegression.formula` is **Bambi's own string, passed through verbatim.** Behavio's
formula language does not apply to it and nothing translates between the two. The refusal is
deliberate and it is not symmetrical:

- *Downward*, a `DesignSpec` is one fixed matrix with no varying-effect representation, so
  `(1|subject)` and `bs(x, df=5)` have no image in it. Translating Bambi's formula into
  Behavio's would drop exactly the terms the wrapper exists for.
- *Upward*, Behavio's `lag()` and `kernel()` are **sequence-aware**: they read trial order
  and reset at subject and session boundaries. `formulae` has no notion of trial order, so
  `lag(choice, 1)` has no Bambi formula that means the same thing — only a materialised
  column that Behavio computes and a Bambi formula then names.

To fit history-dependent features with this wrapper, add them to the `Study` as columns
**inside the training fold** and name them in the Bambi formula. The wrapper will not do it
for you, because doing it for you would mean computing them across a fold boundary.

### Priors

`priors={"stimulus": PriorSpec.normal(0.0, 1.0)}` — Behavio's own `PriorSpec` (`normal`,
`half-normal`, `beta`, `uniform`), keyed by Bambi term name and translated to the
identically parameterised `bambi.Prior`. The correspondence is published as
`prior_correspondence()`. Everything richer is delegated to Bambi and stated rather than
approximated:

- A prior on a **group-specific** term is refused. Bambi spells one as a `Normal` whose
  `sigma` is itself a `Prior`; a flat `PriorSpec` can only express a `Normal` with a *fixed*
  sigma, which is not a weaker hierarchical prior but a model with no pooling at all.
- `bambi.Prior` objects are not accepted: they are mutable and carry no JSON-safe form, so a
  fit using one could not be written into a signature or an evidence bundle.

### Where Bambi strains the contract

**Bambi retains no grouping variable, so blocked LOO cannot run on its output.** A Bambi
`InferenceData` has `posterior`, `sample_stats`, `observed_data` and `log_likelihood` and no
`constant_data` at all, so `psis_loo(result, block="trial_subject")` has nothing to resolve
and the only reachable estimand is leave-one-*trial*-out — the one
[`behavio.posterior.loo`](reference/posterior.md) opens by calling wrong for a multi-subject
design. The wrapper writes `trial_subject`, `trial_session`, `trial_in_session` and
`trial_session_order` into `constant_data`, exactly as the first-party PyMC backend does, and
`BLOCKING_VARIABLES` names them. It also renames Bambi's `__obs__` dimension to `trial`, so a
Pareto-k target reads `choice[trial=17]`.

**Default priors are a function of the training fold.** With `auto_scale=True` (Bambi's
default and the wrapper's) the prior scales are computed from the data handed to
`bambi.Model`, so each fold of a prospective evaluation fits a slightly different model and
the signature cannot say so. `auto_scale` is a declared field, `describe()` reports the
situation as a warning, and the *realised* prior specification is recorded on every posterior
as `prior_specification`. Set `auto_scale=False` for one model across folds.

**A random effect can only be predicted for groups the fit saw.** A splitter that holds
subjects fixed lets `(1|subject)` predict out of fold; `(1|session)` cannot, because the test
fold's sessions are new levels whose offsets were never estimated. The wrapper refuses by
default, naming the unseen levels; `sample_new_groups=True` draws them from the hyperprior
instead, which is a prior-predictive claim about a new group rather than a held-out
prediction about a known one.

**A subject-level random effect needs the cohort splitter.** `forward_session_splits` builds
one fold *per subject*, so a `(1|subject)` model fitted inside one of them has a single group
level and its variance is not identified. Use
`cohort_forward_session_splits`, which keeps every subject in every fold.
`describe()` reports the one-level case as a `single_group_level` warning before any sampler
starts, and a fold that samples it anyway will usually fail its own R-hat gate — which is the
gate working, not the wrapper failing.

**A sampled model's parameter vector is a function of its data.** `parameter_names` and
`posterior_parameter_labels` are study-independent properties, and for a mixed-effects model
they cannot be: the coordinates of `1|subject`, the columns of `C(condition)` and the basis
of `bs(x, df=5)` are facts about the data. `BambiRegression` therefore does not claim
`GenerativePosteriorBehaviourModel`. It claims
[`DesignGenerativeBehaviourModel`](reference/contracts.md) instead — the contract's name for
"generative relative to a design" — and `model.bind(design)` returns a
`DesignBoundBambiRegression` that satisfies the generative contract in full. Since
`run_parameter_recovery` already takes the design as its second argument, it binds for you,
and binding states something true rather than working around something false:

```python
report = run_parameter_recovery(model, design, [truth], seed=0)  # binds internally
bound = model.bind(design)  # or bind explicitly
```

The capability matrix says which of the two it is: `can_simulate=False`,
`can_bind_design=True`, `can_recover_parameters=True`. This wrapper reported the gap; the
contract now has a way to express it, so the next wrapper will not reinvent the shim.

`bound.simulate` draws the outcome from **Bambi's own family**, by writing the supplied values
into a one-draw posterior and calling `predict(kind="response")`. Nothing about the family's
random variate is reimplemented, so recovery tests the inference rather than two
implementations of one likelihood.

**Bambi 0.19 cannot compute an out-of-sample log likelihood for the categorical family.**
`compute_log_likelihood(idata, data=new_rows)` raises a PyTensor broadcast error there. The
wrapper does not use that method at all: for a finite-support family the pointwise predictive
density *is* the observed entry of the draw-averaged response simplex, which is one graph
evaluation rather than two and makes `predict` and `pointwise_log_prob` the same computation.
That the result is Bambi's own likelihood is checked, per row, against the `log_likelihood`
group Bambi computed during sampling.

### The conformance harness, run against the sampler directly

This wrapper originally reported that it could not be checked at all.
[`check_behaviour_estimator`](reference/data-adapters.md) is written against
`BehaviourEstimator` — `model_capabilities`, then `model.fit(study)`, then
`model.predict(study, fit)` — while a `PosteriorBehaviourEstimator` has `sample` and takes a
`PosteriorResult`, so reaching the two leakage checks and the filtered/smoothed check
required the caller to write an adapter. `tests/test_bambi_model.py` wrote one, and that
shim was the evidence that the gap was in the harness.

The shim is gone. [`check_posterior_behaviour_estimator`](reference/data-adapters.md) samples,
audits, projects and then runs the identical check bodies with the posterior in the place a
`FitResult` occupies for an optimized model:

```python
from behavio.adapters import assert_posterior_behaviour_estimator_conforms

assert_posterior_behaviour_estimator_conforms(model.bind(study), study, require_complete=True)
```

Every check passes for `BambiRegression`, and exactly one is skipped for a stated reason — a
bernoulli regression predicts no continuous outcome, so there is no density to reconcile
against the choice probabilities. Handing it the *unbound* model is also fine: every check
runs except the simulator one, which is skipped saying that this model is generative only
relative to a design.

### What is not wrapped, and why it is Behavio's gap

`SUPPORTED_FAMILIES` is `("bernoulli", "categorical")`. The limit is Behavio's prediction
vocabulary, not Bambi's:

- `gaussian`, `t`, `gamma`, `wald` predict a continuous outcome, and reaching
  `DensityPrediction` from them would mean tabulating each family's density in the wrapper —
  re-hand-rolling exactly what wrapping a maintained package is supposed to stop.
- `poisson` and `negativebinomial` have **no shape at all**: `ModelPrediction` names nothing
  for an unbounded count, and truncating one to a categorical simplex would be a modelling
  decision disguised as a data structure.

Both are worth fixing in the contract before they are worked around in a wrapper.

## dynamax

```bash
uv sync --extra dynamax          # or: python -m pip install -e ".[dynamax]"
```

```python
from behavio import Study
from behavio.evaluate import evaluate_splits, forward_session_splits
from behavio.foreign.dynamax import DynamaxSwitchingAutoregression

model = DynamaxSwitchingAutoregression(outcome="speed", n_states=3, num_lags=1)

fit = model.fit(study)
states = model.state_probabilities(study, fit)  # predictive, filtered and smoothed
labels = model.most_likely_states(study, fit)  # Viterbi, and a description not a prediction
```

`DynamaxSwitchingAutoregression` describes one **continuous** `Study` column as a switch
between latent regimes, each with its own offset, its own autoregressive coefficients and its
own innovation variance:

$$y_t = b_k + \sum_{\ell} W_{k,\ell}\, y_{t-\ell} + \varepsilon_t, \quad
\varepsilon_t \sim \mathcal{N}(0, s_k), \quad k = z_t.$$

### Why this family

Behavio's only latent-state model is `BernoulliGLMHMM`: Bernoulli emissions, stationary
transitions, a discrete observation. Nothing here described a continuous behavioural time
series at all — running speed, pupil diameter, licking rate, a kinematic component of a pose.

A Gaussian-emission HMM is the obvious on-ramp and it is *included*: `num_lags=0` is exactly
that model. It is not what earns the wrap. The default `num_lags=1` is a **switching linear
autoregression**, and that is the one nothing else in the package can express. Behavio
already models history dependence in choice, through `lag()` and `kernel()` in [its own
formula language](design-formulas.md); it already models regime switching, through the
GLM-HMM. It has never been able to write down a *regime that is itself a dynamical system*,
which is the standard description of continuous behaviour — a mouse that is running has a
different autocorrelation from one that is grooming, not merely a different mean speed. A
Gaussian HMM forced onto that data answers with extra states whose only job is to tile the
autocorrelation.

The nesting is the point of shipping both. `num_lags=0` fixes every weight at zero, its
parameter names are a strict subset, and both are fitted by the same code through the same
contract, so the pair is a **targeted competitor** rather than two unrelated candidates —
which is what `AGENTS.md` demands before a latent state is interpreted.

### Filtered prediction versus smoothed description

This is the first model in Behavio that legitimately declares
`PredictionMode.SMOOTHED`, and it is why the wrapper exists.

| Mode | Mixing weights | What it is |
| --- | --- | --- |
| `FILTERED` (default) | `predicted_probs`, $p(z_t \mid y_{1:t-1})$ | the one-step-ahead predictive density $p(y_t \mid y_{1:t-1})$ |
| `SMOOTHED` | `smoothed_probs`, $p(z_t \mid y_{1:T})$ | a description of a recorded session, conditioned on trial *t* itself and everything after it |

Two consequences, both checked rather than asserted:

- **The filtered scores are the likelihood, decomposed.** `sum_t log p(y_t | y_{1:t-1})` is
  *exactly* the marginal log likelihood dynamax's own forward filter reports — the chain
  rule, not an approximation of it. The test suite asserts the identity to 1e-8.
- **`predicted_probs`, not `filtered_probs`.** `filtered_probs` is $p(z_t \mid y_{1:t})$,
  which is admissible under Behavio's definition of filtering — it reads nothing after *t* —
  but predicting $y_t$ from it conditions the prediction on the observation being predicted.
  It would pass the conformance harness and be worthless. It is reported separately by
  `state_probabilities()`, where it is the right answer to a different question.

### What the conformance harness actually proved

[`check_behaviour_estimator`](reference/data-adapters.md) relabels the second half of every
trial sequence, holds the fit fixed, and re-predicts. Every check passes for this wrapper,
including `smoothed-prediction-uses-future-rows` — and the same check **fails** for the same
model on a different study, which is the more informative result:

> When two states are six standard deviations apart, each observation identifies its own
> state, the backward message carries no information, and the smoothed and filtered state
> posteriors agree to machine precision. The harness then reports that a model claiming to
> smooth did not measurably use the future, and it is right: *on that data* the two
> descriptions are the same object.

That is the check being falsifiable in both directions, on one model, by changing only the
data. It also says something about how to read a `SMOOTHED` declaration: it is a claim about
a model *and* a dataset, not about a model alone. `require_complete=True` rejects a run in
which the perturbation was a no-op; it does not — and today cannot — distinguish "this model
does not smooth" from "smoothing this study gains nothing".

### Shape: `sequence_layout`, and why there is no padded tensor

[`sequence_layout`](reference/study-and-task.md) is built from `Study.chronological_indices()`, so it
cannot disagree with the package's own chronology, and `join(split(v)) == v` exactly.
Everything the wrapper hands dynamax comes from `split` and everything it hands back comes
from `join`.

The round-trip invariant is necessary and **not sufficient**, and the difference matters. It
catches a *misalignment* — a prediction concatenated in sequence order and returned as if the
study had been sorted, which is silently wrong whenever the source table is not already
chronological. It cannot catch a *contamination*, because a contaminated block has exactly
the right length and joins back perfectly. Two contaminations are live here:

**Autoregressive inputs must be built per sequence.** `compute_inputs` lags an array, and
lagging the flat study would make the first trial of one session a function of the last trial
of the previous one — across a night, or across animals. This is the concrete reason the
autoregressive family is where the layout earns its place: for `num_lags=0` only the state
chain resets at a boundary, and a wrapper that got it wrong would be wrong more quietly.

**Padding is not safe, so there is none.** `dynamax.ssm.SSM.fit_em` vmaps its E-step over an
`(n_sequences, num_timesteps, emission_dim)` batch and **takes no mask**, so zero-padding
ragged sequences feeds the forward–backward pass invented observations at an invented
emission value and returns their sufficient statistics to the M-step. Measured, in the test
suite, on four sessions of 40, 33, 26 and 19 trials whose true state offsets are one apart:
zero-padding moves a fitted emission offset by **0.79**, an autoregressive weight by 0.40 and
an innovation variance by 0.13. Behavioural sessions are ragged essentially always.

The wrapper therefore partitions the layout by length, vmaps dynamax's own `e_step` within
each partition, concatenates the sufficient statistics — which are sums over time, and so
length-independent for every component of these families — and calls dynamax's own `m_step`
once. That is `fit_em`'s loop with its batching replaced, and on an equal-length batch the
two agree to floating point, which the test suite asserts.

### Where dynamax strains the contract

**dynamax has no `fit` object, so the wrapper owns the mapping to `FitResult`.** EM is
`initialize(key)` then parameter pytrees. Estimates, a covariance, a convergence verdict, a
canonical state order and every diagnostic are the wrapper's.

**A covariance, out of EM.** dynamax reports a parameter pytree and a log-joint trace, and no
uncertainty. The wrapper differentiates the objective EM maximised — `log_prior(θ)` plus the
summed marginal log likelihood — **twice, with jax**, in dynamax's own unconstrained
coordinates, and carries the result onto the reported natural coordinates by the delta
method with the Jacobian of the constraining map. Unconstrained coordinates are not a
convenience: a simplex has no interior derivative in its natural coordinates, so a Hessian
taken there is singular by construction. When the observed information is not positive
definite — an unoccupied state, a transition probability at zero, an under-iterated run — the
covariance is `NaN` with a message, exactly as PyDDM's wrapper does.

**EM cannot fail, so "converged" had to be measured.** `fit_em` runs a fixed number of
monotone iterations; there is no stopping rule to have fired and no status to read. The
wrapper reports **exact stationarity**: the gradient norm of the log joint at the estimate,
against `gradient_tolerance`. That is stronger than any optimizer flag and it is free,
because the Hessian pass computes the gradient on the way. An under-iterated fit fails it and
the fit's audit fails with it.

**The reported covariance is singular, and that is correct.** `parameter_names` reports whole
simplexes — `initial[k]`, `transition[j->k]` — rather than reference-category logits, so a
reader of `transition[0->1]` is reading a transition probability rather than a contrast
against whichever state happened to sort last. The price is a covariance that is exactly
singular along the sum-to-one directions, which is the right variance for a quantity that
cannot move. It is also why `hessian_condition` reports the condition number of the
*unconstrained observed information*, the matrix actually inverted: the natural covariance's
condition number would be an artefact of the constraint and would raise an ill-conditioning
warning on every healthy fit.

**One emission column, because Behavio has no shape for more.** dynamax's Gaussian and
autoregressive families are multivariate, and a switching *vector* autoregression is what
most of this literature fits. `DensityPrediction` tabulates a density over one continuous
coordinate, so a two-dimensional emission has nothing to be returned as. Fitting and
pointwise scoring would work unchanged; only `predict` has nowhere to go. **That is a gap in
Behavio's prediction vocabulary**, not in dynamax, and it is the same kind of gap Bambi's
`poisson` family reported.

**`compare_models` cannot score an unlabelled density.** `compare_models` computes a Brier
column unconditionally, and a Brier score is a scoring rule for a probability. PyDDM's
density escapes this because it is *defective across the two boundaries*, so integrating the
grid yields genuine choice probabilities. A switching autoregression predicts an unlabelled
continuous density with no discrete margin at all, so `compare_models` raises
`UnscoreableByBrier` — correctly, since there is no number to report, but it also means the
**prospective comparison table is unreachable for every continuous-outcome model**.
`evaluate_splits` and the log score are unaffected, and a nested comparison against
`num_lags=0` runs through them today. This is the second gap in the contract that this
wrapper reports rather than works around.

**The predictive density is unbounded and the grid is not.** `predict` tabulates on a grid
fixed **at fit time** — derived from the training range and the fitted variances, and
retained on the fit as `outcome_grid` — because a grid derived from the study being predicted
would make an early row's reported density a function of later rows, which is exactly the
leak the conformance harness exists to catch. A held-out row outside that range loses tail
mass, which `total_mass` reports per row rather than normalising away, and `grid_truncation`
summarises on the fit. `pointwise_log_prob` is computed in closed form and never off the
grid, so no score is a function of the tabulation; `grid_log_density_gap` reports how far
the tabulation sits from the closed form on the training rows — the same number PyDDM's
wrapper calls `interpolation_gap`, and here a property of the report rather than of the
score.

### Label switching

A hidden Markov model's states are unidentified up to permutation. The package already knows
there are two different answers and that they are not interchangeable:
[`align_latent_states`](reference/latent-and-rl-models.md) aligns inferred posteriors against *simulated
truth* and so cannot run on data, and the hierarchical GLM-HMM keeps labels identified
*during* a joint fit by anchoring each group's emissions to the population's.

Neither applies to a single-group EM fit of a foreign model, so this wrapper does the third
thing, which is what `BernoulliGLMHMM` itself does for a single fit: **canonicalise
afterwards, and report how identified the canonical order is.** States are sorted by
increasing emission bias, ties broken by the rest of the emission row; the permutation is
applied to the initial distribution, to *both axes* of the transition matrix and to every
emission parameter, and is recorded as `canonical_permutation`. Because the log joint is
exactly invariant under simultaneous relabelling, the covariance is computed *after* the
permutation rather than permuted — no relabelling map is needed, where the GLM-HMM needs one
because its coordinates are reference-category logits and relabelling re-references them.

Canonicalisation makes an order; it does not make the order mean anything, and the fit says
so. `label_order_gap` is the smallest distance between adjacent canonical biases and
`label_ambiguous` is true below `label_tolerance`: two states with indistinguishable biases
have an order decided by numerical noise, and reading "state 0" as a behaviour across two
fits of the same animal is then exactly the confident nonsense a latent-state model invites.
`state_occupancy` and `low_occupancy` report the other half of the same problem.

### Determinism

`random_seed` seeds one jax key per restart, `initialisation` and `n_restarts` and
`em_iterations` are all declared fields in the `signature`, and nothing reads a global
stream. Fitting the same study twice with the same configuration gives bit-identical
estimates. The dynamax and jax *versions* are deliberately **not** in the signature — see
`DynamaxSwitchingAutoregression.signature` for why the argument is Bambi's rather than
PyDDM's — and are carried on the fit as provenance.

## Filtered versus smoothed

`ssm.most_likely_states` and `dynamax`'s smoother are smoothed by construction: the state
estimate for trial *t* is a function of the whole sequence, including trials after *t*. A
naive wrapper returns that array from `predict()` and stamps it `PredictionMode.FILTERED`,
and until recently nothing in Behavio could tell.

[`check_behaviour_estimator`](reference/data-adapters.md) can, because the distinction is
behavioural rather than structural. It relabels the observations in the second half of every
trial sequence, holds the fit fixed, and re-predicts: a filtered quantity on the first half
must be unchanged, and a quantity advertised as smoothed must not be. Point it at any
wrapper before trusting its mode declaration:

```python
from behavio.adapters import assert_behaviour_estimator_conforms

assert_behaviour_estimator_conforms(my_wrapper, small_study, require_complete=True)
```

`require_complete=True` additionally rejects a run in which the study was too short or too
uniform for the check to execute, because a skipped leakage check is not evidence of a
filtered prediction.

The dynamax wrapper is the first model in the package that exercises this check in both
directions, and doing so turned up its one honest limit: the check measures whether a
*claimed* smoothed estimate moves when the future moves, and a well-identified state-space
model on well-separated data has a smoothed estimate that does not measurably move. See
[the dynamax section](#what-the-conformance-harness-actually-proved).

## Writing your own wrapper

[Extend Behavio](extensions.md) is the contract; three helpers exist so a wrapper does not
have to re-derive what every wrapper needs.

- `behavio.trials.sequence_layout` derives session boundaries from `Study` once and
  restores source row order exactly. `layout.join(layout.split(values)) == values`, always.
  Use it wherever a foreign package wants a list of per-sequence arrays or a `subj_idx`
  column; do not re-derive boundaries by scanning for changes in a column.
- `behavio.contracts.DensityPrediction` is the prediction type for a continuous outcome — a
  response-time density, a continuous confidence report, an *n*-accumulator race. It carries
  defective densities on an explicit grid, reports the mass a truncated grid lost rather
  than normalising it away, and interpolates at an observed value so a per-trial likelihood
  is not a function of the solver's step size. Return it from `predict()`; it is one of the
  three shapes `ModelPrediction` names, and everything downstream reads it.
- `behavio.adapters.check_behaviour_estimator` executes the estimator half of the
  [compatibility list](extensions.md#compatibility-tests), including the filtered/smoothed
  check above and a cross-check that an integrated `DensityPrediction` reproduces the
  model's own choice probabilities. Its sampled counterpart,
  `behavio.adapters.check_posterior_behaviour_estimator`, runs the same checks against a
  `PosteriorBehaviourEstimator`, so a sampler needs no adapter of its own.

Inside `behavio.foreign` there is one more, `behavio.foreign._shared`, and what is *not* in
it is the interesting part. Three wrappers exist, and only two things were duplicated across
them: `quiet_foreign_package`, which all three need to silence a package's own logging, and
`ForeignCurvature`/`unknown_curvature`, which both *point-estimate* wrappers need because
neither PyDDM nor dynamax reports an uncertainty and both must be able to decline to invent
one. The Bambi wrapper nominated four of its own helpers for the same module on the
reasoning that the next wrapper would want them; the next wrapper fits by expectation
maximization, so it has no posterior groups to repair, and those four still have one user
each. They stayed where they are. Two wraps is thin evidence for an abstraction; three is
where you can see which candidates were real.

And one rule that is not a helper: **do not make the wrapped package a Behavio dependency.**
Add an extra, name it in the error a user meets without it, and put its licence in the table
at the top of this page.
