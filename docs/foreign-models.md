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
`evaluate_splits`, `compare_models`, `run_parameter_recovery` and `describe()` unchanged,
keeps its dependency behind its own extra, and states its licence here before you install
it.

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
| [dynamax](https://github.com/probml/dynamax) 1.0 | — | Not wrapped | MIT | Permitted | Unpinned jax. A smoother by default — see [filtered versus smoothed](#filtered-versus-smoothed) |
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

Behavio's answer is structural rather than diplomatic: **no wrapper's dependency is ever a
Behavio dependency**, and each lives behind its own extra. A user who needs two conflicting
packages runs two environments and moves `Study` tables and evidence bundles between them,
which they can, because both are plain data. A user who needs neither pays nothing. This is
the same rule that keeps NWB, DANDI and ONE optional, applied to models.

If Behavio ever adds a wrapper whose extra conflicts with an existing one, the conflict goes
in the table above and the two extras are documented as mutually exclusive. There is no
version of this problem that a library can solve for its users by resolving harder.

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
`GenerativePosteriorBehaviourModel`. `model.bind(design)` returns a
`DesignBoundBambiRegression` that does — and since `run_parameter_recovery` already takes the
design as its second argument, binding states something true rather than working around
something false:

```python
bound = model.bind(design)
report = run_parameter_recovery(bound, design, [truth], seed=0)
```

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

### The conformance harness has no sampled entry point

[`check_behaviour_estimator`](reference/data-adapters.md) is written against
`BehaviourEstimator`: it calls `model_capabilities`, then `model.fit(study)`, then
`model.predict(study, fit)`. A `PosteriorBehaviourEstimator` has `sample` and takes a
`PosteriorResult` where the harness passes a `FitResult`, so **no sampled model can be run
through the harness without an adapter**, and that includes the two leakage checks and the
filtered/smoothed check — the ones a wrapper most needs.

`tests/test_bambi_model.py` writes that adapter and runs the harness against it. Nothing is
faked: `fit` runs the real sampler, the real convergence audit and the real `point_summary`,
and returns a `FitResult` subclass that also carries the posterior. Every check passes, and
exactly one is skipped for a stated reason — a bernoulli regression predicts no continuous
outcome, so there is no density to reconcile against the choice probabilities. The adapter
lives in the test rather than in the wrapper on purpose: the gap is in the harness.

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
  model's own choice probabilities.

And one rule that is not a helper: **do not make the wrapped package a Behavio dependency.**
Add an extra, name it in the error a user meets without it, and put its licence in the table
at the top of this page.
