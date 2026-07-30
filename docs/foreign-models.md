# Wrapped models: compatibility and licences

Behavio's contribution is falsification, not likelihoods. Prospective splitters that block
correctly, exact-design parameter and model recovery, simulation-based calibration,
sensitivity, test-retest reliability, recovery gates that block a named claim, and
content-addressed replayable evidence — none of the mature modelling packages in this
ecosystem ship those. Several of them compute a density better than Behavio can hand-roll
one. So where a maintained package already does the arithmetic, Behavio wraps it rather
than reimplementing it, and puts the falsification machinery around the result.

`behavio.foreign` is where those wrappers live. Each is a real
[`BehaviourEstimator`](reference/contracts.md): it flows through `evaluate_splits`,
`compare_models`, `run_parameter_recovery` and `describe()` unchanged, keeps its dependency
behind its own extra, and states its licence here before you install it.

## Compatibility matrix

| Package | Extra | Status | Licence | Commercial use | Notes |
| --- | --- | --- | --- | --- | --- |
| [PyDDM](https://pyddm.readthedocs.io) 0.9.x | `pyddm` | **Shipped** — `behavio.foreign.pyddm.PyDDMDriftDiffusion` | MIT | Permitted | Adaptive Navarro–Fuss first-passage density; MIT like Behavio, so installing the extra changes nothing about your obligations |
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
  through a [`DensityPrediction`](reference/data-adapters.md) instead.
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

- `behavio.adapters.sequence_layout` derives session boundaries from `Study` once and
  restores source row order exactly. `layout.join(layout.split(values)) == values`, always.
  Use it wherever a foreign package wants a list of per-sequence arrays or a `subj_idx`
  column; do not re-derive boundaries by scanning for changes in a column.
- `behavio.adapters.DensityPrediction` is the prediction type for a continuous outcome — a
  response-time density, a continuous confidence report, an *n*-accumulator race. It carries
  defective densities on an explicit grid, reports the mass a truncated grid lost rather
  than normalising it away, and interpolates at an observed value so a per-trial likelihood
  is not a function of the solver's step size.
- `behavio.adapters.check_behaviour_estimator` executes the estimator half of the
  [compatibility list](extensions.md#compatibility-tests), including the filtered/smoothed
  check above and a cross-check that an integrated `DensityPrediction` reproduces the
  model's own choice probabilities.

And one rule that is not a helper: **do not make the wrapped package a Behavio dependency.**
Add an extra, name it in the error a user meets without it, and put its licence in the table
at the top of this page.
