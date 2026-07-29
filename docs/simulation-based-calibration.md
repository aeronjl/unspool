# Simulation-based calibration

Simulation-based calibration (SBC) asks whether an inference implementation recovers the
distribution that generated its inputs. Repeatedly draw parameters from the declared
prior, simulate one complete `Study`, run the same posterior inference users will run, and
rank each simulated truth among the retained posterior draws.

<figure class="doc-figure" data-figure-kind="Conceptual">
  <img src="assets/sbc-workflow.svg" alt="A repeated simulation-based calibration loop: draw a latent truth from the prior, simulate an observed Study, infer a labelled posterior, randomize the truth rank among posterior draws, and retain both ranks and failures.">
  <figcaption>A prior-SBC repetition tests the joint simulator–inference pipeline. The
  finite rank histogram is evidence to inspect, not an automatic certificate.</figcaption>
</figure>

This implements the rank formulation introduced by
[Talts et al.](https://arxiv.org/abs/1804.06788). The runner is backend-neutral: simulation
returns Behavio's canonical `Study`, inference returns its labelled `PosteriorResult`, and
declared test quantities connect latent truth to posterior variables.

## The workflow

The public boundary has three explicit pieces:

1. `simulator(seed)` draws from the prior predictive distribution and returns an
   `SBCSimulation(study, truth)`;
2. `inference(study, seed)` fits that simulated study and returns a `PosteriorResult`; and
3. each `SBCTestQuantity` declares which truth and posterior quantity are ranked.

For a conjugate beta–binomial check, the complete pattern is:

```python
import numpy as np

from behavio import (
    PosteriorGroup,
    PosteriorParameterQuantity,
    PosteriorResult,
    PosteriorVariable,
    SBCSimulation,
    Study,
    run_simulation_based_calibration,
)


def simulate(seed: int) -> SBCSimulation:
    rng = np.random.default_rng(seed)
    probability = rng.uniform()
    n_trials = 20
    study = Study(
        {
            "subject": ["mouse"] * n_trials,
            "session": ["session"] * n_trials,
            "trial": np.arange(n_trials),
            "session_order": np.zeros(n_trials, dtype=int),
            "choice": rng.binomial(1, probability, size=n_trials),
        }
    )
    return SBCSimulation(study, {"probability": probability})


def infer(study: Study, seed: int) -> PosteriorResult:
    rng = np.random.default_rng(seed)
    successes = int(np.sum(study["choice"]))
    failures = len(study) - successes
    values = rng.beta(1 + successes, 1 + failures, size=(2, 1_000))
    probability = PosteriorVariable(
        "probability",
        values,
        ("chain", "draw"),
        {"chain": [0, 1], "draw": np.arange(1_000)},
    )
    return PosteriorResult(
        model_name="beta-binomial",
        model_signature="beta-binomial[uniform-prior]",
        inference_library="numpy",
        inference_library_version=np.__version__,
        parameter_names=("probability",),
        groups=(PosteriorGroup("posterior", (probability,)),),
    )


report = run_simulation_based_calibration(
    simulate,
    infer,
    (PosteriorParameterQuantity("probability"),),
    repeats=500,
    seed=419,
    simulation_signature="beta-binomial-prior-predictive[v1]",
    inference_signature="beta-binomial-conjugate[v1]",
)
```

Simulation, inference, and tie-breaking receive deterministic but separate seeds. Their
human-readable signatures are retained in the report so a rank histogram cannot become
detached from the exact pipeline it evaluated.

## Inspect ranks, coverage, and failures

Each `SBCRank` retains the replicate, labelled target, simulated truth, randomized rank,
posterior draw count, posterior mean and standard deviation, central interval, and coverage
indicator. Vector parameters preserve their named posterior coordinates, for example
`bias[subject='mouse-a']`.

```python
summary = report.summary(bins=10)[0]
print(summary.mean_normalized_rank)
print(summary.interval_coverage)
print(summary.histogram_counts)
print(summary.n_replicates, summary.repeats_requested, summary.retained_fraction)
print(report.n_successful, report.n_unconverged, report.n_other_failures)
```

The raw rank is an integer from zero through $M$, inclusive, for $M$ retained posterior
draws. Exact ties are broken uniformly at random across all admissible ranks; this avoids
systematic distortion for discrete or numerically rounded quantities. `normalized_rank`
places that discrete result at the midpoint of its unit-interval cell.

The summaries are deliberately descriptive. Behavio does not turn a finite histogram or
coverage proportion into a universal pass/fail threshold.

Simulation, inference, and evaluation exceptions become immutable `SBCFailure` records.
They count against `success_rate`; they are never silently dropped or replaced by ranks.
Use `report.to_dict()` for JSON-safe archival of every rank, failure, summary, and
uniformity assessment.

## Unconverged replicates are excluded, not pooled

SBC's validity argument is conditional on the sampler being correct. A replicate whose
chains did not mix, or that diverged, contributes a rank drawn from an unknown
distribution — and the resulting histogram deviation is indistinguishable from the model
error SBC exists to detect. Every replicate is therefore audited with
[`audit_posterior`](posterior-diagnostics.md), and a replicate whose audit status is `FAIL`
is retained as an `"audit"`-stage `SBCFailure` carrying its `audit_issue_codes`, and kept
out of the ranks:

```python
for failure in report.failures:
    if failure.stage == "audit":
        print(failure.replicate, failure.audit_issue_codes)

print(report.unconverged_replicates)
```

Because the excluded replicates are counted rather than quietly dropped, a histogram built
from forty of one hundred intended replicates reports `n_replicates=40`,
`repeats_requested=100` and `retained_fraction=0.4`; it never presents as though one
hundred replicates survived. Under the default `PosteriorAuditPolicy`, divergences,
R-hat exceedance and non-finite diagnostics are errors that exclude a replicate; low bulk
or tail effective sample size and tree-depth saturation are warnings that do not.

Pass an explicit policy to change those thresholds, or `audit_policy=None` to disable the
check entirely. The choice is recorded on the report and in `to_dict()`, so a reader can
always see whether the histogram was convergence-filtered:

```python
from behavio.posterior_diagnostics import PosteriorAuditPolicy

report = run_simulation_based_calibration(
    simulate,
    infer,
    (PosteriorParameterQuantity("probability"),),
    repeats=500,
    seed=419,
    simulation_signature="beta-binomial-prior-predictive[v1]",
    inference_signature="beta-binomial-conjugate[v1]",
    audit_policy=PosteriorAuditPolicy(max_rhat=1.01, min_ess_bulk=400.0),
)
```

## Assess uniformity against its exact null

A mean normalized rank near one half is a weak diagnostic: it is blind to the symmetric
U-shape of an over-dispersed posterior and to the symmetric cap of an under-dispersed one,
which are the two commonest SBC failure modes. `report.uniformity()` therefore reports the
empirical-CDF difference against a *simultaneous* confidence band, following
[Säilynoja, Bürkner and Vehtari](https://arxiv.org/abs/2103.10522), alongside a binned
chi-square:

```python
check = report.uniformity(bins=10, confidence_level=0.95)[0]
print(check.null, check.n_posterior_draws)
print(check.max_absolute_difference, check.n_points_outside_band)
print(check.chi_square, check.chi_square_dof, check.chi_square_p_value)
```

`evaluation_points`, `ecdf_difference`, `lower_difference_band` and
`upper_difference_band` are the four aligned curves you would plot. Behavio ships no
plotting module; the arrays are the deliverable.

Three properties of the reference distribution matter:

- The null is the **exact discrete uniform** over the $M + 1$ rank cells whenever every
  retained replicate has the same posterior draw count, reported as
  `null="discrete-uniform"`. It falls back to `"continuous-uniform"` only when the draw
  counts differ and no single discrete null exists.
- The band is **simultaneous, not pointwise**. The empirical CDF is evaluated at many
  points at once, so a pointwise 95% interval applied at every point is exceeded far more
  often than 5% of the time. The reported `pointwise_level` is calibrated downwards until
  the whole curve stays inside the envelope with probability `confidence_level` under the
  null, which is simulated exactly from the multinomial law of the counts between
  consecutive evaluation points. `n_band_simulations` and `band_seed` are recorded, so
  repeated calls on one report agree.
- `chi_square_dof` and `min_expected_bin_count` are reported alongside the statistic, so a
  reader can see whether the chi-square approximation is even applicable at that bin count.

Nothing here returns a verdict. `n_points_outside_band` and `chi_square_p_value` are
evidence to weigh against the declared design, not a certificate.

## Thinning and near-independence

SBC assumes the retained draws are near-independent; an autocorrelated chain produces a
non-uniform histogram that cannot be told apart from a genuine model error. The
[Stan SBC guide](https://mc-stan.org/docs/stan-users-guide/simulation-based-calibration.html)
therefore recommends approximately independent draws. `thin=` makes an explicit stride part
of report provenance, and Behavio now records whether that stride actually worked rather
than assuming it did.

Each `SBCRank` carries `thinned_ess`, the bulk effective sample size of that target's
thinned draws with their chain structure preserved, and `relative_ess`, that value per
thinned draw. `SBCSummary` aggregates them as `mean_relative_ess` and `min_relative_ess`:

```python
print(summary.mean_relative_ess, summary.min_relative_ess)
```

A value near one means the stride achieved near-independence. A small value means the
histogram may be non-uniform because the chain is autocorrelated, not because the model is
wrong — increase `thin`, or draw more samples. This is a recorded diagnostic, never a hard
failure: it is null when ArviZ is not installed, and it never excludes a replicate on its
own.

## Choose test quantities deliberately

`PosteriorParameterQuantity("learning_rate")` is the natural default when simulation truth
and posterior variable share a name. A custom `SBCTestQuantity` can instead check a derived
quantity, prediction, or transformation. It must provide a stable `name` and `signature`,
return its simulated truth, and return one labelled posterior variable whose first axes are
`chain` and `draw`.

This choice is scientifically consequential. As
[Modrák et al.](https://arxiv.org/abs/2211.02383) show, SBC's ability to expose a bug depends
on the test quantities. Checking only population means can miss an error in individual
effects; checking only natural parameters can miss a faulty derived prediction. Declare
quantities that exercise the inferential claims the model will support.

## What SBC does—and does not—establish

| Question | Appropriate evidence |
| --- | --- |
| Does posterior inference reproduce the declared generative distribution? | Prior SBC |
| Can a fixed, realistic experimental design estimate a parameter? | Exact-design parameter recovery |
| Can the design distinguish candidate explanations? | Model recovery |
| Does the fitted model reproduce behaviourally meaningful summaries? | Posterior-predictive checks |
| Does it predict later sessions or new animals? | Prospective validation |
| Are conclusions stable to plausible priors and analysis choices? | Sensitivity analysis |

Good prior SBC validates the joint implementation under its own prior predictive
distribution. It does not show that the prior is scientifically plausible, the model fits
real animals, parameters are recoverable in the intended design, or conclusions transport
to future sessions. Those are separate Behavio evidence objects rather than conclusions
inferred from a single diagnostic.

The current PyMC hierarchical history-GLM adapter intentionally cannot be presented as an
end-to-end prior-SBC example: its intercept has flat prior semantics, so it does not define
a proper joint distribution from which SBC repetitions can be drawn. Behavio will not
invent a simulation prior merely to produce a reassuring histogram. A future explicitly
proper-prior model can plug into this runner without changing the SBC result contract.
