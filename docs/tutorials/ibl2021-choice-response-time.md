# Does one decision process predict both choice and response time?

!!! warning "A worked diagnostic, not a population estimate"

    This example follows one outcome-blindly selected animal. It demonstrates a joint
    choice/response-time evidence path; it does not estimate an IBL-wide decision process.

## Scientific question

Can a stationary Wiener drift-diffusion model fitted to two late-training sessions predict
both choices and movement-onset latencies in the next untouched session? Does adding an
explicit response-contaminant component improve that forecast?

The source is the International Brain Laboratory's standardized visual two-choice task.
The experimental unit is a trial from subject `CSHL045`, the lexicographically first
eligible animal in Unspool's frozen 78-animal manifest. Subject selection reads no choices
or response times. Each of the six checksum-pinned sessions contributes at most its first
150 source rows before any outcome filter is applied.

Response time is `firstMovement_times - goCue_times`, in seconds. We retain finite
left/right-choice trials from 50 ms through 3 s. This fixed window defines the analysis
population; it is not learned from the test session.

## Prospective boundary and estimand

Endpoint positions 3 and 4 fit the model; position 5 is opened once for evaluation. The
estimand is mean held-out **joint log density per eligible trial** for the observed choice
and response time. It is not comparable to a choice-only log loss.

The naive and contaminant-aware models share a covariate-dependent drift, one boundary,
starting bias, stationary nondecision time, and fixed diffusion scale of 1. The robust
model adds a uniform 50 ms–3 s response component with a fitted mixture probability no
larger than 0.3. Posterior contaminant responsibilities remain probabilities—not automatic
trial exclusions.

<figure class="doc-figure doc-figure--wide" data-figure-kind="Literature-shaped">
  <img src="../../assets/ibl-choice-response-time.svg" alt="Four panels compare observed and posterior-predictive response times, observed and predicted conditional accuracy, model-dependent contaminant responsibilities, and held-out joint log density for naive and robust diffusion models in one IBL animal.">
  <figcaption><strong>Literature-shaped · a joint outcome needs a joint diagnostic.</strong> The display keeps physical response-time units, choice accuracy, model-dependent responsibilities, and untouched-session evidence together. One posterior-predictive draw illustrates fitted implications rather than uncertainty; one selected animal cannot establish a population process.<span class="doc-figure__meta"><strong>Unit:</strong> eligible trial from one animal · <strong>n:</strong> 153 training and 111 untouched trials · <strong>Estimand:</strong> mean held-out joint choice/response-time log density · <a href="../../reference/figure-provenance/">provenance</a></span></figcaption>
</figure>

## Result

The two training sessions contain 153 eligible trials and the untouched session contains
111. The naive model achieves mean joint log density `−0.3045`; the contaminant-aware
model achieves `−0.3429`, a difference of `−0.0385` in favour of the simpler fit. The
robust model estimates contaminant probability `0.2520`, and its held-out responsibilities
sum to `40.60` expected contaminant assignments.

That is a useful negative result. A component that improves recovery under matched
contamination does not receive a free pass on public data. Here its extra flexibility does
not improve the predeclared future-session score. The robust fit also reaches the drift
coefficient boundary and therefore carries a warning-level audit.

## What this does—and does not—support

The example shows that Unspool can preserve response-time units, declare an eligibility
window before fitting, score the complete joint observation, generate a predictive draw,
retain trialwise responsibilities, and compare models prospectively.

It does not establish that the retained trials arise from a stationary diffusion process,
that high-responsibility trials are measurement errors, or that this animal represents a
population. The validity window also conditions the target population; the likelihood is
not a model of discarded anticipatory or very late movements. A confirmatory study should
freeze the design and evaluate it across animals with a subject-level estimand.

## Reproduce it

```bash
uv run --extra ibl python -m benchmarks.ibl2021_decision_models.benchmark
uv run --group docs python -m scripts.plot_documentation_figures --skip-cell
```

The [benchmark directory](https://github.com/aeronjl/unspool/tree/main/benchmarks/ibl2021_decision_models)
versions the dataset UUIDs, checksum manifest, eligibility counts, fit audits, pointwise
held-out arrays, parameters, and random seed. See the
[IBL source study](https://doi.org/10.7554/eLife.63711) and the
[drift-diffusion method contract](../drift-diffusion.md) for context.
