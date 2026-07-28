# How to read a model evidence panel

A fitted curve is not a complete scientific result. Unspool's reporting target is a chain
of evidence that lets a reader move from observed behaviour to a bounded model claim
without losing the validation boundary or the design's known ambiguities.

<figure class="doc-figure doc-figure--wide" data-figure-kind="Synthetic benchmark">
  <img src="../../assets/choice-model-evidence-atlas.svg" alt="Four rows for static GLM, smooth GLM, GLM-HMM, and Q-learning, each showing observed session choice rates, fitted model-specific structure, predicted and observed choices in an untouched fifth session, calibration residuals, and dense-design model-recovery scores.">
  <figcaption><strong>Synthetic benchmark · four model families, one evidential grammar.</strong> Each row is a deterministic representative fit from the dense four-family recovery design. The panels show a single run; the recovery column comes from the committed matched benchmark and prevents the representative example from standing in for repeated evidence.<span class="doc-figure__meta"><strong>Unit:</strong> simulated study · <strong>n:</strong> 4 generating families under one 300-trial design · <strong>Estimand:</strong> fitted structure, future prediction, calibration, and family recovery · <a href="../../reference/figure-provenance/">provenance</a></span></figcaption>
</figure>

## 1. Observed behaviour

The first column shows session-level choice rates from the simulated study. It is a check
on chronology and outcome geometry, not a sufficient statistic for fitting. Trial rows,
stimuli, rewards, and history remain in the underlying `Study`.

## 2. Fitted structure

The second column deliberately changes meaning by model family:

| Family | Displayed fitted object | Scientific question |
| --- | --- | --- |
| Static GLM | Bias, stimulus, and choice-history coefficients | Is one stable decision rule sufficient? |
| Smooth GLM | Bias and stimulus paths across declared session knots | Does a continuous trajectory improve future prediction? |
| GLM-HMM | Filtered probability of one canonicalized latent state | Do persistent discrete regimes organize the choices? |
| Q-learning | Pre-choice value difference | Does reward-driven value updating explain sequential choice? |

Putting every model into the same generic “fit line” would hide what its latent structure
actually asserts. A standardized report should preserve comparability around that central,
family-specific panel rather than erase the difference.

## 3. Untouched future prediction

All four fits use sessions 1–4 and forecast session 5. Predicted probabilities are binned
only for display; likelihood comparison retains pointwise scores. Observed choices in the
future session are never used to choose or refit the displayed model.

## 4. Calibration residuals

The fourth column plots observed minus predicted choice rate in four probability groups.
Points above zero indicate underprediction and points below zero indicate overprediction.
This compact display can expose systematic miscalibration that a mean log score alone
would conceal, but sixty future trials make it descriptive rather than definitive.

## 5. Recovery in the declared design

The final column returns the model to competition. It shows the prospective mean log
probability for all four candidates when each row's family generated the dense 300-trial
design. The candidate with the highest value is selected. The matching sparse design is
less successful, so this column is evidence about this design—not a universal property of
the model family.

!!! warning "Simulation is necessary but not sufficient"

    These rows demonstrate that the software can recover interpretable structure and
    future predictions when the generator is known. They do not establish that any family
    generated an animal's behaviour. Public-data analyses still require the same
    prospective and diagnostic panels, followed by design-matched recovery.

## Coverage boundary

This standardized atlas covers the four choice-only families that participate in one
matched recovery grid. Drift-diffusion models score choice and response time jointly and
therefore need an expanded panel containing the response-time distribution, conditional
accuracy, contaminant responsibilities, and physical-unit checks. The public
[choice/response-time study](ibl2021-choice-response-time.md) now provides that extension;
it complements, rather than replaces, the repeated
[drift-diffusion recovery evidence](../drift-diffusion.md#recovery-evidence).

## Reproduce it

```bash
uv run python -m benchmarks.recovery_grid.benchmark
uv run --group docs python -m scripts.plot_documentation_figures --skip-cell
```

The plotting script refits one deterministic dense-design example per family. The recovery
scores are read from the committed benchmark result, whose seeds, candidate scores,
audits, and selection matrix are versioned independently.
