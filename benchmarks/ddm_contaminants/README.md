# Contaminant-aware drift-diffusion benchmark

This benchmark asks whether explicitly representing a small response-time contaminant
process protects Wiener parameter recovery and future-session prediction. It compares
contaminant-aware and naive fits on the same generated trials; it is not a claim that one
uniform component describes every lapse or outlier process.

## Contract

- One subject completes five 200-trial sessions with standard-normal stimulus evidence.
- The Wiener truth is drift `0.2 + 1.2 * stimulus`, boundary `1.2`, starting bias `0.45`,
  and non-decision time `0.25` seconds.
- Five percent of trials instead have an evidence-independent random choice and response
  time uniform from `0.05` to `3.0` seconds.
- The contaminant-aware model receives that fixed support and a fixed non-decision-time
  search interval before seeing data; it estimates the mixture probability jointly with
  the Wiener parameters.
- The naive model is the previous fixed-parameter Wiener family. It must place
  non-decision time below the fastest observed response because it has no alternative
  component for anticipatory trials.
- Both models fit the first 800 trials and score the same held-out fifth session. Twenty
  paired repetitions retain every seed, estimate, standard error, restart objective, fit
  audit, latent contaminant count, and posterior-responsibility summary in `result.json`.

## Result

All 20 contaminant-aware fits pass audit. Ten naive fits pass and ten retain warnings.
The contaminant-aware model has lower RMSE for every shared parameter:

| Parameter | Contaminant-aware RMSE | Naive RMSE |
| --- | ---: | ---: |
| Drift intercept | 0.09058 | 0.12867 |
| Stimulus drift | 0.07886 | 0.30749 |
| Boundary | 0.02545 | 0.44610 |
| Starting bias | 0.01485 | 0.03183 |
| Non-decision time | 0.00390 | 0.15799 |

The contaminant-probability RMSE is `0.00751`. Mean posterior contaminant responsibility
is `0.54089` on generated contaminant trials and `0.02361` on ordinary trials. These
responsibilities express uncertainty; they are not hard outlier labels.

The contaminant-aware fit also has lower future-session mean negative joint log density in
all 20 repetitions (`0.40759` versus `0.89213`; paired naive-minus-aware difference
`0.48454`). These density scores are comparable because both candidates declare the same
choice/response-time observation and physical unit.

The naive penalty is smaller than it was before the simulator's Brownian-bridge absorption
test was added: the paired forecast gap fell from `1.61255` to `0.48454`, almost entirely
because the naive model's own forecast improved from `2.06248` to `0.89213` while the
contaminant-aware forecast barely moved. The direction of every comparison is unchanged,
and the naive fit still degrades most on boundary and non-decision time.

## Interpretation boundary

The support is part of the scientific model and must be fixed from task timing, equipment
limits, or a training-only rule. Choosing it from the held-out session would leak future
information. The independent uniform component is a robustness account, not a mechanistic
theory of lapses, and its posterior responsibilities should not be treated as observed
ground truth in real data.

Ratcliff and Tuerlinckx showed that unmodelled contaminants can substantially distort
diffusion fits and advocated explicit contaminant modeling
([2002](https://doi.org/10.3758/BF03196302)). Unspool follows that methodological principle
but uses a deliberately compact, prospectively fixed joint mixture as its first executable
contract.

## Run

```bash
uv run python -m benchmarks.ddm_contaminants.benchmark
```
