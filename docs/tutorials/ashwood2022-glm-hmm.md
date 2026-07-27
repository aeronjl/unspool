# Do persistent latent strategies predict a future session?

!!! warning "Structural analogue, not reproduction"

    This worked study borrows its scientific question from Ashwood et al. (2022), but uses
    a smaller covariate set, one animal, a different session boundary, and Unspool's own
    prospective state-count procedure. It does not reproduce their paper.

## Literature-shaped question

[Ashwood et al.](https://doi.org/10.1038/s41593-021-01007-z) used a GLM-HMM to argue that
mouse choices alternate among persistent strategies and selected model structure with
held-out likelihood. Here we ask a narrower software question: after choosing the number
of latent states entirely in earlier sessions, does a GLM-HMM predict an untouched IBL
session better than a stationary history GLM?

The public source and outcome-blind subject rule are identical to the
[choice/response-time study](ibl2021-choice-response-time.md). The experimental unit is a
left/right-choice trial from `CSHL045`. Up to 150 source rows per session are fixed before
no-go trials are removed, leaving 893 eligible choices across six ordinal endpoint
windows.

## Nested prospective boundary

The comparison has two validation layers:

1. fit 2-, 3-, and 4-state GLM-HMMs on positions 0–3 and select by mean log loss on
   position 4;
2. refit only the selected state count on positions 0–4, then compare it with a stationary
   stimulus-plus-choice-history GLM on untouched position 5.

Both candidates score choice only. Within the test session, probabilities are filtered
one step ahead: an observed choice may update the state distribution used for the next
trial, but no future choice is visible before it is scored.

<figure class="doc-figure doc-figure--wide">
  <img src="../../assets/ibl-glmhmm-states.svg" alt="Four panels show filtered latent-state probabilities in the untouched session, fitted state-specific emission coefficients, the fitted transition matrix, and inner state-count selection beside outer static-GLM and GLM-HMM log losses.">
  <figcaption><strong>Latent structure stays attached to its validation boundary.</strong> State probabilities and coefficients describe the selected fit; the two loss panels show how it was selected and whether that procedure survived the untouched session.</figcaption>
</figure>

## Result

Position-4 selection log losses are `0.67037`, `0.49956`, and `0.49954` for 2, 3, and 4
states. The declared rule therefore selects four states, although the three- and
four-state results are practically tied to five decimal places.

After refitting, the selected GLM-HMM achieves test log loss `0.4447`, compared with
`0.6692` for the static GLM—an improvement of `0.2245` per trial across 150 held-out
choices. Its filtered distribution spends 93.8% of the test session in the highest
sensory-weight state and changes maximum-probability label three times.

The fit audit is a central part of the result, not a footnote. It warns about an
ill-conditioned Hessian, boundary estimates, and disagreement among converged restarts.
The evidence supports better prediction by this selected procedure for this one session;
it does not support a confident claim that four biological strategies exist.

## Interpretation boundary

State labels are canonicalized by fitted stimulus weight. The plotted filtered
probabilities condition on choices already observed in the held-out session and are
model-dependent summaries, not directly measured neural states. Near-tied state-count
selection and numerical warnings should motivate design-matched state and model recovery
before substantive interpretation.

This bounded example stakes out the intended workflow: literature supplies the question;
chronology, nested selection, untouched prediction, and recovery determine the claim.
Scaling it to the replicated cohort and adding targeted smooth-drift and learning-agent
competitors remain separate confirmatory work.

## Reproduce it

```bash
uv run --extra ibl python -m benchmarks.ibl2021_decision_models.benchmark
uv run --group docs python -m scripts.plot_documentation_figures --skip-cell
```

The [committed benchmark](https://github.com/aeronjl/unspool/tree/main/benchmarks/ibl2021_decision_models)
retains all candidate scores, restart evidence, fit audits, transition and emission
parameters, and trialwise predictive and filtered state probabilities. Read it alongside
the [GLM-HMM method contract](../glm-hmm.md) and the
[model-recovery study](model-recovery-design.md).
