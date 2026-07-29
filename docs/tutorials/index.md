# Worked studies

These tutorials are executable scientific narratives rather than isolated API snippets.
Each identifies its source, experimental unit, cohort rule, estimand, validation boundary,
result, and limitations.

The [literature-recipe standard](recipe-contract.md) explains the common structure and the
minimum evidence required before a new example joins this list.

## Cell 2025 flagship forecast

Start with a panel-level published-parity reproduction of Figure 1G and 1I from 30 mice.
The worked audit traces the paper PDF, released notebook revision, source checksums, exact
windows, continuous colour mapping, and every intentional rendering difference.

[Reproduce Cell Figure 1G and 1I](cell2025-figure1gi-reproduction.md)

Next, replay the released Gaussian-process trajectories and soft-DTW centroids from
Figure 1H and 1J. The panel audit preserves their distinct coordinate systems, traces the
exact numerical environment, and keeps the three retrospective summaries subordinate to
the continuous animal-level paths.

[Reproduce Cell Figure 1H and 1J](cell2025-figure1hj-reproduction.md)

Then forecast each animal's final five observed sessions from its first eight days using a
completed historical cohort. The flagship chapter combines source-level provenance,
animal-balanced model comparison, exact-design recovery, and explicit unresolved claims.

[Open the Cell flagship study](cell2025-learning-trajectories.md)

## Replicated IBL learning trajectories

Construct an outcome-blind cohort of 78 animals across nine labs, then inspect change over
six ordinal endpoint windows without treating the result as an unbiased learning estimate.

[Open the IBL trajectory study](ibl2021-learning-trajectories.md)

## Prospective drift and nested selection

Ask whether session-varying individual trajectories improve prediction for represented
animals and for animals in an entirely held-out lab. Candidate smoothness is selected only
inside outer training data.

[Open the prospective IBL study](ibl2021-prospective-selection.md)

## Joint choice and response time

Fit naive and contaminant-aware Wiener drift-diffusion models to two late-training
sessions from one outcome-blindly selected IBL animal, then score both choice and
movement-onset latency in an untouched session. The negative robust-versus-naive result
and warning-level fit audit remain visible.

[Open the choice/response-time study](ibl2021-choice-response-time.md)

## Published parity: Ashwood et al. 2022 latent strategies

Reproduce the published GLM-HMM result from the paper's own public data. The cohort
reproduces exactly, and the model-derived quantities are compared against the values
printed in the paper under tolerances frozen before any fit was run. Six of fourteen
checkable claims fail, each in the direction its declared substitution predicts, and
the failures are retained rather than tuned away.

[Open the GLM-HMM parity study](ashwood2022-glm-hmm.md)

## Published parity: IBL 2021 psychometrics

Reproduce the standardised training curves and psychometric summaries from the IBL's
2021 reproducibility paper. Five of six checkable values reproduce; the number of mice
reaching proficiency does not, because a substantial part of the published cohort is
absent from the public release.

[Open the IBL parity study](ibl2021-psychometrics.md)

## Restless-bandit reinforcement learning

Fit bias, choice-history, reward-history, and Q-learning accounts on seven sessions from
32 mice, then compare animal-balanced forecasts in the untouched eighth session. A matched
recovery experiment shows where the WSLS/Q-learning distinction is identifiable while the
empirical paired interval remains unresolved.

[Open the restless-bandit study](chen2021-bandit.md)

## Can the design distinguish the models?

Compare static, smooth, GLM-HMM, and Q-learning generators under sparse and dense versions
of the same prospective design.

[Open the recovery study](model-recovery-design.md)
