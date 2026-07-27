# Worked studies

These tutorials are executable scientific narratives rather than isolated API snippets.
Each identifies its source, experimental unit, cohort rule, estimand, validation boundary,
result, and limitations.

The [literature-recipe standard](recipe-contract.md) explains the common structure and the
minimum evidence required before a new example joins this list.

## Cell 2025 flagship forecast

Reproduce the public behavioural results from 30 mice in Liebana, Laffere et al., then
forecast each animal's final five sessions from its first eight days using a completed
historical cohort. The chapter combines source-level provenance, animal-balanced model
comparison, exact-design recovery, and explicit unresolved claims.

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

## Literature-shaped latent strategies

Turn the question posed by Ashwood et al. into a bounded prospective GLM-HMM example.
State count is selected in an earlier session, the selected model is compared with a
static history GLM in a later untouched session, and near-tied selection and numerical
warnings constrain interpretation.

[Open the GLM-HMM study](ashwood2022-glm-hmm.md)

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
