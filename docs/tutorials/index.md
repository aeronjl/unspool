# Worked studies

These tutorials are executable scientific narratives rather than isolated API snippets.
Each identifies its source, experimental unit, cohort rule, estimand, validation boundary,
result, and limitations.

## Cell 2025 learning strategies

Reproduce the relationship between early choice bias and later psychometric asymmetry in 30
mice from Liebana, Laffere et al. The example foregrounds exact trial exclusions and a
preserved source-session identity collision.

[Open the Cell study](cell2025-learning-trajectories.md)

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

## Can the design distinguish the models?

Compare static, smooth, GLM-HMM, and Q-learning generators under sparse and dense versions
of the same prospective design.

[Open the recovery study](model-recovery-design.md)
