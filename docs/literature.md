# Literature guide

Unspool's worked examples are anchored in methodological and empirical literature, but the
library does not present citations as validation by association. Each literature-shaped
workflow must still define an estimand, pass a numerical contract, and state what was not
reproduced.

## Longitudinal strategy formation

Liebana, Laffere et al. (2025) followed individual learning trajectories and related early
choice strategy to later psychometric structure. Unspool independently reproduces the
bounded behavioural result from Figure 1 using the public trial table. It does not yet
reproduce the complete behavioural clustering or any neural analysis.

- [Worked study](tutorials/cell2025-learning-trajectories.md)
- [Source article](https://doi.org/10.1016/j.cell.2025.05.025)

## Standardized learning across laboratories

The International Brain Laboratory's standardized decision-making study supplies public
trial tables across multiple institutions. Unspool uses an outcome-blind endpoint-window
cohort to test retrieval, chronology, cross-lab structure, future-session prediction, and
training-only model selection. Conditioning on protocol transition and the finite set of
labs remain explicit limitations.

- [Trajectory study](tutorials/ibl2021-learning-trajectories.md)
- [Prospective study](tutorials/ibl2021-prospective-selection.md)
- [Source article](https://doi.org/10.7554/eLife.63711)

## Latent states, reinforcement learning, and recovery

GLM-HMMs and reinforcement-learning agents are common explanations of nonstationary choice.
Unspool makes them compete with observable history and smooth-drift accounts, then tests
whether the study design recovers the generating family. A bounded public-data analogue
of Ashwood et al. selects GLM-HMM state count in an earlier session and scores the selected
procedure in an untouched session. It does not claim that a latent state or learning rate
is mechanistically identified.

- [Literature-shaped GLM-HMM study](tutorials/ashwood2022-glm-hmm.md)
- [Recovery study](tutorials/model-recovery-design.md)
- [GLM-HMM assumptions](glm-hmm.md)
- [Q-learning assumptions](q-learning.md)
- [Ashwood et al. source article](https://doi.org/10.1038/s41593-021-01007-z)

## Choice and response time

Diffusion models turn accuracy and latency into one joint predictive claim. Unspool's
public IBL example declares movement-onset response-time units and eligibility before
fitting, compares naive and contaminant-aware accounts prospectively, and retains the
negative result rather than treating robustification as automatically superior.

- [Choice/response-time study](tutorials/ibl2021-choice-response-time.md)
- [Drift-diffusion assumptions](drift-diffusion.md)

## Documentation commitments

Future literature examples should prioritize:

1. a full prospective refactor of the Cell behavioural analysis;
2. an independently held-out smoothness confirmation after exact-design recovery;
3. a cohort-level confirmation of the public choice/response-time design; and
4. targeted smooth, history, and learning competitors for the public latent-state design.

These are roadmap commitments, not currently supported empirical claims.
