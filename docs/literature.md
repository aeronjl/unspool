# Literature guide

Behavio's worked examples are anchored in methodological and empirical literature, but the
library does not present citations as validation by association. Each literature-shaped
workflow must still define an estimand, pass a numerical contract, and state what was not
reproduced.

## Longitudinal strategy formation

Liebana, Laffere et al. (2025) followed individual learning trajectories and related early
choice strategy to later psychometric structure. Behavio independently reproduces the
bounded behavioural result from Figure 1 using the public trial table, and checks the
recomputed correlations against the values printed in the paper through
[`published_claims.json`](https://github.com/aeronjl/behavio/blob/main/benchmarks/cell2025/published_claims.json).
It does not yet reproduce the complete behavioural clustering or any neural analysis.

- [Worked study](tutorials/cell2025-learning-trajectories.md)
- [Source article](https://doi.org/10.1016/j.cell.2025.05.025)

## Standardized learning across laboratories

The International Brain Laboratory's standardized decision-making study supplies public
trial tables across multiple institutions. Behavio uses an outcome-blind endpoint-window
cohort to test retrieval, chronology, cross-lab structure, future-session prediction, and
training-only model selection. Conditioning on protocol transition and the finite set of
labs remain explicit limitations.

- [Trajectory study](tutorials/ibl2021-learning-trajectories.md)
- [Prospective study](tutorials/ibl2021-prospective-selection.md)
- [Source article](https://doi.org/10.7554/eLife.63711)

## Latent states, reinforcement learning, and recovery

GLM-HMMs and reinforcement-learning agents are common explanations of nonstationary choice.
Behavio makes them compete with observable history and smooth-drift accounts, then tests
whether the study design recovers the generating family. The Ashwood et al. benchmark goes
further than an analogue: it recomputes the paper's own published values from the paper's
own public data, under tolerances frozen before any fit was run. The cohort reproduces
exactly and eight of fourteen checkable claims pass; the six that fail are retained, each
moving in the direction its declared modelling substitution predicts. It does not claim
that a latent state or learning rate is mechanistically identified.

- [Ashwood 2022 GLM-HMM parity study](tutorials/ashwood2022-glm-hmm.md)
- [IBL 2021 psychometrics parity study](tutorials/ibl2021-psychometrics.md)
- [Recovery study](tutorials/model-recovery-design.md)
- [GLM-HMM assumptions](glm-hmm.md)
- [Q-learning assumptions](q-learning.md)
- [Ashwood et al. source article](https://doi.org/10.1038/s41593-021-01007-z)

Covariate-dependent transitions follow the non-homogeneous HMM convention: each source
state's next-state probabilities are a multinomial-logit function of exogenous covariates.
Subject heterogeneity in transition dynamics is represented by Gaussian effects on the
complete transition regression, consistent with mixed/multilevel HMM practice. Behavio uses
isometric log-ratio coordinates so that an isotropic penalty is unchanged by the arbitrary
choice and ordering of destination states. This covers observed transition drivers and
partial pooling. The separate `SessionDynamicBernoulliGLMHMM` follows the recent dynamic
learning model's Gaussian random walk for emissions and Dirichlet session deviations around
a global transition matrix. `HierarchicalSessionDynamicBernoulliGLMHMM` is an explicitly new
mixed-effects extension: a Gaussian population path plus evolving subject deviations, with
population plug-in prediction for unseen subjects. It is informed by mixed-effects HMM
practice but is not attributed to the dynamic paper.
`LabHierarchicalSessionDynamicBernoulliGLMHMM` extends that construction again with one
exchangeable evolving lab deviation shared by subjects nested in the same laboratory. Its
Gaussian random effects and higher cluster level follow mixed and multilevel HMM practice;
the exact population/lab/subject dynamic path is a Behavio specification. It requires
within-lab subject replication and uses laboratory-joint prediction for a wholly unseen lab.
This is distinct from treating observed labs as fixed contrasts. None of the dynamic classes
claims temporal smoothness of transition matrices or a causal source for lab differences.

Dynamic-path uncertainty follows incomplete-data rather than fixed-responsibility
curvature. Behavio differentiates the observed state-marginalized path objective, uses
Louis' identity and, when necessary, supplemented EM for Gaussian scale information, and
estimates transition concentration through conditional Dirichlet-multinomial evidence.
Because HMM posteriors are invariant to state permutations, the optimized models' labelled
intervals are explicitly conditional on one whole-path modal labelling rather than averaged
across labels. Those intervals are local empirical-Bayes approximations.

`PyMCBernoulliGLMHMM` is the distinct full-posterior route. It follows the standard HMM
strategy of exactly marginalizing the finite discrete path before NUTS, uses proper priors
and non-centred Gaussian innovations, and samples the population, laboratory, subject,
session, transition, and variance-component layers supported by the wrapped model. Complete
posterior draws are relabelled together after sampling; minimum gaps and path crossings stay
in the posterior as ambiguity evidence. This implements posterior propagation on fitted
paths. It does not yet propagate new dynamic paths into unseen sessions, subjects, or labs,
and a few tiny contract tests are not a calibration claim: repeated design-specific SBC is
still required.

- [Non-homogeneous HMM transition regression](https://doi.org/10.1016/j.csda.2019.106840)
- [hmmTMB covariate and random-effect HMMs](https://doi.org/10.18637/jss.v114.i05)
- [Mixed HMMs for longitudinal processes](https://doi.org/10.1198/016214506000001086)
- [Bayesian multilevel mixed HMM](https://doi.org/10.1002/sim.6039)
- [Mixed non-homogeneous HMM](https://pubmed.ncbi.nlm.nih.gov/22302505/)
- [Dynamic GLM-HMM learning study](https://pmc.ncbi.nlm.nih.gov/articles/PMC11623682/)
- [IBL standardized multi-lab behaviour](https://doi.org/10.7554/eLife.63711)
- [Random-lab replicability assessment](https://doi.org/10.1371/journal.pbio.3002082)
- [Isometric log-ratio coordinates](https://doi.org/10.1023/A:1023818214614)
- [Observed information for EM](https://doi.org/10.1111/j.2517-6161.1982.tb01203.x)
- [Supplemented EM](https://doi.org/10.1080/01621459.1991.10475130)
- [Dirichlet concentration estimation](https://tminka.github.io/papers/dirichlet/)
- [Label switching in mixture and HMM posteriors](https://doi.org/10.18637/jss.v069.c01)
- [Forward marginalization for finite HMMs](https://mc-stan.org/docs/functions-reference/hidden_markov_models.html)
- [Post-sampling relabeling rather than naive posterior means](https://doi.org/10.1111/1467-9868.00265)
- [Non-centred hierarchical parameterizations](https://doi.org/10.1214/088342307000000014)
- [The No-U-Turn Sampler](https://www.jmlr.org/papers/v15/hoffman14a.html)

## Choice and response time

Diffusion models turn accuracy and latency into one joint predictive claim. Behavio's
public IBL example declares movement-onset response-time units and eligibility before
fitting, compares naive and contaminant-aware accounts prospectively, and retains the
negative result rather than treating robustification as automatically superior.

- [Choice/response-time study](tutorials/ibl2021-choice-response-time.md)
- [Drift-diffusion assumptions](drift-diffusion.md)

## Calibrating an inference implementation

Simulation-based calibration tests a complete prior-simulation and posterior-inference
pipeline by asking whether simulated truths have uniform randomized ranks among posterior
draws. Behavio follows Talts et al.'s rank formulation, preserves the failed repetitions,
and makes the tested quantities explicit because Modrák et al. show that diagnostic
sensitivity depends on that choice. This checks computational faithfulness under the
declared generative distribution; it is not an empirical model check.

- [Simulation-based calibration guide](simulation-based-calibration.md)
- [Talts et al. (2018)](https://arxiv.org/abs/1804.06788)
- [Modrák et al. (2023)](https://arxiv.org/abs/2211.02383)

`PyMCBinaryQLearning` is the first first-party sampled estimator whose every free parameter
has a normalized prior and whose simulator draws the complete prior joint. Its filtered
sequential likelihood follows the standard Q-learning construction used in Bayesian
behavioural modelling; its SBC route tests the implementation, not the psychological truth
of Q-learning or the universal suitability of its default priors.

- [hBayesDM](https://doi.org/10.1162/CPSY_a_00002)
- [PyMC reinforcement-learning example](https://www.pymc.io/projects/examples/en/latest/case_studies/reinforcement_learning.html)

## Censoring and categorical calibration

Right-censored likelihood contributions use the survival/CCDF at the observation limit,
not the event density there. `CensoredDensityPrediction` retains that quantity explicitly
and `PatchLeaving` is the first model to emit it.

- [Stan survival-model guide](https://mc-stan.org/docs/stan-users-guide/survival.html)

Multiclass calibration is not one scalar property. Behavio retains pooled confidence
calibration, top-label calibration conditional on the predicted label, and one-vs-rest
classwise calibration separately. Fixed-bin ECE is retained as a descriptive summary with
its populated bins, not treated as a unique or unbiased calibration estimand.

- [Vaicenavicius et al. (2019)](https://proceedings.mlr.press/v89/vaicenavicius19a.html)
- [Gupta and Ramdas (2021)](https://arxiv.org/abs/2107.08353)
- [Guo et al. (2017)](https://proceedings.mlr.press/v70/guo17a.html)

Sensitivity to prior, likelihood, preprocessing, and model choices is a distinct workflow
stage. Behavio's first contract uses explicit exact refits and common scalar summaries; it
can represent a small targeted sensitivity analysis or a larger multiverse without
pretending that every possible fork is equally defensible. Power-scaling methods are a
complementary efficient diagnostic, not silently substituted for those refits.

- [Analysis sensitivity guide](sensitivity-analysis.md)
- [Schad, Betancourt, and Vasishth (2021)](https://doi.org/10.1037/met0000275)
- [Kallioinen et al. (2024)](https://doi.org/10.1007/s11222-023-10366-5)

## Reliability of individual computational measures

Stable group effects do not imply stable individual differences. Computational parameters
can show high ordering consistency but poor absolute agreement, and plug-in estimates can
confound trial-level estimation error with true between-occasion change. Behavio therefore
reports several named consistency and agreement quantities, preserves the paired subjects,
and keeps hierarchical joint reliability modelling as a distinct extension.

- [Test-retest reliability guide](test-retest-reliability.md)
- [Chen et al. (2021)](https://doi.org/10.1016/j.neuroimage.2021.118647)
- [Schaaf et al. (2024)](https://doi.org/10.3758/s13428-023-02203-4)
- [Williams et al. (2025)](https://doi.org/10.3758/s13428-025-02599-1)

## Coverage signals from the current field

Conference programmes are useful coverage audits, not evidence that a method is valid.
The [Cosyne 2025 workshops](https://www.cosyne.org/workshops-program-2025) foregrounded
multi-timescale behavioural flexibility and re-examining reinforcement learning in the
brain; the [Cosyne 2026 workshops](https://www.cosyne.org/workshops-program) included
inferring neural latent states from behaviour and learning-to-execution sequences. Those
themes reinforce a basic package need: observable history, smooth change, reward-driven
updates, and latent regimes must be easy to compare under the same longitudinal contract.
They do not justify adding a novel mechanism without a validated implementation.

For evidence-accumulation models, the multi-author
[expert task-design guide](https://doi.org/10.1177/25152459251336127) treats experimental
design, model adequacy, and parameter recovery as part of the analysis rather than
post-fit decoration. Behavio uses that as a documentation standard: response-time origin,
units, eligibility, candidate confusion, and recovery belong in each DDM recipe.

The resulting orientation layer is deliberately practical:

- [choose a model by the claim](model-choice-guide.md);
- inspect common-format [model cards](model-cards.md);
- follow the [literature-recipe standard](tutorials/recipe-contract.md); and
- preserve established implementations with the [migration](migration-guides.md) and
  [extension](extensions.md) guides.

## Documentation commitments

Future literature examples should prioritize:

1. an independently held-out smoothness confirmation after exact-design recovery;
2. a cohort-level confirmation of the public choice/response-time design; and
3. targeted smooth and learning competitors for the public latent-state design.

The first public bandit commitment is now implemented in the
[Chen et al. restless-bandit recipe](tutorials/chen2021-bandit.md): common bias, history,
and Q-learning accounts share one prospective animal-balanced comparison and exact-design
recovery contract.

These are roadmap commitments, not currently supported empirical claims.
