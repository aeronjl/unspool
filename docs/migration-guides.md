# Migration guides

Behavio is not a drop-in replacement for every behavioural-model package. Migration means
preserving a scientific analysis while making its data semantics, prediction boundary,
diagnostics, and recovery evidence explicit. Start by reproducing the old result under an
equivalent specification; introduce new longitudinal claims only in a second step.

## The migration sequence

1. Freeze the source data, task coding, old software version, model equation, parameter
   transforms, priors or penalties, optimizer/sampler settings, and current output.
2. Represent trials as a `Study` with explicit subject, session, trial, and session order.
3. Declare the observed event in `TaskSpec`, including reward, omissions, availability,
   response-time origin, units, and reset boundaries.
4. Match the old model's likelihood and parameterization before changing validation.
5. Compare per-trial predictions or log likelihoods on a small fixed fixture—not just the
   final parameter vector.
6. Add fit or posterior diagnostics, simulation, parameter recovery, and model recovery.
7. Replace retrospective evaluation with a deployment-matched prospective split.
8. Record intended differences and retain the old artifact beside the new evidence bundle.

An inability to match pointwise predictions is a specification difference to explain, not
a tolerance to wave away.

## Hand-written SciPy likelihoods

| Existing element | Behavio destination |
| --- | --- |
| Flat arrays or DataFrame | `Study` plus `TaskSpec` |
| Ad hoc design-matrix construction | fixed `DesignSpec` terms and fold-fitted transforms |
| Natural parameter vector | `ParameterSpace` with named transforms and bounds |
| `scipy.optimize.minimize` closure | `DeterministicProblem` and a backend such as SciPy L-BFGS-B |
| Best returned vector | complete `OptimizationRun` plus `FitAudit` |
| Scalar total likelihood | finite pointwise log probabilities on labelled trials |
| One random train/test split | a session-, subject-, lab-, or historical-cohort splitter |

Keep the original likelihood callable during migration. Compare natural-to-optimizer round
trips, analytic versus finite-difference gradients, objective values at fixed parameters,
pointwise scores, and simulations under fixed seeds. A matching optimum alone can conceal
different resets, trial ordering, or outcome coding.

## `ssm` GLM-HMM workflows

The [Linderman lab `ssm` repository](https://github.com/lindermanlab/ssm) is a general state-
space toolkit supporting HMMs, input-output HMMs, and several observation families.
Behavio's `BernoulliGLMHMM` is narrower: it is a behavioural binary-choice model with
explicit session resets, state-specific input-driven emissions, stationary transitions,
filtered prospective scoring, and Behavio recovery evidence.

| `ssm` concept | Behavio concept | Migration check |
| --- | --- | --- |
| One observation/input array per sequence | one `Study` with subject/session boundaries | sequence starts equal session resets |
| `K` hidden states | `n_states` | state count selected only in training data |
| Bernoulli/input-driven observations | state-specific Bernoulli GLM emissions | covariate coding and intercept convention |
| transition parameters | stationary transition matrix and optional sticky prior | row orientation and self-transition meaning |
| `most_likely_states` | filtered or smoothed state evidence | smoothed descriptions not used as forecasts |
| raw EM traces or chosen restart | retained deterministic restarts and fit audit | every attempted optimum remains visible |

Do not compare raw state labels. Align states by their emissions or a declared assignment
rule, and rerun label-aware recovery. Behavio does not currently cover the full `ssm`
catalogue—ARHMM, HSMM, LDS, SLDS, recurrent transitions, or general observation families
should remain in `ssm` or enter through an external estimator adapter.

## hBayesDM workflows

[hBayesDM](https://doi.org/10.1162/CPSY_a_00002) provides task-specific hierarchical
Bayesian decision models through a compact R interface. Its convenience depends on each
named task fixing choices about columns, equations, priors, hierarchy, and generated
quantities. Preserve those choices explicitly when moving to Behavio.

| hBayesDM artifact | Behavio destination | Important boundary |
| --- | --- | --- |
| Task-specific input table | `Study` plus task adapter and `TaskSpec` | preserve action/reward coding and missing trials |
| Named task/model function | explicit `BinaryRLAgent` component assembly or external estimator | there may be no first-party equation match |
| Hierarchical Stan posterior | `PosteriorResult` through an adapter | retain chain/draw/subject labels and prior semantics |
| Model output summaries | posterior diagnostics, PPC, PSIS-LOO, and evidence bundle | do not reduce the posterior to a point estimate |
| Subject parameters | labelled posterior quantities and reliability workflow | transformations and shrinkage must remain explicit |

Behavio's current first-party RL models are binary and non-hierarchical. If the hBayesDM
model contains task-specific state variables, counterfactual updates, model-based planning,
or hierarchical priors, adapt the established backend to Behavio's contracts instead of
approximating it with a superficially similar agent.

## HDDM workflows

[HDDM](https://hddm.readthedocs.io/en/stable/) specializes in hierarchical Bayesian drift-
diffusion inference and related extensions. Behavio's Wiener family emphasizes explicit
longitudinal task semantics, prospective scoring, deterministic reference fits, and
design-specific recovery. These strengths are complementary.

| HDDM concept | Behavio destination | Migration check |
| --- | --- | --- |
| `subj_idx`, response, RT table | `Study` identity plus `TaskSpec.response_time` | response coding, RT origin, and time unit |
| `depends_on` or regression formula | explicit DDM covariates or fixed-knot paths | condition contrasts and link functions |
| hierarchical posterior | external estimator returning labelled `PosteriorResult` | priors, non-centering, chain/draw labels |
| outlier mixture | explicit contaminant support and probability | support and scored event must match |
| posterior predictive data | common posterior-predictive discrepancy contract | group by subject/session where relevant |

Do not convert an HDDM posterior into a deterministic Behavio fit and call the analyses
equivalent. If the inferential target is hierarchical Bayesian DDM, retain HDDM as the
backend and adapt its predictions, pointwise likelihood, diagnostics, and posterior groups.
The first-party `HierarchicalSmoothWienerDriftDiffusion` is not a drop-in HDDM replacement.

## PyDDM workflows

[PyDDM](https://pyddm.readthedocs.io/) supports generalized DDM components, including
time- or position-dependent drift, noise, and bounds. Behavio's first-party Wiener solver
is deliberately more constrained.

| PyDDM concept | Behavio destination | Migration check |
| --- | --- | --- |
| `Sample` conditions and responses | `Study` plus choice/RT `TaskSpec` | correct/error versus upper/lower coding |
| `gddm()` components | matching first-party Wiener specification or external estimator | noise scale, bound convention, starting point |
| `Fittable` ranges | natural and optimizer bounds in `ParameterSpace` | transforms and fixed parameters |
| solution PDFs | pointwise joint log probabilities and predictions | tail treatment and RT discretization |
| time-dependent bound/drift | external PyDDM adapter unless exactly represented | across-trial smoothness is not within-trial time dependence |

The last distinction is crucial: Behavio's smooth DDM changes parameters across decisions
on a longitudinal clock. A PyDDM component depending on within-decision time `t` changes the
accumulation process inside one decision. They are different models.

## Parity report

Every migration should leave a short machine-readable or tabular parity report containing:

- fixture identity and number of trials, sessions, and subjects;
- old and new package versions;
- matched and intentionally changed model assumptions;
- parameter-name and transform mapping;
- maximum absolute pointwise-log-probability difference at fixed parameters;
- simulation-summary differences under declared seeds;
- fit and posterior diagnostics from both paths; and
- claims enabled by the migration and claims still excluded.

After parity, use the [literature-recipe standard](tutorials/recipe-contract.md) to turn the
migrated analysis into public documentation. If the established implementation should
remain authoritative, expose it through [Behavio's extension contracts](extensions.md)
rather than copying its numerical core.
