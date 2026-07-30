# The first modelling contract

Behavio's first model is intentionally ordinary: a static Bernoulli GLM. Its purpose is
to establish what every more elaborate model must expose before smooth drift, latent
states, reinforcement learning, or population structure are added.

## Common interfaces

Objects satisfying `BehaviourEstimator` provide fitting, prediction, and pointwise scoring.
`GenerativeBehaviourModel` adds named parameters and simulation; `BehaviourModel` remains
the backwards-compatible name for this full generative contract.
Models that expose [`ParameterSpaceProvider`](parameter-spaces.md) additionally distinguish
scientific natural parameters from transformed optimizer coordinates and publish portable
bounds, fixed values, and priors for inference adapters.

| Method or property | Contract |
| --- | --- |
| `fit(study)` | Return estimates and visible numerical diagnostics, including failed convergence and boundary warnings. |
| `predict(study, fit, mode=...)` | State whether predictions are filtered or smoothed rather than conflating the two. |
| `pointwise_log_prob(study, fit, mode=...)` | Return one score per observed trial for validation and comparison. |
| `scored_columns` | Declare the complete observed event represented by each likelihood contribution. |
| `signature` | Prevent a fit from being silently reused with a different model specification. |

Generative models additionally provide:

| Method or property | Contract |
| --- | --- |
| `simulate(design, parameters, seed=...)` | Generate outcomes from named parameters while retaining the supplied study design. |
| `parameter_names` | Give simulation truth and fitted estimates the same stable coordinate system. |

`FitResult` retains parameter estimates, approximate standard errors, covariance, sample
size, optimizer status and message, iteration count, objective, gradient norm, Hessian
condition number, and a large-coefficient boundary warning. Arrays are copied and exposed
read-only. An optimizer's failure remains part of the result rather than being discarded.

Calling `fit.audit()` derives one common `FitAudit` from those fields and any retained
restart, occupancy, or label evidence. Its `pass`, `warning`, or `fail` status never removes
the underlying diagnostics, and its stable issue codes make heterogeneous fits comparable
in reports. See the [fit-audit guide](diagnostics.md) for the complete rules and
interpretation boundary.

The [estimator contract guide](estimator-contract.md) documents plugin compatibility,
machine-readable capabilities, fit-result invariants, and why choice-only and joint
choice/response-time likelihoods cannot be ranked as if they scored the same event.

## Static Bernoulli history GLM

`BernoulliHistoryGLM` models a binary choice using an intercept, named numeric covariates,
and zero or more previous choices:

\[
\operatorname{logit} P(y_t=1) = \beta_0 + x_t^\top\beta
  + \sum_{k=1}^{K}\gamma_k(2y_{t-k}-1).
\]

Choice history is constructed in explicit trial order and reset at every subject/session
boundary. Unavailable history at a session's beginning is encoded as zero. Simulation is
recursive: a generated choice changes the history seen by the next trial. Prediction is
one-step-ahead and filtered: the prediction at trial *t* may use observed choices before
*t*, never later choices.

The coefficients are static across subjects and sessions. That restriction is the model's
scientific role, not a claim about learning. It provides a stationary account that smooth
drift and discrete-state models must outperform under prospective evaluation.

The first such competitor, `smooth(model, over=...)`, represents each parameter on an
explicit fixed-knot time basis with a random-walk roughness penalty. Its assumptions,
subject-alignment safeguards, and prospective use are detailed in
[Smooth change as a competing explanation](smooth-drift.md).

`hierarchical(model, over="subject")` supplies the constrained population model. It
jointly estimates population parameters and Gaussian-penalized group deviations with the
scales fixed before fitting, and it takes a declaration of *which* parameters vary. Seen
subjects use their fitted deviations; unseen subjects use an explicitly recorded
population plug-in. See the [partial-pooling guide](hierarchical-glm.md).

The two axes compose: `hierarchical(smooth(model), over="subject")` is a fixed-knot
population trajectory plus shrunken, smooth subject-deviation trajectories. Its
shared-clock assumption, penalty structure, unseen-subject policy, and factorial recovery
evidence are detailed in [Partially pooled trajectories](hierarchical-smooth-glm.md), and
the combinators themselves in [Composing models](composing-models.md).

`BernoulliGLMHMM` supplies the first discrete switching competitor: state-specific
Bernoulli GLM emissions, a stationary learned transition matrix, session-reset initial
states, deterministic multi-restart fitting, filtered state probabilities, and explicit
label/occupancy diagnostics. Its assumptions and non-interpretive label convention are
detailed in the [fixed-transition GLM-HMM guide](glm-hmm.md).

`BinaryQLearning` supplies the first reward-learning competitor. It uses an explicit
action-contingent reward environment for simulation, session-reset action values, filtered
updates, and recoverable learning-rate, temperature, bias, and perseveration parameters.
See the [session-reset Q-learning guide](q-learning.md).

`MultinomialLogit` supplies the categorical reference likelihood. It consumes the shared
`ChoiceSpec` coordinate, respects trial-specific action availability, optionally retains
omissions as an additional category, and passes full probability vectors through the same
prospective comparison, protocol, and recovery machinery. It is also composable, so
`smooth()` and `hierarchical()` supply its drifting and per-subject forms. See the
[multinomial and omission-aware choice guide](multinomial.md).

`WienerDriftDiffusion` supplies the first joint choice/response-time family. Covariates
control drift while boundary separation, relative starting bias, and non-decision time are
shared across trials. It declares both observed columns, converts explicit physical time
units to seconds, retains deterministic restart evidence, and supports generative recovery.
An optional fixed-support contaminant component estimates the probability of independent
choice/RT responses and retains posterior trial responsibilities without silently removing
them.
See the [joint choice and response-time guide](drift-diffusion.md).

`SmoothWienerDriftDiffusion` places selected drift coefficients, boundary separation, and
starting bias on fixed-knot paths over an explicit study clock while retaining stationary
non-decision time. Its future-knot persistence forecast, single-subject default, and strict
distinction between across-trial trajectories and within-decision dynamics are detailed in
the [session-varying drift-diffusion guide](smooth-ddm.md).

`HierarchicalSmoothWienerDriftDiffusion` adds smooth Gaussian-shrunken animal deviations
around those population paths. Represented animals use their fitted paths; unseen animals
use an explicitly recorded population-trajectory plug-in by default. Its separate
`predict_new_subjects()` contract integrates fitted random-effect paths and reports
subject-joint likelihoods with Monte Carlo diagnostics. The fixed- or estimated-scale MAP
contract, supplemented scale uncertainty, natural-bound safeguards, arrowhead local
uncertainty, and recovery evidence are detailed in the
[hierarchical drift-diffusion guide](hierarchical-smooth-ddm.md).

The GLM implementations use SciPy's deterministic L-BFGS-B optimizer. An optional L2
penalty applies to non-intercept coefficients. Standard errors and 95% coverage summaries
use a local Hessian approximation; when penalization is nonzero they are approximate rather
than exact frequentist intervals.

## Prospective evaluation

```python
from behavio import evaluate_splits, forward_session_splits

splits = forward_session_splits(study, min_train_sessions=2)
evaluations = evaluate_splits(model, study, splits)

for evaluation in evaluations:
    print(evaluation.split.test_sessions, evaluation.mean_log_loss)
    print(evaluation.fit.audit().status, evaluation.fit.audit().issue_codes)
```

`evaluate_splits` requires prospective folds by default. Passing leave-one-session-out
folds raises an error unless `require_prospective=False` explicitly acknowledges the
interpolation analysis. Each fold refits the model from scratch and retains its full fit.

For within-session rolling origins, evaluation replays the observed current-session prefix
to construct filtered history, then returns predictions and scores only for future target
trials. A multi-trial horizon is evaluated sequentially as outcomes become observed; it is
not an open-loop simulation from the origin. See the [validation guide](validation.md).

This guarantee covers model fitting and scoring. Covariates supplied to the model must
also be causally available at prediction time. Learned normalization, feature selection,
and learning landmarks still require training-only estimation. The first fold-fitted
landmark contract is described in the
[clock and transform guide](clocks-and-transforms.md).

Several candidates can be evaluated as one matched scientific object with
`compare_models`. It retains equal-unit and pooled scores, paired unit-bootstrap
differences, fit audits, and fold provenance. Candidate or hyperparameter selection for a
final forecast belongs inside `nested_select_model`, which supplies only the outer training
study to its inner splitter. See the [prospective comparison guide](comparison.md).

## Design-specific parameter recovery

```python
from behavio import run_parameter_recovery

report = run_parameter_recovery(
    model,
    design,
    parameter_sets=[
        {"intercept": -0.2, "stimulus": 0.8, "choice_lag_1": 0.2},
        {"intercept": 0.2, "stimulus": 1.2, "choice_lag_1": 0.6},
    ],
    repeats=10,
    seed=123,
)

for row in report.summary():
    print(row.parameter, row.bias, row.rmse, row.correlation, row.coverage_95)
```

The report stores every truth, estimate, standard error, convergence flag, optimizer
message, complete fit audit, and child random seed. Warning fits remain eligible for
summary; failed audits remain visible but are excluded. Coverage reports its own finite-
uncertainty denominator. The report also records the model signature and the design's
number of trials and subjects. Bias, RMSE, truth-estimate correlation, and approximate 95%
coverage are summaries of those retained runs—not a universal identifiability certificate.

To test whether the design can distinguish whole model families, use prospective
cross-model simulation rather than parameter recovery alone. The contract and its explicit
unresolved outcomes are described in [Prospective model recovery](model-recovery.md).

Run the complete synthetic example with:

```bash
uv run python examples/static_glm.py
```
