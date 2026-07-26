# The first modelling contract

Unspool's first model is intentionally ordinary: a static Bernoulli GLM. Its purpose is
to establish what every more elaborate model must expose before smooth drift, latent
states, reinforcement learning, or hierarchical pooling are added.

## Common interface

Objects satisfying `BehaviourModel` provide:

| Method or property | Contract |
| --- | --- |
| `simulate(design, parameters, seed=...)` | Generate outcomes from named parameters while retaining the supplied study design. |
| `fit(study)` | Return estimates and visible numerical diagnostics, including failed convergence and boundary warnings. |
| `predict(study, fit, mode=...)` | State whether predictions are filtered or smoothed rather than conflating the two. |
| `pointwise_log_prob(study, fit, mode=...)` | Return one score per observed trial for validation and comparison. |
| `parameter_names` | Give simulation truth and fitted estimates the same stable coordinate system. |
| `signature` | Prevent a fit from being silently reused with a different model specification. |

`FitResult` retains parameter estimates, approximate standard errors, covariance, sample
size, optimizer status and message, iteration count, objective, gradient norm, Hessian
condition number, and a large-coefficient boundary warning. Arrays are copied and exposed
read-only. An optimizer's failure remains part of the result rather than being discarded.

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

The first such competitor, `SmoothBernoulliHistoryGLM`, represents each coefficient on an
explicit fixed-knot time basis with a random-walk roughness penalty. Its assumptions,
subject-alignment safeguards, and prospective use are detailed in
[Smooth change as a competing explanation](smooth-drift.md).

Fitting uses SciPy's deterministic L-BFGS-B optimizer. An optional L2 penalty applies to
non-intercept coefficients. Standard errors and 95% coverage summaries use a local Hessian
approximation; when penalization is nonzero they are approximate rather than exact
frequentist intervals.

## Prospective evaluation

```python
from unspool import evaluate_splits, forward_session_splits

splits = forward_session_splits(study, min_train_sessions=2)
evaluations = evaluate_splits(model, study, splits)

for evaluation in evaluations:
    print(evaluation.split.test_sessions, evaluation.mean_log_loss)
    print(evaluation.fit.diagnostics.converged)
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

## Design-specific parameter recovery

```python
from unspool import run_parameter_recovery

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
message, and child random seed. It also records the model signature and the design's number
of trials and subjects. Bias, RMSE, truth-estimate correlation, and approximate 95%
coverage are summaries of those retained runs—not a universal identifiability certificate.

To test whether the design can distinguish whole model families, use prospective
cross-model simulation rather than parameter recovery alone. The contract and its explicit
unresolved outcomes are described in [Prospective model recovery](model-recovery.md).

Run the complete synthetic example with:

```bash
uv run python examples/static_glm.py
```
