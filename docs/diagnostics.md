# Fit audits

Every fitted Unspool model retains its raw optimizer evidence in `FitDiagnostics`.
Multi-restart and latent-state models also retain their model-specific evidence on their
fit-result subclasses. `FitAudit` adds a normalized view across those objects; it does not
replace, edit, or conditionally discard them.

<figure class="doc-figure" data-figure-kind="Conceptual">
  <img src="../assets/diagnostic-layers.svg" alt="Four stacked evidence layers: numerical fit, prospective prediction, parameter recovery, and model recovery, culminating in bounded interpretation.">
  <figcaption><strong>Evidence is layered.</strong> Numerical convergence is necessary but cannot certify prediction, parameter recovery, model discrimination, or interpretation. This is a conceptual audit hierarchy.</figcaption>
</figure>

```python
fit = model.fit(study)
audit = fit.audit()

print(audit.status)
print(audit.issue_codes)
print(audit.restarts)
print(audit.latent_states)
```

The equivalent function form is `audit_fit(fit)`. `audit.to_dict()` produces a fresh,
JSON-serializable record for reports and recovery grids.

## Status is numerical, not cognitive

| Status | Meaning |
| --- | --- |
| `pass` | No implemented numerical or model-specific rule fired. |
| `warning` | The fit exists, but at least one issue limits uncertainty estimates or interpretation. |
| `fail` | Nonconvergence or non-finite core fit evidence makes the numerical result unusable. |

A pass is not evidence that the model is the right explanation, globally identifiable, or
prospectively useful. A warning does not always invalidate prediction: label ambiguity,
for example, blocks stable naming of latent states but does not automatically make the
model's marginal predictions meaningless. Consumers decide which issue codes block their
particular claim; Unspool keeps the decision visible.

## Stable issue codes

| Code | Severity | Trigger |
| --- | --- | --- |
| `optimizer_nonconvergence` | error | The selected optimizer result did not converge. |
| `nonfinite_estimates` | error | At least one selected parameter estimate is non-finite. |
| `nonfinite_objective` | error | The selected objective is non-finite. |
| `nonfinite_gradient` | error | The selected gradient norm is non-finite. |
| `nonfinite_uncertainty` | warning | The local covariance or standard errors are non-finite. |
| `invalid_standard_errors` | warning | A reported standard error is negative. |
| `nonfinite_hessian_condition` | warning | The local Hessian condition number is non-finite. |
| `ill_conditioned_hessian` | warning | The condition number crosses the explicit audit threshold. |
| `boundary_estimate` | warning | The model's own parameter-boundary rule fired. |
| `restart_nonconvergence` | warning/error | Some/all deterministic restarts failed. |
| `restart_objective_disagreement` | warning | Converged restarts reached materially different objectives. |
| `low_state_occupancy` | warning | A latent-state model's configured occupancy rule fired. |
| `label_ambiguity` | warning | The model's configured state-label ordering rule fired. |

`RestartAudit` reports restart count, convergence count, selected restart and objective,
failed messages, and the absolute and relative range among finite converged objectives.
It does not pretend that objective agreement proves parameter agreement. `LatentStateAudit`
reports state count, minimum occupancy, emission separation, label-order gap, and the
model's ambiguity flags. Truth-aware permutation alignment is deliberately absent from a
fit audit: it is available only in simulation recovery, where reference states exist.

## Explicit policy

Two continuous rules have shared defaults and can be changed without changing the fitted
model:

```python
from unspool import FitAuditPolicy

audit = fit.audit(
    policy=FitAuditPolicy(
        hessian_condition_warning=1e10,
        restart_relative_objective_warning=1e-4,
    )
)
```

Boundary, occupancy, and label-ambiguity thresholds remain model configuration because
their meaning depends on the parameterization. The audit consumes the resulting flags; it
does not silently impose a second threshold.

## Interpretation boundary

The audit checks retained numerical evidence, not posterior calibration, parameter
recovery, model recovery, predictive generalization, or causal interpretation. Those
remain separate design-specific contracts. In particular, a clean GLM-HMM audit does not
license cognitive names for its states, and a failed audit must remain visible rather than
being removed from recovery denominators.
