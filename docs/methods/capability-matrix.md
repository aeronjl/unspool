# Capability matrix

**Supported** means the capability has a stable data contract, prospective scoring,
diagnostics, tests, and direct benchmark evidence. **Experimental** means it is usable but
its inferential or design coverage remains deliberately narrow.

| Scientific capability | Status | Evidence boundary |
| --- | --- | --- |
| Canonical trial/session/animal representation | **Supported** | Identity, chronology, and source order are validated. |
| Forward-session, within-session, and historical-cohort prospective validation | **Supported** | Learned preprocessing must still be fitted within folds; historical-cohort claims require the declared deployment order. |
| Complete-subject and complete-lab holdout | **Supported** | Does not by itself create population-of-labs inference. |
| Static Bernoulli history GLM | **Supported** | Choice-only likelihood with declared covariates and lags. |
| Smooth longitudinal Bernoulli GLM | **Supported** | Fixed clock and knots; future knots are not learned from test data. |
| Hierarchical static and smooth GLMs | **Supported** | Gaussian partial pooling with population plug-in for unseen subjects. |
| Training-only nested candidate selection | **Supported** | Estimates the selection procedure, not one retrospectively named model. |
| Parameter and model recovery | **Supported** | Evidence is specific to the simulated design and parameter regime. |
| GLM-HMM | **Experimental** | Fixed transitions; state interpretation requires alignment and competitors. |
| Binary Q-learning | **Experimental** | Compact two-action, session-reset implementation. |
| Static and smooth drift diffusion | **Experimental** | Joint choice/RT data with explicit units and fixed support assumptions. |
| Hierarchical smooth drift diffusion | **Experimental** | Plug-in population prediction and bounded variance-component estimation. |
| Threshold learning landmarks | **Experimental** | Fold-fitted plug-in bootstrap; unresolved draws remain visible. |
| Cross-lab trajectory geometry | **Experimental** | Fixed empirical labs, no population-of-labs generalization. |
| Session-varying GLM-HMM parameters | **Planned** | Requires targeted state and trajectory recovery first. |
| Hierarchical lab effects | **Planned** | Requires more labs and a population-level sampling model. |
| Full Bayesian uncertainty propagation | **Planned** | Current implementations use Laplace/local or Monte Carlo approximations. |
| Neural-signal modelling | **Out of scope** | Neural observations should enter through explicit companion models or packages. |

A planned capability should not be approximated by forcing data through a supported method
with a different estimand.
