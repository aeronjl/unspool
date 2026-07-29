# Plotting API

`behavio.plot` renders the reports the rest of the package produces. Every central
diagnostic here is inherently visual: a rank histogram, an ECDF-difference band, a
Pareto-\(k\) vector, a posterior-predictive reference distribution, and a recovery scatter
all lose most of their meaning when reduced to a tuple of numbers, because the *shape* of
the deviation is the finding.

## Installation

matplotlib is an optional dependency, not a core one:

```bash
pip install 'behavio[plots]'
```

`import behavio` and `import behavio.plot` both work without matplotlib installed. Only
calling a plotting function raises `MatplotlibUnavailableError`, whose message names the
extra. This mirrors how `behavio.posterior` treats ArviZ.

## Contract

Every function takes a report object and returns a `matplotlib.figure.Figure`. The functions
that draw a single panel also accept an optional `ax`, so a display can be composed into a
larger figure.

- **No side effects.** Nothing calls `show`, writes a file, or mutates global `rcParams` at
  import. Figures are built from `matplotlib.figure.Figure` directly rather than through
  `pyplot`, so nothing is registered in a global registry and nothing needs closing.
- **The house style is scoped.** `figure_style()` applies the figure standard for the
  duration of a call and restores the previous `rcParams` afterwards.
  `configure_figure_style()` is the opt-in global version, for scripts that own their whole
  process.
- **Plots present; they do not compute.** A number that is not already on the report is not
  invented here. The SBC band is the report's simultaneous band, the ELPD interval is the
  report's paired interval, and the Pareto-\(k\) threshold is the report's `good_k`.
- **Failure stays visible.** Where a report carries an audit status, an issue list, or a
  replicate accounting, the figure carries it too. A display drawn from a `FAIL`ed
  posterior is watermarked, so it cannot be cropped or re-captioned into clean evidence.

## Coverage

| Display | Reads | Function |
| --- | --- | --- |
| SBC rank histogram | `SBCReport.summary()` | `plot_sbc_rank_histogram` |
| SBC ECDF difference with its simultaneous band | `SBCReport.uniformity()` | `plot_sbc_ecdf_difference` |
| Pareto-\(k\) by observation or block | `psis_loo` | `plot_pareto_k` |
| Paired ELPD differences | `compare_posterior_models` | `plot_elpd_differences` |
| Posterior-predictive reference distributions | `posterior_predictive_check` | `plot_predictive_check`, `plot_predictive_checks` |
| Parameter recovery, true against estimated | `run_parameter_recovery` | `plot_parameter_recovery`, `plot_parameter_recovery_grid` |
| Aggregate probability calibration | `fit_model` | `plot_calibration` |
| R-hat and effective sample size | `audit_posterior` | `plot_rhat`, `plot_ess`, `plot_convergence` |

### The SBC band is a band

`plot_sbc_ecdf_difference` fills the region between
`SBCUniformity.lower_difference_band` and `SBCUniformity.upper_difference_band`. It is a
*simultaneous* envelope: under the null the whole difference curve stays inside it with
probability `confidence_level`. Drawing it as error bars would invite a pointwise reading,
which would be exceeded far more often than the nominal level.

The rank histogram deliberately carries only its exact discrete-uniform expectation line and
no second envelope, for the same reason.

### Recovery intervals are never pooled

`ParameterRecoveryReport.interval_kind` records whether coverage came from a Wald interval
built on the standard errors or from the equal-tailed 95% quantiles of the posterior draws.
The two are different quantities, so they are drawn with different colours, different cap
styles, and an explicit legend label naming the kind.

### Calibration is aggregate only

`CalibrationSummary` retains mean predicted probability, observed rate, Brier score, and
expected calibration error. The ten-bin decomposition behind the expected calibration error
is computed inside `behavio.runner` and then discarded, so `plot_calibration` draws the
aggregate point against the diagonal rather than a full reliability curve. Retaining per-bin
counts and rates on `CalibrationSummary` is the change that would upgrade this display.

## Figure standard

The visual contract — palette, typography, and deterministic SVG export — lives in
`behavio.plot.style` and is described in the
[scientific figure standard](figure-standard.md). The documentation figure generator in
`scripts/plot_documentation_figures.py` is a caller of that module rather than the owner of
the style.

::: behavio.plot
    options:
      members_order: source
      show_root_heading: false
      show_source: false
