# Contributing

Unspool is pre-alpha. Design discussion is welcome, but the public API should grow only
around tested scientific use cases.

## Set up

Install [uv](https://docs.astral.sh/uv/) and run:

```bash
uv sync --locked --all-groups
```

This creates an isolated environment from the committed lockfile. Do not install project
dependencies globally or mutate the system Python.

## Check a change

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv build
```

Use `uv run ruff format .` to apply formatting.

## Documentation changes

Build the documentation in strict mode before submitting a change:

```bash
uv sync --group docs --locked
uv run --group docs mkdocs build --strict
```

Worked studies should identify the scientific question, experimental unit, estimand,
validation boundary, result, and limitations. If a page adds or changes a figure:

- regenerate it with `uv run --group docs python -m scripts.plot_documentation_figures`;
- provide conclusion-bearing alternative text and a caption;
- classify it as empirical or conceptual in the
  [figure provenance register](docs/reference/figure-provenance.md); and
- keep the source data or frozen benchmark artifact traceable without committing fetched
  datasets.

Use `--skip-cell` to regenerate every figure except the Cell reproduction when its
checksum-pinned source table is not present locally.

## Scientific changes

A new model or inference path should include:

- a generative simulation;
- a recovery test under at least one explicit design;
- convergence and failure diagnostics;
- a comparison to the simplest relevant alternative; and
- documentation that separates predictive, identificatory, and mechanistic claims.

Data-derived preprocessing, clocks, and landmarks must be fitted inside validation folds
when used for held-out evaluation.

## Scope

Keep reusable package code in `src/unspool/`, tests in `tests/`, and scientific rationale
in `docs/`. Dataset downloads, generated reports, notebooks, and paper-specific analyses
should not enter the core package without a clear reusable contract.
