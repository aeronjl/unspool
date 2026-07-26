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
