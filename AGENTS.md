# Repository instructions

## Purpose

Unspool is a process-first Python library for fitting and falsifying behavioural models
across learning. Scientific validity and a small stable API take priority over model count.

## Workflow

- Use `uv` for Python, environments, dependencies, tools, and lockfiles.
- Support Python 3.11 and newer; use the pinned Python 3.12 interpreter for local work.
- Add dependencies with `uv add` or `uv add --dev` and commit `uv.lock`.
- Do not add containers, a second package manager, or a task runner without a demonstrated
  need.
- Do not commit, push, publish, or configure a remote unless the user explicitly asks.

## Architecture

- Keep reusable library code in `src/unspool/`.
- Keep data-source adapters optional; do not make NWB, ONE, or DANDI core dependencies.
- Require models to expose compatible simulation, fitting, prediction, pointwise scoring,
  and diagnostics contracts before expanding the model catalogue.
- Keep paper-specific analyses outside the importable package.

## Scientific requirements

- Preserve trial order and subject/session boundaries.
- Fit learned preprocessing and landmarks within training folds.
- Distinguish filtered predictions from smoothed descriptions.
- Treat parameter and model recovery as design-specific evidence.
- Keep optimization failures, boundary estimates, and latent-label ambiguity visible.
- Do not interpret latent states without targeted smooth, history, learning, and observable
  behavioural competitors.

## Validation

Before handing off changes, run:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv build
```
