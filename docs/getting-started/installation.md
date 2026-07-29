# Installation

Behavio requires Python 3.11 or newer. It is currently a pre-release research package,
so install it from GitHub rather than PyPI. The core install contains only NumPy and
SciPy; data adapters and probabilistic backends remain optional.

## Use Behavio in an analysis

Create an isolated environment, then install the current source snapshot and Matplotlib
for the figures in the [first analysis](first-analysis.md):

=== "macOS and Linux"

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install \
      "behavio @ git+https://github.com/aeronjl/behavio.git@main" \
      "matplotlib>=3.9"
    ```

=== "Windows PowerShell"

    ```powershell
    py -3.12 -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install `
      "behavio @ git+https://github.com/aeronjl/behavio.git@main" `
      "matplotlib>=3.9"
    ```

Verify the environment:

```bash
python -c "import behavio; print(behavio.__version__)"
```

!!! tip "Pin a scientific analysis"

    `@main` follows active development. Replace it with a commit SHA when an analysis
    must be exactly reproducible, and record that SHA with the data and protocol.

## Contribute to Behavio

The repository uses [uv](https://docs.astral.sh/uv/) and a committed lockfile. GitHub
access is public over HTTPS; use SSH if your GitHub key is already configured.

```bash
git clone git@github.com:aeronjl/behavio.git
cd behavio
uv sync --locked --all-groups
uv run pytest
```

The common verification commands are:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv build
uv run --group docs mkdocs build --strict
```

## Optional integrations

Reading a CSV or TSV trial table with
[`read_table()`](../interoperability.md#csv-tsv-and-parquet-files) needs none of the
following: delimited ingest is part of the base install, because a trial table is how most
behavioural data reaches Python rather than an optional data source.

Install only the interface used by the analysis:

| Need | uv | pip from a checkout |
| --- | --- | --- |
| Parquet trial tables | `uv sync --extra parquet` | `python -m pip install -e ".[parquet]"` |
| Pose and ethogram readers | `uv sync --extra readers` | `python -m pip install -e ".[readers]"` |
| NWB read/write | `uv sync --extra nwb` | `python -m pip install -e ".[nwb]"` |
| DANDI-backed NWB | `uv sync --extra dandi` | `python -m pip install -e ".[dandi]"` |
| IBL ONE | `uv sync --extra ibl` | `python -m pip install -e ".[ibl]"` |
| PyBADS optimization | `uv sync --extra optimization` | `python -m pip install -e ".[optimization]"` |
| ArviZ interchange | `uv sync --extra probabilistic` | `python -m pip install -e ".[probabilistic]"` |
| PyMC backend | `uv sync --extra bayesian` | `python -m pip install -e ".[bayesian]"` |
| Diagnostic figures | `uv sync --extra plots` | `python -m pip install -e ".[plots]"` |

The `plots` extra is exactly the `matplotlib>=3.9` pin used above. `import behavio` and
`import behavio.plot` both work without it; only calling a
[plotting function](../reference/plots.md) raises, and the error names the extra.

These extras do not alter the canonical [`Study`](../data-contract.md) or
[`TaskSpec`](../task-contract.md) contracts. They add a source, optimizer, result backend,
or figure renderer at the edge of the same workflow.

## Troubleshooting

- `ModuleNotFoundError: behavio` usually means the environment was not activated or the
  install command ran under a different Python. Compare `python -c "import sys;
  print(sys.executable)"` with the environment you created.
- Adapter import errors should name the missing extra. Install that extra rather than
  adding every optional dependency.
- On Apple silicon, use a native arm64 Python where possible; NumPy and SciPy wheels are
  available for supported Python versions.

[Run the first analysis](first-analysis.md){ .md-button .md-button--primary }
[Understand the data contract](../data-contract.md){ .md-button }
