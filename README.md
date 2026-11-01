# Is Jev Enough for ML Compiler Decisions?

An evaluation of a general-purpose typed decision model on TVM TenSet and XLA
TpuGraphs.

This repository contains the course project for CS 521: ML and Compilers at the
University of Illinois Urbana-Champaign.

## Team

- Jung In Chang (`jichang2@illinois.edu`)
- Christian Teo (`chteo2@illinois.edu`)
- Jonghu Jeong (`jonghuj2@illinois.edu`)

## Development setup

This project requires Python 3.11 and
[uv](https://docs.astral.sh/uv/getting-started/installation/).

Install the project and development dependencies:

```bash
uv sync
```

Install the required Git pre-commit hook:

```bash
uv run pre-commit install
```

Verify the development setup:

```bash
uv run pre-commit run --all-files
```

Every commit must pass Ruff formatting, Ruff linting, and the complete pytest
suite. CI runs the same pre-commit checks on every push and pull request.

## Status

The evaluation framework and example experiments will be added incrementally.
