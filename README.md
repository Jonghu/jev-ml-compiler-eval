# Is Jev Enough for ML Compiler Decisions?

An evaluation of a general-purpose typed decision model on TVM TenSet and XLA
TpuGraphs.

This repository contains the course project for CS 521: ML and Compilers at the
University of Illinois Urbana-Champaign.

## Team

- Jung In Chang (`jichang2@illinois.edu`)
- Christian Teo (`chteo2@illinois.edu`)
- Jonghu Jeong (`jonghuj2@illinois.edu`)

## Research questions

- **RQ1:** Can Jev make competitive tensor scheduling decisions on TVM
  workloads?
- **RQ2:** Can Jev make competitive layout and tiling configuration decisions
  on XLA workloads?

The RQ1 evaluation uses TenSet and compares Jev with Random and an XGBoost cost
model. The RQ2 evaluation uses TpuGraphs and compares Jev with Random and a GNN
performance model.

## Current status

The shared evaluation framework, YAML experiment configuration, mock dataset,
Random baseline, and Jev integration are implemented. TenSet and TpuGraphs
workstreams remain to be implemented.

See [Project status](docs/project-status.md) for the completed-work checklist
and the remaining tasks for each benchmark.

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

## Run the mock experiments

Run the Random-only smoke test without credentials:

```bash
uv run python scripts/run_experiment.py \
  --config experiments/mock/random-smoke-test.yaml
```

To compare Random and Jev, create `.env` from the example and set a TypeSafe API
key:

```bash
cp .env.example .env
```

```dotenv
TYPESAFE_API_KEY=your-api-key
```

Then run:

```bash
uv run python scripts/run_experiment.py \
  --config experiments/mock/random-vs-jev.yaml
```

The CLI loads `.env` without overriding an already exported environment
variable. Experiment output directories are never overwritten. Change
`experiment.output_dir` before rerunning an experiment whose result directory
already contains files.

## Repository layout

```text
examples/                 Committed mock input data
experiments/              Version-controlled experiment YAML files
scripts/                  Command-line entry points
src/jev_compiler_eval/    Evaluation package
tests/                    Unit and integration tests
data/                     Local benchmark data, excluded from Git
results/                  Generated experiment results, excluded from Git
docs/                     Interfaces, configuration, protocol, and status
```

## Documentation

- [Configuration reference](docs/configuration.md)
- [Extension interfaces](docs/interfaces.md)
- [Experiment protocol](docs/experiment-protocol.md)
- [Project status and remaining work](docs/project-status.md)
