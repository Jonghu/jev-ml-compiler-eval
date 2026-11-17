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

- **RQ1:** Can Jev select faster tensor schedules than Random from program and
  schedule descriptions on TVM workloads?
- **RQ2:** Can Jev make competitive layout and tiling configuration decisions
  on XLA workloads?

The RQ1 evaluation uses TenSet to compare Jev with Random. TenSet provides
recorded runtime labels and metric definitions; specialized-model comparison
and training are outside the current RQ1 scope. The retained local XGBoost
adapter is optional legacy tooling. The RQ2 evaluation uses TpuGraphs and
plans to compare Jev with Random and a GNN performance model.

## Current status

The shared evaluation framework, YAML experiment configuration, mock dataset,
Random baseline, and Jev integration are implemented. The TenSet workstream
includes record preparation, portable metadata export, label-free serialization,
single-question and tournament selection, benchmark metrics, and comparison plots.
Legacy TVM is needed only for the optional local XGBoost adapter.

Completed Jev-versus-Random results include the E5-2673 study on the same 120
workloads at **16, 32, and 50 valid schedules**, and the Platinum-8272 tournament
on **484 workloads and 187,679 test records**. The 50-schedule result is a
historical anchor reused in the size study. These protocols are reported
separately in the [RQ1 report](docs/rq1-report.md), with
[candidate-size results](docs/results/tenset/size-sweep/README.md) and
[tournament results](docs/results/tenset/tutorial-500-platinum-test/README.md).

Remaining TenSet work is the offline export and review of concrete tensor and
schedule case narratives, followed by final report review. The
[finish guide](docs/run-tenset-finish-rq1.md) distinguishes those steps from
reproduction commands for already completed paid runs. Offline analysis uses
the saved prepared data and result bundles; further Jev runs require credentials.
The TpuGraphs implementation is on the separate `tpu-graph` branch and awaits
integration.

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

- [TenSet completed runs and reviewed results](docs/tenset-runlog.md)
- [Completed TenSet 16/32/50 candidate-size results](docs/results/tenset/size-sweep/README.md)
- [Finish the Jev-versus-Random RQ1 analyses](docs/run-tenset-finish-rq1.md)
- [Run the reconstructed TenSet 500-file tutorial test pools](docs/run-tenset-tutorial-reference.md)
- [Configuration reference](docs/configuration.md)
- [Extension interfaces](docs/interfaces.md)
- [Experiment protocol](docs/experiment-protocol.md)
- [Project status and remaining work](docs/project-status.md)
- [Run the TenSet pilot on macOS](docs/run-tenset-on-macos.md)
- [TenSet setup, training, evaluation, and analysis](docs/tenset.md)
- [RQ1 methodology and experimental status](docs/rq1-report.md)
