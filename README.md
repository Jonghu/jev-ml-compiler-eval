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

Both workstreams directly compare Jev with Random on fixed local candidate
pools. TenSet's published MLP and TpuGraphs' published GNN/MLP values provide
external references; those trained checkpoints were not reproduced locally.
The retained local XGBoost adapter is optional legacy tooling.

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

The primary [ResNet-50 evaluation](docs/results/tenset/paper-resnet50/README.md)
is also complete: 27 distinct tasks, 56 task occurrences and 100,126 schedule
records. Jev's network Top-1/Top-5 endpoint scores are **0.0977 / 0.2746**,
versus Random's **0.0915 / 0.3771**. This is a small observed Top-1 gain and a
Top-5 deficit. The published MLP + Ranking loss reference is **0.8823 / 0.9456**;
its original inventory and checkpoint remain unverified. The network metric
differs from the earlier workload-prefix metrics.

The [controlled context-ablation study](docs/results/tenset/context-ablation/README.md)
is complete: seven conditions with three fresh repeats on the frozen E5
120-workload, 50-schedule cohort, plus three repeats of filtered A5 on all
27 ResNet-50 task pools. Naive input has the highest small-study mean Top-1
score (0.3582 versus the saved Random draw's 0.3324); additional descriptions
do not provide consistent gains. The new filtered network scores are
**0.1134 / 0.3432**, below the current Random-42 reference's
**0.1373 / 0.4063**. These are separate from the historical Random-0 comparison
above. All 2,601 requests were live, with an estimated **$5.04 in input cost**
at the recorded assumed rate. The
[ablation guide](docs/run-tenset-context-ablation.md) documents offline
preflight, paid execution, resume, and result sharing. The TenSet structural
diversity filter differs from TpuGraphs' large-tile heuristic.

The TpuGraphs pipeline is included in this tree. Its logged full `tile:xla`
validation experiment covers **676 kernels from seven programs**, with
1,038,376 deduplicated configurations. Jev has higher mean Top-1 error than
Random (413.6% versus 268.4%) and lower Top-5 error (48.9% versus 57.6%). See
the [experiment log](docs/runlog.md); these are logged RQ2 results, not a new
independent raw-bundle audit.

Remaining work includes concrete TenSet tensor/schedule case narratives,
the consolidated RQ2/team report, metric reconciliation and final review. The
[finish guide](docs/run-tenset-finish-rq1.md) distinguishes those steps from
reproduction commands for already completed paid runs. Offline analysis uses
the saved prepared data and result bundles; further Jev runs require credentials.
The [ResNet-50 guide](docs/run-tenset-paper-resnet50.md) documents preparation,
offline preflight and optional reproduction. Preserve each historical run's
code/data hashes: merged implementation changes do not turn an old frozen
collection into a resumable new-code experiment.

See [Project status](docs/project-status.md) for the completed-work checklist
and the remaining tasks for each benchmark.
The [one-page progress report](docs/reports/progress-2026-11-19.pdf) summarizes
the repository review, results and remaining work as of 19 November 2026.

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
- [Completed TenSet context, order, and filtering ablation](docs/results/tenset/context-ablation/README.md)
- [Context-ablation execution, cost preflight, and sharing guide](docs/run-tenset-context-ablation.md)
- [Completed main-paper ResNet-50 comparison and reproduction guide](docs/run-tenset-paper-resnet50.md)
- [Main-paper protocol and metric source audit](docs/tenset-paper-protocol.md)
- [Finish the Jev-versus-Random RQ1 analyses](docs/run-tenset-finish-rq1.md)
- [Run the reconstructed TenSet 500-file tutorial test pools](docs/run-tenset-tutorial-reference.md)
- [Configuration reference](docs/configuration.md)
- [Extension interfaces](docs/interfaces.md)
- [Experiment protocol](docs/experiment-protocol.md)
- [Project status and remaining work](docs/project-status.md)
- [Run the TenSet pilot on macOS](docs/run-tenset-on-macos.md)
- [TenSet setup, training, evaluation, and analysis](docs/tenset.md)
- [RQ1 methodology and experimental status](docs/rq1-report.md)
