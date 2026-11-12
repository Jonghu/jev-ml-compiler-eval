# Project status

This checklist separates implemented software, audited TenSet experiments,
and the benchmark evidence still needed for the final report. See the
[macOS TenSet guide](run-tenset-on-macos.md) for the current execution workflow.

## Shared repository and infrastructure — Person A

### Repository and environment

- [x] Initialize the private GitHub repository and project structure.
- [x] Set up Python 3.11, uv, and a committed lockfile.
- [x] Enforce Ruff formatting, Ruff linting, and pytest through pre-commit.
- [x] Run the same checks in GitHub Actions.
- [x] Define common workload, candidate, runtime, decision, and evaluation-case
  types.
- [x] Define explicit dataset and model interfaces and registries.
- [x] Load version-controlled YAML experiment configurations.
- [x] Save manifests, per-workload predictions, aggregate metrics, and CSV
  summaries.
- [x] Add a committed mock dataset and end-to-end example experiments.

### Jev integration

- [x] Use the official TypeSafe Python SDK.
- [x] Handle authentication, retries, timeouts, API errors, and response
  validation.
- [x] Cache successful requests without storing credentials.
- [x] Reuse and close the TypeSafe client across workload decisions.
- [x] Serialize generic workload context and candidate configurations.
- [x] Convert Choice probabilities into complete candidate rankings.
- [x] Record resolved model, confidence, probabilities, token usage, request ID,
  cache status, and latency.
- [x] Add configurable TenSet candidate/workload budgets and a local input-byte
  guard.
- [ ] Validate TypeSafe service limits and agree on policies across datasets.
- [ ] Add candidate batching or staged selection if a benchmark exceeds those
  limits.

### Shared evaluation

- [x] Implement the reproducible Random baseline.
- [x] Implement runtime regret and mock Top-1 accuracy.
- [x] Measure per-decision and aggregate latency.
- [x] Record API token usage and cache status.
- [x] Run multiple models on identical candidate sets.
- [x] Implement and test the TenSet Top-1/Top-5 Score and Pairwise Accuracy
  definitions.
- [ ] Integrate and validate TpuGraphs metrics in the shared main branch.
- [ ] Convert token usage into monetary cost using a recorded pricing snapshot.
- [x] Add deterministic TenSet candidate sampling and evaluation budgets.
- [ ] Agree on candidate sampling and budgets across both benchmarks.
- [ ] Support `fail_fast: false` with explicit partial-result semantics.
- [x] Implement TenSet comparison tables and plots from saved result bundles.
- [ ] Generate reviewed benchmark comparison tables and plots across datasets.

### Integration and final analysis

- [x] Review and merge the TenSet workstream into main.
- [ ] Review and merge the TpuGraphs workstream.
- [ ] Verify reproducibility across both benchmark pipelines.
- [ ] Analyze the overall performance-cost tradeoff.
- [ ] Consolidate cross-dataset tables and plots for the final report.

## RQ1: TVM TenSet — Person B

The portable dataset and metadata pipeline is implemented. The Mac pilot and
a 120-test-workload, 50-schedule long-context comparison have completed. The
uploaded results of both runs were independently audited. The full held-out,
all-candidate experiment and the follow-up ablations remain pending. The primary experiment is
Jev versus Random; published XGBoost results are an external literature
reference, with their protocol differences stated explicitly. The optional
local XGBoost implementation does not block this experiment.

See the [macOS run guide](run-tenset-on-macos.md), [implementation reference](tenset.md),
and [RQ1 report section](rq1-report.md).

### User-reported Mac pilot

The console output from `experiments/tenset/jev-vs-random.yaml` reports:

| Model | Workloads | Schedules | Top-1 Score | Top-5 Score | Pairwise Accuracy | Runtime regret | Mean decision latency (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random | 1 | 8 | 1.0 | 1.0 | 0.821429 | 0.0 | 0.0355 |
| Jev `jev-1.14.0` | 1 | 8 | 1.0 | 1.0 | 0.571429 | 0.0 | 555.784792 |

Both selected a fastest schedule in this candidate pool. This is a setup pilot
using `split: all`, not the held-out benchmark or evidence of overall model
competitiveness. The subsequently uploaded pilot bundle was inspected: both
models selected the unique sampled optimum, Jev used the requested model, and
its request was uncached. The saved metric values recomputed correctly.

### Audited 120-workload comparison

The run at commit `6a4fe91` uses 120 project-test workloads, 50 identical sampled
schedules per workload, seed 42, full program metadata, and experimental Jev
long-context mode. All 240 prediction rows agree with the manifest and the
reported metric definitions. Random rankings reproduce from their recorded seed.

| Model | Top-1 Score | Top-5 Score | Pairwise Accuracy | Runtime regret |
|---|---:|---:|---:|---:|
| Random-42 | 0.332440 | 0.498775 | 0.491959 | 6.058854 |
| Jev 1.14 long context | 0.336867 | 0.512829 | 0.519456 | 5.081011 |

Jev selects a faster schedule on 58 workloads, a slower one on 60, and ties
on two; both models select the sampled optimum on five workloads. Exploratory
paired confidence intervals include no improvement for Top-1, Top-5, and mean
regret against the observed Random-42 ranking. The modest Pairwise Accuracy
gain remains positive in workload and DAG-hash-family resampling. Exact uniform
Random expectation, a separate analytical reference, has mean regret 6.762689.

There are 110 live and ten cached Jev responses. Mean live total decision
latency is 465.156 ms; the 426.488 ms console mean mixes both. Long context and
these accounting distinctions must be explicit in the report. Raw measurements
and actual request payloads were not part of the uploaded audit bundle.

The [follow-up guide](run-tenset-followup-experiments.md) provides field-definition,
context/order, forced-group tournament, and resumable all-candidate workflows.
These are prepared execution profiles, not completed experimental results.

### Dataset preparation

- [x] Download, prepare, and run TenSet on the user's Mac, as demonstrated by
  the user-reported pilot.
- [x] Implement extraction of tensor workloads, candidate schedules, and
  measured runtimes, with fixture validation and a user-reported real-data pilot.
- [x] Select Intel E5-2673 v4 as the initial recorded hardware target.
- [ ] Establish a representative benchmark workload set and report its coverage.
- [x] Implement workload metadata, schedule configuration, and runtime extraction.
- [x] Define reproducible evaluation subsets and train/test splits.

### Workload serialization

- [x] Represent tensor shapes, operation types, loop structures, and scheduling
  transformations in `Workload.context` and `Candidate.configuration`.
- [x] Validate serialization against real TVM record structure and focused fixtures.
- [x] Verify that measured runtimes never appear in model-visible data.
- [x] Export semantic program metadata with portable Python and native format fixtures.
- [x] Measure serialized input bytes and enforce a configurable local budget.
- [ ] Verify actual service limits and full-dataset serialization coverage.

### Optional local XGBoost baseline

The implementation is retained for a future matched comparison. Training and
running it locally are optional; current experiments use Jev and Random.

- [x] Reuse the paper or official TVM XGBoost cost-model implementation where
  feasible.
- [ ] Optionally train or load the model with a documented training split.
- [ ] Optionally generate and validate rankings on the same held-out candidates.
- [x] Integrate XGBoost through `BaseDecisionModel` with a persistent legacy worker.

### Experiments and analysis

- [x] Run the user-reported Jev-versus-Random setup pilot on one workload and
  eight shared candidates.
- [x] Run and audit the 120 x 50 Jev-versus-Random evaluation on held-out workloads.
- [ ] Run field-definition and context/order ablations.
- [ ] Run the all-candidate tournament on the full eligible test collection.
- [ ] Select and verify published XGBoost results for a separately cited
  literature comparison, without treating them as paired measurements.
- [ ] Optionally evaluate the local XGBoost baseline on identical candidates.
- [x] Implement official Top-1/Top-5 Score and Pairwise Accuracy with parity tests.
- [x] Obtain user-reported Top-1/Top-5 Score, Pairwise Accuracy, and runtime
  regret for the one-workload pilot.
- [x] Independently inspect the pilot and 120 x 50 predictions and provenance.
- [ ] Compute benchmark metrics and regret over the full agreed evaluation set.
- [ ] Compare workload types and candidate-set sizes.
- [ ] Identify Jev success and failure cases.
- [ ] Analyze variation across workloads and seeds with appropriate statistics.
- [ ] Validate live/cached decision latency, API usage, and monetary cost.
- [x] Implement per-workload exports, aggregate analysis, and comparison plots.
- [x] Write the RQ1 methodology and explicitly pending results/discussion section.
- [ ] Complete the RQ1 results and discussion with real benchmark evidence.

### RQ1 deliverables

- [x] TenSet preprocessing and serialization pipeline.
- [x] Official XGBoost cost-model baseline integration (legacy execution pending).
- [ ] TenSet result tables and plots.
- [x] RQ1 report section with methodology, references, and explicit pending results.

## RQ2: XLA TpuGraphs — Person C

The separate [TpuGraphs branch](https://github.com/Jonghu/jev-ml-compiler-eval/tree/tpu-graph)
at `031371e` contains a `tile:xla` downloader, graph/configuration serialization,
official-validation-split experiment profiles, and Top-K Error/Kendall's Tau
implementations. That work has not been merged into main. The unchecked items
below remain main-branch integration or experiment deliverables; they do not
mean that the separate branch has no implementation. The GraphSAGE baseline
and final RQ2 results remain pending.

### Dataset preparation

- [ ] Download and set up TpuGraphs.
- [ ] Explore graph representations, candidate configurations, and runtime
  labels.
- [ ] Start with the `tile:xla` collection.
- [ ] Expand to layout configurations only if time and feasibility permit.
- [ ] Extract graph features, configuration parameters, and runtime labels.
- [ ] Use and document the official dataset splits.

### Graph serialization

- [ ] Represent operation types, graph connectivity, tensor shapes, and tile
  sizes in `Workload.context` and `Candidate.configuration`.
- [ ] Develop a compact summary for graphs that exceed Jev input limits.
- [ ] Validate serialized representations against source data.
- [ ] Verify that measured runtimes never appear in model-visible data.
- [ ] Analyze the effect of serialization and input size on decisions.

### GNN baseline

- [ ] Set up the official TpuGraphs GNN performance-prediction baseline.
- [ ] Train or load the provided GraphSAGE model where feasible.
- [ ] Generate candidate rankings.
- [ ] Integrate the GNN through `BaseDecisionModel`.

### Experiments and analysis

- [ ] Evaluate Jev, Random, and GNN on identical candidate sets.
- [ ] Implement and compute official Top-1 Error.
- [ ] Implement and compute Top-5/10 Error and Kendall's Tau.
- [ ] Compute runtime regret.
- [ ] Compare graph sizes, configuration types, and candidate-set sizes.
- [ ] Investigate Jev success and failure cases.
- [ ] Export per-graph predictions and aggregate results.
- [ ] Write the RQ2 results and discussion section.

### RQ2 deliverables

- [ ] TpuGraphs preprocessing and graph serialization pipeline.
- [ ] GNN performance-model baseline.
- [ ] TpuGraphs result tables and plots.
- [ ] RQ2 report section.

## Cross-team experiment agreement

### Before benchmark experiments

- [x] Define the common decision interface and result format.
- [x] Ensure models receive identical candidate sets within an experiment.
- [ ] Agree on candidate sampling strategies, evaluation budgets, and random
  seeds for both benchmarks.
- [x] Confirm TenSet's official metric formulas and references with parity tests.
- [ ] Complete the cross-dataset metric and reporting agreement, including the
  different TenSet Top-5 Score and TpuGraphs Top-5 Error definitions.

### During benchmark experiments

- [ ] Review each workstream's implementation and methodology.
- [ ] Review serialization design and API-limit handling.
- [ ] Maintain experiment logs for runtime, latency, usage, and cost.

### Final report

- [ ] Person A: methodology, Jev integration, evaluation framework, and overall
  comparison.
- [ ] Person B: TenSet experiment, results, and RQ1 discussion.
- [ ] Person C: TpuGraphs experiment, results, and RQ2 discussion.
- [ ] All members: introduction, related work, limitations, conclusions, and
  final proofreading.
