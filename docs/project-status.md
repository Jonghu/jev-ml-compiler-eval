# Project status

This checklist distinguishes completed shared infrastructure from the original
remaining plans for the TenSet and TpuGraphs benchmark workstreams.

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
- [ ] Add candidate-count and input-size policies for TypeSafe API limits.
- [ ] Add candidate batching or staged selection if a benchmark exceeds those
  limits.

### Shared evaluation

- [x] Implement the reproducible Random baseline.
- [x] Implement runtime regret and mock Top-1 accuracy.
- [x] Measure per-decision and aggregate latency.
- [x] Record API token usage and cache status.
- [x] Run multiple models on identical candidate sets.
- [ ] Implement benchmark-specific Top-K and ranking metrics.
- [ ] Convert token usage into monetary cost using a recorded pricing snapshot.
- [ ] Add configurable candidate sampling strategies and evaluation budgets.
- [ ] Support `fail_fast: false` with explicit partial-result semantics.
- [ ] Generate cross-model comparison tables and plots.

### Integration and final analysis

- [ ] Review and merge the TenSet workstream.
- [ ] Review and merge the TpuGraphs workstream.
- [ ] Verify reproducibility across both benchmark pipelines.
- [ ] Analyze the overall performance-cost tradeoff.
- [ ] Consolidate cross-dataset tables and plots for the final report.

## RQ1: TVM TenSet — Person B

### Dataset preparation

- [ ] Download and set up TenSet.
- [ ] Explore tensor workloads, candidate schedules, and measured runtimes.
- [ ] Select representative workloads and a target hardware platform.
- [ ] Extract workload metadata, schedule configurations, and runtime labels.
- [ ] Define reproducible evaluation subsets and train/test splits.

### Workload serialization

- [ ] Represent tensor shapes, operation types, loop structures, and scheduling
  transformations in `Workload.context` and `Candidate.configuration`.
- [ ] Validate that serialization preserves relevant optimization information.
- [ ] Verify that measured runtimes never appear in model-visible data.
- [ ] Measure serialized input sizes against Jev API limits.

### XGBoost baseline

- [ ] Reuse the paper or official TVM XGBoost cost-model implementation where
  feasible.
- [ ] Train or load the model with an explicitly documented training split.
- [ ] Generate candidate rankings.
- [ ] Integrate XGBoost through `BaseDecisionModel`.

### Experiments and analysis

- [ ] Evaluate Jev, Random, and XGBoost on identical candidate sets.
- [ ] Implement and compute official Top-1 Score.
- [ ] Implement and compute Top-5 Score and Pairwise Accuracy.
- [ ] Compute runtime regret.
- [ ] Compare workload types and candidate-set sizes.
- [ ] Identify Jev success and failure cases.
- [ ] Export per-workload predictions and aggregate results.
- [ ] Write the RQ1 results and discussion section.

### RQ1 deliverables

- [ ] TenSet preprocessing and serialization pipeline.
- [ ] XGBoost cost-model baseline.
- [ ] TenSet result tables and plots.
- [ ] RQ1 report section.

## RQ2: XLA TpuGraphs — Person C

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
- [ ] Confirm the exact official metric formulas and references.

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
