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

The current research question is **Jev versus Random**. Specialized-model
comparison and training are outside the agreed RQ1 scope. Completed evidence
includes the E5 120-workload single-question experiment, its 16/32/50-candidate
size study, and the Platinum 484-workload record-test tournament. Their
protocols and aggregations remain separate. The new size-study upload contains
analysis exports; it does not include the new raw prediction/manifest bundles.

The [RQ1 report](rq1-report.md) now contains the completed methodology, results,
uncertainty, cost, and limitations. The [run log](tenset-runlog.md) retains
historical runs and the [finish guide](run-tenset-finish-rq1.md) gives the
remaining semantic-case export commands and records how the completed size
study was executed. Prepared optional profiles remain distinct from completed
experiments.

### Completed E5 single-question experiment

The run at `6a4fe91` uses 120 project-test workloads, 50 valid distinct sampled
schedules each, seed 42, program metadata, no field definitions, and long
context. Both models receive identical pools. Pairwise Accuracy and runtime
regret are reported for this full-ranking, valid-only protocol.

| Model | Top-1 Score | Top-5 Score | Pairwise Accuracy | Runtime regret |
|---|---:|---:|---:|---:|
| Random-42 | 0.332440 | 0.498775 | 0.491959 | 6.058854 |
| Jev 1.14 long context | 0.336867 | 0.512829 | 0.519456 | 5.081011 |

The Pairwise Accuracy difference is 2.75 percentage points. Exploratory paired
intervals include zero improvement for Top-1, Top-5, and regret. Jev chooses a
faster schedule on 58 workloads, a slower one on 60, and an equal runtime on two;
both models select the sampled optimum on 5/120 cases. Outliers affect the
lower Jev mean regret. There are 110 live and ten cached responses; the live
total decision mean is 465.156 ms. See the
[reviewed E5 outputs](results/tenset/w120-c50-s42-long-context/metrics.csv).

### Completed E5 16/32/50 candidate-size study

The user completed the new 16- and 32-schedule conditions and uploaded their
combined analysis with the historical 50-schedule anchor. Every budget has
120 workloads for each model: six aggregate rows and 720 per-workload rows,
representing the same 120 workload IDs across budgets. The source analysis
records the audited prepared-file checksum and the fixed E5/seed-42 protocol.
The 50-schedule row is reused evidence, not a new API repeat.

| Schedules | Model | Top-1 Score | Top-5 Score | Pairwise Accuracy | Runtime regret |
|---:|---|---:|---:|---:|---:|
| 16 | Random-42 | 0.407439 | 0.615892 | 0.504097 | 5.308322 |
| 16 | Jev | 0.456949 | 0.645309 | 0.533403 | 3.535879 |
| 32 | Random-42 | 0.348193 | 0.553470 | 0.503461 | 5.317561 |
| 32 | Jev | 0.394553 | 0.563148 | 0.529536 | 3.993304 |
| 50 | Random-42 | 0.332440 | 0.498775 | 0.491959 | 6.058854 |
| 50 | Jev | 0.336867 | 0.512829 | 0.519456 | 5.081011 |

Use the [reviewed size-study exports](results/tenset/size-sweep/README.md) and
dedicated size figure; the generic overall comparison pools budgets. Jev has
higher observed scores/Pairwise Accuracy and lower mean regret within each
budget. Exploratory paired workload-bootstrap intervals are positive for
Pairwise Accuracy at each size and include zero for Top-1 and Top-5 at all
sizes. The 32-schedule regret interval excludes zero; the other two include
zero. These intervals are unadjusted for several comparisons and do not model
verified DAG-family dependence. Cross-budget intervals do not establish a
change in Jev's advantage. Changing pool oracles also prevents interpreting
a cross-size score change as a physical speedup.

The upload also provides operation-name breakdowns for 27 literal operation
signatures and metric-only success/failure examples. Full tensor shapes and
chosen/oracle schedule configurations are still absent. The new aggregate and
per-workload exports can be checked internally, but new raw predictions,
manifests, input hashes, and prepared candidate IDs were not uploaded for an
independent source-level repetition of the original 50-schedule audit. Exact
new execution timestamps are not supplied by these analysis exports.

### Completed Platinum record-test tournament

The run at `ed73f19` uses the tutorial's batch-size-1 file-selection procedure,
500 sampled workload files, 484 retained workloads, and 187,679 test records.
The reverse-order 90/10 split is within each workload. It retains error records;
the reported test pools include 2,013 error records. This is not a
workload-disjoint test set or proof of identity with unpublished original
artifacts. Jev and Random use the same reconstructed pools.

Jev uses field definitions, long context, neutral aliases, shuffled presentation,
and a maximum of 255 options per question with five advancing from each group.
468 workloads require two rounds and 16 require one. The primary results weight
each workload by its number of test records.

| Model | Record-weighted Top-1 Score | Record-weighted Top-5 Score |
|---|---:|---:|
| Random-0 | 0.235045 | 0.385021 |
| Jev 1.14 tournament | 0.269356 | 0.388245 |

The Jev-minus-Random exploratory 95% paired weighted intervals are
[+0.009015, +0.059739] for Top-1 and [−0.018344, +0.024936] for Top-5.
Exact uniform-Random expectation, a separate analytical reference, is
0.239404/0.378379; the positive Top-1 signal remains under that reference.
Jev selects the test-pool optimum on 9/484 cases, versus 6/484 for Random.
Among 468 two-round cases, the optimum is eliminated in round 0 on 442.
These are saved-path diagnostics, not evidence of a counterfactual winner.

The completed run records 1,840 live and 16 cached requests. New input usage
is 179,522,529 tokens; total stored-response input usage including earlier
pilot responses is 181,403,479. At the assumed $0.042 per million input tokens
with free output, estimates are $7.539946 new and $7.618946 including the pilot.
They overlap and must not be added; neither is an invoice or a spending cap.

See the [reviewed Platinum results](results/tenset/tutorial-500-platinum-test/README.md).
Global Pairwise Accuracy is not defined by the tournament's deterministic
elimination tail. Failure-sentinel penalties are not interpreted as physical
runtime slowdowns. The older one-workload pilot remains a setup check, and
old numerical specialized-model comparisons remain historical context only.

### Dataset and serialization

- [x] Download, prepare, and evaluate real TenSet CPU measurements from the Mac.
- [x] Preserve exact E5-2673 and Platinum-8272 hardware-target metadata.
- [x] Implement and document both project-workload and within-workload record splits.
- [x] Preserve distinct-valid-schedule and tutorial record-level policies separately.
- [x] Serialize shapes, dtypes, operation names, tensor-expression graphs, loops,
  and scheduling transformations with portable Python and native format fixtures.
- [x] Keep runtime/error/record-index labels outside model-visible fields in the
  implemented interface and verify shared recorded input fingerprints.
- [x] Measure input sizes; implement token-aware grouping and explicit long context.
- [x] Obtain E5 operation-name coverage and per-operation scores from the
  completed size-analysis export; 27 literal signatures are represented.
- [ ] Complete tensor/schedule semantic case exports from the Mac's frozen data.

The uploaded audit files contain no raw measurements, prepared shards, private
float32 throughput labels, or actual request payloads. Saved-record consistency
is verified; independent raw-content and request-label-isolation audits remain
limited by the supplied artifacts.

### Metrics, experiments, and reviewed outputs

- [x] Run and audit the original setup pilot and 120 × 50 single-question comparison.
- [x] Run and audit the 484-workload Platinum record-test tournament.
- [x] Report Top-1/Top-5, Pairwise Accuracy, and valid-only runtime regret for E5.
- [x] Report record-weighted Top-1/Top-5 and separate macro scores for Platinum.
- [x] Compute paired workload uncertainty and separately labelled exact Random expectations.
- [x] Inspect tournament elimination, coarse probabilities, optimum hits, and
  failure-sentinel selection counts without additional API calls.
- [x] Separate live/cached latency and usage; report assumed-rate cost estimates.
- [x] Publish reviewed per-workload/aggregate exports, figures, and provenance.
- [x] Prepare offline operation/type and paired-case export commands.
- [x] Review the uploaded E5 operation-name and per-budget metric breakdowns.
- [ ] Run and review the Mac's concrete tensor/schedule case-study exports.
- [x] Prepare a controlled 16/32/50 candidate-size sweep on the frozen E5 cohort.
- [x] Complete the user-run 16/32 conditions, reuse the 50-candidate archive,
  and obtain the combined per-workload tables and dedicated size plot.
  Raw new prediction bundles were not uploaded for a source-level audit.
- [x] Write the RQ1 report with completed results and explicit remaining analyses.
- [x] Incorporate the completed size and operation-name analyses into the report.
- [ ] Add and review concrete tensor/schedule explanations for the selected cases.

### Prepared main-paper ResNet-50 comparison

The [new workflow](run-tenset-paper-resnet50.md) targets one main-paper Table 4
column: ResNet-50, batch 1, input224, Platinum-8272. Its MLP + Ranking loss
reference is 0.8823/0.9456 for Top-1/Top-5. This reference is separate from
the completed tutorial experiment and remains unverified against the original
checkpoint and record inventory.

- [x] Prepare complete network task/weight/record inventories from ZIP or raw data.
- [x] Reconstruct the pinned network-weighted best-of-K endpoint evaluator.
- [x] Keep intermediate prefix metrics separate and suppress full-network
  reference comparisons for pilot subsets.
- [x] Prepare a three-row full-run comparison export: local Jev, local Random,
  and the separately labelled published MLP + Ranking loss reference.
- [x] Provide offline-first validation and token/cost scenarios, plus explicitly
  selected execution with verified per-task resume.
- [ ] Run local-data preflight and review the actual scope and estimated cost.
- [ ] Execute and review the pilot and full-network Jev/Random experiment.

No new benchmark outcome or paid model usage is claimed. The
[protocol audit](tenset-paper-protocol.md) documents the inaccessible appendices,
native label/error behavior and remaining limits of the published comparison.

### Optional extensions and retained tooling

Field-only ablations, context/order controls, forced-group tournament controls,
fresh Jev repeats, and all-source-schedule experiments remain available in the
[follow-up guide](run-tenset-followup-experiments.md). They are useful extensions,
not required completion items under the present Jev-versus-Random scope.
The specialized-model adapter remains optional legacy code; its training,
artifacts, and numerical comparison are not RQ1 blockers. Historical tutorial
commands may still acknowledge external-reference provenance without changing
the measured comparison to a three-model study.

### RQ1 deliverables

- [x] Reproducible preprocessing, metadata, serialization, and evaluation workflows.
- [x] Audited Jev-versus-Random tables, figures, per-workload results, and accounting.
- [x] RQ1 methodology, completed-results discussion, and stated limitations.
- [x] Completed candidate-size and operation-name evidence with dedicated plots.
- [ ] Reviewed semantic tensor/schedule case narratives incorporated into final RQ1.

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
- [x] Person B: standalone TenSet methodology, results, and RQ1 discussion draft.
- [ ] Person B: integrate the RQ1 draft into the team report and add the
  remaining tensor/schedule case explanations before final proofreading.
- [ ] Person C: TpuGraphs experiment, results, and RQ2 discussion.
- [ ] All members: introduction, related work, limitations, conclusions, and
  final proofreading.
