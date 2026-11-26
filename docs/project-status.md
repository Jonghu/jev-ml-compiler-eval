# Project status

Status reviewed **2026-11-25**, including the completed TenSet context ablation.
This checklist separates implemented software,
completed experiments, reviewed artifacts, and remaining report work. Both
benchmark implementations are integrated in this revision. TpuGraphs PR #1
entered `origin/main` at `fbd67c1`; this integration retains that implementation
alongside the completed TenSet work. The 2026-11-19 integration passed **740
tests**, with Ruff formatting and lint checks. The current integration pull
request records the additional ablation checks and final hosted CI outcome.

See the [TenSet finish guide](run-tenset-finish-rq1.md),
[RQ1 report](rq1-report.md), and [TpuGraphs run log](runlog.md) for evidence and
commands. Historical run logs retain the observations made at each run.

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
- [x] Convert single-question Choice probabilities into complete rankings;
  keep tournament probabilities scoped to their individual questions.
- [x] Record resolved model, confidence, probabilities, token usage, request ID,
  cache status, and latency.
- [x] Add configurable TenSet candidate/workload budgets and a local input-byte
  guard.
- [x] Implement token-aware candidate grouping, multi-question request packing,
  tournament advancement, and client-side request/token rate limits.
- [x] Exercise the recorded budgets in full TpuGraphs validation and TenSet runs.
- [ ] Independently verify a current provider limit/pricing snapshot and review
  actual request payloads where the uploaded evidence does not include them.

### Shared evaluation

- [x] Implement the reproducible Random baseline.
- [x] Implement runtime regret and mock Top-1 accuracy.
- [x] Measure per-decision and aggregate latency.
- [x] Record API token usage and cache status.
- [x] Run multiple models on identical candidate sets.
- [x] Implement and test the TenSet Top-1/Top-5 Score and Pairwise Accuracy
  definitions.
- [x] Implement shared TpuGraphs Top-1/5/10 Error and Kendall's Tau; these are
  present in the merged TpuGraphs implementation.
- [x] Convert recorded live/cached input usage into explicitly assumed-rate cost
  estimates; keep cached prior charges separate from new successful usage.
- [ ] Verify those estimates against a dated provider price or billing record.
- [x] Add deterministic TenSet candidate sampling and evaluation budgets.
- [x] Version each benchmark's seed, candidate policy, token budgets, and scope.
- [ ] Finalize the cross-dataset reporting rationale for the different sampling,
  API modes, metrics, and aggregation policies.
- [ ] Optional: support `fail_fast: false` with explicit partial-result semantics;
  current experiments deliberately use fail-fast execution.
- [x] Implement TenSet comparison tables and plots from saved result bundles.
- [ ] Generate reviewed benchmark comparison tables and plots across datasets.

### Integration and final analysis

- [x] Merge the earlier TenSet implementation into main and prepare the later
  audited studies, analysis tooling, and presentations for integration.
- [x] Merge the TpuGraphs workstream into main through PR #1 (`fbd67c1`).
- [x] Resolve shared-code conflicts and validate the combined TenSet/TpuGraphs
  implementation with all 740 tests.
- [x] Preserve per-run configurations, implementation identities, input hashes,
  and reviewed TenSet prediction/summary audits.
- [ ] Consolidate the project-wide performance-cost narrative and final tables;
  component run logs and an earlier cross-dataset table already exist.

## RQ1: TVM TenSet — Person B

The local empirical comparison is **Jev versus Random**. Published learned-model
values may be shown as explicitly external literature references; they are not
locally trained or paired baselines. Matching or training those models is not a
required blocker for the agreed report scope.

Completed evidence includes the E5 120-workload single-question experiment, its
16/32/50-candidate size study, the Platinum 484-workload record-test tournament,
the main-paper-protocol ResNet-50 network evaluation, and the completed
three-repeat context ablation with its filtered full-network follow-up. Their
scopes, metric formulas, and aggregations remain separate. The size-study upload
contains analysis exports rather than new raw prediction/manifest bundles.

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

### Completed main-paper ResNet-50 network evaluation

The [ResNet-50 workflow](run-tenset-paper-resnet50.md) has completed at commit
`f6a2a49`. It covers all 27 network tasks, total occurrence weight 56, and
100,126 candidate records (25 pools of 4,000 records, plus pools of 20 and 106)
on Platinum-8272. Both models use the same complete pools.

The primary endpoint is the pinned evaluator's task-frequency-weighted network
best-of-K score, including its throughput floor. It differs from the E5 macro
prefix score and the 484-workload record-weighted prefix score; do not compare
those aggregates as one metric.

| Model | Network best-of-1 Score | Network best-of-5 Score |
|---|---:|---:|
| Random-0 | 0.091462 | 0.377121 |
| Jev tournament | 0.097724 | 0.274583 |
| Published MLP + Ranking loss, external reference | 0.8823 | 0.9456 |

Jev's single-run Top-1 score is slightly higher than Random's, while its Top-5
score is lower. In 100,000 offline uniform-Random draws, approximately 95.3%
score above the fixed Jev Top-5 result. That is a conditional random-reference
frequency, not a p-value or an estimate of fresh Jev response variation.
Only 2/27 task selections put the best available finalist in the first five;
25/27 final questions have a probability tie at the fifth-place cutoff. These
are recorded-path diagnostics, not proof that a different algorithm would win.

The audit reproduces all 108 saved network task decompositions and checks 40
recorded implementation hashes. The full run records 673 stored responses:
631 live and 42 cached. Its 95,175,703 new input tokens correspond to an
estimated $3.997380 at the assumed $0.042 per million input tokens with free
output. Raw measurement/feature and original paper-checkpoint identity limits
remain explicit. The external MLP values are a literature reference, not a
locally reproduced or paired model result.

- [x] Complete local-data preparation, preflight, and full-network evaluation.
- [x] Verify full task/weight coverage, candidate identity, and endpoint arithmetic.
- [x] Audit saved scores, request usage, finalists, and probability-cutoff ties.
- [x] Prepare the current four-slide presentation and reviewed network outputs.

See [reviewed ResNet-50 results](results/tenset/paper-resnet50/README.md),
[protocol notes](tenset-paper-protocol.md), and the
[current presentation](presentations/tenset-rq1/README.md). No additional paid
experiment is implied by the remaining report tasks.

### Optional extensions and retained tooling

The controlled candidate-definition, graph-specification, presentation-order,
and structural-filter study with three fresh repeats is completed; see the
[reviewed ablation results](results/tenset/context-ablation/README.md). The older
no-program-context profile, forced-group tournament and tie-breaking controls,
repeats on other cohorts, and expanded-network experiments remain extensions
in the [follow-up guide](run-tenset-followup-experiments.md).
The specialized-model adapter remains optional legacy code. Published MLP
values are clearly labelled external references in the network study; local
specialized-model training or matching artifacts is optional reproduction work,
not an agreed RQ1 completion blocker. Historical tutorial status fields are
provenance rather than a requirement to execute another model.

### RQ1 deliverables

- [x] Reproducible preprocessing, metadata, serialization, and evaluation workflows.
- [x] Audited Jev-versus-Random tables, figures, per-workload results, and accounting.
- [x] RQ1 methodology, completed-results discussion, and stated limitations.
- [x] Completed candidate-size and operation-name evidence with dedicated plots.
- [ ] Reviewed semantic tensor/schedule case narratives incorporated into final RQ1.

## RQ2: XLA TpuGraphs — Person C

The `tile:xla` pipeline and experiments R1–R7 are implemented and recorded in
[the run log](runlog.md). TpuGraphs PR #1 is merged into `origin/main` at
`fbd67c1`; both implementations are present in this integration revision.
The previous status text describing an unmerged `031371e` branch was obsolete.

### Completed dataset, serialization, and metrics

- [x] Implement the official `v0` train/valid/test downloader with archive
  checksum verification.
- [x] Evaluate on labelled official `valid`: 676 kernels from seven programs.
  The public `test` runtime labels are zero and are not used for scored runs.
- [x] Serialize node opcodes, shapes/dtypes, graph connectivity, layouts,
  convolution dimensions, and tile-configuration bounds without timing labels.
- [x] Compact serialization by removing trailing padding and derivable features.
- [x] Preserve the official runtime-normalizer ranking target and document
  merging duplicate configuration rows using the fastest normalized timing.
- [x] Implement Top-1/5/10 Error, runtime regret, Top-1 accuracy, and Kendall's Tau.
- [x] Exercise the serialization and budgets on the complete labelled split.
- [ ] Optional: extend beyond `tile:xla` to layout configurations or graph summaries.

### Completed experiments and interpretation

- [x] R1–R4: Random and Jev connection/subset runs, including a fresh five-kernel
  rerun that observed changed probabilities despite unchanged Top-1 choices.
- [x] R5: separate no-context and shuffled-order variants on 120 kernels.
- [x] R6: field-description comparison on the same 120-kernel candidate pools.
- [x] R7: blind shuffled-ID tournament on all 676 validation kernels and all
  1,038,376 distinct configurations, using at most 255 choices per question.
- [x] Report per-program and candidate-count breakdowns, probability ties,
  selection failures, outliers, latency, cache status, and input/output usage.
- [x] Record completed bundle paths in the run log and commit the log/configs;
  generated raw bundles remain outside Git.

R6 reduced Jev's mean Top-1 error from 4.237 to 0.904 on the fixed subset;
Random's was 0.766. In full R7, Jev has mean Top-1 error **413.6%** versus
Random's **268.4%**, but lower Top-5 error **48.9% versus 57.6%** and lower
Top-10 error **24.2% versus 35.1%**. R7 reports 1,819 requests, including 16
cached probe responses; estimated input cost is approximately $2.25 new or
$2.28 including the probe, using the recorded assumed rate. These are run-log
reports; the raw TpuGraphs result bundles were not supplied for the independent
TenSet-style audit.

The R7 global Kendall value is diagnostic only: eliminated candidates are not
a globally learned ranking. The comparison has the paper's overall validation
scope but merges duplicate configurations, so it must not be described as an
identical original-paper evaluation pool.

### References and remaining report work

- [x] Cite published EarlySAGE/GraphSAGE paper values as external references.
- [x] Analyze compiler-default configuration selection offline as a project
  reference; it is not a registered decision model.
- [ ] Optional: train/load/integrate a local GraphSAGE model if a paired learned
  baseline reproduction is chosen. No such model or training run is implemented.
- [ ] Write the standalone RQ2 final report section from the completed run log
  and review its reference/metric caveats with the team.
- [ ] Incorporate reviewed RQ2 figures and exports into the final team report.
- [ ] Optional: extend fresh-request variance and field/ID-order causal controls;
  the existing five-kernel repeat and combined R7 changes do not isolate all effects.

## Cross-team experiment agreement

### Before benchmark experiments

- [x] Define the common decision interface and result format.
- [x] Ensure models receive identical candidate sets within an experiment.
- [x] Record benchmark-specific candidate policies, evaluation budgets, and seeds
  in version-controlled configurations.
- [ ] Explain those differences in the final cross-dataset reporting agreement.
- [x] Confirm TenSet's official metric formulas and references with parity tests.
- [ ] Complete the cross-dataset metric and reporting agreement, including the
  different TenSet Top-5 Score and TpuGraphs Top-5 Error definitions.

### During benchmark experiments

- [x] Maintain both workstreams' run logs, configurations, latency and usage records.
- [x] Implement candidate/label separation, cache-order identity, and token-aware
  tournament handling; retain the limitations of raw-payload evidence.
- [ ] Complete final integration validation and the shared methodology review.
- [ ] Verify current pricing assumptions and final performance-cost comparisons.

### Final report

- [ ] Person A: methodology, Jev integration, evaluation framework, and overall
  comparison.
- [x] Person B: standalone TenSet methodology, results, and RQ1 discussion draft.
- [ ] Person B: integrate the RQ1 draft into the team report and add the
  remaining tensor/schedule case explanations before final proofreading.
- [ ] Person C: TpuGraphs experiment, results, and RQ2 discussion.
- [ ] All members: introduction, related work, limitations, conclusions, and
  final proofreading.

## Completed TenSet context-engineering ablation (2026-11-25)

Integration validation passes all 818 local tests, Ruff formatting and linting.
The reporting regression checks cover Random reference lines and complete
error bars in the comparison plots. Publication required no new Jev calls.

The [context-ablation workflow](run-tenset-context-ablation.md) has completed
seven conditions with three fresh repeats on the frozen E5 120 × 50 cohort,
followed by the required three-repeat, filtered full ResNet-50 evaluation.
Candidate and graph descriptions are independent factors. Ordering and
filtering use runtime-blind schedule structure; this diversity filter differs
from TpuGraphs' large-tile heuristic. The full stage covers 27 ResNet-50 task
pools, occurrence weight 56 and 100,126 original candidate records; it shows
at most 50 schedules per task. It is not a full-network factorial or the whole
TenSet dataset.

The small phase favors naive context on mean Top-1 score (0.358186), versus
0.321786 with candidate definitions, 0.355713 with graph specification and
0.331918 with both. Graph specification has the highest endpoint Top-5 mean
(0.634412). Ordering effects are uneven across workloads. The filter retains
a pool optimum on 22/120 small-phase workloads and 1/27 full-phase tasks;
its full-network oracle score ceiling is 0.713744.

| Full ResNet-50 follow-up | Network Top-1 score | Network Top-5 score |
|---|---:|---:|
| A5 structural diversity filter, mean ± sample SD | 0.113422 ± 0.007755 | 0.343155 ± 0.024797 |
| Random-42, one seeded draw | 0.137327 | 0.406334 |

The uploaded accounting reports 2,601 live requests, zero cached requests,
119,921,175 input tokens and an input-cost estimate of **$5.03668935** at the
assumed $0.042/million rate. This completed-study accounting includes its
saved pilot decisions; a separate pilot charge must not be added again.
The small/full-stage estimates are $4.858342902/$0.178346448. Output fees,
unrecorded retries and billing reconciliation are not included.

- [x] Prepare the study configuration, offline cost preflight, and explicit
  paid-execution commands.
- [x] Complete the Mac preparation, preflight, pilot and both experiment stages.
- [x] Complete all seven small-stage conditions and the full-stage A5 condition,
  with three fresh repeats each.
- [x] Review the uploaded derived summaries, paired comparisons, filter baselines,
  usage and figures; incorporate them into the RQ1 report.
- [ ] Optional independent verification: supply raw predictions, completion
  receipts and Jev caches for a request-level audit. The summary-only upload
  limits verification, but does not make the completed runs incomplete.
- [ ] Manually share the raw experiment archive with Jonghu; no delivery is
  claimed by this repository update.

See the [reviewed ablation publication](results/tenset/context-ablation/README.md)
for source identity and tables. Three-repeat variation is now measured on
these fixed inputs. Exploratory sign tests address workload win/loss directions;
they are not tests of mean-effect magnitude or guarantees of broad generalization.
The earlier four-slide contribution remains the historical ResNet-50 deck and
has not been regenerated for this ablation publication. Semantic tensor/schedule
case narratives, final team-report integration and billing validation remain
separate report tasks.
