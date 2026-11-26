# Reviewed TenSet results

The [TenSet run log](../../tenset-runlog.md) records completed runs and their
interpretation. This directory contains reviewed derived outputs for the final
report, following the project convention of keeping generated raw bundles
outside Git. Publication was explicitly requested by the TenSet workstream owner.

## 484 Platinum workloads: Jev versus Random

The completed [Platinum test-pool result publication](tutorial-500-platinum-test/)
contains record-weighted Top-1/Top-5 results, paired uncertainty, exact
uniform-Random expectations, natural candidate-count breakdowns, tournament
diagnostics, reviewed usage/cost tables, and three reproducible PNG/SVG figures.
It covers 484 retained workloads and 187,679 test records. The research scope is
Jev versus Random; no specialized-model comparison is required.

This study retains error records and uses tournament selection metrics. Keep
its aggregates separate from the earlier valid-only E5 single-question study
below, which supplies full-ranking Pairwise Accuracy and runtime regret.

## 120 E5 workloads at 16, 32, and 50 schedules

The completed [candidate-size comparison](size-sweep/README.md) keeps all
three budgets separate on the same 120 workload IDs. It publishes six model
means, all 720 model/workload/budget rows, per-budget curves, paired workload
uncertainty, operation-signature breakdowns, post-hoc examples, and accounting.
The 50-schedule anchor reuses the historical run below. The new 16/32 upload
contains analysis exports rather than raw predictions and manifests, so its
independent verification is limited to those exports.

Jev's observed means favor it at all budgets. Exploratory Pairwise Accuracy
intervals are positive; Top-1 and Top-5 intervals include zero. The apparent
larger Top-1 advantage at smaller budgets is not established by the paired
cross-budget intervals. Full tensor/schedule explanations still need the
offline prepared-data case export.

## Completed main-paper ResNet-50 network study

The [reviewed ResNet-50 publication](paper-resnet50/README.md) covers 27 network
task types, occurrence weight 56 and 100,126 candidate records. The historical
full-pool Jev tournament scores are 0.097724/0.274583 at network Top-1/Top-5,
versus Random-0 at 0.091462/0.377121. The
[source audit](../../tenset-paper-protocol.md) defines the full-record pools,
occurrence-weighted best-of-K endpoint and external MLP + Ranking loss reference.
The [paper run guide](../../run-tenset-paper-resnet50.md) retains reproduction and
offline preflight instructions. Tutorial and candidate-size scores use different
aggregations and cannot substitute for these network results.

## Completed context-engineering ablation and filtered network follow-up

The [context-ablation publication](context-ablation/README.md) contains seven
conditions × three fresh repeats on the same E5 120 × 50 pools, plus three
repeats of structural diversity filtering on the complete declared ResNet-50
inventory. Candidate definitions, graph specification and both are tested
separately; ordering uses schedule structure, and filtering is compared with
Random inside the retained pool and a filter-oracle bound.

Naive context has the highest small-phase mean Top-1 score (0.358186); graph
specification has the highest endpoint Top-5 mean (0.634412). Added candidate
definitions do not improve the cohort mean. The full filtered-stage network
scores are 0.113422 ± 0.007755 and 0.343155 ± 0.024797, versus one Random-42 draw
at 0.137327 and 0.406334. This full stage is A5 only, not a full factorial or
the whole TenSet benchmark. It is distinct from the older Random-0 tournament.

The uploaded summaries report 2,601 live requests and 119,921,175 input tokens,
an estimated $5.03668935 at $0.042/million input tokens, including the study's
saved pilot decisions. Only derived summaries and the frozen study manifest
were provided; raw predictions, completion receipts and caches were not uploaded
for independent request-level audit. Raw sharing with Jonghu remains a manual
step. The existing presentation has not been regenerated for these new results.

## 120 test workloads x 50 schedules, long context

Directory: [w120-c50-s42-long-context](w120-c50-s42-long-context/).

| File | Meaning |
|---|---|
| `provenance.json` | Recorded run commit, environment, input checksums, scope and derivation notes |
| `metrics.csv` | Metrics recomputed from the uploaded predictions; latency here is the original mixed live/cache mean |
| `paired-workloads.csv` | Paired per-workload metrics, candidate counts, selected timings, outcomes and recorded input hashes |
| `jev-accounting.csv` | Response tokens versus new live usage, cache status, and latency per workload |
| `integrity-checks.json` | Saved-record consistency checks and their limitations |
| `paired-statistics.json` | Independent metric recomputation and 50,000 paired workload-bootstrap replicates |
| `family-cluster-sensitivity.json` | Exploratory uncertainty using leading DAG-hash families |
| `expected-random.json` | Separately labelled exact uniform-Random expectations and per-workload values |
| `expected-random-family-sensitivity.json` | Cluster sensitivity for the exact expected-Random reference |
| `audit-summary.json` | Audit findings, usage totals, outlier diagnostics and public-rate cost estimates |
| `paired-comparison.png` / `.svg` | Publication-ready paired slowdown and regret-difference figure |

These are derived audit exports, not original copies of `metrics.json`,
`summary.csv`, or the original five-file result bundle. The original full-run
`predictions.jsonl` and `manifest.json` were uploaded for audit but remain outside
Git; their SHA-256 values are recorded in `provenance.json`. The tracked YAML at
the recorded commit is a reference configuration, not a supplied saved YAML.
Raw measurements, the prepared artifact, and actual request payloads were not
available for independent content validation.

The numerical audit JSON files are preserved. One temporary `/tmp` reference
in `expected-random-family-sensitivity.json` is replaced by the relative published
filename. No experimental values were changed by that path adjustment.

Bootstrap intervals are exploratory, paired by workload, conditional on the
cohort and saved decisions, and unadjusted for multiple metrics. DAG-hash families
are not verified independent networks. The analytically expected Random result
does not replace the observed `random-42` result or count as another experiment.
Prices are public-rate estimates, not billing records. Cached response tokens
are previous usage, not newly charged inputs in the full run.

[cross-dataset-comparison.csv](cross-dataset-comparison.csv) compares the shared
regret metric with TpuGraphs run-log aggregates. It labels the different scopes
and evidence levels; it is not a paired cross-dataset experiment.
