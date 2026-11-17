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
It covers 484 retained workloads and 187,679 test records. The empirical paired
comparison is Jev versus Random. Published XGBoost scores are reported separately
as external context, rather than as a third locally evaluated model.

This study retains error records and uses tournament selection metrics. Keep
its aggregates separate from the earlier valid-only E5 single-question study
below, which supplies full-ranking Pairwise Accuracy and runtime regret.

## Published XGBoost tutorial reference

The [separate reference record](tutorial-500-platinum-test/xgboost-tutorial-reference.json)
cites the pinned official 500-file tutorial: XGBoost Top-1 Score 0.829146,
Top-5 Score 0.904636, and Pairwise Accuracy 0.846439. Its reported training/test
record counts are 1,689,022/187,679. The original inventory, exact test records,
and checkpoint have not been matched, so the values are external context and
are not labelled paper-table results or a controlled three-model comparison.

The tutorial contains no published Random baseline. Our Random-0 and Random-42
results are local measurements paired with Jev on our own pools. Adding this
reference requires no new API calls or model training and does not alter the
frozen experimental outputs below.

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
