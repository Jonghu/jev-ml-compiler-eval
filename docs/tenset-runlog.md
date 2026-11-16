# Run log — RQ1 TenSet

A record of completed TenSet experiments and separately labelled offline
analyses. This follows the TpuGraphs [run-log format](https://github.com/Jonghu/jev-ml-compiler-eval/blob/61fe406/docs/runlog.md).
Dates below use America/Chicago; UTC completion timestamps are also given.
The current comparison is Jev versus Random. Prepared follow-up profiles are
not completed runs; older optional comparisons remain historical context.

## E5 shared setup — R1–R3

| Item | Value |
|---|---|
| Dataset | Official TenSet CPU v3.3, Intel E5-2673 v4 collection |
| Exact target | `llvm -keys=cpu -link-params=0 -mcpu=core-avx2 -model=e5-2673` |
| Experimental unit | One tensor workload and its candidate schedules |
| Evaluation | Offline selection against existing CPU measurements; requests run from the user's Mac |
| Measurements | Mean timing repetitions per record, then mean repeated schedule records; seconds converted to milliseconds |
| Invalid records | Failed, empty, nonfinite, nonpositive, and failure-sentinel timings excluded |
| Splits | Project-defined workload-key hash, seed 42: approximately 70% train, 10% validation, 20% test |
| Candidate sampling | Seeded hash of workload ID and schedule ID, independent of runtime values |
| Models | Uniform Random ranking, seed 42; pinned `jev-1.14.0` |
| Jev question | "Which candidate is most likely to have the lowest runtime for this workload?" |

Top-1 and Top-5 Score are higher-is-better normalized-throughput ratios.
Top-5 Score averages the best score at each prefix of the first five choices;
it is not just the best-of-five endpoint. Pairwise Accuracy follows TenSet's
strict comparisons, with a separately named tie-aware diagnostic. Runtime
regret is `selected / fastest_candidate - 1`, where the oracle is limited to
the candidates presented in that run. R1–R3 results are workload macro means;
R4 below reports test-record-weighted selection scores separately.

Original generated bundles remain outside Git, as in the TpuGraphs workstream.
Reviewed tables, figures, audit methods, and input checksums are published under
[docs/results/tenset](results/tenset/README.md). A published derived table is
not an additional experiment or an original result-bundle file.

## R1 — One-workload setup pilot, eight schedules

- Completed: 2026-11-05, 22:31 CST (`2026-11-06T04:31:48.980732Z`).
- Recorded code commit: `de0fb6ae34e340696ce867938c2733564ca52e58`.
- Config: `experiments/tenset/jev-vs-random.yaml` at that commit.
- Output: `results/tenset/random-vs-jev-c8-s42/`, outside Git.
- Scope: one workload, eight schedules, `split: all`; this case is train-labelled
  by the project hash split and is a setup check, not a held-out benchmark.
- Jev used default context; requested and resolved versions were `jev-1.14.0`.
- One live Jev call: 16,460 input tokens, 275 output tokens; API latency
  554.943 ms and total decision latency 555.785 ms.

| Model | Workloads | Schedules | Top-1 Score | Top-5 Score | Pairwise Accuracy | Runtime regret |
|---|---:|---:|---:|---:|---:|---:|
| Random-42 | 1 | 8 | 1.000000 | 1.000000 | 0.821429 | 0.000000 |
| Jev 1.14 | 1 | 8 | 1.000000 | 1.000000 | 0.571429 | 0.000000 |

The uploaded five-file bundle was independently inspected. Both models selected
the unique fastest candidate, 0.23013625 ms; the second-fastest took
0.479597125 ms. Jev correctly ordered 16 of 28 pairs, versus Random's 23.
The second-fastest schedule appeared last in Jev's ranking.

The perfect selection score is exact, with no runtime ties or rounding issue.
It describes one successful choice, not 100% accuracy across TenSet. The optimum
also happened to be the first presented candidate, so this case alone cannot
rule out presentation-order effects. Metadata enrichment preserved an outdated
status string in the original provenance; it did not affect metric computation.

## R2 — Default-context, 50-schedule canary rejected

- CLI profile: `experiments/tenset/local/jev-vs-random-w10-c50-s42.yaml`,
  generated locally before the long-context change.
- The API returned HTTP 400 with `error_type: max_tokens_exceeded`.
- No completed result bundle or success metrics were supplied for this run.

Offline diagnostics showed request sizes of approximately 26–216 KB among
the first ten cases. One program context alone was about 80 KB, so reducing
candidate count alone could not guarantee admission. The local 262,144-byte
guard was not a service token limit. No claim is made about charges for failed
requests, which are not included in the successful-response accounting below.

## R3 — 120 test workloads, 50 sampled schedules, long context

- Completed: 2026-11-11, 23:20 CST (`2026-11-12T05:20:00.163188Z`).
- Recorded code commit: `6a4fe91585df2f91e347bfbc7be2fb8bde325df0`.
- Tracked reference config at that commit:
  `experiments/tenset/jev-vs-random-w120-c50-s42-long-context.yaml`.
- Original output: `results/tenset/random-vs-jev-w120-c50-s42-long-context/`,
  outside Git. The original saved `experiment.yaml` was not uploaded for this
  run; the tracked reference is not represented as a copy of that file.
- Environment: Python 3.11.18, TypeSafe SDK 0.7.5, NumPy 2.4.6,
  Matplotlib 3.11.2.
- All 120 decisions requested and resolved `jev-1.14.0`; prediction metadata
  records `long_context: true` throughout. This opts into the documented
  `TypeSafe-Danger-Zone-Long-Context: true` header.
- Scope: 120 project-test workloads, exactly 50 sampled schedules each,
  required tensor-program metadata, seed 42, one Choice question per workload.
- Source preparation indexed 2,308 platform files and 448 test groups. It
  processed 122 groups to retain 120 eligible workloads; the two exclusions
  had 19 and 20 valid distinct schedules. Processed records: 448,713 total,
  439,211 valid, 9,502 invalid.

| Model | Workloads | Schedules each | Top-1 Score | Top-5 Score | Pairwise Accuracy | Tie-aware Pairwise Accuracy | Runtime regret |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random-42 | 120 | 50 | 0.332440 | 0.498775 | 0.491959 | 0.491959 | 6.058854 |
| Jev 1.14, long context | 120 | 50 | 0.336867 | 0.512829 | 0.519456 | 0.519704 | 5.081011 |

Exact values are in [metrics.csv](results/tenset/w120-c50-s42-long-context/metrics.csv).

| Selection diagnostic | Value |
|---|---:|
| Jev selects a faster schedule than Random | 58 / 120 |
| Jev selects the same runtime | 2 / 120 |
| Jev selects a slower schedule | 60 / 120 |
| Jev selects the unique sampled optimum | 5 / 120 (4.17%) |
| Random selects the unique sampled optimum | 5 / 120 (4.17%) |
| Jev optimum appears in its first five predictions | 14 / 120 |
| Random optimum appears in its first five predictions | 13 / 120 |

Top-1 Score 0.336867 is not a 33.7% optimum-hit rate. Mean regret is 16.1%
lower than the observed Random-42 result, but that average is sensitive to
large gains on a few workloads. Median regret is 3.081662 for Jev and
3.025676 for Random. All 120 cases remain in the primary result.

![Paired slowdown and regret differences for all 120 workloads](results/tenset/w120-c50-s42-long-context/paired-comparison.png)

### Integrity audit

The uploaded `predictions.jsonl` and `manifest.json` contain 240 predictions,
120 per model. Every pair has identical ordered candidate IDs, runtime maps,
and recorded model-input fingerprints. All five metrics recompute exactly for
all rows. Every Random ranking reproduces from seed 42 and its workload hash;
Jev probabilities cover the exact candidate set, are finite, sum to one, and
agree with the selected candidate and saved ranking.

The manifest's source workload keys reproduce the declared test splits and
workload IDs. The [provenance record](results/tenset/w120-c50-s42-long-context/provenance.json)
includes input-file and prepared-data SHA-256 hashes. Raw measurements, the
prepared artifact, and actual request payloads were not uploaded, so runtime
aggregation, semantic completeness, and absence of labels inside actual
requests were not independently rechecked from these two files.

## A1 — Offline paired uncertainty and outlier analysis

This analysis uses the saved R3 choices and labels, with no additional API calls.
The primary diagnostic resamples workload pairs together, with 50,000 bootstrap
replicates and seed 20261112. Intervals are percentile 95% intervals, exploratory
and unadjusted for multiple metrics. Deltas below are **Jev minus Random-42**;
negative regret differences favor Jev.

| Metric | Mean difference | Paired workload-bootstrap 95% interval |
|---|---:|---:|
| Top-1 Score | +0.004427 | −0.059112 to +0.068515 |
| Top-5 Score | +0.014055 | −0.036992 to +0.065099 |
| Pairwise Accuracy | +0.027497 | +0.014292 to +0.040782 |
| Runtime regret | −0.977843 | −2.982630 to +0.906991 |

The Pairwise Accuracy difference is 2.75 percentage points. It remains positive
in a separate sensitivity check resampling 56 leading DAG-hash families,
including a largest family of 25 workloads. These hashes are not verified
independent networks. The intervals therefore describe exploratory uncertainty
for this cohort and do not establish performance on unseen programs or hardware.

The three largest favorable regret differences are 57.502979, 35.828305,
and 32.024633. Together they exceed the total net regret improvement across
all cases. Removing those cases changes the mean Random-minus-Jev regret gap
to −0.068502. This is a post-hoc sensitivity diagnostic, not a reason to remove
them from the reported experiment. Median paired runtime ratio Jev/Random is
1.025154, consistent with the nearly even per-workload win/loss count.

Probability ties are frequent: each Jev response has only three to nine distinct
probability values, and 30 workloads have multiple candidates tied for maximum
probability. The official and tie-aware aggregate accuracies are nevertheless
close; the report retains both definitions.

## A2 — Exact uniform-Random expectation, separate analytical reference

All 50 candidate runtimes are available for each R3 workload. Uniform Random's
expected Top-1 Score and regret can therefore be computed exactly by averaging
over its 50 possible first choices. Expected Top-5 Score uses the exact minimum
order-statistic distribution for each prefix, without enumerating 50! rankings.
This is not another run and does not replace the saved Random-42 values.

| Reference | Top-1 Score | Top-5 Score | Runtime regret |
|---|---:|---:|---:|
| Exact uniform-Random expectation | 0.318459 | 0.494099 | 6.762689 |
| Observed Random-42 | 0.332440 | 0.498775 | 6.058854 |
| Observed Jev 1.14, long context | 0.336867 | 0.512829 | 5.081011 |

Against this exact reference, mean Jev-minus-expected-Random regret is
−1.681678. Its exploratory DAG-family bootstrap interval is −2.476098 to
−0.629261; Top-1 and Top-5 difference intervals still include zero. This supports
lower average regret than uniform Random expectation on the fixed cohort,
while selection performance against the single observed Random-42 ranking
remains mixed. The reference method and per-workload values are published in
[expected-random.json](results/tenset/w120-c50-s42-long-context/expected-random.json).

## A3 — Live usage, cached responses, latency, and estimated cost

Ten cached responses are exactly the first ten workload inputs, consistent
with the preceding canary. R3 itself generated 110 live responses. There are
120 distinct request IDs across the saved responses.

| Accounting item | Value |
|---|---:|
| Live / cached decisions in the full run | 110 / 10 |
| New live input / output tokens | 5,418,271 / 163,609 |
| Input / output tokens across all 120 distinct responses | 5,943,452 / 178,485 |
| Previously cached input tokens, not new usage | 525,181 |
| Mean live API latency | 462.818 ms |
| Mean live total decision latency | 465.156 ms |
| Mean cached total decision latency | 1.139 ms |
| Mixed live/cached mean printed by the CLI | 426.488 ms |
| Input tokens per response, minimum / maximum | 15,936 / 138,439 |

At the [listed TypeSafe rate](https://docs.typesafe.ai/models) of $0.042 per
million input tokens, with output unbilled, the successful **new live inputs**
are estimated at **$0.227567**. All 120 distinct stored responses correspond
to **$0.249625**, including the ten responses paid for previously. These two
estimates must not be added together. They are not invoices and exclude failed
requests, other runs, custom pricing, taxes, and credits. Per-workload accounting
is in [jev-accounting.csv](results/tenset/w120-c50-s42-long-context/jev-accounting.csv).

Report live latency separately from cached access. Long context is an explicit
experimental condition; the [provider documentation](https://docs.typesafe.ai/models#long-context)
describes slower and less accurate behavior at longer contexts. Most R3 inputs
are above the default context budget, so these measurements cannot be labelled
a default-context reproduction of TpuGraphs.

## A4 — Cross-dataset comparison with the TpuGraphs run log

TpuGraphs Top-1 Error has the same definition as TenSet runtime regret.
The following values use fractional regret throughout; TpuGraphs R7 percentages
were divided by 100. The TpuGraphs values are run-log reports, not independently
audited prediction bundles.

| Experiment | Workloads | Jev regret | Random regret | Jev regret change versus its Random baseline |
|---|---:|---:|---:|---:|
| TenSet R3, 50 sampled schedules, long context | 120 | 5.081 | 6.059 | −16.1% |
| TpuGraphs R4, at most 50 configurations | 120 | 4.237 | 0.766 | +453.1% |
| TpuGraphs R6, field descriptions | 120 | 0.904 | 0.766 | +18.0% |
| TpuGraphs R7, full blind tournament | 676 | 4.136 | 2.684 | +54.1% |

The pools, data distributions, and request protocols differ. TenSet samples
50 schedules, TpuGraphs R4/R6 filter kernels whose pools exceed 50, and R7 uses
all configurations through multiple questions at default context limits.
TenSet Pairwise Accuracy and TpuGraphs Kendall's tau are not interchangeable.
The linked [comparison CSV](results/tenset/cross-dataset-comparison.csv) retains
scope and evidence labels alongside the reported means.

## Historical findings from R1–R3

1. The setup pilot's perfect score was a correct one-case result, not an
   overall accuracy claim.
2. The 120 x 50 long-context experiment shows a modest positive ranking signal,
   with mixed selection performance and important outliers.
3. Jev has lower mean regret than exact uniform-Random expectation on these
   pools, while its small Top-1/Top-5 gains remain uncertain.
4. Cache-aware accounting matters: 426 ms is a mixed mean; the live total
   decision mean is approximately 465 ms.
5. Published XGBoost results are an external literature reference. No matched
   XGBoost evaluation or competitiveness claim is supported by these runs.

## Optional R3 follow-ups prepared, not reported as completed

- [ ] Field-description ablation on the same 120 x 50 cohort.
- [ ] No-context and shuffled-presentation controls.
- [ ] Blind single-question versus forced-group tournament comparison.
- [ ] Validation pilot for protocol development.
- [ ] Full eligible project-test collection, retaining every valid schedule.

Commands and methodological constraints are in the
[follow-up experiment guide](run-tenset-followup-experiments.md). These are
exploratory extensions of inspected test results; reserve untouched cases for
confirmatory claims after freezing further choices.

## R4 — 500-file Platinum record-test tournament

- Completed: 2026-11-16, 15:38 CST (`2026-11-16T21:38:42.173585Z`).
- Recorded code commit: `ed73f19184f0513d8ca3f88e381435d28c3229a8`.
- Config: `experiments/tenset/jev-tutorial-500-test.yaml`.
- Original output: `results/tenset/tutorial-500-platinum-test/`, outside Git.
- Uploaded for audit: `experiment.yaml`, `manifest.json`, `predictions.jsonl`,
  `official-weighted-summary.json`, and `official-weighted-summary.csv`.
- Platform: Intel Platinum 8272; exact recorded target
  `llvm -keys=cpu -link-params=0 -mcpu=skylake-avx512 -model=platinum-8272`.
- Environment: Python 3.11.18, TypeSafe SDK 0.7.5, NumPy 2.4.6,
  Matplotlib 3.11.2.
- Scope: 500 sampled batch-size-1 workload files, seed 0; 484 retained workload
  pools with 187,679 test records. 468 pools contain 400 records; the remaining
  16 contain 6–88. The minimum retained source workload has 48 records.
- Split: reverse source-record order, the first `int(0.9*N)` reversed indices
  assigned to the unused training portion and the remaining indices to test.
  This is within each workload, not a workload-disjoint holdout.
- Preparation reports 1,689,022 training records, 187,679 test records,
  1,876,701 retained records, and 340 records in excluded small workloads.
  It preserves measurement-error records and duplicates. The observed test
  pools contain 2,013 reported error records and no duplicate configurations.
- Jev uses field definitions, seeded shuffled neutral aliases, long context,
  at most 255 options per question, and five candidates advancing per group.
  Token-aware packing can use smaller groups. There are 468 two-round cases
  and 16 single-question cases.
- All 1,856 saved questions/requests resolve to `jev-1.14.0`.

The current research scope is **Jev versus Random**. The tutorial supplies a
reproducible selection/split procedure and metric context; no numerical
specialized-model comparison is required. Original tutorial artifact identity
and native TVM feature reproduction remain unverified and are not inferred
from matching record counts. This is not labelled a paper-table reproduction.

Reviewed derived outputs are in
[the Platinum result directory](results/tenset/tutorial-500-platinum-test/README.md).

| Model | Workloads | Test records | Record-weighted Top-1 Score | Record-weighted Top-5 Score |
|---|---:|---:|---:|---:|
| Random-0 | 484 | 187,679 | 0.235045 | 0.385021 |
| Jev 1.14, fields and long-context tournament | 484 | 187,679 | 0.269356 | 0.388245 |

The primary aggregation weights workloads by their number of test records.
The separate macro means are 0.249948/0.399532 for Random and
0.282998/0.400940 for Jev. The saved canonical runtime-based scores and the
private-label weighted summaries have distinct names and aggregation scopes.

Top-1 Score is a performance ratio rather than an optimum-hit percentage.
Jev selects the unique test-pool optimum on 9/484 cases, versus 6/484 for
Random. Jev has a higher Top-1 score on 278 cases and a lower score on 206;
its Top-5 score wins 238 and loses 246. There are no paired score ties.

Global Pairwise Accuracy is not reported: probabilities are only comparable
within a tournament question, and the eliminated tail is deterministically
ordered. Failure-sentinel penalties are not physical runtime measurements;
no average physical-runtime regret or speedup is inferred from them.

### Integrity audit

All 968 predictions are present, 484 per model. Every paired row has identical
ordered candidate IDs, runtime maps, recorded model-input fingerprints, and
byte counts. The manifest's realized pools agree with the predictions, Random-0
rankings reproduce exactly, and all saved generic Top-1/Top-5 metrics recompute
within `1.11e-16`.

Every tournament's seeded presentation, per-round partitions, question
probabilities, choice-first rankings, top-five advancement, final question,
and deterministic elimination order reconstruct from the saved metadata.
Probabilities are finite and sum to one in every question. All recorded package
source fingerprints agree with the recorded commit. The uploaded weighted
summary's prediction checksum and all collection-index hash references agree.
Runtime-ratio recomputation agrees with the private-label weighted scores
within `5.3e-10`; this is not exact independent reconstruction of the private
float32 labels.

Raw measurements, prepared shards, private throughput labels, actual request
payloads, and the request cache were not supplied. Consequently raw aggregation,
semantic completeness, and actual request-label isolation were not independently
rechecked. All audit conclusions retain those content-validation limits.

### Paired uncertainty and exact uniform-Random expectation

The paired workload bootstrap samples 484 workload pairs with replacement,
retains each sampled case's test-record weight, and recomputes weighted ratios
of sums. There are 50,000 percentile resamples with seed 20261116. Intervals
are exploratory, conditional on this file sample and saved responses, and
unadjusted for multiple metrics. Related workloads may be correlated; a
DAG-family mapping was not recoverable from the uploaded artifacts.

| Metric | Weighted Jev minus Random-0 | Exploratory 95% paired interval |
|---|---:|---:|
| Top-1 Score | +0.034310 | +0.009015 to +0.059739 |
| Top-5 Score | +0.003224 | −0.018344 to +0.024936 |

All saved runtime labels also support an exact uniform-Random expectation over
every candidate permutation. Top-1 is a direct average; Top-5 uses the minimum
order-statistic distribution for each prefix. This is an analytical reference,
not a fresh API experiment or a replacement for the observed Random-0 ranking.

| Analytical reference | Weighted Top-1 Score | Weighted Top-5 Score |
|---|---:|---:|
| Exact uniform-Random expectation | 0.239404 | 0.378379 |
| Observed Random-0 | 0.235045 | 0.385021 |
| Observed Jev | 0.269356 | 0.388245 |

Jev minus expected Random is +0.029951 for Top-1, with interval
[+0.012579, +0.048034], and +0.009865 for Top-5, with interval
[−0.005705, +0.025934]. The positive Top-1 mean signal persists against this
reference; Top-5 remains uncertain. Jev exceeds expected Random on 219 workloads
and falls below it on 265, so a positive weighted mean is not a universal
per-workload improvement.

### Tournament and probability diagnostics

The test-pool optimum is unique in all 484 workloads. Among 468 two-round
cases, the optimum is eliminated in round 0 on 442 (94.44%). The other 26
advance it; six select it and 20 do not. Among 16 single-question cases, three
select the optimum. This is recorded-path attribution, not evidence of what a
hypothetical single-question or different-grouping strategy would choose.

| Options per question | Questions | Option probabilities equal to zero | Median distinct probability values |
|---|---:|---:|---:|
| At most 50 | 479 | 0.00% | 9 |
| 51–150 | 1,189 | 25.38% | 3 |
| 151–255 | 188 | 62.57% | 4 |

Across all 1,856 questions, 61,591 of 194,539 option-probability entries are
zero (31.66%), and 837 questions have multiple candidates tied for maximum
probability. These bins differ in workload and tournament round and do not
constitute an isolated candidate-size ablation. Zero recorded probability is
not proof that a schedule cannot be optimal.

The saved runtime maps retain 2,013 failure-sentinel records across 145
workloads, matching the reported test error count. Jev selects a sentinel
record on eight cases, versus seven for Random; at least one appears in the
first five choices on 28 versus 22 cases. No claim that Jev avoids failed
configurations is supported.

### Usage, latency, and estimated input cost

| Accounting item | Value |
|---|---:|
| Live / cached requests in the completed run | 1,840 / 16 |
| New live input / output tokens | 179,522,529 / 1,974,500 |
| Previously cached input / output tokens | 1,880,950 / 13,018 |
| All distinct saved-response input / output tokens | 181,403,479 / 1,987,518 |
| Mean live API latency, per request | 1.022 s |
| Mean total decision latency, per workload, mixed live/cache | 4.102 s |
| Estimated new successful input cost | $7.539946 |
| Estimated total including earlier pilot responses | $7.618946 |

The 16 cached requests correspond to the first three workload inputs, with
8/4/4 saved responses. The pricing assumption is $0.042 per million input
tokens with output free. No invoice or dated provider-pricing artifact was
supplied. These estimates cover logged successful responses, excluding failed
or other requests, custom pricing, taxes, and credits. They overlap and must
not be added; cached tokens are prior usage rather than new full-run charges.

### Interpretation and current remaining work

R4 supports a positive weighted Top-1 signal against Random on these fixed
pools, with no clear Top-5 gain. It combines field definitions, blind
presentation, long context, and tournament selection. Comparing R4 with R3
cannot isolate any of those factors because hardware, workloads, splits,
pool sizes, record policy, and aggregation also change.

The [RQ1 report](rq1-report.md) now treats both completed protocols separately
under a Jev-versus-Random scope. The
[finish guide](run-tenset-finish-rq1.md) prepares the remaining operation/type
exports, concrete paired schedule cases, and controlled 16/32/50 E5 size study.
Those new exports and 16/32 runs are prepared, not reported as completed.
Optional prompt/order/algorithm controls, fresh repeats, and all-source-schedule
tournaments are not required completion items for the revised scope.
