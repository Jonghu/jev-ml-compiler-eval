# Run log — RQ1 TenSet

A record of completed TenSet experiments and separately labelled offline
analyses. This follows the TpuGraphs [run-log format](https://github.com/Jonghu/jev-ml-compiler-eval/blob/61fe406/docs/runlog.md).
Dates below use America/Chicago; UTC completion timestamps are also given.
The paired comparison is Jev versus Random. The main-paper MLP + Ranking loss
values are an external published reference, not a locally trained baseline.
The completed primary ResNet-50 run is R6 below; R1–R5 retain their historical
cohorts and metrics. Prepared follow-up profiles are not completed runs.

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

The [RQ1 report](rq1-report.md) treats both protocols separately under a
Jev-versus-Random scope. At R4 publication, the
[finish guide](run-tenset-finish-rq1.md) prepared operation/type exports, concrete
paired schedule cases, and the controlled 16/32/50 E5 size study. The subsequent
user-completed size analysis is recorded as R5 below. Full semantic
case-study exports remain separate from its metric-only examples.
Optional prompt/order/algorithm controls, fresh repeats, and all-source-schedule
tournaments are not required completion items for the revised scope.

## R5 — Completed E5 16/32/50 candidate-size study

- Completion evidence: the user uploaded the combined local analysis after
  executing the 16- and 32-candidate conditions. Exact new execution dates,
  raw prediction bundles, and new run manifests were not supplied.
- Reported protocol: the same 120 E5 workloads at each budget; seed 42, no
  field definitions, single-question Jev, and long context. The uploaded
  analysis supports scope/count checks; the new raw configs are not included.
- Source input: the analysis records prepared-file SHA-256
  `1f949c2c9f788b507c4e395925720c7beaa7063c3efb6ac09ae5ed4855db6275`.
- The local sweep method declares nested 16/32/50 prefixes of the frozen
  candidate order. Both models appear on all 120 workload IDs at each size.
- The 50-candidate condition reuses R3's historical results. It is not a new
  Jev repeat and is not counted as another 120 independent workloads.
- Uploaded outputs include six candidate-budget/model aggregate rows,
  720 per-workload rows, operation-name tables, metric-only case examples,
  accounting, and a dedicated size-comparison figure with a method record.
- Reviewed publication: [size-study results](results/tenset/size-sweep/README.md).

### Per-budget results

All values below are workload macro means within one budget. With equal
candidate counts per workload, record-weighted and macro means coincide
within each row. Do not use the generic analyzer's across-budget overall
means as the candidate-size comparison.

| Candidates | Model | Workloads | Top-1 Score | Top-5 Score | Pairwise Accuracy | Tie-aware Pairwise Accuracy | Runtime regret |
|---:|---|---:|---:|---:|---:|---:|---:|
| 16 | Random-42 | 120 | 0.407439 | 0.615892 | 0.504097 | 0.504097 | 5.308322 |
| 16 | Jev 1.14, long context | 120 | 0.456949 | 0.645309 | 0.533403 | 0.533715 | 3.535879 |
| 32 | Random-42 | 120 | 0.348193 | 0.553470 | 0.503461 | 0.503461 | 5.317561 |
| 32 | Jev 1.14, long context | 120 | 0.394553 | 0.563148 | 0.529536 | 0.529940 | 3.993304 |
| 50 | Random-42, reused R3 | 120 | 0.332440 | 0.498775 | 0.491959 | 0.491959 | 6.058854 |
| 50 | Jev 1.14, reused R3 | 120 | 0.336867 | 0.512829 | 0.519456 | 0.519704 | 5.081011 |

Jev has higher observed Top-1, Top-5, and Pairwise Accuracy and lower mean
regret than Random within each budget. The observed Top-1 gaps are +0.049510,
+0.046360, and +0.004427 for 16, 32, and 50 candidates. Top-5 gaps are
+0.029417, +0.009678, and +0.014055, respectively. These mean differences
alone are not significance claims or evidence of fresh API repeatability.

Exploratory paired workload-bootstrap intervals use 50,000 resamples and
seed 20261116, with common workload draws across budgets. All Top-1 and
Top-5 intervals include zero. Strict Pairwise Accuracy differences have
positive intervals at 16 [+0.002708, +0.055625], 32 [+0.010232, +0.042104],
and 50 [+0.014293, +0.040789]. The mean-regret difference at 32 is
−1.324258 [−2.647424, −0.112540]; the other regret intervals include zero.
These are unadjusted comparisons conditional on saved responses, without
verified DAG-family clustering. Both 16-minus-50 and 32-minus-50 contrasts
of model advantages include zero for every tested metric. The smaller
budget's larger observed Top-1 gap is not an established scaling effect.

A supplementary 56-group leading DAG-hash bootstrap uses the historical
120-workload mapping, 50,000 shared group draws, and the same seed. It keeps
members together and uses a ratio of summed differences to workload counts.
Pairwise Accuracy intervals remain positive (the 16-schedule lower bound is
only +0.000926), while Top-1/Top-5 and budget-advantage contrasts still cross
zero. The 32-schedule regret interval remains negative. These hash groups
are not verified independent networks; the supplement is a sensitivity check,
not a multiplicity correction. See the published cluster-sensitivity tables.

Both models' normalized selection scores decrease across the larger pools,
but every budget has its own candidate-pool oracle. A score change between
sizes is not a physical runtime speedup or slowdown. Statistical comparisons
must pair the same workload IDs and must not treat the 720 rows as 720
independent tensor programs. The dedicated size figure uses per-budget
values; the uploaded generic overall figure pools sizes and serves a
different descriptive purpose.

### Operation-name and case outputs

The generic analyzer now provides 162 per-operation rows: 27 literal operation
signatures for two models at three budgets. These names come from the saved
program metadata and include fused-operation signatures. They support a
descriptive workload-type breakdown; ambiguous names such as `compute` are
not silently interpreted as a more specific operator family.

The uploaded `jev_cases.csv` supplies ten post-hoc metric examples with
workload IDs, operation names, candidate counts, and score/regret differences.
It does not contain tensor shapes, selected candidate transformation arguments,
or complete Jev/Random/oracle schedules. Full semantic case narratives still
require the Mac's prepared data and the offline case exporter.

### Verification limits and completion status

Aggregate means can be checked against the uploaded per-workload rows, and
the plot-method hashes bind the supplied analysis tables. The 50-candidate
aggregate matches the earlier audited R3 result. The local analysis declares
the fixed prepared-data identity and controlled nested-pool method.

The new 16/32 prediction rows, model-input hashes, request metadata, raw
runtime maps, manifests, and prepared shards were not supplied. The cloud
review therefore does not independently repeat the source-level schedule,
label, model-version, or API-usage audit for those new runs. No exact new
execution timestamps or billing totals are inferred from aggregate tables.
These limits do not turn the user-completed runs back into prepared profiles;
they distinguish completed analysis evidence from a full raw-artifact audit.

The exported accounting reports 120 live requests at each new size, with
2,750,775 input tokens at 16 candidates and 4,255,904 at 32. Their combined
7,006,679 input tokens correspond to an estimated $0.294281 at the assumed
$0.042 per million input tokens, with free output. No invoice or dated
pricing snapshot was supplied. The c50 accounting describes historical
calls and is excluded from incremental sweep usage.

The requested candidate-size comparison and operation-name breakdown are now
completed and exported. The remaining substantive RQ1 task is the offline
semantic tensor/schedule case export and its final report narrative. Optional
fresh repeats or prompt/algorithm controls do not require rerunning these
completed size conditions merely to regenerate their plots.

## R6 — Completed main-paper-aligned ResNet-50 network study

- Aggregate manifest: 2026-11-18, 13:46 CST
  (`2026-11-18T19:46:01.580450Z`). This is the recorded manifest timestamp,
  not an inferred first-request or last-request timestamp.
- Execution commit: `f6a2a49f6b2d0dc8b3ae2e9f87ca6b3250d929ea`.
- Config: `experiments/tenset/paper-resnet50/full.yaml` at that commit.
- Original output: `results/tenset/paper-resnet50/full/`, retained outside Git.
- Reviewed publication: [paper-resnet50](results/tenset/paper-resnet50/README.md).
- Presentation: [four-slide setup/workflow/results/diagnosis contribution](presentations/tenset-rq1/README.md).

### Cohort and protocol

One reconstructed ResNet-50 network, batch 1, input `(1, 3, 224, 224)`, has
27 distinct tasks with occurrence weights summing to 56. It contains 100,126
candidate records: 25 pools of 4,000, one of 20, and one of 106. Its exact
recorded target is
`llvm -keys=cpu -link-params=0 -mcpu=skylake-avx512 -model=platinum-8272`.
All declared task records are retained, without 500-file sampling, a 90/10
record split, deduplication, or filtering by error code. The manifest reports
67 errors. Runtime maps contain 67 expected failure-sentinel values; raw error
flags were not supplied for independent status verification. None of the
selected best-of-1/best-of-5 labels is below the evaluator throughput floor.

Jev 1.14.0 uses full program/schedule context, field descriptions, opaque
aliases, seeded candidate presentation, long context, at most 255 choices per
question, and five promotions per group. Token guards may produce smaller
groups. Questions rank the explicit choice first, then decreasing probability,
then seeded presentation order for ties. The same frozen candidate pools are
used by Random-0. The Mac runs the API/evaluation client and does not newly
execute or time the CPU kernels.

The [pinned network evaluator](https://github.com/tlc-pack/tenset/blob/35774ed373f9693cc6efcbe83f5703141e8ab0c8/scripts/eval_model_on_dataset.py)
computes `sum(weight * full-task minimum) / sum(weight * best first-K latency)`.
Latency is recovered in float64 from float32-origin throughput labels with
floor `1e-5`. These are task-frequency-weighted network endpoint scores, not
the older tutorial's record-weighted prefix-average scores. Use
`paper-network-summary.json` rather than the generic collection diagnostics.

### Results and external reference

| Model and evidence | Network Top-1 ↑ | Network Top-5 ↑ |
|---|---:|---:|
| Random-0, paired local run | 0.091461940470 | 0.377121152526 |
| Jev, paired local run | 0.097724226968 | 0.274582561989 |
| MLP + Ranking loss, external published Table 4 reference | 0.8823 | 0.9456 |
| Jev minus saved Random | +0.006262286499 | −0.102538590537 |

MLP + Ranking loss is a neural-network cost model, not XGBoost. The paper's
original inventory and checkpoint are unverified, and the MLP was not trained
or evaluated locally. Its larger published values provide cross-study context;
they do not establish a matched MLP-versus-Jev effect or exact Table 4 reproduction.
The reconstruction covers only the ResNet-50 column, not all five held-out networks.

The optional aggregate proxy regret `1 / Network Score@1 - 1` is 9.232877 for
Jev and 9.933510 for Random. It refers to the same occurrence-weighted latency
sums, not physical end-to-end network execution or a mean of task regrets.
The reciprocal identity must not be applied to the older macro score means.

### Independent saved-output checks and sensitivity

The audit verifies all 54 complete rankings, exact paired candidate order,
runtime maps and model-input hashes for every task, and exact deterministic
replay of all 27 Random-0 rankings. All 108 task/K decompositions and both
network score pairs match independent float32-label/float64-latency arithmetic
exactly. All 40 implementation source hashes in the manifest match the
recorded checkout. No concrete scoring or setup defect was found in the
supplied exports; their gap from the paper reference is not evidence of a bug.

| Original uploaded artifact | SHA-256 |
|---|---|
| `predictions.jsonl` | `a87404603313bf84f3057ce91cd9363fb8e61ba7af90ac28c58e168d09201fcc` |
| `manifest.json` | `f3d6f73a3af166042707837b4ab7ff34526ee87fe26ba22d359ffbb2eb1dae5c` |
| `paper-network-summary.json` | `cc16e48fa1097fce92e3c3b22500f148088fd6e4d91f8e0e2c03bb41d76954f0` |
| `paper-task-decompositions.csv` | `58eebade5f71719ed00cd0f91865ba5f7f11a805f84e32ea267bee231ab789d0` |

The declared collection index hash is
`d010a639fdd0d499511e4c9ac6b5137dd7825dae660c87211c4e143e3b28bb3c`;
the effective configuration hash is
`82b4d9532d9ae0b2e8e9319b6055a0d2f6e18f60062614e8685f6f3aa811cdb3`.
Publication preserves these original identities. Later integration changes
code hashes, so replay/resume uses the historical checkout and dependencies;
old provenance must not be rewritten to authorize a newer implementation.

Jev has a lower latency proxy on 14/27 tasks at K=1 and 10/27 at K=5.
Omitting either of two influential tasks reverses the small Top-1 advantage.
The Top-5 deficit persists under all one-task and leading-DAG-hash-group
omissions. Task-mixture resampling is descriptive sensitivity, not a confidence
interval across networks: 27 tasks and 56 occurrences are components of one
network, not independent network observations.

A separate 100,000-draw offline uniform-Random reference samples ordered choices
without replacement within each fixed task pool, using seed 20261118. Mean
scores are 0.074093103281/0.347948805434. Fixed Jev Top-1 exceeds 77.934% of
Random draws. At K=5, 95.320% of draws exceed fixed Jev, with no score ties.
These fractions and percentile ranges are not p-values, population confidence
intervals, or fresh Jev inference replicates.

### Tournament and finalist diagnosis

There are 23 two-round, two one-round, and two three-round task tournaments,
with 673 questions in total. All 25 multi-round tasks lose a pool-optimal
record in round 0: 22 below the probability cutoff and three tied at it but
excluded. Final Top-5 cutoff ties occur in 25/27 tasks. A runtime-best surviving
finalist is included in the actual Top-5 in only two tasks; it is tied at the
cutoff but excluded in 14 and below the cutoff in 11. Median distinct probability
levels per question are four. Do not equate the 25 boundary ties with 25
best-finalist exclusions by ties.

The retrospective oracle-aware score within surviving final pools is 0.742401
versus achieved Jev Top-5 0.274583. The final-ranking stage accounts for 86.866%
of saved excess weighted Top-5 latency proxy. These label-dependent upper
bounds/decompositions describe recorded decisions, not attainable predictions,
causal ablations, or model reruns. Probabilities are only comparable within a
question, so no global tournament Pairwise Accuracy is reported.

### Usage and cost

| Accounting item | Recorded value |
|---|---:|
| Successful live / cached requests | 631 / 42 |
| New live input tokens | 95,175,703 |
| Historical cached input tokens | 5,585,849 |
| All stored input tokens | 100,761,552 |
| New live output tokens | 1,061,953 |
| Assumed input rate | $0.042 per million |
| Estimated new successful input cost | $3.997379526 |
| Total recorded decision latency, including cached cases | 1,133.867633 seconds |

The estimate covers new input only. Output charges, unobserved retries/errors,
custom pricing, taxes and credits are excluded; cached tokens represent prior
usage. Confirm the actual tariff and reconcile billing before calling this an
invoice total. Decision latency is client/model time, not physical ResNet runtime.

### Evidence limits and remaining work

Raw source measurements, the original task pickle, preparation state/index,
prepared shards, and transmitted request bodies were not uploaded. The audit
checks consistency of saved outputs and the matching implementation; it does
not independently validate original paper pool/checkpoint identity or actual
wire contents. The [source audit](tenset-paper-protocol.md) follows the pinned
executable evaluator; supplemental appendices were inaccessible. Missing raw
artifacts are a verification limit, not a demonstrated code defect.

The completed run, numerical review, plots, accounting, and slides need no
additional paid calls. Remaining report deliverables are the Mac's concrete
tensor/schedule case exports and reviewed semantic explanations, integration
into the team report, and pricing/billing validation. Fresh Jev calls and
controlled order, tie-breaking, context/field-description, and grouping
ablations are optional follow-ups. Matching trained-baseline reproduction
is outside the current scope; the MLP reference remains external.

## R7 — Completed three-repeat context ablation and filtered network follow-up

This entry updates the historical prepared-only follow-up status recorded
above without rewriting those earlier observations. Results were reviewed on
2026-11-25 from the uploaded `tenset-context-ablation-summary` bundle. The
[reviewed publication](results/tenset/context-ablation/README.md) contains the
compact descriptor of the frozen study, derived CSV summaries and PNG/SVG figures; the
[workflow guide](run-tenset-context-ablation.md) retains execution commands.

- Protocol: `tenset_context_ablation_v1`; model `jev-1.14.0`, long context,
  opaque aliases and seed 42.
- Frozen study identity:
  `5b7e11785c8e2deaa109d8226a663e5e2e4c02fc6dc01aa1bf32245c4abed3bd`.
- Small stage: the existing E5 120 × 50 original pools; A0 naive, A1 candidate
  definitions, A2 graph specification, A3 both, A4 ascending/descending
  structural order and A5 structural diversity filtering to ten schedules.
- Three fresh repeats per condition with fixed presentation seed; every shown
  pool fits one Choice question. The uploaded accounting reports no cache hits.
- Required full stage: three A5 repeats on all 27 ResNet-50 task pools,
  occurrence weight 56 and 100,126 original candidate records; at most 50
  schedules are shown per task. This is not a full-network factorial or the
  whole TenSet dataset.
- Ordering/filtering use schedule structure without runtime labels. The
  diversity filter is a different rule from TpuGraphs' large-tile heuristic.
- All scores retain the original pool denominator. Full-stage scoring uses
  the private float32 normalizers and throughput floor of the pinned paper
  evaluator. Hidden filter tails do not support global Pairwise Accuracy.

### Small-stage observations

Mean scores over the fixed 120 workloads; Top-1 ± is the sample standard
deviation across three run means. Prefix Top-5 averages the first five
prefix-best ratios; endpoint Top-5 uses the best of the first five picks.

| Condition | Top-1 Score, mean ± SD | Prefix Top-5 | Endpoint Top-5 | Mean regret |
|---|---:|---:|---:|---:|
| A0 naive | 0.358186 ± 0.004510 | 0.518579 | 0.629279 | 5.073117 |
| A1 candidate definitions | 0.321786 ± 0.002126 | 0.500551 | 0.613974 | 5.374682 |
| A2 graph specification | 0.355713 ± 0.002391 | 0.522320 | 0.634412 | 4.960330 |
| A3 both | 0.331918 ± 0.007598 | 0.497509 | 0.604623 | 5.222691 |
| A4 ascending structural order | 0.338018 ± 0.004870 | 0.503314 | 0.618841 | 4.537227 |
| A4 descending structural order | 0.317597 ± 0.020314 | 0.483385 | 0.601717 | 7.087368 |
| A5 diversity filter | 0.340978 ± 0.003721 | 0.495122 | 0.597623 | 5.520364 |
| Random-42, one seeded draw | 0.332440 | 0.498775 | 0.622915 | 6.058854 |

A1 and A3 have lower cohort mean Top-1 than A0 in every repeat. Across
workloads, A1 improves 32, worsens 41 and ties 47; A3 improves 34, worsens 39
and ties 47. Their exploratory sign-test p-values are 0.349 and 0.640. These
win/loss tests do not test mean-effect magnitude: a nonsignificant sign test
is not evidence that the observed mean decline is random. A2 differs little
from A0 on Top-1 and has a small endpoint Top-5 advantage. The results do not
establish a general benefit from adding candidate or graph descriptions.

Ascending structural order has lower mean regret than descending, but the
per-workload Top-1 direction is mixed: descending improves 54 cases, worsens
60 and ties six relative to ascending. The Pairwise Accuracy gap
(0.519934 versus 0.508059) has an unadjusted sign-test p = 0.0353 among many
comparisons. Structural order must not be read as TPU tile size. Unfiltered
same-pick agreement over the three repeats ranges from 65.0% to 70.8%; A5
agreement is 82.5%. These are limited fresh-response repetitions, not a claim
that the service is deterministic.

A5 retains an original-pool optimum on 22/120 workloads. Exact uniform Random
inside its retained pool has Top-1 score 0.308215 and endpoint Top-5 score
0.601597. Jev's Top-1 is higher, but its endpoint Top-5 is slightly lower.
A5 uses about 60.1% fewer input tokens than A3; the mean run-median decision
latency is 189.1 ms versus 485.7 ms. Its endpoint Top-5 is also lower
(0.597623 versus 0.604623), so this is a cost/coverage tradeoff rather than
an across-the-board quality gain.

### Full-stage observations

| Model | Weighted network Top-1 | Weighted network Top-5 |
|---|---:|---:|
| A5, three-repeat mean ± SD | 0.113422 ± 0.007755 | 0.343155 ± 0.024797 |
| Random-42, one seeded draw | 0.137327 | 0.406334 |

A5 is below this paired Random draw on both network endpoints. The separate
macro task scores are 0.267820/0.433057 and must not replace the weighted
network values. The filter retains an optimum on 1/27 tasks; its weighted
filter-oracle score ceiling is 0.713744. Full-stage first-pick agreement is
66.7% over three repeats. The earlier R6 Jev tournament result
0.097724/0.274583 and Random-0 result 0.091462/0.377121 remain historical
measurements: changing context, seed and selection algorithm prevents treating
the cross-run difference as a controlled filter-only effect.

### Usage, review limits and remaining work

| Phase | Live requests | Input tokens | Output tokens | Input-cost estimate |
|---|---:|---:|---:|---:|
| Small: seven conditions × three repeats | 2,520 | 115,674,831 | 1,059,840 | $4.858342902 |
| Full: A5 × three repeats | 81 | 4,246,344 | 37,422 | $0.178346448 |
| Complete study | 2,601 | 119,921,175 | 1,097,262 | $5.036689350 |

The assumed input rate is $0.042/million tokens. These completed-study totals
include saved pilot decisions reused through the runner's receipt-based
resume; do not add an earlier pilot estimate again. The upload reports zero
cached requests. Output fees, unrecorded retries and invoice reconciliation
are excluded. No additional Jev calls were made for this review/publication.

Only the frozen study manifest and derived summaries were uploaded. Their
aggregate arithmetic and coverage can be reviewed, but raw predictions,
completion receipts and request/response caches are still needed to reproduce
request-level integrity and label-boundary checks independently. Raw archive
sharing with Jonghu remains a manual delivery step; this publication does
not send it. The RQ1 report incorporates the findings. Concrete semantic
schedule cases, team-report integration, billing validation and any update
to the historical presentation remain separate tasks.
