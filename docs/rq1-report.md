# RQ1: Tensor schedule selection on TenSet

## Research question and scope

Can Jev select faster tensor schedules than uniform Random from serialized
program and schedule descriptions? The current RQ1 comparison is **Jev versus
Random** on the same original evaluation pools within each experiment.
Filtered conditions also report uniform Random inside the retained pools. The
primary completed study reconstructs the main paper's ResNet-50 network evaluator.
The published MLP + Ranking loss result is a separate external reference;
that model was not trained or evaluated locally and is not a paired baseline.

Completed evidence includes the ResNet-50 network study, the E5 120-workload
single-question study with a 16/32/50 candidate-size sweep, the Platinum
484-workload tutorial test-record tournament, and a controlled three-repeat
context ablation with a filtered full-network follow-up. Their uploaded results
answer related questions and are reported separately because their cohorts,
candidate policies, and aggregation rules differ.
The [run log](tenset-runlog.md) records their provenance; the
[finish guide](run-tenset-finish-rq1.md) provides the execution and remaining
semantic case-analysis commands. The 16- and 32-schedule runs are completed
measurements; the original 50-schedule results are reused.

The [audited ResNet-50 results](results/tenset/paper-resnet50/README.md) and
[four-slide contribution](presentations/tenset-rq1/README.md) describe the
completed historical primary experiment. The
[context-ablation publication](results/tenset/context-ablation/README.md) adds
the completed repeated study. The existing slides have not been regenerated
for that addition. No new Jev requests were made to review or publish the
saved results.

## Dataset, serialization, and evaluation protocols

TenSet contains 51,577,248 program measurements across six hardware platforms
[1]. We use existing CPU measurements, with the exact target preserved in each
workload. A workload is a tensor program; a schedule is a compiler configuration
for that program. The Mac runs the evaluation client, while Jev inference runs
through the API. No CPU kernels are newly timed on the Mac.

The historical studies use the following protocols. The complete ResNet-50
network protocol and its distinct endpoint metric are specified below.

| Property | E5 single-question experiment | Platinum tutorial tournament experiment |
|---|---|---|
| Recorded hardware | Intel E5-2673 v4, AVX2 | Intel Platinum 8272, AVX-512 |
| Scope | 120 project-test workloads | 500 sampled files; 484 retained workloads |
| Candidates | Nested pools of 16, 32, and 50 valid distinct schedules; 1,920 / 3,840 / 6,000 per budget | All reconstructed test records; 6–400 per workload, 187,679 total |
| Split | Workload-key hash: approximately 70/10/20 train/validation/test | Reverse record-order 90/10 split within each workload |
| Record policy | Invalid timings/errors removed; duplicate schedule records averaged | Duplicate and error records retained; 2,013 test error records reported |
| Sampling seed | 42 | 0, batch-size-1 workload preset |
| Jev presentation | Original deterministic order, one question, no field definitions | Neutral aliases, seeded shuffled order, field definitions, tournament |
| API mode | Long context, pinned `jev-1.14.0` | Long context, pinned `jev-1.14.0` |
| Primary aggregation | Workload macro mean | Test-record-count weighted mean |
| Reported metrics | Top-1, Top-5, Pairwise Accuracy, runtime regret | Top-1 and Top-5 selection scores |

The E5 subset freezes 50 schedules independently of their runtime values, with
16- and 32-schedule prefixes for the size sweep. It
samples from workloads with sufficient valid candidates. This is a
project-defined workload split, not the official tutorial's within-task split.
The Platinum preparation follows the pinned tutorial's file-selection and
record-split procedures [2], including its minimum of 48 records per retained
workload. It reconstructs test pools rather than proving identity with the
original authors' unpublished artifacts. Matching the published record counts
alone does not establish a reproduction of a paper table. The 90% portion is
not supplied to Jev as training data, and these 484 cases are not 484 unseen
programs under a workload-disjoint split.

The implemented serialization retains tensor shapes and dtypes, operation names,
complete tensor-expression graphs, loop information, and the exact TVM target.
Schedule serialization retains initial state and each scheduling transformation
with its code and arguments. Workload keys alone are insufficient descriptions.
Measured timings, error codes, and record indices remain private labels rather
than model-visible fields. Paired predictions record identical candidate pools
and model-input fingerprints. Actual API payloads were not uploaded for an
independent content audit.

The tournament compares at most 255 options per question and advances five
from each group. Local token estimates can require smaller groups. In the
historical Platinum tutorial run, 468 workloads required two rounds and 16 required one;
all 1,856 saved requests resolved to the pinned model. The definitions and
tournament were evaluated together, so this run does not isolate the effect
of either change.

## Metrics and aggregation

For the historical E5 and 500-file tutorial studies, let `t*` be the minimum
runtime in a workload's evaluated pool and `b_j` the best runtime among its
first `j` predicted choices. Their task-level score is [3]:

```text
Top-k Score = mean(t* / b_j, j = 1..min(k, candidate_count))
runtime_regret = selected_runtime / t* - 1
```

Their Top-5 averages the prefix-best scores. The ResNet-50 network study
instead uses the best-of-five endpoint described below. Scores are normalized
performance ratios, not
optimum-hit accuracy. The oracle is restricted to each original evaluated
pool; the new filtered conditions retain that original-pool denominator.
For these historical task metrics, the reciprocal relationship between Top-1
and regret applies per workload, not by inverting an aggregate mean.

For the E5 single-question run, Pairwise Accuracy follows the official strict
comparison convention and preserves model score ties. A separately named
tie-aware diagnostic excludes equal measured timings and gives half credit
for predicted ties. Valid-only runtime regret describes measured schedules.

For both Platinum tournament protocols, probabilities are comparable within each
question. Eliminated candidates are appended using a deterministic rule;
that tail is not a global learned ranking. We therefore do not report global
Pairwise Accuracy. Retained failure-sentinel costs are artificial penalties,
so they are not interpreted as physical program runtime or used to claim a
mean runtime speedup. Selection scores and failure-selection counts remain
explicitly separate outcomes.

The E5 runs average workload metrics equally. Every case within a budget
has the same candidate count; the original anchor has 50 candidates per case.
The historical Platinum tutorial result weights each workload by its number
of test records, as in the tutorial evaluator [4]. Its unweighted macro scores are separately
labelled. These aggregations and the different pool/invalid-record policies
must not be mixed into one cross-run result.

## Completed primary study: ResNet-50 network selection

The user completed the full ResNet-50 run at commit
`f6a2a49f6b2d0dc8b3ae2e9f87ca6b3250d929ea`; its aggregate manifest was created
on 2026-11-18 at `19:46:01.580450Z`. The run contains **27 distinct tasks,
56 network task occurrences, and 100,126 schedule records** for batch 1 and
input `(1, 3, 224, 224)`, using recorded Platinum-8272 measurements. Twenty-five
pools have 4,000 records, one has 20, and one has 106. Every declared task and
its complete record pool is evaluated; no 500-file sampling, within-task
90/10 split, valid-only filtering, or schedule deduplication is applied.
The manifest reports 67 error records. The Mac runs the client; it does not
newly execute the measured CPU kernels.

The [published audit](results/tenset/paper-resnet50/README.md) verifies 54 saved
model predictions, identical ordered candidate pools/runtime maps/input hashes
for all 27 Jev/Random pairs, and exact replay of every Random-0 ranking. All
108 task/K decompositions match an independent reconstruction exactly, and
all 40 recorded implementation hashes match the experiment checkout. These
checks establish consistency of the supplied outputs and implementation.
The raw task pickle, source measurements, prepared shards, and transmitted
request bodies were not supplied for independent source/content verification.
No concrete scoring or setup defect was found in the supplied evidence.

### Network endpoint metric and comparison

The pinned network evaluator [6] weights each task by its occurrence count,
not its number of candidate records. Let `m_i` be the full-pool minimum
float32 cost and `q_ic = float32(m_i / cost_ic)` the normalized throughput.
It recovers the latency proxy `ell_ic = m_i / max(q_ic, 1e-5)` in float64 and
reports:

```text
Network Score@K = sum_i(w_i * m_i) /
                  sum_i(w_i * min(ell_ic for c in the first K choices))
```

Top-5 is the best of the first five choices. This differs from the historical
prefix-average task scores and record-weighted tutorial aggregate. The
shared runner's ordinary `top_1_score`/`top_5_score` remain intermediate
prefix diagnostics; the primary outputs are
`paper_network_best_of_1_score` and `paper_network_best_of_5_score` in
`paper-network-summary.json`. The original tutorial result cannot be inserted
into this network comparison.

| Model and evidence | Network Top-1 ↑ | Network Top-5 ↑ |
|---|---:|---:|
| Random-0, locally paired | 0.091462 | 0.377121 |
| Jev 1.14, locally paired | 0.097724 | 0.274583 |
| MLP + Ranking loss, external published reference | 0.8823 | 0.9456 |

Jev minus saved Random is **+0.006262** for Top-1 and **−0.102539** for Top-5.
The small observed Top-1 lead does not establish reliable superiority.
The derived network latency-proxy regret `1 / Network Score@1 - 1` is
9.232877 for Jev and 9.933510 for Random. This identity applies to these
same network latency sums; it is not a mean task regret, measured end-to-end
network slowdown, or a rule for inverting the older macro-averaged scores.

The MLP row comes from Section 5.2/Table 4 of the main paper [5]. It is a
trained neural-network cost model with ranking loss, not XGBoost. The paper
excludes the union of tasks from five held-out networks during training;
this reconstruction covers only its ResNet-50 column. Original task/candidate
inventory and trained checkpoint identity remain unverified. The published
values are numerically higher but do not support a matched MLP-versus-Jev
significance test or a claim of reproducing Table 4.

### Random variation, concentration, and tournament diagnostics

An additional offline reference samples 100,000 uniform ordered Random choices
without replacement within each fixed task pool (seed 20261118). Its mean
network scores are 0.074093/0.347949. The saved Jev Top-1 exceeds 77.934% of
these draws, while **95.320% of draws score higher than saved Jev on Top-5**.
These are Random-choice reference distributions conditional on this network,
its pools/weights, and a fixed Jev response set. They are not p-values,
population confidence intervals, or fresh Jev replications.

Jev chooses a lower latency proxy on 14/27 tasks at Top-1 and 10/27 at Top-5.
Omitting either of two influential tasks reverses the Top-1 advantage.
The Top-5 deficit persists under every single-task and leading-DAG-hash-group
omission. These are descriptive changes to the task mixture, not additional
network experiments. The 27 components, 56 occurrences, and 100,126 schedules
do not constitute independent network observations.

The recorded tournament uses 23 two-round, two one-round, and two three-round
cases, with 673 questions in total. All 25 multi-round cases lose an original
pool-optimal record in round 0. Final Top-5 boundary probability ties occur
in 25/27 tasks; only two tasks include a runtime-best surviving finalist in
the reported Top-5. A median of four distinct probability values per question
makes the deterministic tie rule relevant. The explicit Choice answer comes
first, followed by descending probability and then seeded presentation order.
Probabilities from different questions are not globally comparable, so no
global tournament Pairwise Accuracy is reported. These diagnostics describe
saved paths; they do not establish that ordering or tie-breaking caused the
outcome or that an alternative tournament would recover the optima.

### Recorded usage and reproducibility

| ResNet-50 accounting item | Recorded value |
|---|---:|
| Successful live / cached requests | 631 / 42 |
| New live input tokens | 95,175,703 |
| Historical cached input tokens | 5,585,849 |
| New live output tokens | 1,061,953 |
| Estimated new input cost at $0.042/million | $3.997380 |
| Sum of recorded decision latencies, including cached cases | 1,133.868 seconds |

This is an input-only estimate, not a total invoice. Output charges,
unobserved retries/failures, and billing adjustments are excluded. Confirm
current provider pricing and reconcile usage before reporting actual monetary
cost. Recorded decision time is inference/client latency, not ResNet execution.

The [protocol](tenset-paper-protocol.md), [run guide](run-tenset-paper-resnet50.md),
and [four-slide workflow/results contribution](presentations/tenset-rq1/README.md)
provide reproduction and presentation details. The experiment remains bound
to its original commit and artifact hashes. Integration changes alter code
fingerprints; historical execution/resume must use the recorded checkout and
dependencies, and must not rewrite old provenance to match newer code.

## Completed E5 single-question results

The run at commit `6a4fe91` evaluated Random-42 and Jev on the same 120 × 50
pools. Reviewed outputs are in the
[E5 result directory](results/tenset/w120-c50-s42-long-context/metrics.csv).

| Model | Top-1 Score ↑ | Top-5 Score ↑ | Pairwise Accuracy ↑ | Runtime regret ↓ |
|---|---:|---:|---:|---:|
| Random-42 | 0.332440 | 0.498775 | 0.491959 | 6.058854 |
| Jev 1.14, long context | 0.336867 | 0.512829 | 0.519456 | 5.081011 |

Jev's Pairwise Accuracy is higher by 0.027497, or 2.75 percentage points.
Exploratory paired workload-bootstrap intervals are positive for this ranking
metric and include zero improvement for Top-1, Top-5, and mean regret. Mean
regret is 16.1% below the observed Random result but is sensitive to a few large
wins: Jev chooses a faster schedule on 58 workloads, a slower one on 60, and
an equal runtime on two. Both select the unique sampled optimum on 5/120 cases.
This supports a modest ranking signal with mixed selection performance.
The exact uniform-Random expectation and outlier/family sensitivity are separate
offline analyses in the run log, not additional model runs.

### Completed candidate-size comparison

The additional E5 runs keep the same 120 workload IDs and use 16 and 32
schedules from the frozen 50-schedule pool. Seed 42, no field definitions,
single-question selection, pinned model, and long-context mode follow the
prepared sweep workflow. The 50-schedule member reuses the original saved
responses. The uploaded export contains 720 rows: two models, three budgets,
and 120 workloads. Its six aggregate rows and plotted values are recomputed
from those per-workload rows in the
[reviewed sweep results](results/tenset/size-sweep/README.md).

| Schedules | Model | Top-1 Score ↑ | Top-5 Score ↑ | Pairwise Accuracy ↑ | Runtime regret ↓ |
|---|---|---:|---:|---:|---:|
| 16 | Random-42 | 0.407439 | 0.615892 | 0.504097 | 5.308322 |
| 16 | Jev | 0.456949 | 0.645309 | 0.533403 | 3.535879 |
| 32 | Random-42 | 0.348193 | 0.553470 | 0.503461 | 5.317561 |
| 32 | Jev | 0.394553 | 0.563148 | 0.529536 | 3.993304 |
| 50 | Random-42 | 0.332440 | 0.498775 | 0.491959 | 6.058854 |
| 50 | Jev | 0.336867 | 0.512829 | 0.519456 | 5.081011 |

Jev's observed Top-1 advantage is +0.049510, +0.046360, and +0.004427
at 16, 32, and 50 schedules. Pairwise Accuracy improves by 2.93, 2.61,
and 2.75 percentage points. Mean regret is 33.4%, 24.9%, and 16.1% lower
than the respective Random means; these reductions are not equivalent
percentages of execution speedup. Compare the models within each budget.
The oracle changes with the pool, and five choices cover a smaller fraction
of a larger pool, so falling absolute scores do not alone establish worse
decision quality as candidate count increases.

An exploratory paired workload bootstrap uses 50,000 resamples, seed
20261116, and the same sorted-workload draws across budgets. The following
percentile 95% intervals describe Jev-minus-Random differences:

| Schedules | Top-1 difference and 95% interval | Pairwise Accuracy difference and 95% interval |
|---|---|---|
| 16 | +0.049510 [−0.024313, +0.123654] | +0.029306 [+0.002708, +0.055625] |
| 32 | +0.046360 [−0.009531, +0.102266] | +0.026075 [+0.010232, +0.042104] |
| 50 | +0.004427 [−0.059905, +0.068146] | +0.027497 [+0.014293, +0.040789] |

Top-1 and Top-5 intervals cross zero at every budget; the strongest
consistent signal is in Pairwise Accuracy. The mean-regret difference at
32 schedules has interval [−2.647424, −0.112540], while regret intervals
at 16 and 50 schedules cross zero. These are unadjusted exploratory
intervals over several metrics and budgets, conditional on saved responses.
Program-family dependence is not modeled; operation-name signatures are not
a substitute for verified DAG-family IDs. They are not fresh API replicates.

Shared resampling also estimates whether the Jev-minus-Random advantage
changes between budgets. Every tested 16-minus-50 and 32-minus-50
advantage interval includes zero. For Top-1, these contrasts are +0.045084
[−0.051515, +0.143318] and +0.041934 [−0.033347, +0.118212]. Thus the
larger observed Top-1 gap at smaller pools is descriptive, not an established
candidate-size interaction. Full metric and contrast tables accompany the
reviewed sweep exports.

A separate [hash-family sensitivity analysis](results/tenset/size-sweep/family-cluster-sensitivity.json)
joins the historical E5 mapping by workload ID: 56 leading DAG-hash groups,
with 25 workloads in the largest group. Resampling 56 groups with replacement
retains all member workloads and computes each mean as summed paired
differences divided by summed workload counts. It uses 50,000 shared draws
and seed 20261116. Pairwise Accuracy intervals remain positive at 16
[+0.000926, +0.057576], 32 [+0.010891, +0.041882], and 50
[+0.014962, +0.042180], with the 16-schedule lower bound close to zero.
Top-1/Top-5 and cross-budget advantage intervals still include zero; the
32-schedule regret interval remains negative. This is a sensitivity check
using hash-group proxies, not proof of independent networks or an adjustment
for the several metric comparisons. It leaves the qualitative conclusion
unchanged: modest ranking evidence with uncertain selection-score gains.

The upload includes analysis exports and plot provenance, but not the new
16/32 runs' raw predictions, execution manifests, or frozen sweep manifest.
Export consistency and matched workload IDs can be checked independently;
the exact new schedule identities, nesting, wire inputs, and resolved API
model cannot be independently re-audited from this archive alone. The
prepared workflow checks those invariants on the Mac. Its declared prepared
file checksum matches the historical E5 artifact. No new model calls were
made during the export review.

### Workload-type results and examples

The uploaded per-workload export supplies 27 literal joined operation
signatures, consistent across all three budgets. They identify groups of
ComputeOp names, not independent program families or fully inspected tensor
semantics. The largest signature, `PaddedInput+Conv2dOutput+T_add+T_relu`,
contains 40/120 workloads. Its mean Jev-minus-Random Top-1 differences are
+0.0171, +0.0214, and +0.0521 at 16, 32, and 50 schedules. Other groups
contain fewer cases and can reverse direction across budgets. These
descriptive results do not support a blanket claim that Jev is better for
all convolution workloads. Group sizes and all five metrics are exported
separately for each candidate budget.

At 50 schedules, `tenset-e6cd98ac634fba7bab1d171c` has operation signature
`PaddedInput+Conv2dOutput+T_add`. Jev selects a test-pool optimum, scoring
1.000000 versus Random's 0.051030. A contrasting case,
`tenset-1c06dd347db18aa8ffcdc91b`, has signature `compute`: Random scores
1.000000 and Jev 0.200940. These are post-hoc performance examples, not
representative samples or explanations of why specific transformations
succeed. Tensor shapes and complete selected/oracle schedule transformations
still require the context-aware case export from the prepared data.

## Completed historical Platinum tutorial tournament results

The run at commit `ed73f19` evaluated both models on all 484 retained workloads
and 187,679 reconstructed test records. Reviewed exports and figures are in
[the Platinum result directory](results/tenset/tutorial-500-platinum-test/README.md).

| Model | Record-weighted Top-1 Score ↑ | Record-weighted Top-5 Score ↑ |
|---|---:|---:|
| Random-0 | 0.235045 | 0.385021 |
| Jev 1.14, field definitions and long-context tournament | 0.269356 | 0.388245 |
| Jev minus Random-0 | +0.034310 | +0.003224 |

Paired workload resampling retains each sampled workload's test-record weight
and recomputes weighted ratios of sums. With 50,000 percentile-bootstrap
replicates and seed 20261116, the exploratory 95% intervals for Jev minus
Random-0 are **[+0.009015, +0.059739]** for Top-1 and
**[−0.018344, +0.024936]** for Top-5. They are unadjusted for multiple metrics
and conditional on this file sample and saved responses. Related workloads
may be correlated; DAG-family clustering could not be recovered from the
uploaded artifacts.

Jev has a higher Top-1 score on 278 workloads and a lower score on 206. For
Top-5, it wins 238 and loses 246. It selects the unique test-pool optimum on
9/484 workloads, versus 6/484 for Random. The separately reported macro means
are 0.282998/0.400940 for Jev and 0.249948/0.399532 for Random. They are not the
weighted values in the primary table.

### Analytical Random reference and failure diagnostics

Saved runtime maps permit computing the exact expectation of a uniform random
permutation under those recorded labels. The weighted expected Random scores are 0.239404
for Top-1 and 0.378379 for Top-5. Top-5 uses the exact prefix minimum
order-statistic distribution. This analytical reference does not replace the
observed Random-0 run or count as extra experimental replicates.

Jev minus expected Random is +0.029951 for Top-1, with an exploratory paired
95% interval **[+0.012579, +0.048034]**. Its Top-5 difference is +0.009865,
with interval **[−0.005705, +0.025934]**. The Top-1 signal persists against
uniform Random expectation; there is no clear Top-5 gain under either reference.

The tournament loses the unique test-pool optimum in the first round on
442 of its 468 two-round workloads. Of the 26 optima that advance, six are
selected in the final question and 20 are not. Three of the 16 single-question
cases select the optimum. This describes the recorded paths and does not prove
that a hypothetical single-question alternative would select those optima.

Coarse probabilities are common: 837 of 1,856 questions have ties for maximum
probability, and 31.66% of recorded option probabilities are zero. Option-count
bins differ in workload and round, so this is a descriptive diagnostic rather
than a controlled candidate-size result. Failure-sentinel choices occur eight
times for Jev and seven for Random. We do not claim that Jev avoids failed
configurations. Schedule-level semantic explanations remain to be completed
from the prepared workload metadata.

## Historical tutorial and E5 decision cost and latency

| Platinum tutorial accounting item | Recorded value |
|---|---:|
| Live / cached requests in the full run | 1,840 / 16 |
| New live input tokens | 179,522,529 |
| Previously cached input tokens | 1,880,950 |
| All distinct saved-response input tokens | 181,403,479 |
| Estimated new successful input cost | $7.539946 |
| Estimated total including earlier pilot responses | $7.618946 |

Prices assume $0.042 per million input tokens and free output. No invoice or
dated provider-pricing artifact was supplied; these are estimates for logged
successful responses, excluding failed or other requests, custom pricing,
taxes, and credits. The two cost totals overlap and must not be added. Cached
tokens represent prior usage rather than new full-run charges.

The mean total Platinum decision latency is 4.102 seconds per workload,
including cached cases. Mean live API latency is 1.022 seconds per request.
Those are different units and scopes. The E5 live total decision mean is
465.156 ms for its smaller single-question pools. These values describe their
respective protocols, not a controlled latency comparison between algorithms.

The additional 16- and 32-schedule E5 runs report 120 live requests each,
with 2,750,775 and 4,255,904 input tokens. Together they use 7,006,679
new input tokens, an estimated **$0.294281** at the same assumed input rate.
The 50-schedule anchor's old 110-live/10-cached accounting is historical;
reusing its saved results adds no new sweep requests. Mean total decision
latencies in the uploaded exports are 243.222 ms and 342.073 ms at 16 and
32 schedules. The historical 50-schedule mean of 426.488 ms includes cached
cases, so this is not a controlled latency-scaling experiment.

## Completed controlled context-engineering ablation

### Design and evidence

The completed [context-ablation study](results/tenset/context-ablation/README.md)
uses the same frozen E5 120-workload, 50-schedule pools as the historical anchor,
with three fresh Jev requests per workload and condition. A0 contains the raw
program and candidate schedules, A1 adds candidate definitions, A2 adds a
separate graph-field specification, and A3 adds both. A4 changes only the
presentation order of A3, using ascending or descending lexicographic schedule
features. A5 retains a deterministic structurally diverse subset of ten schedules
and uses the A3 descriptions. The question, pinned model, long-context mode,
opaque aliases, source pools, seed 42 and original-pool scoring denominator
remain fixed. Each shown pool fits one question.

Ordering and filtering use transformation/annotation/split structure rather
than measured timings. The diversity rule was fixed without fitting training
labels or using test runtimes. It differs from TpuGraphs' training-informed
large-tile heuristic, so the two filtering studies test different rules.
The three repeats keep input order fixed and measure fresh API variation;
separate repeat caches prevent a cached response from posing as a new repeat.

The supplied frozen manifest and derived summaries report all 24 planned
runs: seven small-stage conditions × three repeats, plus three full-stage A5
runs. Raw predictions, completion receipts and request/response caches were
not uploaded. Our review can reconcile summary arithmetic and planned coverage;
it cannot independently repeat the runner's receipt, private-label or
request-body checks. The [workflow guide](run-tenset-context-ablation.md) gives
reproduction, preflight and archive commands.

### Small-stage results

These are macro workload means on 120 × 50 original schedules. The ± values
are sample standard deviations across three run means, not confidence intervals.
The prefix Top-5 and endpoint Top-5 definitions are kept separate.

| Condition | Top-1 Score, mean ± SD | Prefix Top-5 Score | Endpoint Top-5 Score | Mean runtime regret |
|---|---:|---:|---:|---:|
| A0 naive | 0.358186 ± 0.004510 | 0.518579 | 0.629279 | 5.073117 |
| A1 candidate definitions | 0.321786 ± 0.002126 | 0.500551 | 0.613974 | 5.374682 |
| A2 graph specification | 0.355713 ± 0.002391 | 0.522320 | 0.634412 | 4.960330 |
| A3 both | 0.331918 ± 0.007598 | 0.497509 | 0.604623 | 5.222691 |
| A4 ascending structural order | 0.338018 ± 0.004870 | 0.503314 | 0.618841 | 4.537227 |
| A4 descending structural order | 0.317597 ± 0.020314 | 0.483385 | 0.601717 | 7.087368 |
| A5 diversity filter, keep 10 | 0.340978 ± 0.003721 | 0.495122 | 0.597623 | 5.520364 |
| Random-42, one seeded draw | 0.332440 | 0.498775 | 0.622915 | 6.058854 |

Candidate definitions reduce mean Top-1 score by 0.036399 relative to A0;
both descriptions reduce it by 0.026267. Every A1 and A3 aggregate Top-1 repeat
is below every A0 repeat, so those fixed-cohort mean differences exceed the
observed repeat spread. Their workload effects are mixed: A1 improves 32,
worsens 41 and ties 47 cases; A3 improves 34, worsens 39 and ties 47.
The corresponding exploratory sign-test p-values are 0.349 and 0.640.
These tests concern win/loss direction, not mean-effect magnitude; they neither
prove the observed mean declines are random nor establish uniform harm.
A2 is close to A0 on Top-1 (difference −0.002472, 28 wins/27 losses/65 ties),
with a small endpoint Top-5 advantage of 0.005133. More description is not a
consistent improvement under this serialization.

Ordering changes which losses dominate the mean. Ascending and descending
structural orders have mean regret 4.537227 and 7.087368, but the descending
Top-1 score is higher on 54 workloads, lower on 60 and tied on six. The mean
gap is not evidence of a uniform ordering advantage. Pairwise Accuracy is
0.519934 ascending versus 0.508059 descending; its sign-test p = 0.0353 is an
unadjusted exploratory result among many comparisons. These structural orders
must not be interpreted as TPU tile-size orders. Across unfiltered variants,
65.0–70.8% of workloads keep the same first choice in all three repeats;
A5 has 82.5% agreement. Aggregate stability does not imply deterministic choices.
Top-1 Score is a performance ratio, not optimum-hit accuracy: A0's actual
optimum-pick accuracy is 3.61%.

Filtering trades input cost for lost search coverage. It retains a pool
optimum in 22/120 workloads. Filtered-pool uniform Random has expected Top-1
score 0.308215 and endpoint Top-5 score 0.601597, versus A5's 0.340978 and
0.597623. Thus Jev adds Top-1 selection signal within the retained pools but
has no observed endpoint Top-5 advantage over that exact expectation. Filtered
or heuristic-completed tails are not used to claim full-ranking Pairwise Accuracy.

### Required full ResNet-50 follow-up

The full stage applies A5 with at most 50 shown schedules to all 27 declared
ResNet-50 task pools: occurrence weight 56 and 100,126 original candidate
records. It preserves the private float32 normalizers and throughput floor
used by the pinned network evaluator. Both Jev and Random are scored against
the original complete pools. This stage is not a full-network factorial,
not an evaluation of every TenSet workload, and not an unchanged repeat of
the historical full-pool tournament.

| Model | Weighted network Top-1 Score | Weighted network Top-5 Score |
|---|---:|---:|
| A5 diversity filter, three-repeat mean ± SD | 0.113422 ± 0.007755 | 0.343155 ± 0.024797 |
| Random-42, one seeded draw | 0.137327 | 0.406334 |

Jev is below this paired Random draw on both weighted network endpoints.
Its separate task-macro Top-1/endpoint Top-5 means are 0.267820/0.433057;
these must not replace the weighted values above. The filter retains a
pool optimum on only 1/27 tasks. Its frequency-weighted filter-oracle score
ceiling is 0.713744, showing headroom beyond both the filter's exclusions and
Jev's achieved choices. The historical Jev tournament's 0.097724/0.274583 and
Random-0's 0.091462/0.377121 remain a separate saved experiment; their seed,
context and selection procedure differ, so a cross-run difference is not a
controlled filter-only effect.

### Usage and scope of completion

The uploaded accounting reports 2,601 live requests, zero cached requests,
119,921,175 input tokens and 1,097,262 output tokens. At the assumed
$0.042/million input rate, the complete study's input estimate is
**$5.03668935**: $4.858342902 for the 2,520-request small stage and
$0.178346448 for the 81-request full stage. Completed-study totals already
include saved pilot decisions; do not add a separate pilot estimate again.
Output fees, unrecorded retries and billing reconciliation are excluded.

Small-stage A3 uses 18,353,679 input tokens across its three repeats, versus
7,318,197 for A5, approximately 60.1% fewer. Mean run-median decision latency
falls from 485.7 ms to 189.1 ms, while endpoint Top-5 falls from 0.604623 to
0.597623. This is the measured performance-cost tradeoff for this particular
filter, not evidence that filtering improves schedule quality generally.

All requested ablation conditions and the required full-stage follow-up are
completed under these scopes. Raw archive sharing with Jonghu remains manual;
no transmission is claimed. Semantic tensor/schedule explanations and team
report integration remain distinct deliverables. The historical presentation
has not been regenerated for these new results.

## Remaining report deliverables and optional experiments

The implementation and completed results cover the requested Top-1/Top-5,
Pairwise Accuracy, regret, per-workload predictions, aggregate exports,
candidate-size comparison, an E5 operation-signature breakdown, and the
controlled three-repeat context/order/filter ablation with a full-network
filtered follow-up under their stated protocols. The historical ResNet-50
result, saved-output audits, Random-choice variability, tournament diagnostics
and its presentation are finished; the deck has not been updated for the
new ablation results. The following report deliverables still need completion:

- Export the full selected/oracle schedule and tensor metadata using the
  context-aware case analyzer on the Mac. The uploaded E5 operation labels
  support grouping, but do not supply shapes or transformation arguments.
- Review those semantic examples before attributing good or poor performance
  to particular scheduling decisions. Do not infer causes from scores or
  operation names alone.
- If extending type-level conclusions to the Platinum study, export and review
  its own metadata/type breakdown. E5 labels do not describe that separate cohort.
- Integrate this RQ1 section, the primary ResNet-50 results, and the separate
  historical studies into the team report; agree on cross-dataset metric labels
  and complete final proofreading.
- Verify the provider tariff, output-token charges, and invoice reconciliation
  before replacing the explicitly labelled input-cost estimates with actual costs.

The [finish guide](run-tenset-finish-rq1.md) supplies offline case-export commands.
The three-repeat candidate-definition, graph-specification, presentation-order
and filtering study is completed. The older no-program-context control,
tie-breaking/grouping ablations, fresh repeats on other cohorts, and additional
networks or expanded historical cohorts remain new extensions; the primary
ResNet-50 run already uses complete declared task pools. Changing platform and
pool policies between existing studies is not a candidate-size ablation.
Training or reproducing a matched specialized model is outside the current
scope; the paper MLP stays an external published reference.

## Limitations and interpretation

The primary ResNet-50 run has a small observed Jev Top-1 lead and a lower
Top-5 score than saved Random. The Top-1 lead is sensitive to individual tasks;
the offline Random reference does not establish reliable model superiority.
The historical runs show a modest ranking signal across all three E5 budgets
and a positive weighted Top-1 signal on the tutorial test reconstruction.
E5 Top-1/Top-5 improvements and historical Platinum Top-5 gains remain
uncertain under the exploratory intervals. E5 mean regret is lower for Jev
at 32 schedules under an unadjusted interval; the other budgets
remain uncertain. This is evidence conditional on selected workloads and API
responses, not proof of broad superiority across tensor programs or hardware.
Fresh API repeatability was not measured for the historical one-run studies.
The new context ablation supplies three fresh repeats on its fixed inputs,
with 65.0–82.5% identical first picks across the small-stage conditions and
66.7% in the full stage. Three repeats describe limited response variation;
fixed sampling seeds and cached replay do not establish deterministic new
responses. The historical protocols differ in target, split, pool size,
invalid-record handling, presentation, field definitions, algorithm, and aggregation. Their difference
cannot identify the contribution of any one factor; the new small-stage
factorial isolates its stated context changes within the frozen E5 cohort.

No new kernel execution was performed. Recorded runtimes and source/request
fingerprints were audited for saved-record consistency. Raw measurements,
prepared shards, complete private source labels, and actual API request
payloads were not supplied for independent content verification. The context
ablation upload contains derived summaries and a frozen manifest rather than
raw predictions, completion receipts or caches, so its independent review is
limited to summary-level consistency. All 108 ResNet-50 endpoint
decompositions match runtime-based float32-label
reconstruction exactly. Separately, recomputation agrees with the historical
tutorial weighted scores within 5.3e-10; that historical check is not exact
reconstruction of its private labels. The portable reader fails on unsupported
semantic formats rather than silently
removing them. Local byte/token guards are planning policies, not guarantees
of service admission. The published MLP provides external context without
changing the paired Jev-versus-Random research question. Optional legacy tools
do not imply a requirement to train a matching specialized model.

## References

1. Zheng et al., *TenSet: A Large-scale Program Performance Dataset for Learned
   Tensor Compilers*, NeurIPS Datasets and Benchmarks, 2021.
   [Official dataset repository](https://github.com/tlc-pack/tenset).
2. [Pinned tutorial and record-preparation procedure](https://github.com/tlc-pack/tenset/blob/35774ed373f9693cc6efcbe83f5703141e8ab0c8/docs/try_pretrained_models_on_aws_c59x.md).
3. [Official metric definitions](https://github.com/tlc-pack/tenset/blob/35774ed373f9693cc6efcbe83f5703141e8ab0c8/python/tvm/auto_scheduler/cost_model/metric.py).
4. [Reference evaluation and aggregation](https://github.com/tlc-pack/tenset/blob/35774ed373f9693cc6efcbe83f5703141e8ab0c8/scripts/train_model.py).
5. [Main-paper Section 5.2/Table 4, page 7](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/a684eceee76fc522773286a895bc8436-Paper-round1.pdf#page=7).
6. [Pinned network-weighted endpoint evaluator](https://github.com/tlc-pack/tenset/blob/35774ed373f9693cc6efcbe83f5703141e8ab0c8/scripts/eval_model_on_dataset.py).
