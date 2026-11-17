# RQ1: Tensor schedule selection on TenSet

## Research question and scope

Can Jev select faster tensor schedules than uniform Random from serialized
program and schedule descriptions? The current RQ1 comparison is **Jev versus
Random**. TenSet supplies measured runtime labels and metric definitions.
The empirical results below compare these two models on identical candidate
pools within each experiment.

Two evaluation protocols have completed. The E5 study now also includes a
matched 16/32/50 candidate-size sweep. Their uploaded results answer related
questions and are reported separately, with the limits of each audit stated.
The [run log](tenset-runlog.md) records their provenance; the
[finish guide](run-tenset-finish-rq1.md) provides the execution and remaining
semantic case-analysis commands. The 16- and 32-schedule runs are completed
measurements; the original 50-schedule results are reused.

## Dataset, serialization, and evaluation protocols

TenSet contains 51,577,248 program measurements across six hardware platforms
[1]. We use existing CPU measurements, with the exact target preserved in each
workload. A workload is a tensor program; a schedule is a compiler configuration
for that program. The Mac runs the evaluation client, while Jev inference runs
through the API. No CPU kernels are newly timed on the Mac.

| Property | E5 single-question experiment | Platinum tournament experiment |
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

Program serialization retains tensor shapes and dtypes, operation names,
complete tensor-expression graphs, loop information, and the exact TVM target.
Schedule serialization retains initial state and each scheduling transformation
with its code and arguments. Workload keys alone are insufficient descriptions.
Measured timings, error codes, and record indices remain private labels rather
than model-visible fields. Paired predictions record identical candidate pools
and model-input fingerprints. Actual API payloads were not uploaded for an
independent content audit.

The tournament compares at most 255 options per question and advances five
from each group. Local token estimates can require smaller groups. In the
completed Platinum run, 468 workloads required two rounds and 16 required one;
all 1,856 saved requests resolved to the pinned model. The definitions and
tournament were evaluated together, so this run does not isolate the effect
of either change.

## Metrics and aggregation

Let `t*` be the minimum runtime in a workload's evaluated pool and `b_j` the
best runtime among its first `j` predicted choices. TenSet's score is [3]:

```text
Top-k Score = mean(t* / b_j, j = 1..min(k, candidate_count))
runtime_regret = selected_runtime / t* - 1
```

Top-5 therefore averages the prefix-best scores; it is not just the endpoint
best-of-five ratio. These scores are normalized performance ratios, not
optimum-hit accuracy. The oracle is restricted to each presented test pool.
The reciprocal relationship between Top-1 and regret applies per workload,
not by inverting an aggregate mean.

For the E5 single-question run, Pairwise Accuracy follows the official strict
comparison convention and preserves model score ties. A separately named
tie-aware diagnostic excludes equal measured timings and gives half credit
for predicted ties. Valid-only runtime regret describes measured schedules.

For the Platinum tournament, probabilities are comparable within each
question. Eliminated candidates are appended using a deterministic rule;
that tail is not a global learned ranking. We therefore do not report global
Pairwise Accuracy. Error records carry failure-sentinel costs, so their large
penalties are not interpreted as physical program runtime or used to claim a
mean runtime speedup. Selection scores and failure-selection counts remain
explicitly separate outcomes.

The E5 runs average workload metrics equally. Every case within a budget
has the same candidate count; the original anchor has 50 candidates per case.
The Platinum primary result weights each workload by its number of test records,
as in the tutorial evaluator [4]. Its unweighted macro scores are separately
labelled. These aggregations and the different pool/invalid-record policies
must not be mixed into one cross-run result.

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

## Completed Platinum tournament results

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

## Decision cost and latency

| Platinum accounting item | Recorded value |
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

## Remaining required analyses and prepared work

The implementation and completed results cover the requested Top-1/Top-5,
Pairwise Accuracy, regret, per-workload predictions, aggregate exports,
candidate-size comparison, and an E5 operation-signature breakdown under
their stated protocols. Reviewed size results and post-hoc performance
examples are published. The following evidence still needs completion:

- Export the full selected/oracle schedule and tensor metadata using the
  context-aware case analyzer on the Mac. The uploaded E5 operation labels
  support grouping, but do not supply shapes or transformation arguments.
- Review those semantic examples before attributing good or poor performance
  to particular scheduling decisions. Do not infer causes from scores or
  operation names alone.
- If extending type-level conclusions to the Platinum study, export and review
  its own metadata/type breakdown. E5 labels do not describe that separate cohort.

The [finish guide](run-tenset-finish-rq1.md) supplies the commands. Additional
field-description, context/order, forced-group controls, fresh repeats, and
all-source-schedule tournaments remain optional extensions. Changing platform
and workload/pool policies between the E5 and Platinum experiments is not a
candidate-size ablation. No specialized-baseline training is required for this
revised RQ1 scope.

## Limitations and interpretation

The completed runs show a modest Jev ranking signal across all three E5
budgets and a positive weighted Top-1 signal on the Platinum test
reconstruction. E5 Top-1/Top-5 improvements and Platinum Top-5 gains remain
uncertain under the exploratory intervals. E5 mean regret is lower for Jev
at 32 schedules under an unadjusted interval; the other budgets
remain uncertain. This is evidence conditional on selected workloads and API
responses, not proof of broad superiority across tensor programs or hardware.
Fresh API repeatability has not been measured. Fixed sampling seeds and cached
replay do not establish that new live responses will be identical.
The protocols differ in target, split, pool size, invalid-record handling,
presentation, field definitions, algorithm, and aggregation. Their difference
cannot identify the contribution of any one of those factors.

No new kernel execution was performed. Recorded runtimes and source/request
fingerprints were audited for saved-record consistency. Raw measurements,
prepared shards, private float32 throughput labels, and actual API request
payloads were not supplied for independent content verification. Runtime-ratio
recomputation agrees with the uploaded Platinum weighted scores within
5.3e-10, but that is not exact reconstruction of the private labels. The
portable reader fails on unsupported semantic formats rather than silently
removing them. Local byte/token guards are planning policies, not guarantees
of service admission. Optional legacy tools and historical numerical
comparisons do not change the current Jev-versus-Random research question.

## References

1. Zheng et al., *TenSet: A Large-scale Program Performance Dataset for Learned
   Tensor Compilers*, NeurIPS Datasets and Benchmarks, 2021.
   [Official dataset repository](https://github.com/tlc-pack/tenset).
2. [Pinned tutorial and record-preparation procedure](https://github.com/tlc-pack/tenset/blob/35774ed373f9693cc6efcbe83f5703141e8ab0c8/docs/try_pretrained_models_on_aws_c59x.md).
3. [Official metric definitions](https://github.com/tlc-pack/tenset/blob/35774ed373f9693cc6efcbe83f5703141e8ab0c8/python/tvm/auto_scheduler/cost_model/metric.py).
4. [Reference evaluation and aggregation](https://github.com/tlc-pack/tenset/blob/35774ed373f9693cc6efcbe83f5703141e8ab0c8/scripts/train_model.py).
