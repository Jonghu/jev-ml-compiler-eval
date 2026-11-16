# RQ1: Tensor schedule selection on TenSet

## Research question and scope

Can Jev select faster tensor schedules than uniform Random from serialized
program and schedule descriptions? The current RQ1 comparison is **Jev versus
Random**. TenSet supplies measured labels and metric definitions. Specialized
cost-model training and a numerical XGBoost comparison are outside this scope;
the retained legacy adapter is optional tooling rather than unfinished RQ1 work.

Two experiments have completed and their uploaded results were audited. They
answer related questions under different protocols and are reported separately.
The [run log](tenset-runlog.md) records their provenance; the
[finish guide](run-tenset-finish-rq1.md) provides the remaining local analysis and
candidate-size commands. Prepared commands are not completed measurements.

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
| Candidates | 50 valid distinct schedules each; 6,000 total | All reconstructed test records; 6–400 per workload, 187,679 total |
| Split | Workload-key hash: approximately 70/10/20 train/validation/test | Reverse record-order 90/10 split within each workload |
| Record policy | Invalid timings/errors removed; duplicate schedule records averaged | Duplicate and error records retained; 2,013 test error records reported |
| Sampling seed | 42 | 0, batch-size-1 workload preset |
| Jev presentation | Original deterministic order, one question, no field definitions | Neutral aliases, seeded shuffled order, field definitions, tournament |
| API mode | Long context, pinned `jev-1.14.0` | Long context, pinned `jev-1.14.0` |
| Primary aggregation | Workload macro mean | Test-record-count weighted mean |
| Reported metrics | Top-1, Top-5, Pairwise Accuracy, runtime regret | Top-1 and Top-5 selection scores |

The E5 subset freezes 50 schedules independently of their runtime values. It
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

The E5 run averages workload metrics equally; all cases contain 50 candidates.
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

## Remaining required analyses and prepared work

The implementation and existing results cover the requested Top-1/Top-5,
Pairwise Accuracy, regret, per-workload predictions, and aggregate exports
under their stated protocols. The following evidence still needs completion:

- Run and review the prepared operation/type breakdowns on the Mac's frozen
  prepared data. The uploaded hashes and opaque IDs cannot supply operation
  names or verify coverage of workload types.
- Complete a controlled 16/32/50 candidate-size comparison using the same
  frozen E5 workloads, seed, metadata, no-description single-question prompt,
  and long-context mode. Reuse the archived 50-candidate run; the 16/32 runs
  are prepared, not completed. Each size has its own explicitly bounded oracle.
- Inspect concrete paired successes and failures, including tensor shapes and
  selected/oracle schedule transformations, rather than inferring causes from
  aggregate scores or identifiers alone.
- Add those reviewed results to this section and the publication exports.

The [finish guide](run-tenset-finish-rq1.md) supplies the commands. Additional
field-description, context/order, forced-group controls, fresh repeats, and
all-source-schedule tournaments remain optional extensions. Changing platform
and workload/pool policies between the E5 and Platinum experiments is not a
candidate-size ablation. No specialized-baseline training is required for this
revised RQ1 scope.

## Limitations and interpretation

The completed runs show a modest Jev ranking signal on the E5 pools and a
positive weighted Top-1 signal on the Platinum test reconstruction. Top-5 gains
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
