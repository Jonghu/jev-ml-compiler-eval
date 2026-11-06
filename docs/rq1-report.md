# RQ1: Tensor schedule selection on TenSet

## Research question and experimental status

Can Jev select competitive tensor schedules using serialized program and schedule
information, compared with Random and a specialized XGBoost cost model?

The repository implements the TenSet experiment pipeline. Full benchmark results
are **pending**. A user-reported pilot completed on a MacBook Air using one real
TenSet workload and eight candidate schedules. Both Random and Jev selected a
sampled optimum with zero regret; Random had higher pairwise accuracy on that
case. This verifies the setup path, not general competitiveness. Saved artifacts,
cache status, token usage, and costs have not been independently inspected.
See [project status](project-status.md) for the reported pilot output.

Portable Python metadata export reads the supplied registry without installing
TVM on the Mac. Native-generated fixtures validate format compatibility; their
values are not experimental evidence about TenSet or Jev.

## Dataset and experimental unit

TenSet contains 51,577,248 program measurements across six hardware platforms
[1]. Our initial target is the Intel E5-2673 v4 CPU collection from the official
CPU v3.3 release. The experimental unit is one tensor program and a fixed set
of successfully measured schedules on one exact TVM target. The target string
is preserved from the records rather than inferred from the current machine.
This is offline evaluation: labels come from the dataset, so the evaluation
host does not need to reproduce the measurement hardware.

We reject failed measurements, empty timing arrays, nonfinite or nonpositive
times, and TVM failure sentinels. Each record's timing repetitions are averaged;
duplicate schedule records are averaged with equal record weight. Times are
converted from seconds to milliseconds. Invalid and excluded record counts,
source hashes, and preparation settings are recorded for inspection.

## Splits and candidate sets

The split is assigned by a seeded hash of the original workload key: approximately
70% training, 10% validation, and 20% testing. This is a project-defined split
by workload, distinct from the default within-task split in TenSet's training
script. No tensor program crosses splits. Validation is reserved for development
and final test labels are evaluated after predictions. Jev and Random receive
no project-specific training. The optional local XGBoost adapter uses only
training workloads if a matched baseline is run in the future.

Within each test workload, candidates are ordered by a seeded hash independent
of runtime. Taking prefixes produces nested candidate sets. The planned sizes
are 8, 16, and 32, with seeds 42, 7, and 2026. All models in a run receive the
same IDs and configurations. For cross-size comparisons, restrict to workloads
with at least 32 valid schedules and compare paired workloads. Do not select
the fastest measured candidates or report different pools as a paired comparison.

## Representations and models

The workload representation includes the exact target, workload key, available
tensor shapes and operation metadata, and complete tensor-expression graphs and
loop descriptions decoded from the official task registry. A hash-based key alone is insufficient
to identify program semantics; metadata coverage must be reported. The schedule
representation preserves initial state and every scheduling transformation's
code, name, and positional arguments. Measured runtime, measurement status,
timestamps, and source-record indices never enter model-visible inputs.

Jev uses the existing shared decision interface and pinned model version.
Random uses the shared seeded baseline. Our measured comparison is Jev versus
Random. Published TenSet XGBoost results will be cited as an external reference
with their hardware, split, candidate pool, and aggregation protocol; they must
not be presented as a paired comparison with our sampled candidates.

The optional specialized adapter calls the official TenSet TVM `XGBModel`:
per-store features, normalized-throughput labels, and the pack-sum RMSE objective
[2]. Its persistent subprocess and checkpoint/split provenance remain available
for a future matched baseline. Local XGBoost training is not a prerequisite for
the current Jev experiment.

## Metrics and aggregation

Let `t*` be the fastest measured runtime in the sampled candidate set, and let
`b_j` be the minimum measured runtime among the first `j` predicted schedules.
The official per-workload average peak score is:

```text
Top-k Score = mean(t* / b_j for j in 1..min(k, candidate_count))
runtime_regret = selected_runtime / t* - 1
```

Top-5 Score therefore measures the whole five-choice prefix rather than only
the best fifth-choice endpoint. Higher scores and lower regret are better.
The oracle is the sampled candidate set, not every schedule in the archive.
Top-1 Score equals `1 / (1 + runtime_regret)`.

Pairwise Accuracy follows TenSet's strict pair comparisons [3]. Its treatment
of ties can depend on original candidate order. We additionally report the
explicitly named `pairwise_accuracy_tie_aware` diagnostic, which excludes equal
measured runtimes and awards half credit for predicted ties. Score ties are
preserved when the model exports scores or probabilities; otherwise its full
ranking defines the prediction. Top-k follows each model's exported ranking,
including its deterministic tie resolution.

The shared runner macro-averages workloads. TenSet's reference script weights
workloads by their number of measurements [4]; with equal candidate counts these
averages coincide. Where counts vary, report both aggregation rules rather than
claiming direct numerical equivalence with the paper.

## Results and discussion to complete after the benchmark run

No Jev-versus-XGBoost comparison or competitiveness claim is currently supported.
The analysis command produces aggregate and per-workload tables, breakdowns by
operation and candidate count, plots, and paired Jev success/failure examples.
Populate this section from those outputs after the live comparison completes.
Report the number of workloads per split and operation, candidate counts,
serialization byte distributions, metadata coverage, checkpoint provenance,
Jev model version, seed variation, and all excluded workloads.

Discuss whether performance changes with operation type or candidate count and
whether poor choices correlate with missing program metadata or large inputs.
Show concrete schedule choices from paired cases rather than inferring causes
from aggregate scores alone. Treat decision latency separately from selected
program runtime: legacy process startup, feature extraction, API latency, and
cached Jev calls have different meanings. Dollar costs remain unknown unless
a dated pricing snapshot is supplied; missing prices must never be shown as zero.

## Limitations

The initial collection covers one CPU platform. The hash split is reproducible
but not stratified and can yield empty splits on tiny subsets. Excluding invalid
schedules evaluates selection among valid candidates rather than prediction of
compilability. Sampling bounds input size but changes the oracle. The configured
byte cap is a local experiment budget, not a verified TypeSafe service limit.
Serialization retains scheduling transformations but may lack low-level compiler
features seen by the specialized baseline. The real archive and API path require
integration verification before publishing benchmark claims. The portable reader
rejects unsupported registry formats and tensor nodes instead of dropping semantics.

## References

1. Zheng et al., *TenSet: A Large-scale Program Performance Dataset for Learned
   Tensor Compilers*, NeurIPS Datasets and Benchmarks, 2021.
   [Official repository](https://github.com/tlc-pack/tenset).
2. [Official XGBoost implementation](https://github.com/tlc-pack/tenset/blob/35774ed373f9693cc6efcbe83f5703141e8ab0c8/python/tvm/auto_scheduler/cost_model/xgb_model.py).
3. [Official metric definitions](https://github.com/tlc-pack/tenset/blob/35774ed373f9693cc6efcbe83f5703141e8ab0c8/python/tvm/auto_scheduler/cost_model/metric.py).
4. [Reference evaluation and aggregation](https://github.com/tlc-pack/tenset/blob/35774ed373f9693cc6efcbe83f5703141e8ab0c8/scripts/train_model.py).
