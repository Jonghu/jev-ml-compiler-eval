# TenSet main-paper ResNet-50 protocol

This document defines the prepared evaluation and its limits. It contains no
new Jev results. The empirical comparison is Jev versus Random on identical
record pools. The published reference is **MLP + Ranking loss**, ResNet-50,
**Top-1 0.8823; Top-5 0.9456**, from the main paper's
[Section 5.2, Table 4, printed page 7][paper]. It is an external reference,
not a locally evaluated, paired baseline. MLP + Ranking loss has the highest
ResNet-50 Top-5 entry in that table; MLP + MSE has a higher Top-1 entry, so the
chosen reference must not be called the best published ResNet-50 Top-1 result.

## Published setting and source evidence

The paper's Section 5.2 uses measurements from Intel Xeon Platinum-8272 and
holds out five networks: ResNet-50, MobileNet-V2, ResNext-50, BERT-tiny and
BERT-base. Training excludes the union of their tasks. Images have batch size
1 and resolution 224; text inputs have sequence length 128. This is distinct
from Section 5.1/Table 3 and from the repository's 500-file tutorial.

The reconstruction covers **one Table 4 network column**, ResNet-50 with
network key `('resnet_50', [(1, 3, 224, 224)])`. It does not reproduce all five
held-out networks, train the paper model, or establish the identity of its
original checkpoint and measurement inventory.

| Primary source | What it establishes |
| --- | --- |
| [Main paper, Sections 4.2 and 5.2/Table 4][paper] | Best measured candidate among the predicted top K; published setting and reference scores |
| [Pinned network evaluator][evaluator] | Complete network task pools, task-frequency weights, throughput floor and endpoint network metric |
| [Pinned file conventions][common] | Network task inventory and platform-specific measurement filenames |
| [Pinned dataset implementation][dataset] | Record loading and native feature/label extraction; network evaluation sets minimum sample size to zero |
| [Native feature extraction][feature-cpp] and [mean helper][utils] | Float32 task minima and normalized labels derived from mean record costs |
| [Record reader][records] and [MeasureResult constructor][measure] | Recorded costs and error codes are read independently; no error-code-based cost replacement |
| [Python feature decoder][feature-py] | Float32-origin labels and minima unpacked into default float64 arrays |

All code citations are pinned to TenSet commit
`35774ed373f9693cc6efcbe83f5703141e8ab0c8`. The official
[supplemental ZIP][supplement] could not be inspected in this environment:
the web reader does not support ZIP content and the authorized download
returned HTTP 403. Appendix B/C definitions are therefore **not independently
verified** here. The detailed metric contract below is derived from the
pinned executable evaluator, consistent with the main paper's best-of-K
description; it is not presented as a quotation from the appendix.

## Frozen evaluation inputs

Read the network's `.task.pkl` inventory from `network_info`, including its
task occurrence weights. Resolve every listed task to its complete measurement
file under `measure_records/platinum-8272`. Require one exact LLVM target with
`-model=platinum-8272` and `-mcpu=skylake-avx512` throughout the cohort.

Use **every record in each resolved file**, preserving source order and
duplicate schedules as distinct record candidates. There is no 500-file
sampling, within-file 90/10 split, valid-only filtering, candidate cap, or
minimum-48-record filter. Both Jev and Random receive the same complete pools.
The inventory, occurrence weights, measurement files, prepared shards and
effective configuration are hash-bound for validation and resume. Report the
observed number of tasks and records; a count matching the paper's illustrative
27 ResNet-50 tasks does not prove source identity.

Workload context contains program/target metadata. Candidate inputs contain
schedule transformations and opaque identifiers. Record runtimes, normalized
throughput labels, task minima, source record positions and error flags remain
private evaluation data. Field descriptions, deterministic candidate shuffling
and opaque identifiers reduce accidental ordering/identifier cues. The seed
`0` is a project choice, not a verified paper seed.

The native reader preserves an error record's supplied costs, even if they
are not the usual `1e10` sentinel. Feature extraction does not filter on error
code or deduplicate records. Failed feature extraction can yield zero feature
rows without removing their labels. An all-error pool is not automatically
dropped by this path; it must not be described as a validated physical runtime
oracle. Empty native task output can instead arise when task reconstruction
fails. The portable preparation rejects empty, nonpositive or nonfinite cost
data rather than silently changing the cohort. Any additional no-success-task
guard must be reported as a stricter local validation rule.

The shared decision adapter currently requires at least two candidate records
per task. Preparation therefore fails early on a singleton task file instead
of excluding it. This is a local adapter limitation, not a paper eligibility
rule; such a failure cannot be reported as a complete network experiment.

## Endpoint network scores

For task \(i\), let \(t_{ic}\) be candidate \(c\)'s mean recorded cost in
seconds, rounded to float32, and let \(m_i=\min_c t_{ic}\). The native mean
accumulates the original cost array in float64 before the float32 cast. Define
the float32-origin normalized throughput label
\(q_{ic}=\operatorname{float32}(m_i/t_{ic})\). Recover the evaluator's latency
proxy using float64 arithmetic:

\[
\ell_{ic}=\frac{m_i}{\max(q_{ic},10^{-5})}.
\]

For predicted ordering \(\pi_i\), task occurrence weight \(w_i\), and
\(k\in\{1,5\}\), report:

\[
S_k=\frac{\sum_i w_i m_i}
{\sum_i w_i\min_{1\le j\le\min(k,n_i)}\ell_{i,\pi_i(j)}}.
\]

Higher is better. Top-5 uses the single best latency among the first five
predictions. It is **not** an average of prefix scores, a mean of per-task
ratios, a record-count-weighted mean, or classification accuracy. The floor
also means the denominator must not be replaced directly with raw sentinel
costs. Task weights come from network extraction frequency, not pool size.

The original evaluator prints a warning and skips missing tasks. This project
requires complete inventory coverage for a full network score. A pilot must
be explicitly labeled as a subset and must not carry the published full-network
reference. The collection runner's ordinary `top_1_score`/`top_5_score`
outputs remain intermediate diagnostics; use **`paper-network-summary.json`**,
**`paper-network-summary.csv`** and the task decomposition export for this
protocol's endpoint network results.

## Preparation, preflight and execution boundaries

[Preparation](../scripts/prepare_tenset_paper.py) reads the existing CPU ZIP or
extracted directory and reconstructs program metadata without installing
legacy TVM or training a cost model. The Mac's processor does not need to match
Platinum-8272: this evaluates selection against recorded Platinum measurements,
not runtimes newly measured on the Mac.

[The runner](../scripts/run_tenset_paper.py) defaults to offline preflight. Both
the default mode and `--preflight-only` stop before loading `.env` or creating
an API client. The [three-task pilot profile][pilot] and
[full-network profile][full] both use complete per-task pools. They retain
workload context and field descriptions, `jev-1.14.0`, long context, at most
255 options per question and five promotions per tournament question. These
are project inference choices; Table 4 does not evaluate Jev or tournaments.

Preflight validates all source pools, reports actual coverage and first-round
packing, and estimates later rounds under two size-based promotion scenarios.
Its character-based token estimates and assumed input-token price are planning
figures, not measured usage or exhaustive cost bounds. They exclude output
charges, retries and cache savings. A preflight estimate does not run the
experiment; live execution requires the explicit `--execute` option.

## Permitted comparison and remaining limits

A completed full-network run can place its endpoint scores alongside the
published MLP + Ranking loss values, with the label **external published
reference; original inventory/checkpoint unverified**. Jev versus Random is
the paired comparison on the locally frozen pools. This supports a
paper-metric-aligned reconstruction, not a claim of exact Table 4 reproduction
or statistical superiority over a model that was not evaluated locally.

The earlier E5-2673 120-task experiments and Platinum 500-file tutorial run
have different cohorts and aggregation rules. Their saved results remain
separate; their Top-K scores cannot be directly substituted into the Table 4
network comparison. A pilot, truncated pool or incomplete network also cannot
be promoted to a full-network paper comparison.

[paper]: https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/a684eceee76fc522773286a895bc8436-Paper-round1.pdf#page=7
[supplement]: https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/a684eceee76fc522773286a895bc8436-Supplemental-round1.zip
[evaluator]: https://github.com/tlc-pack/tenset/blob/35774ed373f9693cc6efcbe83f5703141e8ab0c8/scripts/eval_model_on_dataset.py
[common]: https://github.com/tlc-pack/tenset/blob/35774ed373f9693cc6efcbe83f5703141e8ab0c8/scripts/common.py
[dataset]: https://github.com/tlc-pack/tenset/blob/35774ed373f9693cc6efcbe83f5703141e8ab0c8/python/tvm/auto_scheduler/dataset.py
[feature-cpp]: https://github.com/tlc-pack/tenset/blob/35774ed373f9693cc6efcbe83f5703141e8ab0c8/src/auto_scheduler/feature.cc
[utils]: https://github.com/tlc-pack/tenset/blob/35774ed373f9693cc6efcbe83f5703141e8ab0c8/src/auto_scheduler/utils.h
[records]: https://github.com/tlc-pack/tenset/blob/35774ed373f9693cc6efcbe83f5703141e8ab0c8/src/auto_scheduler/measure_record.cc
[measure]: https://github.com/tlc-pack/tenset/blob/35774ed373f9693cc6efcbe83f5703141e8ab0c8/src/auto_scheduler/measure.cc
[feature-py]: https://github.com/tlc-pack/tenset/blob/35774ed373f9693cc6efcbe83f5703141e8ab0c8/python/tvm/auto_scheduler/feature.py
[pilot]: ../experiments/tenset/paper-resnet50/pilot.yaml
[full]: ../experiments/tenset/paper-resnet50/full.yaml
