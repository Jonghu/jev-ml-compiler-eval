# Completed TenSet ResNet-50 evaluation and reproduction guide

This extension targets one column of the main paper's Section 5.2/Table 4:
ResNet-50, batch 1, input `(1, 3, 224, 224)`, using recorded Platinum-8272
measurements. The published **MLP + Ranking loss** reference is **0.8823
Top-1 / 0.9456 Top-5**. These are external published scores; the original
checkpoint and exact original test inventory have not been matched.

The [protocol and source audit](tenset-paper-protocol.md) explains the full
record pools, task-frequency weights, numerical conventions, and limits of
the comparison. The user completed this run at commit
`f6a2a49f6b2d0dc8b3ae2e9f87ca6b3250d929ea`; its aggregate manifest is dated
2026-11-18. The [audited results](results/tenset/paper-resnet50/README.md),
[run log](tenset-runlog.md#r6--completed-main-paper-aligned-resnet-50-network-study),
and [four-slide contribution](presentations/tenset-rq1/README.md) are available.

The full run contains **27 tasks, 56 occurrences, and 100,126 schedule records**.
Jev's network endpoint Top-1/Top-5 are **0.097724/0.274583**, versus paired
Random-0 **0.091462/0.377121**. Its 631 live and 42 cached requests contain
95,175,703 new input tokens, estimated at **$3.997380 input-only** using the
assumed $0.042/million rate. Output charges and invoice reconciliation remain
outside this estimate. No new calls were made to audit or publish the run.

Sections 1–4 document reproduction and new experiments. The completed run does
not need to be executed again to obtain its results. For remaining offline
tensor/schedule cases and final report integration, use the
[finish guide](run-tenset-finish-rq1.md).

## 1. Preserve the historical checkout for exact replay or resume

The existing data, result bundles, and caches must remain intact. Historical
resume is bound to the original source files, effective configuration,
Python version, and package versions. Integration changes code fingerprints;
do not rewrite old provenance to make changed code accept old execution state.
For historical reproduction from a clean repository folder:

```bash
git fetch origin
git switch --detach f6a2a49f6b2d0dc8b3ae2e9f87ca6b3250d929ea
uv sync --locked --extra tenset --python 3.11.18
```

If Git refuses the switch because of local edits, preserve those edits first.
Check the archived manifest's environment before resuming. For a deliberately
changed experiment under current integrated code, prepare a fresh collection
and choose new configuration/result paths; the default paths below belong to
the original completed experiment and must not be overwritten.

## 2. Prepare the full network offline

For the existing ZIP in Downloads:

```bash
uv run python scripts/prepare_tenset_paper.py \
  --archive "$HOME/Downloads/dataset_cpu_v3.3.zip" \
  --output-dir data/tenset/paper-resnet50-platinum
```

Use the actual ZIP filename if yours differs. The helper safely extracts the
Platinum measurement files and network task metadata into
`data/tenset/paper-resnet50-platinum-raw/`. It then prepares the complete
ResNet-50 task inventory and all records in each task's measurement file.
No separate `metadata.json`, legacy TVM installation, model training, API key,
or new on-device kernel measurements are needed for this preparation.

If the CPU dataset is already extracted, reuse that raw root instead:

```bash
uv run python scripts/prepare_tenset_paper.py \
  --raw-root /path/to/extracted/dataset_cpu \
  --output-dir data/tenset/paper-resnet50-platinum
```

That root must contain the `network_info` and
`measure_records/platinum-8272` directories, possibly below one enclosing
dataset directory. A prepared collection's `index.json` is not the raw root.
Do not run both preparation commands into the same nonempty output directory.

Preparation prints progress and the resulting index path. If interrupted,
repeat the same command with `--resume`, keeping the source, extraction path,
and code unchanged. Resuming verifies the frozen source hashes and completed
shards; it does not select a new subset. Changing code after preparation may
require a fresh output directory instead of resuming the old preparation.

The new collection is:

```text
data/tenset/paper-resnet50-platinum/index.json
```

This is not a 500-file sample or a 90/10 record split. The reference paper
describes 27 ResNet-50 tasks; preparation reports the actual task count as a
diagnostic. It never drops a task merely to match that count. Unsupported or
missing records fail before live execution instead of silently shrinking the
benchmark. Original-paper identity remains unverified even if counts match.

## 3. Check the pilot and full cost without API calls

```bash
uv run python scripts/run_tenset_paper.py \
  --config experiments/tenset/paper-resnet50/pilot.yaml \
  --preflight-only
```

The pilot uses the first three tasks in the frozen network inventory, with
their complete per-task candidate pools. It is a software/inference pilot,
not a full ResNet-50 result comparable to the published reference.

Save the full-network preflight for cost review:

```bash
mkdir -p results/tenset/paper-resnet50
uv run python scripts/run_tenset_paper.py \
  --config experiments/tenset/paper-resnet50/full.yaml \
  --preflight-only \
  | tee results/tenset/paper-resnet50/preflight-full.txt
```

The runner defaults to offline preflight even when `--preflight-only` is
omitted. It does not read `.env` or construct an API client in this mode.
Both profiles validate the complete source inventory, including tasks outside
the pilot, before requests are possible.

Inspect:

- `full_network_tasks` and `full_network_total_candidates`: actual frozen scope.
- `model_plans`: first-round requests/tokens and later-round promotion scenarios.
- `cost_assumptions`: first-round estimated input cost and scenario totals.
- `warnings` and `reference_alignment_status`: estimation and comparison limits.

The default input price is an **assumption of $0.042 per million input tokens**.
Override it with `--input-price-per-million` if your account has a different
rate. These are character-based planning estimates, not provider token counts,
invoices, or guaranteed upper bounds. Later-round candidates depend on Jev's
choices; the two serialized-size promotion scenarios are not exhaustive. Output
charges and retries are excluded, and cache savings are not assumed. Earlier
484-workload token usage cannot determine this new pool's price.

Share `results/tenset/paper-resnet50/preflight-full.txt` to review the actual
scope and estimates before making a new paid run. It contains local file paths
and hashes but no API key. The CPU ZIP and prepared shards need not be uploaded.

## 4. Reproduction only: execute the prepared profiles

The commands in this section **make paid Jev API calls**. The original full run
has already completed. Use them only for an intended new execution or an
interrupted historical run with its exact recorded identity.

The existing `TYPESAFE_API_KEY` in the repository-root `.env` is used for live
execution. An exported shell value takes precedence over `.env`; unset it if
you intend to use the file's value, without printing either value.

Run the pilot first:

```bash
uv run python scripts/run_tenset_paper.py \
  --config experiments/tenset/paper-resnet50/pilot.yaml \
  --execute
```

After reviewing the pilot, execute the complete network:

```bash
uv run python scripts/run_tenset_paper.py \
  --config experiments/tenset/paper-resnet50/full.yaml \
  --execute
```

For an interrupted run, repeat its exact command with `--resume`. Keep the
config, source data, prepared collection, code, and environment unchanged.
The shared runner verifies completed per-task bundles and reuses their saved
results. Use a new output directory for a deliberately changed experiment.
Pilot and full profiles use separate result paths and share a content-addressed
API cache. Identical successful pilot requests can be reused by the full run.
Preflight remains conservative by assuming no cache savings; retain the cache
and inspect recorded live/cached usage after execution.

## 5. Use the endpoint network summary

Full-network outputs are under `results/tenset/paper-resnet50/full/`:

| File | Purpose |
|---|---|
| `paper-network-summary.json` | Endpoint network scores, coverage, paired Jev/Random differences, and separately labelled published MLP reference |
| `paper-network-summary.csv` | Each model's endpoint network scores, coverage and weighted latency terms |
| `paper-task-decompositions.csv` | Task weights, private normalizers, selected candidates and weighted latency contributions |
| `predictions.jsonl` | Per-task predictions, shared candidate sets, input fingerprints, usage and request/cache metadata |
| `manifest.json`, `collection-state.json` | Data/config/code/environment identity and execution state |
| `metrics.json`, `summary.csv` | Shared runner's intermediate task/prefix metrics; not the paper network score |

The primary metric names are `paper_network_best_of_1_score` and
`paper_network_best_of_5_score`. They use the network's task occurrence
weights and the best candidate at the K-choice endpoint. This differs from
the earlier tutorial's record-weighted mean of prefix-average scores.
Do not paste the generic intermediate Top-1/Top-5 into a Table 4 comparison.

Display the complete-network score table after execution:

```bash
cat results/tenset/paper-resnet50/full/paper-network-summary.csv
```

Pilot metrics are named `pilot_network_*` and omit the full-network published
reference. No global Pairwise Accuracy is inferred from the tournament tail.
Reported weighted latency terms reconstruct the reference evaluator's proxy;
they are not new measured end-to-end network inference latencies.

The completed full run can be discussed against the published MLP reference as
a reconstruction aligned with the pinned network evaluator. It does not
establish a locally paired comparison with the original checkpoint or exact
Table 4 reproduction. Jev versus Random on the frozen local pools is the paired
comparison. Keep this extension separate from the completed 484-workload
tutorial and 120-workload candidate-size studies.

The full run also exports `paper-reference-comparison.csv` and
`paper-reference-comparison.md`: a three-row table for **Jev, Random, and
the published MLP + Ranking loss reference**. Each row identifies its evidence
source and whether its candidate pool is verified against the local experiment.
The MLP row is external, not an additional locally evaluated model. Only Jev
versus Random supplies paired results. These comparison tables are omitted
for the pilot.

## 6. Finish analysis and reporting without another paid run

Saved-output checks confirm identical paired pools, exact Random replay,
and all 108 task/K decompositions. The small Top-1 lead is sensitive to
individual tasks; Top-5 is lower than Random. In 100,000 offline Random-choice
draws on the same fixed network, 95.320% exceed saved Jev Top-5. That fraction
is a reference distribution, not a p-value or fresh Jev repeatability check.
The original paper inventory/checkpoint and raw transmitted request contents
remain unverified; that limit is not a demonstrated scoring defect.

Remaining deliverables are concrete tensor/schedule case exports and reviewed
semantic narratives, integration into the team report, and verification of
actual pricing/output charges and billing. The published MLP is a trained
neural-network ranking model used as an external reference; training a matched
model is outside the current scope. Fresh Jev, presentation-order, tie-breaking,
context, and grouping experiments remain optional follow-ups.
