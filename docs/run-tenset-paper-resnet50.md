# Prepare the TenSet paper's ResNet-50 network evaluation

This extension targets one column of the main paper's Section 5.2/Table 4:
ResNet-50, batch 1, input `(1, 3, 224, 224)`, using recorded Platinum-8272
measurements. The published **MLP + Ranking loss** reference is **0.8823
Top-1 / 0.9456 Top-5**. These are external published scores; the original
checkpoint and exact original test inventory have not been matched.

The [protocol and source audit](tenset-paper-protocol.md) explains the full
record pools, task-frequency weights, numerical conventions, and limits of
the comparison. This setup is implemented and tested offline. No new Jev
benchmark results or measured cost estimates have been produced yet.

## 1. Update the Mac checkout

From your repository folder:

```bash
git fetch origin
git switch feat/tenset-paper-resnet50
git pull --ff-only
uv sync --locked --extra tenset
```

Keep the earlier prepared data, results, and caches. If Git refuses a switch
because of local edits, preserve those edits before continuing. This extension
writes to new paths and does not resume or overwrite the older experiments.

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

## 4. Execute later using the prepared profiles

The commands in this section **make paid Jev API calls**. They are execution
instructions, not evidence that those runs have already happened.

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

A completed full run can be discussed against the published MLP reference as
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
