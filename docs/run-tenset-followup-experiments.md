# TenSet field descriptions, controls, and full-candidate tournament

These profiles extend the completed 120-workload, 50-schedule Jev-versus-Random
experiment. They do not rerun or replace its saved results. All API calls below
are manual: preparing data, preflight, and analysis do not call Jev.

The shared tournament and request controls are reused from the TpuGraphs
workstream at commit `61fe406`, with TenSet long-context support and
question-scoped probability reporting.

Use Python 3.11 and the committed environment, from the repository root:

```bash
uv sync --locked --extra tenset
```

The fixed-cohort profiles expect the existing artifact:

```text
data/tenset/prepared-w120-c50-s42-metadata.json
```

Keep `.env` with `TYPESAFE_API_KEY` configured as in the completed experiment.
Do not repeat extraction or download the CPU archive again.

## Experimental conditions

All new Jev profiles pin `jev-1.14.0`, keep seed 42, and explicitly use long
context. New requests use `.cache/jev-tenset-v3-long-context`, whose cache keys
preserve the order sent to the model. The older cache and results remain an
archive of the completed experiment; the new client does not migrate or delete
them. It rejects incompatible cache schemas instead of guessing the old order.

| Stage | Workloads and candidates | Factor being tested |
| --- | --- | --- |
| Field definitions | Same 120 test workloads, same 50 schedules | Official syntax and argument definitions added once to the context |
| No-context / shuffled-order controls | Same 120 x 50 | Each variant changes one input factor from the archived baseline |
| Blind single-question control | Same 120 x 50 | Neutral aliases and shuffled order, matching the tournament presentation |
| Forced-group tournament | Same 120 x 50 | At most 25 choices per question, advancing five, to exercise elimination |
| Full-candidate pilot | First three eligible workloads of a prepared collection | Packing, API limits, resumable execution, and accounting |
| Full-candidate evaluation | All eligible held-out workload groups; all valid distinct schedules | Larger evaluation scope using at most 255 choices per question |

The field definitions come from pinned TenSet TVM sources, including precise
step argument order, annotation codes, and evolving stage/iterator indices.
They provide neither performance-direction advice nor measured timings. All
models in a run receive the same cases and configurations. Candidate IDs are
already opaque hashes; the tournament additionally uses neutral aliases and a
seeded presentation order.

These are follow-up, exploratory experiments on the already-inspected 120
test cases. Develop further prompt changes on validation workloads and freeze
them before evaluating an untouched test cohort. The full test collection can
include the previously inspected cases and is not automatically a new
confirmatory holdout.

## 1. Field-description experiment

First validate all 120 cases without an API client:

```bash
uv run python scripts/preflight_experiment.py \
  --config experiments/tenset/jev-field-descriptions-w120-c50.yaml
```

Then run ten workloads and, after they succeed, all 120. The larger profile
reuses successful responses from the first ten through the common cache.

```bash
uv run python scripts/run_experiment.py \
  --config experiments/tenset/jev-field-descriptions-w10-c50.yaml

uv run python scripts/run_experiment.py \
  --config experiments/tenset/jev-field-descriptions-w120-c50.yaml
```

The original prepared artifact, workload IDs, schedules, and timings are
unchanged. Dataset provenance records the description flag, and input hashes
reflect the added definitions.

## 2. Context and order controls

This profile contains Random, Jev without program context, and Jev with a
different seeded candidate presentation order. It does not add field definitions,
so each Jev condition changes one factor relative to the archived baseline.

```bash
uv run python scripts/preflight_experiment.py \
  --config experiments/tenset/jev-context-order-ablation-w120-c50.yaml

uv run python scripts/run_experiment.py \
  --config experiments/tenset/jev-context-order-ablation-w120-c50.yaml
```

Order-sensitive caching is essential: reordered choices are distinct requests.
The exact sent request and per-request hash are retained in the new cache.
Model metadata records the context/order options and presented candidate IDs.

After both 120-workload experiments complete, compare them with the archived
baseline. Analyze the full bundles, not their ten-workload canaries:

```bash
uv run python scripts/analyze_tenset.py \
  --results \
    results/tenset/random-vs-jev-w120-c50-s42-long-context \
    results/tenset/jev-field-descriptions-w120-c50 \
    results/tenset/jev-context-order-ablation-w120-c50 \
  --dataset data/tenset/prepared-w120-c50-s42-metadata.json \
  --output-dir results/tenset/followups-c50-analysis
```

Repeated Random rows are references for paired runs, not additional independent
workloads. Use `by_run.csv`, `per_workload.csv`, and `unique_workloads` when
reporting the comparisons. Jev responses can vary between live requests;
cached repeatability is not evidence that the service is deterministic.

## 3. Tournament on the existing 120 x 50 cohort

First run a blind single-question control. It uses the same neutral aliases,
shuffled order, field definitions, and API mode as the forced-group tournament:

```bash
uv run python scripts/preflight_experiment.py \
  --config experiments/tenset/jev-tournament-single-question-w120-c50.yaml

uv run python scripts/run_experiment.py \
  --config experiments/tenset/jev-tournament-single-question-w120-c50.yaml
```

Then exercise elimination before expanding the candidate pools:

```bash
uv run python scripts/preflight_experiment.py \
  --config experiments/tenset/jev-tournament-w120-c50.yaml

uv run python scripts/run_experiment.py \
  --config experiments/tenset/jev-tournament-w120-c50.yaml

uv run python scripts/analyze_tenset.py \
  --results \
    results/tenset/jev-tournament-single-question-w120-c50 \
    results/tenset/jev-tournament-w120-c50 \
  --dataset data/tenset/prepared-w120-c50-s42-metadata.json \
  --output-dir results/tenset/jev-tournament-w120-c50-analysis
```

With 50 candidates, a 255-option limit might require only one question. The
25-option control explicitly exercises grouping and advancing finalists. Compare
it with the blind single-question control to hold presentation fixed. It
is a separate algorithm ablation, not the same group-size condition as the
TpuGraphs 255-option experiment. Field definitions are enabled in both the
field-description run and this tournament run.

Tournament profiles use Top-1 Score, Top-5 Score, and runtime regret. Their
probabilities are only comparable within individual questions. The full returned
ranking places finalists first and eliminated candidates in deterministic
round order; the tail is not a global learned ordering. Global Pairwise Accuracy
is therefore omitted, and selection-only tournament bundles are analyzed
separately from the five-metric single-question bundles.

## 4. Prepare every eligible held-out workload with all schedules

The frozen 120 x 50 artifact discarded the other schedules. Removing a YAML
candidate cap cannot recover them. Prepare a new collection from the existing
raw E5-2673 records:

```bash
uv run python scripts/prepare_tenset_collection.py \
  --raw-root data/tenset/raw \
  --platform e5-2673 \
  --target-from-prepared data/tenset/prepared-w120-c50-s42-metadata.json \
  --task-registry data/tenset/raw/network_info/all_tasks.pkl \
  --split test \
  --seed 42 \
  --min-candidates 2 \
  --output-dir data/tenset/full-test
```

This step makes no API requests. It freezes source hashes and preparation
settings, aggregates each workload's repeated records before writing its shard,
attaches semantic metadata with the portable reader, and retains every valid
distinct schedule. It keeps one workload pool in memory instead of the whole
collection. The complete `index.json` is published only after preparation
finishes. The eligible count is reported; 448 indexed test groups from the
earlier preparation do not guarantee 448 eligible workloads.

If interrupted, repeat the same command with `--resume`. Completed shards are
verified and reused; changed sources, metadata, settings, or preparation code
are rejected. Keep the original data and collection files. Missing or unsupported
metadata fails preparation by default; explicitly excluding unsupported programs
requires `--skip-unsupported-programs` and is recorded as a scope restriction.

For development on validation rather than test, use `--split validation` and
a separate output directory. A source-workload pilot can use `--workload-limit 3`
with its own output directory; it is not the complete test collection.

The optional validation pilot has a matching profile:

```bash
uv run python scripts/prepare_tenset_collection.py \
  --raw-root data/tenset/raw \
  --platform e5-2673 \
  --target-from-prepared data/tenset/prepared-w120-c50-s42-metadata.json \
  --task-registry data/tenset/raw/network_info/all_tasks.pkl \
  --split validation --seed 42 --min-candidates 2 --workload-limit 3 \
  --output-dir data/tenset/validation-pilot

uv run python scripts/run_tenset_collection.py \
  --config experiments/tenset/jev-tournament-validation-pilot.yaml \
  --preflight-only

uv run python scripts/run_tenset_collection.py \
  --config experiments/tenset/jev-tournament-validation-pilot.yaml
```

## 5. Preflight and run the full-candidate tournament

The collection needs a dedicated runner. It evaluates one shard at a time,
reuses the model clients and rate limiter, and checkpoints completed workloads.
It uses the same common evaluation code and writes a canonical aggregate result
bundle only when every selected workload succeeds. Failures are not silently
excluded.

Validate the entire collection without API calls before starting the pilot:

```bash
uv run python scripts/run_tenset_collection.py \
  --config experiments/tenset/jev-tournament-full-test.yaml \
  --preflight-only
```

Preflight reports realized workload/candidate counts and first-round packing
estimates. Those character-based token estimates are conservative planning
heuristics, not a verified tokenizer. Later rounds depend on actual Jev choices.
A context that cannot fit even with a small candidate group fails preflight;
splitting candidates cannot remove an oversized program context.

Run the three-workload pilot, then the complete collection after checking its
usage, output, and latency:

```bash
uv run python scripts/run_tenset_collection.py \
  --config experiments/tenset/jev-tournament-full-test-pilot.yaml

uv run python scripts/run_tenset_collection.py \
  --config experiments/tenset/jev-tournament-full-test.yaml
```

To continue an interrupted evaluation, repeat its exact command with `--resume`.
The pilot and complete evaluation have separate output directories and share
the request cache. Completed child bundles are hash-checked before reuse. A
different index, model configuration, candidate policy, or scope requires a new
experiment directory. Pilot plus full results must not be pooled as extra cases.
Resume also checks source-code and environment fingerprints: keep the same
implementation, Python, and package versions until that evaluation completes.

Full-candidate TenSet requests contain much more schedule data than TpuGraphs
tile bounds. Inspect the pilot and preflight counts before committing to the
full run; its request count and cost need not resemble TpuGraphs'. Long context
raises the service limit, but a workload can still exceed it.

Analyze the complete result, using the collection index as its dataset:

```bash
uv run python scripts/analyze_tenset.py \
  --results results/tenset/jev-tournament-full-test \
  --dataset data/tenset/full-test/index.json \
  --output-dir results/tenset/jev-tournament-full-test-analysis
```

`overall.csv` is the macro mean over workload decisions. `by_candidate_count.csv`
and `by_operation.csv` provide breakdowns. Request-level accounting separates
live and cached tournament calls. Each workload's original child result bundle
remains available alongside the complete aggregate. Published XGBoost values
remain an external reference rather than a matched baseline for this cohort.
