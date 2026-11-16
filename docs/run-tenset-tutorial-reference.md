# Run the reconstructed TenSet 500-file tutorial test pools

This workflow reconstructs the official repository's Platinum-8272, batch-size-1
example at TenSet commit `35774ed373f9693cc6efcbe83f5703141e8ab0c8`. It evaluates
Jev and Random on the reconstructed test records. The published XGBoost numbers
remain an **external tutorial reference**: the original selected-file inventory,
test-record identities, and matching XGBoost artifacts have not been verified.
This is neither a controlled three-model comparison nor a verified reproduction
of a paper table. Matching XGBoost predictions or a checkpoint with matching
training provenance is still required for a paired baseline comparison.

Data preparation, preflight, and reading saved outputs make no API calls.
Live execution requires `--allow-external-reference`, which explicitly preserves
the limitation above in the saved results. The flag does not supply an XGBoost
model or certify that the original tutorial subset has been recovered.

## 1. Update the code on the Mac

Run these commands from the repository. Keep the existing E5-2673 prepared data,
cache, and 120-workload results; this workflow uses separate directories.

```bash
cd "$HOME/Projects/jev-ml-compiler-eval"
git fetch origin
git switch feat/tenset-tutorial-reference
git pull --ff-only
uv sync --locked --extra tenset
```

Git will stop if switching branches would overwrite local changes. Preserve
those changes before continuing; these instructions do not discard them.

## 2. Extract Platinum measurements and network inventories

The downloaded CPU archive contains the measurement hardware's recorded labels.
The Mac runs the evaluation client; it does not need a Platinum CPU. The filename
below assumes the official ZIP is still in Downloads. Adjust only `--archive`
if your local filename or location differs.

```bash
uv run python scripts/download_tenset.py \
  --archive "$HOME/Downloads/dataset_cpu_v3.3.zip" \
  --platform platinum-8272 \
  --include-network-tasks \
  --output-dir data/tenset/raw-tutorial-platinum
```

Run extraction once into an empty destination. It preserves archive and file
checksums and refuses to overwrite an existing extraction. If the extraction
already completed, proceed to preparation instead of running it again. No
additional download is needed. The network `.task.pkl` inventories are needed
to reconstruct the preset's ordered file population; a platform-only extraction
from the earlier E5-2673 workflow does not contain all required inputs.

## 3. Prepare the complete reconstructed test collection offline

```bash
uv run python scripts/prepare_tenset_tutorial.py \
  --raw-root data/tenset/raw-tutorial-platinum \
  --output-dir data/tenset/tutorial-500-platinum
```

This uses the tutorial's default 500-file selection and seed 0. It reads the
known task-registry format with the portable data-only decoder, so no legacy TVM
installation or separately created `metadata.json` is needed. It does not train
XGBoost or reproduce TVM's native performance features.

The reconstruction preserves record identity and ordering, repeated schedules,
and timing-error records. A retained workload needs at least 48 records. The
native default split reverses the record sequence, takes `int(0.9 * N)` records
for training, and uses the remainder for testing. Consequently the test portion
is the first approximately 10% of the original records, in reversed order;
it is not a random schedule split or our project-defined workload hash split.
Timing-error records retain their recorded reference costs, including failure
sentinels; those values are not successful execution latencies measured on the Mac.
Invalid empty, nonfinite, or nonpositive timing labels fail preparation rather
than silently changing the reference pool. Measured values and source-record
indices remain private and never enter the Jev context or candidate configuration.

Preparation prints and saves `data/tenset/tutorial-500-platinum/reference-status.json`.
The tutorial's published example reports 484 retained workloads, 1,689,022
training records, and 187,679 test records. The status file compares the realized
counts with these reference counts. Equal counts alone do not establish identical
test records or matching XGBoost predictions; count differences must be reported.
Do not rename a changed pool as the original published experiment.

If preparation is interrupted, repeat the same command with `--resume`:

```bash
uv run python scripts/prepare_tenset_tutorial.py \
  --raw-root data/tenset/raw-tutorial-platinum \
  --output-dir data/tenset/tutorial-500-platinum \
  --resume
```

Keep the same sources, preparation settings, and implementation when resuming.
The complete `index.json` is published after preparation finishes.

## 4. Validate all test pools without API calls

```bash
uv run python scripts/run_tenset_tutorial.py \
  --config experiments/tenset/jev-tutorial-500-test.yaml \
  --preflight-only
```

Check the realized workload and candidate counts and the first-round request
packing before live execution. The planner uses character-based token estimates,
not a verified service tokenizer. Later rounds depend on Jev's choices. These
estimates neither certify API admission nor enforce a billed-token or dollar cap.
Oversized program contexts fail preflight rather than dropping candidates.

## 5. Configure the TypeSafe API key

The key belongs in `.env` at the repository root, alongside `pyproject.toml`.
Keep an existing `.env`; the commands below only create one if it is absent.

```bash
if [ ! -f .env ]; then
  cp .env.example .env
fi
nano .env
```

Set this line in the editor, replacing the placeholder with your actual key:

```dotenv
TYPESAFE_API_KEY=your-api-key
```

In nano, save with Control-O, press Enter, then exit with Control-X. The file is
ignored by Git. Do not commit or share its contents. An already exported
`TYPESAFE_API_KEY` takes precedence over `.env`. To use the key in `.env` for
these runs, remove that shell override without printing any credential:

```bash
unset TYPESAFE_API_KEY
```

Preflight does not need the key and does not load `.env` or create an API client.

## 6. Run the three-workload pilot

This command makes Jev API calls. It retains every test candidate for the first
three prepared workloads rather than resampling 50 schedules.

```bash
uv run python scripts/run_tenset_tutorial.py \
  --config experiments/tenset/jev-tutorial-500-pilot.yaml \
  --allow-external-reference
```

Inspect the saved predictions, usage, cache status, and weighted scores under
`results/tenset/tutorial-500-platinum-pilot/` before proceeding. The previously
discussed approximately $8 cost for the complete test portion was a linear
estimate from another run, not a quote or spending cap. Use the actual pilot
usage and realized collection size to refine it. Cached responses are previous
usage; do not charge their stored token counts again as new calls.

The CLI prints the record-weighted scores. Read their exact values with:

```bash
cat results/tenset/tutorial-500-platinum-pilot/official-weighted-summary.csv
```

For the shared token, cache, and latency analysis, run:

```bash
uv run python scripts/analyze_tenset.py \
  --results results/tenset/tutorial-500-platinum-pilot \
  --dataset data/tenset/tutorial-500-platinum/index.json \
  --output-dir results/tenset/tutorial-500-platinum-pilot-analysis
```

This analysis uses the runtime-based canonical metrics. Keep using
`official-weighted-summary.csv` for the reference-formula score comparison.

## 7. Run the complete test portion

```bash
uv run python scripts/run_tenset_tutorial.py \
  --config experiments/tenset/jev-tutorial-500-test.yaml \
  --allow-external-reference
```

The pilot and complete evaluation have separate result directories and share
the tutorial-specific request cache. Do not pool the pilot with the complete
run as additional independent workloads. Continue an interrupted evaluation
with its exact original command plus `--resume`, for example:

```bash
uv run python scripts/run_tenset_tutorial.py \
  --config experiments/tenset/jev-tutorial-500-test.yaml \
  --allow-external-reference \
  --resume
```

Keep the same code, Python environment, configuration, and prepared files until
the run completes. Source and environment identities are checked during resume.

## Outputs and interpretation

The completed run writes its canonical bundle under
`results/tenset/tutorial-500-platinum-test/`, including `predictions.jsonl`,
`manifest.json`, `metrics.json`, `summary.csv`, and `experiment.yaml`.
The tutorial wrapper additionally writes:

| File | Meaning |
|---|---|
| `official-weighted-summary.csv` / `.json` | Top-1 and Top-5 recomputed from private float32 reference labels, with test-record-count weighted means and separately named workload macro means |
| `external-xgboost-reference.json` | Published tutorial XGBoost scores, source link, and explicit unmatched-reference status |

Use the weighted Top-1/Top-5 columns when discussing the tutorial's aggregation.
The canonical `metrics.json` remains runtime-based and workload-macro; its
scores can also differ slightly because of floating-point rounding. The wrapper
does not invent XGBoost predictions or infer XGBoost regret from its aggregate
Top-1 Score.

The profiles report Top-1 and Top-5 only. They omit runtime regret because the
reference pool retains failed-measurement sentinel costs, and no published
XGBoost regret is available for this example.

The Jev profile uses long context, field descriptions, neutral shuffled candidate
IDs, at most 255 choices per question, and five advancing candidates per group.
Its eliminated tail is not a global learned ranking, so global tournament
Pairwise Accuracy is omitted. A published XGBoost Pairwise Accuracy value must
not be compared with an arbitrary cross-group tournament ordering.

The published reference source is the
[official 500-file tutorial](https://github.com/tlc-pack/tenset/blob/35774ed373f9693cc6efcbe83f5703141e8ab0c8/docs/try_pretrained_models_on_aws_c59x.md#L78).
Keep the completed E5-2673 [120-workload results](tenset-runlog.md) separate.
