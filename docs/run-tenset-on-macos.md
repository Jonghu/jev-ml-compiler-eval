# Run the TenSet experiment on macOS

This guide starts from `main` and an existing `dataset_cpu_v3.3.zip` in your
Downloads folder. It runs Random and Jev on the same measured CPU schedules,
with program metadata exported by the portable Python reader. You do not need a
GPU, TVM installation, Docker, or an XGBoost checkpoint for this workflow.

The first run evaluates one workload with eight candidate schedules. Treat it
as a pipeline pilot. A larger experiment and a complete report are separate
steps. All commands below run in Terminal from the repository root unless a
step explicitly changes directories.

## 1. Install tools and get the private repository

If Homebrew is not installed, follow its instructions at
[brew.sh](https://brew.sh/), including the PATH setup printed by its installer.
Open a new Terminal window afterward. Install the tools:

```bash
brew install git gh uv
```

Sign in to the GitHub account that has access to the private repository. Follow
GitHub CLI's browser prompts:

```bash
gh auth login --hostname github.com --git-protocol https --web
gh auth setup-git
```

For a fresh checkout:

```bash
mkdir -p "$HOME/Projects"
cd "$HOME/Projects"
gh repo clone Jonghu/jev-ml-compiler-eval
cd jev-ml-compiler-eval
git switch main
git pull --ff-only origin main
```

If you already cloned the repository, open Terminal in that existing folder and
run `git status --short`, `git switch main`, and `git pull --ff-only origin main`.
Keep existing local changes and experiment artifacts. If Git reports a local
change or branch conflict, resolve that before continuing; do not reset or
clean the checkout to follow this guide.

## 2. Install the Python environment

```bash
uv python install 3.11
uv sync --locked --extra tenset
uv run python --version
uv run pre-commit install
uv run pre-commit run --all-files
```

Python should report version 3.11. `uv run` uses the project environment, so you
do not need to activate `.venv`. The `tenset` extra includes plotting and download
tools; importing your existing ZIP does not download the dataset again.

## 3. Extract the E5-2673 collection

Check the ZIP filename and available disk space:

```bash
ls -lh "$HOME/Downloads/dataset_cpu_v3.3.zip"
df -h . "$HOME/Downloads"
```

If your filename differs, change the `--archive` path below. Extraction creates
additional files alongside the large ZIP, so allow space for the selected CPU
collection. Run this once:

```bash
uv run python scripts/download_tenset.py \
  --archive "$HOME/Downloads/dataset_cpu_v3.3.zip" \
  --platform e5-2673 \
  --output-dir data/tenset/raw
```

This extracts only the selected platform's measurement records and the task
registry. Expected paths are:

```text
data/tenset/raw/measure_records/e5-2673/*.json
data/tenset/raw/network_info/all_tasks.pkl
data/tenset/raw/download-manifest.json
```

The command refuses to overwrite a nonempty destination. If you already
completed this extraction, keep those files and proceed to the next step.

## 4. Prepare a small, fixed source pool

If `data/tenset/prepared-pilot.json` already exists from your earlier pilot,
keep it and skip this step. Enrichment in the next step preserves its workload
IDs, splits, candidate IDs, schedules, and measured runtimes.

For a fresh pilot, this command reads the first four sorted workload files,
detects the exact serialized target, and requires at least eight valid distinct
schedules in each retained workload. The target describes the dataset's Intel
CPU; it should not be changed to your Mac's CPU.

```bash
uv run python - <<'PY'
import json
from pathlib import Path

from jev_compiler_eval.datasets.tenset import prepare_records
from jev_compiler_eval.datasets.tenset_download import sha256_file

raw = Path("data/tenset/raw")
output = Path("data/tenset/prepared-pilot.json")
if output.exists():
    raise SystemExit(f"Keep the existing artifact: {output}")
paths = sorted((raw / "measure_records/e5-2673").glob("*.json"))[:4]
if len(paths) != 4:
    raise SystemExit("Expected at least four extracted E5-2673 measurement files")
target = None
for path in paths:
    with path.open(encoding="utf-8") as records:
        for line in records:
            if line.strip() and not line.lstrip().startswith("#"):
                target = json.loads(line)["i"][0][1]
                break
    if target is not None:
        break
if target is None:
    raise SystemExit("No measurement records were found in the selected files")
prepared = prepare_records(paths, target=target, seed=42, min_candidates=8)
prepared["provenance"].update({
    "platform": "e5-2673",
    "purpose": "small source pool for a metadata-enriched pilot",
    "source_file_selection": "first four sorted measurement filenames",
    "source_files_selected": len(paths),
    "dataset_license": "CC BY 4.0",
    "download_manifest_sha256": sha256_file(raw / "download-manifest.json"),
})
with output.open("x", encoding="utf-8") as stream:
    json.dump(prepared, stream, indent=2, allow_nan=False)
    stream.write("\n")
print("Exact target:", target)
print("Prepared workloads:", len(prepared["workloads"]))
print("Splits:", prepared["provenance"]["split_counts"])
print("Saved:", output)
PY
```

The source pool is deliberately small and is not a representative dataset
sample. If none of these four files has enough valid schedules, preparation
fails without creating the output; increase the source-file count explicitly
and record that choice before continuing. These are real dataset measurements,
so do not mark this artifact `synthetic`.

## 5. Export program metadata and enrich the same cases

The portable exporter reads the registry as data and reconstructs tensor shapes,
operations, computation expressions, and loops. It does not import TVM or run
tensor programs. `--prepared` limits semantic conversion to your selected
workloads, although the registry itself is read first.

```bash
uv run python scripts/export_tenset_metadata.py \
  --backend portable \
  --task-registry data/tenset/raw/network_info/all_tasks.pkl \
  --prepared data/tenset/prepared-pilot.json \
  --output data/tenset/metadata-pilot.json

uv run python scripts/enrich_tenset.py \
  --prepared data/tenset/prepared-pilot.json \
  --metadata data/tenset/metadata-pilot.json \
  --output data/tenset/prepared-metadata.json
```

Keep the original pilot artifact. The new artifact has the same cases and
labels, with program metadata added. Export also writes
`data/tenset/metadata-pilot.json.manifest.json`; enrichment records the parent
artifact and metadata hashes.

These commands refuse existing outputs. Skip a command you already completed
for these exact inputs. For changed inputs, use new filenames and update the
experiment's dataset path. If the exporter reports an unsupported registry or
expression, preserve the error and stop here: a schedule-only fallback does not
satisfy the committed profile's full-program-metadata requirement.

## 6. Check the pilot without making an API request

The committed profile is `experiments/tenset/jev-vs-random.yaml`. It uses:

- `data/tenset/prepared-metadata.json`;
- Random and the pinned `jev-1.14.0` model;
- one workload, eight candidates, seed 42, and `split: all`;
- required program metadata and a local input-size guard;
- zero automatic Jev retries.

Load the exact selected case before entering an API key:

```bash
uv run python - <<'PY'
from jev_compiler_eval.config import load_experiment_config
from jev_compiler_eval.datasets import get_dataset_class

config = load_experiment_config("experiments/tenset/jev-vs-random.yaml")
assert {model.model_type for model in config.models} == {"random", "jev"}
dataset = get_dataset_class(config.dataset.dataset_type)(config.dataset.config)
cases = dataset.load()
assert len(cases) == 1
assert len(cases[0].candidates) == 8
print("Workload:", cases[0].workload.workload_id)
print("Operation:", cases[0].workload.context["operation"])
print("Candidates:", len(cases[0].candidates))
print("Input bytes:", dataset.manifest_metadata()["realized_cases"][0]["serialized_input_bytes"])
print("Offline preflight passed. No API request was made.")
PY
```

If preflight fails, resolve its error before running Jev. The byte guard is a
local policy; it is not a verified TypeSafe token or service limit.

## 7. Configure the TypeSafe API key

Create or open the ignored `.env` file without replacing existing contents:

```bash
touch .env
chmod 600 .env
nano .env
```

Add or update this line with your actual key:

```dotenv
TYPESAFE_API_KEY=your_actual_key_here
```

In nano, press Control+O, Enter, then Control+X to save and exit. Keep the key
out of experiment YAML, Git commits, screenshots, and messages. An already
exported `TYPESAFE_API_KEY` takes precedence over `.env`.

## 8. Run the one-workload comparison and analysis

This run can make one live Jev decision if the exact request is not already
cached. API access, model availability, and account usage charges depend on
your TypeSafe account; the repository does not assume a dollar price.

If `results/tenset/random-vs-jev-c8-s42` already contains your successful pilot,
keep it and skip the run command below. Existing results can be analyzed only
with the exact prepared artifact used for that run: check `dataset.config.path`
in its saved `experiment.yaml`. An older schedule-only run must not be paired
with the new `prepared-metadata.json`. Use step 9 to run the enriched pilot into
a new result directory. If an analysis destination already contains files, use
a new analysis directory as well.

```bash
uv run python scripts/run_experiment.py \
  --config experiments/tenset/jev-vs-random.yaml

uv run python scripts/analyze_tenset.py \
  --results results/tenset/random-vs-jev-c8-s42 \
  --dataset data/tenset/prepared-metadata.json \
  --output-dir results/tenset/random-vs-jev-c8-s42-analysis
```

The experiment directory contains `experiment.yaml`, `manifest.json`,
`predictions.jsonl`, `metrics.json`, and `summary.csv`. Analysis adds CSV tables,
`summary.json`, and `comparison.png`/`comparison.svg` in its separate directory.
Open the plot on your Mac:

```bash
open results/tenset/random-vs-jev-c8-s42-analysis/comparison.png
```

Inspect Jev's cache and usage fields without displaying your key:

```bash
uv run python - <<'PY'
import json
from pathlib import Path

path = Path("results/tenset/random-vs-jev-c8-s42/predictions.jsonl")
for line in path.read_text(encoding="utf-8").splitlines():
    prediction = json.loads(line)
    if prediction["model_type"] == "jev":
        metadata = prediction["metadata"]
        print("Model:", metadata.get("resolved_model"))
        print("Cache hit:", metadata.get("cache_hit"))
        print("Input/output tokens:", metadata.get("input_tokens"), metadata.get("output_tokens"))
        print("Decision latency (ms):", prediction["decision_latency_ms"])
        print("Metrics:", prediction["metrics"])
PY
```

`cache_hit: true` means the exact response was reused. Keep cached and live
latency separate; cached response token counts do not represent new API usage.
The analysis `decision_accounting.csv` separates those fields. Enriching an
older schedule-only input changes its request, so an old pilot cache entry
will not automatically cover the enriched case.

A Top-1 or Top-5 Score of 1 on one workload only describes that sampled case.
Use the per-workload rankings and Pairwise Accuracy as well; this pilot does
not establish that Jev is competitive across TenSet.

## 9. Repeat without overwriting results

The runner and analyzer reject nonempty output directories. Keep previous
results and create a new experiment YAML with a distinct name and destination:

```bash
uv run python - <<'PY'
from pathlib import Path
import yaml

source = Path("experiments/tenset/jev-vs-random.yaml")
output = Path("experiments/tenset/local/repeat-c8-s42.yaml")
config = yaml.safe_load(source.read_text(encoding="utf-8"))
config["experiment"]["name"] += "-repeat-01"
config["experiment"]["output_dir"] += "-repeat-01"
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("x", encoding="utf-8") as stream:
    yaml.safe_dump(config, stream, sort_keys=False)
print(output)
PY

uv run python scripts/run_experiment.py \
  --config experiments/tenset/local/repeat-c8-s42.yaml

uv run python scripts/analyze_tenset.py \
  --results results/tenset/random-vs-jev-c8-s42-repeat-01 \
  --dataset data/tenset/prepared-metadata.json \
  --output-dir results/tenset/random-vs-jev-c8-s42-repeat-01-analysis
```

The same candidates and model reuse the existing cache. A new output directory
does not create a fresh live request. If you deliberately need another live
latency measurement, use a new `models[].config.cache_dir` for Jev in the new
YAML and account for the resulting API usage. Preserve the previous cache.
For another repetition, change the suffix and copied YAML filename again.

## 10. Expand to ten workloads, then design the reported experiment

Changing `max_workloads` to 10 does not create more workloads in a four-file
artifact. First prepare a larger source pool, export metadata for that pool,
and enrich it into new files. This example selects up to 16 source files with
the existing seeded preprocessing CLI and preserves the pilot artifacts.
Preparation retains all valid schedules from these source files in memory;
`max_workloads` limits evaluation afterward and does not bound preparation
memory. Start with 16 files on your MacBook Air. If preflight finds fewer than
ten eligible workloads, increase the file budget to 32, then 64 if needed,
using fresh filenames for each larger pool.

```bash
uv run python - <<'PY'
import json
import subprocess
import sys
from pathlib import Path

target = json.loads(Path("data/tenset/prepared-pilot.json").read_text(encoding="utf-8"))["provenance"]["target"]
subprocess.run([
    sys.executable, "scripts/prepare_tenset.py",
    "--raw-root", "data/tenset/raw", "--platform", "e5-2673",
    "--target", target, "--seed", "42", "--min-candidates", "8",
    "--file-limit", "16", "--output", "data/tenset/prepared-pool16.json",
], check=True)
PY

uv run python scripts/export_tenset_metadata.py \
  --backend portable \
  --task-registry data/tenset/raw/network_info/all_tasks.pkl \
  --prepared data/tenset/prepared-pool16.json \
  --output data/tenset/metadata-pool16.json

uv run python scripts/enrich_tenset.py \
  --prepared data/tenset/prepared-pool16.json \
  --metadata data/tenset/metadata-pool16.json \
  --output data/tenset/prepared-pool16-metadata.json
```

Create and preflight a ten-workload profile before spending API calls:

```bash
uv run python - <<'PY'
from pathlib import Path
import yaml
from jev_compiler_eval.datasets.tenset import TenSetDataset

config = yaml.safe_load(Path("experiments/tenset/jev-vs-random.yaml").read_text(encoding="utf-8"))
config["experiment"].update({
    "name": "tenset-e5-2673-random-vs-jev-c8-s42-w10",
    "description": "Ten-workload pilot with full tensor-program metadata",
    "output_dir": "results/tenset/random-vs-jev-c8-s42-w10",
})
config["dataset"]["config"].update({
    "path": "data/tenset/prepared-pool16-metadata.json",
    "max_workloads": 10,
})
cases = TenSetDataset(config["dataset"]["config"]).load()
if len(cases) != 10:
    raise SystemExit(f"Only {len(cases)} eligible workloads; prepare a larger source pool first")
output = Path("experiments/tenset/local/jev-vs-random-c8-s42-w10.yaml")
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("x", encoding="utf-8") as stream:
    yaml.safe_dump(config, stream, sort_keys=False)
print("Offline preflight passed for", len(cases), "workloads")
print(output)
PY

uv run python scripts/run_experiment.py \
  --config experiments/tenset/local/jev-vs-random-c8-s42-w10.yaml

uv run python scripts/analyze_tenset.py \
  --results results/tenset/random-vs-jev-c8-s42-w10 \
  --dataset data/tenset/prepared-pool16-metadata.json \
  --output-dir results/tenset/random-vs-jev-c8-s42-w10-analysis
```

This stage can make up to ten live decisions, depending on cache hits. Keep it
as a pilot: `split: all` intentionally permits a small pool without requiring a
nonempty test split. For reported RQ1 experiments, choose and record a larger
source budget, switch to `split: test`, check the actual eligible workload count,
and commit each experiment YAML before running it. Use a new output directory
for every candidate-count/seed setting. Keep candidate sets identical between
Random and Jev, and check metadata coverage and serialization sizes first.

To compare 8, 16, and 32 candidates on the same workloads, prepare with
`--min-candidates 32`, then use the same prepared artifact and seed for each
candidate-count profile. Report workload/operation coverage and seed variation,
not only an aggregate score. XGBoost numbers from the TenSet paper can be cited
as external context; they are not a matched baseline on your sampled workloads.

See [the TenSet method and metric reference](tenset.md),
[the RQ1 report section](rq1-report.md), and
[the shared experiment protocol](experiment-protocol.md) before writing results.

## Troubleshoot `max_tokens_exceeded`

A 400 response with `error_type: max_tokens_exceeded` means the combined
workload state and candidate question exceeded the service context limit.
Increasing retries, timeouts, or the local byte cap does not fix that limit.

To retain a 50-candidate experiment, use Jev's documented long-context option:

```yaml
models:
  - id: jev-1.14.0-long-context
    type: jev
    config:
      model: jev-1.14.0
      cache_dir: .cache/jev-long-context
      max_retries: 0
      request_timeout_seconds: 120.0
      long_context: true
```

Keep the existing Random entry, dataset path, split, seed, and candidate count.
Use a new experiment name and result directory for this mode. Run a small
workload check before the full comparison. Long-context responses have distinct
cache keys even if both modes share a cache directory. The runner records the
flag in prediction metadata and saves the model configuration with results.

The [official documentation](https://docs.typesafe.ai/models#long-context)
raises both context limits to 262k tokens with this option, and describes slower
and less accurate behavior at longer contexts. Report the mode and measured
usage explicitly. Successful responses before a failed run may remain cached;
the runner writes a complete result bundle only after all model decisions
succeed. Keep existing caches and prepared data.

If the frozen `data/tenset/prepared-w120-c50-s42-metadata.json` artifact already
exists from the 120-workload setup, the committed profiles preserve that cohort
and candidate sampling. Run the ten-workload check first:

```bash
uv run python scripts/run_experiment.py \
  --config experiments/tenset/jev-vs-random-w10-c50-s42-long-context.yaml
```

After it succeeds, run the full comparison:

```bash
uv run python scripts/run_experiment.py \
  --config experiments/tenset/jev-vs-random-w120-c50-s42-long-context.yaml

uv run python scripts/analyze_tenset.py \
  --results results/tenset/random-vs-jev-w120-c50-s42-long-context \
  --dataset data/tenset/prepared-w120-c50-s42-metadata.json \
  --output-dir results/tenset/random-vs-jev-w120-c50-s42-long-context-analysis
```

Both profiles use the same cache, zero automatic retries, a 120-second request
timeout, and new result directories. Successful long-context responses from
the smaller check can be reused by the full run. These profiles reference
existing prepared data; they do not download or prepare a new cohort.
