# TenSet evaluation (RQ1)

This workstream extends the shared runner with `dataset.type: tenset`, the
`tenset_xgb` model, official TenSet metrics, and result analysis. The existing
Jev and Random models consume the same prepared cases. See
[RQ1 report section](rq1-report.md) for methodology, references, and result status.
For a complete first run on a Mac, follow the
[macOS pilot guide](run-tenset-on-macos.md).

## Environment and data

Install the locked Python 3.11 environment, including download and plotting tools:

```bash
uv sync --locked --extra tenset
```

Download the official CPU v3.3 archive and extract the Intel E5-2673 v4 collection:

```bash
uv run python scripts/download_tenset.py --output-dir data/tenset/raw
```

The archive is large. Check available storage first. The command downloads the
complete CPU archive, then extracts only `measure_records/e5-2673` and the task
registry. It does not download a small remote subset. Google Drive and its
download redirect hosts must be allowed by the execution environment. An
existing archive can instead be imported:

```bash
uv run python scripts/download_tenset.py \
  --archive /absolute/path/dataset_cpu_v3.3.zip \
  --output-dir data/tenset/raw
```

The extractor rejects unsafe paths, symlinks, collisions, and nonempty outputs.
It records archive and extracted-file hashes, the official URL, platform, and
CC BY 4.0 attribution in `download-manifest.json`. An optional `--sha256` verifies
a checksum obtained independently; the repository does not invent an official
checksum. Data and generated artifacts are excluded from Git.

## Official TVM/XGBoost runtime

This section is optional for Jev evaluation. The default metadata exporter now
runs in the main Python 3.11 environment without native TVM, Docker, or XGBoost.
The legacy runtime below is needed only to train/run the specialized baseline
or regenerate native reference fixtures.

The original TenSet TVM fork uses historical Python, NumPy, and XGBoost APIs.
Current `apache-tvm` wheels are not a compatible replacement. Keep the evaluator
on Python 3.11 and build a separate Python 3.8 environment:

```bash
uv venv --python 3.8 .venv-tenset
uv pip install --python .venv-tenset/bin/python -r requirements-tenset-tvm.txt
git clone https://github.com/tlc-pack/tenset.git .cache/tenset
git -C .cache/tenset checkout 35774ed373f9693cc6efcbe83f5703141e8ab0c8
git -C .cache/tenset submodule update --init --recursive
```

Build this checkout using its
[source installation guide](https://github.com/tlc-pack/tenset/blob/35774ed373f9693cc6efcbe83f5703141e8ab0c8/docs/install/from_source.rst).
It requires a C++ toolchain, CMake, and compatible LLVM configuration. The build
is a prerequisite, not an action performed by the evaluator. After building,
make the pinned TVM Python bindings available to the legacy interpreter:

```bash
export PYTHONPATH="$PWD/.cache/tenset/python${PYTHONPATH:+:$PYTHONPATH}"
.venv-tenset/bin/python -c 'import tvm, xgboost; print(tvm.__version__, xgboost.__version__)'
```

The worker uses official `XGBModel.update`, `save`, `load`, and `predict` with
per-store features and the pack-sum objective. It loads checkpoints before
prediction to disable TVM's random warmup behavior. The registry and checkpoint
are Python pickle files: use the official trusted registry and locally trained
checkpoints. A persistent worker loads them once and closes with the shared
model lifecycle. The first decision includes startup; worker initialization
and feature/prediction time are also reported separately.

## Export semantic metadata on macOS and prepare records

TenSet's workload keys are usually opaque hashes. The CPU archive already
contains their tensor programs in `network_info/all_tasks.pkl`. The portable
exporter reads stored TVM SaveJSON graphs into inert data containers and extracts
tensor shapes, operation names, loop extents, complete expression graphs, and
initial scheduling-stage order. No source pickle constructors or TVM code run.
Unsupported globals or graph schemas fail explicitly instead of silently
omitting program information.

```bash
uv run python scripts/export_tenset_metadata.py \
  --task-registry data/tenset/raw/network_info/all_tasks.pkl \
  --output data/tenset/metadata.json
```

This uses the main environment on Intel and Apple silicon Macs; no separate
dataset is downloaded. The output is keyed by the original workload key and has
a `metadata.json.manifest.json` sidecar with registry/output hashes and exporter
provenance. The reader supports pickle protocols 2–5 and known TVM object states.
Its default registry-size limit is 512 MiB, adjustable with `--max-registry-mib`.

If a prepared pilot already exists, export only its tensor programs and attach
the metadata without changing workloads, candidates, splits, or labels:

```bash
uv run python scripts/export_tenset_metadata.py \
  --task-registry data/tenset/raw/network_info/all_tasks.pkl \
  --prepared data/tenset/prepared-pilot.json \
  --output data/tenset/metadata.json

uv run python scripts/enrich_tenset.py \
  --prepared data/tenset/prepared-pilot.json \
  --metadata data/tenset/metadata.json \
  --output data/tenset/prepared-metadata.json
```

Enrichment requires complete metadata coverage, preserves the parent artifact,
and records parent-data and metadata hashes. It refuses output overwrites.
Expression-graph node IDs are separate from schedule-stage indices;
`compute_dag.stage_order` maps the initial stages, while later transformations
can change them. The representation preserves expressions instead of inferring
program descriptions from workload hashes.

For native reference comparison use `--backend tvm` together with
`--python-executable .venv-tenset/bin/python`. Existing commands that explicitly
supply `--python-executable` retain their native-backend behavior.

Inspect the exact target strings in the extracted raw records. Pass one exact
string to the preparation command; `llvm` alone is not interchangeable with a
target carrying model and CPU flags. For an initial bounded run:

```bash
uv run python scripts/prepare_tenset.py \
  --raw-root data/tenset/raw \
  --platform e5-2673 \
  --target 'EXACT_TARGET_FROM_THE_RECORDS' \
  --metadata data/tenset/metadata.json \
  --min-candidates 32 \
  --file-limit 256 \
  --seed 42 \
  --output data/tenset/prepared.json
```

`--file-limit` selects workload files by seeded filename hash before loading
them. This bounds the initial run without inspecting labels. Preparation streams
lines but retains selected schedules in memory, so apply a source-file budget
for large collections. `--workload-limit` limits the final artifact after
parsing and does not bound peak parsing memory. Increase budgets after inspecting
the split and operation coverage; the initial subset is not stratified.

Preparation averages each record's timing repetitions, rejects failed or
invalid measurements, deduplicates schedule states, and averages repeated
records. It preserves original hidden MeasureInputs for XGBoost while exposing
only allowlisted workload context and scheduling transformations to Jev.
No measured labels, statuses, timestamps, or record offsets are transmitted to
decision models. Metadata is optional for parser development; missing operation
metadata is explicitly `unknown`, and missing shapes are not guessed.
When preparing development fixtures, pass `--synthetic` so analysis and plots
identify them as software tests rather than benchmark evidence.

The prepared schema contains `schema_version`, `dataset`, `provenance`, and
`workloads`. Each workload stores its ID, split, context, and candidates. A
candidate's `configuration` is model-visible; `runtime_ms`, `record_input`, and
`record_version` remain outside it. Source and metadata hashes, exclusion counts,
runtime aggregation, split settings, and split counts are retained.

## Train and evaluate

The current primary experiment compares Random and Jev; published XGBoost
results are a separately cited literature reference. They are not paired
measurements on this repository's chosen split and candidate pools.

After enriching a prepared pilot and setting `TYPESAFE_API_KEY` in `.env`:

```bash
uv run python scripts/run_experiment.py \
  --config experiments/tenset/jev-vs-random.yaml
```

This profile requires complete program metadata and uses one workload, eight
candidates, and zero retries. `split: all` makes it a setup pilot, not the final
held-out evaluation. For formal runs, prepare a sufficiently large subset,
change the profile to `split: test`, set agreed workload/candidate budgets, and
choose a new experiment name/output directory. No XGBoost checkpoint is loaded.

### Optional local specialized baseline

Train the official baseline on all and only the prepared training workloads:

```bash
uv run python scripts/train_tenset_xgb.py \
  --prepared data/tenset/prepared.json \
  --checkpoint data/tenset/xgb.pkl \
  --task-registry data/tenset/raw/network_info/all_tasks.pkl \
  --python-executable .venv-tenset/bin/python \
  --seed 42
```

Training writes a checkpoint and a `.json` provenance sidecar. Evaluation checks
the checkpoint, prepared-data, and registry hashes; training IDs must match the
prepared training split. Ranking a training workload or mismatched candidate
fails before inference. Validation/test labels never cross the prediction pipe.

Run Random without TVM or credentials:

```bash
uv run python scripts/run_experiment.py --config experiments/tenset/random.yaml
```

For the three-model comparison, configure `TYPESAFE_API_KEY` through the existing
`.env` mechanism and verify the legacy interpreter path in the YAML:

```bash
uv run python scripts/run_experiment.py --config experiments/tenset/comparison.yaml
```

The dataset adapter samples candidates once before the shared runner executes
models, so `candidate_selection.strategy` remains `all`. Its config supports:

| Setting | Default | Meaning |
| --- | --- | --- |
| `path` | Required | Prepared JSON artifact |
| `split` | `test` | `train`, `validation`, `test`, or `all` |
| `candidate_limit` | `32` | Seeded prefix size; `null` keeps all |
| `seed` | `42` | Candidate and workload ordering seed |
| `max_input_bytes` | `262144` | Fail if serialized Jev state/questions exceed local budget |
| `max_workloads` | `null` | Optional evaluation workload budget |
| `require_program_metadata` | `false` | Fail before model calls if program semantics are missing |

Cases with fewer schedules use all available candidates. Prepare with
`--min-candidates 32` for fixed-size 8/16/32 comparisons. The input byte check
includes the generic Jev state/question envelope but excludes SDK framing. It
does not establish service token or candidate-count limits. Adjust the budget
to the actual service contract before spending API credits; oversized cases fail
without silent truncation or runtime-based pruning.

For the planned matrix, copy the committed comparison YAML and change
`dataset.config.candidate_limit` to 8, 16, or 32 and the sampling/Random seeds to
42, 7, or 2026. Give every variant a distinct name/output directory and commit
its YAML before running. Keep the preparation split and XGBoost checkpoint fixed
while varying candidate sampling. Analyze seeds separately before reporting
their variation. Outputs are never overwritten.

## Tables, plots, and report

```bash
uv run python scripts/analyze_tenset.py \
  --results results/tenset/comparison-c32-s42 \
  --dataset data/tenset/prepared.json \
  --output-dir results/tenset/analysis-c32-s42
```

The command saves aggregate and per-workload CSVs, operation and candidate-count
breakdowns, paired Jev cases, decision accounting, a JSON summary, and PNG/SVG
comparison plots. `--no-plots` needs no Matplotlib. It labels synthetic fixture
inputs and does not convert them into benchmark results. Cached calls are
separated from billable usage. Prices are not assumed, so monetary cost remains
unknown without Person A's pricing snapshot.

The common runner additionally saves the full runtime map, model-visible input
fingerprint and byte count, package versions, and dataset provenance. Its
`model_input_bytes` counts the common model-input fingerprint envelope; the
TenSet provenance records actual Jev serialization sizes used by the byte guard.
Use those sizes rather than treating fingerprint bytes as API token usage.
Per-workload predictions retain Jev probabilities, scores, latency, usage,
resolved model version, and cache status through the existing shared metadata.

## Verification and limits of this branch

```bash
uv run pre-commit run --all-files
```

Tests cover raw-record handling, unit conversion and duplicate aggregation,
label isolation, disjoint splits, nested candidate sets, official metric parity,
artifact integrity, subprocess protocol/lifecycle, archive extraction, and
analysis outputs. Portable metadata parsing is checked against 13 tensor
programs serialized by the pinned native TenSet TVM build, including reductions,
fused convolution, branching, multiple outputs, scalar computes, and Unicode
names. The official archive and live Jev credentials are required for the final
integration and reported experiment; native TVM is optional for Jev runs.
Do not present synthetic smoke outputs as the RQ1 results.

The raw parser was also checked against the pinned upstream file
`tutorials/auto_scheduler/ci_logs/resnet-50-NHWC-B1-llvm.json`
(SHA-256 `a6263606470539b8692cb754eb008450d192a42324a2a0cd855c6fa5265eeb4f`).
All 26 records parsed. Each workload has only one schedule, so this confirms
format compatibility and supplies no ranking-performance evidence.
