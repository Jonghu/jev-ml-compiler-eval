# Experiment configuration

Every experiment is defined by a YAML file committed under `experiments/` and
run through the shared entry point:

```bash
uv run python scripts/run_experiment.py --config experiments/path/to/file.yaml
```

## Schema

The current schema version is `1`.

```yaml
schema_version: 1

experiment:
  name: mock-random-vs-jev
  description: Compare Random and Jev on the committed mock dataset
  seed: 42
  output_dir: results/mock/random-vs-jev

dataset:
  type: mock
  config:
    path: examples/mock_dataset.json

candidate_selection:
  strategy: all

models:
  - id: random
    type: random
    config:
      seed: 42

  - id: jev-1.14.0
    type: jev
    config:
      model: jev-1.14.0
      cache_dir: .cache/jev
      max_retries: 3
      request_timeout_seconds: 30.0

metrics:
  - runtime_regret
  - top_1_accuracy

execution:
  fail_fast: true
```

## Sections

### `experiment`

- `name`: unique human-readable experiment name.
- `description`: optional description.
- `seed`: shared experiment seed recorded in the manifest.
- `output_dir`: result bundle destination. A nonempty destination is never
  overwritten.

### `dataset`

`type` selects a registered `BaseDataset` implementation. `config` is passed to
that implementation without dataset-specific interpretation by the YAML
loader.

### `candidate_selection`

Only `strategy: all` is currently implemented. Candidate sampling strategies
for TenSet are applied once in its dataset adapter before the common runner
passes the resulting fixed cases to every model. See [TenSet](tenset.md) for
its candidate limits, seeds, input budgets, and workload budgets.

### `models`

Each entry has a unique experiment `id`, a registered implementation `type`,
and implementation-specific `config`. Multiple entries may use the same type
with different configurations.

Random accepts only `seed`. Jev requires `model`, `cache_dir`, `max_retries`,
and `request_timeout_seconds`.

### `metrics`

The shared metrics include `runtime_regret` and mock `top_1_accuracy`. TenSet adds
`top_1_score`, `top_5_score`, and official `pairwise_accuracy`, plus a separately
named `pairwise_accuracy_tie_aware` diagnostic. Its Top-5 score averages the best
normalized throughput reached at each prefix. See [RQ1](rq1-report.md) for exact
formulas and aggregation. TpuGraphs metrics remain to be implemented.

### `execution`

Only `fail_fast: true` is currently supported. Continuing after individual
model failures remains future work.

## Credentials

The CLI reads `TYPESAFE_API_KEY` from the process environment or the repository
`.env` file. `.env` is excluded from Git. Never put API keys in experiment YAML
files.
