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
must be added before large benchmark runs.

### `models`

Each entry has a unique experiment `id`, a registered implementation `type`,
and implementation-specific `config`. Multiple entries may use the same type
with different configurations.

Random accepts only `seed`. Jev requires `model`, `cache_dir`, `max_retries`,
and `request_timeout_seconds`.

### `metrics`

Each metric is computed per workload and averaged across workloads. Every
model must rank all candidates, so ranking metrics see the full ranking.

| Name | Definition |
|---|---|
| `runtime_regret` | Selected runtime divided by optimal runtime, minus one. |
| `top_1_accuracy` | One when the selected candidate ties for the optimal runtime. |
| `top_1_error`, `top_5_error`, `top_10_error` | Fastest runtime among the K highest-ranked candidates divided by optimal runtime, minus one. Zero when a workload has at most K candidates. |
| `kendall_tau` | Kendall's tau-b between ranking positions and runtimes; one means the ranking is ordered from fastest to slowest. |

`top_1_error` equals `runtime_regret`. The Top-K errors and Kendall's tau
follow the TpuGraphs definitions (Phothilimthana et al., arXiv:2308.13490).
`kendall_tau` raises an error for workloads with fewer than two candidates or
identical runtimes, where it is undefined. TenSet Top-1 Score, Top-5 Score,
and Pairwise Accuracy are not yet implemented.

### `execution`

Only `fail_fast: true` is currently supported. Continuing after individual
model failures remains future work.

## Credentials

The CLI reads `TYPESAFE_API_KEY` from the process environment or the repository
`.env` file. `.env` is excluded from Git. Never put API keys in experiment YAML
files.
