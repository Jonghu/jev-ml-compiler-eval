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
The Jev-only TenSet pilot additionally sets `require_program_metadata: true`
so missing program semantics fail before any API request.

### `models`

Each entry has a unique experiment `id`, a registered implementation `type`,
and implementation-specific `config`. Multiple entries may use the same type
with different configurations.

Random accepts only `seed`. Jev requires `model`, `cache_dir`, `max_retries`,
and `request_timeout_seconds`. It also accepts the optional boolean
`long_context` (default `false`). When enabled, the client sends
`TypeSafe-Danger-Zone-Long-Context: true`, separates its cache from default-mode
requests, and records `long_context: true` in prediction metadata.

[TypeSafe's model documentation](https://docs.typesafe.ai/models#long-context)
lists default limits of 64k tokens per request and 32k per question, including
the shared state. The long-context header raises both limits to 262k tokens.
TypeSafe documents slower and less accurate behavior at longer contexts, with
the same price per input token. Record the mode as an experiment condition.
The dataset's `max_input_bytes` is a local byte budget, not a token count.

Jev also accepts `include_context` (boolean, default `true`) and
`candidate_order` (`original` or `shuffled`, default `original`). A shuffled
order requires an integer `order_seed`; an original order rejects an unused
order seed. The shuffle is deterministic per workload. Cache schema v3 hashes
the exact request order and retains the sent body; old cache files are not
automatically migrated. Use a new cache directory for these controls.

`type: jev-tournament` groups candidates into token-budgeted Choice questions,
advances each group's top candidates, and repeats until one final question.
It accepts long context explicitly and records probabilities per question,
alongside request usage and cache references. Its eliminated-candidate tail is
a deterministic bookkeeping order rather than a globally scored ranking.
Use selection metrics instead of global Pairwise Accuracy for this model.
The committed [TenSet follow-up profiles](run-tenset-followup-experiments.md)
provide complete settings, offline preflight, and collection execution commands.

TenSet additionally accepts the boolean dataset setting
`describe_config_fields` (default `false`). It adds the pinned TVM syntax and
argument definitions to the workload context without changing IDs, sampling,
schedule configurations, or measured values. Full-candidate collections use
`scripts/run_tenset_collection.py` with a completed collection index instead of
loading all candidate pools through the ordinary runner at once.

Single-question Jev records shuffled input IDs as `presented_candidate_ids`.
Its probability ties preserve original dataset order. The tournament instead
uses each question's seeded presentation order for probability ties and records
`last_question_probabilities`; those values are not a global score map.

### `metrics`

The shared runner computes metrics per workload and averages them across
workloads. Tournament rankings include a deterministic eliminated-candidate
tail: selection metrics remain meaningful, but global ranking metrics do not.

| Name | Definition |
|---|---|
| `runtime_regret` | Selected runtime divided by optimal runtime, minus one. |
| `top_1_accuracy` | One when the selected candidate ties for the optimal runtime. |
| `top_1_score`, `top_5_score` | TenSet prefix-average performance ratios; higher is better. |
| `pairwise_accuracy` | TenSet's strict pairwise comparison convention. |
| `pairwise_accuracy_tie_aware` | Separate pairwise diagnostic with explicit tie handling. |
| `top_1_error`, `top_5_error`, `top_10_error` | Fastest runtime among the K highest-ranked candidates divided by optimal runtime, minus one. Zero when a workload has at most K candidates. |
| `kendall_tau` | Kendall's tau-b between ranking positions and runtimes; one means the ranking is ordered from fastest to slowest. |

`top_1_error` equals `runtime_regret`. The Top-K errors and Kendall's tau
follow the TpuGraphs definitions (Phothilimthana et al., arXiv:2308.13490).
`kendall_tau` raises an error for workloads with fewer than two candidates or
identical runtimes, where it is undefined. The runner rejects global Pairwise
Accuracy for tournaments; tournament Kendall's tau is at most a historical
manufactured-order diagnostic, not a learned full-ranking result.

The [ResNet-50 workflow](run-tenset-paper-resnet50.md) separately exports
`paper_network_best_of_1_score` and `paper_network_best_of_5_score`, using
task-frequency-weighted network latency sums and the best-of-K endpoint.
These are not the shared runner's macro/prefix-average scores. Only for the
same network aggregation is proxy regret `1 / network_score - 1`; do not apply
that transformation to a macro mean of per-workload scores.

### `execution`

Only `fail_fast: true` is currently supported. Continuing after individual
model failures remains future work.

## Credentials

The CLI reads `TYPESAFE_API_KEY` from the process environment or the repository
`.env` file. `.env` is excluded from Git. Never put API keys in experiment YAML
files.
