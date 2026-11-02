# Experiment protocol

This protocol applies to Jev and every baseline on both benchmark workstreams.

## Fair comparison

- Load each workload and candidate set once per experiment.
- Present the identical candidate IDs and configurations to every model.
- Apply candidate sampling before model execution, never inside one model.
- Record the experiment seed and all model-specific seeds.
- Keep measured runtimes separate from model-visible workload and candidate
  data.
- Fail if a model ranking omits, duplicates, or invents a candidate.

## Reproducibility

- Commit every experiment YAML file.
- Pin the Jev model version for reported experiments.
- Record the Git commit in the result manifest.
- Use the committed `uv.lock` environment.
- Do not overwrite nonempty result directories.
- Preserve per-workload predictions before computing aggregate metrics.

## Metrics

The mock pipeline currently provides runtime regret and Top-1 accuracy:

```text
runtime_regret = selected_runtime / optimal_runtime - 1
```

Any candidate tied for the minimum runtime is a correct Top-1 selection and has
zero regret.

Benchmark owners must implement and cite the official benchmark definitions:

- TenSet: Top-1 Score, Top-5 Score, Pairwise Accuracy.
- TpuGraphs: Top-1 Error, Top-5/10 Error, Kendall's Tau.

Do not reuse the mock Top-1 accuracy name for an official benchmark metric with
a different definition.

## Jev accounting

Record for every Jev decision:

- requested and resolved model versions;
- probability for every candidate and response confidence;
- input and output token usage;
- request ID when available;
- live or cached status;
- API and total decision latency.

Cached responses must not be counted as new API cost. API keys must never be
written to YAML, cache files, result files, logs, or commits.

## Result bundle

Each completed experiment writes:

```text
experiment.yaml
manifest.json
predictions.jsonl
metrics.json
summary.csv
```

Generated results are excluded from Git by default. Only reviewed aggregate
results intended for the final report should be added through an explicit team
decision.
