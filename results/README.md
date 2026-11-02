# Generated results

Experiment result bundles are written below this directory and excluded from
Git by default.

```text
results/<dataset>/<experiment>/
├── experiment.yaml
├── manifest.json
├── predictions.jsonl
├── metrics.json
└── summary.csv
```

The runner refuses to overwrite a nonempty output directory. Use a distinct
`experiment.output_dir` for a new run.

Only reviewed aggregate outputs needed by the final report should be committed,
and only after an explicit team decision.
