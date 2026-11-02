# Version-controlled experiments

Every experiment must be represented by a committed YAML file. Use the shared
runner rather than adding dataset-specific command-line scripts:

```bash
uv run python scripts/run_experiment.py --config experiments/path/to/file.yaml
```

Experiment files must pin model versions used for reported results, record all
seeds, select a unique output directory, and avoid embedding credentials.

The `mock/` configurations are reference implementations for new benchmark
workstreams.
