"""Run an experiment from a version-controlled YAML configuration."""

import argparse

from jev_compiler_eval.evaluation import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to experiment YAML")
    args = parser.parse_args()

    result = run_experiment(args.config)
    print(f"Experiment: {result.experiment_name}")
    print(f"Dataset: {result.dataset_type}")
    for summary in result.summaries:
        print(f"Model: {summary.model_id}")
        print(f"Workloads: {summary.workloads}")
        for name, value in summary.metrics.items():
            print(f"{name}: {value:.6f}")
        print(f"mean_decision_latency_ms: {summary.mean_decision_latency_ms:.6f}")
    print(f"Results: {result.output_dir}")


if __name__ == "__main__":
    main()
