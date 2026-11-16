"""Run one guarded TenSet size-sweep member or validate it without API calls."""

import argparse
import json
from pathlib import Path

from jev_compiler_eval.config.environment import load_cli_environment
from jev_compiler_eval.evaluation.tenset_size_sweep import (
    run_tenset_size_sweep_member,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("experiments/tenset/local/size-sweep/manifest.json"),
    )
    parser.add_argument("--candidate-count", required=True, type=int)
    parser.add_argument("--preflight-only", "--dry-run", action="store_true")
    args = parser.parse_args()
    result = run_tenset_size_sweep_member(
        args.manifest,
        args.candidate_count,
        preflight_only=args.preflight_only,
        load_environment=lambda: load_cli_environment(PROJECT_ROOT / ".env"),
    )
    if isinstance(result, dict):
        print(json.dumps(result, indent=2))
        print("No API client was constructed or request sent.")
        return
    print(f"Experiment: {result.experiment_name}")
    for model in result.summaries:
        print(f"Model: {model.model_id}; workloads: {model.workloads}")
        for name, value in model.metrics.items():
            print(f"{name}: {value:.6f}")
    print(f"Results: {result.output_dir}")


if __name__ == "__main__":
    main()
