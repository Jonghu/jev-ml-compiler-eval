"""Run a complete TenSet collection, retaining one workload in memory at a time."""

import argparse
import json
from pathlib import Path

from jev_compiler_eval.config.environment import load_cli_environment
from jev_compiler_eval.evaluation.tenset_collection import (
    preflight_as_dict,
    preflight_tenset_collection,
    run_tenset_collection,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--workload-limit", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", "--dry-run", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        result = preflight_tenset_collection(
            args.config, workload_limit=args.workload_limit, output_dir=args.output_dir
        )
        print(json.dumps(preflight_as_dict(result), indent=2))
        print("Offline validation only; no API client was created.")
        return
    load_cli_environment(PROJECT_ROOT / ".env")

    def report(done: int, total: int, workload: str, reused: bool) -> None:
        status = "reused completed results" if reused else "completed"
        print(f"[{done}/{total}] {workload}: {status}", flush=True)

    result = run_tenset_collection(
        args.config,
        workload_limit=args.workload_limit,
        resume=args.resume,
        output_dir=args.output_dir,
        progress=report,
    )
    print(f"Experiment: {result.experiment_name}")
    for summary in result.summaries:
        print(f"Model: {summary.model_id}; workloads: {summary.workloads}")
        for name, value in summary.metrics.items():
            print(f"{name}: {value:.6f}")
    print(f"Results: {result.output_dir}")


if __name__ == "__main__":
    main()
