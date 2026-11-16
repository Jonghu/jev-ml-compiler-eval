"""Preflight or evaluate the reconstructed TenSet tutorial test-record pools."""

import argparse
import json
from pathlib import Path

from jev_compiler_eval.config.environment import load_cli_environment
from jev_compiler_eval.evaluation.tenset_tutorial import (
    SUMMARY_CSV,
    SUMMARY_JSON,
    preflight_tenset_tutorial,
    run_tenset_tutorial,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--workload-limit", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", "--dry-run", action="store_true")
    parser.add_argument(
        "--allow-external-reference",
        action="store_true",
        help="Label published XGBoost aggregates as external reference only",
    )
    args = parser.parse_args()
    if args.preflight_only:
        result = preflight_tenset_tutorial(
            args.config, workload_limit=args.workload_limit, output_dir=args.output_dir
        )
        print(json.dumps(result, indent=2))
        print("Offline validation only; no API client or environment file was loaded.")
        return
    if not args.allow_external_reference:
        parser.error(
            "--allow-external-reference is required: original test identities and "
            "XGBoost artifacts are unverified; this is not a matched XGBoost comparison"
        )
    # Check the complete selected scope before loading credentials or SDK clients.
    preflight_tenset_tutorial(
        args.config, workload_limit=args.workload_limit, output_dir=args.output_dir
    )
    load_cli_environment(PROJECT_ROOT / ".env")

    def report(done: int, total: int, workload: str, reused: bool) -> None:
        status = "reused completed results" if reused else "completed"
        print(f"[{done}/{total}] {workload}: {status}", flush=True)

    print("Published XGBoost is external reference only; original pool is unverified.")
    result = run_tenset_tutorial(
        args.config,
        allow_external_reference=True,
        workload_limit=args.workload_limit,
        resume=args.resume,
        output_dir=args.output_dir,
        progress=report,
    )
    print(f"Experiment: {result.experiment_name}")
    summary = json.loads((result.output_dir / SUMMARY_JSON).read_text())
    for model in summary["models"]:
        print(f"Model: {model['model_id']}; workloads: {model['workloads']}")
        print(f"Test records: {model['test_records']}")
        for name, value in model["record_weighted_metrics"].items():
            print(f"Record-weighted {name}: {value:.6f}")
    print(
        f"Record-weighted reference-formula scores: {result.output_dir / SUMMARY_CSV}"
    )
    print(f"Results: {result.output_dir}")


if __name__ == "__main__":
    main()
