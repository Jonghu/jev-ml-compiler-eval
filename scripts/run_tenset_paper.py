"""Preflight or explicitly execute the reconstructed ResNet-50 network protocol."""

import argparse
import json
from pathlib import Path

from jev_compiler_eval.config.environment import load_cli_environment
from jev_compiler_eval.evaluation.tenset_paper import (
    INPUT_PRICE_PER_MILLION,
    SUMMARY_JSON,
    preflight_tenset_paper,
    run_tenset_paper,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--execute", action="store_true", help="Enable paid API calls")
    choice.add_argument("--preflight-only", "--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--input-price-per-million",
        type=float,
        default=INPUT_PRICE_PER_MILLION,
        help="Assumed USD per million input tokens; a planning estimate only",
    )
    args = parser.parse_args()
    if args.resume and not args.execute:
        parser.error("--resume requires --execute")
    # Both default and explicit offline modes stop before .env/SDK initialization.
    preflight = preflight_tenset_paper(
        args.config,
        output_dir=args.output_dir,
        input_price_per_million=args.input_price_per_million,
    )
    print(json.dumps(preflight, indent=2))
    if not args.execute:
        print("Offline preflight only; no API client or environment file was loaded.")
        return
    load_cli_environment(PROJECT_ROOT / ".env")

    def report(done: int, total: int, workload: str, reused: bool) -> None:
        status = "reused completed results" if reused else "completed"
        print(f"[{done}/{total}] {workload}: {status}", flush=True)

    result = run_tenset_paper(
        args.config,
        execute=True,
        resume=args.resume,
        output_dir=args.output_dir,
        progress=report,
        expected_identity=preflight["execution_identity"],
    )
    print(f"Experiment: {result.experiment_name}")
    print(f"Endpoint network-score summary: {result.output_dir / SUMMARY_JSON}")
    print(f"Results: {result.output_dir}")


if __name__ == "__main__":
    main()
