"""Estimate or explicitly execute the frozen TenSet context ablation."""

import argparse
import json
from pathlib import Path

from jev_compiler_eval.config.environment import load_cli_environment
from jev_compiler_eval.evaluation.tenset_ablation import (
    preflight_ablation,
    run_ablation,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study",
        type=Path,
        default=Path("results/tenset/context-ablation/study.json"),
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--execute", action="store_true", help="Enable paid API calls")
    modes.add_argument("--preflight-only", "--dry-run", action="store_true")
    parser.add_argument("--phase", choices=("small", "full", "all"), default="all")
    parser.add_argument("--variants", nargs="+")
    parser.add_argument("--repeats", type=int, nargs="+")
    parser.add_argument("--max-workloads", type=int)
    parser.add_argument("--calibrate-from-pilot", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--budget-usd", type=float)
    args = parser.parse_args()
    if args.resume and not args.execute:
        parser.error("--resume requires --execute")
    if args.execute and args.budget_usd is None:
        parser.error("--execute requires --budget-usd (input-only planning allowance)")
    selection = {
        "phase": args.phase,
        "variants": args.variants,
        "repeats": args.repeats,
        "max_workloads": args.max_workloads,
    }
    if not args.execute:
        print(
            json.dumps(
                preflight_ablation(
                    args.study,
                    **selection,
                    calibrate_from_pilot=args.calibrate_from_pilot,
                ),
                indent=2,
            )
        )
        print("Offline preflight only; no API client or environment file was loaded.")
        return

    def report(done, total, phase, variant, repeat, workload, reused):
        status = "verified existing result" if reused else "completed"
        print(
            f"[{done}/{total}] {phase}/{variant}/rep{repeat} {workload}: {status}",
            flush=True,
        )

    result = run_ablation(
        args.study,
        execute=True,
        **selection,
        resume=args.resume,
        budget_usd=args.budget_usd,
        load_environment=lambda: load_cli_environment(PROJECT_ROOT / ".env"),
        progress=report,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
