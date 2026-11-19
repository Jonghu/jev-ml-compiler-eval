"""Prepare a fixed-cohort TenSet candidate-size comparison without API calls."""

import argparse
import json
from pathlib import Path

from jev_compiler_eval.evaluation.tenset_size_sweep import (
    AUDITED_PREPARED_SHA256,
    prepare_tenset_size_sweep,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepared",
        type=Path,
        default=Path("data/tenset/prepared-w120-c50-s42-metadata.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("experiments/tenset/local/size-sweep")
    )
    parser.add_argument(
        "--results-root", type=Path, default=Path("results/tenset/size-sweep")
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=[16, 32, 50])
    parser.add_argument(
        "--expected-prepared-sha256",
        default=AUDITED_PREPARED_SHA256,
        help=(
            "Explicit frozen input checksum; defaults to the audited 120-workload file"
        ),
    )
    parser.add_argument(
        "--baseline-results",
        type=Path,
        help="Existing 50-schedule bundle to validate and reuse without modifying it",
    )
    args = parser.parse_args()
    path = prepare_tenset_size_sweep(
        args.prepared,
        output_dir=args.output_dir,
        results_root=args.results_root,
        sizes=tuple(args.sizes),
        expected_prepared_sha256=args.expected_prepared_sha256,
        baseline_results=args.baseline_results,
    )
    plan = json.loads(path.read_text())
    print(f"Frozen sweep manifest: {path}")
    print("Offline preparation passed; no model client was created or request sent.")
    for budget in plan["budgets"]:
        print(f"\nCandidates per workload: {budget['candidates_per_workload']}")
        if budget["action"] == "reuse_verified_anchor":
            print(f"Reuse verified existing results: {budget['results_dir']}")
        else:
            print(budget["preflight_command"])
            print(budget["run_command"])
    print("\nAfter the new runs complete:")
    print(plan["analysis_command"])
    print(plan["plot_command"])
    print(plan["analysis_note"])


if __name__ == "__main__":
    main()
