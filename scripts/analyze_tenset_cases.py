"""Export offline TenSet workload-type breakdowns and concrete schedule cases."""

import argparse
from pathlib import Path

from jev_compiler_eval.evaluation.tenset_case_analysis import analyze_tenset_cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--examples-per-outcome", type=int, default=3)
    args = parser.parse_args()
    result = analyze_tenset_cases(
        args.results,
        args.dataset,
        args.output_dir,
        examples_per_outcome=args.examples_per_outcome,
    )
    print(f"TenSet case analysis saved to {result}")
    print("Offline analysis only; no API calls or model decisions.")


if __name__ == "__main__":
    main()
