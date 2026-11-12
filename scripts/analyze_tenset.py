"""Generate TenSet comparison tables and plots from completed result bundles."""

import argparse
from pathlib import Path

from jev_compiler_eval.evaluation.tenset_analysis import analyze_tenset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    result = analyze_tenset(
        args.results, args.dataset, args.output_dir, create_plots=not args.no_plots
    )
    print(f"TenSet analysis saved to {result}")


if __name__ == "__main__":
    main()
