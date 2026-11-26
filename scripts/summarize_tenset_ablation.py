"""Summarize and optionally archive saved TenSet ablations without API calls."""

import argparse
import json
from pathlib import Path

from jev_compiler_eval.evaluation.tenset_ablation_analysis import (
    package_ablation,
    summarize_ablation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--phase", choices=("small", "full"))
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    result = summarize_ablation(
        args.root,
        output_dir=args.output_dir,
        phase=args.phase,
        create_plots=not args.no_plots,
        overwrite=args.overwrite,
        allow_incomplete=args.allow_incomplete,
    )
    print(json.dumps(result, indent=2))
    if args.archive is not None:
        print(f"Raw team-review archive: {package_ablation(args.root, args.archive)}")
    print("Offline analysis only; no model client or environment file was loaded.")


if __name__ == "__main__":
    main()
