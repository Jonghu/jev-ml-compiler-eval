"""Plot controlled TenSet candidate-size curves from completed offline analysis."""

import argparse
from pathlib import Path

from jev_compiler_eval.evaluation.tenset_size_sweep import plot_tenset_size_sweep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("experiments/tenset/local/size-sweep/manifest.json"),
    )
    parser.add_argument(
        "--analysis-dir", type=Path, default=Path("results/tenset/size-sweep/analysis")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/tenset/size-sweep/size-comparison"),
    )
    args = parser.parse_args()
    output = plot_tenset_size_sweep(args.manifest, args.analysis_dir, args.output_dir)
    print(f"Candidate-size curves: {output}")
    print("Offline plotting only; no decisions or API calls were made.")


if __name__ == "__main__":
    main()
