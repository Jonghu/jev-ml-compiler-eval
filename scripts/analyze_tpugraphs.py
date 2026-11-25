"""Explain which Jev picks drive the Top-1 error of a TpuGraphs tile run."""

import argparse
from pathlib import Path

from jev_compiler_eval.evaluation.tpugraphs_analysis import analyze_tpugraphs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="Result bundle that contains predictions.jsonl",
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--jev-model", required=True)
    parser.add_argument("--random-model", default="random")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Jev cache directory; enables tournament traces",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--worst", type=int, default=15)
    parser.add_argument("--traces", type=int, default=3)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    output_dir = analyze_tpugraphs(
        args.run_dir / "predictions.jsonl",
        args.dataset,
        args.output_dir or args.run_dir / "analysis",
        jev_model_id=args.jev_model,
        random_model_id=args.random_model,
        cache_dir=args.cache_dir,
        worst_count=args.worst,
        trace_count=args.traces,
        create_plots=not args.no_plots,
    )
    print(f"TpuGraphs analysis saved to {output_dir}")


if __name__ == "__main__":
    main()
