"""Prepare tutorial-derived Platinum test records without TVM or API calls."""

import argparse
import json
from pathlib import Path

from jev_compiler_eval.datasets.tenset_tutorial import prepare_tutorial_collection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/tenset/tutorial-500-platinum")
    )
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    index = prepare_tutorial_collection(
        args.raw_root,
        args.output_dir,
        sample_size=args.sample_size,
        seed=args.seed,
        resume=args.resume,
        progress=lambda message: print(message, flush=True),
    )
    status = json.loads((index.parent / "reference-status.json").read_text())
    print(f"Collection: {index}")
    print(json.dumps(status, indent=2))
    print("Offline preparation only; no Jev API calls or XGBoost training.")
    print(
        "Published XGBoost scores are an external reference. The original tutorial "
        "test inventory and trained model have not been independently matched."
    )


if __name__ == "__main__":
    main()
