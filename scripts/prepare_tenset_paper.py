"""Prepare the complete ResNet-50 Platinum record pools, offline and without TVM."""

import argparse
import json
from pathlib import Path

from jev_compiler_eval.datasets.tenset_paper import prepare_paper_collection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--raw-root", type=Path, help="Extracted TenSet CPU root")
    source.add_argument("--archive", type=Path, help="Existing TenSet CPU ZIP")
    parser.add_argument(
        "--extract-dir", type=Path, help="Persistent raw destination for ZIP input"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/tenset/paper-resnet50-platinum")
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.extract_dir is not None and args.archive is None:
        parser.error("--extract-dir requires --archive")
    index = prepare_paper_collection(
        args.archive or args.raw_root,
        args.output_dir,
        extract_dir=args.extract_dir,
        resume=args.resume,
        progress=lambda message: print(message, flush=True),
    )
    print(f"Collection: {index}")
    print(
        json.dumps(
            json.loads((index.parent / "reference-status.json").read_text()), indent=2
        )
    )
    print("Offline preparation only; no API requests, native TVM or model training.")
    print(
        "Original paper inventory and matching MLP reference artifacts "
        "remain unverified."
    )


if __name__ == "__main__":
    main()
