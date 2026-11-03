"""Download and extract checksum-verified TpuGraphs tile:xla data."""

import argparse
from pathlib import Path

from jev_compiler_eval.datasets.tpugraphs_download import (
    TILE_XLA_ARCHIVES,
    download_tile_archives,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=sorted(TILE_XLA_ARCHIVES),
        default=["train", "valid", "test"],
        help="Splits to download (default: all)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "tpugraphs",
        help="Destination directory (default: data/tpugraphs)",
    )
    args = parser.parse_args()

    download_tile_archives(args.data_dir, args.splits)


if __name__ == "__main__":
    main()
