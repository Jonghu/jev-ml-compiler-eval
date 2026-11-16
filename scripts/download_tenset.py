"""Download the official TenSet CPU v3.3 archive or extract a local copy."""

import argparse
from pathlib import Path

from jev_compiler_eval.datasets.tenset_download import (
    download_cpu_archive,
    extract_cpu_archive,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Existing official CPU ZIP")
    parser.add_argument(
        "--download-to", type=Path, default=Path("data/tenset/dataset_cpu_v3.3.zip")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/tenset/raw"),
        help="Empty destination for selected records and the workload registry",
    )
    parser.add_argument("--platform", default="e5-2673")
    parser.add_argument(
        "--include-network-tasks",
        action="store_true",
        help="Also extract network task inventories for the tutorial preset",
    )
    parser.add_argument("--sha256", help="Expected SHA-256 from a trusted source")
    args = parser.parse_args()
    archive = args.archive
    if archive is None:
        archive = download_cpu_archive(args.download_to)
    manifest = extract_cpu_archive(
        archive,
        args.output_dir,
        platform=args.platform,
        expected_sha256=args.sha256,
        include_network_tasks=args.include_network_tasks,
    )
    print(f"Extracted {len(manifest['files'])} files to {args.output_dir}")
    print(f"Archive SHA-256: {manifest['archive_sha256']}")
    print(f"Provenance: {args.output_dir / 'download-manifest.json'}")


if __name__ == "__main__":
    main()
