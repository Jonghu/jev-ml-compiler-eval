"""Prepare label-separated TenSet cases from one hardware platform's records."""

import argparse
import hashlib
import json
from pathlib import Path

from jev_compiler_eval.datasets.tenset import prepare_records
from jev_compiler_eval.datasets.tenset_download import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--platform", default="e5-2673")
    parser.add_argument(
        "--target",
        required=True,
        help="Exact TVM target string appearing in the measurement records",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--metadata", type=Path, help="Label-free workload metadata JSON"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--min-candidates", type=int, default=2)
    parser.add_argument("--workload-limit", type=int)
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Mark development fixtures as synthetic, never benchmark measurements",
    )
    parser.add_argument(
        "--file-limit",
        type=int,
        help="Seeded source-file budget applied before reading measurement records",
    )
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"prepared output already exists: {args.output}")
    if args.file_limit is not None and args.file_limit < 1:
        parser.error("--file-limit must be positive")
    if not args.platform or Path(args.platform).name != args.platform:
        parser.error("--platform must be a single directory name")
    record_dirs = [
        directory
        for directory in args.raw_root.rglob(args.platform)
        if directory.is_dir() and directory.parent.name == "measure_records"
    ]
    if len(record_dirs) != 1:
        raise ValueError(
            f"expected one measure_records/{args.platform} directory beneath "
            f"{args.raw_root}; found {len(record_dirs)}"
        )
    record_dir = record_dirs[0]
    paths = sorted(set(record_dir.glob("*.json")) | set(record_dir.glob("*.jsonl")))
    if not paths:
        raise ValueError(f"no measurement JSON files found in {record_dir}")
    files_available = len(paths)
    if args.file_limit is not None:
        paths.sort(
            key=lambda path: hashlib.sha256(
                f"{args.seed}\0{path.name}".encode("utf-8")
            ).digest()
        )
        paths = paths[: args.file_limit]
    prepared = prepare_records(
        paths,
        target=args.target,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        min_candidates=args.min_candidates,
        workload_limit=args.workload_limit,
        metadata_path=args.metadata,
    )
    provenance = prepared["provenance"]
    provenance.update(
        {
            "platform": args.platform,
            "source_file_budget": args.file_limit,
            "source_files_available": files_available,
            "source_files_selected": len(paths),
            "dataset_license": "CC BY 4.0",
        }
    )
    if args.synthetic:
        provenance["synthetic"] = True
    download_manifest = args.raw_root / "download-manifest.json"
    if download_manifest.is_file():
        provenance["download_manifest"] = {
            "path": str(download_manifest.resolve()),
            "sha256": sha256_file(download_manifest),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(prepared, output, indent=2, allow_nan=False)
        output.write("\n")
    print(f"Prepared {len(prepared['workloads'])} workloads: {args.output}")
    print(f"Splits: {json.dumps(provenance['split_counts'], sort_keys=True)}")


if __name__ == "__main__":
    main()
