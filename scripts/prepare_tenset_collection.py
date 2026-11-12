"""Prepare all distinct valid schedules in a resumable TenSet workload collection."""

import argparse
from pathlib import Path

from jev_compiler_eval.datasets.tenset_collection import (
    load_collection_index,
    prepare_collection,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--platform", default="e5-2673")
    targets = parser.add_mutually_exclusive_group(required=True)
    targets.add_argument(
        "--target", help="Exact target string from measurement records"
    )
    targets.add_argument(
        "--target-from-prepared",
        type=Path,
        help="Use only the exact hardware target from an earlier artifact",
    )
    programs = parser.add_mutually_exclusive_group(required=True)
    programs.add_argument(
        "--metadata", type=Path, help="Label-free program metadata JSON"
    )
    programs.add_argument(
        "--task-registry", type=Path, help="Original all_tasks.pkl, decoded without TVM"
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", choices=("test", "validation"), default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-candidates", type=int, default=2)
    parser.add_argument(
        "--workload-limit",
        type=int,
        help="Optional source-workload budget selected independently of labels",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--skip-unsupported-programs",
        action="store_true",
        help="Explicitly exclude and report undecodable/missing program metadata",
    )
    args = parser.parse_args()
    index_path = prepare_collection(
        args.raw_root,
        output_dir=args.output_dir,
        platform=args.platform,
        target=args.target,
        target_from_prepared=args.target_from_prepared,
        split=args.split,
        seed=args.seed,
        min_candidates=args.min_candidates,
        metadata_path=args.metadata,
        task_registry_path=args.task_registry,
        workload_limit=args.workload_limit,
        resume=args.resume,
        skip_unsupported_programs=args.skip_unsupported_programs,
        progress_callback=lambda message: print(message, flush=True),
    )
    index = load_collection_index(index_path)
    print(f"Collection index: {index_path}")
    print(f"Eligible workloads: {len(index['shards'])}")
    print(f"Distinct candidate schedules: {index['provenance']['candidate_count']}")
    print(
        f"Excluded source workload groups: {index['provenance']['excluded_workloads']}"
    )


if __name__ == "__main__":
    main()
