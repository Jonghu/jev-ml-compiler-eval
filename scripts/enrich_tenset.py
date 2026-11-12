"""Attach exported semantic metadata to existing TenSet cases without resampling."""

import argparse
import json
from pathlib import Path

from jev_compiler_eval.datasets.tenset import (
    _file_sha256,
    _load_metadata,
    _reject_label_keys,
    load_prepared,
)

SEMANTIC_FIELDS = {"operation", "tensors", "compute_dag", "loop_structure"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"enriched output already exists: {args.output}")
    prepared = load_prepared(args.prepared)
    metadata = _load_metadata(args.metadata)
    provenance = prepared.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("prepared TenSet provenance must be a mapping")
    for workload in prepared["workloads"]:
        if not isinstance(workload, dict) or not isinstance(
            workload.get("context"), dict
        ):
            raise ValueError("prepared workloads must contain context mappings")
        context = workload["context"]
        key = context.get("workload_key")
        if not isinstance(key, str) or not key:
            raise ValueError("each prepared workload needs its original workload_key")
        if key not in metadata:
            raise ValueError(f"semantic metadata is missing workload_key: {key}")
        description = metadata[key]
        missing = SEMANTIC_FIELDS - description.keys()
        if missing:
            raise ValueError(
                "semantic metadata is incomplete for workload_key "
                f"{key}: missing {', '.join(sorted(missing))}"
            )
        context.update({field: description[field] for field in sorted(SEMANTIC_FIELDS)})
        _reject_label_keys(context)
    provenance.update(
        {
            "metadata_sha256": _file_sha256(args.metadata),
            "parent_prepared_sha256": _file_sha256(args.prepared),
            "metadata_enrichment": {
                "workloads": len(prepared["workloads"]),
                "fields": sorted(SEMANTIC_FIELDS),
                "preserved": "workload order, IDs, splits, candidates and runtimes",
            },
        }
    )
    # Serialize before exclusive creation so invalid values cannot leave a partial file.
    encoded = json.dumps(prepared, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        output.write(encoded)
    print(f"Enriched {len(prepared['workloads'])} workloads: {args.output}")
    print("Workload IDs, splits, candidate sets and measured runtimes are preserved.")


if __name__ == "__main__":
    main()
