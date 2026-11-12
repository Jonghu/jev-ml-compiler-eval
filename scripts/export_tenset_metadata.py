"""Export TenSet program metadata with a portable Python or native TVM reader."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from jev_compiler_eval.models.tenset_xgb import WORKER_PATH


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _portable_export(args: argparse.Namespace) -> dict[str, object]:
    from jev_compiler_eval.datasets.tenset import load_prepared
    from jev_compiler_eval.datasets.tenset_metadata import metadata_from_tensor_graph
    from jev_compiler_eval.datasets.tenset_registry import load_tenset_registry

    output = Path(args.output)
    manifest_path = Path(str(output) + ".manifest.json")
    if output.exists() or manifest_path.exists():
        raise FileExistsError("metadata and manifest destinations must not exist")
    wanted = None
    if args.prepared:
        prepared = load_prepared(args.prepared)
        wanted = {entry["context"]["workload_key"] for entry in prepared["workloads"]}
    records = load_tenset_registry(
        args.task_registry, max_bytes=args.max_registry_mib * 1024 * 1024
    )
    available = {entry["workload_key"] for entry in records}
    if wanted is not None and wanted - available:
        missing = len(wanted - available)
        raise ValueError(f"task registry is missing {missing} prepared workload key(s)")
    metadata = {}
    for entry in records:
        key = entry["workload_key"]
        if wanted is not None and key not in wanted:
            continue
        try:
            metadata[key] = metadata_from_tensor_graph(entry["tensor_graph"])
        except (ValueError, TypeError, KeyError) as error:
            raise ValueError(f"cannot decode tensor program {key}: {error}") from error
    if not metadata:
        raise ValueError("task registry contains no selected tensor programs")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    manifest = {
        "schema_version": 1,
        "backend": "portable",
        "exporter": "tenset-savejson-metadata-v1",
        "registry_sha256": _sha256(Path(args.task_registry)),
        "metadata_sha256": _sha256(output),
        "registry_workloads": len(records),
        "exported_workloads": len(metadata),
        "prepared_sha256": _sha256(Path(args.prepared)) if args.prepared else None,
        "representation": "structured TVM tensor expression graph",
        "source_format_reference": (
            "tlc-pack/tenset@35774ed373f9693cc6efcbe83f5703141e8ab0c8"
        ),
    }
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return {"output": str(output), "workloads": len(metadata)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-registry", required=True, help="Trusted all_tasks.pkl")
    parser.add_argument("--output", required=True, help="New metadata JSON destination")
    parser.add_argument(
        "--backend",
        choices=("portable", "tvm"),
        help="Default: portable; supplying --python-executable selects native TVM",
    )
    parser.add_argument("--python-executable", help="Interpreter for --backend tvm")
    parser.add_argument(
        "--prepared", help="Export only workload keys from this prepared artifact"
    )
    parser.add_argument(
        "--max-registry-mib",
        type=int,
        default=512,
        help="Maximum registry size for the portable reader (default: 512 MiB)",
    )
    args = parser.parse_args()
    if not Path(args.task_registry).is_file():
        parser.error("task registry input file does not exist")
    if Path(args.output).exists():
        parser.error("metadata destination must not exist")
    if args.max_registry_mib < 1:
        parser.error("--max-registry-mib must be positive")
    backend = args.backend or ("tvm" if args.python_executable else "portable")
    if backend == "portable":
        if args.python_executable:
            parser.error("--python-executable requires --backend tvm")
        result = _portable_export(args)
        print(f"Metadata: {result['output']}")
        print(f"Workloads: {result['workloads']}")
        print("Backend: portable Python; no TVM or XGBoost required")
        return
    if args.prepared:
        parser.error("--prepared selection is supported by --backend portable")
    completed = subprocess.run(
        [
            args.python_executable or sys.executable,
            str(WORKER_PATH),
            "metadata",
            "--task-registry",
            args.task_registry,
            "--output",
            args.output,
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
    )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise SystemExit("TenSet metadata worker returned invalid JSON") from error
    if completed.returncode or "error" in result:
        raise SystemExit(result.get("error", "TenSet metadata export failed"))
    print(f"Metadata: {result['output']}")
    print(f"Workloads: {result['workloads']}")


if __name__ == "__main__":
    main()
