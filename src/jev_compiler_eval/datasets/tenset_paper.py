"""Offline, full-record ResNet-50 benchmark reconstruction from TenSet CPU data.

The network task inventory supplies the entire cohort and its official weights.
Record errors and duplicate schedules remain separate candidates. No API, TVM,
feature extraction, training, sampling or within-file train/test split is used.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Mapping
from math import isfinite
from pathlib import Path

from jev_compiler_eval.datasets.tenset import (
    _CONTEXT_FIELDS,
    TRANSFORMATIONS,
    _context,
    _digest,
    _reject_label_keys,
)
from jev_compiler_eval.datasets.tenset_collection import (
    _fingerprint,
    _program_metadata,
    _publish,
    _read_json,
    _sha256,
    _shard_path,
    _source_descriptor,
    load_collection_shard,
)
from jev_compiler_eval.datasets.tenset_download import extract_cpu_archive
from jev_compiler_eval.datasets.tenset_registry import load_tenset_task_info
from jev_compiler_eval.datasets.tenset_tutorial import (
    _directory,
    _float32,
)
from jev_compiler_eval.datasets.tenset_tutorial import (
    _record_pool as _tutorial_record_pool,
)
from jev_compiler_eval.datasets.tenset_tutorial_protocol import (
    TUTORIAL_COMMIT,
    _measurement_filename,
    clean_name,
    f32,
)

PROTOCOL = "tenset_paper_resnet50_full_record_pool"
ALIGNMENT = "unverified_original_paper_artifacts"
NETWORK_KEY = ("resnet_50", [(1, 3, 224, 224)])
NETWORK_KEY_JSON = ["resnet_50", [[1, 3, 224, 224]]]
PLATFORM = "platinum-8272"
TASK_INFO_FILENAME = clean_name((NETWORK_KEY, "llvm")) + ".task.pkl"
_SHA = re.compile(r"[0-9a-f]{64}")
_PREPROCESSING_FILES = (
    "tenset_paper.py",
    "tenset_tutorial.py",
    "tenset_tutorial_protocol.py",
    "tenset_registry.py",
    "tenset_metadata.py",
    "tenset_collection.py",
    "tenset_download.py",
    "tenset.py",
)


def _preprocessing_sources() -> dict[str, str]:
    return {
        name: _sha256(Path(__file__).with_name(name)) for name in _PREPROCESSING_FILES
    }


def _record_pool(path: Path) -> tuple[list, list[dict]]:
    return _tutorial_record_pool(path, allow_zero_throughput=True)


def _platinum_target(target: object) -> bool:
    if not isinstance(target, str):
        return False
    tokens = target.split()
    return (
        bool(tokens)
        and tokens[0] == "llvm"
        and {"-model=platinum-8272", "-mcpu=skylake-avx512"} <= set(tokens)
    )


def _raw_source(source: Path, destination: Path, extract_dir: Path | None) -> tuple:
    if source.is_dir():
        if extract_dir is not None:
            raise ValueError("extract_dir applies only to an archive source")
        root = source.resolve()
        identity = {"kind": "directory", "path": str(root)}
    elif source.is_file():
        archive_hash = _sha256(source)
        root = (
            extract_dir or destination.with_name(destination.name + "-raw")
        ).absolute()
        if root.is_symlink():
            raise ValueError("archive extraction directory must not be a symlink")
        manifest_path = root / "download-manifest.json"
        if root.exists() and any(root.iterdir()):
            if not manifest_path.is_file():
                raise FileExistsError("existing extraction has no download manifest")
            manifest = _read_json(manifest_path)
            if (
                manifest.get("archive_sha256") != archive_hash
                or manifest.get("platform") != PLATFORM
                or manifest.get("include_network_tasks") is not True
            ):
                raise ValueError(
                    "existing extraction differs from the requested CPU ZIP"
                )
        else:
            extracted = extract_cpu_archive(
                source,
                root,
                platform=PLATFORM,
                include_network_tasks=True,
            )
            if extracted["archive_sha256"] != archive_hash:
                raise ValueError("CPU archive changed during extraction")
        identity = {
            "kind": "archive",
            "path": str(source.resolve()),
            "archive_sha256": archive_hash,
        }
    else:
        raise FileNotFoundError(f"TenSet CPU directory or ZIP does not exist: {source}")
    manifest_path = root / "download-manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.is_file() else None
    if manifest is not None:
        identity["download_manifest_sha256"] = _sha256(manifest_path)
        identity["declared_archive_sha256"] = manifest.get("archive_sha256")
    return root, identity, manifest


def _check_extracted_hashes(manifest: dict | None, inputs: list[dict]) -> None:
    if manifest is None:
        return
    hashes = {entry["path"]: entry["sha256"] for entry in manifest["files"]}
    for entry in inputs:
        if hashes.get(entry["path"]) != entry["sha256"]:
            raise ValueError("extracted paper source differs from its archive manifest")


def _counts(candidates: list[dict]) -> dict[str, int]:
    return {
        "eligible_workloads": 1,
        "candidate_records": len(candidates),
        "source_records": len(candidates),
        "error_records": sum(row["source_error_code"] != 0 for row in candidates),
        "duplicate_configurations": len(candidates)
        - len({_digest(row["configuration"]) for row in candidates}),
    }


def _validate_document(document: dict, descriptor: Mapping, state: dict) -> None:
    provenance = document.get("provenance", {})
    workload = document["workloads"][0]
    context = workload.get("context", {})
    records = workload.get("candidates", [])
    minimum = descriptor.get("full_task_min_latency_seconds")
    if (
        provenance.get("experiment_protocol") != PROTOCOL
        or provenance.get("reference_alignment_status") != ALIGNMENT
        or provenance.get("preparation_fingerprint") != _fingerprint(state)
        or provenance.get("source") != descriptor.get("source")
        or provenance.get("task_weight") != descriptor.get("task_weight")
        or provenance.get("task_index") != descriptor.get("task_index")
        or provenance.get("full_task_min_latency_seconds") != minimum
        or context.get("target") != state["target"]
        or context.get("workload_key") != descriptor["workload_key"]
        or set(context) - _CONTEXT_FIELDS
        or len(records) != descriptor.get("candidate_count")
        or len(records) != descriptor["source"]["records"]
        or [row.get("candidate_id") for row in records]
        != descriptor.get("candidate_ids")
    ):
        raise ValueError("paper shard differs from its frozen full-task cohort")
    _reject_label_keys(context)
    _program_metadata(
        descriptor["workload_key"],
        {
            descriptor["workload_key"]: {
                key: context.get(key)
                for key in ("operation", "tensors", "compute_dag", "loop_structure")
            }
        },
        None,
    )
    if type(minimum) not in (int, float) or not isfinite(minimum) or minimum <= 0:
        raise ValueError("paper full-task minimum must be finite and positive")
    costs = []
    for ordinal, row in enumerate(records):
        raw = row.get("record_input")
        cost = row.get("reference_cost_seconds")
        if (
            type(row.get("source_record_index")) is not int
            or row["source_record_index"] != ordinal
            or type(row.get("source_error_code")) is not int
            or not isinstance(raw, list)
            or len(raw) != 2
            or not isinstance(raw[0], list)
            or len(raw[0]) < 2
            or raw[0][:2] != [descriptor["workload_key"], state["target"]]
            or not isinstance(raw[1], list)
            or len(raw[1]) != 2
            or not isinstance(raw[1][1], list)
            or type(cost) not in (int, float)
            or not isfinite(cost)
            or cost <= 0
        ):
            raise ValueError(
                "paper record identity, ordering or private label is invalid"
            )
        configuration = {"initial_state": raw[1][0], "transformations": []}
        for step in raw[1][1]:
            if (
                not isinstance(step, list)
                or not step
                or not isinstance(step[0], str)
                or step[0] not in TRANSFORMATIONS
            ):
                raise ValueError("paper schedule contains an invalid transformation")
            configuration["transformations"].append(
                {
                    "code": step[0],
                    "name": TRANSFORMATIONS.get(step[0], "unknown"),
                    "arguments": step[1:],
                }
            )
        if (
            row.get("configuration") != configuration
            or row.get("candidate_id") != "record-" + _digest([raw, ordinal])[:24]
            or _float32(cost) != cost
            or row.get("runtime_ms") != cost * 1000
            or type(row.get("official_throughput")) not in (int, float)
            or row["official_throughput"] != f32(minimum / cost)
        ):
            raise ValueError("paper record schedule or float32 labels disagree")
        _reject_label_keys(configuration)
        costs.append(cost)
    if (
        not costs
        or min(costs) != minimum
        or provenance.get("counts") != _counts(records)
    ):
        raise ValueError("paper full-task normalizer or record counts disagree")


def load_paper_index(path: str | Path) -> dict:
    """Validate the frozen complete cohort without loading all candidate pools."""
    path = Path(path)
    index = _read_json(path)
    state_path = path.parent / "preparation.json"
    state = _read_json(state_path)
    if state.get("preprocessing_sources") != _preprocessing_sources():
        raise ValueError(
            "paper preparation source code differs from installed preprocessing "
            "modules; prepare a fresh artifact with the current code"
        )
    provenance = index.get("provenance", {})
    tasks, shards = state.get("network_tasks"), index.get("shards")
    if (
        index.get("schema_version") != 1
        or index.get("dataset") != "tenset_collection"
        or index.get("status") != "complete"
        or provenance.get("experiment_protocol") != PROTOCOL
        or provenance.get("reference_alignment_status") != ALIGNMENT
        or provenance.get("original_paper_inventory_verified") is not False
        or provenance.get("network_key") != NETWORK_KEY_JSON
        or provenance.get("platform") != PLATFORM
        or not _platinum_target(provenance.get("target"))
        or provenance.get("target") != state.get("target")
        or provenance.get("preparation_sha256") != _sha256(state_path)
        or provenance.get("preparation_fingerprint") != _fingerprint(state)
        or state.get("experiment_protocol") != PROTOCOL
        or state.get("network_key") != NETWORK_KEY_JSON
        or not isinstance(tasks, list)
        or not tasks
        or provenance.get("network_tasks") != tasks
        or not isinstance(shards, list)
        or len(shards) != len(tasks)
        or len(state.get("sources", [])) != len(tasks)
    ):
        raise ValueError("invalid or incomplete ResNet-50 paper benchmark index")
    seen = set()
    for number, (task, shard) in enumerate(zip(tasks, shards, strict=True)):
        expected_id = (
            "tenset-paper-"
            + _digest([NETWORK_KEY_JSON, task["workload_key"], state["target"]])[:24]
        )
        if (
            not isinstance(shard, dict)
            or not isinstance(task, dict)
            or task.get("task_index") != number
            or shard.get("task_index") != number
            or task.get("workload_id") != expected_id
            or shard.get("workload_id") != expected_id
            or task.get("workload_key") in seen
            or shard.get("workload_key") != task.get("workload_key")
            or type(task.get("task_weight")) is not int
            or task["task_weight"] <= 0
            or shard.get("task_weight") != task["task_weight"]
            or shard.get("task_indices") != [number]
            or shard.get("split") != "test"
            or shard.get("source") != state["sources"][number]
            or type(shard.get("candidate_count")) is not int
            or shard["candidate_count"] < 2
            or shard["candidate_count"] != shard["source"]["records"]
            or not isinstance(shard.get("candidate_ids"), list)
            or len(shard["candidate_ids"]) != shard["candidate_count"]
            or any(not isinstance(value, str) for value in shard["candidate_ids"])
            or len(set(shard["candidate_ids"])) != shard["candidate_count"]
            or not isinstance(shard.get("sha256"), str)
            or _SHA.fullmatch(shard["sha256"]) is None
        ):
            raise ValueError(
                "paper task descriptor differs from the complete task inventory"
            )
        _shard_path(path, shard)
        seen.add(task["workload_key"])
    if provenance.get("observed_counts", {}).get("eligible_workloads") != len(tasks):
        raise ValueError("paper benchmark task counts disagree")
    return index


def load_paper_shard(index_path: str | Path, descriptor: Mapping[str, object]) -> dict:
    """Verify a complete, ordered task pool and its private float32 labels."""
    index_path = Path(index_path)
    index = load_paper_index(index_path)
    if descriptor not in index["shards"]:
        raise ValueError("paper descriptor is not in the frozen complete cohort")
    document = load_collection_shard(index_path, descriptor)
    _validate_document(
        document, descriptor, _read_json(index_path.parent / "preparation.json")
    )
    return document


def prepare_paper_collection(
    source: str | Path,
    output_dir: str | Path,
    *,
    extract_dir: str | Path | None = None,
    resume: bool = False,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Prepare every task and every record for ResNet-50 batch 1, 224x224.

    ``source`` is either an extracted CPU root or a local CPU ZIP. ZIP extraction
    reuses the existing safe selective platform extractor and a persistent raw
    directory, allowing shard-level preparation to resume without re-extraction.
    The published paper's exact original artifact identity remains unverified.
    """
    source, destination = Path(source).resolve(), Path(output_dir).absolute()
    if type(resume) is not bool:
        raise ValueError("resume must be a boolean")
    if destination.is_symlink():
        raise ValueError("paper output directory must not be a symlink")
    if not resume and destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            "paper output is nonempty; use --resume with identical inputs"
        )
    state_path = destination / "preparation.json"
    if resume and not state_path.is_file():
        raise FileNotFoundError("resume requires an existing paper preparation.json")
    root, input_source, archive_manifest = _raw_source(
        source, destination, Path(extract_dir) if extract_dir is not None else None
    )
    info_dir = _directory(root, "network_info")
    measurement_dir = _directory(root, PLATFORM, "measure_records")
    task_path = info_dir / TASK_INFO_FILENAME
    if not task_path.is_file():
        raise FileNotFoundError(
            f"ResNet-50 batch-1 task inventory is missing: {task_path}"
        )
    task_hash = _sha256(task_path)
    tasks = load_tenset_task_info(task_path)
    if _sha256(task_path) != task_hash:
        raise ValueError("network task inventory changed during loading")
    if progress:
        progress(f"Hashing all measurement records for {len(tasks)} network tasks")
    paths = [
        measurement_dir / _measurement_filename(task["workload_key"]) for task in tasks
    ]
    if len(set(paths)) != len(paths):
        raise ValueError("network tasks collide in measurement filenames")
    sources = [_source_descriptor(path, root) for path in paths]
    targets = {entry["target"] for entry in sources}
    if len(targets) != 1 or not _platinum_target(next(iter(targets))):
        raise ValueError(
            "all paper tasks require one exact Platinum-8272 AVX-512 target"
        )
    target = next(iter(targets))
    for task, descriptor in zip(tasks, sources, strict=True):
        if (
            descriptor["records"] < 1
            or descriptor["workload_key"] != task["workload_key"]
        ):
            raise ValueError("paper measurement source does not match its network task")
        if descriptor["records"] < 2:
            raise ValueError(
                "current paper runner adapter requires at least two records per "
                "network task; the full cohort cannot be dropped, padded or altered"
            )
    inventory = {"path": task_path.relative_to(root).as_posix(), "sha256": task_hash}
    _check_extracted_hashes(archive_manifest, [inventory, *sources])
    network_tasks = [
        {
            "task_index": number,
            "workload_key": task["workload_key"],
            "task_weight": task["task_weight"],
            "workload_id": "tenset-paper-"
            + _digest([NETWORK_KEY_JSON, task["workload_key"], target])[:24],
        }
        for number, task in enumerate(tasks)
    ]
    state = {
        "schema_version": 1,
        "experiment_protocol": PROTOCOL,
        "network_key": NETWORK_KEY_JSON,
        "platform": PLATFORM,
        "target": target,
        "network_tasks": network_tasks,
        "task_info": inventory,
        "input_source": input_source,
        "sources": sources,
        "candidate_selection": "all_source_records_in_source_order",
        "record_split": "none",
        "candidate_deduplication": False,
        "error_record_filtering": False,
        "preprocessing_sources": _preprocessing_sources(),
    }
    if resume and _read_json(state_path) != state:
        raise ValueError(
            "paper inputs, task weights, settings or preparation code changed on resume"
        )
    _publish(state_path, state)
    fingerprint = _fingerprint(state)
    shards, total = [], Counter()
    for number, (task, source_info, path, identity) in enumerate(
        zip(tasks, sources, paths, network_tasks, strict=True), 1
    ):
        identifier = identity["workload_id"]
        receipt_path = destination / "receipts" / f"{identifier}.json"
        base = {
            **identity,
            "source": source_info,
            "preparation_fingerprint": fingerprint,
        }
        if receipt_path.exists():
            receipt = _read_json(receipt_path)
            if (
                any(receipt.get(key) != value for key, value in base.items())
                or receipt.get("status") != "included"
            ):
                raise ValueError(
                    "paper receipt differs from the frozen complete cohort"
                )
            shard = receipt["shard"]
            document = load_collection_shard(destination / "index.json", shard)
            _validate_document(document, shard, state)
            if (
                any(shard.get(key) != value for key, value in identity.items())
                or shard.get("source") != source_info
            ):
                raise ValueError("paper receipt shard belongs to a different task")
            if receipt.get("counts") != document["provenance"]["counts"]:
                raise ValueError("paper receipt counts differ from verified shard")
            counts = document["provenance"]["counts"]
        else:
            raw_task, records = _record_pool(path)
            if (
                raw_task[:2] != [task["workload_key"], target]
                or len(records) != source_info["records"]
                or _sha256(path) != source_info["sha256"]
            ):
                raise ValueError("paper measurement source changed during preparation")
            context = _context(raw_task)
            context.update(
                _program_metadata(
                    task["workload_key"],
                    None,
                    {task["workload_key"]: task["tensor_graph"]},
                )
            )
            _reject_label_keys(context)
            counts = _counts(records)
            minimum = min(record["reference_cost_seconds"] for record in records)
            document = {
                "schema_version": 1,
                "dataset": "tenset",
                "provenance": {
                    **base,
                    "experiment_protocol": PROTOCOL,
                    "reference_alignment_status": ALIGNMENT,
                    "full_task_min_latency_seconds": minimum,
                    "counts": counts,
                    "runtime_aggregation": (
                        "float32(mean(record costs)) seconds to ms; no deduplication"
                    ),
                    "official_throughput": (
                        "float32(full-task min float32 cost / record float32 cost)"
                    ),
                    "record_split": "none_full_measurement_file",
                },
                "workloads": [
                    {
                        "workload_id": identifier,
                        "split": "test",
                        "context": context,
                        "candidates": records,
                    }
                ],
            }
            shard_path = destination / "workloads" / f"{identifier}.json"
            shard = {
                **identity,
                "task_indices": [identity["task_index"]],
                "path": shard_path.relative_to(destination).as_posix(),
                "split": "test",
                "source": source_info,
                "candidate_count": len(records),
                "candidate_ids": [record["candidate_id"] for record in records],
                "full_task_min_latency_seconds": minimum,
            }
            _validate_document(document, shard, state)
            _publish(shard_path, document)
            shard["sha256"] = _sha256(shard_path)
            _publish(
                receipt_path,
                {**base, "status": "included", "shard": shard, "counts": counts},
            )
        shards.append(shard)
        total.update(counts)
        del document
        if "records" in locals():
            del records
        if progress:
            progress(f"Prepared {number}/{len(tasks)} complete network task pools")
    status = {
        "experiment_protocol": PROTOCOL,
        "reference_alignment_status": ALIGNMENT,
        "original_paper_inventory_verified": False,
        "matching_paper_model_predictions_available": False,
        "diagnostic_paper_resnet50_tasks": 27,
        "observed_network_tasks": len(tasks),
        "diagnostic_task_count_matches": len(tasks) == 27,
        "counts_note": (
            "Task counts alone do not prove identity with original paper artifacts."
        ),
    }
    _publish(destination / "reference-status.json", status)
    index = {
        "schema_version": 1,
        "dataset": "tenset_collection",
        "status": "complete",
        "provenance": {
            **status,
            "network_key": NETWORK_KEY_JSON,
            "network_tasks": network_tasks,
            "platform": PLATFORM,
            "target": target,
            "source_commit": TUTORIAL_COMMIT,
            "preparation_sha256": _sha256(state_path),
            "preparation_fingerprint": fingerprint,
            "candidate_selection": "all_source_records_in_source_order",
            "record_split": "none",
            "candidate_deduplication": False,
            "error_record_filtering": False,
            "observed_counts": dict(total),
            "candidate_count": total["candidate_records"],
            "dataset_license": "CC BY 4.0",
        },
        "shards": shards,
    }
    index_path = destination / "index.json"
    _publish(index_path, index)
    load_paper_index(index_path)
    return index_path
