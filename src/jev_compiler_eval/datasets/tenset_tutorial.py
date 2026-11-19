"""Portable reconstruction of the official 500-file tutorial's record test pools.

This intentionally does not call prepare_records: the reference keeps duplicate
and failed measurement records, and splits records before any candidate sampling.
Native TVM features and the original trained XGBoost model are not reconstructed.
"""

from __future__ import annotations

import json
import struct
from collections import Counter
from collections.abc import Callable
from math import isfinite
from pathlib import Path

from jev_compiler_eval.datasets.tenset import (
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
    _source_descriptor,
    load_collection_shard,
)
from jev_compiler_eval.datasets.tenset_registry import load_tenset_registry
from jev_compiler_eval.datasets.tenset_tutorial_protocol import (
    f32,
    tutorial_file_sample,
)

PROTOCOL = "tenset_500file_tutorial_record_test_reconstruction"
ALIGNMENT = "unverified_original_tutorial_artifacts"
PUBLISHED_COUNTS = {
    "sampled_files": 500,
    "eligible_workloads": 484,
    "training_records": 1689022,
    "test_records": 187679,
}


def _float32(value: float) -> float:
    try:
        result = struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, struct.error) as error:
        raise ValueError("timing cannot be represented as finite float32") from error
    if not isfinite(result) or result <= 0:
        raise ValueError("timing/throughput must be finite and positive in float32")
    return result


def tutorial_test_indices(count: int) -> list[int]:
    """Mirror reverse-order within_task(train_ratio=.9, shuffle_time=False)."""
    if type(count) is not int or count < 1:
        raise ValueError("record count must be a positive integer")
    return list(range(count - 1, -1, -1))[int(0.9 * count) :]


def _record_pool(
    path: Path, *, allow_zero_throughput: bool = False
) -> tuple[list, list[dict]]:
    if type(allow_zero_throughput) is not bool:
        raise ValueError("allow_zero_throughput must be a boolean")
    task = None
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            try:
                row = json.loads(line)
                measure_input = row["i"]
                current_task, state = measure_input
                costs = row["r"][0]
                error_code = row["r"][1]
                if (
                    not isinstance(current_task, list)
                    or len(current_task) < 2
                    or not all(isinstance(x, str) and x for x in current_task[:2])
                    or not isinstance(state, list)
                    or len(state) != 2
                    or not isinstance(state[1], list)
                    or not isinstance(costs, list)
                    or not costs
                    or type(error_code) is not int
                    or any(
                        type(value) not in (int, float)
                        or not isfinite(value)
                        or value <= 0
                        for value in costs
                    )
                ):
                    raise ValueError("unsupported record shape or timing values")
                if task is None:
                    task = current_task
                elif task != current_task:
                    raise ValueError("measurement file changes task metadata")
                configuration = {"initial_state": state[0], "transformations": []}
                for step in state[1]:
                    if (
                        not isinstance(step, list)
                        or not step
                        or not isinstance(step[0], str)
                    ):
                        raise ValueError("unsupported transformation record")
                    configuration["transformations"].append(
                        {
                            "code": step[0],
                            "name": TRANSFORMATIONS.get(step[0], "unknown"),
                            "arguments": step[1:],
                        }
                    )
                _reject_label_keys(configuration)
                cost = _float32(sum(costs) / len(costs))
                ordinal = len(records)
                records.append(
                    {
                        "candidate_id": "record-"
                        + _digest([measure_input, ordinal])[:24],
                        "configuration": configuration,
                        "runtime_ms": cost * 1000,
                        "reference_cost_seconds": cost,
                        "source_record_index": ordinal,
                        "source_error_code": error_code,
                        "record_input": measure_input,
                        "record_version": row.get("v"),
                    }
                )
            except (KeyError, IndexError, TypeError, ValueError) as error:
                raise ValueError(
                    f"cannot preserve tutorial record {path}:{line_number}: {error}"
                ) from error
    if task is None:
        raise ValueError(f"empty tutorial source file: {path}")
    minimum = min(record["reference_cost_seconds"] for record in records)
    for record in records:
        ratio = minimum / record["reference_cost_seconds"]
        if allow_zero_throughput:
            throughput = f32(ratio)
            if not isfinite(throughput) or not 0 <= throughput <= 1:
                raise ValueError("throughput must be finite and in [0, 1]")
            record["official_throughput"] = throughput
        else:
            record["official_throughput"] = _float32(ratio)
    return task, records


def _directory(root: Path, name: str, parent: str | None = None) -> Path:
    paths = [root] if root.name == name else list(root.rglob(name))
    matches = [
        p for p in paths if p.is_dir() and (parent is None or p.parent.name == parent)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one {name} directory below {root}; found {len(matches)}"
        )
    return matches[0]


def prepare_tutorial_collection(
    raw_root: str | Path,
    output_dir: str | Path,
    *,
    sample_size: int = 500,
    seed: int = 0,
    resume: bool = False,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Freeze tutorial-derived test records, preserving labels outside Jev inputs.

    Matching the published counts is diagnostic, not proof that the source pool
    or model equals the authors' original run. Preparation never calls Jev/TVM.
    """
    root, destination = Path(raw_root).resolve(), Path(output_dir).absolute()
    if type(resume) is not bool:
        raise ValueError("resume must be a boolean")
    if destination.is_symlink():
        raise ValueError("output directory must not be a symlink")
    if not resume and destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            "tutorial output is nonempty; use --resume for identical inputs"
        )
    state_path = destination / "preparation.json"
    if resume and not state_path.is_file():
        raise FileNotFoundError("resume requires an existing tutorial preparation.json")
    network_info = _directory(root, "network_info")
    measurement_dir = _directory(root, "platinum-8272", "measure_records")
    selected, selection = tutorial_file_sample(
        network_info, measurement_dir, sample_size=sample_size, seed=seed
    )
    expected_keys = selection.get("selected_workload_keys")
    if (
        not isinstance(expected_keys, list)
        or len(expected_keys) != len(selected)
        or any(not isinstance(key, str) or not key for key in expected_keys)
        or len(set(expected_keys)) != len(expected_keys)
    ):
        raise ValueError("tutorial selection must bind each file to a unique task key")
    if progress:
        progress(f"Hashing {len(selected)} sampled measurement files (offline)")
    sources = [_source_descriptor(path, root) for path in selected]
    targets = {source["target"] for source in sources if source["records"]}
    if len(targets) != 1:
        raise ValueError("tutorial sources must contain one exact CPU target")
    target_tokens = next(iter(targets)).split()
    if (
        target_tokens[0] != "llvm"
        or "-model=platinum-8272" not in target_tokens
        or "-mcpu=skylake-avx512" not in target_tokens
    ):
        raise ValueError("tutorial sources must use the Platinum-8272 AVX-512 target")
    registry_path = network_info / "all_tasks.pkl"
    state = {
        "schema_version": 1,
        "experiment_protocol": PROTOCOL,
        "selection": selection,
        "sources": sources,
        "registry_sha256": _sha256(registry_path),
        "preprocessing_sources": {
            name: _sha256(Path(__file__).with_name(name))
            for name in (
                "tenset_tutorial.py",
                "tenset_tutorial_protocol.py",
                "tenset_registry.py",
                "tenset_metadata.py",
                "tenset.py",
                "tenset_collection.py",
            )
        },
    }
    if resume and _read_json(state_path) != state:
        raise ValueError(
            "tutorial inputs, settings, or preparation code changed on resume"
        )
    _publish(state_path, state)
    fingerprint = _fingerprint(state)
    if progress:
        progress("Decoding program metadata without TVM")
    graphs = {
        entry["workload_key"]: entry["tensor_graph"]
        for entry in load_tenset_registry(registry_path)
    }
    if _sha256(registry_path) != state["registry_sha256"]:
        raise ValueError("task registry changed during preparation")
    shards, exclusions = [], []
    counts = Counter(sampled_files=len(selected))
    for number, (path, source, key) in enumerate(
        zip(selected, sources, expected_keys, strict=True), 1
    ):
        target = source["target"]
        if source["records"] and source["workload_key"] != key:
            raise ValueError(
                f"tutorial source header differs from sampled task key: {path}"
            )
        identifier = "tenset-tutorial-" + _digest([key, target])[:24]
        receipt_path = destination / "receipts" / f"{identifier}.json"
        base = {"workload_id": identifier, "source": source, "fingerprint": fingerprint}
        if receipt_path.exists():
            receipt = _read_json(receipt_path)
            if any(receipt.get(k) != v for k, v in base.items()):
                raise ValueError("tutorial receipt differs from frozen preparation")
            if receipt.get("status") == "included":
                shard = receipt["shard"]
                document = load_collection_shard(destination / "index.json", shard)
                provenance = document["provenance"]
                if (
                    provenance.get("preparation_fingerprint") != fingerprint
                    or provenance.get("source") != source
                    or provenance.get("counts") != receipt.get("counts")
                ):
                    raise ValueError(
                        "tutorial receipt differs from verified shard provenance"
                    )
                counts.update(receipt["counts"])
                shards.append(shard)
            elif (
                receipt.get("status") == "excluded"
                and source["records"] < 48
                and receipt.get("reason") == "fewer_than_48_records"
            ):
                exclusions.append(receipt)
                counts["excluded_records"] += source["records"]
            else:
                raise ValueError("invalid tutorial receipt status")
        elif source["records"] < 48:
            receipt = {**base, "status": "excluded", "reason": "fewer_than_48_records"}
            _publish(receipt_path, receipt)
            exclusions.append(receipt)
            counts["excluded_records"] += source["records"]
        else:
            task, records = _record_pool(path)
            if len(records) != source["records"] or task[:2] != [key, target]:
                raise ValueError("tutorial source identity or record count changed")
            if _sha256(path) != source["sha256"]:
                raise ValueError("tutorial source changed during preparation")
            indices = tutorial_test_indices(len(records))
            test_records = [records[index] for index in indices]
            context = _context(task)
            context.update(_program_metadata(key, None, graphs))
            _reject_label_keys(context)
            group_counts = {
                "eligible_workloads": 1,
                "training_records": len(records) - len(indices),
                "test_records": len(indices),
                "retained_records": len(records),
                "test_error_records": sum(
                    r["source_error_code"] != 0 for r in test_records
                ),
                "test_duplicate_configurations": len(test_records)
                - len({_digest(r["configuration"]) for r in test_records}),
            }
            provenance = {
                "experiment_protocol": PROTOCOL,
                "reference_alignment_status": ALIGNMENT,
                "preparation_fingerprint": fingerprint,
                "source": source,
                "counts": group_counts,
                "train_source_record_indices": list(
                    range(len(records) - 1, len(indices) - 1, -1)
                ),
                "test_source_record_indices": indices,
                "runtime_aggregation": (
                    "float32(mean(record costs)) seconds to ms; no deduplication"
                ),
                "official_throughput": (
                    "float32(full-task min float32 cost / record float32 cost)"
                ),
            }
            document = {
                "schema_version": 1,
                "dataset": "tenset",
                "provenance": provenance,
                "workloads": [
                    {
                        "workload_id": identifier,
                        "split": "test",
                        "context": context,
                        "candidates": test_records,
                    }
                ],
            }
            shard_path = destination / "workloads" / f"{identifier}.json"
            _publish(shard_path, document)
            shard = {
                "path": shard_path.relative_to(destination).as_posix(),
                "sha256": _sha256(shard_path),
                "workload_id": identifier,
                "workload_key": key,
                "split": "test",
                "candidate_count": len(indices),
            }
            _publish(
                receipt_path,
                {**base, "status": "included", "shard": shard, "counts": group_counts},
            )
            shards.append(shard)
            counts.update(group_counts)
            del records, test_records, document
        if progress:
            progress(
                f"Prepared {number}/{len(selected)} files; "
                f"{len(shards)} retained workloads"
            )
    if not shards:
        raise ValueError("no tutorial workloads meet the 48-record minimum")
    status = {
        "experiment_protocol": PROTOCOL,
        "reference_alignment_status": ALIGNMENT,
        "original_test_inventory_verified": False,
        "matching_xgboost_predictions_available": False,
        "published_tutorial_counts": PUBLISHED_COUNTS,
        "observed_counts": dict(counts),
        "published_counts_match": all(
            counts[k] == v for k, v in PUBLISHED_COUNTS.items()
        ),
        "counts_note": (
            "Count agreement alone does not establish identical original "
            "test records or a matching model."
        ),
        "native_tvm_features_reproduced": False,
    }
    _publish(destination / "reference-status.json", status)
    _publish(destination / "exclusions.json", {"excluded": exclusions})
    index = {
        "schema_version": 1,
        "dataset": "tenset_collection",
        "status": "complete",
        "provenance": {
            **status,
            "platform": "platinum-8272",
            "target": next(iter(targets)),
            "seed": seed,
            "candidate_selection": (
                "all_tutorial_test_records_preserving_duplicates_and_errors"
            ),
            "candidate_count": counts["test_records"],
            "preparation_fingerprint": fingerprint,
        },
        "shards": shards,
    }
    _publish(destination / "index.json", index)
    return destination / "index.json"
