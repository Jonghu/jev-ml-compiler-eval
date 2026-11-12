"""Prepare resumable TenSet collections while retaining one candidate pool at a time."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path, PurePosixPath

from jev_compiler_eval.datasets.tenset import (
    _load_metadata,
    _reject_label_keys,
    load_prepared,
    prepare_records,
    workload_split,
)
from jev_compiler_eval.datasets.tenset_metadata import metadata_from_tensor_graph
from jev_compiler_eval.datasets.tenset_registry import load_tenset_registry

SCHEMA_VERSION = 1
COLLECTION_DATASET = "tenset_collection"


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _publish(path: Path, value: object) -> None:
    """Publish a complete JSON file atomically, refusing to replace any destination."""

    data = _json_bytes(value)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != data:
            raise FileExistsError(f"existing collection artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".tenset-", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        # Link creation is atomic and cannot overwrite a simultaneously published
        # file. Temporary and destination are on the same local filesystem.
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid collection JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"collection artifact must be a JSON object: {path}")
    return value


def _source_descriptor(path: Path, raw_root: Path) -> dict:
    """Group by the first source header; verify every subsequent header agrees.

    Header identity and source order never inspect measurement values or errors.
    Hashing and validation stream the file, retaining only its workload identity.
    """

    digest = hashlib.sha256()
    identity = None
    records = 0
    with path.open("rb") as stream:
        for number, line in enumerate(stream, 1):
            digest.update(line)
            if not line.strip() or line.lstrip().startswith(b"#"):
                continue
            try:
                record = json.loads(line)
                task = record["i"][0]
                if not isinstance(task, list):
                    raise ValueError("source task header must be a list")
                header = task[:2]
                if len(header) != 2 or any(
                    not isinstance(item, str) or not item for item in header
                ):
                    raise ValueError("workload key and target must be nonempty strings")
            except (ValueError, TypeError, KeyError, IndexError) as error:
                raise ValueError(f"invalid source header at {path}:{number}") from error
            if identity is None:
                identity = header
            elif header != identity:
                raise ValueError(
                    f"source mixes workload keys or targets: {path}:{number}"
                )
            records += 1
    return {
        "path": path.relative_to(raw_root).as_posix(),
        "sha256": digest.hexdigest(),
        "workload_key": identity[0] if identity else None,
        "target": identity[1] if identity else None,
        "records": records,
    }


def _discover_sources(
    raw_root: Path,
    platform: str,
    progress_callback: Callable[[str], None] | None = None,
) -> list[dict]:
    directories = [
        path
        for path in raw_root.rglob(platform)
        if path.is_dir() and path.parent.name == "measure_records"
    ]
    if len(directories) != 1:
        raise ValueError(
            f"expected one measure_records/{platform} directory; "
            f"found {len(directories)}"
        )
    paths = sorted(
        set(directories[0].glob("*.json")) | set(directories[0].glob("*.jsonl"))
    )
    if not paths:
        raise ValueError("no TenSet measurement files found")
    if progress_callback:
        progress_callback(f"Hashing and indexing {len(paths)} source files")
    descriptors = []
    for number, path in enumerate(paths, 1):
        descriptors.append(_source_descriptor(path, raw_root))
        if progress_callback and (number % 100 == 0 or number == len(paths)):
            progress_callback(f"Indexed source files: {number}/{len(paths)}")
    return descriptors


def _target_from_artifact(path: Path) -> str:
    prepared = load_prepared(path)
    targets = {entry["context"]["target"] for entry in prepared["workloads"]}
    if len(targets) != 1:
        raise ValueError("target reference must contain exactly one hardware target")
    target = targets.pop()
    declared = prepared.get("provenance", {}).get("target", target)
    if not isinstance(target, str) or not target or declared != target:
        raise ValueError("target reference provenance does not match its workloads")
    return target


def _verify_group_sources(
    raw_root: Path, sources: list[dict], measured_sources: list[dict] | None = None
) -> None:
    expected = {
        str((raw_root / source["path"]).resolve()): source["sha256"]
        for source in sources
    }
    actual = (
        {source["path"]: source["sha256"] for source in measured_sources}
        if measured_sources is not None
        else {path: _sha256(Path(path)) for path in expected}
    )
    if actual != expected:
        raise ValueError("source changed during collection preparation")


def _shard_path(index_path: Path, entry: Mapping[str, object]) -> Path:
    relative = entry.get("path")
    workload_id = entry.get("workload_id")
    if (
        not isinstance(workload_id, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", workload_id) is None
        or not isinstance(relative, str)
        or relative != f"workloads/{workload_id}.json"
        or PurePosixPath(relative).is_absolute()
    ):
        raise ValueError("invalid or escaping collection shard path")
    root = index_path.parent.resolve()
    path = root / relative
    if (
        path.is_symlink()
        or path.parent.is_symlink()
        or not path.resolve().is_relative_to(root)
    ):
        raise ValueError("collection shard path escapes its collection directory")
    return path


def load_collection_index(path: str | Path) -> dict:
    """Read a completed index without loading any workload candidate pool."""

    path = Path(path)
    index = _read_json(path)
    if (
        index.get("schema_version") != SCHEMA_VERSION
        or index.get("dataset") != COLLECTION_DATASET
        or index.get("status") != "complete"
        or not isinstance(index.get("provenance"), dict)
        or not isinstance(index.get("shards"), list)
        or not index["shards"]
    ):
        raise ValueError("invalid or incomplete TenSet collection index")
    identifiers = set()
    for entry in index["shards"]:
        if not isinstance(entry, dict):
            raise ValueError("collection shard descriptor must be a mapping")
        _shard_path(path, entry)
        if entry["workload_id"] in identifiers:
            raise ValueError("duplicate collection workload ID")
        identifiers.add(entry["workload_id"])
        if (
            not isinstance(entry.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is None
            or type(entry.get("candidate_count")) is not int
            or entry["candidate_count"] < 2
            or entry.get("split") not in {"test", "validation"}
            or not isinstance(entry.get("workload_key"), str)
            or not entry["workload_key"]
        ):
            raise ValueError("invalid collection shard descriptor")
    return index


def load_collection_shard(
    index_path: str | Path, descriptor: Mapping[str, object]
) -> dict:
    """Validate and load exactly one ordinary prepared TenSet document."""

    path = _shard_path(Path(index_path), descriptor)
    if _sha256(path) != descriptor.get("sha256"):
        raise ValueError(f"collection shard checksum mismatch: {path}")
    prepared = load_prepared(path)
    if len(prepared["workloads"]) != 1:
        raise ValueError("collection shard must contain exactly one workload")
    workload = prepared["workloads"][0]
    if (
        workload.get("workload_id") != descriptor.get("workload_id")
        or workload.get("split") != descriptor.get("split")
        or workload.get("context", {}).get("workload_key")
        != descriptor.get("workload_key")
        or len(workload.get("candidates", [])) != descriptor.get("candidate_count")
    ):
        raise ValueError("collection shard content does not match its descriptor")
    return prepared


def iter_collection_workloads(path: str | Path) -> Iterator[dict]:
    """Yield validated workloads in frozen index order, loading one shard at a time."""

    index = load_collection_index(path)
    for descriptor in index["shards"]:
        yield load_collection_shard(path, descriptor)["workloads"][0]


def _program_metadata(key: str, metadata: dict | None, graphs: dict | None) -> dict:
    if metadata is not None:
        if key not in metadata:
            raise ValueError("program metadata does not contain the selected workload")
        result = metadata[key]
    else:
        if key not in graphs:
            raise ValueError("task registry does not contain the selected workload")
        result = metadata_from_tensor_graph(graphs[key])
    if set(result) != {"operation", "tensors", "compute_dag", "loop_structure"}:
        raise ValueError(
            "program metadata must describe operations, tensors, DAG and loops"
        )
    if (
        result["operation"] in (None, "", "unknown")
        or not isinstance(result["tensors"], list)
        or not result["tensors"]
        or any(
            not isinstance(tensor, Mapping)
            or "shape" not in tensor
            or not tensor.get("dtype")
            for tensor in result["tensors"]
        )
        or not result["compute_dag"]
        or not isinstance(result["loop_structure"], list)
    ):
        raise ValueError("selected workload requires complete program metadata")
    _reject_label_keys(result)
    return result


def prepare_collection(
    raw_root: str | Path,
    *,
    output_dir: str | Path,
    platform: str = "e5-2673",
    target: str | None = None,
    target_from_prepared: str | Path | None = None,
    split: str = "test",
    seed: int = 42,
    min_candidates: int = 2,
    metadata_path: str | Path | None = None,
    task_registry_path: str | Path | None = None,
    workload_limit: int | None = None,
    resume: bool = False,
    skip_unsupported_programs: bool = False,
    progress_callback: Callable[[str], None] | None = None,
) -> Path:
    """Prepare every valid distinct schedule for each selected project workload.

    Selection uses only exact target, project hash split and an optional seeded
    workload budget. Preparation retains one workload's candidate pool at a time.
    The portable registry is loaded once; only selected tensor graphs are decoded.
    Unsupported or missing program metadata fails by default. Explicitly opting
    into ``skip_unsupported_programs`` records those exclusions, independently of
    runtime values. Completed shards and exclusions have immutable receipts, so
    resuming validates source hashes and settings before reusing prior work.
    """

    raw_root, output_dir = Path(raw_root).resolve(), Path(output_dir)
    if output_dir.is_symlink():
        raise ValueError("collection output directory must not be a symlink")
    if (
        type(seed) is not int
        or seed < 0
        or type(min_candidates) is not int
        or min_candidates < 2
    ):
        raise ValueError("seed must be nonnegative and min_candidates must be >= 2")
    if split not in {"test", "validation"}:
        raise ValueError("collection split must be test or validation")
    if not platform or Path(platform).name != platform:
        raise ValueError("platform must be a single directory name")
    if workload_limit is not None and (
        type(workload_limit) is not int or workload_limit < 1
    ):
        raise ValueError("workload_limit must be a positive integer")
    if (target is None) == (target_from_prepared is None):
        raise ValueError("provide exactly one of target or target_from_prepared")
    if (metadata_path is None) == (task_registry_path is None):
        raise ValueError("provide exactly one of metadata_path or task_registry_path")
    if type(resume) is not bool or type(skip_unsupported_programs) is not bool:
        raise ValueError("resume and skip_unsupported_programs must be booleans")
    preparation_path = output_dir / "preparation.json"
    if resume and not preparation_path.is_file():
        raise FileNotFoundError(
            "resume requires an existing collection preparation.json"
        )
    if not resume and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            "collection output is nonempty; use --resume with identical inputs"
        )
    if target_from_prepared is not None:
        target = _target_from_artifact(Path(target_from_prepared))
    if not isinstance(target, str) or not target.strip():
        raise ValueError("target must be a nonempty exact TVM target string")
    sources = _discover_sources(raw_root, platform, progress_callback)
    groups = {}
    for source in sources:
        if source["target"] == target:
            groups.setdefault(source["workload_key"], []).append(source)
    selected = [key for key in groups if workload_split(key, seed=seed) == split]
    selected.sort(
        key=lambda key: hashlib.sha256(
            json.dumps([seed, key], separators=(",", ":")).encode()
        ).hexdigest()
    )
    split_groups = len(selected)
    if workload_limit is not None:
        selected = selected[:workload_limit]
    provider_path = Path(metadata_path or task_registry_path).resolve()
    settings = {
        "platform": platform,
        "target": target,
        "split": split,
        "seed": seed,
        "min_candidates": min_candidates,
        "workload_limit": workload_limit,
        "validation_fraction": 0.1,
        "test_fraction": 0.2,
        "skip_unsupported_programs": skip_unsupported_programs,
    }
    inputs = {
        "sources": sources,
        "preprocessing_sources": {
            name: _sha256(Path(__file__).with_name(name))
            for name in (
                "tenset_collection.py",
                "tenset.py",
                "tenset_metadata.py",
                "tenset_registry.py",
            )
        },
        "program_metadata": {
            "kind": "metadata_json" if metadata_path else "portable_registry",
            "sha256": _sha256(provider_path),
            "path": str(provider_path),
        },
        "target_reference": (
            {
                "sha256": _sha256(Path(target_from_prepared)),
                "path": str(Path(target_from_prepared).resolve()),
            }
            if target_from_prepared
            else None
        ),
    }
    state = {
        "schema_version": 1,
        "dataset": "tenset_collection_preparation",
        "settings": settings,
        "inputs": inputs,
        "selected_workload_keys": selected,
    }
    if resume and _read_json(preparation_path) != state:
        raise ValueError(
            "collection resume settings or source hashes differ from frozen preparation"
        )
    if not selected:
        raise ValueError(
            "no source workload groups match the exact target and project split"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    _publish(preparation_path, state)
    preparation_fingerprint = _fingerprint(state)
    metadata, graphs = None, None
    provider_loaded = False
    shards, exclusions = [], []
    counts = Counter()

    def report_group(number: int) -> None:
        if progress_callback:
            progress_callback(
                f"Prepared source groups: {number}/{len(selected)}; "
                f"eligible workloads: {len(shards)}; exclusions: {len(exclusions)}"
            )

    for number, key in enumerate(selected, 1):
        workload_id = (
            "tenset-"
            + hashlib.sha256(
                json.dumps(
                    [key, target], sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest()[:24]
        )
        receipt_path = output_dir / "receipts" / f"{workload_id}.json"
        base = {
            "workload_id": workload_id,
            "workload_key": key,
            "split": split,
            "preparation_fingerprint": preparation_fingerprint,
            "sources": groups[key],
        }
        if receipt_path.exists():
            receipt = _read_json(receipt_path)
            if any(receipt.get(field) != value for field, value in base.items()):
                raise ValueError("collection receipt does not match frozen inputs")
            if receipt.get("status") == "included":
                descriptor = receipt.get("shard")
                if not isinstance(descriptor, Mapping) or any(
                    descriptor.get(field) != base[field]
                    for field in ("workload_id", "workload_key", "split")
                ):
                    raise ValueError(
                        "collection receipt shard identity differs from its workload"
                    )
                verified = load_collection_shard(output_dir / "index.json", descriptor)
                provenance = verified.get("provenance", {})
                if (
                    provenance.get("collection_preparation_fingerprint")
                    != preparation_fingerprint
                    or provenance.get("program_metadata_sha256")
                    != inputs["program_metadata"]["sha256"]
                    or provenance.get("target") != target
                    or not isinstance(provenance.get("sources"), list)
                ):
                    raise ValueError(
                        "verified shard provenance differs from frozen inputs"
                    )
                _verify_group_sources(raw_root, groups[key], provenance["sources"])
                verified_counts = provenance.get("counts")
                if (
                    not isinstance(verified_counts, Mapping)
                    or receipt.get("counts") != verified_counts
                ):
                    raise ValueError(
                        "collection receipt counts differ from verified shard counts"
                    )
                shards.append(descriptor)
                counts.update(verified_counts)
                del verified
            elif receipt.get("status") == "excluded":
                exclusions.append(receipt)
            else:
                raise ValueError("invalid collection receipt status")
            report_group(number)
            continue
        try:
            prepared = prepare_records(
                [raw_root / source["path"] for source in groups[key]],
                target=target,
                seed=seed,
                min_candidates=min_candidates,
            )
        except ValueError as error:
            if str(error) != "no workloads meet the target and minimum candidate count":
                raise
            _verify_group_sources(raw_root, groups[key])
            receipt = {
                **base,
                "status": "excluded",
                "reason": "below_minimum_valid_distinct_candidates",
            }
            _publish(receipt_path, receipt)
            exclusions.append(receipt)
            report_group(number)
            continue
        _verify_group_sources(raw_root, groups[key], prepared["provenance"]["sources"])
        workloads = prepared["workloads"]
        if len(workloads) != 1 or workloads[0]["workload_id"] != workload_id:
            raise ValueError(
                "source group did not produce its expected single workload"
            )
        workload = workloads[0]
        if not provider_loaded:
            if progress_callback:
                progress_callback("Loading portable program metadata once")
            metadata = _load_metadata(metadata_path) if metadata_path else None
            if task_registry_path:
                graphs = {
                    entry["workload_key"]: entry["tensor_graph"]
                    for entry in load_tenset_registry(provider_path)
                }
            if _sha256(provider_path) != inputs["program_metadata"]["sha256"]:
                raise ValueError(
                    "program metadata changed during collection preparation"
                )
            provider_loaded = True
        try:
            program = _program_metadata(key, metadata, graphs)
        except (ValueError, TypeError, KeyError) as error:
            if not skip_unsupported_programs:
                raise ValueError(
                    f"cannot decode program metadata for {key}: {error}"
                ) from error
            receipt = {
                **base,
                "status": "excluded",
                "reason": "unsupported_program_metadata",
                "detail": str(error),
                "candidate_count": len(workload["candidates"]),
            }
            _publish(receipt_path, receipt)
            exclusions.append(receipt)
            prepared = workloads = workload = None
            report_group(number)
            continue
        workload["context"].update(program)
        prepared["provenance"].update(
            {
                "platform": platform,
                "collection_preparation_fingerprint": preparation_fingerprint,
                "program_metadata_sha256": inputs["program_metadata"]["sha256"],
                "candidate_selection": "all_valid_distinct_schedules",
            }
        )
        path = output_dir / "workloads" / f"{workload_id}.json"
        _publish(path, prepared)
        descriptor = {
            "path": path.relative_to(output_dir).as_posix(),
            "sha256": _sha256(path),
            "workload_id": workload_id,
            "workload_key": key,
            "split": split,
            "candidate_count": len(workload["candidates"]),
        }
        receipt = {
            **base,
            "status": "included",
            "shard": descriptor,
            "counts": prepared["provenance"]["counts"],
        }
        _publish(receipt_path, receipt)
        shards.append(descriptor)
        counts.update(receipt["counts"])
        # No collection-wide list retains prepared workloads or candidate arrays.
        del prepared, workloads, workload
        report_group(number)
    exclusions_path = output_dir / "exclusions.json"
    _publish(
        exclusions_path,
        {
            "schema_version": 1,
            "preparation_fingerprint": preparation_fingerprint,
            "excluded": exclusions,
        },
    )
    if not shards:
        raise ValueError(
            "no eligible workloads remain; inspect collection exclusions.json"
        )
    index = {
        "schema_version": 1,
        "dataset": COLLECTION_DATASET,
        "status": "complete",
        "provenance": {
            "settings": settings,
            "preparation_sha256": _sha256(preparation_path),
            "input_inventory_sha256": _fingerprint(inputs),
            "source_files_available": len(sources),
            "source_files_empty": sum(source["records"] == 0 for source in sources),
            "source_files_other_targets": sum(
                source["target"] not in {target, None} for source in sources
            ),
            "source_records_indexed": sum(source["records"] for source in sources),
            "selected_source_files": sum(len(groups[key]) for key in selected),
            "source_groups_matching_target": len(groups),
            "source_groups_matching_split": split_groups,
            "selected_source_groups": len(selected),
            "eligible_workloads": len(shards),
            "excluded_workloads": len(exclusions),
            "candidate_count": sum(entry["candidate_count"] for entry in shards),
            "counts_for_included_workloads": dict(counts),
            "counts_scope": "included_workload_groups",
            "candidate_selection": "all_valid_distinct_schedules",
            "split_definition": "project_hash_split_not_official_tenset_split",
            "dataset_license": "CC BY 4.0",
        },
        "exclusions": {
            "path": "exclusions.json",
            "sha256": _sha256(exclusions_path),
            "count": len(exclusions),
        },
        "shards": shards,
    }
    index_path = output_dir / "index.json"
    _publish(index_path, index)
    load_collection_index(index_path)
    return index_path
