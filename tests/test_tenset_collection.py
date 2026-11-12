"""Synthetic collection boundaries; these measurements are not benchmark results."""

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

import jev_compiler_eval.datasets.tenset_collection as collection
from jev_compiler_eval.datasets.tenset import TenSetDataset, workload_split

TARGET = "llvm -keys=cpu -mcpu=synthetic-test"


def _key(split="test", index=0):
    keys = (
        json.dumps([f"synthetic-{number}", [16, 8], [8, 4]]) for number in range(10000)
    )
    return [key for key in keys if workload_split(key) == split][index]


def _record(key, index=0, *, cost=0.001, error=0, target=TARGET):
    return {
        "i": [[key, target], [[], [["SP", 0, 0, 16, [index + 1, 1], 1]]]],
        "r": [[cost], error, 0.0, 0.0],
        "v": "v0.6",
    }


def _raw(root, name, records):
    directory = root / "measure_records" / "e5-2673"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(
        "# Synthetic test input; not TenSet benchmark results.\n"
        + "\n".join(json.dumps(row) for row in records)
        + "\n",
        encoding="utf-8",
    )
    return path


def _metadata(tmp_path, keys):
    path = tmp_path / "metadata.json"
    path.write_text(
        json.dumps(
            {
                key: {
                    "operation": "synthetic_matmul",
                    "tensors": [{"name": "a", "shape": [16, 8], "dtype": "float32"}],
                    "compute_dag": "Synthetic DAG for collection boundary tests.",
                    "loop_structure": [],
                }
                for key in keys
            }
        )
    )
    return path


def _fixture(tmp_path, count=2):
    root = tmp_path / "raw"
    keys = [_key(index=index) for index in range(count)]
    for index, key in enumerate(keys):
        _raw(
            root,
            f"task-{index}.json",
            [_record(key, schedule) for schedule in range(3)],
        )
    metadata = _metadata(tmp_path, keys)
    kwargs = {
        "target": TARGET,
        "metadata_path": metadata,
        "output_dir": tmp_path / "collection",
    }
    return root, keys, kwargs


def test_groups_all_matching_files_and_retains_all_valid_distinct_schedules(tmp_path):
    root, keys, kwargs = _fixture(tmp_path, count=1)
    key = keys[0]
    _raw(root, "repeat.jsonl", [_record(key, 0, cost=0.003), _record(key, 3)])
    _raw(
        root, "invalid.json", [_record(key, 99, error=1), _record(key, 100, cost=1e10)]
    )
    train_key = _key("train")
    _raw(root, "train.json", [_record(train_key, i) for i in range(3)])
    _raw(
        root,
        "other-hardware.json",
        [_record(key, i, target="llvm -mcpu=other") for i in range(3)],
    )
    index_path = collection.prepare_collection(root, **kwargs)
    index = collection.load_collection_index(index_path)
    assert len(index["shards"]) == 1
    descriptor = index["shards"][0]
    assert descriptor["candidate_count"] == 4
    prepared = collection.load_collection_shard(index_path, descriptor)
    workload = prepared["workloads"][0]
    assert sorted(candidate["runtime_ms"] for candidate in workload["candidates"]) == [
        1,
        1,
        1,
        2,
    ]
    assert workload["context"]["operation"] == "synthetic_matmul"
    assert prepared["provenance"]["counts"]["invalid_measurements"] == 2
    assert len(prepared["provenance"]["sources"]) == 3
    case = TenSetDataset(
        {
            "path": str(index_path.parent / descriptor["path"]),
            "split": "test",
            "candidate_limit": None,
            "max_input_bytes": None,
            "require_program_metadata": True,
        }
    ).load()[0]
    assert len(case.candidates) == 4
    assert all(
        "runtime_ms" not in candidate.configuration for candidate in case.candidates
    )


def test_insufficient_schedule_groups_have_explicit_exclusions(tmp_path):
    root, keys, kwargs = _fixture(tmp_path, count=2)
    # Metadata for an ineligible group is unnecessary: filter valid distinct
    # schedules first, then require program semantics for eligible workloads.
    kwargs["metadata_path"] = _metadata(tmp_path, keys[:1])
    _raw(root, "task-1.json", [_record(keys[1], 0), _record(keys[1], 0, cost=0.002)])
    index_path = collection.prepare_collection(root, **kwargs)
    index = collection.load_collection_index(index_path)
    assert index["provenance"]["selected_source_groups"] == 2
    assert index["provenance"]["eligible_workloads"] == 1
    assert index["provenance"]["excluded_workloads"] == 1
    assert index["provenance"]["counts_scope"] == "included_workload_groups"
    exclusions = json.loads((index_path.parent / "exclusions.json").read_text())[
        "excluded"
    ]
    assert exclusions[0]["reason"] == "below_minimum_valid_distinct_candidates"
    assert exclusions[0]["workload_key"] == keys[1]


def test_complete_resume_checks_artifacts_and_skips_reprocessing(tmp_path, monkeypatch):
    root, _, kwargs = _fixture(tmp_path)
    index_path = collection.prepare_collection(root, **kwargs)
    original = index_path.read_bytes()
    shard_paths = list((index_path.parent / "workloads").glob("*.json"))
    timestamps = {path: path.stat().st_mtime_ns for path in shard_paths}

    def must_not_prepare(*args, **kwargs):
        raise AssertionError("completed workload must not be reprocessed")

    monkeypatch.setattr(collection, "prepare_records", must_not_prepare)
    monkeypatch.setattr(collection, "_load_metadata", must_not_prepare)
    assert collection.prepare_collection(root, **kwargs, resume=True) == index_path
    assert index_path.read_bytes() == original
    assert {path: path.stat().st_mtime_ns for path in shard_paths} == timestamps
    with pytest.raises(FileExistsError, match="nonempty"):
        collection.prepare_collection(root, **kwargs)


@pytest.mark.parametrize(
    "tamper,expected",
    [
        ("counts", "receipt counts differ"),
        ("descriptor_alias", "shard identity differs"),
        ("fingerprint", "provenance differs"),
        ("source_hash", "source changed during"),
    ],
)
def test_resume_binds_receipt_diagnostics_and_identity_to_verified_shard(
    tmp_path,
    tamper,
    expected,
):
    root, _, kwargs = _fixture(tmp_path, count=2)
    index_path = collection.prepare_collection(root, **kwargs)
    index = collection.load_collection_index(index_path)
    descriptor = index["shards"][0]
    receipt_path = index_path.parent / "receipts" / f"{descriptor['workload_id']}.json"
    receipt = json.loads(receipt_path.read_text())
    # Recreate the state before final index publication, where a corrupted
    # receipt previously could publish false aggregate measurement counts.
    index_path.unlink()
    if tamper == "counts":
        receipt["counts"]["records_seen"] += 999
    elif tamper == "descriptor_alias":
        receipt["shard"] = index["shards"][1]
    else:
        shard_path = index_path.parent / descriptor["path"]
        shard = json.loads(shard_path.read_text())
        if tamper == "fingerprint":
            shard["provenance"]["collection_preparation_fingerprint"] = "f" * 64
        else:
            shard["provenance"]["sources"][0]["sha256"] = "f" * 64
        shard_path.write_text(json.dumps(shard))
        receipt["shard"]["sha256"] = hashlib.sha256(shard_path.read_bytes()).hexdigest()
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match=expected):
        collection.prepare_collection(root, **kwargs, resume=True)
    assert not index_path.exists()


def test_interruption_publishes_no_index_and_resume_processes_only_remaining_groups(
    tmp_path, monkeypatch
):
    root, _, kwargs = _fixture(tmp_path, count=3)
    original = collection.prepare_records
    calls = []

    def interrupt(paths, **settings):
        calls.append(paths)
        if len(calls) == 2:
            raise RuntimeError("synthetic interruption")
        return original(paths, **settings)

    monkeypatch.setattr(collection, "prepare_records", interrupt)
    with pytest.raises(RuntimeError, match="interruption"):
        collection.prepare_collection(root, **kwargs)
    assert not (kwargs["output_dir"] / "index.json").exists()
    assert len(list((kwargs["output_dir"] / "receipts").glob("*.json"))) == 1
    calls.clear()

    def observe(paths, **settings):
        calls.append(paths)
        return original(paths, **settings)

    monkeypatch.setattr(collection, "prepare_records", observe)
    index_path = collection.prepare_collection(root, **kwargs, resume=True)
    assert len(calls) == 2
    assert len(collection.load_collection_index(index_path)["shards"]) == 3


@pytest.mark.parametrize("change", ["source", "metadata", "seed", "limit"])
def test_resume_rejects_changed_sources_metadata_and_settings(tmp_path, change):
    root, _, kwargs = _fixture(tmp_path)
    collection.prepare_collection(root, **kwargs)
    if change == "source":
        path = root / "measure_records" / "e5-2673" / "task-0.json"
        path.write_text(path.read_text() + "# changed source bytes\n")
    elif change == "metadata":
        path = kwargs["metadata_path"]
        path.write_text(path.read_text() + " ")
    elif change == "seed":
        kwargs["seed"] = 43
    else:
        kwargs["workload_limit"] = 1
    with pytest.raises(ValueError, match="resume settings or source hashes"):
        collection.prepare_collection(root, **kwargs, resume=True)


def test_corrupt_completed_shard_is_never_reused(tmp_path):
    root, _, kwargs = _fixture(tmp_path)
    index_path = collection.prepare_collection(root, **kwargs)
    entry = collection.load_collection_index(index_path)["shards"][0]
    shard_path = index_path.parent / entry["path"]
    shard_path.write_text(shard_path.read_text() + " ")
    with pytest.raises(ValueError, match="checksum"):
        collection.prepare_collection(root, **kwargs, resume=True)


def test_orphan_shard_after_interruption_is_validated_without_overwrite(tmp_path):
    root, _, kwargs = _fixture(tmp_path, count=1)
    index_path = collection.prepare_collection(root, **kwargs)
    receipt = next((index_path.parent / "receipts").glob("*.json"))
    shard = next((index_path.parent / "workloads").glob("*.json"))
    timestamp = shard.stat().st_mtime_ns
    receipt.unlink()
    index_path.unlink()
    collection.prepare_collection(root, **kwargs, resume=True)
    assert shard.stat().st_mtime_ns == timestamp
    assert receipt.is_file()


def test_metadata_failures_are_fatal_unless_exclusion_is_explicit(tmp_path):
    root, keys, kwargs = _fixture(tmp_path, count=2)
    kwargs["metadata_path"] = _metadata(tmp_path, keys[:1])
    with pytest.raises(ValueError, match="cannot decode program metadata"):
        collection.prepare_collection(root, **kwargs)
    assert not (kwargs["output_dir"] / "index.json").exists()
    kwargs["output_dir"] = tmp_path / "explicit-exclusions"
    index_path = collection.prepare_collection(
        root, **kwargs, skip_unsupported_programs=True
    )
    index = collection.load_collection_index(index_path)
    assert index["provenance"]["excluded_workloads"] == 1
    assert index["provenance"]["settings"]["skip_unsupported_programs"] is True
    exclusions = json.loads((index_path.parent / "exclusions.json").read_text())[
        "excluded"
    ]
    assert exclusions[0]["reason"] == "unsupported_program_metadata"


def test_registry_is_loaded_once_and_only_selected_programs_are_decoded(
    tmp_path, monkeypatch
):
    fixture = (
        Path(__file__).parent / "fixtures" / "tenset_registry" / "native_registry.pkl"
    )
    records = collection.load_tenset_registry(fixture)
    selected = next(
        record for record in records if workload_split(record["workload_key"]) == "test"
    )
    root = tmp_path / "raw"
    _raw(
        root,
        "native-shape.json",
        [_record(selected["workload_key"], i) for i in range(3)],
    )
    loads, decodes = [], []
    original_decoder = collection.metadata_from_tensor_graph

    def registry_once(path):
        loads.append(path)
        return [selected, {"workload_key": _key("train"), "tensor_graph": {}}]

    def decoder(graph):
        decodes.append(graph)
        return original_decoder(graph)

    monkeypatch.setattr(collection, "load_tenset_registry", registry_once)
    monkeypatch.setattr(collection, "metadata_from_tensor_graph", decoder)
    index_path = collection.prepare_collection(
        root,
        output_dir=tmp_path / "collection",
        target=TARGET,
        task_registry_path=fixture,
    )
    assert len(loads) == len(decodes) == 1
    workload = next(collection.iter_collection_workloads(index_path))
    assert workload["context"]["operation"] not in {"unknown", ""}
    assert isinstance(workload["context"]["compute_dag"], dict)


def test_target_can_come_from_frozen_artifact_without_reusing_its_candidate_pool(
    tmp_path,
):
    root, _, kwargs = _fixture(tmp_path, count=1)
    first_index = collection.prepare_collection(root, **kwargs)
    descriptor = collection.load_collection_index(first_index)["shards"][0]
    reference = first_index.parent / descriptor["path"]
    kwargs.pop("target")
    kwargs["target_from_prepared"] = reference
    kwargs["output_dir"] = tmp_path / "second"
    index_path = collection.prepare_collection(root, **kwargs)
    index = collection.load_collection_index(index_path)
    assert index["provenance"]["settings"]["target"] == TARGET
    assert index["shards"][0]["candidate_count"] == 3


def test_lazy_iteration_loads_one_pool_per_next(tmp_path, monkeypatch):
    root, _, kwargs = _fixture(tmp_path, count=3)
    index_path = collection.prepare_collection(root, **kwargs)
    original = collection.load_collection_shard
    calls = []

    def observe(path, descriptor):
        calls.append(descriptor["workload_id"])
        return original(path, descriptor)

    monkeypatch.setattr(collection, "load_collection_shard", observe)
    iterator = collection.iter_collection_workloads(index_path)
    assert calls == []
    next(iterator)
    assert len(calls) == 1
    next(iterator)
    assert len(calls) == 2


def test_mixed_workload_source_is_rejected_instead_of_losing_hidden_groups(tmp_path):
    root, keys, kwargs = _fixture(tmp_path, count=2)
    _raw(root, "mixed.json", [_record(keys[0]), _record(keys[1])])
    with pytest.raises(ValueError, match="mixes workload keys"):
        collection.prepare_collection(root, **kwargs)


def test_label_changes_do_not_change_group_or_candidate_identity(tmp_path):
    root, _, kwargs = _fixture(tmp_path, count=3)
    first_index = collection.prepare_collection(root, **kwargs, workload_limit=2)
    first = list(collection.iter_collection_workloads(first_index))
    for path in (root / "measure_records" / "e5-2673").glob("*.json"):
        rows = [
            json.loads(line)
            for line in path.read_text().splitlines()
            if not line.startswith("#")
        ]
        for index, row in enumerate(rows):
            row["r"][0] = [float(index + 1)]
        _raw(root, path.name, rows)
    kwargs["output_dir"] = tmp_path / "different-labels"
    second_index = collection.prepare_collection(root, **kwargs, workload_limit=2)
    second = list(collection.iter_collection_workloads(second_index))
    assert [workload["workload_id"] for workload in first] == [
        workload["workload_id"] for workload in second
    ]
    assert [
        [candidate["candidate_id"] for candidate in workload["candidates"]]
        for workload in first
    ] == [
        [candidate["candidate_id"] for candidate in workload["candidates"]]
        for workload in second
    ]


def test_index_refuses_escaping_paths_and_descriptor_mismatch(tmp_path):
    root, _, kwargs = _fixture(tmp_path, count=1)
    index_path = collection.prepare_collection(root, **kwargs)
    index = collection.load_collection_index(index_path)
    entry = copy.deepcopy(index["shards"][0])
    entry["candidate_count"] += 1
    with pytest.raises(ValueError, match="does not match"):
        collection.load_collection_shard(index_path, entry)
    index["shards"][0]["path"] = "../../outside.json"
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="escaping"):
        collection.load_collection_index(index_path)


def test_progress_reports_inventory_and_each_group(tmp_path):
    root, _, kwargs = _fixture(tmp_path, count=3)
    messages = []
    collection.prepare_collection(root, **kwargs, progress_callback=messages.append)
    assert messages[0] == "Hashing and indexing 3 source files"
    assert "Indexed source files: 3/3" in messages
    assert messages.count("Loading portable program metadata once") == 1
    groups = [
        message for message in messages if message.startswith("Prepared source groups:")
    ]
    assert len(groups) == 3
    assert (
        groups[-1]
        == "Prepared source groups: 3/3; eligible workloads: 3; exclusions: 0"
    )


def test_sources_cannot_change_between_inventory_and_preparation(tmp_path, monkeypatch):
    root, _, kwargs = _fixture(tmp_path, count=1)
    original = collection.prepare_records

    def mutate_source(paths, **settings):
        path = paths[0]
        path.write_text(path.read_text() + "# concurrent source edit\n")
        return original(paths, **settings)

    monkeypatch.setattr(collection, "prepare_records", mutate_source)
    with pytest.raises(ValueError, match="source changed during"):
        collection.prepare_collection(root, **kwargs)
    assert not (kwargs["output_dir"] / "index.json").exists()


def test_cli_prepares_and_resumes_with_visible_progress(tmp_path):
    root, _, kwargs = _fixture(tmp_path, count=1)
    project_root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        str(project_root / "scripts" / "prepare_tenset_collection.py"),
        "--raw-root",
        str(root),
        "--target",
        TARGET,
        "--metadata",
        str(kwargs["metadata_path"]),
        "--output-dir",
        str(kwargs["output_dir"]),
    ]
    first = subprocess.run(
        command, cwd=project_root, capture_output=True, text=True, check=True
    )
    assert "Hashing and indexing 1 source files" in first.stdout
    assert "Eligible workloads: 1" in first.stdout
    assert "Distinct candidate schedules: 3" in first.stdout
    second = subprocess.run(
        command + ["--resume"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Eligible workloads: 1" in second.stdout
    assert "Loading portable program metadata once" not in second.stdout
