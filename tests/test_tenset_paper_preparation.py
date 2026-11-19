"""Full-network preparation boundaries; all record costs here are synthetic."""

import copy
import json
import pickle
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

import pytest
import test_tenset_registry as registry_fixtures

import jev_compiler_eval.datasets.tenset_paper as paper
from jev_compiler_eval.datasets.tenset import _digest
from jev_compiler_eval.datasets.tenset_collection import _fingerprint, _sha256
from jev_compiler_eval.datasets.tenset_registry import load_tenset_registry
from jev_compiler_eval.datasets.tenset_tutorial import _float32
from jev_compiler_eval.datasets.tenset_tutorial import (
    _record_pool as original_tutorial_record_pool,
)
from jev_compiler_eval.datasets.tenset_tutorial_protocol import _measurement_filename

registry_factory = registry_fixtures.registry_factory
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET = "llvm -keys=cpu -mcpu=skylake-avx512 -model=platinum-8272"


def _record(key, index, *, target=TARGET, error=0, cost=None, schedule=None):
    return {
        "i": [
            [key, target],
            [[], [["SP", 2, 0, 4, [(index if schedule is None else schedule) + 1], 1]]],
        ],
        "r": [[(index + 1) * 0.001 if cost is None else cost], error, 0, 0],
        "v": "v0.6",
    }


def _write_records(path, records):
    path.write_text(
        "# Synthetic timing fixtures.\n\n"
        + "\n".join(json.dumps(record) for record in records)
        + "\n"
    )


@pytest.fixture
def paper_fixture(tmp_path, registry_factory):
    """Real restricted task pickle plus native-generated semantic graph fixtures."""
    make_task, _, executed = registry_factory
    native = load_tenset_registry(
        PROJECT_ROOT / "tests/fixtures/tenset_registry/native_registry.pkl"
    )

    def make(weights=(3, 7), counts=(5, 4)):
        root = tmp_path / "raw"
        info = root / "network_info"
        measurements = root / "measure_records" / paper.PLATFORM
        info.mkdir(parents=True, exist_ok=True)
        measurements.mkdir(parents=True, exist_ok=True)
        keys = [entry["workload_key"] for entry in native[: len(weights)]]
        tasks = [
            make_task(entry["workload_key"], graph_value=entry["tensor_graph"])
            for entry in native[: len(weights)]
        ]
        (info / paper.TASK_INFO_FILENAME).write_bytes(
            pickle.dumps((tasks, list(weights)), protocol=4)
        )
        paths = []
        for key, count in zip(keys, counts, strict=True):
            path = measurements / _measurement_filename(key)
            _write_records(path, [_record(key, index) for index in range(count)])
            paths.append(path)
        assert executed == []
        return root, paths, keys, tmp_path / "prepared"

    return make


def test_complete_network_weights_and_full_source_record_order(paper_fixture):
    root, _, keys, output = paper_fixture()
    index_path = paper.prepare_paper_collection(root, output)
    index = paper.load_paper_index(index_path)
    assert [row["workload_key"] for row in index["shards"]] == keys
    assert [row["task_weight"] for row in index["shards"]] == [3, 7]
    assert [row["task_indices"] for row in index["shards"]] == [[0], [1]]
    assert [row["candidate_count"] for row in index["shards"]] == [5, 4]
    assert index["provenance"]["network_key"] == ["resnet_50", [[1, 3, 224, 224]]]
    assert index["provenance"]["record_split"] == "none"
    assert index["provenance"]["candidate_deduplication"] is False
    assert index["provenance"]["original_paper_inventory_verified"] is False
    assert index["provenance"]["diagnostic_task_count_matches"] is False
    assert index["provenance"]["observed_counts"]["source_records"] == 9
    for descriptor in index["shards"]:
        document = paper.load_paper_shard(index_path, descriptor)
        candidates = document["workloads"][0]["candidates"]
        assert [row["source_record_index"] for row in candidates] == list(
            range(len(candidates))
        )
        assert descriptor["candidate_ids"] == [
            row["candidate_id"] for row in candidates
        ]
        assert document["provenance"]["task_weight"] == descriptor["task_weight"]


def test_duplicates_errors_all_error_pools_and_float32_labels_are_retained(
    paper_fixture,
):
    root, paths, keys, output = paper_fixture(weights=(9,), counts=(4,))
    records = [
        _record(keys[0], i, error=1, cost=cost, schedule=0)
        for i, cost in enumerate((0.15, 0.3, 1e10, 0.001))
    ]
    records[0]["r"][0] = [0.1, 0.2]
    _write_records(paths[0], records)
    index_path = paper.prepare_paper_collection(root, output)
    descriptor = paper.load_paper_index(index_path)["shards"][0]
    document = paper.load_paper_shard(index_path, descriptor)
    rows = document["workloads"][0]["candidates"]
    assert len(rows) == len({row["candidate_id"] for row in rows}) == 4
    assert len({_digest(row["configuration"]) for row in rows}) == 1
    assert [row["source_error_code"] for row in rows] == [1, 1, 1, 1]
    minimum = _float32(0.001)
    assert descriptor["full_task_min_latency_seconds"] == minimum
    for row, original in zip(rows, records, strict=True):
        cost = _float32(sum(original["r"][0]) / len(original["r"][0]))
        assert row["reference_cost_seconds"] == cost
        assert row["runtime_ms"] == cost * 1000
        assert row["official_throughput"] == _float32(minimum / cost)
    assert document["provenance"]["counts"]["error_records"] == 4
    assert document["provenance"]["counts"]["duplicate_configurations"] == 3
    visible = json.dumps(
        {
            "context": document["workloads"][0]["context"],
            "configurations": [row["configuration"] for row in rows],
        }
    )
    for private in (
        "runtime_ms",
        "official_throughput",
        "reference_cost_seconds",
        "source_record_index",
        "source_error_code",
        "record_input",
    ):
        assert private not in visible


def test_native_throughput_underflow_zero_is_retained_only_when_opted_in(paper_fixture):
    root, paths, keys, output = paper_fixture(weights=(3,), counts=(2,))
    _write_records(
        paths[0], [_record(keys[0], 0, cost=1e-30), _record(keys[0], 1, cost=1e30)]
    )
    with pytest.raises(ValueError, match="finite and positive"):
        original_tutorial_record_pool(paths[0])
    with pytest.raises(ValueError, match="must be a boolean"):
        original_tutorial_record_pool(paths[0], allow_zero_throughput=1)
    index_path = paper.prepare_paper_collection(root, output)
    descriptor = paper.load_paper_index(index_path)["shards"][0]
    rows = paper.load_paper_shard(index_path, descriptor)["workloads"][0]["candidates"]
    assert [row["official_throughput"] for row in rows] == [1.0, 0.0]
    assert all(row["reference_cost_seconds"] > 0 for row in rows)


@pytest.mark.parametrize(
    "mutation", ["missing", "wrong_key", "wrong_target", "mixed_target", "empty"]
)
def test_missing_or_mismatched_cohort_sources_fail_without_complete_index(
    paper_fixture, mutation
):
    root, paths, keys, output = paper_fixture()
    if mutation == "missing":
        paths[1].unlink()
    elif mutation == "empty":
        paths[1].write_text("# No records\n")
    else:
        rows = [_record(keys[1], i) for i in range(4)]
        if mutation == "wrong_key":
            for row in rows:
                row["i"][0][0] = "wrong-workload"
        elif mutation == "wrong_target":
            for row in rows:
                row["i"][0][1] = "llvm -model=e5-2673 -mcpu=core-avx2"
        else:
            rows[-1]["i"][0][1] = "llvm -model=other"
        _write_records(paths[1], rows)
    with pytest.raises((ValueError, FileNotFoundError)):
        paper.prepare_paper_collection(root, output)
    assert not (output / "index.json").exists()


@pytest.mark.parametrize("weights", [[3, 7], [3, 3]])
def test_duplicate_network_task_keys_are_rejected_without_weight_aggregation(
    paper_fixture, registry_factory, weights
):
    root, _, _, output = paper_fixture()
    make_task, _, _ = registry_factory
    native = load_tenset_registry(
        PROJECT_ROOT / "tests/fixtures/tenset_registry/native_registry.pkl"
    )[0]
    task = make_task(native["workload_key"], graph_value=native["tensor_graph"])
    (root / "network_info" / paper.TASK_INFO_FILENAME).write_bytes(
        pickle.dumps(([task, task], weights), protocol=4)
    )
    with pytest.raises(ValueError, match="duplicate"):
        paper.prepare_paper_collection(root, output)
    assert not (output / "index.json").exists()


def test_missing_network_inventory_is_not_replaced_with_a_tutorial_preset(
    paper_fixture,
):
    root, _, _, output = paper_fixture()
    (root / "network_info" / paper.TASK_INFO_FILENAME).unlink()
    with pytest.raises(FileNotFoundError, match="ResNet-50"):
        paper.prepare_paper_collection(root, output)
    assert not (output / "index.json").exists()


def test_single_record_task_exposes_runner_limitation_without_altering_cohort(
    paper_fixture,
):
    root, _, _, output = paper_fixture(counts=(5, 1))
    with pytest.raises(
        ValueError, match="current paper runner adapter requires at least two"
    ):
        paper.prepare_paper_collection(root, output)
    assert not (output / "preparation.json").exists()
    assert not (output / "index.json").exists()


def test_unknown_semantics_fail_without_dropping_a_network_task(
    paper_fixture, registry_factory
):
    root, _, keys, output = paper_fixture()
    make_task, _, _ = registry_factory
    native = load_tenset_registry(
        PROJECT_ROOT / "tests/fixtures/tenset_registry/native_registry.pkl"
    )
    graph = copy.deepcopy(native[1]["tensor_graph"])
    graph["nodes"][graph["root"]]["type_key"] = "UnsupportedGraph"
    tasks = [
        make_task(keys[0], graph_value=native[0]["tensor_graph"]),
        make_task(keys[1], graph_value=graph),
    ]
    (root / "network_info" / paper.TASK_INFO_FILENAME).write_bytes(
        pickle.dumps((tasks, [3, 7]), protocol=4)
    )
    with pytest.raises(ValueError, match="unsupported"):
        paper.prepare_paper_collection(root, output)
    assert not (output / "index.json").exists()


def test_resume_reuses_verified_shards_and_rejects_receipt_count_tampering(
    paper_fixture, monkeypatch
):
    root, _, _, output = paper_fixture()
    original = paper._record_pool
    calls = []

    def interrupt(path):
        calls.append(path)
        if len(calls) == 2:
            raise RuntimeError("fixture interruption")
        return original(path)

    monkeypatch.setattr(paper, "_record_pool", interrupt)
    with pytest.raises(RuntimeError, match="interruption"):
        paper.prepare_paper_collection(root, output)
    assert not (output / "index.json").exists()
    shard = next((output / "workloads").glob("*.json"))
    timestamp = shard.stat().st_mtime_ns
    monkeypatch.setattr(paper, "_record_pool", original)
    index = paper.prepare_paper_collection(root, output, resume=True)
    assert shard.stat().st_mtime_ns == timestamp
    assert len(paper.load_paper_index(index)["shards"]) == 2
    receipt_path = next((output / "receipts").glob("*.json"))
    receipt = json.loads(receipt_path.read_text())
    receipt["counts"]["source_records"] += 10
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="counts"):
        paper.prepare_paper_collection(root, output, resume=True)


@pytest.mark.parametrize(
    "mutation", ["source", "weights", "shard", "candidate_order", "labels"]
)
def test_hashes_order_and_private_labels_detect_changes(paper_fixture, mutation):
    root, paths, _, output = paper_fixture()
    index_path = paper.prepare_paper_collection(root, output)
    index = paper.load_paper_index(index_path)
    descriptor = index["shards"][0]
    if mutation == "source":
        paths[0].write_text(paths[0].read_text() + "# changed\n")
        with pytest.raises(ValueError, match="changed on resume"):
            paper.prepare_paper_collection(root, output, resume=True)
    elif mutation == "weights":
        index["provenance"]["network_tasks"][0]["task_weight"] += 1
        index_path.write_text(json.dumps(index))
        with pytest.raises(ValueError, match="benchmark index"):
            paper.load_paper_index(index_path)
    else:
        path = output / descriptor["path"]
        if mutation == "shard":
            path.write_text(path.read_text() + " ")
            with pytest.raises(ValueError, match="checksum"):
                paper.load_paper_shard(index_path, descriptor)
        else:
            document = json.loads(path.read_text())
            records = document["workloads"][0]["candidates"]
            if mutation == "candidate_order":
                records.reverse()
                descriptor["candidate_ids"].reverse()
            else:
                records[0]["official_throughput"] /= 2
            path.write_text(json.dumps(document))
            descriptor["sha256"] = _sha256(path)
            index_path.write_text(json.dumps(index))
            with pytest.raises(ValueError, match="ordering|labels"):
                paper.load_paper_shard(index_path, descriptor)


def test_complete_resume_never_overwrites_existing_artifacts(
    paper_fixture, monkeypatch
):
    root, _, _, output = paper_fixture()
    index = paper.prepare_paper_collection(root, output)
    frozen = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in output.rglob("*.json")
    }

    def must_not_parse(path):
        raise AssertionError("completed record pool must not be parsed again")

    monkeypatch.setattr(paper, "_record_pool", must_not_parse)
    assert paper.prepare_paper_collection(root, output, resume=True) == index
    assert all(
        (path.read_bytes(), path.stat().st_mtime_ns) == before
        for path, before in frozen.items()
    )
    with pytest.raises(FileExistsError, match="nonempty"):
        paper.prepare_paper_collection(root, output)


@pytest.mark.parametrize("mismatch", ["changed", "missing", "extra"])
def test_self_consistent_old_code_commitments_fail_current_validation(
    paper_fixture, monkeypatch, mismatch
):
    """Simulate an older installation, including its internally valid hashes."""
    import yaml

    import jev_compiler_eval.models.jev_tournament as tournament
    from jev_compiler_eval.evaluation.tenset_paper import preflight_tenset_paper

    root, _, _, output = paper_fixture()
    current_sources = paper._preprocessing_sources
    earlier_sources = current_sources()
    if mismatch == "changed":
        earlier_sources = {name: "0" * 64 for name in earlier_sources}
    elif mismatch == "missing":
        earlier_sources.pop("tenset_download.py")
    else:
        earlier_sources["obsolete_preparation_module.py"] = "0" * 64
    # Under the simulated old installation, preparation, shards, receipts and
    # index all agree. Restoring the current code must reject that artifact.
    monkeypatch.setattr(paper, "_preprocessing_sources", lambda: earlier_sources)
    index_path = paper.prepare_paper_collection(root, output)
    assert paper.load_paper_index(index_path)["status"] == "complete"
    monkeypatch.setattr(paper, "_preprocessing_sources", current_sources)
    with pytest.raises(ValueError, match="differs from installed preprocessing"):
        paper.load_paper_index(index_path)

    config = yaml.safe_load(
        (PROJECT_ROOT / "experiments/tenset/paper-resnet50/full.yaml").read_text()
    )
    config["dataset"]["config"]["path"] = str(index_path)
    config["experiment"]["output_dir"] = str(output.parent / "results")
    config_path = output.parent / "paper.yaml"
    config_path.write_text(yaml.safe_dump(config))

    def no_client(*args, **kwargs):
        raise AssertionError("an incompatible preparation must fail before any client")

    monkeypatch.setattr(tournament, "JevApiClient", no_client)
    with pytest.raises(ValueError, match="differs from installed preprocessing"):
        preflight_tenset_paper(config_path)


def test_local_zip_to_real_cli_preserves_archive_hash_and_never_loads_api(
    paper_fixture, tmp_path
):
    root, _, _, _ = paper_fixture()
    archive = tmp_path / "Downloads with spaces" / "dataset_cpu_v3.3.zip"
    archive.parent.mkdir()
    with ZipFile(archive, "w") as output:
        for path in root.rglob("*"):
            if path.is_file():
                output.write(path, "dataset_cpu/" + path.relative_to(root).as_posix())
    destination = tmp_path / "from-zip"
    extracted = tmp_path / "extracted"
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/prepare_tenset_paper.py"),
            "--archive",
            str(archive),
            "--extract-dir",
            str(extracted),
            "--output-dir",
            str(destination),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "no API requests" in result.stdout
    index = paper.load_paper_index(destination / "index.json")
    state = json.loads((destination / "preparation.json").read_text())
    assert state["input_source"]["archive_sha256"] == _sha256(archive)
    assert index["provenance"]["candidate_count"] == 9
    assert state["preprocessing_sources"]["tenset_paper.py"] == _sha256(
        PROJECT_ROOT / "src/jev_compiler_eval/datasets/tenset_paper.py"
    )
    assert state["preprocessing_sources"]["tenset_download.py"] == _sha256(
        PROJECT_ROOT / "src/jev_compiler_eval/datasets/tenset_download.py"
    )
    assert _fingerprint(state) == index["provenance"]["preparation_fingerprint"]
    assert not (destination / "predictions.jsonl").exists()


def test_zip_traversal_is_rejected_by_the_shared_safe_extractor(tmp_path):
    archive = tmp_path / "unsafe.zip"
    with ZipFile(archive, "w") as output:
        output.writestr("../escaped.json", "{}")
    with pytest.raises(ValueError, match="unsafe archive path"):
        paper.prepare_paper_collection(archive, tmp_path / "prepared")
    assert not (tmp_path / "escaped.json").exists()
