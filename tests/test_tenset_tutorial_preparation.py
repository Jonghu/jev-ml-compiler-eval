"""Portable tutorial reconstruction boundaries; all timings are synthetic."""

import json
import pickle
import re
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

import pytest
import test_tenset_registry as registry_fixtures
import yaml

import jev_compiler_eval.datasets.tenset_tutorial as tutorial
from jev_compiler_eval.datasets.tenset import TenSetDataset
from jev_compiler_eval.datasets.tenset_collection import (
    load_collection_index,
    load_collection_shard,
)
from jev_compiler_eval.datasets.tenset_download import extract_cpu_archive
from jev_compiler_eval.datasets.tenset_registry import load_tenset_registry
from jev_compiler_eval.datasets.tenset_tutorial_protocol import (
    batch_size_one_network_keys,
    clean_name,
)
from jev_compiler_eval.evaluation.tenset_collection import preflight_tenset_collection
from jev_compiler_eval.evaluation.tenset_tutorial import preflight_tenset_tutorial

TARGET = "llvm -keys=cpu -mcpu=skylake-avx512 -model=platinum-8272"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
registry_factory = registry_fixtures.registry_factory


def _record(key, index, *, costs=None, error=0, configuration_index=None):
    schedule = index if configuration_index is None else configuration_index
    return {
        "i": [[key, TARGET], [[], [["SP", 0, 0, 16, [schedule + 1, 1], 1]]]],
        "r": [[(index + 1) / 1000] if costs is None else costs, error, 0.0, 0.0],
        "v": "v0.6",
    }


def _write_records(path, rows):
    path.write_text(
        "# Synthetic records, not a TenSet experiment.\n\n"
        + "\n".join(json.dumps(row) for row in rows)
        + "\n"
    )


@pytest.fixture
def tutorial_fixture(tmp_path, monkeypatch):
    def make(counts=(48, 49, 50)):
        root = tmp_path / "raw"
        info = root / "network_info"
        info.mkdir(parents=True, exist_ok=True)
        (info / "all_tasks.pkl").write_bytes(b"synthetic registry placeholder")
        measurements = root / "measure_records" / "platinum-8272"
        measurements.mkdir(parents=True, exist_ok=True)
        keys = [
            json.dumps([f"synthetic_task_{index}", [16, 8]])
            for index in range(len(counts))
        ]
        paths = []
        for number, (key, count) in enumerate(zip(keys, counts, strict=True)):
            path = measurements / f"task-{number}.json"
            _write_records(path, [_record(key, index) for index in range(count)])
            paths.append(path)

        def sample(network_info_dir, measurement_dir, *, sample_size=500, seed=0):
            assert network_info_dir == info
            assert measurement_dir == measurements
            assert sample_size == 500
            return list(paths), {
                "preset": "synthetic-test-pool",
                "seed": seed,
                "sample_size": len(paths),
                "selected_filenames": [path.name for path in paths],
                "selected_workload_keys": list(keys),
                "original_tutorial_inventory_verified": False,
            }

        def metadata(key, explicit, graphs):
            if key not in graphs:
                raise ValueError("task registry does not contain the selected workload")
            return {
                "operation": "synthetic_matmul",
                "tensors": [{"name": "a", "shape": [16, 8], "dtype": "float32"}],
                "compute_dag": "Synthetic tensor program for boundary testing.",
                "loop_structure": [],
            }

        monkeypatch.setattr(tutorial, "tutorial_file_sample", sample)
        monkeypatch.setattr(
            tutorial,
            "load_tenset_registry",
            lambda path: [{"workload_key": key, "tensor_graph": {}} for key in keys],
        )
        monkeypatch.setattr(tutorial, "_program_metadata", metadata)
        return root, paths, keys, tmp_path / "prepared"

    return make


@pytest.mark.parametrize(
    "count,expected",
    [
        (48, [4, 3, 2, 1, 0]),
        (49, [4, 3, 2, 1, 0]),
        (50, [4, 3, 2, 1, 0]),
        (51, [5, 4, 3, 2, 1, 0]),
    ],
)
def test_record_split_matches_reference_reverse_order(count, expected):
    assert tutorial.tutorial_test_indices(count) == expected


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_record_split_rejects_invalid_counts(count):
    with pytest.raises(ValueError, match="positive integer"):
        tutorial.tutorial_test_indices(count)


def test_preparation_preserves_original_record_indices_and_training_disjointness(
    tutorial_fixture,
):
    root, _, _, output = tutorial_fixture()
    index_path = tutorial.prepare_tutorial_collection(root, output)
    index = load_collection_index(index_path)
    assert len(index["shards"]) == 3
    for count, descriptor in zip((48, 49, 50), index["shards"], strict=True):
        document = load_collection_shard(index_path, descriptor)
        provenance = document["provenance"]
        test = provenance["test_source_record_indices"]
        train = provenance["train_source_record_indices"]
        assert test == [4, 3, 2, 1, 0]
        assert train == list(range(count - 1, 4, -1))
        assert set(test).isdisjoint(train)
        assert set(test + train) == set(range(count))
        candidates = document["workloads"][0]["candidates"]
        assert [candidate["source_record_index"] for candidate in candidates] == test
        assert descriptor["candidate_count"] == 5
    counts = index["provenance"]["observed_counts"]
    assert counts["training_records"] == 132
    assert counts["test_records"] == 15


def test_duplicate_and_failed_records_remain_unique_candidates_without_label_leaks(
    tutorial_fixture,
):
    root, paths, keys, output = tutorial_fixture((48,))
    rows = [_record(keys[0], index) for index in range(48)]
    rows[0] = _record(keys[0], 0, costs=[0.1, 0.2])
    rows[1] = _record(keys[0], 1, costs=[0.3], configuration_index=0)
    rows[2] = _record(keys[0], 2, costs=[1e10], error=2)
    rows[-1] = _record(keys[0], 47, costs=[0.001])
    _write_records(paths[0], rows)
    index_path = tutorial.prepare_tutorial_collection(root, output)
    index = load_collection_index(index_path)
    descriptor = index["shards"][0]
    document = load_collection_shard(index_path, descriptor)
    candidates = document["workloads"][0]["candidates"]
    by_index = {candidate["source_record_index"]: candidate for candidate in candidates}
    assert len(candidates) == 5
    assert len({candidate["candidate_id"] for candidate in candidates}) == 5
    assert all(
        re.fullmatch(r"record-[0-9a-f]{24}", candidate["candidate_id"])
        for candidate in candidates
    )
    assert by_index[0]["configuration"] == by_index[1]["configuration"]
    assert by_index[0]["candidate_id"] != by_index[1]["candidate_id"]
    assert by_index[2]["source_error_code"] == 2
    assert by_index[2]["runtime_ms"] == 1e13
    assert by_index[0]["runtime_ms"] == 150.00000596046448
    assert by_index[0]["official_throughput"] == 0.006666666828095913
    counts = document["provenance"]["counts"]
    assert counts["test_error_records"] == 1
    assert counts["test_duplicate_configurations"] == 1

    case = TenSetDataset(
        {
            "path": str(index_path.parent / descriptor["path"]),
            "split": "test",
            "candidate_limit": None,
            "max_input_bytes": None,
            "require_program_metadata": True,
        }
    ).load()[0]
    visible = json.dumps(
        {
            "context": dict(case.workload.context),
            "candidates": {
                candidate.candidate_id: dict(candidate.configuration)
                for candidate in case.candidates
            },
        }
    )
    for field in (
        "runtime_ms",
        "source_error_code",
        "source_record_index",
        "record_input",
        "record_version",
        "official_throughput",
    ):
        assert field not in visible
    assert len(case.candidates) == 5
    assert any(runtime.runtime_ms == 1e13 for runtime in case.runtimes)


def test_groups_below_48_records_have_explicit_exclusions(tutorial_fixture):
    root, paths, _, output = tutorial_fixture((47, 48))
    index_path = tutorial.prepare_tutorial_collection(root, output)
    index = load_collection_index(index_path)
    assert len(index["shards"]) == 1
    exclusions = json.loads((output / "exclusions.json").read_text())["excluded"]
    assert len(exclusions) == 1
    assert exclusions[0]["reason"] == "fewer_than_48_records"
    assert exclusions[0]["source"]["path"].endswith(paths[0].name)
    assert exclusions[0]["source"]["records"] == 47
    assert index["provenance"]["observed_counts"]["excluded_records"] == 47


def test_source_header_must_match_key_selected_from_network_registry(
    tutorial_fixture, monkeypatch
):
    root, paths, keys, output = tutorial_fixture((48,))
    unexpected = json.dumps(["different_registered_workload", [16, 8]])
    _write_records(paths[0], [_record(unexpected, index) for index in range(48)])
    monkeypatch.setattr(
        tutorial,
        "load_tenset_registry",
        lambda path: [
            {"workload_key": key, "tensor_graph": {}} for key in [keys[0], unexpected]
        ],
    )
    with pytest.raises(ValueError, match="workload key|selected|header|identity"):
        tutorial.prepare_tutorial_collection(root, output)
    assert not (output / "index.json").exists()


def test_empty_sampled_file_is_excluded_without_changing_sample_selection(
    tutorial_fixture,
):
    root, _, keys, output = tutorial_fixture((0, 48))
    index_path = tutorial.prepare_tutorial_collection(root, output)
    index = load_collection_index(index_path)
    assert len(index["shards"]) == 1
    assert index["shards"][0]["workload_key"] == keys[1]
    exclusions = json.loads((output / "exclusions.json").read_text())["excluded"]
    assert len(exclusions) == 1
    assert exclusions[0]["source"]["records"] == 0
    assert exclusions[0]["reason"] == "fewer_than_48_records"
    assert index["provenance"]["observed_counts"]["sampled_files"] == 2


@pytest.mark.parametrize(
    "bad_costs",
    [[], [0], [-1], [float("nan")], [float("inf")], [True], ["0.001"], [1e300]],
)
def test_unrepresentable_records_fail_without_silent_dropping(
    tutorial_fixture, bad_costs
):
    root, paths, keys, output = tutorial_fixture((48,))
    rows = [_record(keys[0], index) for index in range(48)]
    rows[20] = _record(keys[0], 20, costs=bad_costs)
    _write_records(paths[0], rows)
    with pytest.raises(ValueError, match="cannot preserve tutorial record"):
        tutorial.prepare_tutorial_collection(root, output)
    assert not (output / "index.json").exists()
    assert not list((output / "workloads").glob("*.json"))


def test_missing_record_result_and_missing_registry_coverage_fail(
    tutorial_fixture, monkeypatch
):
    root, paths, keys, output = tutorial_fixture((48,))
    rows = [_record(keys[0], index) for index in range(48)]
    del rows[0]["r"]
    _write_records(paths[0], rows)
    with pytest.raises(ValueError, match="cannot preserve tutorial record"):
        tutorial.prepare_tutorial_collection(root, output)
    _write_records(paths[0], [_record(keys[0], index) for index in range(48)])
    monkeypatch.setattr(tutorial, "load_tenset_registry", lambda path: [])
    with pytest.raises(ValueError, match="registry does not contain"):
        tutorial.prepare_tutorial_collection(root, output.parent / "other-output")


def test_complete_resume_verifies_artifacts_without_reprocessing(
    tutorial_fixture, monkeypatch
):
    root, _, _, output = tutorial_fixture((48, 50))
    index_path = tutorial.prepare_tutorial_collection(root, output)
    before = {
        path.relative_to(output): path.read_bytes() for path in output.rglob("*.json")
    }

    def no_reprocessing(*args, **kwargs):
        raise AssertionError("completed source must not be reprocessed")

    monkeypatch.setattr(tutorial, "_record_pool", no_reprocessing)
    assert tutorial.prepare_tutorial_collection(root, output, resume=True) == index_path
    assert {
        path.relative_to(output): path.read_bytes() for path in output.rglob("*.json")
    } == before
    with pytest.raises(FileExistsError, match="nonempty"):
        tutorial.prepare_tutorial_collection(root, output)


@pytest.mark.parametrize("tamper", ["shard", "receipt_counts", "source", "registry"])
def test_resume_rejects_tampered_artifacts_and_changed_inputs(tutorial_fixture, tamper):
    root, sources, _, output = tutorial_fixture((48, 50))
    index_path = tutorial.prepare_tutorial_collection(root, output)
    index = load_collection_index(index_path)
    descriptor = index["shards"][0]
    if tamper == "shard":
        path = output / descriptor["path"]
        value = json.loads(path.read_text())
        value["workloads"][0]["candidates"][0]["runtime_ms"] += 1
        path.write_text(json.dumps(value))
    elif tamper == "receipt_counts":
        path = output / "receipts" / f"{descriptor['workload_id']}.json"
        value = json.loads(path.read_text())
        value["counts"]["training_records"] += 1
        path.write_text(json.dumps(value))
    elif tamper == "source":
        sources[0].write_text(sources[0].read_text() + "\n")
    else:
        path = root / "network_info" / "all_tasks.pkl"
        path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum|receipt|changed on resume"):
        tutorial.prepare_tutorial_collection(root, output, resume=True)


def test_partial_resume_reuses_receipt_before_final_index_exists(
    tutorial_fixture, monkeypatch
):
    root, _, _, output = tutorial_fixture((48, 50))

    def interrupt(message):
        if message.startswith("Prepared 1/"):
            raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        tutorial.prepare_tutorial_collection(root, output, progress=interrupt)
    assert not (output / "index.json").exists()
    original = tutorial._record_pool
    calls = []

    def record_call(path):
        calls.append(path.name)
        return original(path)

    monkeypatch.setattr(tutorial, "_record_pool", record_call)
    index = tutorial.prepare_tutorial_collection(root, output, resume=True)
    assert index.is_file()
    assert calls == ["task-1.json"]


@pytest.mark.parametrize("dependency", ["tenset.py", "tenset_collection.py"])
def test_resume_rejects_changed_shared_preprocessing_code(
    tutorial_fixture, monkeypatch, dependency
):
    root, _, _, output = tutorial_fixture((48,))
    tutorial.prepare_tutorial_collection(root, output)
    original_sha256 = tutorial._sha256

    def changed_dependency(path):
        if path.name == dependency:
            return "f" * 64
        return original_sha256(path)

    monkeypatch.setattr(tutorial, "_sha256", changed_dependency)
    with pytest.raises(ValueError, match="preparation code changed on resume"):
        tutorial.prepare_tutorial_collection(root, output, resume=True)


def test_partial_resume_rejects_tampered_exclusion_reason(tutorial_fixture):
    root, _, _, output = tutorial_fixture((47, 48))

    def interrupt(message):
        if message.startswith("Prepared 1/"):
            raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        tutorial.prepare_tutorial_collection(root, output, progress=interrupt)
    receipt_path = next((output / "receipts").glob("*.json"))
    receipt = json.loads(receipt_path.read_text())
    assert receipt["reason"] == "fewer_than_48_records"
    receipt["reason"] = "unsupported_program_semantics"
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="receipt|exclusion"):
        tutorial.prepare_tutorial_collection(root, output, resume=True)


def test_matching_diagnostic_counts_never_claims_original_artifacts_or_model(
    tutorial_fixture, monkeypatch
):
    root, _, _, output = tutorial_fixture((48, 50))
    monkeypatch.setattr(
        tutorial,
        "PUBLISHED_COUNTS",
        {
            "sampled_files": 2,
            "eligible_workloads": 2,
            "training_records": 88,
            "test_records": 10,
        },
    )
    index_path = tutorial.prepare_tutorial_collection(root, output)
    status = json.loads((output / "reference-status.json").read_text())
    assert status["published_counts_match"] is True
    assert status["original_test_inventory_verified"] is False
    assert status["matching_xgboost_predictions_available"] is False
    assert status["native_tvm_features_reproduced"] is False
    assert (
        status["reference_alignment_status"] == "unverified_original_tutorial_artifacts"
    )
    assert (
        load_collection_index(index_path)["provenance"][
            "original_test_inventory_verified"
        ]
        is False
    )


def test_prepared_tutorial_collection_passes_offline_runner_preflight(
    tutorial_fixture, tmp_path
):
    root, _, _, output = tutorial_fixture((48, 50))
    index_path = tutorial.prepare_tutorial_collection(root, output)
    config = {
        "schema_version": 1,
        "experiment": {
            "name": "synthetic-tutorial-preflight",
            "seed": 0,
            "output_dir": str(tmp_path / "results"),
        },
        "dataset": {
            "type": "tenset",
            "config": {
                "path": str(index_path),
                "split": "test",
                "seed": 0,
                "candidate_limit": None,
                "max_input_bytes": None,
                "require_program_metadata": True,
            },
        },
        "candidate_selection": {"strategy": "all"},
        "models": [{"id": "random-0", "type": "random", "config": {"seed": 0}}],
        "metrics": ["top_1_score", "top_5_score", "runtime_regret"],
        "execution": {"fail_fast": True},
    }
    path = tmp_path / "preflight.yaml"
    path.write_text(yaml.safe_dump(config))
    result = preflight_tenset_collection(path)
    assert result.selected_workloads == 2
    assert result.total_candidates == 10
    assert result.min_candidates == result.max_candidates == 5
    assert result.api_requests_made == 0


def test_native_metadata_zip_to_real_cli_and_tournament_preflight(
    tmp_path, registry_factory
):
    """Exercise the Mac preparation chain with no mocked production readers.

    Source task-info constructors match upstream reducers, while the all-task
    registry and TE expression graph were generated independently by native TVM.
    Timing records are synthetic and are not benchmark results.
    """

    make_task, _, executed = registry_factory
    native_path = (
        PROJECT_ROOT / "tests" / "fixtures" / "tenset_registry" / "native_registry.pkl"
    )
    native = load_tenset_registry(native_path)[0]
    key = native["workload_key"]
    task = make_task(key, graph_value=native["tensor_graph"])
    task_info = pickle.dumps(([task], [1]), protocol=4)
    rows = [_record(key, index) for index in range(48)]
    for row in rows:
        row["i"][1] = [[], []]
    record_text = "\n".join(json.dumps(row) for row in rows) + "\n"

    download_dir = tmp_path / "Downloads with spaces"
    download_dir.mkdir()
    archive_path = download_dir / "dataset_cpu_v3.3.zip"
    measurement_name = clean_name((key, "llvm")) + ".json"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "dataset_cpu/network_info/all_tasks.pkl", native_path.read_bytes()
        )
        for network in batch_size_one_network_keys():
            name = clean_name((network, "llvm")) + ".task.pkl"
            archive.writestr(f"dataset_cpu/network_info/{name}", task_info)
        archive.writestr(
            f"dataset_cpu/measure_records/platinum-8272/{measurement_name}", record_text
        )
    assert executed == []
    raw = tmp_path / "extracted"
    manifest = extract_cpu_archive(
        archive_path, raw, platform="platinum-8272", include_network_tasks=True
    )
    assert len(manifest["files"]) == 42

    output = tmp_path / "prepared-native"
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "prepare_tenset_tutorial.py"),
            "--raw-root",
            str(raw),
            "--output-dir",
            str(output),
            "--sample-size",
            "1",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "no Jev API calls or XGBoost training" in result.stdout
    index_path = output / "index.json"
    index = load_collection_index(index_path)
    assert len(index["shards"]) == 1
    descriptor = index["shards"][0]
    assert descriptor["candidate_count"] == 5
    document = load_collection_shard(index_path, descriptor)
    assert document["workloads"][0]["context"]["operation"] == "matmul"
    state = json.loads((output / "preparation.json").read_text())
    assert state["selection"]["selected_workload_keys"] == [key]
    assert len(state["selection"]["network_task_info"]) == 40

    profile = yaml.safe_load(
        (
            PROJECT_ROOT / "experiments" / "tenset" / "jev-tutorial-500-pilot.yaml"
        ).read_text()
    )
    profile["experiment"]["name"] = "synthetic-native-tutorial-preflight"
    profile["experiment"]["output_dir"] = str(tmp_path / "results")
    profile["dataset"]["config"]["path"] = str(index_path)
    profile["dataset"]["config"]["max_workloads"] = 1
    config_path = tmp_path / "native-preflight.yaml"
    config_path.write_text(yaml.safe_dump(profile))
    plan = preflight_tenset_tutorial(config_path)
    assert plan["api_requests_made"] == 0
    assert plan["selected_workloads"] == 1
    assert plan["test_record_count"] == 5
    assert plan["paired_xgboost_comparison"] is False
    model_plan = next(iter(plan["model_plans"].values()))
    assert model_plan["first_round_requests"] == 1
