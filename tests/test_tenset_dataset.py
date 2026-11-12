"""Synthetic TVM-format fixtures test preprocessing, not TenSet performance."""

import copy
import json

import pytest

from jev_compiler_eval.datasets.tenset import (
    TenSetDataset,
    load_prepared,
    prepare_records,
    serialized_input_bytes,
    workload_split,
)

TARGET = "llvm -keys=cpu -mcpu=core-avx2"


def record(index=0, *, workload="matmul", costs=None, error=0, target=TARGET):
    return {
        "i": [
            [
                json.dumps([workload, [128, 64], [64, 32]]),
                target,
                [8, 64, 64, 0, 0, 0, 0, 0],
                "",
                2,
            ],
            [
                [],
                [
                    ["SP", 0, 0, 128, [index + 1, 1], 1],
                    ["RE", 0, [0, 2, 1]],
                    ["PR", 0, 0, "auto_unroll_max_step$64"],
                ],
            ],
        ],
        "r": [costs if costs is not None else [0.001, 0.003], error, 999, 12345],
        "v": "v0.6",
    }


def raw_file(tmp_path, records):
    path = tmp_path / "records.json"
    path.write_text(
        "# Synthetic measurements, not benchmark results.\n"
        + "\n".join(json.dumps(item) for item in records)
        + "\n",
        encoding="utf-8",
    )
    return path


def prepared_file(tmp_path, records, **kwargs):
    root = prepare_records([raw_file(tmp_path, records)], target=TARGET, **kwargs)
    path = tmp_path / "prepared.json"
    path.write_text(json.dumps(root), encoding="utf-8")
    return path, root


def visible(case):
    return {
        "workload_id": case.workload.workload_id,
        "context": case.workload.context,
        "candidates": [
            (candidate.candidate_id, candidate.configuration)
            for candidate in case.candidates
        ],
    }


def test_extracts_tvm_records_and_converts_seconds_to_milliseconds(tmp_path):
    path, root = prepared_file(tmp_path, [record(0), record(1)])
    cases = TenSetDataset({"path": str(path), "split": "all"}).load()
    assert len(cases) == 1
    assert cases[0].workload.context["operation"] == "matmul"
    assert cases[0].workload.context["tensors"] == [
        {"shape": [128, 64]},
        {"shape": [64, 32]},
    ]
    assert cases[0].workload.context["hardware_params"] == [8, 64, 64, 0, 0, 0, 0, 0]
    assert all(runtime.runtime_ms == 2 for runtime in cases[0].runtimes)
    config = cases[0].candidates[0].configuration
    assert [step["name"] for step in config["transformations"]] == [
        "split",
        "reorder",
        "pragma",
    ]
    assert config["transformations"][0]["arguments"][0:3] == [0, 0, 128]
    assert root["workloads"][0]["candidates"][0]["record_input"]
    assert root["provenance"]["sources"][0]["sha256"]


def test_deduplicates_schedule_and_averages_repeated_measurements(tmp_path):
    path, root = prepared_file(
        tmp_path,
        [
            record(0, costs=[0.001, 0.003]),
            record(0, costs=[0.004]),
            record(1),
        ],
    )
    cases = TenSetDataset({"path": str(path), "split": "all"}).load()
    assert len(cases[0].candidates) == 2
    assert sorted(runtime.runtime_ms for runtime in cases[0].runtimes) == [2, 3]
    assert root["provenance"]["counts"]["valid_measurements"] == 3


@pytest.mark.parametrize(
    "costs,error",
    [
        ([0], 0),
        ([-1], 0),
        ([float("nan")], 0),
        ([float("inf")], 0),
        ([1e10], 0),
        ([], 0),
        ([True], 0),
        ([0.001], 4),
    ],
)
def test_filters_failed_nonfinite_nonpositive_and_sentinel_labels(
    tmp_path, costs, error
):
    _, root = prepared_file(
        tmp_path,
        [
            record(0),
            record(1),
            record(2, costs=costs, error=error),
        ],
    )
    assert len(root["workloads"][0]["candidates"]) == 2
    assert root["provenance"]["counts"]["invalid_measurements"] == 1


def test_target_filter_prevents_cross_hardware_aggregation(tmp_path):
    path, root = prepared_file(
        tmp_path,
        [
            record(0),
            record(1),
            record(2, target="cuda"),
        ],
    )
    case = TenSetDataset({"path": str(path), "split": "all"}).load()[0]
    assert len(case.candidates) == 2
    assert case.workload.context["target"] == TARGET
    assert root["provenance"]["counts"]["other_targets"] == 1


def test_labels_timestamps_and_measure_input_stay_out_of_model_inputs(tmp_path):
    path, _ = prepared_file(tmp_path, [record(0), record(1)])
    case = TenSetDataset({"path": str(path), "split": "all"}).load()[0]
    text = json.dumps(visible(case))
    for hidden in ("runtime", "record_input", "record_version", "12345", "999"):
        assert hidden not in text
    assert serialized_input_bytes(case.workload, case.candidates) > 0


def test_changing_labels_cannot_change_ids_splits_sampling_or_serialization(tmp_path):
    records = [
        record(index, workload=f"matmul-{workload}")
        for workload in range(30)
        for index in range(8)
    ]
    path, original = prepared_file(tmp_path, records)
    config = {"path": str(path), "split": "all", "candidate_limit": 3}
    before = TenSetDataset(config).load()
    for index, item in enumerate(records):
        item["r"] = [[0.1 * (index + 1)], 0, index, index]
    path, changed = prepared_file(tmp_path, list(reversed(records)))
    after = TenSetDataset(config).load()
    assert [visible(case) for case in before] == [visible(case) for case in after]
    assert [workload["split"] for workload in original["workloads"]] == [
        workload["split"] for workload in changed["workloads"]
    ]


def test_candidate_limits_are_nested_prefixes_and_manifest_records_them(tmp_path):
    path, _ = prepared_file(tmp_path, [record(index) for index in range(20)])
    small_dataset = TenSetDataset(
        {
            "path": str(path),
            "split": "all",
            "candidate_limit": 5,
        }
    )
    small = small_dataset.load()[0]
    large = TenSetDataset(
        {
            "path": str(path),
            "split": "all",
            "candidate_limit": 10,
        }
    ).load()[0]
    assert small.candidates == large.candidates[:5]
    manifest = small_dataset.manifest_metadata()
    assert manifest["prepared_file_sha256"]
    assert manifest["realized_cases"][0]["candidate_ids"] == [
        candidate.candidate_id for candidate in small.candidates
    ]


def test_workload_splits_are_disjoint_and_do_not_depend_on_hardware(tmp_path):
    records = [
        record(index, workload=f"program-{workload}")
        for workload in range(80)
        for index in range(2)
    ]
    path, _ = prepared_file(tmp_path, records)
    sets = []
    for split in ("train", "validation", "test"):
        cases = TenSetDataset({"path": str(path), "split": split}).load()
        sets.append({case.workload.context["workload_key"] for case in cases})
    assert len(set.union(*sets)) == 80
    assert all(
        not left & right
        for index, left in enumerate(sets)
        for right in sets[index + 1 :]
    )
    for program in set.union(*sets):
        assert (
            program
            in sets[
                ("train", "validation", "test").index(
                    workload_split(program),
                )
            ]
        )


def test_opaque_hash_has_unknown_operation_until_metadata_is_supplied(tmp_path):
    opaque = "d7b65649a4dd54becea0a52aabbc5af5"
    records = [record(0, workload=opaque), record(1, workload=opaque)]
    _, root = prepared_file(tmp_path, records)
    assert root["workloads"][0]["context"]["operation"] == "unknown"
    key = records[0]["i"][0][0]
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                key: {
                    "operation": "matmul",
                    "compute_dag": "C[i,j] += A[i,k] * B[k,j]",
                    "loop_structure": [{"extent": 128, "kind": "spatial"}],
                }
            }
        ),
        encoding="utf-8",
    )
    _, augmented = prepared_file(tmp_path, records, metadata_path=metadata)
    context = augmented["workloads"][0]["context"]
    assert context["operation"] == "matmul"
    assert "compute_dag" in context
    assert augmented["provenance"]["metadata_sha256"]


def test_rejects_label_bearing_metadata(tmp_path):
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({"key": {"runtime_ms": 2}}), encoding="utf-8")
    with pytest.raises(ValueError, match="label-bearing"):
        prepared_file(tmp_path, [record(0), record(1)], metadata_path=metadata)


def test_schedule_specific_state_is_kept_in_candidate_not_workload(tmp_path):
    records = [record(0), record(1)]
    records[0]["i"][1][0] = [["synthetic-stage", 1]]
    path, root = prepared_file(tmp_path, records)
    case = TenSetDataset({"path": str(path), "split": "all"}).load()[0]
    assert "initial_state" not in case.workload.context
    assert any(
        candidate.configuration["initial_state"] for candidate in case.candidates
    )
    assert len(root["workloads"]) == 1


def test_rejects_oversized_input_without_silently_truncating_candidates(tmp_path):
    path, _ = prepared_file(tmp_path, [record(0), record(1)])
    dataset = TenSetDataset({"path": str(path), "split": "all", "max_input_bytes": 1})
    with pytest.raises(ValueError, match="exceeds local"):
        dataset.load()


def test_workload_budget_excludes_oversized_unselected_inputs(tmp_path):
    path, root = prepared_file(
        tmp_path,
        [
            record(0, workload="small-program"),
            record(1, workload="small-program"),
            record(0, workload="large-program"),
            record(1, workload="large-program"),
        ],
    )
    selected = TenSetDataset(
        {
            "path": str(path),
            "split": "all",
            "max_workloads": 1,
        }
    ).load()[0]
    input_bytes = serialized_input_bytes(selected.workload, selected.candidates)
    for data in root["workloads"]:
        if data["workload_id"] != selected.workload.workload_id:
            data["context"]["compute_dag"] = "synthetic-large-dag" * input_bytes
    root["workloads"].reverse()
    path.write_text(json.dumps(root), encoding="utf-8")
    config = {
        "path": str(path),
        "split": "all",
        "max_workloads": 1,
        "max_input_bytes": input_bytes,
    }
    assert TenSetDataset(config).load() == (selected,)
    with pytest.raises(ValueError, match="exceeds local"):
        TenSetDataset({**config, "max_workloads": 2}).load()


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"path": ""},
        {"path": "x", "split": "invalid"},
        {"path": "x", "candidate_limit": 0},
        {"path": "x", "seed": True},
        {"path": "x", "unknown": 1},
    ],
)
def test_rejects_invalid_dataset_configuration(config):
    with pytest.raises(ValueError):
        TenSetDataset(config)


def test_rejects_malformed_record_with_source_and_line(tmp_path):
    path = raw_file(tmp_path, [{"i": []}])
    with pytest.raises(ValueError, match=r"records.json:2:"):
        prepare_records([path], target=TARGET)


@pytest.mark.parametrize("max_workloads", [None, 1])
def test_rejects_duplicate_workload_ids_in_prepared_file(tmp_path, max_workloads):
    path, root = prepared_file(tmp_path, [record(0), record(1)])
    root["workloads"].append(copy.deepcopy(root["workloads"][0]))
    path.write_text(json.dumps(root), encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        TenSetDataset(
            {
                "path": str(path),
                "split": "all",
                "max_workloads": max_workloads,
            }
        ).load()


def test_rejects_unsupported_prepared_artifact(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps({"dataset": "mock"}), encoding="utf-8")
    with pytest.raises(ValueError, match="dataset=tenset"):
        load_prepared(path)


def test_rejects_empty_evaluation_split(tmp_path):
    path, _ = prepared_file(
        tmp_path, [record(0), record(1)], test_fraction=0, validation_fraction=0
    )
    with pytest.raises(ValueError, match="no workloads in split=test"):
        TenSetDataset({"path": str(path)}).load()
