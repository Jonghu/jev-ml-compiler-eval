import json
from pathlib import Path

import numpy as np
import pytest

from jev_compiler_eval.datasets import TpuGraphsTileDataset, get_dataset_class
from jev_compiler_eval.datasets.tpugraphs_features import (
    HLO_OPCODE_NAMES,
    NODE_FEATURE_NAMES,
    TILE_CONFIG_FEATURE_NAMES,
)

PARAMETER = next(code for code, name in HLO_OPCODE_NAMES.items() if name == "parameter")
CONVOLUTION = next(
    code for code, name in HLO_OPCODE_NAMES.items() if name == "convolution"
)


def _node(**features: float) -> np.ndarray:
    row = np.zeros(len(NODE_FEATURE_NAMES), dtype=np.float32)
    for name, value in features.items():
        row[NODE_FEATURE_NAMES.index(name)] = value
    return row


def _config(output_bounds: list[int]) -> np.ndarray:
    row = np.zeros(len(TILE_CONFIG_FEATURE_NAMES), dtype=np.float32)
    for dim, value in enumerate(output_bounds):
        row[TILE_CONFIG_FEATURE_NAMES.index(f"output_bounds_{dim}")] = value
    return row


def _write_graph(
    path: Path,
    *,
    configs: list[list[int]],
    runtimes: list[int],
    normalizers: list[int] | None = None,
) -> None:
    node_feat = np.stack(
        [
            _node(
                shape_element_type_is_f32=1,
                shape_dimensions_0=8,
                shape_dimensions_1=4,
                layout_minor_to_major_0=1,
                layout_minor_to_major_1=0,
            ),
            _node(
                is_root=1,
                shape_element_type_is_bf16=1,
                shape_dimensions_0=8,
                shape_dimensions_1=4,
                shape_dimensions_2=3,
                shape_dimensions_3=3,
                layout_minor_to_major_0=0,
                convolution_dim_numbers_input_batch_dim=0,
                convolution_dim_numbers_input_feature_dim=3,
                convolution_dim_numbers_input_spatial_dims_0=1,
                convolution_dim_numbers_input_spatial_dims_1=2,
                feature_group_count=1,
                batch_group_count=1,
                window_size_0=3,
                window_size_1=3,
                window_size_sum=6,
            ),
        ]
    )
    np.savez(
        path,
        node_feat=node_feat,
        node_opcode=np.array([PARAMETER, CONVOLUTION], dtype=np.uint8),
        edge_index=np.array([[1, 0]], dtype=np.int64),
        config_feat=np.stack([_config(bounds) for bounds in configs]),
        config_runtime=np.array(runtimes, dtype=np.int64),
        config_runtime_normalizers=np.array(
            normalizers or [runtimes[0]] * len(runtimes), dtype=np.int64
        ),
    )


def _dataset(path: Path, max_candidates: int = 100) -> TpuGraphsTileDataset:
    return TpuGraphsTileDataset({"path": str(path), "max_candidates": max_candidates})


def test_tpugraphs_tile_dataset_is_registered() -> None:
    assert get_dataset_class("tpugraphs-tile") is TpuGraphsTileDataset


def test_serializes_graph_with_named_features(tmp_path: Path) -> None:
    _write_graph(
        tmp_path / "resnet50.4x4.fp16_-1a2b.npz",
        configs=[[1, 2], [4, 4]],
        runtimes=[2_000_000, 1_000_000],
    )

    (case,) = _dataset(tmp_path).load()

    context = case.workload.context
    assert case.workload.workload_id == "resnet50.4x4.fp16_-1a2b"
    assert context["program"] == "resnet50.4x4.fp16"
    assert context["num_nodes"] == 2
    assert context["num_edges"] == 1
    parameter, convolution = context["nodes"]
    assert parameter == {
        "id": 0,
        "opcode": "parameter",
        "dtype": "f32",
        "shape": [8, 4],
        "layout_minor_to_major": [1, 0],
        "parameter_number": 0,
    }
    assert convolution["opcode"] == "convolution"
    assert convolution["operands"] == [0]
    assert convolution["is_root"] is True
    assert convolution["window_size"] == [3, 3]
    assert convolution["convolution"]["input_batch_dim"] == 0
    assert convolution["convolution"]["input_spatial_dims"] == [1, 2]
    assert "window_size_sum" not in convolution
    assert [dict(c.configuration) for c in case.candidates] == [
        {"output_bounds": [1, 2]},
        {"output_bounds": [4, 4]},
    ]


def test_converts_nanoseconds_to_milliseconds(tmp_path: Path) -> None:
    _write_graph(
        tmp_path / "graph_1.npz",
        configs=[[1], [2]],
        runtimes=[2_000_000, 1_500_000],
    )

    (case,) = _dataset(tmp_path).load()

    assert [r.runtime_ms for r in case.runtimes] == [2.0, 1.5]


def test_ranks_by_normalized_runtime(tmp_path: Path) -> None:
    _write_graph(
        tmp_path / "graph_1.npz",
        configs=[[1], [2], [3]],
        runtimes=[1_000_000, 990_000, 1_010_000],
        normalizers=[1_000_000, 900_000, 1_100_000],
    )

    (case,) = _dataset(tmp_path).load()

    runtimes = {r.candidate_id: r.runtime_ms for r in case.runtimes}
    assert runtimes["config_0"] == pytest.approx(1.0)
    assert runtimes["config_1"] == pytest.approx(1.1)
    assert runtimes["config_2"] == pytest.approx(1010 / 1100)
    assert min(runtimes, key=runtimes.__getitem__) == "config_2"


def test_merges_duplicate_configs_keeping_fastest_runtime(tmp_path: Path) -> None:
    _write_graph(
        tmp_path / "graph_1.npz",
        configs=[[1, 2], [4, 4], [1, 2]],
        runtimes=[3_000_000, 2_000_000, 1_000_000],
        normalizers=[3_000_000] * 3,
    )

    (case,) = _dataset(tmp_path).load()

    assert [c.candidate_id for c in case.candidates] == ["config_0", "config_1"]
    assert {r.candidate_id: r.runtime_ms for r in case.runtimes} == {
        "config_0": 1.0,
        "config_1": 2.0,
    }


def test_skips_graphs_with_too_many_candidates(tmp_path: Path) -> None:
    _write_graph(tmp_path / "small_1.npz", configs=[[1], [2]], runtimes=[2, 1])
    _write_graph(tmp_path / "large_1.npz", configs=[[1], [2], [3]], runtimes=[3, 2, 1])

    cases = _dataset(tmp_path, max_candidates=2).load()

    assert [case.workload.workload_id for case in cases] == ["small_1"]


def test_keeps_runtime_labels_out_of_model_visible_data(tmp_path: Path) -> None:
    _write_graph(
        tmp_path / "graph_1.npz",
        configs=[[1], [2]],
        runtimes=[123_456_789, 987_654_321],
    )

    (case,) = _dataset(tmp_path).load()

    visible = json.dumps(
        {
            "context": case.workload.context,
            "candidates": [dict(c.configuration) for c in case.candidates],
        }
    )
    assert "runtime" not in visible
    assert "123456789" not in visible
    assert "987654321" not in visible


def test_rejects_unlabeled_test_split_graphs(tmp_path: Path) -> None:
    _write_graph(tmp_path / "graph_1.npz", configs=[[1], [2]], runtimes=[0, 0])

    with pytest.raises(ValueError, match="the test split has no labels"):
        _dataset(tmp_path).load()


def test_rejects_unknown_config_fields(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must contain path and max_candidates"):
        TpuGraphsTileDataset({"path": str(tmp_path)})


@pytest.mark.parametrize("max_candidates", [0, -1, True, 1.5])
def test_rejects_invalid_max_candidates(tmp_path: Path, max_candidates: object) -> None:
    with pytest.raises(ValueError, match="max_candidates must be a positive integer"):
        TpuGraphsTileDataset({"path": str(tmp_path), "max_candidates": max_candidates})


def test_missing_directory_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _dataset(tmp_path / "missing").load()


def test_rejects_directory_without_npz_files(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"no \.npz files found"):
        _dataset(tmp_path).load()
