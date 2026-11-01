import json
from pathlib import Path

import pytest

from jev_compiler_eval.datasets import MockDataset, get_dataset_class

PROJECT_ROOT = Path(__file__).parent.parent
MOCK_DATASET_PATH = PROJECT_ROOT / "examples" / "mock_dataset.json"


def test_mock_dataset_is_registered() -> None:
    assert get_dataset_class("mock") is MockDataset


def test_loads_committed_mock_dataset() -> None:
    cases = MockDataset({"path": str(MOCK_DATASET_PATH)}).load()

    assert len(cases) == 2
    assert cases[0].workload.workload_id == "mock_matmul_001"
    assert cases[0].workload.context["operation"] == "matmul"
    assert cases[1].workload.workload_id == "mock_conv2d_001"
    assert cases[1].workload.context["operation"] == "conv2d"
    assert all(len(case.candidates) == 3 for case in cases)
    assert all(len(case.runtimes) == 3 for case in cases)


def test_keeps_runtime_labels_out_of_model_visible_data() -> None:
    cases = MockDataset({"path": str(MOCK_DATASET_PATH)}).load()

    for case in cases:
        assert "runtime_ms" not in case.workload.context
        for candidate in case.candidates:
            assert "runtime_ms" not in candidate.configuration


def test_rejects_unknown_config_fields() -> None:
    with pytest.raises(
        ValueError, match='mock dataset config must contain only "path"'
    ):
        MockDataset({"path": str(MOCK_DATASET_PATH), "split": "test"})


def test_missing_dataset_file_raises_file_not_found(tmp_path: Path) -> None:
    dataset = MockDataset({"path": str(tmp_path / "missing.json")})

    with pytest.raises(FileNotFoundError):
        dataset.load()


def test_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text("{", encoding="utf-8")

    with pytest.raises(ValueError, match="failed to parse mock dataset"):
        MockDataset({"path": str(path)}).load()


def test_rejects_missing_workloads(tmp_path: Path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps({"dataset": "mock"}), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=r"mock dataset root is missing required field\(s\): workloads",
    ):
        MockDataset({"path": str(path)}).load()
