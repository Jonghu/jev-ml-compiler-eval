"""Checks for the official-model boundary without requiring a historical TVM build."""

import ast
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import jev_compiler_eval.models.tenset_xgb as adapter
from jev_compiler_eval.models.tenset_xgb import (
    IMPLEMENTATION,
    TenSetXGBModel,
    file_sha256,
)
from jev_compiler_eval.types import Candidate, Workload


def _configuration(index):
    return {
        "initial_state": [],
        "transformations": [{"code": "SP", "name": "split", "arguments": [index]}],
    }


@pytest.fixture
def artifacts(tmp_path):
    prepared = tmp_path / "prepared.json"
    checkpoint = tmp_path / "model.pkl"
    registry = tmp_path / "all_tasks.pkl"
    checkpoint.write_bytes(b"fake trusted local checkpoint")
    registry.write_bytes(b"fake trusted local registry")
    entries = []
    for workload_id, split in (("train-w", "train"), ("test-w", "test")):
        entries.append(
            {
                "workload_id": workload_id,
                "split": split,
                "context": {"target": "llvm", "workload_key": workload_id},
                "candidates": [
                    {
                        "candidate_id": candidate_id,
                        "configuration": _configuration(index),
                        "runtime_ms": 1000.0 + index,
                        "record_input": [[workload_id, "llvm"], [[], [["SP", index]]]],
                        "record_version": "v0.6",
                    }
                    for index, candidate_id in enumerate(("a", "b", "c"))
                ],
            }
        )
    prepared.write_text(json.dumps({"dataset": "tenset", "workloads": entries}))
    provenance = {
        "implementation": IMPLEMENTATION,
        "checkpoint_sha256": file_sha256(checkpoint),
        "prepared_sha256": file_sha256(prepared),
        "task_registry_sha256": file_sha256(registry),
        "training_workload_ids": ["train-w"],
    }
    Path(str(checkpoint) + ".json").write_text(json.dumps(provenance))
    return {
        "prepared_path": str(prepared),
        "checkpoint_path": str(checkpoint),
        "task_registry_path": str(registry),
        "python_executable": sys.executable,
    }


def _case(workload_id="test-w"):
    return (
        Workload(workload_id, {"target": "llvm", "workload_key": workload_id}),
        tuple(
            Candidate(candidate_id, _configuration(index))
            for index, candidate_id in enumerate(("a", "b", "c"))
        ),
    )


def _fake_worker(tmp_path, monkeypatch, scores=(0.2, 0.8, 0.8)):
    worker = tmp_path / "worker.py"
    payload = json.dumps({"scores": list(scores), "feature_prediction_ms": 2.0})
    worker.write_text(
        "import json, sys\n"
        "from pathlib import Path\n"
        "print(json.dumps({'status':'ready','versions':{'tvm':'fake'}}), flush=True)\n"
        "for line in sys.stdin:\n"
        "    with Path(__file__).with_suffix('.requests').open('a') as f:\n"
        "        f.write(line)\n"
        f"    print({payload!r}, flush=True)\n"
    )
    monkeypatch.setattr(adapter, "WORKER_PATH", worker)
    return worker.with_suffix(".requests")


def test_persistent_worker_receives_no_labels_and_breaks_ties_stably(
    artifacts, tmp_path, monkeypatch
):
    requests_path = _fake_worker(tmp_path, monkeypatch)
    model = TenSetXGBModel(artifacts)
    try:
        first = model.rank(*_case())
        process = model._process
        second = model.rank(*_case())
        assert model._process is process
        assert first.ranked_candidate_ids == ("b", "c", "a")
        assert second.ranked_candidate_ids == first.ranked_candidate_ids
        assert first.metadata["scores"] == {"a": 0.2, "b": 0.8, "c": 0.8}
        assert first.metadata["worker_initialization_ms"] > 0
        assert second.metadata["worker_initialization_ms"] == 0
        requests = [json.loads(line) for line in requests_path.read_text().splitlines()]
        assert len(requests) == 2
        assert requests[0]["records"] == requests[1]["records"]
        for record in requests[0]["records"]:
            assert set(record) == {"i", "v"}
            assert record["i"][0] == ["test-w", "llvm"]
    finally:
        model.close()
    assert process.poll() is not None
    model.close()


def test_train_overlap_and_candidate_mismatch_fail_before_worker(artifacts):
    model = TenSetXGBModel(artifacts)
    try:
        with pytest.raises(ValueError, match="training workload"):
            model.rank(*_case("train-w"))
        workload, candidates = _case()
        with pytest.raises(ValueError, match="candidate does not match"):
            model.rank(workload, (Candidate("a", {"tile": 999}), *candidates[1:]))
        with pytest.raises(ValueError, match="nonempty, unique"):
            model.rank(workload, (candidates[0], candidates[0]))
        assert model._process is None
    finally:
        model.close()


@pytest.mark.parametrize("corrupt", ["workload", "schedule"])
def test_hidden_measure_input_must_match_visible_case(artifacts, corrupt):
    prepared_path = Path(artifacts["prepared_path"])
    prepared = json.loads(prepared_path.read_text())
    candidate = prepared["workloads"][1]["candidates"][0]
    if corrupt == "workload":
        candidate["record_input"][0][0] = "another-workload"
    else:
        candidate["record_input"][1][1][0][1] = 999
    prepared_path.write_text(json.dumps(prepared))
    provenance_path = Path(artifacts["checkpoint_path"] + ".json")
    provenance = json.loads(provenance_path.read_text())
    provenance["prepared_sha256"] = file_sha256(prepared_path)
    provenance_path.write_text(json.dumps(provenance))
    model = TenSetXGBModel(artifacts)
    try:
        with pytest.raises(ValueError, match="hidden TVM input does not match"):
            model.rank(*_case())
        assert model._process is None
    finally:
        model.close()


def test_workload_key_overlap_is_rejected_even_with_distinct_ids(artifacts):
    prepared_path = Path(artifacts["prepared_path"])
    prepared = json.loads(prepared_path.read_text())
    prepared["workloads"][1]["context"]["workload_key"] = "train-w"
    prepared_path.write_text(json.dumps(prepared))
    provenance_path = Path(artifacts["checkpoint_path"] + ".json")
    provenance = json.loads(provenance_path.read_text())
    provenance["prepared_sha256"] = file_sha256(prepared_path)
    provenance_path.write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="workload keys overlap"):
        TenSetXGBModel(artifacts)


@pytest.mark.parametrize(
    "field", ["checkpoint_path", "prepared_path", "task_registry_path"]
)
def test_tampered_artifact_is_rejected(artifacts, field):
    path = Path(artifacts[field])
    if field == "prepared_path":
        path.write_text(path.read_text() + " ")
    else:
        path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="does not match its artifact"):
        TenSetXGBModel(artifacts)


def test_worker_nonfinite_scores_are_rejected(artifacts, tmp_path, monkeypatch):
    _fake_worker(tmp_path, monkeypatch, scores=(0.2, float("inf"), 0.3))
    model = TenSetXGBModel(artifacts)
    try:
        with pytest.raises(ValueError, match="nonfinite"):
            model.rank(*_case())
    finally:
        model.close()


def test_worker_missing_tvm_has_actionable_error(artifacts, tmp_path):
    # Empty import path plus a fake tvm package guarantees the check is independent
    # of whether the host happens to have another version of TVM installed.
    fake_modules = tmp_path / "imports"
    fake_modules.mkdir()
    (fake_modules / "tvm.py").write_text("raise ImportError('test missing TVM')")
    import os
    import subprocess

    completed = subprocess.run(
        [
            sys.executable,
            str(adapter.WORKER_PATH),
            "serve",
            "--checkpoint",
            artifacts["checkpoint_path"],
            "--task-registry",
            artifacts["task_registry_path"],
        ],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(fake_modules)},
    )
    assert completed.returncode == 1
    assert "original tlc-pack/tenset TVM fork" in json.loads(completed.stdout)["error"]


def _worker_module():
    spec = importlib.util.spec_from_file_location(
        "tenset_worker_under_test", adapter.WORKER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_training_uses_only_training_labels_and_saves_provenance(
    artifacts, tmp_path, monkeypatch
):
    worker = _worker_module()
    calls = []

    class FakeModel:
        def __init__(self, **kwargs):
            self.model = SimpleNamespace(xgb_params={})

        def update(self, inputs, results):
            calls.append((inputs, results))

        def save(self, path):
            Path(path).write_bytes(b"trained")

        def load(self, path):
            calls.append(("reload", path))

    def read_record(value):
        record = json.loads(value)
        return record["i"], record["r"]

    monkeypatch.setattr(
        worker,
        "_dependencies",
        lambda _: (FakeModel, read_record, None, {"tvm": "fake"}),
    )
    checkpoint = tmp_path / "new-model.pkl"
    args = SimpleNamespace(
        prepared=artifacts["prepared_path"],
        checkpoint=str(checkpoint),
        task_registry=artifacts["task_registry_path"],
        seed=0,
    )
    provenance = worker._train(args)
    assert provenance["training_workload_ids"] == ["train-w"]
    assert provenance["training_candidates"] == 3
    inputs, results = calls[0]
    assert all(inp[0][0] == "train-w" for inp in inputs)
    assert results[0][0] == [1.0]
    assert calls[1] == ("reload", str(checkpoint))
    assert provenance["checkpoint_sha256"] == file_sha256(checkpoint)
    assert Path(str(checkpoint) + ".json").is_file()
    with pytest.raises(FileExistsError):
        worker._train(args)


def test_inference_record_uses_dummy_runtime_even_if_label_present():
    worker = _worker_module()
    raw = json.loads(
        worker._measure_record(
            {"i": ["state"], "v": "v0.6", "runtime_ms": 999999}, False
        )
    )
    assert raw == {"i": ["state"], "v": "v0.6", "r": [[1.0], 0, 0.0, 0.0]}


def test_training_rejects_input_that_describes_another_schedule(artifacts):
    worker = _worker_module()
    entry = json.loads(Path(artifacts["prepared_path"]).read_text())["workloads"][0]
    candidate = entry["candidates"][0]
    worker._validate_training_input(entry, candidate)
    candidate["record_input"][1][1][0][1] = 999
    with pytest.raises(ValueError, match="visible candidate"):
        worker._validate_training_input(entry, candidate)


def test_metadata_exports_shapes_all_loop_axes_and_untruncated_dag(
    tmp_path, monkeypatch
):
    worker = _worker_module()
    spatial = SimpleNamespace(var="i", dom=SimpleNamespace(extent=64))
    reduction = SimpleNamespace(var="k", dom=SimpleNamespace(extent=128))
    operation = SimpleNamespace(name="dense", axis=[spatial], reduce_axis=[reduction])
    tensor = SimpleNamespace(op=operation, shape=[64, 128], dtype="float32")
    dag = SimpleNamespace(tensors=[tensor], ops=[operation])
    task = SimpleNamespace(workload_key='["hash",64,128]', compute_dag=dag)
    complete_dag = "dense(i,j) += input(i,k)*weight(k,j)" + " " * 1000
    tvm = SimpleNamespace(get_global_func=lambda _: lambda dag, simple: complete_dag)
    monkeypatch.setattr(worker, "_register_tasks", lambda _: ([task], tvm))
    output = tmp_path / "metadata.json"
    result = worker._metadata(
        SimpleNamespace(task_registry="trusted.pkl", output=str(output))
    )
    metadata = json.loads(output.read_text())
    description = metadata[task.workload_key]
    assert result["workloads"] == 1
    assert set(description) == {"operation", "tensors", "compute_dag", "loop_structure"}
    assert description["compute_dag"] == complete_dag
    assert description["tensors"][0]["shape"] == [64, 128]
    assert description["loop_structure"][0]["reduction"] == [
        {"variable": "k", "extent": 128}
    ]


def test_standalone_worker_uses_python38_syntax():
    ast.parse(adapter.WORKER_PATH.read_text(), feature_version=(3, 8))
