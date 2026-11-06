"""Portable metadata export/enrichment with native-generated tensor programs."""

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from jev_compiler_eval.config import load_experiment_config
from jev_compiler_eval.datasets.tenset import TenSetDataset, prepare_records

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/tenset_registry"


def _run(script, *arguments):
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *map(str, arguments)],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )


@pytest.fixture
def prepared(tmp_path):
    references = json.loads((FIXTURES / "native_reference.json").read_text())["cases"]
    selected = [
        entry for entry in references if entry["name"] in {"matmul", "shared_branch"}
    ]
    records = []
    for entry in selected:
        for factor in range(1, 9):
            records.append(
                {
                    "i": [
                        [entry["workload_key"], "llvm"],
                        [[], [["SP", 0, 0, 8, [factor], 1]]],
                    ],
                    "r": [[factor * 0.001], 0, 0, 0],
                    "v": "v0.6",
                }
            )
    source = tmp_path / "synthetic-schedule-records.jsonl"
    source.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    data = prepare_records([source], target="llvm", min_candidates=8)
    data["provenance"]["synthetic"] = True
    path = tmp_path / "prepared.json"
    path.write_text(json.dumps(data))
    return path, data


def test_portable_cli_exports_only_selected_real_program_graphs(tmp_path, prepared):
    path, data = prepared
    output = tmp_path / "metadata.json"
    result = _run(
        "export_tenset_metadata.py",
        "--task-registry",
        FIXTURES / "native_registry.pkl",
        "--prepared",
        path,
        "--output",
        output,
    )
    assert result.returncode == 0, result.stderr
    assert "no TVM or XGBoost required" in result.stdout
    metadata = json.loads(output.read_text())
    assert set(metadata) == {
        entry["context"]["workload_key"] for entry in data["workloads"]
    }
    for value in metadata.values():
        assert value["tensors"]
        assert value["compute_dag"]["stage_order"]
        assert value["compute_dag"]["format"] == "tvm-te-expression-graph-v1"
    manifest = json.loads(Path(str(output) + ".manifest.json").read_text())
    assert manifest["exported_workloads"] == 2
    assert manifest["registry_workloads"] == 13
    assert manifest["prepared_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert (
        manifest["metadata_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    )
    before = output.read_bytes()
    repeated = _run(
        "export_tenset_metadata.py",
        "--task-registry",
        FIXTURES / "native_registry.pkl",
        "--output",
        output,
    )
    assert repeated.returncode != 0
    assert output.read_bytes() == before


def test_complete_metadata_workflow_preserves_cases_and_passes_preflight(
    tmp_path, prepared
):
    path, original = prepared
    metadata = tmp_path / "metadata.json"
    result = _run(
        "export_tenset_metadata.py",
        "--task-registry",
        FIXTURES / "native_registry.pkl",
        "--prepared",
        path,
        "--output",
        metadata,
    )
    assert result.returncode == 0, result.stderr
    config = {
        "path": str(path),
        "split": "all",
        "candidate_limit": 8,
        "require_program_metadata": True,
    }
    with pytest.raises(ValueError, match="complete program metadata is required"):
        TenSetDataset(config).load()
    enriched = tmp_path / "prepared-metadata.json"
    result = _run(
        "enrich_tenset.py",
        "--prepared",
        path,
        "--metadata",
        metadata,
        "--output",
        enriched,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(enriched.read_text())
    for before, after in zip(original["workloads"], data["workloads"], strict=True):
        assert before["workload_id"] == after["workload_id"]
        assert before["split"] == after["split"]
        assert before["candidates"] == after["candidates"]
    config["path"] = str(enriched)
    cases = TenSetDataset(config).load()
    assert len(cases) == 2
    assert all(case.workload.context["operation"] != "unknown" for case in cases)


def test_missing_selected_program_never_creates_metadata(tmp_path, prepared):
    path, data = prepared
    mutated = copy.deepcopy(data)
    mutated["workloads"][0]["context"]["workload_key"] = '["unavailable"]'
    path.write_text(json.dumps(mutated))
    output = tmp_path / "metadata.json"
    result = _run(
        "export_tenset_metadata.py",
        "--task-registry",
        FIXTURES / "native_registry.pkl",
        "--prepared",
        path,
        "--output",
        output,
    )
    assert result.returncode != 0
    assert "missing 1 prepared workload" in result.stderr
    assert not output.exists()


def test_committed_jev_comparison_needs_no_xgboost_checkpoint():
    config = load_experiment_config(ROOT / "experiments/tenset/jev-vs-random.yaml")
    assert [model.model_type for model in config.models] == ["random", "jev"]
    assert config.dataset.config["require_program_metadata"] is True
    assert config.dataset.config["max_workloads"] == 1


def test_metadata_requirement_flag_must_be_boolean(prepared):
    path, _ = prepared
    with pytest.raises(ValueError, match="must be a boolean"):
        TenSetDataset({"path": str(path), "require_program_metadata": "yes"})
