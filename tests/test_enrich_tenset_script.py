"""Enrichment changes metadata without changing synthetic evaluation cases."""

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/enrich_tenset.py"
SEMANTIC_FIELDS = {"operation", "tensors", "compute_dag", "loop_structure"}


def fixture_files(tmp_path: Path):
    prepared = {
        "schema_version": 1,
        "dataset": "tenset",
        "provenance": {
            "synthetic": True,
            "seed": 42,
            "sources": [{"path": "synthetic-only.json", "sha256": "not-a-benchmark"}],
        },
        "workloads": [],
    }
    metadata = {}
    for index, split in enumerate(("test", "train", "validation")):
        key = json.dumps([f"synthetic-key-{index}", [8, 8]])
        prepared["workloads"].append(
            {
                "workload_id": f"synthetic-workload-{index}",
                "split": split,
                "context": {
                    "format": "tvm-auto-scheduler-json",
                    "workload_key": key,
                    "target": "llvm -model=synthetic",
                    "operation": "unknown",
                    "tensors": [],
                    "hardware_params": [8, 64, 64, 0, 0, 0, 0, 0],
                },
                "candidates": [
                    {
                        "candidate_id": f"synthetic-schedule-{factor}",
                        "configuration": {
                            "initial_state": [],
                            "transformations": [
                                {
                                    "code": "SP",
                                    "name": "split",
                                    "arguments": [0, 0, 8, [factor], 1],
                                }
                            ],
                        },
                        "runtime_ms": factor * 0.3,
                        "record_input": [
                            [key, "llvm -model=synthetic"],
                            [[], [["SP", 0, 0, 8, [factor], 1]]],
                        ],
                        "record_version": "v0.6",
                    }
                    for factor in (2, 4, 8)
                ],
            }
        )
        metadata[key] = {
            "operation": f"synthetic_matmul_{index}",
            "tensors": [{"name": "A", "shape": [8, 8], "dtype": "float32"}],
            "compute_dag": "C[i,j] = sum(A[i,k] * B[k,j], k=0..7)",
            "loop_structure": [{"operation": "C", "spatial": [8, 8], "reduction": [8]}],
        }
    prepared_path = tmp_path / "prepared-pilot.json"
    metadata_path = tmp_path / "metadata.json"
    prepared_path.write_text(json.dumps(prepared), encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return prepared_path, metadata_path, prepared, metadata


def run_enrichment(prepared: Path, metadata: Path, output: Path):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--prepared",
            str(prepared),
            "--metadata",
            str(metadata),
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )


def test_enrichment_preserves_ids_splits_candidates_labels_and_order(tmp_path: Path):
    prepared_path, metadata_path, prepared, metadata = fixture_files(tmp_path)
    before_bytes = prepared_path.read_bytes()
    output = tmp_path / "enriched/prepared.json"
    result = run_enrichment(prepared_path, metadata_path, output)
    assert result.returncode == 0, result.stderr
    enriched = json.loads(output.read_text())
    assert prepared_path.read_bytes() == before_bytes
    assert enriched["dataset"] == prepared["dataset"]
    assert enriched["schema_version"] == prepared["schema_version"]
    for before, after in zip(prepared["workloads"], enriched["workloads"], strict=True):
        assert before["workload_id"] == after["workload_id"]
        assert before["split"] == after["split"]
        assert before["candidates"] == after["candidates"]
        key = before["context"]["workload_key"]
        expected = copy.deepcopy(before["context"])
        expected.update(metadata[key])
        assert after["context"] == expected
    provenance = enriched["provenance"]
    assert (
        provenance["parent_prepared_sha256"] == hashlib.sha256(before_bytes).hexdigest()
    )
    assert (
        provenance["metadata_sha256"]
        == hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    )
    assert provenance["metadata_enrichment"]["workloads"] == 3
    assert set(provenance["metadata_enrichment"]["fields"]) == SEMANTIC_FIELDS
    for key, value in prepared["provenance"].items():
        assert provenance[key] == value


def test_enrichment_refuses_existing_output_and_preserves_it(tmp_path: Path):
    prepared, metadata, _, _ = fixture_files(tmp_path)
    output = tmp_path / "existing.json"
    output.write_text("keep existing", encoding="utf-8")
    result = run_enrichment(prepared, metadata, output)
    assert result.returncode != 0
    assert "already exists" in result.stderr
    assert output.read_text() == "keep existing"
    alias = tmp_path / "alias.json"
    alias.symlink_to(output)
    result = run_enrichment(prepared, metadata, alias)
    assert result.returncode != 0
    assert "already exists" in result.stderr
    assert output.read_text() == "keep existing"


@pytest.mark.parametrize(
    "invalid", ["missing_workload", "missing_fields", "labels", "nested_labels"]
)
def test_enrichment_rejects_incomplete_or_label_bearing_metadata(
    tmp_path: Path, invalid
):
    prepared_path, metadata_path, _, metadata = fixture_files(tmp_path)
    key = next(iter(metadata))
    if invalid == "missing_workload":
        del metadata[key]
    elif invalid == "missing_fields":
        del metadata[key]["compute_dag"]
    elif invalid == "labels":
        metadata[key]["runtime_ms"] = 1.5
    else:
        metadata[key]["tensors"][0]["runtime_ms"] = 1.5
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    output = tmp_path / "enriched.json"
    before = prepared_path.read_bytes()
    result = run_enrichment(prepared_path, metadata_path, output)
    assert result.returncode != 0
    assert any(
        word in result.stderr for word in ("missing", "incomplete", "label-bearing")
    )
    assert not output.exists()
    assert prepared_path.read_bytes() == before
