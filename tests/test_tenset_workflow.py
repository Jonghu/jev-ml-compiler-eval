"""Exercise preparation, the shared CLI, and analysis without external services."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run(script, *arguments):
    completed = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / script), *map(str, arguments)],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        env={**os.environ, "MPLCONFIGDIR": "/tmp/tenset-workflow-matplotlib"},
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_tenset_cli_workflow_preserves_fair_cases_and_synthetic_status(tmp_path):
    raw = tmp_path / "raw" / "measure_records" / "e5-2673"
    raw.mkdir(parents=True)
    records = []
    for operation in ("synthetic_matmul", "synthetic_conv"):
        for tile in (1, 2, 4, 8, 16, 32):
            records.append(
                {
                    "i": [
                        [json.dumps([operation, [32, 32]]), "llvm"],
                        [[], [["SP", 0, 0, 32, [tile], 1]]],
                    ],
                    "r": [[0.001 * (33 - tile)], 0, 0, 0],
                    "v": "v0.6",
                }
            )
    (raw / "fixture.json").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n"
    )
    prepared = tmp_path / "prepared.json"
    _run(
        "prepare_tenset.py",
        "--raw-root",
        tmp_path / "raw",
        "--target",
        "llvm",
        "--output",
        prepared,
        "--synthetic",
    )
    output = tmp_path / "run"
    experiment = yaml.safe_load(
        (PROJECT_ROOT / "experiments/tenset/random.yaml").read_text()
    )
    experiment["experiment"]["output_dir"] = str(output)
    experiment["dataset"]["config"].update({"path": str(prepared), "split": "all"})
    experiment["models"].append(
        {"id": "random-7", "type": "random", "config": {"seed": 7}}
    )
    config = tmp_path / "experiment.yaml"
    config.write_text(yaml.safe_dump(experiment))
    _run("run_experiment.py", "--config", config)
    predictions = [
        json.loads(line)
        for line in (output / "predictions.jsonl").read_text().splitlines()
    ]
    assert len(predictions) == 4
    for workload_id in {prediction["workload_id"] for prediction in predictions}:
        paired = [p for p in predictions if p["workload_id"] == workload_id]
        assert paired[0]["candidate_ids"] == paired[1]["candidate_ids"]
        assert paired[0]["model_input_sha256"] == paired[1]["model_input_sha256"]
        for prediction in paired:
            metrics = prediction["metrics"]
            assert metrics["top_1_score"] == pytest.approx(
                1 / (1 + metrics["runtime_regret"])
            )
            assert len(prediction["runtime_by_candidate_ms"]) == 6
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["dataset_provenance"]["provenance"]["synthetic"] is True
    assert manifest["package_versions"]["typesafe-sdk"]
    analysis = tmp_path / "analysis"
    _run(
        "analyze_tenset.py",
        "--results",
        output,
        "--dataset",
        prepared,
        "--output-dir",
        analysis,
        "--no-plots",
    )
    summary = json.loads((analysis / "summary.json").read_text())
    assert summary["data_status"] == "synthetic_fixture"
    assert (analysis / "aggregate.csv").is_file()
