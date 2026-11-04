import csv
import hashlib
import json
from pathlib import Path

import pytest

from jev_compiler_eval.evaluation.tenset_analysis import analyze_tenset


def write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    data = tmp_path / "prepared.json"
    data.write_text(
        json.dumps(
            {
                "dataset": "tenset",
                "provenance": {"synthetic": True},
                "workloads": [
                    {
                        "workload_id": "w",
                        "context": {"operation": "matmul"},
                        "candidates": [
                            {"candidate_id": "a", "configuration": {}, "runtime_ms": 1},
                            {"candidate_id": "b", "configuration": {}, "runtime_ms": 2},
                            {"candidate_id": "c", "configuration": {}, "runtime_ms": 4},
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    result_dir = tmp_path / "run"
    result_dir.mkdir()
    (result_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_type": "tenset",
                "experiment_name": "synthetic-smoke",
                "seed": 42,
                "model_ids": ["random", "jev"],
            }
        ),
        encoding="utf-8",
    )
    predictions = [
        {
            "model_id": "random",
            "model_type": "random",
            "workload_id": "w",
            "candidate_ids": ["a", "b", "c"],
            "predicted_ranking": ["c", "b", "a"],
            "decision_latency_ms": 1,
            "metadata": {},
        },
        {
            "model_id": "jev",
            "model_type": "jev",
            "workload_id": "w",
            "candidate_ids": ["a", "b", "c"],
            "predicted_ranking": ["a", "b", "c"],
            "decision_latency_ms": 3,
            "metadata": {
                "cache_hit": True,
                "input_tokens": 100,
                "output_tokens": 20,
                "probabilities": {"a": 0.6, "b": 0.3, "c": 0.1},
            },
        },
    ]
    write_predictions(result_dir, predictions)
    return data, result_dir


def write_predictions(result_dir: Path, predictions: list[dict]) -> None:
    (result_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in predictions), encoding="utf-8"
    )


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def test_analysis_recomputes_official_metrics_and_labels_synthetic_data(
    tmp_path,
) -> None:
    data, run = write_inputs(tmp_path)
    output = tmp_path / "analysis"
    assert analyze_tenset([run], data, output, create_plots=False) == output
    aggregate = {row["model_id"]: row for row in read_csv(output / "aggregate.csv")}
    assert float(aggregate["random"]["top_1_score"]) == 0.25
    assert float(aggregate["random"]["top_5_score"]) == pytest.approx(1.75 / 3)
    assert float(aggregate["jev"]["top_1_score"]) == 1
    assert float(aggregate["jev"]["pairwise_accuracy"]) == 1
    assert read_csv(output / "by_operation.csv")[0]["operation"] == "matmul"
    summary = json.loads((output / "summary.json").read_text())
    assert summary["data_status"] == "synthetic_fixture"
    assert len(summary["prepared_sha256"]) == 64
    examples = read_csv(output / "jev_cases.csv")
    assert float(examples[0]["top_1_score_delta"]) == 0.75


def test_accounting_excludes_cached_response_tokens_from_live_usage(tmp_path) -> None:
    data, run = write_inputs(tmp_path)
    output = tmp_path / "analysis"
    analyze_tenset([run], data, output, create_plots=False)
    rows = {
        row["model_id"]: row for row in read_csv(output / "decision_accounting.csv")
    }
    assert rows["jev"]["response_input_tokens"] == "100"
    assert rows["jev"]["live_input_tokens"] == "0"
    assert rows["jev"]["live_decisions"] == "0"
    assert rows["jev"]["cached_decisions"] == "1"
    assert rows["jev"]["monetary_cost_usd"] == ""


@pytest.mark.parametrize(
    "mutation", ["candidate_set", "candidate_order", "ranking", "model", "runtime"]
)
def test_analysis_rejects_invalid_or_unfair_comparisons(tmp_path, mutation) -> None:
    data, run = write_inputs(tmp_path)
    predictions = [
        json.loads(line)
        for line in (run / "predictions.jsonl").read_text().splitlines()
    ]
    if mutation == "candidate_set":
        predictions[1]["candidate_ids"] = ["a", "b"]
    elif mutation == "candidate_order":
        predictions[1]["candidate_ids"] = ["b", "a", "c"]
    elif mutation == "ranking":
        predictions[1]["predicted_ranking"] = ["a", "b"]
    elif mutation == "model":
        predictions[1]["model_id"] = "unlisted"
    else:
        predictions[1]["runtime_by_candidate_ms"] = {"a": 999, "b": 2, "c": 4}
    write_predictions(run, predictions)
    with pytest.raises(ValueError):
        analyze_tenset([run], data, tmp_path / "analysis", create_plots=False)
    assert not (tmp_path / "analysis").exists()


def test_analysis_refuses_to_overwrite_results_or_duplicate_bundles(tmp_path) -> None:
    data, run = write_inputs(tmp_path)
    output = tmp_path / "analysis"
    analyze_tenset([run], data, output, create_plots=False)
    with pytest.raises(FileExistsError):
        analyze_tenset([run], data, output, create_plots=False)
    with pytest.raises(ValueError, match="distinct"):
        analyze_tenset([run, run], data, tmp_path / "second", create_plots=False)


@pytest.mark.parametrize(
    "mutation", ["prepared_checksum", "input_checksum", "input_size", "split"]
)
def test_analysis_validates_saved_provenance_when_available(tmp_path, mutation) -> None:
    data, run = write_inputs(tmp_path)
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["dataset_provenance"] = {
        "prepared_file_sha256": hashlib.sha256(data.read_bytes()).hexdigest()
    }
    if mutation == "prepared_checksum":
        manifest["dataset_provenance"]["prepared_file_sha256"] = "wrong"
    elif mutation == "split":
        manifest["dataset_provenance"]["split"] = "test"
    else:
        predictions = [
            json.loads(line)
            for line in (run / "predictions.jsonl").read_text().splitlines()
        ]
        if mutation == "input_checksum":
            predictions[0]["model_input_sha256"] = "wrong"
        else:
            predictions[0]["model_input_bytes"] = 0
        write_predictions(run, predictions)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        analyze_tenset([run], data, tmp_path / "analysis", create_plots=False)
    assert not (tmp_path / "analysis").exists()


def test_analysis_generates_exportable_plots_when_installed(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    data, run = write_inputs(tmp_path)
    output = tmp_path / "analysis"
    analyze_tenset([run], data, output)
    assert (output / "comparison.png").read_bytes().startswith(b"\x89PNG")
    svg = (output / "comparison.svg").read_text()
    assert "<svg" in svg
    assert "Synthetic software test — not benchmark results" in svg
