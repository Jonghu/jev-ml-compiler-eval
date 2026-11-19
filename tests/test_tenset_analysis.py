import csv
import hashlib
import json
import shutil
from pathlib import Path

import pytest
import yaml

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


SELECTION_METRICS = ("top_1_score", "top_5_score", "runtime_regret")


def _selection_experiment(run):
    (run / "experiment.yaml").write_text(
        yaml.safe_dump({"metrics": list(SELECTION_METRICS)}), encoding="utf-8"
    )


def _predictions(run):
    return [
        json.loads(line)
        for line in (run / "predictions.jsonl").read_text().splitlines()
    ]


def test_tournament_analysis_uses_selection_metrics_and_request_accounting(tmp_path):
    pytest.importorskip("matplotlib")
    data, run = write_inputs(tmp_path)
    _selection_experiment(run)
    predictions = _predictions(run)
    prediction = predictions[1]
    prediction["model_type"] = "jev-tournament"
    prediction["decision_latency_ms"] = 35
    prediction["metadata"] = {
        "probability_scope": "within_question_only",
        # These values deliberately disagree with the total ranking. They are
        # diagnostics from separate questions, not global ranking scores.
        "last_question_probabilities": {"a": 0.01, "b": 0.99, "c": 0.7},
        "request_reports": [
            {
                "cache_hit": True,
                "input_tokens": 100,
                "output_tokens": 20,
                "api_latency_ms": 0.1,
                "request_id": "cached",
            },
            {
                "cache_hit": False,
                "input_tokens": 200,
                "output_tokens": 30,
                "api_latency_ms": 10,
                "request_id": "live-1",
            },
            {
                "cache_hit": False,
                "input_tokens": 300,
                "output_tokens": 40,
                "api_latency_ms": 20,
                "request_id": "live-2",
            },
        ],
    }
    write_predictions(run, predictions)
    output = tmp_path / "analysis"
    analyze_tenset([run], data, output)
    aggregate = {row["model_id"]: row for row in read_csv(output / "overall.csv")}
    assert float(aggregate["jev"]["top_1_score"]) == 1
    assert float(aggregate["jev"]["runtime_regret"]) == 0
    assert "pairwise_accuracy" not in aggregate["jev"]
    svg = (output / "comparison.svg").read_text()
    assert "Runtime regret" in svg
    assert "Pairwise Accuracy" not in svg
    usage = {
        row["model_id"]: row for row in read_csv(output / "decision_accounting.csv")
    }["jev"]
    assert usage["cache_hit"] == "mixed"
    assert usage["live_decisions"] == "1"
    assert usage["cached_decisions"] == "0"
    assert usage["live_requests"] == "2"
    assert usage["cached_requests"] == "1"
    assert usage["response_input_tokens"] == "600"
    assert usage["response_output_tokens"] == "90"
    assert usage["live_input_tokens"] == "500"
    assert usage["live_output_tokens"] == "70"
    assert float(usage["live_api_latency_ms"]) == 30
    assert usage["request_id"] == "cached;live-1;live-2"
    summary = json.loads((output / "summary.json").read_text())
    assert summary["metrics"] == list(SELECTION_METRICS)
    assert summary["pairwise_definition"] == "not reported by this experiment"
    assert read_csv(output / "jev_cases.csv")[0]["jev_model"] == "jev"


def test_tournament_analysis_rejects_global_pairwise_metrics(tmp_path):
    data, run = write_inputs(tmp_path)
    predictions = _predictions(run)
    predictions[1]["model_type"] = "jev-tournament"
    predictions[1]["metadata"] = {}
    write_predictions(run, predictions)
    with pytest.raises(ValueError, match="global Pairwise Accuracy"):
        analyze_tenset([run], data, tmp_path / "analysis", create_plots=False)
    assert not (tmp_path / "analysis").exists()


def test_analysis_rejects_pooling_different_declared_metric_sets(tmp_path):
    data, run = write_inputs(tmp_path)
    other = tmp_path / "other-run"
    shutil.copytree(run, other)
    _selection_experiment(other)
    with pytest.raises(ValueError, match="different metric sets"):
        analyze_tenset([run, other], data, tmp_path / "analysis", create_plots=False)
    assert not (tmp_path / "analysis").exists()


def test_analysis_rejects_unsupported_declared_metrics(tmp_path):
    data, run = write_inputs(tmp_path)
    (run / "experiment.yaml").write_text("metrics: [top_1_accuracy]\n")
    with pytest.raises(ValueError, match="supported TenSet metric set"):
        analyze_tenset([run], data, tmp_path / "analysis", create_plots=False)


def test_analysis_validates_saved_metrics_against_recomputation(tmp_path):
    data, run = write_inputs(tmp_path)
    _selection_experiment(run)
    predictions = _predictions(run)
    predictions[0]["metrics"] = {
        "top_1_score": 0.5,
        "top_5_score": 1.75 / 3,
        "runtime_regret": 3,
    }
    write_predictions(run, predictions)
    with pytest.raises(ValueError, match="saved metric does not match"):
        analyze_tenset([run], data, tmp_path / "analysis", create_plots=False)


@pytest.mark.parametrize("enabled", [False, True])
def test_analysis_reconstructs_description_context_before_hash_validation(
    tmp_path, enabled
):
    from jev_compiler_eval.datasets.tenset_descriptions import (
        tenset_configuration_fields,
    )

    data, run = write_inputs(tmp_path)
    prepared = json.loads(data.read_text())["workloads"][0]
    context = {
        **prepared["context"],
        "configuration_fields": tenset_configuration_fields(),
    }
    model_input = json.dumps(
        {
            "workload_id": "w",
            "context": context,
            "candidates": {
                item["candidate_id"]: item["configuration"]
                for item in prepared["candidates"]
            },
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    predictions = _predictions(run)
    for prediction in predictions:
        prediction["model_input_sha256"] = hashlib.sha256(model_input).hexdigest()
        prediction["model_input_bytes"] = len(model_input)
    write_predictions(run, predictions)
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["dataset_provenance"] = {
        "prepared_file_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
        "describe_config_fields": enabled,
    }
    manifest_path.write_text(json.dumps(manifest))
    output = tmp_path / "analysis"
    if enabled:
        analyze_tenset([run], data, output, create_plots=False)
        assert (output / "overall.csv").is_file()
    else:
        with pytest.raises(ValueError, match="input checksum"):
            analyze_tenset([run], data, output, create_plots=False)


def _collection_fixture(data, tmp_path):
    prepared = json.loads(data.read_text())
    prepared["schema_version"] = 1
    workload = prepared["workloads"][0]
    workload["split"] = "test"
    workload["context"]["workload_key"] = '["matmul"]'
    collection = tmp_path / "collection"
    shards = collection / "workloads"
    shards.mkdir(parents=True)
    path = shards / "w.json"
    path.write_text(json.dumps(prepared))
    other = shards / "unused.json"
    other.write_text("This unused shard must never be loaded by the subset analysis.")
    index = collection / "index.json"
    index.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": "tenset_collection",
                "status": "complete",
                "provenance": {"synthetic": True},
                "shards": [
                    {
                        "path": "workloads/w.json",
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "workload_id": "w",
                        "workload_key": '["matmul"]',
                        "split": "test",
                        "candidate_count": 3,
                    },
                    {
                        "path": "workloads/unused.json",
                        "sha256": "0" * 64,
                        "workload_id": "unused",
                        "workload_key": '["unused"]',
                        "split": "test",
                        "candidate_count": 3,
                    },
                ],
            }
        )
    )
    return index, path


def test_collection_analysis_loads_only_referenced_verified_shards(tmp_path):
    data, run = write_inputs(tmp_path)
    index, _ = _collection_fixture(data, tmp_path)
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["dataset_provenance"] = {
        "prepared_file_sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
        "split": "test",
    }
    manifest_path.write_text(json.dumps(manifest))
    output = tmp_path / "analysis"
    analyze_tenset([run], index, output, create_plots=False)
    assert (
        json.loads((output / "summary.json").read_text())["data_status"]
        == "synthetic_fixture"
    )
    assert len(read_csv(output / "per_workload.csv")) == 2


def test_collection_analysis_rejects_a_changed_used_shard(tmp_path):
    data, run = write_inputs(tmp_path)
    index, shard = _collection_fixture(data, tmp_path)
    shard.write_text(shard.read_text() + " ")
    with pytest.raises(ValueError, match="checksum|hash|SHA"):
        analyze_tenset([run], index, tmp_path / "analysis", create_plots=False)
    assert not (tmp_path / "analysis").exists()
