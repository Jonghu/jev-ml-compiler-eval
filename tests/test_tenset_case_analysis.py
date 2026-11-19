"""Offline case reports verify frozen pools and do not rerun decision models."""

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from jev_compiler_eval.datasets.tenset import _digest
from jev_compiler_eval.evaluation import tenset_analysis as generic
from jev_compiler_eval.evaluation.tenset_case_analysis import analyze_tenset_cases
from jev_compiler_eval.metrics import resolve_metrics
from jev_compiler_eval.types import DecisionResult


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_predictions(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def read_csv(path):
    return list(csv.DictReader(path.open()))


@pytest.fixture
def inputs(tmp_path):
    def make(*, collection=False, official=True, missing_metadata=False):
        data = tmp_path / ("index.json" if collection else "prepared.json")
        workloads = []
        for number, count in enumerate((2, 4)):
            candidates = []
            for index in range(count):
                candidate = {
                    "candidate_id": f"record-{index}",
                    "runtime_ms": float(2**index),
                    "configuration": {
                        "initial_state": [],
                        "transformations": [
                            {"code": "SP", "name": "split", "arguments": [index, 1]}
                        ],
                    },
                }
                if official:
                    # The full-task oracle is absent from this test pool.
                    candidate["official_throughput"] = 0.5 / candidate["runtime_ms"]
                    candidate["source_error_code"] = 0
                    candidate["source_record_index"] = index
                candidates.append(candidate)
            context = {
                "format": "tvm-auto-scheduler-json",
                "workload_key": json.dumps(["a" * 32, number, 16]),
                "target": "llvm -model=synthetic",
                "operation": "conv2d+relu",
                "tensors": [{"name": "input", "shape": [16, 8], "dtype": "float32"}],
                "compute_dag": "Synthetic TE graph for offline boundary testing.",
                "loop_structure": [],
            }
            if missing_metadata:
                context.pop("operation")
                context["workload_key"] = '["not-a-dag-hash",16]'
                context["tensors"] = []
            workloads.append(
                {
                    "workload_id": f"conv2d-looking-id-{number}",
                    "split": "test",
                    "context": context,
                    "candidates": candidates,
                }
            )
        root = {"schema_version": 1, "dataset": "tenset", "workloads": workloads}
        if collection:
            descriptors = []
            (tmp_path / "workloads").mkdir()
            for workload in workloads:
                path = tmp_path / "workloads" / (workload["workload_id"] + ".json")
                path.write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "dataset": "tenset",
                            "workloads": [workload],
                        }
                    )
                )
                descriptors.append(
                    {
                        "workload_id": workload["workload_id"],
                        "workload_key": workload["context"]["workload_key"],
                        "split": "test",
                        "path": "workloads/" + path.name,
                        "sha256": digest(path),
                        "candidate_count": len(workload["candidates"]),
                    }
                )
            root = {
                "schema_version": 1,
                "dataset": "tenset_collection",
                "status": "complete",
                "provenance": {},
                "shards": descriptors,
            }
        data.write_text(json.dumps(root))
        run = tmp_path / "run"
        run.mkdir()
        config = {
            "schema_version": 1,
            "dataset": {
                "type": "tenset",
                "config": {
                    "path": str(data),
                    "split": "test",
                    "seed": 42,
                    "candidate_limit": None,
                    "max_workloads": None,
                    "describe_config_fields": False,
                },
            },
            "models": [
                {"id": "random", "type": "random", "config": {"seed": 42}},
                {"id": "jev", "type": "jev-tournament", "config": {}},
            ],
            "metrics": ["top_1_score", "top_5_score"],
        }
        (run / "experiment.yaml").write_text(yaml.safe_dump(config))
        manifest = {
            "dataset_type": "tenset",
            "experiment_name": "synthetic-cases",
            "seed": 42,
            "model_ids": ["random", "jev"],
            "dataset_provenance": {
                **config["dataset"]["config"],
                "prepared_file_sha256": digest(data),
            },
        }
        (run / "manifest.json").write_text(json.dumps(manifest))
        rows = []
        for model in ("random", "jev"):
            for number, workload in enumerate(workloads):
                candidates = workload["candidates"]
                ordered = sorted(
                    [candidate["candidate_id"] for candidate in candidates],
                    key=lambda key: _digest([42, workload["workload_id"], key]),
                )
                ranking = [candidate["candidate_id"] for candidate in candidates]
                if (model == "jev" and number == 0) or (
                    model == "random" and number == 1
                ):
                    ranking.reverse()
                prediction = {
                    "model_id": model,
                    "model_type": "random" if model == "random" else "jev-tournament",
                    "workload_id": workload["workload_id"],
                    "candidate_ids": ordered,
                    "predicted_ranking": ranking,
                    "selected_candidate_id": ranking[0],
                    "decision_latency_ms": 1,
                    "metadata": {},
                }
                case = generic._case_from_prediction(workload, prediction)
                payload = json.dumps(
                    {
                        "workload_id": workload["workload_id"],
                        "context": dict(case.workload.context),
                        "candidates": {
                            c.candidate_id: dict(c.configuration)
                            for c in case.candidates
                        },
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
                runtimes = {c.candidate_id: c.runtime_ms for c in case.runtimes}
                prediction.update(
                    {
                        "model_input_sha256": hashlib.sha256(payload).hexdigest(),
                        "model_input_bytes": len(payload),
                        "runtime_by_candidate_ms": runtimes,
                        "selected_runtime_ms": runtimes[ranking[0]],
                        "metrics": {
                            name: function(case, DecisionResult(tuple(ranking), 1, {}))
                            for name, function in resolve_metrics(
                                config["metrics"]
                            ).items()
                        },
                    }
                )
                rows.append(prediction)
        write_predictions(run / "predictions.jsonl", rows)
        return data, run, workloads, rows, tmp_path / "analysis"

    return make


@pytest.mark.parametrize("collection", [False, True])
def test_metadata_cases_and_weighting_support_selection_only_tournament(
    inputs, collection
):
    data, run, _, _, output = inputs(collection=collection)
    assert analyze_tenset_cases(run, data, output) == output
    overall = {row["model_id"]: row for row in read_csv(output / "overall.csv")}
    assert float(overall["jev"]["weighted_top_1_score"]) == pytest.approx(5 / 6)
    assert float(overall["jev"]["macro_top_1_score"]) == 0.75
    assert float(overall["random"]["weighted_top_1_score"]) == pytest.approx(5 / 12)
    assert overall["jev"]["test_records"] == "6"
    assert overall["jev"]["score_basis"] == "private_official_throughput"
    assert float(overall["jev"]["weighted_top_5_score"]) == pytest.approx(11 / 12)
    per_workload = read_csv(output / "per_workload.csv")
    assert {row["dag_family_hash"] for row in per_workload} == {"a" * 32}
    assert {row["operation_family"] for row in per_workload} == {"convolution"}
    assert all(row["workload_key"].startswith('["aaaaaaaa') for row in per_workload)
    cases = json.loads((output / "case_studies.json").read_text())
    assert {case["outcome"] for case in cases} == {"success", "failure"}
    success = next(case for case in cases if case["outcome"] == "success")
    assert success["context"]["tensors"][0]["shape"] == [16, 8]
    assert success["jev"]["candidate_id"] == "record-0"
    assert success["random"]["candidate_id"] == "record-3"
    assert success["oracle"]["configuration"]["transformations"][0]["code"] == "SP"
    assert "placeholders" in (output / "case_studies.md").read_text()
    summary = json.loads((output / "analysis-manifest.json").read_text())
    assert summary["api_requests_made"] == 0
    assert summary["global_tournament_pairwise_accuracy"] == "not computed"
    assert "not a controlled size ablation" in summary["candidate_size_comparison"]


def test_missing_metadata_is_not_inferred_from_workload_id(inputs):
    data, run, _, _, output = inputs(missing_metadata=True)
    analyze_tenset_cases(run, data, output)
    rows = read_csv(output / "per_workload.csv")
    assert {row["operation"] for row in rows} == {"unknown"}
    assert {row["operation_family"] for row in rows} == {"unknown"}
    assert {row["dag_family_hash"] for row in rows} == {""}
    assert {row["metadata_complete"] for row in rows} == {"False"}


def test_valid_runtime_only_cohort_has_explicit_fallback_basis(inputs):
    data, run, _, _, output = inputs(official=False)
    analyze_tenset_cases(run, data, output)
    rows = read_csv(output / "overall.csv")
    assert {row["score_basis"] for row in rows} == {"prepared_runtime_ratio"}


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("input_hash", "input checksum"),
        ("input_bytes", "input byte count"),
        ("runtime", "runtimes disagree"),
        ("missing_case", "entire configured workload cohort"),
        ("selected_id", "selected candidate"),
        ("selected_cost", "selected runtime"),
        ("model_type", "model type"),
        ("missing_fingerprint", "requires saved"),
        ("prepared_hash", "checksum"),
        ("settings", "settings differ"),
        ("ranking", "ranking"),
    ],
)
def test_invalid_or_incomplete_result_does_not_publish_output(inputs, mutation, match):
    data, run, _, rows, output = inputs()
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if mutation == "input_hash":
        rows[0]["model_input_sha256"] = "wrong"
    elif mutation == "input_bytes":
        rows[0]["model_input_bytes"] += 1
    elif mutation == "runtime":
        rows[0]["runtime_by_candidate_ms"]["record-0"] = 99
    elif mutation == "missing_case":
        rows = [row for row in rows if row["workload_id"] != rows[0]["workload_id"]]
    elif mutation == "selected_id":
        rows[0]["selected_candidate_id"] = "not-the-selected-one"
    elif mutation == "selected_cost":
        rows[0]["selected_runtime_ms"] += 1
    elif mutation == "model_type":
        rows[0]["model_type"] = "jev"
    elif mutation == "missing_fingerprint":
        rows[0].pop("model_input_sha256")
    elif mutation == "prepared_hash":
        manifest["dataset_provenance"]["prepared_file_sha256"] = "wrong"
    elif mutation == "settings":
        config_path = run / "experiment.yaml"
        config = yaml.safe_load(config_path.read_text())
        config["dataset"]["config"]["seed"] = 43
        config_path.write_text(yaml.safe_dump(config))
    else:
        rows[0]["predicted_ranking"] = rows[0]["predicted_ranking"][:-1]
    write_predictions(run / "predictions.jsonl", rows)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=match):
        analyze_tenset_cases(run, data, output)
    assert not output.exists()


def test_truncated_matching_pools_cannot_masquerade_as_full_candidate_evaluation(
    inputs,
):
    data, run, workloads, rows, output = inputs()
    for row in rows:
        if row["workload_id"] == rows[0]["workload_id"]:
            continue
        remove = "record-1"
        row["candidate_ids"].remove(remove)
        row["predicted_ranking"].remove(remove)
        row["runtime_by_candidate_ms"].pop(remove)
        workload = next(w for w in workloads if w["workload_id"] == row["workload_id"])
        case = generic._case_from_prediction(workload, row)
        payload = json.dumps(
            {
                "workload_id": workload["workload_id"],
                "context": dict(case.workload.context),
                "candidates": {
                    c.candidate_id: dict(c.configuration) for c in case.candidates
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        row["model_input_sha256"] = hashlib.sha256(payload).hexdigest()
        row["model_input_bytes"] = len(payload)
        row["metrics"] = {
            name: function(case, DecisionResult(tuple(row["predicted_ranking"]), 1, {}))
            for name, function in resolve_metrics(list(row["metrics"])).items()
        }
    write_predictions(run / "predictions.jsonl", rows)
    with pytest.raises(ValueError, match="configured frozen pool"):
        analyze_tenset_cases(run, data, output)
    assert not output.exists()


def test_record_error_flags_remain_visible_in_case_reports(inputs):
    data, run, _, _, output = inputs()
    document = json.loads(data.read_text())
    document["workloads"][0]["candidates"][1]["source_error_code"] = 1
    data.write_text(json.dumps(document))
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["dataset_provenance"]["prepared_file_sha256"] = digest(data)
    manifest_path.write_text(json.dumps(manifest))
    analyze_tenset_cases(run, data, output)
    cases = json.loads((output / "case_studies.json").read_text())
    failure = next(case for case in cases if case["outcome"] == "failure")
    assert failure["jev"]["is_failure"] is True
    assert failure["jev"]["source_error_code"] == 1
    assert failure["jev"]["failure_status_basis"] == "source_error_code"
    assert failure["jev"]["suspected_failure_sentinel"] is False
    overall = {row["model_id"]: row for row in read_csv(output / "overall.csv")}
    assert overall["jev"]["selected_failure_records"] == "1"
    assert overall["random"]["selected_failure_records"] == "0"


@pytest.mark.parametrize("label", [0, -1, float("nan"), float("inf"), True])
def test_invalid_private_throughput_labels_fail_without_output(inputs, label):
    data, run, _, _, output = inputs()
    document = json.loads(data.read_text())
    document["workloads"][0]["candidates"][0]["official_throughput"] = label
    data.write_text(json.dumps(document))
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["dataset_provenance"]["prepared_file_sha256"] = digest(data)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="finite and in"):
        analyze_tenset_cases(run, data, output)
    assert not output.exists()


def test_collection_shard_tampering_fails_integrity_validation(inputs):
    data, run, _, _, output = inputs(collection=True)
    index = json.loads(data.read_text())
    shard = data.parent / index["shards"][0]["path"]
    shard.write_text(shard.read_text() + " ")
    with pytest.raises(ValueError, match="checksum"):
        analyze_tenset_cases(run, data, output)


def test_partial_private_label_pool_is_rejected(inputs):
    data, run, _, _, output = inputs()
    document = json.loads(data.read_text())
    document["workloads"][0]["candidates"][0].pop("official_throughput")
    data.write_text(json.dumps(document))
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["dataset_provenance"]["prepared_file_sha256"] = digest(data)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="entire test pool"):
        analyze_tenset_cases(run, data, output)


def test_analysis_does_not_overwrite_prior_outputs(inputs):
    data, run, _, _, output = inputs()
    analyze_tenset_cases(run, data, output)
    with pytest.raises(FileExistsError):
        analyze_tenset_cases(run, data, output)


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_case_budget_must_be_a_positive_integer(inputs, value):
    data, run, _, _, output = inputs()
    with pytest.raises(ValueError, match="positive integer"):
        analyze_tenset_cases(run, data, output, examples_per_outcome=value)


def test_cli_runs_offline_without_creating_models_or_loading_dotenv(
    inputs, monkeypatch
):
    data, run, _, _, output = inputs()
    import jev_compiler_eval.config.environment as environment
    import jev_compiler_eval.models.jev_tournament as tournament

    def forbidden(*args, **kwargs):
        raise AssertionError("offline report initialized a model or read credentials")

    monkeypatch.setattr(environment, "load_cli_environment", forbidden)
    monkeypatch.setattr(tournament, "JevApiClient", forbidden)
    path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_tenset_cases.py"
    spec = importlib.util.spec_from_file_location("case_cli_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        "sys.argv",
        [
            "analyze_tenset_cases.py",
            "--results",
            str(run),
            "--dataset",
            str(data),
            "--output-dir",
            str(output),
        ],
    )
    module.main()
    assert (output / "case_studies.json").exists()
