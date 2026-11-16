"""Tutorial execution keeps record labels private and aggregation explicit."""

import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

import jev_compiler_eval.models.jev_tournament as tournament_module
from jev_compiler_eval.evaluation import tenset_collection as collection
from jev_compiler_eval.evaluation import tenset_tutorial as tutorial
from jev_compiler_eval.jev import JevApiResult
from jev_compiler_eval.models import BaseDecisionModel, register_model
from jev_compiler_eval.types import DecisionResult


class TutorialTestModel(BaseDecisionModel):
    calls = []

    @property
    def model_type(self):
        return "tutorial-test-model"

    def validate_config(self):
        if self.config:
            raise ValueError("test model requires an empty configuration")

    def rank(self, workload, candidates):
        self.calls.append(
            {
                "workload": workload.workload_id,
                "context": dict(workload.context),
                "candidates": [
                    dict(candidate.configuration) for candidate in candidates
                ],
            }
        )
        ranking = sorted(
            candidates,
            key=lambda item: item.configuration["transformations"][0]["arguments"][0],
            reverse=True,
        )
        return DecisionResult(tuple(item.candidate_id for item in ranking), 1.0)


register_model("tutorial-test-model", TutorialTestModel)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture(autouse=True)
def reset_model():
    TutorialTestModel.calls.clear()


@pytest.fixture
def tutorial_fixture(tmp_path):
    root = tmp_path / "tutorial"
    (root / "workloads").mkdir(parents=True)
    descriptors = []
    for number, count in enumerate((2, 4)):
        workload_id = f"tutorial-{number}"
        key = f'["synthetic-{number}",16]'
        candidates = [
            {
                "candidate_id": f"record-{index}",
                "source_record_index": index + 100,
                "official_throughput": 1.0 / 2**index,
                "runtime_ms": float(2**index),
                "configuration": {
                    "initial_state": [],
                    "transformations": [
                        {"code": "SP", "name": "split", "arguments": [index]}
                    ],
                },
            }
            for index in range(count)
        ]
        document = {
            "schema_version": 1,
            "dataset": "tenset",
            "provenance": {"synthetic": True},
            "workloads": [
                {
                    "workload_id": workload_id,
                    "split": "test",
                    "context": {
                        "format": "tvm-auto-scheduler-json",
                        "workload_key": key,
                        "target": "llvm",
                        "operation": "synthetic",
                        "tensors": [{"shape": [16], "dtype": "float32"}],
                    },
                    "candidates": candidates,
                }
            ],
        }
        shard = root / "workloads" / f"{workload_id}.json"
        shard.write_text(json.dumps(document))
        descriptors.append(
            {
                "workload_id": workload_id,
                "workload_key": key,
                "split": "test",
                "candidate_count": count,
                "path": f"workloads/{workload_id}.json",
                "sha256": digest(shard),
            }
        )
    index = root / "index.json"
    index.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": "tenset_collection",
                "status": "complete",
                "provenance": {
                    "experiment_protocol": tutorial.EXPERIMENT_PROTOCOL,
                    "reference_alignment_status": tutorial.ALIGNMENT_STATUS,
                },
                "shards": descriptors,
            }
        )
    )
    config = {
        "schema_version": 1,
        "experiment": {
            "name": "tutorial-synthetic",
            "seed": 42,
            "output_dir": str(tmp_path / "results"),
        },
        "dataset": {
            "type": "tenset",
            "config": {
                "path": str(index),
                "split": "test",
                "seed": 42,
                "candidate_limit": None,
                "max_workloads": None,
                "max_input_bytes": None,
            },
        },
        "candidate_selection": {"strategy": "all"},
        "models": [
            {"id": "random-42", "type": "random", "config": {"seed": 42}},
            {"id": "tutorial-model", "type": "tutorial-test-model", "config": {}},
        ],
        "metrics": ["top_1_score", "top_5_score", "runtime_regret"],
        "execution": {"fail_fast": True},
    }
    path = tmp_path / "tutorial.yaml"
    path.write_text(yaml.safe_dump(config))
    return path, config, index


def change_shard(index_path, update, position=0):
    index = json.loads(index_path.read_text())
    descriptor = index["shards"][position]
    shard = index_path.parent / descriptor["path"]
    document = json.loads(shard.read_text())
    update(document["workloads"][0])
    shard.write_text(json.dumps(document))
    descriptor["sha256"] = digest(shard)
    index_path.write_text(json.dumps(index))


def forbid(*args, **kwargs):
    raise AssertionError("offline operation initialized a live model or loaded secrets")


def test_offline_preflight_validates_all_private_records_without_clients(
    tutorial_fixture, monkeypatch
):
    path, _, _ = tutorial_fixture
    monkeypatch.setattr(TutorialTestModel, "__init__", forbid)
    result = tutorial.preflight_tenset_tutorial(path)
    assert result["api_requests_made"] == 0
    assert result["test_record_count"] == 6
    assert result["selected_workloads"] == 2
    assert result["comparison_scope"] == "Jev versus Random"
    assert result["paired_xgboost_comparison"] is False
    assert result["live_execution_requires_allow_external_reference"] is False
    assert "published_xgboost_reference_source" not in result
    assert result["reference_alignment_status"] == tutorial.ALIGNMENT_STATUS
    assert not TutorialTestModel.calls


def test_invalid_private_pool_precedes_model_initialization_without_reference_flag(
    tutorial_fixture, monkeypatch
):
    path, config, index = tutorial_fixture
    change_shard(
        index, lambda workload: workload["candidates"][0].update(official_throughput=0)
    )
    monkeypatch.setattr(collection, "run_tenset_collection", forbid)
    with pytest.raises(ValueError, match="throughput"):
        tutorial.run_tenset_tutorial(path)
    assert not Path(config["experiment"]["output_dir"]).exists()


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("official_throughput", None, "throughput"),
        ("official_throughput", 0, "throughput"),
        ("official_throughput", float("nan"), "throughput"),
        ("official_throughput", True, "throughput"),
        ("official_throughput", 0.25, "disagree"),
        ("source_record_index", None, "record indices"),
        ("source_record_index", True, "record indices"),
    ],
)
def test_private_label_and_record_identity_errors_fail_before_calls(
    tutorial_fixture, monkeypatch, field, value, match
):
    path, _, index = tutorial_fixture
    change_shard(
        index, lambda workload: workload["candidates"][0].update({field: value})
    )
    monkeypatch.setattr(collection, "run_tenset_collection", forbid)
    with pytest.raises(ValueError, match=match):
        tutorial.run_tenset_tutorial(path, allow_external_reference=True)


def test_duplicate_record_indices_are_rejected(tutorial_fixture):
    path, _, index = tutorial_fixture
    change_shard(
        index,
        lambda workload: workload["candidates"][1].update(source_record_index=100),
    )
    with pytest.raises(ValueError, match="record indices"):
        tutorial.preflight_tenset_tutorial(path)


@pytest.mark.parametrize("key", ["experiment_protocol", "reference_alignment_status"])
def test_nonreference_collection_cannot_use_tutorial_runner(tutorial_fixture, key):
    path, _, index = tutorial_fixture
    document = json.loads(index.read_text())
    document["provenance"][key] = "project_workload_hash_split"
    index.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="tutorial"):
        tutorial.preflight_tenset_tutorial(path)


@pytest.mark.parametrize(
    "setting,value", [("candidate_limit", 2), ("split", "validation")]
)
def test_sampling_or_alternative_split_is_not_a_reference_test_pool(
    tutorial_fixture, setting, value
):
    path, config, _ = tutorial_fixture
    config["dataset"]["config"][setting] = value
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        tutorial.preflight_tenset_tutorial(path)


def test_execution_writes_weighted_and_macro_scores_as_distinct_outputs(
    tutorial_fixture,
):
    path, config, _ = tutorial_fixture
    result = tutorial.run_tenset_tutorial(path)
    output = Path(config["experiment"]["output_dir"])
    assert result.output_dir == output
    summary = json.loads((output / tutorial.SUMMARY_JSON).read_text())
    model = next(
        row for row in summary["models"] if row["model_id"] == "tutorial-model"
    )
    assert model["test_records"] == 6
    assert model["workloads"] == 2
    assert model["record_weighted_metrics"]["top_1_score"] == pytest.approx(0.25)
    assert model["macro_metrics"]["top_1_score"] == pytest.approx(0.3125)
    assert model["record_weighted_metrics"]["top_5_score"] == pytest.approx(0.5625)
    assert model["macro_metrics"]["top_5_score"] == pytest.approx(0.609375)
    assert summary["global_tournament_pairwise_accuracy_reported"] is False
    assert summary["comparison_scope"] == "Jev versus Random"
    assert summary["paired_xgboost_comparison"] is False
    assert "external_reference_file" not in summary
    assert summary["predictions_sha256"] == digest(output / "predictions.jsonl")
    rows = list(csv.DictReader((output / tutorial.SUMMARY_CSV).open()))
    assert len(rows) == 2
    assert float(rows[1]["weighted_top_1_score"]) == pytest.approx(0.25)
    assert not (output / "external-xgboost-reference.json").exists()
    assert [
        row["model_id"]
        for row in json.loads((output / "metrics.json").read_text())["models"]
    ] == ["random-42", "tutorial-model"]
    visible = json.dumps(TutorialTestModel.calls)
    for private in ("official_throughput", "source_record_index", "runtime_ms"):
        assert private not in visible
    assert len(TutorialTestModel.calls) == 2


def test_resume_reuses_results_and_recreates_only_derived_summaries(tutorial_fixture):
    path, config, _ = tutorial_fixture
    tutorial.run_tenset_tutorial(path, allow_external_reference=True)
    output = Path(config["experiment"]["output_dir"])
    original_predictions = (output / "predictions.jsonl").read_bytes()
    legacy_reference = output / "external-xgboost-reference.json"
    historical_contents = b'{"status":"historical reference; preserve unchanged"}\n'
    legacy_reference.write_bytes(historical_contents)
    (output / tutorial.SUMMARY_JSON).unlink()
    tutorial.run_tenset_tutorial(path, allow_external_reference=True, resume=True)
    assert len(TutorialTestModel.calls) == 2
    assert (output / "predictions.jsonl").read_bytes() == original_predictions
    assert (output / tutorial.SUMMARY_JSON).is_file()
    assert legacy_reference.read_bytes() == historical_contents


def rewrite_predictions_and_completion(output, update):
    path = output / "predictions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    update(rows)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    state_path = output / collection.STATE_FILE
    state = json.loads(state_path.read_text())
    state["files"]["predictions.jsonl"] = digest(path)
    state_path.write_text(json.dumps(state))


def test_missing_prediction_is_not_silently_omitted_from_weighted_summary(
    tutorial_fixture,
):
    path, config, _ = tutorial_fixture
    tutorial.run_tenset_tutorial(path, allow_external_reference=True)
    output = Path(config["experiment"]["output_dir"])
    rewrite_predictions_and_completion(output, lambda rows: rows.pop())
    with pytest.raises(ValueError, match="missing"):
        tutorial.summarize_tenset_tutorial(path)


def test_changed_private_labels_in_predictions_are_rejected(tutorial_fixture):
    path, config, _ = tutorial_fixture
    tutorial.run_tenset_tutorial(path, allow_external_reference=True)
    output = Path(config["experiment"]["output_dir"])

    def corrupt(rows):
        identity = rows[0]["candidate_ids"][0]
        rows[0]["runtime_by_candidate_ms"][identity] *= 2

    rewrite_predictions_and_completion(output, corrupt)
    with pytest.raises(ValueError, match="labels differ"):
        tutorial.summarize_tenset_tutorial(path)


def test_changed_completed_prediction_file_fails_checksum_validation(tutorial_fixture):
    path, config, _ = tutorial_fixture
    tutorial.run_tenset_tutorial(path, allow_external_reference=True)
    output = Path(config["experiment"]["output_dir"])
    with (output / "predictions.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="checksums"):
        tutorial.summarize_tenset_tutorial(path)


def test_duplicate_configurations_survive_as_separate_record_candidates(
    tutorial_fixture,
):
    path, _, index = tutorial_fixture

    def duplicate_configuration(workload):
        workload["candidates"][1]["configuration"] = copy.deepcopy(
            workload["candidates"][0]["configuration"]
        )

    change_shard(index, duplicate_configuration)
    result = tutorial.preflight_tenset_tutorial(path)
    assert result["test_record_count"] == 6
    assert result["total_candidates"] == 6


def test_tournament_global_pairwise_accuracy_is_rejected(tutorial_fixture):
    path, config, _ = tutorial_fixture
    config["metrics"].append("pairwise_accuracy")
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="selection metrics"):
        tutorial.preflight_tenset_tutorial(path)


def test_real_tournament_interface_runs_and_resumes_using_an_offline_sdk_stub(
    tutorial_fixture, monkeypatch, tmp_path
):
    path, config, _ = tutorial_fixture
    config["models"][1] = {
        "id": "jev-tournament",
        "type": "jev-tournament",
        "config": {
            "model": "jev-1.14.0",
            "cache_dir": str(tmp_path / "cache"),
            "max_retries": 0,
            "request_timeout_seconds": 30,
            "order_seed": 42,
            "candidate_ids": "opaque",
            "max_options": 2,
            "advance_per_question": 1,
            "max_request_tokens": 200000,
            "max_state_question_tokens": 180000,
            "chars_per_token": 1.4,
            "max_requests_per_minute": 1000000,
            "max_tokens_per_second": 1000000,
            "long_context": True,
        },
    }
    path.write_text(yaml.safe_dump(config))
    clients = []

    class OfflineClient:
        def __init__(self, settings):
            assert settings.long_context is True
            self.calls = []
            self.closed = 0
            clients.append(self)

        def decide(self, *, state, questions):
            self.calls.append({"state": state, "questions": questions})
            answers = {}
            for question_id, question in questions.items():
                ids = list(question["criteria"])
                chosen = ids[0]
                answers[question_id] = {
                    "type": "choice",
                    "choice": chosen,
                    "confidence": 1.0,
                    "probabilities": {cid: float(cid == chosen) for cid in ids},
                }
            return JevApiResult(
                model="jev-1.14.0",
                answers=answers,
                input_tokens=100,
                output_tokens=10,
                request_id=f"offline-{len(self.calls)}",
                cache_hit=False,
                request_latency_ms=1,
            )

        def close(self):
            self.closed += 1

    monkeypatch.setattr(tournament_module, "JevApiClient", OfflineClient)
    preflight = tutorial.preflight_tenset_tutorial(path)
    assert preflight["model_plans"]["jev-tournament"]["first_round_questions"] == 3
    assert clients == []
    tutorial.run_tenset_tutorial(path, allow_external_reference=True)
    assert len(clients) == 1
    assert len(clients[0].calls) == 3
    assert clients[0].closed == 1
    visible = json.dumps(clients[0].calls)
    for private in ("runtime_ms", "official_throughput", "source_record_index"):
        assert private not in visible
    tutorial.run_tenset_tutorial(path, allow_external_reference=True, resume=True)
    assert len(clients) == 1
    output = Path(config["experiment"]["output_dir"])
    summary = json.loads((output / tutorial.SUMMARY_JSON).read_text())
    assert summary["global_tournament_pairwise_accuracy_reported"] is False
    assert summary["models"][1]["test_records"] == 6


def load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_tenset_tutorial.py"
    spec = importlib.util.spec_from_file_location("tutorial_cli_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_preflight_never_loads_credentials_or_runs_models(
    tutorial_fixture, monkeypatch, capsys
):
    path, _, _ = tutorial_fixture
    module = load_script()
    monkeypatch.setattr(module, "load_cli_environment", forbid)
    monkeypatch.setattr(module, "run_tenset_tutorial", forbid)
    monkeypatch.setattr(
        "sys.argv",
        ["run_tenset_tutorial.py", "--config", str(path), "--preflight-only"],
    )
    module.main()
    assert '"test_record_count": 6' in capsys.readouterr().out


@pytest.mark.parametrize("compatibility_args", [[], ["--allow-external-reference"]])
def test_cli_runs_jev_random_without_requiring_external_reference_flag(
    tutorial_fixture, monkeypatch, capsys, compatibility_args
):
    path, config, _ = tutorial_fixture
    module = load_script()
    credentials = []
    monkeypatch.setattr(module, "load_cli_environment", credentials.append)
    monkeypatch.setattr(
        "sys.argv",
        ["run_tenset_tutorial.py", "--config", str(path), *compatibility_args],
    )
    module.main()
    assert len(credentials) == 1
    assert len(TutorialTestModel.calls) == 2
    assert "Comparison: Jev versus Random" in capsys.readouterr().out
    output = Path(config["experiment"]["output_dir"])
    assert not (output / "external-xgboost-reference.json").exists()
