"""Collection execution preserves paired cases and resumes verified child results."""

import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
import yaml

import jev_compiler_eval.evaluation.tenset_collection as collection_module
import jev_compiler_eval.models.jev_tournament as tournament_module
from jev_compiler_eval.evaluation.tenset_collection import (
    BUNDLE_FILES,
    RECEIPT_FILE,
    STATE_FILE,
    preflight_tenset_collection,
    run_tenset_collection,
)
from jev_compiler_eval.jev import JevApiResult
from jev_compiler_eval.models import BaseDecisionModel, register_model
from jev_compiler_eval.types import DecisionResult

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CollectionTestModel(BaseDecisionModel):
    instances = []
    fail_workload = None

    def __init__(self, config):
        super().__init__(config)
        self.calls = []
        self.close_calls = 0
        self.__class__.instances.append(self)

    @property
    def model_type(self):
        return "collection-test-model"

    def validate_config(self):
        if self.config != {"prefer": "largest"}:
            raise ValueError("invalid collection test model configuration")

    def rank(self, workload, candidates):
        self.calls.append(workload.workload_id)
        if workload.workload_id == self.fail_workload:
            raise RuntimeError("simulated model interruption")
        ordered = sorted(
            candidates,
            key=lambda item: item.configuration["transformations"][0]["arguments"][0],
            reverse=True,
        )
        return DecisionResult(tuple(item.candidate_id for item in ordered), 2.0)

    def close(self):
        self.close_calls += 1


register_model("collection-test-model", CollectionTestModel)


@pytest.fixture(autouse=True)
def reset_fake_model():
    CollectionTestModel.instances.clear()
    CollectionTestModel.fail_workload = None


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collection_fixture(tmp_path, counts=(2, 5, 3)):
    root = tmp_path / "collection"
    (root / "workloads").mkdir(parents=True)
    entries = []
    for index, count in enumerate(counts):
        workload_id = f"workload-{index}"
        key = json.dumps([f"synthetic_op_{index}", [16, 16]])
        shard = root / "workloads" / f"{workload_id}.json"
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
                        "operation": "synthetic_op",
                        "tensors": [{"shape": [16, 16], "dtype": "float32"}],
                    },
                    "candidates": [
                        {
                            "candidate_id": f"schedule-{number}",
                            "configuration": {
                                "initial_state": [],
                                "transformations": [
                                    {
                                        "code": "SP",
                                        "name": "split",
                                        "arguments": [number],
                                    }
                                ],
                            },
                            "runtime_ms": float(number),
                        }
                        for number in range(1, count + 1)
                    ],
                }
            ],
        }
        shard.write_text(json.dumps(document), encoding="utf-8")
        entries.append(
            {
                "path": shard.relative_to(root).as_posix(),
                "sha256": sha256(shard),
                "workload_id": workload_id,
                "workload_key": key,
                "candidate_count": count,
                "split": "test",
            }
        )
    index_path = root / "index.json"
    index_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": "tenset_collection",
                "status": "complete",
                "provenance": {"synthetic": True, "target": "llvm"},
                "shards": entries,
            }
        ),
        encoding="utf-8",
    )
    config = {
        "schema_version": 1,
        "experiment": {
            "name": "collection-test",
            "seed": 42,
            "output_dir": str(tmp_path / "results"),
        },
        "dataset": {
            "type": "tenset",
            "config": {
                "path": str(index_path),
                "split": "test",
                "candidate_limit": None,
                "max_workloads": None,
                "max_input_bytes": None,
                "seed": 42,
            },
        },
        "candidate_selection": {"strategy": "all"},
        "models": [
            {"id": "random-42", "type": "random", "config": {"seed": 42}},
            {
                "id": "fake-jev",
                "type": "collection-test-model",
                "config": {"prefer": "largest"},
            },
        ],
        "metrics": ["top_1_score", "top_5_score", "runtime_regret"],
        "execution": {"fail_fast": True},
    }
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path, config, index_path


def predictions(path):
    return [
        json.loads(line)
        for line in (path / "predictions.jsonl").read_text().splitlines()
    ]


def test_streams_all_shards_and_macro_averages_without_candidate_weighting(tmp_path):
    path, config, index = collection_fixture(tmp_path)
    result = run_tenset_collection(path)
    output = Path(config["experiment"]["output_dir"])
    assert result.output_dir == output
    assert all((output / name).is_file() for name in BUNDLE_FILES)
    assert len(predictions(output)) == 6
    assert Counter(row["model_id"] for row in predictions(output)) == {
        "random-42": 3,
        "fake-jev": 3,
    }
    fake_summary = next(row for row in result.summaries if row.model_id == "fake-jev")
    assert fake_summary.workloads == 3
    assert fake_summary.metrics["runtime_regret"] == pytest.approx((1 + 4 + 2) / 3)
    assert fake_summary.metrics["top_1_score"] == pytest.approx(
        (1 / 2 + 1 / 5 + 1 / 3) / 3
    )
    instance = CollectionTestModel.instances[0]
    assert len(CollectionTestModel.instances) == 1
    assert instance.calls == ["workload-0", "workload-1", "workload-2"]
    assert instance.close_calls == 1
    manifest = json.loads((output / "manifest.json").read_text())
    provenance = manifest["dataset_provenance"]
    assert provenance["prepared_file_sha256"] == sha256(index)
    assert provenance["collection_index_sha256"] == sha256(index)
    assert provenance["complete_collection_evaluated"] is True
    assert len(provenance["realized_cases"]) == 3
    for workload_id in ("workload-0", "workload-1", "workload-2"):
        paired = [
            row for row in predictions(output) if row["workload_id"] == workload_id
        ]
        assert paired[0]["candidate_ids"] == paired[1]["candidate_ids"]
        assert paired[0]["model_input_sha256"] == paired[1]["model_input_sha256"]


def test_failure_and_resume_reuse_completed_cases_without_overwriting_them(tmp_path):
    path, config, _ = collection_fixture(tmp_path)
    output = Path(config["experiment"]["output_dir"])
    CollectionTestModel.fail_workload = "workload-1"
    with pytest.raises(RuntimeError, match="simulated model interruption"):
        run_tenset_collection(path)
    assert not (output / "manifest.json").exists()
    assert CollectionTestModel.instances[0].close_calls == 1
    child = output / "workloads" / "000000"
    before = {
        name: (child / name).stat().st_mtime_ns
        for name in (*BUNDLE_FILES, RECEIPT_FILE)
    }
    state = json.loads((output / STATE_FILE).read_text())
    assert state["status"] == "interrupted"
    assert state["completed_workloads"] == 1
    CollectionTestModel.fail_workload = None
    run_tenset_collection(path, resume=True)
    assert CollectionTestModel.instances[1].calls == ["workload-1", "workload-2"]
    assert CollectionTestModel.instances[1].close_calls == 1
    assert before == {
        name: (child / name).stat().st_mtime_ns
        for name in (*BUNDLE_FILES, RECEIPT_FILE)
    }
    aggregate_times = {
        name: (output / name).stat().st_mtime_ns for name in BUNDLE_FILES
    }
    run_tenset_collection(path, resume=True)
    assert len(CollectionTestModel.instances) == 2
    assert aggregate_times == {
        name: (output / name).stat().st_mtime_ns for name in BUNDLE_FILES
    }


def test_missing_completion_receipt_is_recovered_without_model_calls(tmp_path):
    path, config, _ = collection_fixture(tmp_path)
    run_tenset_collection(path)
    output = Path(config["experiment"]["output_dir"])
    (output / "workloads" / "000000" / RECEIPT_FILE).unlink()
    # An interruption can leave all canonical child files before its receipt.
    state = json.loads((output / STATE_FILE).read_text())
    state["status"] = "interrupted"
    (output / STATE_FILE).write_text(json.dumps(state))
    run_tenset_collection(path, resume=True)
    assert len(CollectionTestModel.instances) == 1


def test_validation_collection_runs_without_being_relabelled_test(tmp_path):
    path, config, index_path = collection_fixture(tmp_path)
    index = json.loads(index_path.read_text())
    for descriptor in index["shards"]:
        shard_path = index_path.parent / descriptor["path"]
        shard = json.loads(shard_path.read_text())
        shard["workloads"][0]["split"] = "validation"
        shard_path.write_text(json.dumps(shard))
        descriptor["split"] = "validation"
        descriptor["sha256"] = sha256(shard_path)
    index_path.write_text(json.dumps(index))
    config["dataset"]["config"]["split"] = "validation"
    path.write_text(yaml.safe_dump(config))
    assert preflight_tenset_collection(path).selected_workloads == 3
    run_tenset_collection(path)
    output = Path(config["experiment"]["output_dir"])
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["dataset_provenance"]["split"] == "validation"
    config["dataset"]["config"]["split"] = "train"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="held-out project split"):
        preflight_tenset_collection(path)


def test_partial_child_output_is_preserved_before_retrying_failed_case(tmp_path):
    path, config, _ = collection_fixture(tmp_path)
    output = Path(config["experiment"]["output_dir"])
    CollectionTestModel.fail_workload = "workload-1"
    with pytest.raises(RuntimeError):
        run_tenset_collection(path)
    partial = output / "workloads" / "000001"
    partial.mkdir()
    (partial / "manifest.json").write_text("interrupted output", encoding="utf-8")
    CollectionTestModel.fail_workload = None

    result = run_tenset_collection(path, resume=True)

    assert all(summary.workloads == 3 for summary in result.summaries)
    preserved = list((output / "interrupted").glob("000001-*/manifest.json"))
    assert len(preserved) == 1
    assert preserved[0].read_text() == "interrupted output"
    assert CollectionTestModel.instances[1].calls == ["workload-1", "workload-2"]


@pytest.mark.parametrize("mutation", ["config", "shard", "child", "aggregate"])
def test_resume_refuses_changed_inputs_or_corrupted_completed_results(
    tmp_path, mutation
):
    path, config, index = collection_fixture(tmp_path)
    run_tenset_collection(path)
    output = Path(config["experiment"]["output_dir"])
    if mutation == "config":
        config["models"][0]["config"]["seed"] = 7
        path.write_text(yaml.safe_dump(config))
    elif mutation == "shard":
        shard = index.parent / "workloads" / "workload-0.json"
        shard.write_text(shard.read_text() + " ")
    elif mutation == "child":
        child = output / "workloads" / "000000" / "predictions.jsonl"
        child.write_text(child.read_text() + " ")
    else:
        (output / "metrics.json").write_text("{}")
    with pytest.raises(ValueError):
        run_tenset_collection(path, resume=True)
    assert len(CollectionTestModel.instances) == 1


@pytest.mark.parametrize("component", ["implementation", "python", "packages"])
def test_resume_rejects_implementation_or_environment_changes_before_models(
    tmp_path, monkeypatch, component
):
    path, config, _ = collection_fixture(tmp_path)
    run_tenset_collection(path)
    output = Path(config["experiment"]["output_dir"])
    original = {name: sha256(output / name) for name in BUNDLE_FILES}
    state = json.loads((output / STATE_FILE).read_text())
    assert "models/jev_tournament.py" in state["identity"]["implementation_sources"]
    assert state["identity"]["python_version"]
    assert state["identity"]["package_versions"]["typesafe-sdk"]
    if component == "implementation":
        changed = collection_module._implementation_fingerprints()
        changed["models/jev_tournament.py"] = "0" * 64
        monkeypatch.setattr(
            collection_module, "_implementation_fingerprints", lambda: changed
        )
    elif component == "python":
        monkeypatch.setattr(
            collection_module.platform, "python_version", lambda: "0.0.0"
        )
    else:
        changed = collection_module._package_versions()
        changed["typesafe-sdk"] = "changed-version"
        monkeypatch.setattr(collection_module, "_package_versions", lambda: changed)

    with pytest.raises(
        ValueError, match="unchanged configuration, collection, and code"
    ):
        run_tenset_collection(path, resume=True)

    assert len(CollectionTestModel.instances) == 1
    assert original == {name: sha256(output / name) for name in BUNDLE_FILES}


def test_offline_preflight_checks_all_cases_without_constructing_models(tmp_path):
    path, config, _ = collection_fixture(tmp_path)
    result = preflight_tenset_collection(path)
    assert result.selected_workloads == result.available_workloads == 3
    assert result.total_candidates == 10
    assert (result.min_candidates, result.max_candidates) == (2, 5)
    assert result.max_serialized_input_bytes > 0
    assert result.api_requests_made == 0
    assert not CollectionTestModel.instances
    assert not Path(config["experiment"]["output_dir"]).exists()


def test_preflight_rejects_later_corrupt_shard_without_model_construction(tmp_path):
    path, _, index = collection_fixture(tmp_path)
    last = index.parent / "workloads" / "workload-2.json"
    last.write_text(last.read_text() + " ")
    with pytest.raises(ValueError, match="checksum"):
        preflight_tenset_collection(path)
    assert not CollectionTestModel.instances


def test_explicit_prefix_and_output_override_record_subset_without_modifying_index(
    tmp_path,
):
    path, config, index = collection_fixture(tmp_path)
    original_hash = sha256(index)
    output = tmp_path / "canary"
    result = run_tenset_collection(path, workload_limit=2, output_dir=output)
    assert all(summary.workloads == 2 for summary in result.summaries)
    manifest = json.loads((output / "manifest.json").read_text())
    provenance = manifest["dataset_provenance"]
    assert provenance["selected_workloads"] == 2
    assert provenance["available_workloads"] == 3
    assert provenance["complete_collection_evaluated"] is False
    assert sha256(index) == original_hash
    assert not Path(config["experiment"]["output_dir"]).exists()


def test_model_scope_mismatch_aborts_instead_of_publishing_aggregate(
    tmp_path, monkeypatch
):
    path, config, _ = collection_fixture(tmp_path)
    original = collection_module.run_experiment

    def invalid_child(config_path, **kwargs):
        result = original(config_path, **kwargs)
        rows = predictions(result.output_dir)
        rows[1]["candidate_ids"].reverse()
        (result.output_dir / "predictions.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
        return result

    monkeypatch.setattr(collection_module, "run_experiment", invalid_child)
    with pytest.raises(ValueError, match="indexed workload"):
        run_tenset_collection(path)
    output = Path(config["experiment"]["output_dir"])
    assert not (output / "manifest.json").exists()
    assert CollectionTestModel.instances[0].close_calls == 1


def test_refuses_nonempty_output_without_resume(tmp_path):
    path, _, _ = collection_fixture(tmp_path)
    run_tenset_collection(path)
    with pytest.raises(FileExistsError, match="not empty"):
        run_tenset_collection(path)
    assert len(CollectionTestModel.instances) == 1


def tournament_config(tmp_path):
    return {
        "model": "jev-1.14.0",
        "cache_dir": str(tmp_path / "api-cache"),
        "max_retries": 0,
        "request_timeout_seconds": 30,
        "order_seed": 42,
        "candidate_ids": "opaque",
        "max_options": 4,
        "advance_per_question": 1,
        "max_request_tokens": 64000,
        "max_state_question_tokens": 32000,
        "chars_per_token": 1.4,
        "max_requests_per_minute": 1000000,
        "max_tokens_per_second": 1000000,
    }


def test_tournament_preflight_and_shared_client_run_without_real_api(
    tmp_path, monkeypatch
):
    path, config, _ = collection_fixture(tmp_path)
    config["models"][1] = {
        "id": "jev-tournament",
        "type": "jev-tournament",
        "config": tournament_config(tmp_path),
    }
    path.write_text(yaml.safe_dump(config))
    clients = []

    class FakeJevClient:
        def __init__(self, settings):
            self.closed = 0
            self.calls = 0
            clients.append(self)

        def decide(self, *, state, questions):
            self.calls += 1
            answers = {}
            for question_id, question in questions.items():
                ids = list(question["criteria"])
                choice = ids[0]
                answers[question_id] = {
                    "type": "choice",
                    "choice": choice,
                    "confidence": 1.0,
                    "probabilities": {cid: float(cid == choice) for cid in ids},
                }
            return JevApiResult(
                model="jev-1.14.0",
                answers=answers,
                input_tokens=100,
                output_tokens=10,
                request_id=f"fake-{self.calls}",
                cache_hit=False,
                request_latency_ms=1,
            )

        def close(self):
            self.closed += 1

    monkeypatch.setattr(tournament_module, "JevApiClient", FakeJevClient)
    offline = preflight_tenset_collection(path)
    assert offline.model_plans["jev-tournament"]["first_round_questions"] == 4
    assert clients == []
    run_tenset_collection(path)
    assert len(clients) == 1
    assert clients[0].closed == 1
    assert clients[0].calls > 0


def test_tournament_collection_rejects_global_pairwise_accuracy_before_clients(
    tmp_path,
):
    path, config, _ = collection_fixture(tmp_path)
    config["models"][1] = {
        "id": "jev-tournament",
        "type": "jev-tournament",
        "config": tournament_config(tmp_path),
    }
    config["metrics"] = ["pairwise_accuracy"]
    path.write_text(yaml.safe_dump(config))

    with pytest.raises(ValueError, match="global ranking metrics"):
        preflight_tenset_collection(path)

    assert CollectionTestModel.instances == []
    assert not Path(config["experiment"]["output_dir"]).exists()


def test_cli_preflight_outputs_counts_without_creating_results(tmp_path):
    path, config, _ = collection_fixture(tmp_path)
    config["models"] = config["models"][:1]
    path.write_text(yaml.safe_dump(config))
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/run_tenset_collection.py"),
            "--config",
            str(path),
            "--preflight-only",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"selected_workloads": 3' in result.stdout
    assert "no API client was created" in result.stdout
    assert not Path(config["experiment"]["output_dir"]).exists()
