"""Frozen complete-pool execution and offline cost planning, without live APIs."""

import copy
import csv
import hashlib
import importlib.util
import json
import pickle
import weakref
from pathlib import Path

import pytest
import test_tenset_registry as registry_fixtures
import yaml

import jev_compiler_eval.models.jev_tournament as tournament_module
from jev_compiler_eval.datasets import tenset_paper as data
from jev_compiler_eval.datasets.tenset_registry import load_tenset_registry
from jev_compiler_eval.datasets.tenset_tutorial_protocol import clean_name
from jev_compiler_eval.evaluation import tenset_collection as collection
from jev_compiler_eval.evaluation import tenset_paper as paper
from jev_compiler_eval.jev import JevApiResult
from jev_compiler_eval.models.random import RandomModel

ROOT = Path(__file__).resolve().parents[1]
TARGET = "llvm -keys=cpu -mcpu=skylake-avx512 -model=platinum-8272"
registry_factory = registry_fixtures.registry_factory


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def forbid(*args, **kwargs):
    raise AssertionError("offline operation loaded credentials or initialized a client")


@pytest.fixture
def paper_fixture(tmp_path, registry_factory):
    """Use the real inert registry, graph, record and collection preparation chain."""

    make_task, _, executed = registry_factory
    native = load_tenset_registry(
        ROOT / "tests/fixtures/tenset_registry/native_registry.pkl"
    )[0]
    root = tmp_path / "raw"
    info, records = root / "network_info", root / "measure_records/platinum-8272"
    info.mkdir(parents=True)
    records.mkdir(parents=True)
    tasks = []
    for number, count in enumerate((6, 8, 260)):
        key = json.dumps([f"synthetic-paper-task-{number}", 16, 8])
        tasks.append(make_task(key, graph_value=native["tensor_graph"]))
        rows = [
            {
                "i": [[key, TARGET], [[], [["SP", 0, 0, 16, [index + 1, 1], 1]]]],
                "r": [[(index + 1) / 1000], 0, 0.0, 0.0],
                "v": "v0.6",
            }
            for index in range(count)
        ]
        # Error records stay in their full pool, with a large runtime label.
        rows[-1]["r"] = [[1e10], 2, 0.0, 0.0]
        path = records / (clean_name((key, "llvm")) + ".json")
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    (info / data.TASK_INFO_FILENAME).write_bytes(
        pickle.dumps((tasks, [2, 7, 1]), protocol=4)
    )
    index = data.prepare_paper_collection(root, tmp_path / "prepared")
    config = yaml.safe_load(
        (ROOT / "experiments/tenset/paper-resnet50/full.yaml").read_text()
    )
    config["dataset"]["config"]["path"] = str(index)
    config["experiment"]["output_dir"] = str(tmp_path / "results")
    config["models"][1]["config"]["cache_dir"] = str(tmp_path / "cache")
    path = tmp_path / "paper.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    assert not executed
    return path, config, index


def edit_config(path, config):
    path.write_text(yaml.safe_dump(config, sort_keys=False))


def corrupt_shard(index_path, update, *, position=0, refresh_hash=False):
    index = json.loads(index_path.read_text())
    entry = index["shards"][position]
    path = index_path.parent / entry["path"]
    document = json.loads(path.read_text())
    update(document)
    path.write_text(json.dumps(document))
    if refresh_hash:
        entry["sha256"] = digest(path)
        index_path.write_text(json.dumps(index))


@pytest.fixture
def offline_clients(monkeypatch):
    clients = []

    class OfflineClient:
        def __init__(self, settings):
            assert settings.model == "jev-1.14.0"
            assert settings.long_context is True
            self.calls, self.closed = [], False
            clients.append(self)

        def decide(self, *, state, questions):
            self.calls.append({"state": state, "questions": questions})
            answers = {}
            for question_id, question in questions.items():
                ids = list(question["criteria"])
                answers[question_id] = {
                    "type": "choice",
                    "choice": ids[0],
                    "confidence": 1.0,
                    "probabilities": {
                        identity: (len(ids) - index) / sum(range(1, len(ids) + 1))
                        for index, identity in enumerate(ids)
                    },
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
            self.closed = True

    monkeypatch.setattr(tournament_module, "JevApiClient", OfflineClient)
    return clients


def test_preflight_checks_full_pools_and_cost_without_client_or_output(
    paper_fixture, monkeypatch
):
    path, config, _ = paper_fixture
    monkeypatch.setattr(tournament_module, "JevApiClient", forbid)
    result = paper.preflight_tenset_paper(path)
    assert result["api_requests_made"] == 0
    assert result["full_network_tasks"] == result["selected_tasks"] == 3
    assert result["selected_candidates"] == 274
    assert result["full_network_task_weight"] == 10
    assert result["metric_contract_validated"] is True
    assert result["scope"] == result["metric_contract_scope"] == "full_network"
    assert result["external_reference"]["model"] == "MLP + Ranking loss"
    assert result["external_reference"]["original_checkpoint_verified"] is False
    assert not Path(config["experiment"]["output_dir"]).exists()
    plan = result["model_plans"][paper.JEV_ID]
    assert plan["first_round"]["questions"] == 4
    assert plan["first_round"]["requests"] >= 3
    for scenario in plan["scenarios"].values():
        assert scenario["input_tokens"] > plan["first_round"]["input_tokens"]
        assert scenario["remaining_input_tokens_after_first_round"] > 0
    price = result["cost_assumptions"]["usd_per_million_input_tokens"]
    assert price == 0.042
    assert result["cost_assumptions"]["input_only"] is True
    repriced = paper.preflight_tenset_paper(path, input_price_per_million=0.084)
    assert repriced["cost_assumptions"]["estimated_first_round_input_cost_usd"] == (
        2 * result["cost_assumptions"]["estimated_first_round_input_cost_usd"]
    )


def test_preflight_releases_each_graph_and_configuration_before_loading_next_task(
    paper_fixture, monkeypatch
):
    path, _, _ = paper_fixture
    original = collection._load_case
    references = []
    loaded = []

    def track_case(*args, **kwargs):
        assert all(reference() is None for reference in references)
        case, metadata = original(*args, **kwargs)
        references.extend(
            weakref.ref(value) for value in (case, case.workload, case.candidates[0])
        )
        loaded.append(case.workload.workload_id)
        return case, metadata

    monkeypatch.setattr(collection, "_load_case", track_case)
    monkeypatch.setattr(tournament_module, "JevApiClient", forbid)
    result = paper.preflight_tenset_paper(path)
    assert len(loaded) == result["full_network_tasks"] == 3
    assert all(reference() is None for reference in references)


def test_pilot_validates_whole_inventory_but_has_no_full_reference(
    paper_fixture, monkeypatch
):
    path, config, index = paper_fixture
    config["dataset"]["config"]["max_workloads"] = 1
    edit_config(path, config)
    monkeypatch.setattr(tournament_module, "JevApiClient", forbid)
    result = paper.preflight_tenset_paper(path)
    assert result["scope"] == "pilot_subset"
    assert result["selected_tasks"] == 1
    assert result["full_network_tasks"] == 3
    assert result["external_reference"] is None
    corrupt_shard(
        index, lambda document: document.update(schema_version=99), position=2
    )
    with pytest.raises(ValueError):
        paper.run_tenset_paper(path, execute=True)


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("experiment", "seed", 42),
        ("dataset", "seed", 42),
        ("dataset", "candidate_limit", 50),
        ("dataset", "describe_config_fields", False),
        ("dataset", "require_program_metadata", False),
        ("dataset", "split", "validation"),
        ("dataset", "max_input_bytes", 999999),
        ("jev", "model", "jev-latest"),
        ("jev", "long_context", False),
        ("jev", "order_seed", 42),
        ("jev", "max_options", 128),
        ("jev", "advance_per_question", 1),
        ("jev", "include_context", False),
        ("random", "seed", 42),
    ],
)
def test_changed_protocol_settings_rejected_before_client(
    paper_fixture, monkeypatch, section, key, value
):
    path, config, _ = paper_fixture
    settings = {
        "experiment": config["experiment"],
        "dataset": config["dataset"]["config"],
        "jev": config["models"][1]["config"],
        "random": config["models"][0]["config"],
    }
    settings[section][key] = value
    edit_config(path, config)
    monkeypatch.setattr(tournament_module, "JevApiClient", forbid)
    with pytest.raises(ValueError):
        paper.run_tenset_paper(path, execute=True)


@pytest.mark.parametrize("field", ["task_weight", "source", "candidate_ids"])
def test_missing_inventory_commitments_fail_before_client(
    paper_fixture, monkeypatch, field
):
    path, _, index = paper_fixture
    document = json.loads(index.read_text())
    del document["shards"][0][field]
    index.write_text(json.dumps(document))
    monkeypatch.setattr(tournament_module, "JevApiClient", forbid)
    with pytest.raises((ValueError, KeyError)):
        paper.run_tenset_paper(path, execute=True)


@pytest.mark.parametrize("private", ["official_throughput", "reference_cost_seconds"])
def test_refreshed_shard_hash_cannot_hide_changed_private_label(
    paper_fixture, monkeypatch, private
):
    path, _, index = paper_fixture

    def update(document):
        document["workloads"][0]["candidates"][0][private] *= 0.5

    corrupt_shard(index, update, refresh_hash=True)
    monkeypatch.setattr(tournament_module, "JevApiClient", forbid)
    with pytest.raises(ValueError):
        paper.run_tenset_paper(path, execute=True)


def test_changed_preparation_source_commitment_is_rejected(paper_fixture, monkeypatch):
    path, _, index = paper_fixture
    preparation = index.parent / "preparation.json"
    document = json.loads(preparation.read_text())
    document["sources"][0]["sha256"] = "f" * 64
    preparation.write_text(json.dumps(document))
    monkeypatch.setattr(tournament_module, "JevApiClient", forbid)
    with pytest.raises(ValueError):
        paper.preflight_tenset_paper(path)


def test_execution_opt_in_is_required_even_with_valid_config(
    paper_fixture, monkeypatch
):
    path, _, _ = paper_fixture
    monkeypatch.setattr(tournament_module, "JevApiClient", forbid)
    with pytest.raises(ValueError, match="execute=True"):
        paper.run_tenset_paper(path)


def test_shared_runner_config_reload_is_bound_to_the_validated_identity(
    paper_fixture, monkeypatch
):
    path, config, _ = paper_fixture
    original = collection.run_tenset_collection

    def mutate_on_handoff(config_path, **kwargs):
        config["dataset"]["config"]["candidate_limit"] = 2
        edit_config(path, config)
        return original(config_path, **kwargs)

    monkeypatch.setattr(collection, "run_tenset_collection", mutate_on_handoff)
    monkeypatch.setattr(RandomModel, "__init__", forbid)
    monkeypatch.setattr(tournament_module, "JevApiClient", forbid)
    with pytest.raises(ValueError, match="validated preflight identity"):
        paper.run_tenset_paper(path, execute=True)
    assert not Path(config["experiment"]["output_dir"]).exists()


def test_real_collection_execution_and_resume_preserve_label_isolation(
    paper_fixture, offline_clients
):
    path, _, _ = paper_fixture
    result = paper.run_tenset_paper(path, execute=True)
    assert len(offline_clients) == 1
    assert offline_clients[0].closed
    visible = json.dumps(offline_clients[0].calls)
    for name in (
        "runtime_ms",
        "official_throughput",
        "source_record_index",
        "reference_cost_seconds",
        "full_task_min_latency_seconds",
        "task_weight",
    ):
        assert f'"{name}":' not in visible
    summary = json.loads((result.output_dir / paper.SUMMARY_JSON).read_text())
    assert summary["scope"] == "full_network"
    assert summary["external_reference"]["comparison_type"].startswith("external")
    assert len(summary["models"]) == 2
    with (result.output_dir / "paper-reference-comparison.csv").open() as stream:
        comparison_rows = list(csv.DictReader(stream))
    assert [row["Model"] for row in comparison_rows] == [
        "Jev",
        "Random",
        "MLP + Ranking loss",
    ]
    assert [
        row["same_as_local_candidate_pool_verified"] for row in comparison_rows
    ] == ["True", "True", "False"]
    assert float(comparison_rows[2]["paper_network_best_of_1_score"]) == 0.8823
    assert float(comparison_rows[2]["paper_network_best_of_5_score"]) == 0.9456
    assert (
        "unverified"
        in (result.output_dir / "paper-reference-comparison.md").read_text()
    )
    for model in summary["models"]:
        assert set(model["metrics"]) == {
            "paper_network_best_of_1_score",
            "paper_network_best_of_5_score",
        }
    child_metrics = json.loads((result.output_dir / "metrics.json").read_text())
    assert set(child_metrics["models"][0]["metrics"]) == {
        "top_1_score",
        "top_5_score",
    }
    original = {
        path.name: digest(path)
        for path in result.output_dir.iterdir()
        if path.is_file()
    }
    paper.run_tenset_paper(path, execute=True, resume=True)
    assert len(offline_clients) == 1
    assert original == {
        path.name: digest(path)
        for path in result.output_dir.iterdir()
        if path.is_file()
    }


def test_pilot_outputs_explicit_subset_metric_names(paper_fixture, offline_clients):
    path, config, _ = paper_fixture
    config["dataset"]["config"]["max_workloads"] = 1
    edit_config(path, config)
    result = paper.run_tenset_paper(path, execute=True)
    summary = json.loads((result.output_dir / paper.SUMMARY_JSON).read_text())
    assert summary["external_reference"] is None
    for model in summary["models"]:
        assert set(model["metrics"]) == {
            "pilot_network_best_of_1_score",
            "pilot_network_best_of_5_score",
        }


def test_bounded_profile_remains_pilot_when_limit_equals_entire_network(
    paper_fixture, offline_clients
):
    path, config, _ = paper_fixture
    config["dataset"]["config"]["max_workloads"] = 3
    edit_config(path, config)
    preflight = paper.preflight_tenset_paper(path)
    assert preflight["selected_tasks"] == preflight["full_network_tasks"] == 3
    assert preflight["scope"] == "pilot_subset"
    assert preflight["external_reference"] is None
    result = paper.run_tenset_paper(path, execute=True)
    summary = json.loads((result.output_dir / paper.SUMMARY_JSON).read_text())
    assert summary["coverage"]["complete_network_coverage"] is True
    assert summary["scope"] == "pilot_subset"
    assert summary["external_reference"] is None
    for model in summary["models"]:
        assert set(model["metrics"]) == {
            "pilot_network_best_of_1_score",
            "pilot_network_best_of_5_score",
        }
    assert not list(result.output_dir.glob("*reference-comparison*"))


@pytest.mark.parametrize("change", ["config", "prepared", "results"])
def test_resume_tampering_fails_before_new_client(
    paper_fixture, offline_clients, monkeypatch, change
):
    path, config, index = paper_fixture
    result = paper.run_tenset_paper(path, execute=True)
    if change == "config":
        config["models"][1]["config"]["chars_per_token"] = 1.5
        edit_config(path, config)
    elif change == "prepared":
        corrupt_shard(index, lambda document: document.update(schema_version=99))
    else:
        with (result.output_dir / "predictions.jsonl").open("a") as stream:
            stream.write("\n")
    monkeypatch.setattr(tournament_module, "JevApiClient", forbid)
    with pytest.raises(ValueError):
        paper.run_tenset_paper(path, execute=True, resume=True)
    assert len(offline_clients) == 1


def load_script():
    spec = importlib.util.spec_from_file_location(
        "paper_cli_under_test", ROOT / "scripts/run_tenset_paper.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("offline_args", [[], ["--preflight-only"]])
def test_cli_defaults_to_offline_without_loading_env(
    paper_fixture, monkeypatch, capsys, offline_args
):
    path, _, _ = paper_fixture
    module = load_script()
    monkeypatch.setattr(module, "load_cli_environment", forbid)
    monkeypatch.setattr(module, "run_tenset_paper", forbid)
    monkeypatch.setattr(
        "sys.argv", ["run_tenset_paper.py", "--config", str(path), *offline_args]
    )
    module.main()
    assert "Offline preflight only" in capsys.readouterr().out


def test_cli_invalid_pool_fails_before_env_or_execution(paper_fixture, monkeypatch):
    path, _, index = paper_fixture
    corrupt_shard(index, lambda document: document.update(schema_version=99))
    module = load_script()
    monkeypatch.setattr(module, "load_cli_environment", forbid)
    monkeypatch.setattr(module, "run_tenset_paper", forbid)
    monkeypatch.setattr(
        "sys.argv", ["run_tenset_paper.py", "--config", str(path), "--execute"]
    )
    with pytest.raises(ValueError):
        module.main()


def test_cli_preserves_printed_preflight_identity_across_environment_loading(
    paper_fixture, monkeypatch, capsys
):
    path, config, _ = paper_fixture
    module = load_script()
    environment_reads = []

    def mutate_while_loading_env(env_path):
        environment_reads.append(env_path)
        # This is still a valid pilot profile, but differs from the printed full
        # network plan and its budget. Execution must preserve the first choice.
        config["dataset"]["config"]["max_workloads"] = 1
        edit_config(path, config)

    monkeypatch.setattr(module, "load_cli_environment", mutate_while_loading_env)
    monkeypatch.setattr(collection, "run_tenset_collection", forbid)
    monkeypatch.setattr(RandomModel, "__init__", forbid)
    monkeypatch.setattr(tournament_module, "JevApiClient", forbid)
    monkeypatch.setattr(
        "sys.argv", ["run_tenset_paper.py", "--config", str(path), "--execute"]
    )
    with pytest.raises(ValueError, match="validated preflight identity"):
        module.main()
    assert environment_reads
    assert '"selected_tasks": 3' in capsys.readouterr().out
    assert not Path(config["experiment"]["output_dir"]).exists()


def test_templates_keep_identical_pools_shared_cache_and_separate_outputs():
    profiles = [
        yaml.safe_load(
            (ROOT / f"experiments/tenset/paper-resnet50/{name}.yaml").read_text()
        )
        for name in ("pilot", "full")
    ]
    pilot, full = copy.deepcopy(profiles)
    assert pilot["dataset"]["config"].pop("max_workloads") == 3
    assert full["dataset"]["config"].pop("max_workloads") is None
    assert pilot["dataset"] == full["dataset"]
    assert pilot["models"][1]["config"].pop("cache_dir") == full["models"][1][
        "config"
    ].pop("cache_dir")
    assert pilot["models"] == full["models"]
    assert pilot["experiment"]["output_dir"] != full["experiment"]["output_dir"]
