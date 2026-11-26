"""End-to-end frozen ablations use synthetic data and an in-memory API double."""

import copy
import csv
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest
import test_tenset_paper_runner as paper_fixtures
import test_tenset_size_sweep as size_fixtures
import yaml

import jev_compiler_eval.models.jev_tournament as tournament_module
from jev_compiler_eval.datasets.tenset import TenSetDataset, _reject_label_keys
from jev_compiler_eval.evaluation import tenset_ablation as ablation
from jev_compiler_eval.evaluation.tenset_ablation_analysis import (
    package_ablation,
    summarize_ablation,
)
from jev_compiler_eval.jev import JevApiResult
from jev_compiler_eval.types import (
    Candidate,
    DecisionResult,
    EvaluationCase,
    RuntimeRecord,
    Workload,
)

ROOT = Path(__file__).resolve().parents[1]
registry_factory = paper_fixtures.registry_factory
paper_fixture = paper_fixtures.paper_fixture


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def forbid(*args, **kwargs):
    raise AssertionError("offline operation initialized an API client or environment")


@pytest.fixture
def ablation_fixture(tmp_path, paper_fixture):
    _, _, full_index = paper_fixture
    data_dir = tmp_path / "small-data"
    data_dir.mkdir()
    small_path = data_dir / "prepared.json"
    document = size_fixtures.make_prepared(small_path, 2)
    settings = yaml.safe_load(
        (ROOT / "experiments/tenset/ablation/study.yaml").read_text()
    )
    settings["study"]["name"] = "synthetic-context-ablation"
    settings["study"]["output_dir"] = str(tmp_path / "ablation-results")
    settings["phases"]["small"].update(
        path=str(small_path), expected_sha256=digest(small_path), workloads=2
    )
    settings["phases"]["full"].update(path=str(full_index), workloads=3, keep=5)
    path = tmp_path / "ablation.yaml"
    path.write_text(yaml.safe_dump(settings, sort_keys=False))
    return path, settings, small_path, document, full_index


def write_settings(path, settings):
    path.write_text(yaml.safe_dump(settings, sort_keys=False))


@pytest.fixture
def fake_clients(monkeypatch):
    clients = []

    class FakeClient:
        def __init__(self, settings):
            self.settings = settings
            self.calls = []
            self.closed = False
            clients.append(self)
            self.instance = len(clients)

        def decide(self, *, state, questions):
            # This validates the actual model-visible payload, not a separate
            # feature dictionary constructed only for the test.
            _reject_label_keys(state)
            _reject_label_keys(questions)
            assert len(questions) == 1
            self.calls.append(copy.deepcopy({"state": state, "questions": questions}))
            answers = {}
            for question_id, question in questions.items():
                aliases = list(question["criteria"])
                assert all(alias.startswith("c") for alias in aliases)
                denominator = sum(range(1, len(aliases) + 1))
                answers[question_id] = {
                    "type": "choice",
                    "choice": aliases[0],
                    "confidence": 1.0,
                    "probabilities": {
                        alias: (len(aliases) - position) / denominator
                        for position, alias in enumerate(aliases)
                    },
                }
            request = f"synthetic-{self.instance}-{len(self.calls)}"
            return JevApiResult(
                model="jev-1.14.0",
                answers=answers,
                input_tokens=100,
                output_tokens=10,
                request_id=request,
                cache_hit=False,
                request_latency_ms=1.0,
                request_sha256=hashlib.sha256(request.encode()).hexdigest(),
                cache_path=str(self.settings.cache_dir / f"{request}.json"),
            )

        def close(self):
            self.closed = True

    monkeypatch.setattr(tournament_module, "JevApiClient", FakeClient)
    return clients


def receipts(study_path):
    return [
        json.loads(path.read_text())["prediction"]
        for path in sorted(study_path.parent.glob("receipts/**/*.json"))
    ]


def test_prepare_and_whole_study_preflight_are_offline_and_include_both_phases(
    ablation_fixture, monkeypatch
):
    path, _, _, _, _ = ablation_fixture
    monkeypatch.setattr(tournament_module, "JevApiClient", forbid)
    monkeypatch.setattr("dotenv.load_dotenv", forbid)
    frozen = ablation.prepare_ablation(path)
    result = ablation.preflight_ablation(frozen)
    assert result["api_requests_made"] == 0
    assert result["selected_decisions"] == 2 * 7 * 3 + 3 * 3
    assert result["completed_decisions"] == 0
    assert result["remaining_requests_no_cache_assumed"] == result["selected_decisions"]
    assert result["by_phase"]["small"]["decisions"] == 42
    assert result["by_phase"]["full"]["decisions"] == 9
    assert result["estimated_remaining_input_cost_usd"] == pytest.approx(
        result["estimated_total_input_tokens"] * 0.042 / 1e6
    )
    study = ablation.load_study(frozen)
    assert sum(row["task_weight"] for row in study["phases"]["full"]["cases"]) == 10
    assert (
        sum(len(row["candidate_ids"]) for row in study["phases"]["full"]["cases"])
        == 274
    )
    assert not list(frozen.parent.glob("receipts/**/*.json"))
    assert not (frozen.parent / "jev-cache").exists()


def test_partial_configuration_cannot_omit_full_benchmark(ablation_fixture):
    path, settings, _, _, _ = ablation_fixture
    del settings["phases"]["full"]
    write_settings(path, settings)
    with pytest.raises(ValueError, match="both small and full"):
        ablation.prepare_ablation(path)


def test_presentation_drift_is_rejected_before_environment_or_api(
    ablation_fixture, monkeypatch
):
    path, _, _, _, _ = ablation_fixture
    frozen = ablation.prepare_ablation(path)
    original = ablation._presentation

    def changed_order(*args, **kwargs):
        workload, candidates = original(*args, **kwargs)
        return workload, tuple(reversed(candidates))

    # Simulate a changed selection/order implementation, including platform
    # differences, while keeping source inputs and candidate pools unchanged.
    monkeypatch.setattr(ablation, "_presentation", changed_order)
    monkeypatch.setattr(tournament_module, "JevApiClient", forbid)
    with pytest.raises(ValueError, match="differs from frozen plan"):
        ablation.run_ablation(
            frozen,
            execute=True,
            phase="small",
            variants=["a0"],
            repeats=[1],
            max_workloads=1,
            budget_usd=1,
            load_environment=forbid,
        )


def test_factorial_changes_only_descriptions_and_preserves_all_candidate_pools(
    ablation_fixture, fake_clients
):
    path, _, _, _, _ = ablation_fixture
    frozen = ablation.prepare_ablation(path)
    ablation.run_ablation(
        frozen,
        execute=True,
        phase="small",
        variants=["a0", "a1", "a2", "a3"],
        repeats=[1],
        max_workloads=1,
        budget_usd=1,
    )
    calls = [call for client in fake_clients for call in client.calls]
    assert len(calls) == 4
    contexts = [call["state"]["context"] for call in calls]
    assert "configuration_fields" not in contexts[0]
    assert "graph_fields" not in contexts[0]
    assert set(contexts[1]) - set(contexts[0]) == {"configuration_fields"}
    assert set(contexts[2]) - set(contexts[0]) == {"graph_fields"}
    assert set(contexts[3]) - set(contexts[0]) == {
        "configuration_fields",
        "graph_fields",
    }
    for context in contexts[1:]:
        assert all(context[key] == value for key, value in contexts[0].items())
    assert "graph_ids" not in contexts[1]["configuration_fields"]["indices"]
    questions = [next(iter(call["questions"].values())) for call in calls]
    assert all(question == questions[0] for question in questions[1:])
    assert len(questions[0]["criteria"]) == 50
    assert len({tuple(row["candidate_ids"]) for row in receipts(frozen)}) == 1
    assert len({tuple(row["shown_candidate_ids"]) for row in receipts(frozen)}) == 1


def test_repeats_make_fresh_calls_in_distinct_cache_directories(
    ablation_fixture, fake_clients
):
    path, _, _, _, _ = ablation_fixture
    frozen = ablation.prepare_ablation(path)
    result = ablation.run_ablation(
        frozen, execute=True, phase="small", variants=["a0"], budget_usd=1
    )
    assert sum(len(client.calls) for client in fake_clients) == 6
    assert all(client.closed for client in fake_clients)
    cache_dirs = {client.settings.cache_dir for client in fake_clients}
    assert {directory.name for directory in cache_dirs} == {"rep1", "rep2", "rep3"}
    assert len(cache_dirs) == 3
    assert result["completed_decisions"] == 6
    assert result["remaining_requests_no_cache_assumed"] == 0
    assert result["recorded_live_input_cost_usd"] == pytest.approx(600 * 0.042 / 1e6)
    grouped = {}
    for row in receipts(frozen):
        grouped.setdefault(row["workload_id"], []).append(row)
    assert all(
        {row["repeat"] for row in rows} == {1, 2, 3} for rows in grouped.values()
    )
    assert all(
        len({tuple(row["ranking"]) for row in rows}) == 1 for rows in grouped.values()
    )


def test_filter_keeps_original_oracle_and_does_not_score_synthetic_tail_as_pairwise(
    ablation_fixture, fake_clients
):
    path, settings, source, document, _ = ablation_fixture
    # Give hidden candidates faster labels before freezing the study. The filter
    # never sees this assignment; an erroneously filtered scoring denominator
    # would report a perfect score instead of 0.01.
    cases = TenSetDataset(
        {"path": str(source), "split": "test", "candidate_limit": 50, "seed": 42}
    ).load()
    shown_by_workload = {
        case.workload.workload_id: {
            candidate.candidate_id
            for candidate in ablation._presentation(case, "a5", 10, 42)[1]
        }
        for case in cases
    }
    for workload in document["workloads"]:
        for candidate in workload["candidates"]:
            candidate["runtime_ms"] = (
                100.0
                if candidate["candidate_id"]
                in shown_by_workload[workload["workload_id"]]
                else 1.0
            )
    source.write_text(json.dumps(document))
    settings["phases"]["small"]["expected_sha256"] = digest(source)
    write_settings(path, settings)
    frozen = ablation.prepare_ablation(path)
    ablation.run_ablation(
        frozen, execute=True, phase="small", variants=["a5"], repeats=[1], budget_usd=1
    )
    for row in receipts(frozen):
        assert len(row["candidate_ids"]) == len(row["ranking"]) == 50
        assert len(row["shown_candidate_ids"]) == 10
        assert set(row["ranking"][:10]) == set(row["shown_candidate_ids"])
        assert min(row["runtimes_ms"].values()) == 1.0
        assert row["metrics"]["endpoint_top_1_score"] == pytest.approx(0.01)
        assert row["metrics"]["endpoint_top_5_score"] == pytest.approx(0.01)
        assert "pairwise_accuracy" not in row["metrics"]


@pytest.mark.parametrize("execute,budget", [(False, 1), (True, 1e-12)])
def test_opt_in_and_input_budget_fail_before_environment_or_client(
    ablation_fixture, monkeypatch, execute, budget
):
    path, _, _, _, _ = ablation_fixture
    frozen = ablation.prepare_ablation(path)
    monkeypatch.setattr(tournament_module, "JevApiClient", forbid)
    with pytest.raises(ValueError, match="execute|budget"):
        ablation.run_ablation(
            frozen, execute=execute, budget_usd=budget, load_environment=forbid
        )


def test_single_question_requirement_fails_during_free_preparation(
    ablation_fixture, monkeypatch
):
    path, settings, _, _, _ = ablation_fixture
    settings["model"]["max_options"] = 10
    write_settings(path, settings)
    monkeypatch.setattr(tournament_module, "JevApiClient", forbid)
    with pytest.raises(ValueError, match="one question"):
        ablation.prepare_ablation(path)
    assert not (Path(settings["study"]["output_dir"]) / "study.json").exists()


def test_changed_source_is_rejected_before_environment_or_client(
    ablation_fixture, monkeypatch
):
    path, _, source, document, _ = ablation_fixture
    frozen = ablation.prepare_ablation(path)
    document["workloads"][0]["candidates"][0]["runtime_ms"] *= 2
    source.write_text(json.dumps(document))
    monkeypatch.setattr(tournament_module, "JevApiClient", forbid)
    with pytest.raises(ValueError, match="checksum"):
        ablation.run_ablation(
            frozen, execute=True, budget_usd=1, load_environment=forbid
        )


def test_rehashed_source_cannot_hide_changed_private_labels(ablation_fixture):
    path, _, source, document, _ = ablation_fixture
    frozen = ablation.prepare_ablation(path)
    document["workloads"][0]["candidates"][0]["runtime_ms"] *= 2
    source.write_text(json.dumps(document))
    study = json.loads(frozen.read_text())
    study["phases"]["small"]["expected_sha256"] = digest(source)
    study["identity"] = ablation._fingerprint(
        {key: value for key, value in study.items() if key != "identity"}
    )
    frozen.write_text(json.dumps(study))
    with pytest.raises(ValueError, match="label inventory changed"):
        ablation.preflight_ablation(frozen)


def test_frozen_manifest_and_implementation_changes_are_rejected(
    ablation_fixture, monkeypatch
):
    path, _, _, _, _ = ablation_fixture
    frozen = ablation.prepare_ablation(path)
    original = frozen.read_text()
    study = json.loads(original)
    study["seed"] += 1
    frozen.write_text(json.dumps(study))
    with pytest.raises(ValueError, match="identity changed"):
        ablation.preflight_ablation(frozen)
    frozen.write_text(original)
    hashes = ablation._sources()
    hashes["models/jev.py"] = "changed"
    monkeypatch.setattr(ablation, "_sources", lambda: hashes)
    with pytest.raises(ValueError, match="code differs"):
        ablation.preflight_ablation(frozen)


def test_completed_receipts_resume_without_loading_environment_or_client(
    ablation_fixture, fake_clients, monkeypatch
):
    path, _, _, _, _ = ablation_fixture
    frozen = ablation.prepare_ablation(path)
    options = {
        "execute": True,
        "phase": "small",
        "variants": ["a1"],
        "repeats": [1],
        "budget_usd": 1,
    }
    ablation.run_ablation(frozen, **options)
    assert sum(len(client.calls) for client in fake_clients) == 2
    monkeypatch.setattr(tournament_module, "JevApiClient", forbid)
    with pytest.raises(FileExistsError, match="resume"):
        ablation.run_ablation(frozen, load_environment=forbid, **options)
    result = ablation.run_ablation(
        frozen, resume=True, load_environment=forbid, **options
    )
    assert result["completed_decisions"] == 2
    assert result["remaining_requests_no_cache_assumed"] == 0


def test_tampered_completion_receipt_fails_before_environment_or_client(
    ablation_fixture, fake_clients, monkeypatch
):
    path, _, _, _, _ = ablation_fixture
    frozen = ablation.prepare_ablation(path)
    options = {
        "execute": True,
        "phase": "small",
        "variants": ["a0"],
        "repeats": [1],
        "max_workloads": 1,
        "budget_usd": 1,
    }
    ablation.run_ablation(frozen, **options)
    receipt_path = next(frozen.parent.glob("receipts/**/*.json"))
    receipt = json.loads(receipt_path.read_text())
    receipt["prediction"]["metrics"]["top_1_score"] = 1.0
    receipt_path.write_text(json.dumps(receipt))
    monkeypatch.setattr(tournament_module, "JevApiClient", forbid)
    with pytest.raises(ValueError, match="receipt was changed"):
        ablation.run_ablation(frozen, resume=True, load_environment=forbid, **options)


def test_full_followup_scores_all_source_records_and_keeps_frequency_weights(
    ablation_fixture, fake_clients
):
    path, _, _, _, _ = ablation_fixture
    frozen = ablation.prepare_ablation(path)
    ablation.run_ablation(frozen, execute=True, phase="full", budget_usd=1)
    assert sum(len(client.calls) for client in fake_clients) == 9
    rows = receipts(frozen)
    assert len(rows) == 9
    assert sum(len(row["candidate_ids"]) for row in rows) == 274 * 3
    assert sum(row["task_weight"] for row in rows) == 10 * 3
    assert {row["runtime_basis"] for row in rows} == {
        "paper_f32_recovered_throughput_floor"
    }
    for row in rows:
        assert row["phase"] == "full"
        assert row["variant"] == "a5"
        assert len(row["shown_candidate_ids"]) == 5
        selected = row["ranking"][0]
        expected = min(row["runtimes_ms"].values()) / row["runtimes_ms"][selected]
        assert row["metrics"]["endpoint_top_1_score"] == pytest.approx(expected)
        assert row["paper_normalizer_ms"] > 0
        assert "pairwise_accuracy" not in row["metrics"]


def test_partial_pilot_calibrates_usage_and_resume_calls_only_remaining_workloads(
    ablation_fixture, fake_clients
):
    path, _, _, _, _ = ablation_fixture
    frozen = ablation.prepare_ablation(path)
    options = {
        "execute": True,
        "phase": "small",
        "variants": ["a1"],
        "repeats": [1],
        "budget_usd": 1,
    }
    ablation.run_ablation(frozen, max_workloads=1, **options)
    assert sum(len(client.calls) for client in fake_clients) == 1
    assert not list(frozen.parent.glob("runs/**/predictions.jsonl"))
    preflight = ablation.preflight_ablation(
        frozen, phase="small", variants=["a1"], repeats=[1], calibrate_from_pilot=True
    )
    assert preflight["completed_decisions"] == 1
    assert preflight["remaining_requests_no_cache_assumed"] == 1
    row = receipts(frozen)[0]
    ratio = 100 / row["request_reports"][0]["estimated_input_tokens"]
    assert preflight["pilot_calibration"][
        "actual_to_estimated_token_ratio"
    ] == pytest.approx(ratio)
    assert preflight["pilot_calibration"][
        "projected_remaining_input_cost_usd"
    ] == pytest.approx(preflight["estimated_remaining_input_cost_usd"] * ratio)
    result = ablation.run_ablation(frozen, resume=True, **options)
    assert result["completed_decisions"] == 2
    assert sum(len(client.calls) for client in fake_clients) == 2
    published = list(frozen.parent.glob("runs/**/predictions.jsonl"))
    assert len(published) == 1
    assert len(published[0].read_text().splitlines()) == 2


def test_pairwise_uses_probabilities_and_top_k_preserves_explicit_choice(
    ablation_fixture, fake_clients, monkeypatch
):
    path, settings, source, document, _ = ablation_fixture
    # Measured runtimes ascend in the dataset candidate order. Strict pairwise
    # comparison with all predicted ties must therefore be zero, while the
    # separate tie-aware diagnostic must be 0.5. Presentation is still shuffled.
    cases = TenSetDataset(
        {"path": str(source), "split": "test", "candidate_limit": 50, "seed": 42}
    ).load()
    labels = {
        case.workload.workload_id: {
            candidate.candidate_id: float(position + 1)
            for position, candidate in enumerate(case.candidates)
        }
        for case in cases
    }
    for workload in document["workloads"]:
        for candidate in workload["candidates"]:
            candidate["runtime_ms"] = labels[workload["workload_id"]][
                candidate["candidate_id"]
            ]
    source.write_text(json.dumps(document))
    settings["phases"]["small"]["expected_sha256"] = digest(source)
    write_settings(path, settings)
    client_type = tournament_module.JevApiClient
    decide = client_type.decide

    def unusual_answer(self, **kwargs):
        response = decide(self, **kwargs)
        answers = copy.deepcopy(response.answers)
        for answer in answers.values():
            aliases = list(answer["probabilities"])
            answer["choice"] = aliases[-1]
            answer["probabilities"] = dict.fromkeys(aliases, 1 / len(aliases))
        return replace(response, answers=answers)

    monkeypatch.setattr(client_type, "decide", unusual_answer)
    frozen = ablation.prepare_ablation(path)
    ablation.run_ablation(
        frozen,
        execute=True,
        phase="small",
        variants=["a0"],
        repeats=[1],
        max_workloads=1,
        budget_usd=1,
    )
    row = receipts(frozen)[0]
    assert row["ranking"][0] == row["shown_candidate_ids"][-1]
    runtimes = row["runtimes_ms"]
    selected = row["ranking"][0]
    expected_top_1 = min(runtimes.values()) / runtimes[selected]
    expected_top_5 = min(runtimes.values()) / min(
        runtimes[key] for key in row["ranking"][:5]
    )
    assert row["metrics"]["endpoint_top_1_score"] == pytest.approx(expected_top_1)
    assert row["metrics"]["endpoint_top_5_score"] == pytest.approx(expected_top_5)
    assert row["metrics"]["pairwise_accuracy"] == 0.0
    assert row["metrics"]["pairwise_accuracy_tie_aware"] == 0.5


def test_scorer_keeps_actual_top_k_separate_from_probability_order():
    # The shared API parser currently rejects a choice below the maximum
    # probability. This direct scorer contract still ensures that computing
    # pairwise diagnostics never changes the original decision's Top-K scores.
    case = EvaluationCase(
        Workload("synthetic", {}),
        tuple(Candidate(key, {}) for key in ("fast", "middle", "slow")),
        tuple(
            RuntimeRecord("synthetic", key, value)
            for key, value in (("fast", 1.0), ("middle", 2.0), ("slow", 4.0))
        ),
    )
    result = DecisionResult(
        ("slow", "fast", "middle"),
        1.0,
        {"last_question_probabilities": {"fast": 0.6, "middle": 0.3, "slow": 0.1}},
    )
    metrics = ablation._metrics(case, result, filtered=False)
    assert metrics["endpoint_top_1_score"] == 0.25
    assert metrics["endpoint_top_5_score"] == 1.0
    assert metrics["pairwise_accuracy"] == metrics["pairwise_accuracy_tie_aware"] == 1.0
    assert result.ranked_candidate_ids == ("slow", "fast", "middle")


@pytest.mark.parametrize("field", ["runtimes_ms", "task_weight"])
def test_rehashed_receipt_cannot_change_private_labels_or_network_weights(
    ablation_fixture, fake_clients, monkeypatch, field
):
    path, _, _, _, _ = ablation_fixture
    frozen = ablation.prepare_ablation(path)
    options = {
        "execute": True,
        "phase": "full",
        "variants": ["a5"],
        "repeats": [1],
        "max_workloads": 1,
        "budget_usd": 1,
    }
    ablation.run_ablation(frozen, **options)
    receipt_path = next(frozen.parent.glob("receipts/**/*.json"))
    receipt = json.loads(receipt_path.read_text())
    row = receipt["prediction"]
    if field == "runtimes_ms":
        first = row["candidate_ids"][0]
        row["runtimes_ms"][first] *= 2
    else:
        row["task_weight"] += 1
    receipt["prediction_sha256"] = ablation._fingerprint(row)
    receipt_path.write_text(json.dumps(receipt))
    monkeypatch.setattr(tournament_module, "JevApiClient", forbid)
    with pytest.raises(ValueError, match="receipt labels or ranking differ"):
        ablation.run_ablation(frozen, resume=True, load_environment=forbid, **options)


def test_concurrent_execution_lock_fails_before_environment_or_client(
    ablation_fixture, monkeypatch
):
    path, _, _, _, _ = ablation_fixture
    frozen = ablation.prepare_ablation(path)
    monkeypatch.setattr(tournament_module, "JevApiClient", forbid)
    with (frozen.parent / ".execution.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with pytest.raises(RuntimeError, match="holds the study lock"):
                ablation.run_ablation(
                    frozen, execute=True, budget_usd=1, load_environment=forbid
                )
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def test_cli_prepare_and_preflight_are_offline_without_api_credentials(
    ablation_fixture, tmp_path
):
    path, settings, _, _, _ = ablation_fixture
    # The child process inherits no pytest monkeypatches. Its startup hook makes
    # credential loading or SDK construction a test failure before any network
    # action, while the actual CLI imports and executes normally.
    bootstrap = tmp_path / "offline-bootstrap"
    bootstrap.mkdir()
    (bootstrap / "sitecustomize.py").write_text(
        "import dotenv\nimport typesafe_sdk\n"
        "def forbid(*args, **kwargs):\n"
        "    raise AssertionError('offline CLI loaded credentials or SDK')\n"
        "class ForbiddenClient:\n"
        "    def __init__(self, *args, **kwargs):\n"
        "        forbid()\n"
        "dotenv.load_dotenv = forbid\n"
        "typesafe_sdk.TypeSafeClient = ForbiddenClient\n"
    )
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"TYPESAFE_API_KEY", "JEV_API_KEY", "OPENAI_API_KEY"}
    }
    environment["PYTHONPATH"] = str(bootstrap)
    frozen = Path(settings["study"]["output_dir"]) / "study.json"
    commands = (
        [
            sys.executable,
            str(ROOT / "scripts/prepare_tenset_ablation.py"),
            "--config",
            str(path),
        ],
        [
            sys.executable,
            str(ROOT / "scripts/run_tenset_ablation.py"),
            "--study",
            str(frozen),
            "--preflight-only",
        ],
    )
    for command in commands:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        data, _ = json.JSONDecoder().raw_decode(
            result.stdout[result.stdout.index("{") :]
        )
        assert data["api_requests_made"] == 0
        assert data["selected_decisions"] == 51
        assert "no API client or environment file was loaded" in result.stdout
    assert not list(frozen.parent.glob("receipts/**/*.json"))


def test_real_runner_pilot_completion_summary_and_portable_team_archive(
    ablation_fixture, fake_clients, tmp_path
):
    path, _, _, _, _ = ablation_fixture
    frozen = ablation.prepare_ablation(path)
    root = frozen.parent

    # A pilot has only a completion receipt, not a completed run.json bundle.
    ablation.run_ablation(
        frozen,
        execute=True,
        phase="full",
        variants=["a5"],
        repeats=[1],
        max_workloads=1,
        budget_usd=1,
    )
    assert not list(root.glob("runs/**/predictions.jsonl"))
    pilot = summarize_ablation(
        root,
        output_dir=root / "pilot-summary",
        create_plots=False,
        allow_incomplete=True,
    )
    assert pilot["decisions"] == pilot["runs"] == 1
    assert pilot["warnings"]
    with (Path(pilot["output_dir"]) / "runs.csv").open() as stream:
        pilot_runs = list(csv.DictReader(stream))
    assert "pilot_network_top_1_score" in pilot_runs[0]
    assert "paper_network_top_1_score" not in pilot_runs[0]
    assert pilot_runs[0]["complete_network_coverage"] == "False"

    # Resume the entire real engine inventory: seven small conditions with
    # three repeats and a three-repeat full-network follow-up.
    ablation.run_ablation(frozen, execute=True, resume=True, budget_usd=1)
    assert sum(len(client.calls) for client in fake_clients) == 51
    summary = summarize_ablation(root, create_plots=False)
    assert summary["decisions"] == 51
    assert summary["runs"] == 24
    assert summary["variants"] == 8
    assert summary["warnings"] == summary["unreported_phases"] == []
    assert summary["usage"]["live_requests"] == 51
    assert summary["usage"]["live_input_tokens"] == 5100
    assert summary["input_cost_estimate"]["known_live_input_cost_usd"] == pytest.approx(
        5100 * 0.042 / 1e6
    )
    output = Path(summary["output_dir"])
    with (output / "runs.csv").open() as stream:
        runs = list(csv.DictReader(stream))
    full = [row for row in runs if row["phase"] == "full"]
    assert len(full) == 3
    saved = receipts(frozen)
    for run in full:
        members = [
            row
            for row in saved
            if row["phase"] == "full" and row["repeat"] == int(run["repeat"])
        ]
        assert len(members) == 3
        numerator = sum(
            row["task_weight"] * row["paper_normalizer_ms"] for row in members
        )
        for k in (1, 5):
            denominator = sum(
                row["task_weight"]
                * min(row["runtimes_ms"][candidate] for candidate in row["ranking"][:k])
                for row in members
            )
            assert float(run[f"paper_network_top_{k}_score"]) == pytest.approx(
                numerator / denominator
            )
        assert run["complete_network_coverage"] == "True"
        assert float(run["network_task_weight"]) == 10
    with (output / "paired.csv").open() as stream:
        pairs = list(csv.DictReader(stream))
    assert {("a0", "a1"), ("a0", "a2"), ("a0", "a3")} <= {
        (row["left"], row["right"]) for row in pairs
    }

    archive_path = package_ablation(root, tmp_path / "team-review.zip")
    review = tmp_path / "another-computer"
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        assert {
            "study.json",
            "config.json",
            "summary/summary.json",
            "SHARING-NOTES.md",
        } <= names
        assert ".execution.lock" not in names
        assert any(
            name.startswith("runs/full/") and name.endswith("predictions.jsonl")
            for name in names
        )
        assert any(name.startswith("receipts/small/") for name in names)
        archive.extractall(review)
    # Analysis rechecks portable run checksums, frozen pools, and completion
    # receipts without requiring original absolute data paths or live clients.
    reviewed = summarize_ablation(
        review, output_dir=review / "review-summary", create_plots=False
    )
    assert reviewed["decisions"] == summary["decisions"]
    assert reviewed["warnings"] == []
    assert reviewed["input_cost_estimate"] == summary["input_cost_estimate"]
