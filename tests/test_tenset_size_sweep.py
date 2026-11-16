"""Frozen nested schedule pools; synthetic measurements are not benchmark data."""

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean

import pytest
import yaml

import jev_compiler_eval.evaluation.tenset_size_sweep as sweep
from jev_compiler_eval.datasets.tenset import TenSetDataset, _digest, workload_split
from jev_compiler_eval.evaluation.runner import _model_input
from jev_compiler_eval.metrics import resolve_metrics
from jev_compiler_eval.models.random import RandomModel
from jev_compiler_eval.types import DecisionResult


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_prepared(path, count):
    keys = (
        json.dumps([f"synthetic_workload_{number}", [16, 8]])
        for number in range(count * 20)
    )
    keys = [key for key in keys if workload_split(key, seed=42) == "test"][:count]
    assert len(keys) == count
    workloads = []
    for key in keys:
        identifier = "tenset-" + _digest([key, sweep.TARGET])[:24]
        candidates = []
        for index in range(50):
            state = [[], [["SP", 0, 0, 16, [index + 1, 1], 1]]]
            candidates.append(
                {
                    "candidate_id": "schedule-" + _digest(state)[:24],
                    "configuration": {
                        "initial_state": [],
                        "transformations": [
                            {
                                "code": "SP",
                                "name": "split",
                                "arguments": state[1][0][1:],
                            }
                        ],
                    },
                    "record_input": [[key, sweep.TARGET], state],
                    "runtime_ms": float(index + 1),
                }
            )
        workloads.append(
            {
                "workload_id": identifier,
                "split": "test",
                "context": {
                    "format": "tvm-auto-scheduler-json",
                    "workload_key": key,
                    "target": sweep.TARGET,
                    "operation": "synthetic_matmul",
                    "tensors": [{"shape": [16, 8], "dtype": "float32"}],
                    "compute_dag": "Synthetic compute DAG, not benchmark data.",
                    "loop_structure": [],
                },
                "candidates": candidates,
            }
        )
    document = {
        "schema_version": 1,
        "dataset": "tenset",
        "provenance": {"synthetic": True, "target": sweep.TARGET},
        "workloads": workloads,
    }
    path.write_text(json.dumps(document))
    return document


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, "COHORT_SIZE", 3)
    prepared = tmp_path / "prepared.json"
    document = make_prepared(prepared, 3)
    kwargs = {
        "output_dir": tmp_path / "profiles",
        "results_root": tmp_path / "results",
        "expected_prepared_sha256": checksum(prepared),
    }
    return prepared, document, kwargs


def write_baseline(prepared, directory):
    directory.mkdir()
    config = sweep._configuration(prepared, 50, directory.parent)
    config["experiment"]["output_dir"] = str(directory)
    (directory / "experiment.yaml").write_text(yaml.safe_dump(config))
    manifest = {
        "dataset_type": "tenset",
        "seed": 42,
        "model_ids": [model["id"] for model in config["models"]],
        "dataset_provenance": {"prepared_file_sha256": checksum(prepared)},
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    cases = TenSetDataset(sweep._dataset_config(prepared, 50)).load()
    functions = resolve_metrics(sweep.METRICS)
    rows = []
    for specification in config["models"]:
        for case in cases:
            ids = [candidate.candidate_id for candidate in case.candidates]
            runtimes = {
                record.candidate_id: record.runtime_ms for record in case.runtimes
            }
            if specification["type"] == "random":
                result = RandomModel({"seed": 42}).rank(case.workload, case.candidates)
            else:
                total = sum(range(1, 51))
                result = DecisionResult(
                    tuple(reversed(ids)),
                    1.0,
                    {
                        "requested_model": sweep.MODEL,
                        "resolved_model": sweep.MODEL,
                        "long_context": True,
                        "probabilities": {
                            candidate: (index + 1) / total
                            for index, candidate in enumerate(ids)
                        },
                    },
                )
            rows.append(
                {
                    "model_id": specification["id"],
                    "model_type": specification["type"],
                    "workload_id": case.workload.workload_id,
                    "candidate_ids": ids,
                    "predicted_ranking": list(result.ranked_candidate_ids),
                    "selected_candidate_id": result.selected_candidate_id,
                    "selected_runtime_ms": runtimes[result.selected_candidate_id],
                    "runtime_by_candidate_ms": runtimes,
                    "model_input_sha256": hashlib.sha256(
                        _model_input(case)
                    ).hexdigest(),
                    "model_input_bytes": len(_model_input(case)),
                    "decision_latency_ms": result.decision_latency_ms,
                    "metadata": dict(result.metadata),
                    "metrics": {
                        name: function(case, result)
                        for name, function in functions.items()
                    },
                }
            )
    (directory / "predictions.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n"
    )
    return config, rows


def test_preparation_freezes_same_workloads_nested_pools_and_all_metrics(
    frozen, monkeypatch
):
    prepared, _, kwargs = frozen

    def forbid(*args, **kwargs):
        raise AssertionError("offline sweep preparation created an API client")

    monkeypatch.setattr("jev_compiler_eval.models.jev.JevApiClient", forbid)
    path = sweep.prepare_tenset_size_sweep(prepared, **kwargs)
    plan = json.loads(path.read_text())
    assert plan["status"] == "prepared_no_api_calls"
    assert plan["prepared_sha256"] == checksum(prepared)
    assert plan["target"] == sweep.TARGET
    assert plan["seed"] == 42 and plan["field_descriptions"] is False
    assert len(plan["budgets"]) == 3
    ids = []
    for size, budget in zip((16, 32, 50), plan["budgets"], strict=True):
        config = yaml.safe_load(Path(budget["config_path"]).read_text())
        assert set(config["metrics"]) == set(sweep.METRICS)
        assert config["dataset"]["config"]["candidate_limit"] == size
        assert config["models"][1]["type"] == "jev"
        assert config["models"][1]["config"]["long_context"] is True
        assert config["models"][1]["config"]["model"] == sweep.MODEL
        assert len(budget["workloads"]) == 3
        ids.append([row["workload_id"] for row in budget["workloads"]])
    assert ids[0] == ids[1] == ids[2]
    for rows in zip(*(row["workloads"] for row in plan["budgets"]), strict=True):
        assert rows[0]["candidate_ids"] == rows[2]["candidate_ids"][:16]
        assert rows[1]["candidate_ids"] == rows[2]["candidate_ids"][:32]


def test_canonical_templates_match_generator():
    root = Path(__file__).resolve().parents[1]
    for size in (16, 32, 50):
        path = (
            root
            / "experiments/tenset/size-sweep"
            / f"jev-vs-random-w120-c{size}-s42.yaml"
        )
        assert yaml.safe_load(path.read_text()) == sweep._configuration(
            Path("data/tenset/prepared-w120-c50-s42-metadata.json"),
            size,
            Path("results/tenset/size-sweep"),
        )


def test_default_requires_the_audited_input_checksum(frozen):
    prepared, _, kwargs = frozen
    kwargs.pop("expected_prepared_sha256")
    with pytest.raises(ValueError, match="checksum"):
        sweep.prepare_tenset_size_sweep(prepared, **kwargs)
    assert not kwargs["output_dir"].exists()


def test_default_cohort_is_exactly_120(tmp_path):
    prepared = tmp_path / "prepared.json"
    make_prepared(prepared, 120)
    path = sweep.prepare_tenset_size_sweep(
        prepared,
        expected_prepared_sha256=checksum(prepared),
        output_dir=tmp_path / "profiles",
        results_root=tmp_path / "results",
    )
    assert len(json.loads(path.read_text())["budgets"][0]["workloads"]) == 120


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda d: d["workloads"].pop(), "exactly"),
        (lambda d: d["workloads"][0].update(split="train"), "test cohort"),
        (
            lambda d: d["workloads"][0]["context"].update(target="llvm -model=other"),
            "test cohort",
        ),
        (
            lambda d: d["workloads"][0]["context"].update(configuration_fields={}),
            "no field definitions",
        ),
        (lambda d: d["workloads"][0]["candidates"].pop(), "at least 50"),
        (
            lambda d: d["workloads"][0]["candidates"][0].update(runtime_ms=1e13),
            "valid-only",
        ),
        (
            lambda d: d["workloads"][0]["candidates"][0].update(source_error_code=1),
            "valid-only",
        ),
        (
            lambda d: d["workloads"][0]["candidates"][0].update(official_throughput=1),
            "valid-only",
        ),
    ],
)
def test_incompatible_cohorts_fail_before_writing(frozen, mutate, match):
    prepared, document, kwargs = frozen
    mutate(document)
    prepared.write_text(json.dumps(document))
    kwargs["expected_prepared_sha256"] = checksum(prepared)
    with pytest.raises(ValueError, match=match):
        sweep.prepare_tenset_size_sweep(prepared, **kwargs)
    assert not kwargs["output_dir"].exists()


@pytest.mark.parametrize(
    "sizes",
    [(), (50,), (32, 16, 50), (16, 16, 50), (16, 32), (0, 50), (51,), (True, 50)],
)
def test_invalid_budgets_fail_closed(frozen, sizes):
    prepared, _, kwargs = frozen
    with pytest.raises(ValueError, match="sizes must"):
        sweep.prepare_tenset_size_sweep(prepared, sizes=sizes, **kwargs)


def test_repeated_preparation_is_identical_and_never_overwrites(frozen):
    prepared, _, kwargs = frozen
    path = sweep.prepare_tenset_size_sweep(prepared, **kwargs)
    original = path.read_bytes()
    mtime = path.stat().st_mtime_ns
    assert sweep.prepare_tenset_size_sweep(prepared, **kwargs) == path
    assert path.read_bytes() == original and path.stat().st_mtime_ns == mtime
    profile = next(kwargs["output_dir"].glob("*.yaml"))
    profile.write_text("different existing content\n")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        sweep.prepare_tenset_size_sweep(prepared, **kwargs)
    assert profile.read_text() == "different existing content\n"


def test_preparation_detects_input_change_before_publishing(frozen, monkeypatch):
    prepared, _, kwargs = frozen
    original = sweep._sha256
    calls = []

    def changing(path):
        if path == prepared:
            calls.append(path)
            if len(calls) > 1:
                return "0" * 64
        return original(path)

    monkeypatch.setattr(sweep, "_sha256", changing)
    with pytest.raises(ValueError, match="changed during"):
        sweep.prepare_tenset_size_sweep(prepared, **kwargs)
    assert not kwargs["output_dir"].exists()


def test_existing_50_result_is_verified_and_left_untouched(frozen, tmp_path):
    prepared, _, kwargs = frozen
    directory = tmp_path / "anchor"
    write_baseline(prepared, directory)
    before = {path.name: checksum(path) for path in directory.iterdir()}
    path = sweep.prepare_tenset_size_sweep(
        prepared, baseline_results=directory, **kwargs
    )
    plan = json.loads(path.read_text())
    assert plan["budgets"][-1]["action"] == "reuse_verified_anchor"
    assert len(list(kwargs["output_dir"].glob("*.yaml"))) == 2
    assert str(directory) in plan["analysis_command"]
    assert {path.name: checksum(path) for path in directory.iterdir()} == before


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("candidate_ids", ["wrong"], "candidate pool"),
        ("model_input_sha256", "0" * 64, "input hash"),
        ("runtime_by_candidate_ms", {}, "runtimes"),
        ("model_type", "jev-tournament", "unexpected rows"),
        ("predicted_ranking", ["wrong"], "ranking"),
        ("selected_candidate_id", "wrong", "selected outcome"),
    ],
)
def test_mismatched_anchor_cannot_be_reused(frozen, tmp_path, field, value, match):
    prepared, _, kwargs = frozen
    directory = tmp_path / "anchor"
    _, rows = write_baseline(prepared, directory)
    rows[0][field] = value
    (directory / "predictions.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n"
    )
    with pytest.raises(ValueError, match=match):
        sweep.prepare_tenset_size_sweep(prepared, baseline_results=directory, **kwargs)
    assert not kwargs["output_dir"].exists()


def test_wrong_baseline_seed_is_rejected(frozen, tmp_path):
    prepared, _, kwargs = frozen
    directory = tmp_path / "anchor"
    config, _ = write_baseline(prepared, directory)
    config["models"][0]["config"]["seed"] = 7
    (directory / "experiment.yaml").write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="seed differs"):
        sweep.prepare_tenset_size_sweep(prepared, baseline_results=directory, **kwargs)


def test_baseline_metric_corruption_is_rejected(frozen, tmp_path):
    prepared, _, kwargs = frozen
    directory = tmp_path / "anchor"
    _, rows = write_baseline(prepared, directory)
    rows[0]["metrics"]["top_1_score"] = 999
    (directory / "predictions.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n"
    )
    with pytest.raises(ValueError, match="metrics do not recompute"):
        sweep.prepare_tenset_size_sweep(prepared, baseline_results=directory, **kwargs)


def test_missing_baseline_rows_are_rejected(frozen, tmp_path):
    prepared, _, kwargs = frozen
    directory = tmp_path / "anchor"
    _, rows = write_baseline(prepared, directory)
    (directory / "predictions.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows[:-1]) + "\n"
    )
    with pytest.raises(ValueError, match="missing frozen"):
        sweep.prepare_tenset_size_sweep(prepared, baseline_results=directory, **kwargs)


def test_cli_validates_and_prints_only_offline_preparation(tmp_path):
    prepared = tmp_path / "prepared.json"
    make_prepared(prepared, 120)
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/prepare_tenset_size_sweep.py"),
            "--prepared",
            str(prepared),
            "--expected-prepared-sha256",
            checksum(prepared),
            "--output-dir",
            str(tmp_path / "profiles"),
            "--results-root",
            str(tmp_path / "results"),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "no model client was created" in completed.stdout
    assert "run_tenset_size_sweep.py" in completed.stdout
    assert "--preflight-only" in completed.stdout
    assert "analyze_tenset.py" in completed.stdout
    assert not (tmp_path / "results").exists()
    preflight = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/run_tenset_size_sweep.py"),
            "--manifest",
            str(tmp_path / "profiles/manifest.json"),
            "--candidate-count",
            "16",
            "--preflight-only",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert preflight.returncode == 0, preflight.stdout + preflight.stderr
    assert '"status": "offline_preflight_passed"' in preflight.stdout
    assert '"api_requests_made": 0' in preflight.stdout
    assert not (tmp_path / "results").exists()


def forbid_execution(*args, **kwargs):
    raise AssertionError("guarded/offline operation reached credentials or a model")


def test_member_preflight_never_loads_credentials_or_models(frozen, monkeypatch):
    prepared, _, kwargs = frozen
    manifest = sweep.prepare_tenset_size_sweep(prepared, **kwargs)
    monkeypatch.setattr(sweep, "run_experiment", forbid_execution)
    result = sweep.run_tenset_size_sweep_member(
        manifest, 16, preflight_only=True, load_environment=forbid_execution
    )
    assert result["status"] == "offline_preflight_passed"
    assert result["workloads"] == 3 and result["candidate_count"] == 16
    assert result["api_requests_made"] == 0
    assert not kwargs["results_root"].exists()


def test_changed_prepared_input_is_rejected_before_credentials(frozen, monkeypatch):
    prepared, _, kwargs = frozen
    manifest = sweep.prepare_tenset_size_sweep(prepared, **kwargs)
    prepared.write_text(prepared.read_text() + "\n")
    monkeypatch.setattr(sweep, "run_experiment", forbid_execution)
    with pytest.raises(ValueError, match="checksum"):
        sweep.run_tenset_size_sweep_member(
            manifest, 16, load_environment=forbid_execution
        )


def test_changed_config_is_rejected_before_credentials(frozen, monkeypatch):
    prepared, _, kwargs = frozen
    manifest = sweep.prepare_tenset_size_sweep(prepared, **kwargs)
    member = json.loads(manifest.read_text())["budgets"][0]
    path = Path(member["config_path"])
    settings = yaml.safe_load(path.read_text())
    settings["models"][1]["config"]["long_context"] = False
    path.write_text(yaml.safe_dump(settings))
    monkeypatch.setattr(sweep, "run_experiment", forbid_execution)
    with pytest.raises(ValueError, match="configuration bytes changed"):
        sweep.run_tenset_size_sweep_member(
            manifest, 16, load_environment=forbid_execution
        )


def test_changed_pool_manifest_is_rejected_before_credentials(frozen, monkeypatch):
    prepared, _, kwargs = frozen
    manifest = sweep.prepare_tenset_size_sweep(prepared, **kwargs)
    plan = json.loads(manifest.read_text())
    plan["budgets"][0]["workloads"][0]["candidate_ids"].reverse()
    manifest.write_text(json.dumps(plan))
    monkeypatch.setattr(sweep, "run_experiment", forbid_execution)
    with pytest.raises(ValueError, match="prefixes"):
        sweep.run_tenset_size_sweep_member(
            manifest, 16, load_environment=forbid_execution
        )


@pytest.mark.parametrize("guard", ["source", "python", "packages"])
def test_changed_implementation_or_environment_is_rejected(frozen, monkeypatch, guard):
    prepared, _, kwargs = frozen
    manifest = sweep.prepare_tenset_size_sweep(prepared, **kwargs)
    if guard == "source":
        monkeypatch.setattr(
            sweep, "_critical_sources", lambda: {"changed.py": "0" * 64}
        )
    elif guard == "python":
        monkeypatch.setattr(sweep.platform, "python_version", lambda: "3.11.changed")
    else:
        monkeypatch.setattr(
            sweep, "_package_versions", lambda: {"typesafe-sdk": "changed"}
        )
    monkeypatch.setattr(sweep, "run_experiment", forbid_execution)
    with pytest.raises(
        ValueError, match="manifest or preparation implementation changed"
    ):
        sweep.run_tenset_size_sweep_member(
            manifest, 16, load_environment=forbid_execution
        )


def test_member_executes_only_selected_profile_after_validation(frozen, monkeypatch):
    prepared, _, kwargs = frozen
    manifest = sweep.prepare_tenset_size_sweep(prepared, **kwargs)
    calls = []
    monkeypatch.setattr(
        sweep, "run_experiment", lambda path: calls.append(("run", path)) or "stubbed"
    )
    result = sweep.run_tenset_size_sweep_member(
        manifest, 32, load_environment=lambda: calls.append(("credentials", None))
    )
    assert result == "stubbed"
    assert [item[0] for item in calls] == ["credentials", "run"]
    assert "c32" in calls[1][1]


def test_credential_callback_cannot_change_inputs_before_model_start(
    frozen, monkeypatch
):
    prepared, _, kwargs = frozen
    manifest = sweep.prepare_tenset_size_sweep(prepared, **kwargs)
    monkeypatch.setattr(sweep, "run_experiment", forbid_execution)

    def change_input():
        prepared.write_text(prepared.read_text() + "\n")

    with pytest.raises(ValueError, match="checksum"):
        sweep.run_tenset_size_sweep_member(manifest, 16, load_environment=change_input)


def test_reused_member_is_read_only_and_revalidated_before_execution(
    frozen, tmp_path, monkeypatch
):
    prepared, _, kwargs = frozen
    directory = tmp_path / "anchor"
    write_baseline(prepared, directory)
    manifest = sweep.prepare_tenset_size_sweep(
        prepared, baseline_results=directory, **kwargs
    )
    monkeypatch.setattr(sweep, "run_experiment", forbid_execution)
    result = sweep.run_tenset_size_sweep_member(
        manifest, 50, load_environment=forbid_execution
    )
    assert result["status"] == "verified_existing_anchor_no_live_execution"
    prediction_path = directory / "predictions.jsonl"
    prediction_path.write_text(prediction_path.read_text() + "\n")
    with pytest.raises(ValueError, match="changed after preparation"):
        sweep.run_tenset_size_sweep_member(
            manifest, 50, load_environment=forbid_execution
        )


def write_synthetic_analysis(manifest_path, directory):
    plan = json.loads(manifest_path.read_text())
    directory.mkdir()
    rows = []
    for budget in plan["budgets"]:
        size = budget["candidates_per_workload"]
        for model, model_type, bump in (
            ("random-42", "random", 0),
            ("jev-1.14.0-long-context", "jev", 0.05),
        ):
            for index, workload in enumerate(budget["workloads"]):
                rows.append(
                    {
                        "candidate_count": size,
                        "model_id": model,
                        "model_type": model_type,
                        "workload_id": workload["workload_id"],
                        "seed": 42,
                        "top_1_score": 0.15 + size / 200 + bump,
                        "top_5_score": 0.5 + index / 100 + bump,
                        "pairwise_accuracy": 0.45 + bump,
                        "pairwise_accuracy_tie_aware": 0.45 + bump,
                        "runtime_regret": 3.0 + (50 - size) / 50 - bump,
                    }
                )

    def write_csv(path, members):
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(members[0]))
            writer.writeheader()
            writer.writerows(members)

    write_csv(directory / "per_workload.csv", rows)
    aggregates = []
    for budget in plan["budgets"]:
        size = budget["candidates_per_workload"]
        for model in ("random-42", "jev-1.14.0-long-context"):
            members = [
                row
                for row in rows
                if row["candidate_count"] == size and row["model_id"] == model
            ]
            aggregates.append(
                {
                    "candidate_count": size,
                    "model_id": model,
                    "model_type": members[0]["model_type"],
                    "decisions": len(members),
                    "unique_workloads": len(members),
                    **{
                        name: mean(row[name] for row in members)
                        for name in sweep.METRICS
                    },
                }
            )
    write_csv(directory / "by_candidate_count.csv", aggregates)
    (directory / "summary.json").write_text(
        json.dumps(
            {
                "prepared_sha256": plan["prepared_sha256"],
                "metrics": list(sweep.METRICS),
                "result_bundles": [entry["results_dir"] for entry in plan["budgets"]],
                "data_status": "synthetic_fixture",
            }
        )
    )
    return rows, write_csv


def test_size_curve_plot_uses_each_budget_not_pooled_overall(frozen, tmp_path):
    pytest.importorskip("matplotlib")
    prepared, _, kwargs = frozen
    manifest = sweep.prepare_tenset_size_sweep(prepared, **kwargs)
    analysis = tmp_path / "analysis"
    write_synthetic_analysis(manifest, analysis)
    (analysis / "overall.csv").write_text("invalid pooled values must never be read\n")
    output = sweep.plot_tenset_size_sweep(manifest, analysis, tmp_path / "plots")
    assert (output / "size-sweep-metrics.png").stat().st_size > 1000
    svg = (output / "size-sweep-metrics.svg").read_text()
    assert "Top-1 Score" in svg and "Runtime regret" in svg
    assert "Candidates per workload" in svg
    with (output / "size-sweep-plot-data.csv").open() as stream:
        data = list(csv.DictReader(stream))
    assert len(data) == 6
    assert {int(row["candidate_count"]) for row in data} == {16, 32, 50}
    assert {int(row["workloads"]) for row in data} == {3}
    method = json.loads((output / "size-sweep-plot-method.json").read_text())
    assert method["overall_pooled_scores_used"] is False
    assert method["data_status"] == "synthetic_fixture"


@pytest.mark.parametrize(
    "mutation", ["missing_budget", "different_workload", "duplicate"]
)
def test_size_curve_plot_rejects_unmatched_cohorts_before_plotting(
    frozen, tmp_path, mutation
):
    prepared, _, kwargs = frozen
    manifest = sweep.prepare_tenset_size_sweep(prepared, **kwargs)
    analysis = tmp_path / "analysis"
    rows, write_csv = write_synthetic_analysis(manifest, analysis)
    if mutation == "missing_budget":
        rows = [row for row in rows if row["candidate_count"] != 32]
    elif mutation == "different_workload":
        rows[0]["workload_id"] = "another-workload"
    else:
        rows.append(dict(rows[0]))
    write_csv(analysis / "per_workload.csv", rows)
    with pytest.raises(ValueError, match="missing a model|unpaired workload"):
        sweep.plot_tenset_size_sweep(manifest, analysis, tmp_path / "plots")
    assert not (tmp_path / "plots").exists()


def test_size_curve_plot_rejects_wrong_prepared_source(frozen, tmp_path):
    prepared, _, kwargs = frozen
    manifest = sweep.prepare_tenset_size_sweep(prepared, **kwargs)
    analysis = tmp_path / "analysis"
    write_synthetic_analysis(manifest, analysis)
    path = analysis / "summary.json"
    summary = json.loads(path.read_text())
    summary["prepared_sha256"] = "0" * 64
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="frozen data"):
        sweep.plot_tenset_size_sweep(manifest, analysis, tmp_path / "plots")


def test_size_curve_plot_rejects_changed_budget_means(frozen, tmp_path):
    prepared, _, kwargs = frozen
    manifest = sweep.prepare_tenset_size_sweep(prepared, **kwargs)
    analysis = tmp_path / "analysis"
    _, write_csv = write_synthetic_analysis(manifest, analysis)
    path = analysis / "by_candidate_count.csv"
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    rows[0]["top_1_score"] = 0.99
    write_csv(path, rows)
    with pytest.raises(ValueError, match="means disagree"):
        sweep.plot_tenset_size_sweep(manifest, analysis, tmp_path / "plots")
