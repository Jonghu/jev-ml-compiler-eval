"""Weighted network endpoint parity and private-label boundary checks."""

import copy
import csv
import json
import math
import os
import struct
from dataclasses import replace
from pathlib import Path

import pytest

from jev_compiler_eval.evaluation.tenset_paper_metrics import (
    REFERENCE_CSV,
    REFERENCE_MD,
    SUMMARY_CSV,
    SUMMARY_JSON,
    TASK_CSV,
    PaperTaskPool,
    compare_paper_network,
    export_paper_network_scores,
    score_paper_network,
)

TARGET = "llvm -keys=cpu -mcpu=skylake-avx512 -model=platinum-8272"
NETWORK = ["resnet_50", [[1, 3, 224, 224]]]


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def pools():
    return [
        PaperTaskPool("a", "key-a", TARGET, 9, 1, {"a0": 1, "a1": 0.5, "a2": 0.25}),
        PaperTaskPool("b", "key-b", TARGET, 1, 9, {"b0": 1, "b1": 0.125, "b2": 0.5}),
    ]


def rankings():
    return {"a": ["a2", "a1", "a0"], "b": ["b0", "b2", "b1"]}


def score(tasks=None, order=None, **kwargs):
    return score_paper_network(
        tasks if tasks is not None else pools(),
        order if order is not None else rankings(),
        network_key=NETWORK,
        target=TARGET,
        expected_task_weights={"a": 9, "b": 1},
        **kwargs,
    )


def test_weighted_network_ratio_is_not_macro_or_record_weighted_score():
    result = score()
    assert result["scope"] == "full_network"
    assert result["coverage"] == {
        "expected_distinct_tasks": 2,
        "evaluated_distinct_tasks": 2,
        "expected_total_task_weight": 10,
        "evaluated_total_task_weight": 10,
        "complete_network_coverage": True,
    }
    # Full-task oracle: 9*1 + 1*9 = 18. Selected: 9*4 + 1*9 = 45.
    assert result["best_weighted_latency_seconds"] == 18
    assert result["selected_weighted_latency_seconds_by_k"] == {"1": 45, "5": 18}
    assert result["metrics"] == {
        "paper_network_best_of_1_score": 0.4,
        "paper_network_best_of_5_score": 1,
    }
    assert result["metrics"]["paper_network_best_of_1_score"] != (0.25 + 1) / 2
    # Existing Top-5 averages prefixes; the official network evaluator uses
    # the best endpoint among K trials, including all candidates when n < K.
    assert result["metrics"]["paper_network_best_of_5_score"] != (0.25 + 0.5 + 1) / 3
    assert [row["evaluated_prefix_count"] for row in result["task_decompositions"]] == [
        1,
        3,
        1,
        3,
    ]


def test_task_frequency_changes_network_contribution_without_record_weighting():
    tasks = [replace(pools()[0], weight=1), replace(pools()[1], weight=9)]
    result = score_paper_network(
        tasks,
        rankings(),
        network_key=NETWORK,
        target=TARGET,
        expected_task_weights={"a": 1, "b": 9},
    )
    assert result["metrics"]["paper_network_best_of_1_score"] == pytest.approx(82 / 85)


def test_oracle_is_full_task_minimum_not_recomputed_from_selected_pool():
    task = PaperTaskPool("w", "key", TARGET, 1, 1, {"slow": 0.25, "better": 0.5})
    result = score_paper_network(
        [task],
        {"w": ["slow", "better"]},
        network_key=NETWORK,
        target=TARGET,
        expected_task_weights={"w": 1},
    )
    assert result["best_weighted_latency_seconds"] == 1
    assert result["metrics"]["paper_network_best_of_1_score"] == 0.25
    assert result["metrics"]["paper_network_best_of_5_score"] == 0.5


@pytest.mark.parametrize("label", [0, -1, -0.25, 1e-8])
def test_zero_negative_and_tiny_labels_use_exact_reference_floor(label):
    task = PaperTaskPool("w", "key", TARGET, 7, 0.25, {"bad": label, "good": 0.5})
    result = score_paper_network(
        [task],
        {"w": ["bad", "good"]},
        network_key=NETWORK,
        target=TARGET,
        expected_task_weights={"w": 7},
    )
    top1, top5 = result["task_decompositions"]
    assert top1["effective_normalized_throughput"] == 1e-5
    assert top1["best_of_k_latency_seconds"] == 0.25 / 1e-5
    assert result["metrics"]["paper_network_best_of_1_score"] == pytest.approx(1e-5)
    assert top5["best_candidate_in_prefix"] == "good"
    assert result["metrics"]["paper_network_best_of_5_score"] == 0.5


def test_float32_origin_is_recovered_with_float64_division():
    task = PaperTaskPool("w", "key", TARGET, 1, 0.03, {"choice": 0.2})
    result = score_paper_network(
        [task],
        {"w": ["choice"]},
        network_key=NETWORK,
        target=TARGET,
        expected_task_weights={"w": 1},
    )
    latency = result["task_decompositions"][0]["best_of_k_latency_seconds"]
    assert latency == f32(0.03) / f32(0.2)
    assert latency != f32(f32(0.03) / f32(0.2))
    assert result["metrics"]["paper_network_best_of_1_score"] == f32(0.2)


def test_complete_network_task_coverage_is_required_and_pilot_is_separately_named():
    with pytest.raises(ValueError, match="every expected task pool"):
        score(pools()[:1], {"a": rankings()["a"]})
    result = score(pools()[:1], {"a": rankings()["a"]}, pilot=True)
    assert result["scope"] == "pilot_subset"
    assert result["coverage"]["complete_network_coverage"] is False
    assert result["coverage"]["expected_distinct_tasks"] == 2
    assert result["coverage"]["evaluated_distinct_tasks"] == 1
    assert result["coverage"]["expected_total_task_weight"] == 10
    assert result["coverage"]["evaluated_total_task_weight"] == 9
    assert set(result["metrics"]) == {
        "pilot_network_best_of_1_score",
        "pilot_network_best_of_5_score",
    }


@pytest.mark.parametrize(
    "order",
    [
        {"a": ["a2", "a1", "a0"]},
        {**rankings(), "extra": ["x"]},
        {**rankings(), "a": ["a0", "a1"]},
        {**rankings(), "a": ["a0", "a1", "a1"]},
        {**rankings(), "a": ["a0", "a1", "unknown"]},
        {**rankings(), "a": "a0"},
    ],
)
def test_missing_extra_duplicate_or_partial_rankings_fail_closed(order):
    with pytest.raises(ValueError, match="rankings|ranking"):
        score(order=order)


@pytest.mark.parametrize(
    "change",
    [
        {"weight": 8},
        {"weight": True},
        {"target": "llvm -model=other"},
        {"workload_key": "key-b"},
        {"workload_id": "b"},
        {"full_task_min_latency_seconds": 0},
        {"full_task_min_latency_seconds": -1},
        {"full_task_min_latency_seconds": math.inf},
        {"full_task_min_latency_seconds": 1e100},
        {"labels_by_candidate": {}},
        {"labels_by_candidate": {"a0": math.nan}},
        {"labels_by_candidate": {"a0": math.inf}},
        {"labels_by_candidate": {"a0": 1.1}},
        {"labels_by_candidate": {"a0": True}},
    ],
)
def test_inconsistent_task_target_weight_or_numeric_input_is_rejected(change):
    with pytest.raises(ValueError):
        score([replace(pools()[0], **change), pools()[1]])


@pytest.mark.parametrize("top_ks", [[], [0], [True], [1, 1], [1.0], "15"])
def test_invalid_top_k_parameters_fail_before_scoring(top_ks):
    with pytest.raises(ValueError, match="top_ks"):
        score(top_ks=top_ks)


def test_model_comparison_keeps_identical_pools_and_local_random_reference():
    tasks = pools()
    before = copy.deepcopy(tasks)
    comparison = compare_paper_network(
        tasks,
        {
            "random-42": rankings(),
            "jev": {"a": ["a0", "a1", "a2"], "b": ["b0", "b1", "b2"]},
        },
        network_key=NETWORK,
        target=TARGET,
        expected_task_weights={"a": 9, "b": 1},
        model_types={"random-42": "random", "jev": "jev-tournament"},
    )
    assert tasks == before
    assert [model["model_id"] for model in comparison["models"]] == ["random-42", "jev"]
    top1, top5 = comparison["paired_random_comparisons"]
    assert top1["random_model_id"] == "random-42"
    assert top1["metric"] == "paper_network_best_of_1_score"
    assert top1["random_score"] == 0.4
    assert top1["model_score"] == 1
    assert top1["score_delta"] == 0.6
    assert top5["score_delta"] == 0
    assert "checkpoint" in comparison["reference_status"]
    assert all("probabilities" not in model for model in comparison["models"])


def test_metric_exports_are_separate_reconstructible_and_resume_without_overwrite(
    tmp_path,
):
    comparison = compare_paper_network(
        pools(),
        {"random-42": rankings()},
        network_key=NETWORK,
        target=TARGET,
        expected_task_weights={"a": 9, "b": 1},
        model_types={"random-42": "random"},
    )
    # Exports may live beside the common evaluator's canonical result bundle.
    (tmp_path / "predictions.jsonl").write_text("preserve canonical predictions\n")
    paths = export_paper_network_scores(tmp_path, comparison)
    assert set(paths) == {SUMMARY_JSON, SUMMARY_CSV, TASK_CSV}
    assert json.loads(paths[SUMMARY_JSON].read_text()) == comparison
    with paths[SUMMARY_CSV].open(newline="") as stream:
        scores = list(csv.DictReader(stream))
    assert len(scores) == 1
    assert float(scores[0]["paper_network_best_of_1_score"]) == 0.4
    with paths[TASK_CSV].open(newline="") as stream:
        tasks = list(csv.DictReader(stream))
    for k in (1, 5):
        members = [row for row in tasks if int(row["top_k"]) == k]
        oracle = sum(float(row["weighted_oracle_latency_seconds"]) for row in members)
        selected = sum(
            float(row["weighted_selected_latency_seconds"]) for row in members
        )
        assert oracle / selected == float(scores[0][f"paper_network_best_of_{k}_score"])
    original = {name: path.read_bytes() for name, path in paths.items()}
    assert export_paper_network_scores(tmp_path, comparison) == paths
    assert original == {name: path.read_bytes() for name, path in paths.items()}
    paths[TASK_CSV].unlink()
    export_paper_network_scores(tmp_path, comparison)
    assert original == {name: path.read_bytes() for name, path in paths.items()}
    assert (
        tmp_path / "predictions.jsonl"
    ).read_text() == "preserve canonical predictions\n"
    paths[SUMMARY_CSV].write_text("changed\n")
    with pytest.raises(FileExistsError, match="differs"):
        export_paper_network_scores(tmp_path, comparison)
    assert paths[SUMMARY_CSV].read_text() == "changed\n"


def test_export_prevalidates_all_files_before_publishing_any(tmp_path):
    comparison = compare_paper_network(
        pools(),
        {"random": rankings()},
        network_key=NETWORK,
        target=TARGET,
        expected_task_weights={"a": 9, "b": 1},
        model_types={"random": "random"},
    )
    (tmp_path / TASK_CSV).write_text("existing task report\n")
    with pytest.raises(FileExistsError):
        export_paper_network_scores(tmp_path, comparison)
    assert not (tmp_path / SUMMARY_JSON).exists()
    assert not (tmp_path / SUMMARY_CSV).exists()


def _export_comparison():
    return compare_paper_network(
        pools(),
        {"random": rankings()},
        network_key=NETWORK,
        target=TARGET,
        expected_task_weights={"a": 9, "b": 1},
        model_types={"random": "random"},
    )


@pytest.mark.parametrize("late_target", ["different_file", "symlink", "directory"])
def test_export_never_clobbers_a_late_target(tmp_path, monkeypatch, late_target):
    original_link = os.link

    def concurrent_link(source, destination):
        destination = Path(destination)
        assert destination.name == SUMMARY_JSON
        if late_target == "different_file":
            destination.write_bytes(b"concurrent result; preserve me\n")
        elif late_target == "symlink":
            other = tmp_path / "other.json"
            other.write_bytes(Path(source).read_bytes())
            destination.symlink_to(other)
        else:
            destination.mkdir()
        return original_link(source, destination)

    monkeypatch.setattr(os, "link", concurrent_link)
    with pytest.raises(FileExistsError, match="differs"):
        export_paper_network_scores(tmp_path, _export_comparison())
    destination = tmp_path / SUMMARY_JSON
    if late_target == "different_file":
        assert destination.read_bytes() == b"concurrent result; preserve me\n"
    elif late_target == "symlink":
        assert destination.is_symlink()
    else:
        assert destination.is_dir()
    assert not (tmp_path / SUMMARY_CSV).exists()
    assert not (tmp_path / TASK_CSV).exists()
    assert not list(tmp_path.glob("tmp*"))


def test_export_accepts_a_late_identical_file_without_replacing_it(
    tmp_path, monkeypatch
):
    original_link = os.link
    concurrent_inode = None

    def concurrent_link(source, destination):
        nonlocal concurrent_inode
        destination = Path(destination)
        if destination.name == SUMMARY_JSON:
            destination.write_bytes(Path(source).read_bytes())
            concurrent_inode = destination.stat().st_ino
        return original_link(source, destination)

    monkeypatch.setattr(os, "link", concurrent_link)
    paths = export_paper_network_scores(tmp_path, _export_comparison())
    assert paths[SUMMARY_JSON].stat().st_ino == concurrent_inode
    assert json.loads(paths[SUMMARY_JSON].read_text()) == _export_comparison()
    assert all(path.is_file() for path in paths.values())
    assert not list(tmp_path.glob("tmp*"))


def test_export_revalidates_files_created_after_initial_checks(tmp_path, monkeypatch):
    original_mkdir = Path.mkdir

    def concurrent_mkdir(path, *args, **kwargs):
        result = original_mkdir(path, *args, **kwargs)
        if path == tmp_path:
            (path / SUMMARY_JSON).write_text("late result; preserve me\n")
        return result

    monkeypatch.setattr(Path, "mkdir", concurrent_mkdir)
    with pytest.raises(FileExistsError, match="differs"):
        export_paper_network_scores(tmp_path, _export_comparison())
    assert (tmp_path / SUMMARY_JSON).read_text() == "late result; preserve me\n"
    assert not (tmp_path / SUMMARY_CSV).exists()


def _with_published_reference(*, pilot=False):
    tasks = pools()[:1] if pilot else pools()
    local_rankings = {
        "random-42": {task.workload_id: rankings()[task.workload_id] for task in tasks},
        "jev": {task.workload_id: list(task.labels_by_candidate) for task in tasks},
    }
    result = compare_paper_network(
        tasks,
        local_rankings,
        network_key=NETWORK,
        target=TARGET,
        expected_task_weights={"a": 9, "b": 1},
        model_types={"random-42": "random", "jev": "jev-tournament"},
        pilot=pilot,
    )
    result["external_reference"] = {
        "model": "MLP + Ranking loss",
        "top_1": 0.8823,
        "top_5": 0.9456,
        "source_url": "https://example.org/tenset-paper.pdf#page=7",
        "original_test_inventory_verified": False,
        "original_checkpoint_verified": False,
    }
    return result


def test_full_reference_exports_have_three_ordered_rows_with_evidence_classes(tmp_path):
    comparison = _with_published_reference()
    paths = export_paper_network_scores(tmp_path, comparison)
    assert set(paths) == {
        SUMMARY_JSON,
        SUMMARY_CSV,
        TASK_CSV,
        REFERENCE_CSV,
        REFERENCE_MD,
    }
    with paths[REFERENCE_CSV].open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["Model"] for row in rows] == ["Jev", "Random", "MLP + Ranking loss"]
    assert [row["Evidenceclass"] for row in rows] == [
        "local paired measurement",
        "local paired measurement",
        "published reference (unpaired)",
    ]
    assert float(rows[0]["paper_network_best_of_1_score"]) == 1
    assert float(rows[1]["paper_network_best_of_1_score"]) == 0.4
    assert float(rows[2]["paper_network_best_of_1_score"]) == 0.8823
    assert float(rows[2]["paper_network_best_of_5_score"]) == 0.9456
    assert [row["same_as_local_candidate_pool_verified"] for row in rows] == [
        "True",
        "True",
        "False",
    ]
    assert rows[0]["source_URL"] == rows[1]["source_URL"] == "predictions.jsonl"
    assert rows[2]["source_URL"] == comparison["external_reference"]["source_url"]
    markdown = paths[REFERENCE_MD].read_text()
    assert (
        "| MLP + Ranking loss | published reference (unpaired) | 0.8823 | 0.9456 | No |"
        in markdown
    )
    assert "inventory and trained checkpoint remain unverified" in markdown
    assert "Only the local Jev-versus-Random comparison is paired" in markdown
    saved = json.loads(paths[SUMMARY_JSON].read_text())
    assert [model["model_id"] for model in saved["models"]] == ["random-42", "jev"]
    assert len(saved["paired_random_comparisons"]) == 2
    assert all(row["model_id"] == "jev" for row in saved["paired_random_comparisons"])
    with paths[SUMMARY_CSV].open(newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 2
    original = {name: path.read_bytes() for name, path in paths.items()}
    assert export_paper_network_scores(tmp_path, comparison) == paths
    assert original == {name: path.read_bytes() for name, path in paths.items()}


def test_pilot_exports_omit_published_reference_table(tmp_path):
    comparison = _with_published_reference(pilot=True)
    comparison["external_reference"] = None
    paths = export_paper_network_scores(tmp_path, comparison)
    assert set(paths) == {SUMMARY_JSON, SUMMARY_CSV, TASK_CSV}
    assert not (tmp_path / REFERENCE_CSV).exists()
    assert not (tmp_path / REFERENCE_MD).exists()


def test_reference_injected_into_pilot_is_rejected_before_export(tmp_path):
    comparison = _with_published_reference(pilot=True)
    with pytest.raises(ValueError, match="complete full network"):
        export_paper_network_scores(tmp_path, comparison)
    assert not list(tmp_path.iterdir())


def test_reference_with_incomplete_coverage_is_rejected(tmp_path):
    comparison = _with_published_reference()
    comparison["coverage"]["complete_network_coverage"] = False
    with pytest.raises(ValueError, match="complete full network"):
        export_paper_network_scores(tmp_path, comparison)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("top_1", math.nan),
        ("top_5", math.inf),
        ("top_1", -0.1),
        ("top_5", 1.1),
        ("top_1", True),
        ("top_5", "0.9456"),
        ("model", ""),
        ("model", "MLP\nInjected row"),
        ("source_url", "javascript:alert(1)"),
        ("source_url", "https://name:secret@example.org/paper.pdf"),
        ("source_url", "https://example.org/paper\n.pdf"),
    ],
)
def test_invalid_published_reference_is_rejected_without_output(tmp_path, field, value):
    comparison = _with_published_reference()
    comparison["external_reference"][field] = value
    with pytest.raises(ValueError):
        export_paper_network_scores(tmp_path, comparison)
    assert not list(tmp_path.iterdir())


def test_reference_table_conflict_prevents_any_new_exports(tmp_path):
    (tmp_path / REFERENCE_MD).write_text("preserve this report\n")
    with pytest.raises(FileExistsError, match="differs"):
        export_paper_network_scores(tmp_path, _with_published_reference())
    assert (tmp_path / REFERENCE_MD).read_text() == "preserve this report\n"
    assert not (tmp_path / SUMMARY_JSON).exists()
    assert not (tmp_path / REFERENCE_CSV).exists()


def test_late_reference_table_conflict_is_not_overwritten(tmp_path, monkeypatch):
    original_link = os.link

    def concurrent_link(source, destination):
        if Path(destination).name == REFERENCE_CSV:
            Path(destination).write_text("late reference table; preserve me\n")
        return original_link(source, destination)

    monkeypatch.setattr(os, "link", concurrent_link)
    with pytest.raises(FileExistsError, match="differs"):
        export_paper_network_scores(tmp_path, _with_published_reference())
    assert (
        tmp_path / REFERENCE_CSV
    ).read_text() == "late reference table; preserve me\n"
    assert not (tmp_path / REFERENCE_MD).exists()
    assert not list(tmp_path.glob("tmp*"))
