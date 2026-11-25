import importlib.util
from pathlib import Path

import pytest

from jev_compiler_eval.types import Candidate, EvaluationCase, RuntimeRecord, Workload

SCRIPT = Path(__file__).parents[1] / "scripts" / "summarize_tpugraphs_ablation.py"
_spec = importlib.util.spec_from_file_location("summarize_ablation", SCRIPT)
assert _spec is not None and _spec.loader is not None
summary = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(summary)


def test_splits_repeat_suffix_from_model_id() -> None:
    assert summary.split_model_id("a3-both-fields-rep2") == ("a3-both-fields", 2)
    assert summary.split_model_id("random") == ("random", None)


def test_tile_size_multiplies_output_and_kernel_bounds() -> None:
    configuration = {
        "output_bounds": [1, 2, 3],
        "kernel_bounds": [4],
        "input_bounds": [9],
    }
    assert summary.tile_size(configuration) == 24


def test_pick_agreement_counts_kernels_with_one_pick() -> None:
    runs = [
        [
            {"workload_id": "w1", "selected_candidate_id": "a"},
            {"workload_id": "w2", "selected_candidate_id": "a"},
        ],
        [
            {"workload_id": "w1", "selected_candidate_id": "a"},
            {"workload_id": "w2", "selected_candidate_id": "b"},
        ],
    ]
    assert summary.pick_agreement(runs) == 0.5


def test_heuristic_baselines_use_hidden_runtimes() -> None:
    sizes = {"config_0": [2], "config_1": [1], "config_2": [4]}
    runtimes = {"config_0": 2.0, "config_1": 4.0, "config_2": 1.0}
    case = EvaluationCase(
        Workload("w", {}),
        tuple(Candidate(cid, {"output_bounds": b}) for cid, b in sizes.items()),
        tuple(RuntimeRecord("w", cid, value) for cid, value in runtimes.items()),
    )

    rows = {row["baseline"]: row for row in summary.heuristic_baselines([case], 2)}

    assert rows["compiler default (config_0)"]["top_1_error"] == pytest.approx(1.0)
    assert rows["largest tile"]["top_1_error"] == pytest.approx(0.0)
    # Largest two are config_2 (1.0) and config_0 (2.0).
    expected = rows["largest-2 filter + random pick (expected)"]["top_1_error"]
    assert expected == pytest.approx(0.5)
    assert rows["largest-2 filter + oracle pick"]["top_1_error"] == pytest.approx(0.0)
    assert rows["random pick (expected)"]["top_1_error"] == pytest.approx(4 / 3)


def test_heuristic_filter_baselines_use_the_shown_pool() -> None:
    sizes = {"config_0": [2], "config_1": [1], "config_2": [4]}
    runtimes = {"config_0": 2.0, "config_1": 1.0, "config_2": 4.0}
    case = EvaluationCase(
        Workload("w", {}),
        tuple(Candidate(cid, {"output_bounds": b}) for cid, b in sizes.items()),
        tuple(RuntimeRecord("w", cid, value) for cid, value in runtimes.items()),
    )

    rows = {
        row["baseline"]: row
        for row in summary.heuristic_baselines(
            [case], 2, {"w": ["config_1", "config_2"]}
        )
    }

    oracle = rows["largest-2 filter + oracle pick"]
    assert oracle["top_1_error"] == pytest.approx(0.0)
    assert oracle["optimal_share"] == pytest.approx(1.0)
    expected = rows["largest-2 filter + random pick (expected)"]
    assert expected["top_1_error"] == pytest.approx(1.5)
    assert expected["optimal_share"] == pytest.approx(0.5)


def test_paired_comparison_counts_and_sign_test() -> None:
    left = {"w1": 1.0, "w2": 1.0, "w3": 1.0, "w4": 0.5}
    right = {"w1": 0.5, "w2": 0.5, "w3": 2.0, "w4": 0.5}

    result = summary.paired_comparison(left, right)

    assert (result["right_better"], result["right_worse"], result["ties"]) == (2, 1, 1)
    assert result["mean_difference"] == pytest.approx(0.0)
    assert result["sign_test_p"] == pytest.approx(1.0)
    assert summary.sign_test(10, 0) == pytest.approx(2 / 1024)
