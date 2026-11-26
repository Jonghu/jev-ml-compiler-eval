"""Structural filtering must not consult hidden measurements or invent tile sizes."""

from dataclasses import replace
from math import log

import pytest

from jev_compiler_eval.evaluation.tenset_ablation_features import (
    FEATURE_NAMES,
    feature_signature,
    schedule_features,
    select_diverse_candidates,
)
from jev_compiler_eval.types import Candidate


def candidate(name, *steps, **extra):
    return Candidate(
        name,
        {
            "initial_state": [],
            "transformations": [
                {"code": step[0], "arguments": list(step[1:])} for step in steps
            ],
            **extra,
        },
    )


def test_features_follow_record_positions_and_do_not_reconstruct_tile_volume():
    item = candidate(
        "schedule",
        ("SP", 2, 3, 128, [4, 8, None, -1, 0], 1),
        ("AN", 2, 0, 2),
        ("PR", 2, 0, "auto_unroll_max_step$64"),
        ("PR", 2, 0, "debug_skip_region"),
        ("RE", 2, [1, 0]),
    )
    features = schedule_features(item)
    assert tuple(features) == FEATURE_NAMES
    assert features["step_count"] == 5
    assert features["step_SP"] == features["step_AN"] == features["step_RE"] == 1
    assert features["step_PR"] == 2
    assert features["split_explicit_count"] == 2
    assert features["split_inferred_count"] == 3
    assert features["split_log_sum"] == pytest.approx(log(5) + log(9))
    assert features["split_log_max"] == pytest.approx(log(9))
    assert features["annotation_2"] == 1
    assert features["unroll_pragma_count"] == 1
    assert features["unroll_log_max"] == pytest.approx(log(65))


def test_unknown_steps_and_annotations_have_explicit_buckets():
    features = schedule_features(candidate("c", ("FUTURE", 9), ("AN", 0, 0, 19)))
    assert features["step_unknown"] == features["annotation_other"] == 1


@pytest.mark.parametrize("length", [True, False, 1.5, "4", [4], {"factor": 4}])
def test_noninteger_split_factors_are_rejected(length):
    with pytest.raises(ValueError, match="SP lengths"):
        schedule_features(candidate("c", ("SP", 0, 0, 8, [length], 1)))


@pytest.mark.parametrize(
    "step",
    [("SP", 0, 0, 8), ("SP", 0, 0, 8, None, 1), ("AN", 0, 0, True), ("PR", 0, 0, 2)],
)
def test_malformed_special_steps_fail_before_filtering(step):
    with pytest.raises(ValueError):
        schedule_features(candidate("c", step))


def test_features_ignore_ids_graph_state_extra_fields_and_hidden_measurements():
    item = candidate("one", ("SP", 0, 0, 64, [4, 8], 1))
    changed = replace(
        item,
        candidate_id="another",
        configuration={
            **item.configuration,
            "initial_state": [{"different": "program"}],
            "runtime_ms": 0.0001,
            "label": 42,
        },
    )
    assert feature_signature(item) == feature_signature(changed)


def test_farthest_first_retains_distinct_schedule_structures_before_duplicates():
    pool = [candidate(f"empty-{i}") for i in range(10)] + [
        candidate("reorder", ("RE", 0, [1, 0])),
        candidate("vectorized", ("AN", 0, 0, 2)),
    ]
    kept = select_diverse_candidates(pool, 3, 42, "workload")
    assert len({feature_signature(item) for item in kept}) == 3
    assert "reorder" in {item.candidate_id for item in kept}
    assert "vectorized" in {item.candidate_id for item in kept}


def test_selection_is_reproducible_order_independent_and_preserves_presentation_order():
    pool = [candidate(str(i), ("SP", 0, 0, 64, [i + 1], 1)) for i in range(30)]
    first = select_diverse_candidates(pool, 8, 42, "workload")
    assert first == select_diverse_candidates(pool, 8, 42, "workload")
    reversed_result = select_diverse_candidates(list(reversed(pool)), 8, 42, "workload")
    assert first == list(reversed(reversed_result))
    assert first == [item for item in pool if item in first]


def test_constant_features_and_singleton_pools_are_handled_without_scaling_errors():
    pool = [candidate(str(i)) for i in range(15)]
    assert len(select_diverse_candidates(pool, 10, 42, "workload")) == 10
    assert select_diverse_candidates(pool[:1], 10, 42, "workload") == pool[:1]
    assert select_diverse_candidates([], 10, 42, "workload") == []
    assert select_diverse_candidates(pool, 30, 42, "workload") == pool


def test_duplicate_ids_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        select_diverse_candidates([candidate("same"), candidate("same")], 1, 42, "w")


@pytest.mark.parametrize(
    "keep,seed,workload", [(True, 42, "w"), (0, 42, "w"), (1, True, "w"), (1, 42, "")]
)
def test_invalid_controls_are_rejected(keep, seed, workload):
    with pytest.raises(ValueError):
        select_diverse_candidates([candidate("one")], keep, seed, workload)
