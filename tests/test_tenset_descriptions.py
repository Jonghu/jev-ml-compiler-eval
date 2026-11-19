"""Format goldens follow the pinned TenSet C++ record writers, not timings."""

import json

import pytest

from jev_compiler_eval.datasets.tenset import TRANSFORMATIONS, _reject_label_keys
from jev_compiler_eval.datasets.tenset_descriptions import (
    tenset_configuration_fields,
)

# Positional fixtures transcribed from each StepNode::WriteToRecord in
# tlc-pack/tenset@35774ed373f9693cc6efcbe83f5703141e8ab0c8. Distinct values make
# accidental argument reordering observable without requiring native TVM.
STEP_RECORDS = [
    (["AN", 2, 3, 6], {"stage_id": 2, "iter_id": 3, "annotation": 6}),
    (["FU", 2, [3, 4]], {"stage_id": 2, "fused_ids": [3, 4]}),
    (
        ["PR", 2, 3, "auto_unroll_max_step$64"],
        {"stage_id": 2, "iter_id": 3, "pragma_type": "auto_unroll_max_step$64"},
    ),
    (["RE", 2, [1, 0, 2]], {"stage_id": 2, "after_ids": [1, 0, 2]}),
    (
        ["SP", 2, 3, 128, [4, 8], 1],
        {
            "stage_id": 2,
            "iter_id": 3,
            "extent": 128,
            "lengths": [4, 8],
            "inner_to_outer": 1,
        },
    ),
    (
        ["FSP", 2, 3, 7, 2],
        {"stage_id": 2, "iter_id": 3, "src_step_id": 7, "n_split": 2},
    ),
    (
        ["FFSP", 2, 3, [7, 9], 1, 0],
        {
            "stage_id": 2,
            "iter_id": 3,
            "src_step_ids": [7, 9],
            "level": 1,
            "factor_or_nparts": 0,
        },
    ),
    (
        ["SA", 2, 3, 32, 1],
        {"stage_id": 2, "iter_id": 3, "factor": 32, "offset": 1},
    ),
    (
        ["CA", 2, 5, 3],
        {"stage_id": 2, "target_stage_id": 5, "target_iter_id": 3},
    ),
    (["CI", 2], {"stage_id": 2}),
    (["CR", 2], {"stage_id": 2}),
    (
        ["CHR", 2, "shared", [5, 7]],
        {"stage_id": 2, "scope_name": "shared", "reader_stage_ids": [5, 7]},
    ),
    (["CHW", 2, "local"], {"stage_id": 2, "scope_name": "local"}),
    (
        ["RF", 2, 3, 1],
        {"stage_id": 2, "iter_id": 3, "factor_iter_id": 1},
    ),
]


@pytest.mark.parametrize("record, expected", STEP_RECORDS, ids=lambda value: str(value))
def test_argument_positions_match_pinned_record_fixtures(record, expected):
    guide = tenset_configuration_fields()
    step = guide["steps"][record[0]]
    assert dict(zip(step["arguments"], record[1:], strict=True)) == expected
    assert step["name"] == TRANSFORMATIONS[record[0]]


def test_annotation_enum_preserves_interleaved_block_and_thread_dimensions():
    codes = tenset_configuration_fields()["annotation_codes"]
    assert {key: codes[str(key)] for key in range(1, 11)} == {
        1: "unroll",
        2: "vectorize",
        3: "parallel",
        4: "vthread",
        5: "blockIdx.x",
        6: "threadIdx.x",
        7: "blockIdx.y",
        8: "threadIdx.y",
        9: "blockIdx.z",
        10: "threadIdx.z",
    }
    assert "none" in codes["0"]
    assert "tensorize" in codes["11"] and "unsupported" in codes["11"]


def test_descriptions_distinguish_history_graph_and_mutating_stage_indices():
    guide = tenset_configuration_fields()
    assert set(guide["steps"]) == set(TRANSFORMATIONS)
    assert "complete transformation history" in guide["indices"]["src_step_id"]
    assert "graph IDs are not stage indices" in guide["indices"]["graph_ids"]
    assert "spatial axes followed by reduction axes" in guide["indices"]["iter_id"]
    assert "entry remains" in guide["steps"]["CI"]["meaning"]
    assert "stage_id+1" in guide["steps"]["CHR"]["meaning"]
    assert "two added stages" in guide["steps"]["CHW"]["meaning"]
    assert "not an existing iterator reference" in guide["steps"]["RF"]["meaning"]
    assert "n_split+1 iterators" in guide["steps"]["FSP"]["meaning"]
    assert (
        "flag 1 selects factor, flag 0 selects nparts"
        in (guide["steps"]["FFSP"]["meaning"])
    )


def test_empty_initial_state_is_omitted_stages_and_extent_zero_is_unspecified():
    guide = tenset_configuration_fields()
    assert "omits stages" in guide["initial_state"]
    assert "0 when unspecified" in guide["steps"]["SP"]["meaning"]
    assert "reverse order with factor splits" in guide["steps"]["SP"]["meaning"]


def test_each_call_is_fresh_deterministic_json_and_passes_label_policy():
    original = tenset_configuration_fields()
    expected = json.dumps(original, sort_keys=True, separators=(",", ":"))
    original["steps"]["SP"]["arguments"].reverse()
    original["annotation_codes"]["6"] = "changed"
    again = tenset_configuration_fields()
    assert json.dumps(again, sort_keys=True, separators=(",", ":")) == expected
    assert json.loads(expected) == again
    _reject_label_keys(again)
    assert again["source"].endswith("35774ed373f9693cc6efcbe83f5703141e8ab0c8")
    assert "src/auto_scheduler/transform_step.cc" in again["source_files"]


def test_provider_validates_nested_fields_before_returning(monkeypatch):
    import jev_compiler_eval.datasets.tenset_descriptions as module

    bad = tenset_configuration_fields()
    bad["steps"]["SP"]["nested"] = {"runtime_ms": 1}
    monkeypatch.setattr(module, "_FIELDS_JSON", json.dumps(bad))
    with pytest.raises(ValueError, match="label-bearing"):
        module.tenset_configuration_fields()
