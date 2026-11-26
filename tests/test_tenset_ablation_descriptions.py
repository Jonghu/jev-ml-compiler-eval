"""Description factors preserve identical program information and hide labels."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from jev_compiler_eval.datasets.tenset_ablation_descriptions import (
    ablation_context,
    tenset_candidate_fields,
    tenset_graph_fields,
)
from jev_compiler_eval.datasets.tenset_descriptions import tenset_configuration_fields
from jev_compiler_eval.datasets.tenset_metadata import metadata_from_tensor_graph
from jev_compiler_eval.datasets.tenset_registry import load_tenset_registry


@pytest.fixture
def context():
    return {
        "operation": "sum",
        "tensors": [{"name": "sum", "shape": [8], "dtype": "float32"}],
        "compute_dag": {
            "format": "tvm-te-expression-graph-v1",
            "nodes": [
                {"type": "ComputeOp", "attrs": {"name": "sum"}},
                {"type": "Array", "items": [0]},
            ],
            "root": 1,
            "stage_order": [0],
        },
        "loop_structure": [],
        "configuration_fields": {"old_guide": "must not reach naive"},
        "graph_fields": {"old_guide": "must not reach naive"},
    }


def test_four_conditions_preserve_exact_raw_program_and_are_independent(context):
    original = deepcopy(context)
    raw = {k: v for k, v in context.items() if not k.endswith("_fields")}
    conditions = {}
    for candidate in (False, True):
        for graph in (False, True):
            result = ablation_context(
                context,
                candidate_definitions=candidate,
                graph_specification=graph,
            )
            conditions[candidate, graph] = result
            assert {k: result[k] for k in raw} == raw
            assert ("configuration_fields" in result) == candidate
            assert ("graph_fields" in result) == graph
    assert context == original
    assert conditions[False, False] == raw
    assert (
        conditions[True, False]["configuration_fields"]
        == conditions[True, True]["configuration_fields"]
    )
    assert (
        conditions[False, True]["graph_fields"]
        == conditions[True, True]["graph_fields"]
    )


def test_candidate_factor_omits_graph_mapping_without_altering_original_guide():
    original = tenset_configuration_fields()
    candidate = tenset_candidate_fields()
    assert "graph_ids" not in candidate["indices"]
    assert "graph_ids" in original["indices"]
    assert candidate["steps"] == original["steps"]
    assert candidate["annotation_codes"] == original["annotation_codes"]
    assert candidate["initial_state"] == original["initial_state"]


def test_graph_guide_covers_exported_node_fields_and_preserves_native_graphs():
    guide = tenset_graph_fields()
    dag = guide["compute_dag"]
    registry = load_tenset_registry(
        Path(__file__).parent / "fixtures/tenset_registry/native_registry.pkl"
    )
    for record in registry:
        metadata = metadata_from_tensor_graph(record["tensor_graph"])
        for node in metadata["compute_dag"]["nodes"]:
            assert set(node) <= set(dag)
        for tensor in metadata["tensors"]:
            assert set(tensor) <= set(guide["tensors"])
        for loop in metadata["loop_structure"]:
            assert set(loop) <= set(guide["loop_structure"])
        enriched = ablation_context(
            metadata, candidate_definitions=False, graph_specification=True
        )
        assert {key: enriched[key] for key in metadata} == metadata
    assert "steps" not in guide
    assert "annotation_codes" not in guide


def test_guides_and_workload_are_deep_copied(context):
    result = ablation_context(
        context, candidate_definitions=True, graph_specification=True
    )
    result["compute_dag"]["stage_order"].append(99)
    result["configuration_fields"]["steps"]["SP"]["arguments"].reverse()
    result["graph_fields"]["compute_dag"]["nodes"] = "changed"
    assert context["compute_dag"]["stage_order"] == [0]
    again = ablation_context(
        context, candidate_definitions=True, graph_specification=True
    )
    assert again["configuration_fields"] == tenset_candidate_fields()
    assert again["graph_fields"] == tenset_graph_fields()
    json.dumps(again, allow_nan=False)


@pytest.mark.parametrize(
    "private",
    [{"runtime_ms": 1}, {"nested": [{"record_input": []}]}],
)
def test_model_visible_labels_are_rejected(context, private):
    context["compute_dag"]["private"] = private
    with pytest.raises(ValueError, match="label-bearing"):
        ablation_context(context, candidate_definitions=True, graph_specification=True)


@pytest.mark.parametrize("candidate,graph", [(1, True), (False, None)])
def test_description_flags_do_not_accept_truthy_non_booleans(context, candidate, graph):
    with pytest.raises(ValueError, match="booleans"):
        ablation_context(
            context,
            candidate_definitions=candidate,
            graph_specification=graph,
        )
