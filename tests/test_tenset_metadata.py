"""Synthetic SaveJSON schemas exercise portable extraction without TVM."""

import base64
import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

from jev_compiler_eval.datasets.tenset_metadata import (
    GRAPH_FORMAT,
    UnsupportedTensorGraph,
    metadata_from_tensor_graph,
)

NATIVE_FIXTURES = Path(__file__).parent / "fixtures" / "tenset_registry"
NATIVE_REFERENCE = json.loads(
    (NATIVE_FIXTURES / "native_reference.json").read_text(encoding="utf-8")
)


class TensorGraphFixture:
    """Build data-only test fixtures following the pinned TVM VisitAttrs schema."""

    def __init__(self):
        self.nodes = [{"type_key": ""}]

    def node(self, kind, **attrs):
        self.nodes.append(
            {
                "type_key": kind,
                "attrs": {key: str(value) for key, value in attrs.items()},
            }
        )
        return len(self.nodes) - 1

    def array(self, values):
        self.nodes.append({"type_key": "Array", "data": list(values)})
        return len(self.nodes) - 1

    def string(self, value):
        node = {"type_key": "runtime.String"}
        if value:
            if value.isascii() and value.isprintable():
                node["repr_str"] = value
            else:
                data = value.encode("utf-8")
                node["repr_b64"] = base64.b64encode(
                    len(data).to_bytes(8, "little") + data
                ).decode("ascii")
        self.nodes.append(node)
        return len(self.nodes) - 1

    def integer(self, value):
        return self.node("IntImm", dtype="int32", value=value, span=0)

    def variable(self, name):
        return self.node(
            "tir.Var", dtype="int32", name=self.string(name), type_annotation=0, span=0
        )

    def axis(self, name, extent, *, minimum=0):
        domain = self.node("Range", min=self.integer(minimum), extent=extent, span=0)
        variable = self.variable(name)
        return self.node(
            "tir.IterVar",
            dom=domain,
            var=variable,
            iter_type=0,
            thread_tag=self.string(""),
            span=0,
        )

    def placeholder(self, name, dimensions):
        shape = self.array([self.integer(value) for value in dimensions])
        op = self.node(
            "PlaceholderOp",
            name=name,
            tag="",
            attrs=self.map(),
            shape=shape,
            dtype="float32",
        )
        return self.node("Tensor", shape=shape, op=op, dtype="float32", value_index=0)

    def map(self, entries=None):
        entries = entries or {}
        self.nodes.append(
            {"type_key": "Map", "keys": list(entries), "data": list(entries.values())}
        )
        return len(self.nodes) - 1

    def load(self, tensor, indices):
        return self.node(
            "tir.ProducerLoad",
            dtype="float32",
            producer=tensor,
            indices=self.array(indices),
            span=0,
        )

    def compute(self, name, dimensions, body, *, reductions=(), attrs=None):
        shape = self.array(dimensions)
        axes = self.array(
            [
                self.axis(f"{name}.ax{index}", extent)
                for index, extent in enumerate(dimensions)
            ]
        )
        op = self.node(
            "ComputeOp",
            name=name,
            tag="",
            attrs=attrs or self.map(),
            axis=axes,
            reduce_axis=self.array(reductions),
            body=self.array(body),
        )
        outputs = [
            self.node("Tensor", shape=shape, op=op, dtype="float32", value_index=index)
            for index in range(len(body))
        ]
        return outputs[0] if len(outputs) == 1 else outputs

    def graph(self, tensors):
        return {
            "root": self.array(tensors),
            "nodes": self.nodes,
            "b64ndarrays": [],
            "attrs": {"tvm_version": "0.8.dev0"},
        }


def elementwise_graph():
    fixture = TensorGraphFixture()
    source = fixture.placeholder("A", [4, 8])
    i, j = fixture.variable("i"), fixture.variable("j")
    load = fixture.load(source, [i, j])
    value = fixture.node("FloatImm", dtype="float32", value="0.125", span=0)
    add = fixture.node("tir.Add", dtype="float32", a=load, b=value, span=0)
    output = fixture.compute(
        "bias_add", [fixture.integer(4), fixture.integer(8)], [add]
    )
    return fixture, fixture.graph([source, output])


def test_extracts_real_names_shapes_dtypes_axes_and_complete_expression_graph():
    _, graph = elementwise_graph()
    metadata = metadata_from_tensor_graph(graph)
    assert metadata["operation"] == "bias_add"
    assert metadata["tensors"] == [
        {"name": "A", "shape": [4, 8], "dtype": "float32"},
        {"name": "bias_add", "shape": [4, 8], "dtype": "float32"},
    ]
    loop = metadata["loop_structure"][0]
    assert loop["stage_id"] == 1
    assert [axis["extent"] for axis in loop["spatial"]] == [4, 8]
    assert [axis["min"] for axis in loop["spatial"]] == [0, 0]
    assert [axis["variable"] for axis in loop["spatial"]] == [
        "bias_add.ax0",
        "bias_add.ax1",
    ]
    dag = metadata["compute_dag"]
    assert dag["format"] == GRAPH_FORMAT
    assert dag["nodes"][dag["stage_order"][1]]["attrs"]["name"] == "bias_add"
    addition = next(node for node in dag["nodes"] if node["type"] == "tir.Add")
    assert addition["attrs"] == {"dtype": "float32"}
    assert set(addition["refs"]) == {"a", "b"}
    assert any(node["type"] == "tir.ProducerLoad" for node in dag["nodes"])
    assert metadata_from_tensor_graph(json.dumps(graph)) == metadata


def test_numeric_literal_attrs_are_never_mistaken_for_node_references():
    _, graph = elementwise_graph()
    metadata = metadata_from_tensor_graph(graph)
    immediates = [
        node for node in metadata["compute_dag"]["nodes"] if node["type"] == "IntImm"
    ]
    assert any(node["attrs"]["value"] == "4" for node in immediates)
    assert all("refs" not in node for node in immediates)


@pytest.mark.parametrize("value", ["1.23456789012345e-20", "inf", "-inf", "nan"])
def test_preserves_source_floating_literal_strings_without_rounding(value):
    _, graph = elementwise_graph()
    original = next(node for node in graph["nodes"] if node["type_key"] == "FloatImm")
    original["attrs"]["value"] = value
    metadata = metadata_from_tensor_graph(graph)
    compact = next(
        node for node in metadata["compute_dag"]["nodes"] if node["type"] == "FloatImm"
    )
    assert compact["attrs"]["value"] == value
    json.dumps(metadata, allow_nan=False)


def test_prunes_unreachable_task_nodes_and_source_spans_only():
    fixture, graph = elementwise_graph()
    source = fixture.node("SourceName", name="unrelated debug source")
    graph["nodes"][1]["attrs"]["span"] = str(source)
    fixture.node("auto_scheduler.SearchTask", measured_runtime="123456")
    metadata = metadata_from_tensor_graph(graph)
    text = json.dumps(metadata)
    assert "SearchTask" not in text
    assert "123456" not in text
    assert "unrelated debug source" not in text
    assert "span" not in text


def test_local_ids_are_stable_when_source_node_indices_change():
    _, graph = elementwise_graph()
    before = metadata_from_tensor_graph(graph)
    # Reindex a fully known data-only fixture. Literal attr values stay unchanged.
    from jev_compiler_eval.datasets.tenset_metadata import _SCHEMAS

    order = [0] + list(reversed(range(1, len(graph["nodes"]))))
    mapping = {old: new for new, old in enumerate(order)}
    changed = copy.deepcopy(graph)
    changed["nodes"] = [copy.deepcopy(graph["nodes"][old]) for old in order]
    changed["root"] = mapping[graph["root"]]
    for node in changed["nodes"]:
        if "data" in node:
            node["data"] = [mapping[value] for value in node["data"]]
        refs = _SCHEMAS.get(node["type_key"], (set(), set()))[0] | {"span"}
        for key in set(node.get("attrs", {})) & refs:
            node["attrs"][key] = str(mapping[int(node["attrs"][key])])
    assert metadata_from_tensor_graph(changed) == before


def test_shared_intermediates_and_branched_outputs_have_native_stage_order():
    fixture = TensorGraphFixture()
    source = fixture.placeholder("A", [4])
    index = fixture.variable("i")
    shared = fixture.compute(
        "shared", [fixture.integer(4)], [fixture.load(source, [index])]
    )
    left = fixture.compute(
        "left", [fixture.integer(4)], [fixture.load(shared, [index])]
    )
    right = fixture.compute(
        "right", [fixture.integer(4)], [fixture.load(shared, [index])]
    )
    metadata = metadata_from_tensor_graph(fixture.graph([source, left, right]))
    dag = metadata["compute_dag"]
    names = [dag["nodes"][index]["attrs"]["name"] for index in dag["stage_order"]]
    assert names == ["A", "shared", "left", "right"]
    assert [loop["stage_id"] for loop in metadata["loop_structure"]] == [1, 2, 3]
    assert metadata["operation"] == "shared+left+right"


def test_scalar_compute_is_retained_with_empty_loops():
    fixture = TensorGraphFixture()
    constant = fixture.node("FloatImm", dtype="float32", value="2", span=0)
    output = fixture.compute("scalar_value", [], [constant])
    metadata = metadata_from_tensor_graph(fixture.graph([output]))
    assert metadata["operation"] == "scalar_value"
    assert metadata["tensors"] == [
        {"name": "scalar_value", "shape": [], "dtype": "float32"}
    ]
    assert metadata["loop_structure"][0]["spatial"] == []


def test_symbolic_dimensions_keep_expression_references_without_guessed_values():
    fixture = TensorGraphFixture()
    size = fixture.variable("n")
    constant = fixture.node("FloatImm", dtype="float32", value="2", span=0)
    output = fixture.compute("dynamic", [size], [constant])
    metadata = metadata_from_tensor_graph(fixture.graph([output]))
    dimension = metadata["tensors"][0]["shape"][0]
    assert set(dimension) == {"expr_ref"}
    node = metadata["compute_dag"]["nodes"][dimension["expr_ref"]]
    assert node["type"] == "tir.Var"
    assert metadata["loop_structure"][0]["spatial"][0]["extent"] == dimension


@pytest.mark.parametrize("name", ["i\nindex", "索引", ""])
def test_decodes_dmlc_prefixed_runtime_strings_and_empty_names(name):
    fixture = TensorGraphFixture()
    dimension = fixture.integer(4)
    axis = fixture.axis(name, dimension)
    constant = fixture.node("FloatImm", dtype="float32", value="2", span=0)
    output = fixture.compute("dynamic", [dimension], [constant])
    operation = int(fixture.nodes[output]["attrs"]["op"])
    fixture.nodes[operation]["attrs"]["axis"] = str(fixture.array([axis]))
    metadata = metadata_from_tensor_graph(fixture.graph([output]))
    assert metadata["loop_structure"][0]["spatial"][0]["variable"] == name


def test_multioutput_computation_preserves_value_indices_and_shared_op():
    fixture = TensorGraphFixture()
    value = fixture.node("FloatImm", dtype="float32", value="1", span=0)
    outputs = fixture.compute("tuple_compute", [fixture.integer(4)], [value, value])
    metadata = metadata_from_tensor_graph(fixture.graph(outputs))
    assert [tensor["name"] for tensor in metadata["tensors"]] == [
        "tuple_compute.v0",
        "tuple_compute.v1",
    ]
    assert len(metadata["compute_dag"]["stage_order"]) == 1
    indices = [
        node["attrs"]["value_index"]
        for node in metadata["compute_dag"]["nodes"]
        if node["type"] == "Tensor"
    ]
    assert sorted(indices) == ["0", "1"]


def test_reduction_preserves_combiner_sources_init_and_condition():
    fixture = TensorGraphFixture()
    source = fixture.placeholder("A", [4, 8])
    i, k = fixture.variable("i"), fixture.variable("k")
    load = fixture.load(source, [i, k])
    axis = fixture.axis("k", fixture.integer(8), minimum=3)
    fixture.nodes[axis]["attrs"]["iter_type"] = "2"
    lhs, rhs = fixture.variable("lhs"), fixture.variable("rhs")
    combination = fixture.node("tir.Add", dtype="float32", a=lhs, b=rhs, span=0)
    zero = fixture.node("FloatImm", dtype="float32", value="0", span=0)
    reducer = fixture.node(
        "tir.CommReducer",
        lhs=fixture.array([lhs]),
        rhs=fixture.array([rhs]),
        result=fixture.array([combination]),
        identity_element=fixture.array([zero]),
        span=0,
    )
    condition = fixture.node("IntImm", dtype="bool", value=1, span=0)
    reduction = fixture.node(
        "tir.Reduce",
        dtype="float32",
        combiner=reducer,
        source=fixture.array([load]),
        init=fixture.array([zero]),
        axis=fixture.array([axis]),
        condition=condition,
        value_index=0,
        span=0,
    )
    output = fixture.compute(
        "sum", [fixture.integer(4)], [reduction], reductions=[axis]
    )
    metadata = metadata_from_tensor_graph(fixture.graph([source, output]))
    loop = metadata["loop_structure"][0]
    assert loop["reduction"][0]["variable"] == "k"
    assert loop["reduction"][0]["min"] == 3
    assert loop["reduction"][0]["extent"] == 8
    node = next(
        node
        for node in metadata["compute_dag"]["nodes"]
        if node["type"] == "tir.Reduce"
    )
    assert set(node["refs"]) == {"combiner", "source", "init", "axis", "condition"}
    assert node["attrs"] == {"dtype": "float32", "value_index": "0"}
    assert any(
        node["type"] == "tir.CommReducer" for node in metadata["compute_dag"]["nodes"]
    )


def test_empty_compute_name_is_preserved_without_inventing_operation_type():
    fixture = TensorGraphFixture()
    value = fixture.node("FloatImm", dtype="float32", value="1", span=0)
    output = fixture.compute("", [], [value])
    assert metadata_from_tensor_graph(fixture.graph([output]))["operation"] == ""


def test_preserves_compute_attrs_and_named_intrinsic_call():
    fixture = TensorGraphFixture()
    op = {"type_key": "Op", "repr_str": "tir.exp"}
    fixture.nodes.append(op)
    op_id = len(fixture.nodes) - 1
    value = fixture.node("FloatImm", dtype="float32", value="1", span=0)
    call = fixture.node(
        "tir.Call", op=op_id, args=fixture.array([value]), dtype="float32", span=0
    )
    attrs = fixture.map({"layout_free_placeholders": fixture.array([])})
    output = fixture.compute("exp", [], [call], attrs=attrs)
    metadata = metadata_from_tensor_graph(fixture.graph([output]))
    text = json.dumps(metadata)
    assert "tir.exp" in text
    assert "layout_free_placeholders" in text
    assert any(node["type"] == "tir.Call" for node in metadata["compute_dag"]["nodes"])


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda graph: graph.update(root=99999), "reference a node"),
        (lambda graph: graph.update(b64ndarrays=["unhandled"]), "NDArray"),
        (lambda graph: graph["nodes"][1]["attrs"].update(value="9999"), None),
    ],
)
def test_validates_reference_indices_without_treating_literals_as_references(
    mutation, match
):
    _, graph = elementwise_graph()
    mutation(graph)
    if match:
        with pytest.raises(UnsupportedTensorGraph, match=match):
            metadata_from_tensor_graph(graph)
    else:
        assert metadata_from_tensor_graph(graph)


def test_unknown_reachable_types_fields_or_missing_semantics_fail_clearly():
    _, graph = elementwise_graph()
    addition = next(node for node in graph["nodes"] if node["type_key"] == "tir.Add")
    addition["type_key"] = "tir.UnknownFutureOperator"
    with pytest.raises(UnsupportedTensorGraph, match="node type.*UnknownFuture"):
        metadata_from_tensor_graph(graph)
    addition["type_key"] = "tir.Add"
    addition["attrs"]["future_semantics"] = "42"
    with pytest.raises(UnsupportedTensorGraph, match="unknown tir.Add fields"):
        metadata_from_tensor_graph(graph)
    del addition["attrs"]["future_semantics"]
    del addition["attrs"]["a"]
    with pytest.raises(UnsupportedTensorGraph, match="missing tir.Add fields"):
        metadata_from_tensor_graph(graph)


def test_extraction_never_imports_tvm_or_executes_registry_objects():
    _, graph = elementwise_graph()
    before = set(sys.modules)
    metadata_from_tensor_graph(graph)
    assert not {
        module
        for module in set(sys.modules) - before
        if module == "tvm" or module.startswith("tvm.")
    }


def test_rejects_non_tensor_graph_root_and_bad_base64():
    fixture = TensorGraphFixture()
    root = fixture.array([fixture.integer(4)])
    with pytest.raises(UnsupportedTensorGraph, match="no other node types"):
        metadata_from_tensor_graph({"nodes": fixture.nodes, "root": root})
    _, graph = elementwise_graph()
    node = next(
        node
        for node in graph["nodes"]
        if node["type_key"] == "runtime.String" and "repr_str" in node
    )
    del node["repr_str"]
    node["repr_b64"] = base64.b64encode(b"not-prefixed").decode("ascii")
    with pytest.raises(UnsupportedTensorGraph, match="DMLC"):
        metadata_from_tensor_graph(graph)


@pytest.mark.parametrize(
    "reference", NATIVE_REFERENCE["cases"], ids=lambda case: case["name"]
)
def test_portable_metadata_matches_original_tvm_generated_reference(reference):
    """Native TVM generated these goldens; this test imports and executes no TVM."""
    from jev_compiler_eval.datasets.tenset_registry import load_tenset_registry

    records = load_tenset_registry(NATIVE_FIXTURES / "native_registry.pkl")
    record = next(
        item for item in records if item["workload_key"] == reference["workload_key"]
    )
    graph = record["tensor_graph"]
    digest = hashlib.sha256(
        json.dumps(
            graph,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    assert digest == reference["tensor_graph_sha256"]
    metadata = metadata_from_tensor_graph(graph)
    assert metadata["tensors"] == reference["tensors"]
    compact = metadata["compute_dag"]
    assert len(compact["stage_order"]) == len(reference["stages"])
    for local_id, stage in zip(
        compact["stage_order"], reference["stages"], strict=True
    ):
        node = compact["nodes"][local_id]
        assert node["type"] == stage["type"]
        assert node["attrs"]["name"] == stage["name"]
    expected_stages = [
        index
        for index, stage in enumerate(reference["stages"])
        if stage["type"] == "ComputeOp"
    ]
    assert [loop["stage_id"] for loop in metadata["loop_structure"]] == expected_stages
    assert metadata["operation"] == "+".join(
        reference["stages"][index]["name"] for index in expected_stages
    )
    for loop in metadata["loop_structure"]:
        stage = reference["stages"][loop["stage_id"]]
        assert loop["operation_ref"] == compact["stage_order"][loop["stage_id"]]
        for field in ("spatial", "reduction"):
            assert [
                {key: axis[key] for key in ("variable", "min", "extent")}
                for axis in loop[field]
            ] == stage[field]


def test_native_same_name_and_shapes_still_preserve_different_arithmetic():
    from jev_compiler_eval.datasets.tenset_registry import load_tenset_registry

    records = {
        item["workload_key"]: item
        for item in load_tenset_registry(NATIVE_FIXTURES / "native_registry.pkl")
    }
    metadata = {}
    for reference in NATIVE_REFERENCE["cases"]:
        if reference["name"] in {"add", "multiply"}:
            metadata[reference["name"]] = metadata_from_tensor_graph(
                records[reference["workload_key"]]["tensor_graph"],
            )
    assert metadata["add"]["operation"] == metadata["multiply"]["operation"]
    assert metadata["add"]["tensors"] == metadata["multiply"]["tensors"]
    assert metadata["add"]["compute_dag"] != metadata["multiply"]["compute_dag"]
    assert "tir.Add" in {
        node["type"] for node in metadata["add"]["compute_dag"]["nodes"]
    }
    assert "tir.Mul" in {
        node["type"] for node in metadata["multiply"]["compute_dag"]["nodes"]
    }


def test_native_validation_helper_remains_python_38_compatible_and_imports_no_tvm():
    import ast
    import runpy

    helper = NATIVE_FIXTURES / "validate_native.py"
    ast.parse(helper.read_text(encoding="utf-8"), feature_version=(3, 8))
    before = set(sys.modules)
    namespace = runpy.run_path(str(helper))
    assert callable(namespace["reconstruct_savejson"])
    assert not {
        module
        for module in set(sys.modules) - before
        if module == "tvm" or module.startswith("tvm.")
    }
