"""Independent syntax guides for controlled TenSet context experiments.

These guides describe the existing serialization; they add no program features
or scheduling advice. The original field guide remains unchanged for historical
experiments. Here its graph-ID explanation belongs to the graph factor only.
"""

import json
from collections.abc import Mapping
from copy import deepcopy

from jev_compiler_eval.datasets.tenset import _reject_label_keys
from jev_compiler_eval.datasets.tenset_descriptions import tenset_configuration_fields

_GRAPH_FIELDS = {
    "format": "tenset-graph-fields-ablation-v1",
    "source": "jev_compiler_eval.datasets.tenset_metadata.metadata_from_tensor_graph",
    "underlying_format": "tvm-te-expression-graph-v1",
    "operation": (
        "Names of the compute operations in initial scheduling-stage order, joined "
        "by '+'. This is a summary; compute_dag preserves the expression graph."
    ),
    "tensors": {
        "scope": "The output tensors in the serialized graph root, in root order.",
        "name": "Operation name, with '.vN' for output N of a multiple-output op.",
        "shape": (
            "Dimension extents. An integer is a constant; an object with expr_ref "
            "identifies the corresponding compute_dag.nodes expression node."
        ),
        "dtype": "The output tensor's element type.",
    },
    "compute_dag": {
        "nodes": (
            "The complete reachable tensor-expression object graph. A node's "
            "zero-based position in this array is its graph-local ID."
        ),
        "root": "Graph-local ID of the output Tensor or Array of output Tensors.",
        "type": (
            "The TVM object type, such as Tensor, PlaceholderOp, ComputeOp, "
            "IntImm, tir.Add, tir.Reduce, Array, or Map."
        ),
        "attrs": (
            "Literal attributes preserved as strings, including names, data "
            "types, and numeric constants. These strings are not graph IDs."
        ),
        "refs": (
            "Named object references to other graph-local node IDs. A null "
            "reference represents no object. Tensor op references an operation; "
            "ComputeOp body and axis references lead to expression and axis arrays."
        ),
        "items": "Ordered graph-local node references of an Array node.",
        "entries": (
            "Map entries are key/value pairs. key_kind='string' means literal "
            "string keys and node-reference values; key_kind='node' means both "
            "keys and values are node references."
        ),
        "key_kind": "Whether a Map has literal string keys or node-reference keys.",
        "repr_str": "Literal contents of a string-backed runtime.String or Op.",
        "repr_b64": (
            "Base64 of a string representation containing an eight-byte "
            "little-endian byte length followed by UTF-8 contents."
        ),
        "stage_order": (
            "Maps each initial scheduling stage index to its operation's "
            "graph-local ID, including placeholder stages. Graph-local IDs are "
            "not stage indices. This initial mapping does not incorporate "
            "stage insertions by subsequent schedule transformations."
        ),
    },
    "loop_structure": {
        "scope": "Initial compute-stage axes, before candidate transformations.",
        "operation": "The compute operation's name.",
        "operation_ref": "Graph-local ID of the compute operation node.",
        "stage_id": "Index of the initial operation in compute_dag.stage_order.",
        "spatial": "Spatial axes in operation order.",
        "reduction": "Reduction axes in operation order.",
        "axis_fields": (
            "variable is the axis variable's name; min and extent give its Range "
            "domain, as integers or expr_ref objects; axis_ref is the graph-local "
            "ID of its tir.IterVar node."
        ),
    },
}


def tenset_candidate_fields() -> dict[str, object]:
    """Return the candidate syntax factor without the graph-field explanation."""
    guide = tenset_configuration_fields()
    guide["format"] = "tenset-candidate-fields-ablation-v1"
    guide["indices"].pop("graph_ids")
    _reject_label_keys(guide)
    return guide


def tenset_graph_fields() -> dict[str, object]:
    """Return a fresh specification of the portable TE graph serialization."""
    guide = deepcopy(_GRAPH_FIELDS)
    _reject_label_keys(guide)
    return guide


def ablation_context(
    context: Mapping[str, object],
    *,
    candidate_definitions: bool,
    graph_specification: bool,
) -> dict[str, object]:
    """Preserve raw workload data and append independently controlled guides.

    Existing appended guides are removed first so already enriched prepared data
    cannot silently turn the naive condition into a descriptions condition.
    """
    if not isinstance(candidate_definitions, bool) or not isinstance(
        graph_specification, bool
    ):
        raise ValueError("ablation description switches must be booleans")
    visible = deepcopy(
        {
            key: value
            for key, value in context.items()
            if key not in {"configuration_fields", "graph_fields"}
        }
    )
    if candidate_definitions:
        visible["configuration_fields"] = tenset_candidate_fields()
    if graph_specification:
        visible["graph_fields"] = tenset_graph_fields()
    _reject_label_keys(visible)
    json.dumps(visible, allow_nan=False)
    return visible
