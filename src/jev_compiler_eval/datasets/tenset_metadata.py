"""Extract TE semantics from TVM SaveJSON without importing or executing TVM.

SaveJSON encodes object references and scalar integers as indistinguishable
decimal strings. The field schemas below follow the TenSet TVM fork's VisitAttrs
methods; numeric strings are never guessed to be references. Unsupported node
types or fields fail explicitly rather than producing incomplete programs.
"""

import base64
import heapq
import json
import re
from collections import deque
from collections.abc import Mapping

GRAPH_FORMAT = "tvm-te-expression-graph-v1"
_INDEX = re.compile(r"0|[1-9][0-9]*")

# Each schema lists reference-backed and literal fields, respectively. Source
# spans are debug metadata, not computation, and are deliberately not traversed.
# Sources: include/tvm/{te/{tensor,operation},tir/{expr,var},ir/{expr,type}}.h
# and src/node/serialization.cc in the pinned TenSet repository.
_SCHEMAS: dict[str, tuple[set[str], set[str]]] = {
    "Tensor": ({"shape", "op"}, {"dtype", "value_index"}),
    "PlaceholderOp": ({"attrs", "shape"}, {"name", "tag", "dtype"}),
    "ComputeOp": (
        {"attrs", "axis", "reduce_axis", "body"},
        {"name", "tag"},
    ),
    "Range": ({"min", "extent"}, set()),
    "IntImm": (set(), {"dtype", "value"}),
    "FloatImm": (set(), {"dtype", "value"}),
    "tir.Var": ({"name", "type_annotation"}, {"dtype"}),
    "tir.SizeVar": ({"name", "type_annotation"}, {"dtype"}),
    "tir.IterVar": ({"dom", "var", "thread_tag"}, {"iter_type"}),
    "tir.StringImm": ({"value"}, {"dtype"}),
    "tir.Any": (set(), {"dtype"}),
    "tir.Cast": ({"value"}, {"dtype"}),
    "tir.Not": ({"a"}, {"dtype"}),
    "tir.Select": ({"condition", "true_value", "false_value"}, {"dtype"}),
    "tir.ProducerLoad": ({"producer", "indices"}, {"dtype"}),
    "tir.Load": ({"buffer_var", "index", "predicate"}, {"dtype"}),
    "tir.Ramp": ({"base", "stride"}, {"dtype", "lanes"}),
    "tir.Broadcast": ({"value"}, {"dtype", "lanes"}),
    "tir.Let": ({"var", "value", "body"}, {"dtype"}),
    "tir.Call": ({"op", "args"}, {"dtype"}),
    "tir.Shuffle": ({"vectors", "indices"}, set()),
    "tir.CommReducer": ({"lhs", "rhs", "result", "identity_element"}, set()),
    "tir.Reduce": (
        {"combiner", "source", "init", "axis", "condition"},
        {"dtype", "value_index"},
    ),
    "PrimType": (set(), {"dtype"}),
    "PointerType": ({"element_type"}, set()),
    "GlobalVar": ({"name_hint", "_checked_type_"}, set()),
    "DictAttrs": ({"__dict__"}, set()),
}
for _binary in (
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Mod",
    "FloorDiv",
    "FloorMod",
    "Min",
    "Max",
    "EQ",
    "NE",
    "LT",
    "LE",
    "GT",
    "GE",
    "And",
    "Or",
):
    _SCHEMAS[f"tir.{_binary}"] = ({"a", "b"}, {"dtype"})


class UnsupportedTensorGraph(ValueError):
    """A malformed or unsupported SaveJSON graph cannot be extracted safely."""


def _fail(message: str) -> None:
    raise UnsupportedTensorGraph(f"unsupported TVM SaveJSON tensor graph: {message}")


def _index(value: object, *, count: int, location: str) -> int:
    if isinstance(value, str) and _INDEX.fullmatch(value):
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < count:
        _fail(f"{location} must reference a node in [0, {count})")
    return value


def _string_repr(node: Mapping[str, object]) -> str:
    """Decode TVM's string representation, including its DMLC length prefix."""
    if "repr_str" in node:
        return str(node["repr_str"])
    if "repr_b64" not in node:
        return ""
    try:
        data = base64.b64decode(node["repr_b64"], validate=True)
    except (ValueError, TypeError) as error:
        raise UnsupportedTensorGraph("invalid SaveJSON string base64") from error
    # DMLC's NativePODStringHandler writes an uint64 length before the bytes.
    # TenSet's published x86 datasets serialize native little-endian values.
    if len(data) < 8 or int.from_bytes(data[:8], "little") != len(data) - 8:
        _fail("repr_b64 must contain a DMLC uint64LE-prefixed string")
    try:
        return data[8:].decode("utf-8")
    except UnicodeDecodeError as error:
        raise UnsupportedTensorGraph("SaveJSON name is not UTF-8") from error


class _TensorGraph:
    def __init__(self, value: str | Mapping[str, object]) -> None:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as error:
                raise UnsupportedTensorGraph("invalid SaveJSON JSON") from error
        if not isinstance(value, Mapping) or set(value) - {
            "root",
            "nodes",
            "b64ndarrays",
            "attrs",
        }:
            _fail("expected root/nodes SaveJSON object")
        nodes = value.get("nodes")
        if not isinstance(nodes, list) or not nodes:
            _fail("nodes must be a non-empty list")
        if value.get("b64ndarrays", []) != []:
            _fail("NDArray-backed tensor graphs are not supported")
        self.nodes = nodes
        self.root = _index(value.get("root"), count=len(nodes), location="root")
        self.ids: dict[int, int] = {}
        self.compact: list[dict] = []
        self.pending: deque[int] = deque()
        root_ref = self.reference(self.root)
        if root_ref is None:
            _fail("root cannot be null")
        while self.pending:
            original = self.pending.popleft()
            self.compact[self.ids[original]] = self._compact_node(original)
        root_type = self.node(self.root)["type_key"]
        if root_type == "Array":
            self.tensors = self.array(self.root)
        elif root_type == "Tensor":
            self.tensors = [self.root]
        else:
            _fail("root must be a Tensor or Array of Tensor nodes")
        if not self.tensors or any(
            self.node(index)["type_key"] != "Tensor" for index in self.tensors
        ):
            _fail("root must contain at least one Tensor and no other node types")

    def node(self, index: int) -> Mapping[str, object]:
        index = _index(index, count=len(self.nodes), location="node index")
        node = self.nodes[index]
        if not isinstance(node, Mapping) or not isinstance(node.get("type_key"), str):
            _fail(f"node {index} must have a string type_key")
        return node

    def reference(self, value: object) -> int | None:
        original = _index(value, count=len(self.nodes), location="object reference")
        if self.node(original)["type_key"] == "":
            if original != 0:
                _fail("only node zero may represent null")
            return None
        if original not in self.ids:
            self.ids[original] = len(self.compact)
            self.compact.append({})
            self.pending.append(original)
        return self.ids[original]

    def attrs(self, index: int) -> Mapping[str, str]:
        attrs = self.node(index).get("attrs", {})
        if not isinstance(attrs, Mapping) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in attrs.items()
        ):
            _fail(f"node {index} attrs must map strings to strings")
        return attrs

    def required(self, index: int, name: str) -> str:
        attrs = self.attrs(index)
        if name not in attrs:
            _fail(f"{self.node(index)['type_key']} node {index} lacks {name}")
        return attrs[name]

    def field(self, index: int, name: str) -> int:
        return _index(
            self.required(index, name),
            count=len(self.nodes),
            location=f"node {index}.{name}",
        )

    def array(self, index: int) -> list[int]:
        node = self.node(index)
        if node["type_key"] != "Array":
            _fail(f"expected Array at node {index}, got {node['type_key']}")
        data = node.get("data", [])
        if not isinstance(data, list):
            _fail(f"Array node {index} data must be a list")
        return [
            _index(item, count=len(self.nodes), location="array item") for item in data
        ]

    def _compact_node(self, index: int) -> dict:
        node = self.node(index)
        if set(node) - {"type_key", "attrs", "data", "keys", "repr_str", "repr_b64"}:
            _fail(f"unexpected fields in node {index}")
        kind = node["type_key"]
        result: dict[str, object] = {"type": kind}
        if kind in {"runtime.String", "Op"}:
            if "attrs" in node or "data" in node or "keys" in node:
                _fail(f"{kind} must use a string representation")
            if "repr_str" in node and "repr_b64" in node:
                _fail(f"node {index} has conflicting string representations")
            for field in ("repr_str", "repr_b64"):
                if field in node:
                    if not isinstance(node[field], str):
                        _fail(f"node {index}.{field} must be a string")
                    result[field] = node[field]
            _string_repr(node)
            return result
        if "repr_str" in node or "repr_b64" in node:
            _fail(f"unsupported string-backed node type {kind}")
        if kind == "Array":
            if self.attrs(index) or "keys" in node:
                _fail(f"Array node {index} has unexpected attrs or keys")
            result["items"] = [self.reference(item) for item in self.array(index)]
            return result
        if kind == "Map":
            if self.attrs(index):
                _fail(f"Map node {index} has unexpected attrs")
            data, keys = node.get("data", []), node.get("keys", [])
            if not isinstance(data, list) or not isinstance(keys, list):
                _fail(f"Map node {index} keys/data must be lists")
            if keys:
                if len(keys) != len(data) or not all(
                    isinstance(key, str) for key in keys
                ):
                    _fail(f"Map node {index} keys/data lengths or key types disagree")
                if len(keys) != len(set(keys)):
                    _fail(f"Map node {index} has duplicate string keys")
                result["entries"] = [
                    [key, self.reference(item)]
                    for key, item in sorted(zip(keys, data, strict=True))
                ]
                result["key_kind"] = "string"
            else:
                if len(data) % 2:
                    _fail(f"Map node {index} object data must have key/value pairs")
                result["entries"] = [
                    [self.reference(data[position]), self.reference(data[position + 1])]
                    for position in range(0, len(data), 2)
                ]
                result["key_kind"] = "node"
            return result
        if kind not in _SCHEMAS:
            _fail(
                f"node type {kind!r} at index {index}; "
                "use a supported TE expression graph"
            )
        if "data" in node or "keys" in node:
            _fail(f"{kind} node {index} has unexpected container fields")
        references, literals = _SCHEMAS[kind]
        attrs = self.attrs(index)
        unknown = set(attrs) - references - literals - {"span"}
        if unknown:
            _fail(f"unknown {kind} fields: {', '.join(sorted(unknown))}")
        missing = (references | literals) - attrs.keys()
        if kind == "tir.Reduce":
            missing -= {"init"}  # Older compatible records predate explicit init.
        if missing:
            _fail(f"missing {kind} fields: {', '.join(sorted(missing))}")
        scalar_values = {key: attrs[key] for key in sorted(literals & attrs.keys())}
        ref_values = {
            key: self.reference(attrs[key]) for key in sorted(references & attrs.keys())
        }
        if scalar_values:
            result["attrs"] = scalar_values
        if ref_values:
            result["refs"] = ref_values
        return result

    def name(self, index: int) -> str:
        if self.node(index)["type_key"] in {"tir.Var", "tir.SizeVar"}:
            name = self.node(self.field(index, "name"))
            if name["type_key"] != "runtime.String":
                _fail("tir.Var name must reference runtime.String")
            return _string_repr(name)
        return self.required(index, "name")

    def expression(self, index: int) -> int | dict[str, int]:
        if self.node(index)["type_key"] == "IntImm":
            value = self.required(index, "value")
            try:
                return int(value)
            except ValueError as error:
                raise UnsupportedTensorGraph(
                    "IntImm value must be an integer"
                ) from error
        return {"expr_ref": self.ids[index]}

    def axes(self, operation: int, field: str) -> list[dict]:
        result = []
        for axis in self.array(self.field(operation, field)):
            if self.node(axis)["type_key"] != "tir.IterVar":
                _fail("ComputeOp axes must contain tir.IterVar nodes")
            domain = self.field(axis, "dom")
            if self.node(domain)["type_key"] != "Range":
                _fail("ComputeOp axis requires a known Range domain")
            variable = self.field(axis, "var")
            if self.node(variable)["type_key"] not in {"tir.Var", "tir.SizeVar"}:
                _fail("IterVar var must reference tir.Var or tir.SizeVar")
            result.append(
                {
                    "variable": self.name(variable),
                    "min": self.expression(self.field(domain, "min")),
                    "extent": self.expression(self.field(domain, "extent")),
                    "axis_ref": self.ids[axis],
                }
            )
        return result

    def expression_children(self, index: int) -> list[int]:
        """Match TIR ExprVisitor traversal order, not SaveJSON field ordering."""
        kind = self.node(index)["type_key"]
        if kind in {
            "IntImm",
            "FloatImm",
            "tir.StringImm",
            "tir.Var",
            "tir.SizeVar",
            "tir.Any",
        }:
            return []
        if kind in {
            f"tir.{binary}"
            for binary in (
                "Add",
                "Sub",
                "Mul",
                "Div",
                "Mod",
                "FloorDiv",
                "FloorMod",
                "Min",
                "Max",
                "EQ",
                "NE",
                "LT",
                "LE",
                "GT",
                "GE",
                "And",
                "Or",
            )
        }:
            return [self.field(index, "a"), self.field(index, "b")]
        fields = {
            "tir.Cast": ("value",),
            "tir.Not": ("a",),
            "tir.Select": ("condition", "true_value", "false_value"),
            "tir.Load": ("index", "predicate"),
            "tir.Ramp": ("base", "stride"),
            "tir.Broadcast": ("value",),
            "tir.Let": ("value", "body"),
        }
        if kind in fields:
            return [self.field(index, field) for field in fields[kind]]
        array_fields = {
            "tir.Call": ("args",),
            "tir.ProducerLoad": ("indices",),
            "tir.Shuffle": ("indices", "vectors"),
        }
        if kind in array_fields:
            return [
                child
                for field in array_fields[kind]
                for child in self.array(self.field(index, field))
            ]
        if kind == "tir.Reduce":
            children = []
            for axis in self.array(self.field(index, "axis")):
                domain = self.field(axis, "dom")
                children.extend(
                    [self.field(domain, "min"), self.field(domain, "extent")]
                )
            children.extend(self.array(self.field(index, "source")))
            if "init" in self.attrs(index):
                children.extend(self.array(self.field(index, "init")))
            children.append(self.field(index, "condition"))
            return children
        _fail(f"unsupported expression visitor type {kind!r}")

    def input_tensors(self, operation: int) -> list[int]:
        """Match ComputeOpNode::InputTensors' postorder and tensor deduplication."""
        if self.node(operation)["type_key"] == "PlaceholderOp":
            return []
        if self.node(operation)["type_key"] != "ComputeOp":
            _fail("only PlaceholderOp and ComputeOp are supported by ComputeDAG")
        tensors, tensor_set = [], set()
        for expression in self.array(self.field(operation, "body")):
            visited: set[int] = set()
            stack = [(expression, False)]
            while stack:
                current, finish = stack.pop()
                if finish:
                    if self.node(current)["type_key"] == "tir.ProducerLoad":
                        tensor = self.field(current, "producer")
                        if self.node(tensor)["type_key"] != "Tensor":
                            _fail("ProducerLoad must reference a Tensor")
                        if tensor not in tensor_set:
                            tensors.append(tensor)
                            tensor_set.add(tensor)
                    continue
                if current in visited:
                    continue
                visited.add(current)
                stack.append((current, True))
                stack.extend(
                    (child, False)
                    for child in reversed(self.expression_children(current))
                )
        return tensors

    def stage_order(self) -> list[int]:
        """Reproduce the pinned TenSet ComputeDAG constructor's initial stages.

        Sources: auto_scheduler/compute_dag.cc TopoSortOps and ComputeDAG(Array),
        te/operation/compute_op.cc InputTensors, te/schedule/graph.cc PostDFSOrder.
        Output selection is consumer-free ComputeOps; placeholders are excluded.
        """
        priority, dependencies, consumers = {}, {}, {}
        stack = [self.field(tensor, "op") for tensor in self.tensors]
        while stack:
            operation = stack.pop()
            if operation in priority:
                continue
            priority[operation] = len(priority)
            tensors = self.input_tensors(operation)
            dependencies[operation] = [self.field(tensor, "op") for tensor in tensors]
            for producer in dependencies[operation]:
                consumers.setdefault(producer, []).append(operation)
                stack.append(producer)
        degree = {operation: len(deps) for operation, deps in dependencies.items()}
        queue = [
            (-priority[operation], operation)
            for operation, count in degree.items()
            if count == 0
        ]
        heapq.heapify(queue)
        topological = []
        while queue:
            _, operation = heapq.heappop(queue)
            topological.append(operation)
            for consumer in consumers.get(operation, []):
                degree[consumer] -= 1
                if degree[consumer] == 0:
                    heapq.heappush(queue, (-priority[consumer], consumer))
        if len(topological) != len(priority):
            _fail("cycle in tensor producer dependencies")
        outputs = [
            operation
            for operation in topological
            if self.node(operation)["type_key"] == "ComputeOp"
            and not consumers.get(operation)
        ]
        stages, visited = [], set()
        for output in outputs:
            stack = [(output, False)]
            while stack:
                operation, finish = stack.pop()
                if finish:
                    stages.append(operation)
                    continue
                if operation in visited:
                    continue
                visited.add(operation)
                stack.append((operation, True))
                stack.extend(
                    (producer, False) for producer in reversed(dependencies[operation])
                )
        return stages


def metadata_from_tensor_graph(graph: str | Mapping[str, object]) -> dict[str, object]:
    """Return tensors, real operation names, axes, and the complete expression DAG.

    ``compute_dag.nodes`` uses its list positions as stable local IDs. ``refs``,
    Array ``items``, and Map node-valued ``entries`` contain IDs (null is None).
    Literal attrs preserve source strings exactly, including floating-point
    constants. Symbolic tensor dimensions and loop bounds refer to those nodes.
    Graph local IDs differ from scheduling stage indices. ``stage_order`` maps
    initial stage indices to operation local IDs using the pinned TVM algorithm.
    Later transformations can create, remove, or reorder scheduling stages.
    """
    parsed = _TensorGraph(graph)
    tensors = []
    for tensor in parsed.tensors:
        operation = parsed.field(tensor, "op")
        kind = parsed.node(operation)["type_key"]
        if kind not in {"PlaceholderOp", "ComputeOp"}:
            _fail("tensor op must be PlaceholderOp or ComputeOp")
        name = parsed.name(operation)
        try:
            output_index = int(parsed.required(tensor, "value_index"))
        except ValueError as error:
            raise UnsupportedTensorGraph(
                "Tensor value_index must be an integer"
            ) from error
        outputs = (
            len(parsed.array(parsed.field(operation, "body")))
            if kind == "ComputeOp"
            else 1
        )
        if not 0 <= output_index < outputs:
            _fail("Tensor value_index must identify an existing operation output")
        if outputs > 1:
            name += f".v{output_index}"
        tensors.append(
            {
                "name": name,
                "shape": [
                    parsed.expression(dimension)
                    for dimension in parsed.array(parsed.field(tensor, "shape"))
                ],
                "dtype": parsed.required(tensor, "dtype"),
            }
        )
    operation_names, loops = [], []
    stages = parsed.stage_order()
    for stage_id, operation in enumerate(stages):
        if parsed.node(operation)["type_key"] == "ComputeOp":
            name = parsed.name(operation)
            operation_names.append(name)
            loops.append(
                {
                    "operation": name,
                    "operation_ref": parsed.ids[operation],
                    "stage_id": stage_id,
                    "spatial": parsed.axes(operation, "axis"),
                    "reduction": parsed.axes(operation, "reduce_axis"),
                }
            )
    return {
        "operation": "+".join(operation_names) if operation_names else "placeholder",
        "tensors": tensors,
        "loop_structure": loops,
        "compute_dag": {
            "format": GRAPH_FORMAT,
            "root": parsed.ids[parsed.root],
            "nodes": parsed.compact,
            "stage_order": [parsed.ids[operation] for operation in stages],
        },
    }
