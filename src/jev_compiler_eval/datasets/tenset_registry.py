"""Read the historical TenSet task registry without importing or executing TVM.

The supported artifact is ``all_tasks.pkl`` from the original TenSet dataset.
Only its known TVM object states are accepted. Their globals are replaced with
local data containers; source constructors, reducers and state setters never run.
Tensor expressions are returned as the original parsed TVM SaveJSON graph.
"""

from __future__ import annotations

import io
import json
import pickle
import pickletools
from pathlib import Path

DEFAULT_MAX_BYTES = 512 * 1024 * 1024


class RegistryFormatError(ValueError):
    """An artifact is outside the deliberately narrow registry format."""


class _DataObject:
    """An inert substitute for a known TVM runtime object."""

    __slots__ = ("state",)

    def _load_state(self, state: object) -> None:
        if hasattr(self, "state"):
            raise RegistryFormatError("registry object has more than one BUILD state")
        if not isinstance(state, dict) or set(state) != {"handle"}:
            raise RegistryFormatError("unsupported TVM runtime object state")
        if state["handle"] is not None and not isinstance(state["handle"], str):
            raise RegistryFormatError(
                "TVM runtime object handle must contain JSON text"
            )
        self.state = state


class _DataTarget(_DataObject):
    __slots__ = ()


class _DataHardwareParams(_DataObject):
    __slots__ = ()


class _DataArray(_DataObject):
    __slots__ = ()


class _DataComputeDAG(_DataObject):
    __slots__ = ()

    def _load_state(self, state: object) -> None:
        if hasattr(self, "state"):
            raise RegistryFormatError("registry object has more than one BUILD state")
        if (
            not isinstance(state, dict)
            or set(state) != {"tensors"}
            or not isinstance(state["tensors"], str)
        ):
            raise RegistryFormatError("unsupported TVM ComputeDAG state")
        self.state = state


class _DataSearchTask(_DataObject):
    __slots__ = ()

    def _load_state(self, state: object) -> None:
        if hasattr(self, "state"):
            raise RegistryFormatError("registry object has more than one BUILD state")
        required = {
            "compute_dag",
            "workload_key",
            "target",
            "target_host",
            "hardware_params",
            "layout_rewrite_option",
            "task_input_names",
        }
        if not isinstance(state, dict) or set(state) != required:
            raise RegistryFormatError("unsupported TVM SearchTask state fields")
        if (
            type(state["compute_dag"]) is not _DataComputeDAG
            or not isinstance(state["workload_key"], str)
            or not state["workload_key"]
            or type(state["target"]) is not _DataTarget
            or (
                state["target_host"] is not None
                and type(state["target_host"]) is not _DataTarget
            )
            or type(state["hardware_params"]) is not _DataHardwareParams
            or type(state["layout_rewrite_option"]) is not int
            or type(state["task_input_names"]) is not _DataArray
        ):
            raise RegistryFormatError("unsupported TVM SearchTask state values")
        self.state = state


class _DataString(str):
    """TVM String content plus its inert default pickle slot state."""

    __slots__ = ("state",)

    def _load_state(self, state: object) -> None:
        if hasattr(self, "state"):
            raise RegistryFormatError("registry string has more than one BUILD state")
        # String is a str subclass with __slots__=['__tvm_object__']; CPython
        # pickles its slot mapping as the second element of this tuple.
        if not isinstance(state, tuple) or len(state) != 2 or state[0] is not None:
            raise RegistryFormatError("unsupported TVM String slot state")
        slots = state[1]
        if (
            not isinstance(slots, dict)
            or set(slots) != {"__tvm_object__"}
            or type(slots["__tvm_object__"]) is not _DataObject
        ):
            raise RegistryFormatError("unsupported TVM String runtime object")
        self.state = slots


_OBJECT_CLASSES = frozenset(
    {
        _DataObject,
        _DataTarget,
        _DataHardwareParams,
        _DataArray,
        _DataComputeDAG,
        _DataSearchTask,
    }
)


def _data_new_object(cls: type) -> _DataObject:
    """The sole allowed reducer; it only creates a local empty data container."""

    if cls not in _OBJECT_CLASSES:
        raise RegistryFormatError("unsupported TVM object reducer class")
    return object.__new__(cls)


_GLOBALS = {
    ("tvm.runtime.object", "_new_object"): _data_new_object,
    ("tvm.runtime.object", "Object"): _DataObject,
    ("tvm.target.target", "Target"): _DataTarget,
    ("tvm.auto_scheduler.search_task", "HardwareParams"): _DataHardwareParams,
    ("tvm.auto_scheduler.search_task", "SearchTask"): _DataSearchTask,
    ("tvm.auto_scheduler.compute_dag", "ComputeDAG"): _DataComputeDAG,
    ("tvm.ir.container", "Array"): _DataArray,
    ("tvm.runtime.container", "String"): _DataString,
}


class _RegistryUnpickler(pickle._Unpickler):
    """Restrict constructor opcodes as well as global resolution.

    CPython's Python implementation is used so REDUCE/NEWOBJ/BUILD dispatch is
    explicitly controlled. In particular, extension-cache objects cannot bypass
    ``find_class`` and source pickle callables are never invoked.
    """

    def find_class(self, module: str, name: str) -> object:
        try:
            return _GLOBALS[(module, name)]
        except KeyError as error:
            raise RegistryFormatError(
                f"unsupported pickle global: {module}.{name}"
            ) from error

    def persistent_load(self, pid: object) -> object:
        raise RegistryFormatError("persistent pickle IDs are not supported")

    def load_reduce(self) -> None:
        arguments = self.stack.pop()
        reducer = self.stack[-1]
        if (
            reducer is not _data_new_object
            or not isinstance(arguments, tuple)
            or len(arguments) != 1
            or arguments[0] not in _OBJECT_CLASSES
        ):
            raise RegistryFormatError("unsupported pickle REDUCE callable or arguments")
        self.stack[-1] = _data_new_object(arguments[0])

    def load_newobj(self) -> None:
        arguments = self.stack.pop()
        cls = self.stack.pop()
        if (
            cls is not _DataString
            or not isinstance(arguments, tuple)
            or len(arguments) != 1
            or not isinstance(arguments[0], str)
        ):
            raise RegistryFormatError("unsupported pickle NEWOBJ class or arguments")
        self.append(str.__new__(_DataString, str(arguments[0])))

    def load_build(self) -> None:
        state = self.stack.pop()
        instance = self.stack[-1]
        if type(instance) not in _OBJECT_CLASSES and type(instance) is not _DataString:
            raise RegistryFormatError("unsupported pickle BUILD instance")
        instance._load_state(state)

    def load_forbidden(self) -> None:
        raise RegistryFormatError("unsupported executable or external pickle opcode")

    dispatch = pickle._Unpickler.dispatch.copy()
    dispatch[pickle.REDUCE[0]] = load_reduce
    dispatch[pickle.NEWOBJ[0]] = load_newobj
    dispatch[pickle.BUILD[0]] = load_build
    for _opcode in (
        pickle.EXT1,
        pickle.EXT2,
        pickle.EXT4,
        pickle.INST,
        pickle.OBJ,
        pickle.NEWOBJ_EX,
        pickle.NEXT_BUFFER,
        pickle.READONLY_BUFFER,
    ):
        dispatch[_opcode[0]] = load_forbidden


def _check_stream(data: bytes) -> None:
    if len(data) < 3 or data[0] != pickle.PROTO[0] or data[1] not in {2, 3, 4, 5}:
        raise RegistryFormatError(
            "TenSet registry requires pickle protocol 2, 3, 4 or 5"
        )
    forbidden = {
        "EXT1",
        "EXT2",
        "EXT4",
        "INST",
        "OBJ",
        "PERSID",
        "BINPERSID",
        "NEWOBJ_EX",
        "NEXT_BUFFER",
        "READONLY_BUFFER",
    }
    try:
        last = None
        for opcode, _, position in pickletools.genops(data):
            if opcode.name in forbidden:
                raise RegistryFormatError(f"unsupported pickle opcode: {opcode.name}")
            last = (opcode.name, position)
        if last != ("STOP", len(data) - 1):
            raise RegistryFormatError("trailing or incomplete TenSet registry data")
    except (ValueError, OverflowError) as error:
        if isinstance(error, RegistryFormatError):
            raise
        raise RegistryFormatError("malformed TenSet registry pickle") from error


def _tensor_graph(task: _DataSearchTask) -> dict[str, object]:
    try:
        graph = json.loads(task.state["compute_dag"].state["tensors"])
    except (AttributeError, KeyError, json.JSONDecodeError, RecursionError) as error:
        raise RegistryFormatError(
            "ComputeDAG tensor state must contain valid JSON"
        ) from error
    if (
        not isinstance(graph, dict)
        or not isinstance(graph.get("nodes"), list)
        or type(graph.get("root")) is not int
        or not 0 <= graph["root"] < len(graph["nodes"])
    ):
        raise RegistryFormatError("unsupported TVM SaveJSON tensor graph root")
    return graph


def _load_registry_data(path: str | Path, max_bytes: int) -> object:
    """Read either supported container through the same inert pickle boundary."""

    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("registry max_bytes must be a positive integer")
    with Path(path).open("rb") as artifact:
        data = artifact.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise RegistryFormatError("TenSet registry exceeds max_bytes")
    _check_stream(data)
    try:
        return _RegistryUnpickler(io.BytesIO(data)).load()
    except RegistryFormatError:
        raise
    except (
        pickle.UnpicklingError,
        EOFError,
        ValueError,
        TypeError,
        AttributeError,
        KeyError,
        IndexError,
        OverflowError,
        RecursionError,
    ) as error:
        raise RegistryFormatError(
            "malformed or unsupported TenSet registry pickle"
        ) from error


def load_tenset_registry(
    path: str | Path, *, max_bytes: int = DEFAULT_MAX_BYTES
) -> tuple[dict[str, object], ...]:
    """Return workload keys and parsed tensor graphs from a restricted registry.

    Binary pickle protocols 2–5 are supported. Unknown globals, reducers,
    constructors, state layouts, extensions, external buffers and persistent
    IDs fail closed. This function performs no TVM imports, tensor execution,
    dynamic module loading or native deserialization. Duplicate workload keys
    with identical graphs are deduplicated; conflicting graphs are rejected.
    ``max_bytes`` bounds the input read, not the expanded Python object size.
    """

    tasks = _load_registry_data(path, max_bytes)
    if not isinstance(tasks, list) or not tasks:
        raise RegistryFormatError("TenSet registry must contain a nonempty task list")
    result = {}
    for task in tasks:
        if type(task) is not _DataSearchTask or not hasattr(task, "state"):
            raise RegistryFormatError("TenSet registry contains an unsupported task")
        workload_key = str(task.state["workload_key"])
        graph = _tensor_graph(task)
        if workload_key in result and result[workload_key] != graph:
            raise RegistryFormatError("conflicting tensor graphs for one workload key")
        result[workload_key] = graph
    return tuple(
        {"workload_key": workload_key, "tensor_graph": graph}
        for workload_key, graph in result.items()
    )


def load_tenset_task_info_keys(
    path: str | Path, *, max_bytes: int = DEFAULT_MAX_BYTES
) -> list[str]:
    """Read ordered keys from an official ``(tasks, weights)`` network pickle.

    The same inert loader used for ``all_tasks.pkl`` rejects executable pickle
    operations. Only the network-info tuple, task list and positive integer
    weights emitted by TenSet are accepted. Task order and duplicate keys are
    preserved; the tutorial applies first-occurrence deduplication across the
    ordered network files after loading them. Tensor graphs are format-checked
    but no TVM deserialization or computation is performed.
    """

    value = _load_registry_data(path, max_bytes)
    if not isinstance(value, tuple) or len(value) != 2:
        raise RegistryFormatError("TenSet task info must contain (tasks, weights)")
    tasks, weights = value
    if (
        not isinstance(tasks, list)
        or not tasks
        or not isinstance(weights, list)
        or len(weights) != len(tasks)
        or any(type(weight) is not int or weight <= 0 for weight in weights)
    ):
        raise RegistryFormatError(
            "TenSet task info requires a nonempty task list and matching "
            "positive integer weights"
        )
    result = []
    for task in tasks:
        if type(task) is not _DataSearchTask or not hasattr(task, "state"):
            raise RegistryFormatError("TenSet task info contains an unsupported task")
        _tensor_graph(task)
        result.append(str(task.state["workload_key"]))
    return result
