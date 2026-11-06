"""Data-only registry reconstruction and hostile pickle rejection checks."""

import builtins
import copyreg
import hashlib
import json
import pickle
import sys
from pathlib import Path
from types import ModuleType

import pytest

from jev_compiler_eval.datasets.tenset_registry import (
    RegistryFormatError,
    load_tenset_registry,
)


@pytest.fixture
def registry_factory(monkeypatch):
    """Reproduce upstream reducers without needing TVM or a native library.

    Source globals deliberately raise if executed during loading. Registry
    loading must resolve their names exclusively to its own inert containers.
    """

    executed = []
    modules = {}
    for name in (
        "tvm",
        "tvm.runtime",
        "tvm.runtime.object",
        "tvm.runtime.container",
        "tvm.auto_scheduler",
        "tvm.auto_scheduler.search_task",
        "tvm.auto_scheduler.compute_dag",
        "tvm.target",
        "tvm.target.target",
        "tvm.ir",
        "tvm.ir.container",
    ):
        module = ModuleType(name)
        module.__path__ = []
        modules[name] = module
        monkeypatch.setitem(sys.modules, name, module)

    def source_new_object(cls):
        executed.append("source reducer")
        raise AssertionError("source pickle reducer must never execute")

    source_new_object.__name__ = "_new_object"
    source_new_object.__qualname__ = "_new_object"
    source_new_object.__module__ = "tvm.runtime.object"
    modules["tvm.runtime.object"]._new_object = source_new_object

    def source_reduce(self):
        return source_new_object, (type(self),), self.state

    def source_setstate(self, state):
        executed.append("source state setter")
        raise AssertionError("source pickle setter must never execute")

    classes = {}
    for module_name, name in (
        ("tvm.runtime.object", "Object"),
        ("tvm.auto_scheduler.search_task", "SearchTask"),
        ("tvm.auto_scheduler.search_task", "HardwareParams"),
        ("tvm.auto_scheduler.compute_dag", "ComputeDAG"),
        ("tvm.target.target", "Target"),
        ("tvm.ir.container", "Array"),
    ):
        cls = type(
            name,
            (),
            {
                "__module__": module_name,
                "__slots__": ("state",),
                "__reduce__": source_reduce,
                "__setstate__": source_setstate,
            },
        )
        setattr(modules[module_name], name, cls)
        classes[name] = cls

    string_cls = type(
        "String",
        (str,),
        {
            "__module__": "tvm.runtime.container",
            "__slots__": ("__tvm_object__",),
        },
    )
    modules["tvm.runtime.container"].String = string_cls

    def node(kind, state):
        instance = object.__new__(classes[kind])
        instance.state = state
        return instance

    graph = {
        "root": 0,
        "nodes": [
            {"type_key": "Array", "data": [1]},
            {"type_key": "Tensor", "attrs": {"dtype": "float32"}},
        ],
        "attrs": {"tvm_version": "0.8.dev0"},
    }

    def make_task(key='["synthetic-workload",64,128]', *, graph_value=None):
        workload_key = string_cls(key)
        workload_key.__tvm_object__ = node(
            "Object", {"handle": '{"root":0,"nodes":[]}'}
        )
        return node(
            "SearchTask",
            {
                "compute_dag": node(
                    "ComputeDAG",
                    {
                        "tensors": json.dumps(
                            graph if graph_value is None else graph_value
                        )
                    },
                ),
                "workload_key": workload_key,
                "target": node("Target", {"handle": '{"root":0,"nodes":[]}'}),
                "target_host": None,
                "hardware_params": node(
                    "HardwareParams", {"handle": '{"root":0,"nodes":[]}'}
                ),
                "layout_rewrite_option": 1,
                "task_input_names": node("Array", {"handle": '{"root":0,"nodes":[]}'}),
            },
        )

    return make_task, graph, executed


def _write_pickle(tmp_path, value, protocol=4):
    path = tmp_path / "all_tasks.pkl"
    path.write_bytes(pickle.dumps(value, protocol=protocol))
    return path


@pytest.mark.parametrize("protocol", [2, 3, 4, 5])
def test_known_upstream_states_reconstruct_without_executing_source(
    tmp_path,
    registry_factory,
    protocol,
):
    make_task, graph, executed = registry_factory
    path = _write_pickle(tmp_path, [make_task()], protocol)
    result = load_tenset_registry(path)
    assert result == (
        {"workload_key": '["synthetic-workload",64,128]', "tensor_graph": graph},
    )
    assert type(result[0]["workload_key"]) is str
    assert executed == []


def test_registry_loading_performs_no_tvm_import(
    tmp_path, registry_factory, monkeypatch
):
    make_task, _, _ = registry_factory
    task = make_task()
    # Normal Python strings and an explicit target host are also emitted by
    # historical TVM configurations, depending on the FFI return path.
    task.state["workload_key"] = str(task.state["workload_key"])
    task.state["target_host"] = task.state["target"]
    path = _write_pickle(tmp_path, [task])
    original_import = builtins.__import__

    def no_tvm_import(name, *args, **kwargs):
        if name == "tvm" or name.startswith("tvm."):
            raise AssertionError("registry loader must not import TVM")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_tvm_import)
    assert len(load_tenset_registry(path)) == 1


def test_native_registry_preserves_every_original_tensor_graph_without_tvm(monkeypatch):
    """Compare native-generated pickles with independent native SaveJSON hashes."""

    fixture_dir = Path(__file__).parent / "fixtures" / "tenset_registry"
    reference = json.loads((fixture_dir / "native_reference.json").read_text())
    assert reference["source_commit"] == "35774ed373f9693cc6efcbe83f5703141e8ab0c8"
    original_import = builtins.__import__

    def no_tvm_import(name, *args, **kwargs):
        if name == "tvm" or name.startswith("tvm."):
            raise AssertionError("native registry fixture must load without TVM")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_tvm_import)
    actual = load_tenset_registry(fixture_dir / "native_registry.pkl")
    expected_hashes = {
        case["workload_key"]: case["tensor_graph_sha256"] for case in reference["cases"]
    }
    actual_hashes = {
        task["workload_key"]: hashlib.sha256(
            json.dumps(
                task["tensor_graph"], sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        for task in actual
    }
    assert len(actual) == len(reference["cases"]) == 13
    assert actual_hashes == expected_hashes


def test_duplicate_keys_deduplicate_identical_graphs_and_reject_conflicts(
    tmp_path,
    registry_factory,
):
    make_task, graph, _ = registry_factory
    path = _write_pickle(tmp_path, [make_task(), make_task()])
    assert len(load_tenset_registry(path)) == 1
    changed = {**graph, "attrs": {"tvm_version": "changed"}}
    path = _write_pickle(tmp_path, [make_task(), make_task(graph_value=changed)])
    with pytest.raises(RegistryFormatError, match="conflicting tensor graphs"):
        load_tenset_registry(path)


def test_unknown_global_never_executes_its_reducer(tmp_path):
    marker = tmp_path / "must-not-exist"

    class Hostile:
        def __reduce__(self):
            return eval, (
                f"__import__('pathlib').Path({str(marker)!r}).write_text('bad')",
            )

    path = _write_pickle(tmp_path, [Hostile()])
    with pytest.raises(RegistryFormatError, match="unsupported pickle global"):
        load_tenset_registry(path)
    assert not marker.exists()


def test_known_class_cannot_be_called_as_arbitrary_reducer(tmp_path):
    path = tmp_path / "all_tasks.pkl"
    path.write_bytes(
        b"\x80\x02ctvm.runtime.container\nString\nX\x01\x00\x00\x00x\x85R."
    )
    with pytest.raises(RegistryFormatError, match="unsupported pickle REDUCE"):
        load_tenset_registry(path)


@pytest.mark.parametrize(
    "opcode",
    [
        b"ctvm.target.target\nTarget\n)\x81",
        b"N}b",
        b"ctvm.runtime.container\nString\n)}\x92",
    ],
)
def test_unknown_constructor_or_build_targets_are_rejected(tmp_path, opcode):
    path = tmp_path / "all_tasks.pkl"
    path.write_bytes(b"\x80\x04" + opcode + b".")
    with pytest.raises(RegistryFormatError, match="unsupported"):
        load_tenset_registry(path)


def test_extension_cache_cannot_bypass_global_allowlist(tmp_path, monkeypatch):
    monkeypatch.setitem(copyreg._extension_cache, 42, eval)
    path = tmp_path / "all_tasks.pkl"
    path.write_bytes(b"\x80\x02\x82\x2a)R.")
    with pytest.raises(RegistryFormatError, match="EXT1"):
        load_tenset_registry(path)


@pytest.mark.parametrize("opcode", [b"Pexternal\n", b"X\x01\x00\x00\x00xQ"])
def test_persistent_ids_are_rejected(tmp_path, opcode):
    path = tmp_path / "all_tasks.pkl"
    path.write_bytes(b"\x80\x02" + opcode + b".")
    with pytest.raises(RegistryFormatError, match="PERSID"):
        load_tenset_registry(path)


@pytest.mark.parametrize("suffix", [b"extra", pickle.dumps([], protocol=4)])
def test_trailing_bytes_are_not_ignored(tmp_path, registry_factory, suffix):
    make_task, _, _ = registry_factory
    path = _write_pickle(tmp_path, [make_task()])
    path.write_bytes(path.read_bytes() + suffix)
    with pytest.raises(RegistryFormatError, match="trailing"):
        load_tenset_registry(path)


@pytest.mark.parametrize("protocol", [0, 1])
def test_legacy_general_reconstructors_are_not_supported(tmp_path, protocol):
    path = _write_pickle(tmp_path, [], protocol)
    with pytest.raises(RegistryFormatError, match="protocol"):
        load_tenset_registry(path)


def test_input_size_limit_and_invalid_limit(tmp_path, registry_factory):
    make_task, _, _ = registry_factory
    path = _write_pickle(tmp_path, [make_task()])
    with pytest.raises(RegistryFormatError, match="max_bytes"):
        load_tenset_registry(path, max_bytes=10)
    for limit in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            load_tenset_registry(path, max_bytes=limit)


@pytest.mark.parametrize("value", [[], {}, ["not a task"], [None]])
def test_registry_root_and_entries_are_restricted(tmp_path, value):
    path = _write_pickle(tmp_path, value)
    with pytest.raises(RegistryFormatError):
        load_tenset_registry(path)


def test_unknown_state_fields_are_rejected(tmp_path, registry_factory):
    make_task, _, executed = registry_factory
    task = make_task()
    task.state["unexpected"] = "unsupported"
    path = _write_pickle(tmp_path, [task])
    with pytest.raises(RegistryFormatError, match="SearchTask state fields"):
        load_tenset_registry(path)
    assert executed == []


@pytest.mark.parametrize(
    "graph", [{}, {"root": 99, "nodes": []}, {"root": True, "nodes": [{}]}]
)
def test_invalid_tensor_graph_is_rejected(tmp_path, registry_factory, graph):
    make_task, _, _ = registry_factory
    path = _write_pickle(tmp_path, [make_task(graph_value=graph)])
    with pytest.raises(RegistryFormatError, match="tensor graph root"):
        load_tenset_registry(path)


def test_truncated_pickle_and_tensor_json_fail_closed(tmp_path, registry_factory):
    make_task, _, _ = registry_factory
    task = make_task()
    task.state["compute_dag"].state["tensors"] = "{"
    path = _write_pickle(tmp_path, [task])
    with pytest.raises(RegistryFormatError, match="valid JSON"):
        load_tenset_registry(path)
    path.write_bytes(path.read_bytes()[:-2])
    with pytest.raises(RegistryFormatError, match="malformed"):
        load_tenset_registry(path)
