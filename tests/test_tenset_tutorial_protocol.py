"""Reference-order file selection and inert network-info pickle boundaries."""

import builtins
import hashlib
import json
import pickle

import pytest
import test_tenset_registry as registry_fixtures

from jev_compiler_eval.datasets.tenset_registry import (
    RegistryFormatError,
    load_tenset_registry,
    load_tenset_task_info_keys,
)
from jev_compiler_eval.datasets.tenset_tutorial_protocol import (
    TUTORIAL_COMMIT,
    batch_size_one_network_keys,
    clean_name,
    f32,
    tutorial_file_sample,
)

# Keep every restricted pickle consumer on the same upstream-format constructors.
registry_factory = registry_fixtures.registry_factory


def test_preset_matches_reference_order_and_network_shapes():
    networks = batch_size_one_network_keys()
    assert len(networks) == 40
    assert [name for name, _ in networks] == [
        "resnet_18",
        "resnet_50",
        "resnet_18",
        "resnet_50",
        "resnet_18",
        "resnet_50",
        "mobilenet_v2",
        "mobilenet_v3",
        "mobilenet_v2",
        "mobilenet_v3",
        "mobilenet_v2",
        "mobilenet_v3",
        "wide_resnet_50",
        "wide_resnet_50",
        "wide_resnet_50",
        "resnext_50",
        "resnext_50",
        "resnext_50",
        "inception_v3",
        "densenet_121",
        "densenet_121",
        "densenet_121",
        "resnet3d_18",
        "resnet3d_18",
        "resnet3d_18",
        "bert_tiny",
        "bert_base",
        "bert_medium",
        "bert_large",
        "bert_tiny",
        "bert_base",
        "bert_medium",
        "bert_large",
        "bert_tiny",
        "bert_base",
        "bert_medium",
        "bert_large",
        "dcgan",
        "dcgan",
        "dcgan",
    ]
    assert networks[0] == ("resnet_18", [(1, 3, 224, 224)])
    assert networks[5] == ("resnet_50", [(1, 3, 256, 256)])
    assert networks[18] == ("inception_v3", [(1, 3, 299, 299)])
    assert networks[22] == ("resnet3d_18", [(1, 3, 112, 112, 16)])
    assert networks[24] == ("resnet3d_18", [(1, 3, 144, 144, 16)])
    assert networks[25] == ("bert_tiny", [(1, 64)])
    assert networks[36] == ("bert_large", [(1, 256)])
    assert networks[-1] == ("dcgan", [(1, 3, 96, 96)])
    networks[0][1].append((999,))
    assert batch_size_one_network_keys()[0][1] == [(1, 3, 224, 224)]


def test_upstream_clean_name_matches_real_network_and_workload_filenames():
    network = ("resnet_18", [(1, 3, 224, 224)])
    assert clean_name((network, "llvm")) == "((resnet_18,[(1,3,224,224)]),llvm)"
    assert clean_name(('["hash", 1, 64]', "llvm")) == "([hash,1,64],llvm)"


@pytest.mark.parametrize("protocol", [2, 3, 4, 5])
def test_network_pickle_is_read_without_source_execution_or_deduplication(
    tmp_path, registry_factory, monkeypatch, protocol
):
    make_task, _, executed = registry_factory
    keys = ['["second",2]', '["first",1]', '["second",2]']
    path = tmp_path / "network.task.pkl"
    path.write_bytes(
        pickle.dumps(([make_task(key) for key in keys], [2, 1, 3]), protocol)
    )
    original_import = builtins.__import__

    def no_tvm_import(name, *args, **kwargs):
        if name == "tvm" or name.startswith("tvm."):
            raise AssertionError("task-info loader must not import TVM")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_tvm_import)
    assert load_tenset_task_info_keys(path) == keys
    assert executed == []
    with pytest.raises(RegistryFormatError, match="nonempty task list"):
        load_tenset_registry(path)


@pytest.mark.parametrize(
    "kind",
    [
        "list_root",
        "missing_weights",
        "empty_tasks",
        "float_weight",
        "bool_weight",
        "zero_weight",
        "bad_task",
    ],
)
def test_network_pickle_rejects_unsupported_container_shapes(
    tmp_path, registry_factory, kind
):
    make_task, _, _ = registry_factory
    task = make_task()
    examples = {
        "list_root": [[task], [1]],
        "missing_weights": ([task], []),
        "empty_tasks": ([], []),
        "float_weight": ([task], [1.0]),
        "bool_weight": ([task], [True]),
        "zero_weight": ([task], [0]),
        "bad_task": (["task"], [1]),
    }
    path = tmp_path / "bad.task.pkl"
    path.write_bytes(pickle.dumps(examples[kind], protocol=4))
    with pytest.raises(RegistryFormatError):
        load_tenset_task_info_keys(path)


def test_network_pickle_rejects_executable_reducer_and_trailing_bytes(tmp_path):
    marker = tmp_path / "must-not-exist"

    class Hostile:
        def __reduce__(self):
            return eval, (
                f"__import__('pathlib').Path({str(marker)!r}).write_text('executed')",
            )

    path = tmp_path / "hostile.task.pkl"
    path.write_bytes(pickle.dumps(([Hostile()], [1]), protocol=4))
    with pytest.raises(RegistryFormatError, match="unsupported pickle global"):
        load_tenset_task_info_keys(path)
    assert not marker.exists()
    path.write_bytes(pickle.dumps(([], []), protocol=4) + b"ignored")
    with pytest.raises(RegistryFormatError, match="trailing"):
        load_tenset_task_info_keys(path)


@pytest.fixture
def tutorial_tree(tmp_path, registry_factory):
    make_task, _, _ = registry_factory
    info = tmp_path / "network_info"
    measurements = tmp_path / "measure_records" / "platinum-8272"
    info.mkdir()
    measurements.mkdir(parents=True)
    keys = ['["shared",1]'] + [f'["task-{index}",1]' for index in range(40)]
    for index, network in enumerate(batch_size_one_network_keys()):
        tasks = [make_task(keys[0]), make_task(keys[index + 1]), make_task(keys[0])]
        name = clean_name((network, "llvm")) + ".task.pkl"
        (info / name).write_bytes(pickle.dumps((tasks, [1, 2, 1]), protocol=4))
    for key in keys:
        (measurements / (clean_name((key, "llvm")) + ".json")).write_text("{}\n")
    return info, measurements, keys


def test_sampling_preserves_reference_population_and_sample_order(tutorial_tree):
    info, measurements, keys = tutorial_tree
    selected, metadata = tutorial_file_sample(info, measurements, sample_size=6)
    expected_indices = [24, 26, 2, 16, 32, 31]
    expected_population = [clean_name((key, "llvm")) + ".json" for key in keys]
    assert metadata["population_filenames"] == expected_population
    assert [path.name for path in selected] == [
        expected_population[index] for index in expected_indices
    ]
    assert metadata["selected_filenames"] == [path.name for path in selected]
    assert metadata["selected_workload_keys"] == [
        keys[index] for index in expected_indices
    ]
    assert metadata["population_size"] == 41
    assert metadata["sample_size"] == 6
    assert metadata["seed"] == 0
    assert metadata["source_commit"] == TUTORIAL_COMMIT
    assert metadata["original_tutorial_inventory_verified"] is False
    assert len(metadata["network_task_info"]) == 40
    first = metadata["network_task_info"][0]
    assert first["filename"] == "((resnet_18,[(1,3,224,224)]),llvm).task.pkl"
    assert (
        first["sha256"]
        == hashlib.sha256((info / first["filename"]).read_bytes()).hexdigest()
    )
    assert first["task_count"] == 3
    json.dumps(metadata)
    assert tutorial_file_sample(info, measurements, sample_size=6) == (
        selected,
        metadata,
    )
    assert (
        tutorial_file_sample(info, measurements, sample_size=6, seed=1)[0] != selected
    )


def test_sampling_rejects_insufficient_population_and_missing_sources(tutorial_tree):
    info, measurements, _ = tutorial_tree
    with pytest.raises(ValueError, match="needs 500.*only 41"):
        tutorial_file_sample(info, measurements)
    first = measurements / "([shared,1],llvm).json"
    first.unlink()
    with pytest.raises(FileNotFoundError, match="measurement file is missing"):
        tutorial_file_sample(info, measurements, sample_size=1)
    first.write_text("{}\n")
    (info / "((resnet_18,[(1,3,224,224)]),llvm).task.pkl").unlink()
    with pytest.raises(FileNotFoundError, match="network task info is missing"):
        tutorial_file_sample(info, measurements, sample_size=1)


@pytest.mark.parametrize("sample_size,seed", [(0, 0), (True, 0), (1, True), (1, 0.5)])
def test_sampling_parameters_are_strict(tmp_path, sample_size, seed):
    with pytest.raises(ValueError, match="tutorial"):
        tutorial_file_sample(tmp_path, tmp_path, sample_size=sample_size, seed=seed)


def test_f32_preserves_reference_single_precision_rounding():
    assert f32(1 / 10) == 0.10000000149011612
    assert f32(2**24 + 1) == 2**24
