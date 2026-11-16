"""Reconstruct the pinned TenSet tutorial's ordered 500-file sampling protocol.

This module selects workload files without importing TVM or executing pickle
reducers. It does not claim identity with the tutorial's unpublished original
file inventory or reconstruct native TVM features and XGBoost predictions.
"""

from __future__ import annotations

import hashlib
import platform
import struct
from pathlib import Path
from random import Random

from jev_compiler_eval.datasets.tenset_registry import load_tenset_task_info_keys

TUTORIAL_COMMIT = "35774ed373f9693cc6efcbe83f5703141e8ab0c8"
_SOURCE_ROOT = f"https://github.com/tlc-pack/tenset/blob/{TUTORIAL_COMMIT}"
TUTORIAL_SOURCE_URLS = {
    "selection": f"{_SOURCE_ROOT}/scripts/make_dataset.py",
    "filenames": f"{_SOURCE_ROOT}/scripts/common.py",
    "record_split": f"{_SOURCE_ROOT}/python/tvm/auto_scheduler/dataset.py",
    "label_calculation": f"{_SOURCE_ROOT}/src/auto_scheduler/feature.cc",
    "published_reference": (
        f"{_SOURCE_ROOT}/docs/try_pretrained_models_on_aws_c59x.md"
    ),
}

NetworkKey = tuple[str, list[tuple[int, ...]]]


def f32(value: float) -> float:
    """Round a scalar through IEEE-754 binary32, as the reference labels do."""

    return struct.unpack("<f", struct.pack("<f", value))[0]


def clean_name(value: object) -> str:
    """Match ``scripts/common.py::clean_name`` exactly."""

    return str(value).replace(" ", "").replace('"', "").replace("'", "")


def batch_size_one_network_keys() -> list[NetworkKey]:
    """Return the 40 reference network keys in their original iteration order."""

    networks: list[NetworkKey] = []
    for image_size in (224, 240, 256):
        for layer in (18, 50):
            networks.append((f"resnet_{layer}", [(1, 3, image_size, image_size)]))
    for image_size in (224, 240, 256):
        for name in ("mobilenet_v2", "mobilenet_v3"):
            networks.append((name, [(1, 3, image_size, image_size)]))
    for name in ("wide_resnet_50", "resnext_50"):
        for image_size in (224, 240, 256):
            networks.append((name, [(1, 3, image_size, image_size)]))
    networks.append(("inception_v3", [(1, 3, 299, 299)]))
    for image_size in (224, 240, 256):
        networks.append(("densenet_121", [(1, 3, image_size, image_size)]))
    for image_size in (112, 128, 144):
        networks.append(("resnet3d_18", [(1, 3, image_size, image_size, 16)]))
    for sequence_length in (64, 128, 256):
        for scale in ("tiny", "base", "medium", "large"):
            networks.append((f"bert_{scale}", [(1, sequence_length)]))
    for image_size in (64, 80, 96):
        networks.append(("dcgan", [(1, 3, image_size, image_size)]))
    return networks


def _file_digest(path: Path) -> str:
    with path.open("rb") as artifact:
        return hashlib.file_digest(artifact, "sha256").hexdigest()


def _measurement_filename(workload_key: str) -> str:
    filename = clean_name((workload_key, "llvm")) + ".json"
    if any(character in filename for character in ("/", "\\", "\x00")):
        raise ValueError("unsupported workload key in tutorial measurement filename")
    return filename


def tutorial_file_sample(
    network_info_dir: Path,
    measurement_dir: Path,
    *,
    sample_size: int = 500,
    seed: int = 0,
) -> tuple[list[Path], dict[str, object]]:
    """Select files using network/task order, first occurrences and seed 0.

    The input directories are the extracted ``network_info`` directory and the
    ``measure_records/platinum-8272`` directory. Every expected network pickle
    and every population measurement file must exist. Source hashes and ordered
    inventories are recorded, so subsequent preparation can freeze the concrete
    selection. Missing files are never silently removed from the population.
    """

    if type(sample_size) is not int or sample_size <= 0:
        raise ValueError("tutorial sample_size must be a positive integer")
    if type(seed) is not int:
        raise ValueError("tutorial seed must be an integer")
    network_info_dir, measurement_dir = Path(network_info_dir), Path(measurement_dir)
    for directory in (network_info_dir, measurement_dir):
        if not directory.is_dir():
            raise FileNotFoundError(
                f"tutorial dataset directory does not exist: {directory}"
            )

    population: list[Path] = []
    workload_keys_by_filename: dict[str, str] = {}
    seen: set[str] = set()
    network_files = []
    for network in batch_size_one_network_keys():
        filename = clean_name((network, "llvm")) + ".task.pkl"
        path = network_info_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"tutorial network task info is missing: {path}")
        digest = _file_digest(path)
        keys = load_tenset_task_info_keys(path)
        if _file_digest(path) != digest:
            raise ValueError(
                f"tutorial network task info changed during loading: {path}"
            )
        network_files.append(
            {"filename": filename, "sha256": digest, "task_count": len(keys)}
        )
        for workload_key in keys:
            if workload_key in seen:
                continue
            seen.add(workload_key)
            path = measurement_dir / _measurement_filename(workload_key)
            if not path.is_file():
                raise FileNotFoundError(f"tutorial measurement file is missing: {path}")
            if path.name in workload_keys_by_filename:
                raise ValueError(
                    "different tutorial workload keys map to one measurement filename"
                )
            population.append(path)
            workload_keys_by_filename[path.name] = workload_key
    if len(population) < sample_size:
        raise ValueError(
            f"tutorial needs {sample_size} sampled files but the preset population "
            f"contains only {len(population)}"
        )
    selected = Random(seed).sample(population, sample_size)
    return selected, {
        "source_commit": TUTORIAL_COMMIT,
        "source_urls": dict(TUTORIAL_SOURCE_URLS),
        "preset": "batch-size-1",
        "seed": seed,
        "sample_size": sample_size,
        "population_size": len(population),
        "population_filenames": [path.name for path in population],
        "selected_filenames": [path.name for path in selected],
        "selected_workload_keys": [
            workload_keys_by_filename[path.name] for path in selected
        ],
        "network_task_info": network_files,
        "sampling_algorithm": (
            "Python random.Random(seed).sample over first-occurrence workload "
            "keys from the ordered batch-size-1 network task lists"
        ),
        "sampling_python_version": platform.python_version(),
        "original_tutorial_inventory_verified": False,
    }
