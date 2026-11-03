"""TpuGraphs tile:xla dataset adapter."""

import re
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import numpy as np

from jev_compiler_eval.datasets.base import BaseDataset
from jev_compiler_eval.datasets.tpugraphs_features import (
    HLO_OPCODE_NAMES,
    NODE_FEATURE_NAMES,
    TILE_CONFIG_FEATURE_NAMES,
)
from jev_compiler_eval.types import (
    Candidate,
    EvaluationCase,
    RuntimeRecord,
    Workload,
)

NANOSECONDS_PER_MILLISECOND = 1_000_000
MAX_CONVOLUTION_SPATIAL_DIMS = 4
EXPECTED_KEYS = {
    "node_feat",
    "node_opcode",
    "edge_index",
    "config_feat",
    "config_runtime",
    "config_runtime_normalizers",
}

_FEATURE_INDEX = {name: index for index, name in enumerate(NODE_FEATURE_NAMES)}
_DTYPE_PREFIX = "shape_element_type_is_"
_DTYPE_COLUMNS = [
    (index, name.removeprefix(_DTYPE_PREFIX))
    for index, name in enumerate(NODE_FEATURE_NAMES)
    if name.startswith(_DTYPE_PREFIX)
]
_CONVOLUTION_PREFIX = "convolution_dim_numbers_"
_DIMENSION_SUFFIX = re.compile(r"^(?P<group>.+)_(?P<dim>\d)$")

# Features serialized explicitly or omitted because they are derivable.
_SPECIAL_FEATURES = {
    "is_root",
    "element_size_in_bits",
    "parameter_number",
    "feature_group_count",
    "batch_group_count",
}
_SPECIAL_GROUPS = {"shape_dimensions", "layout_minor_to_major"}


def _feature_groups() -> tuple[dict[str, list[int]], dict[str, int]]:
    """Split generic node features into per-dimension lists and scalars."""

    lists: dict[str, list[int]] = {}
    scalars: dict[str, int] = {}
    for index, name in enumerate(NODE_FEATURE_NAMES):
        if (
            name in _SPECIAL_FEATURES
            or name.startswith(_DTYPE_PREFIX)
            or name.startswith(_CONVOLUTION_PREFIX)
            or name.endswith(("_sum", "_product"))
        ):
            continue
        match = _DIMENSION_SUFFIX.match(name)
        if match is None:
            scalars[name] = index
            continue
        group = match["group"]
        if group not in _SPECIAL_GROUPS:
            lists.setdefault(group, []).append(index)
    return lists, scalars


_LIST_FEATURES, _SCALAR_FEATURES = _feature_groups()


def _tile_groups() -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for index, name in enumerate(TILE_CONFIG_FEATURE_NAMES):
        match = _DIMENSION_SUFFIX.match(name)
        if match is not None:
            groups.setdefault(match["group"], []).append(index)
    return groups


_TILE_GROUPS = _tile_groups()


def _trim_trailing_zeros(values: list[int]) -> list[int]:
    end = len(values)
    while end and values[end - 1] == 0:
        end -= 1
    return values[:end]


def _columns(row: list[int], indexes: list[int]) -> list[int]:
    return [row[index] for index in indexes]


def _program_name(workload_id: str) -> str:
    """Return the program name without the per-kernel hash suffix."""

    program, separator, _ = workload_id.rpartition("_")
    return program if separator else workload_id


class TpuGraphsTileDataset(BaseDataset):
    """Load TpuGraphs tile:xla kernels as evaluation cases.

    Each kernel becomes one workload. Identical tile configurations are merged
    into one candidate that keeps the fastest normalized runtime, and kernels
    with more distinct configurations than ``max_candidates`` are skipped.
    """

    @property
    def dataset_type(self) -> str:
        return "tpugraphs-tile"

    def validate_config(self) -> None:
        if set(self.config) != {"path", "max_candidates"}:
            raise ValueError(
                "tpugraphs-tile dataset config must contain path and max_candidates"
            )
        path = self.config["path"]
        if not isinstance(path, str) or not path.strip():
            raise ValueError("tpugraphs-tile dataset path must be a non-empty string")
        max_candidates = self.config["max_candidates"]
        if (
            isinstance(max_candidates, bool)
            or not isinstance(max_candidates, int)
            or max_candidates < 1
        ):
            raise ValueError("tpugraphs-tile max_candidates must be a positive integer")

    def load(self) -> tuple[EvaluationCase, ...]:
        directory = Path(cast(str, self.config["path"]))
        if not directory.is_dir():
            raise FileNotFoundError(
                f'tpugraphs-tile dataset directory "{directory}" does not exist'
            )
        paths = sorted(directory.glob("*.npz"))
        if not paths:
            raise ValueError(f'no .npz files found in "{directory}"')

        max_candidates = cast(int, self.config["max_candidates"])
        cases: list[EvaluationCase] = []
        for path in paths:
            case = self._load_case(path)
            if len(case.candidates) <= max_candidates:
                cases.append(case)
        return tuple(cases)

    def _load_case(self, path: Path) -> EvaluationCase:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != EXPECTED_KEYS:
                raise ValueError(f'"{path}" does not contain the tile:xla keys')
            arrays = {key: archive[key] for key in EXPECTED_KEYS}
        self._validate_arrays(path, arrays)

        workload_id = path.stem
        workload = Workload(
            workload_id=workload_id,
            context=self._serialize_graph(workload_id, arrays),
        )

        # Normalizers differ slightly between measurement runs. Rank by the
        # official runtime / normalizer target, rescaled to the first run's
        # default-configuration runtime so that values remain in milliseconds.
        normalizers = arrays["config_runtime_normalizers"]
        normalized_ns = (
            arrays["config_runtime"] / normalizers * float(normalizers[0])
        ).tolist()

        # Merge identical configurations, keyed by their first row index.
        first_index: dict[tuple[int, ...], int] = {}
        fastest_ns: dict[int, float] = {}
        config_rows = arrays["config_feat"].astype(np.int64).tolist()
        for index, (row, runtime_ns) in enumerate(
            zip(config_rows, normalized_ns, strict=True)
        ):
            first = first_index.setdefault(tuple(row), index)
            fastest_ns[first] = min(fastest_ns.get(first, runtime_ns), runtime_ns)

        candidates = tuple(
            Candidate(f"config_{index}", self._serialize_config(config_rows[index]))
            for index in fastest_ns
        )
        runtimes = tuple(
            RuntimeRecord(
                workload_id,
                f"config_{index}",
                runtime_ns / NANOSECONDS_PER_MILLISECOND,
            )
            for index, runtime_ns in fastest_ns.items()
        )
        return EvaluationCase(
            workload=workload,
            candidates=candidates,
            runtimes=runtimes,
        )

    def _validate_arrays(self, path: Path, arrays: Mapping[str, np.ndarray]) -> None:
        node_feat = arrays["node_feat"]
        node_opcode = arrays["node_opcode"]
        edge_index = arrays["edge_index"]
        config_feat = arrays["config_feat"]
        runtime = arrays["config_runtime"]
        normalizers = arrays["config_runtime_normalizers"]

        node_count = node_feat.shape[0] if node_feat.ndim == 2 else -1
        config_count = config_feat.shape[0] if config_feat.ndim == 2 else -1
        if node_feat.shape != (node_count, len(NODE_FEATURE_NAMES)) or node_count < 1:
            raise ValueError(f'"{path}" node_feat must have shape (n, 140)')
        if node_opcode.shape != (node_count,):
            raise ValueError(f'"{path}" node_opcode must have shape (n,)')
        if edge_index.ndim != 2 or edge_index.shape[1] != 2:
            raise ValueError(f'"{path}" edge_index must have shape (m, 2)')
        if edge_index.size and (edge_index.min() < 0 or edge_index.max() >= node_count):
            raise ValueError(f'"{path}" edge_index references unknown nodes')
        if (
            config_feat.shape != (config_count, len(TILE_CONFIG_FEATURE_NAMES))
            or config_count < 1
        ):
            raise ValueError(f'"{path}" config_feat must have shape (c, 24)')
        if runtime.shape != (config_count,) or normalizers.shape != (config_count,):
            raise ValueError(f'"{path}" runtime arrays must have shape (c,)')
        if (runtime <= 0).any() or (normalizers <= 0).any():
            raise ValueError(
                f'"{path}" has non-positive runtimes; the test split has no labels'
            )
        unknown = set(node_opcode.tolist()) - HLO_OPCODE_NAMES.keys()
        if unknown:
            raise ValueError(f'"{path}" contains unknown opcodes: {sorted(unknown)}')
        for name, values in (("node_feat", node_feat), ("config_feat", config_feat)):
            if not np.array_equal(values, np.round(values)):
                raise ValueError(f'"{path}" {name} must contain integer values')

    def _serialize_graph(
        self,
        workload_id: str,
        arrays: Mapping[str, np.ndarray],
    ) -> dict[str, object]:
        operands: dict[int, list[int]] = {}
        for consumer, producer in arrays["edge_index"].tolist():
            operands.setdefault(consumer, []).append(producer)

        node_rows = arrays["node_feat"].astype(np.int64).tolist()
        nodes = [
            self._serialize_node(
                node_id,
                HLO_OPCODE_NAMES[opcode],
                row,
                operands.get(node_id, []),
            )
            for node_id, (opcode, row) in enumerate(
                zip(arrays["node_opcode"].tolist(), node_rows, strict=True)
            )
        ]
        return {
            "collection": "tile:xla",
            "program": _program_name(workload_id),
            "num_nodes": len(nodes),
            "num_edges": int(arrays["edge_index"].shape[0]),
            "nodes": nodes,
        }

    def _serialize_node(
        self,
        node_id: int,
        opcode: str,
        row: list[int],
        operands: list[int],
    ) -> dict[str, object]:
        dtypes = [name for index, name in _DTYPE_COLUMNS if row[index]]
        shape = _trim_trailing_zeros(
            [row[_FEATURE_INDEX[f"shape_dimensions_{dim}"]] for dim in range(6)]
        )
        node: dict[str, object] = {
            "id": node_id,
            "opcode": opcode,
            "dtype": dtypes[0] if len(dtypes) == 1 else dtypes,
            "shape": shape,
        }
        if shape:
            node["layout_minor_to_major"] = [
                row[_FEATURE_INDEX[f"layout_minor_to_major_{dim}"]]
                for dim in range(len(shape))
            ]
        if operands:
            node["operands"] = operands
        if row[_FEATURE_INDEX["is_root"]]:
            node["is_root"] = True
        if opcode == "parameter":
            node["parameter_number"] = row[_FEATURE_INDEX["parameter_number"]]
        if opcode == "convolution":
            node["convolution"] = self._serialize_convolution(row, len(shape))

        for group, indexes in _LIST_FEATURES.items():
            values = _trim_trailing_zeros(_columns(row, indexes))
            if values:
                node[group] = values
        for name, index in _SCALAR_FEATURES.items():
            if row[index]:
                node[name] = row[index]
        return node

    def _serialize_convolution(self, row: list[int], rank: int) -> dict[str, object]:
        """Return convolution dimension numbers, where zero is a valid index."""

        spatial_count = min(max(rank - 2, 0), MAX_CONVOLUTION_SPATIAL_DIMS)

        def value(name: str) -> int:
            return row[_FEATURE_INDEX[_CONVOLUTION_PREFIX + name]]

        def spatial(name: str) -> list[int]:
            return [value(f"{name}_{dim}") for dim in range(spatial_count)]

        return {
            "input_batch_dim": value("input_batch_dim"),
            "input_feature_dim": value("input_feature_dim"),
            "input_spatial_dims": spatial("input_spatial_dims"),
            "kernel_input_feature_dim": value("kernel_input_feature_dim"),
            "kernel_output_feature_dim": value("kernel_output_feature_dim"),
            "kernel_spatial_dims": spatial("kernel_spatial_dims"),
            "output_batch_dim": value("output_batch_dim"),
            "output_feature_dim": value("output_feature_dim"),
            "feature_group_count": row[_FEATURE_INDEX["feature_group_count"]],
            "batch_group_count": row[_FEATURE_INDEX["batch_group_count"]],
        }

    def _serialize_config(self, row: list[int]) -> dict[str, object]:
        configuration: dict[str, object] = {}
        for group, indexes in _TILE_GROUPS.items():
            values = _trim_trailing_zeros(_columns(row, indexes))
            if values:
                configuration[group] = values
        return configuration
