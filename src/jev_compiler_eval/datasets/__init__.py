"""Dataset interfaces and registration."""

from jev_compiler_eval.datasets.base import BaseDataset
from jev_compiler_eval.datasets.mock import MockDataset
from jev_compiler_eval.datasets.registry import get_dataset_class, register_dataset
from jev_compiler_eval.datasets.tenset import TenSetDataset
from jev_compiler_eval.datasets.tpugraphs import TpuGraphsTileDataset

register_dataset("mock", MockDataset)
register_dataset("tenset", TenSetDataset)
register_dataset("tpugraphs-tile", TpuGraphsTileDataset)

__all__ = [
    "BaseDataset",
    "MockDataset",
    "TenSetDataset",
    "TpuGraphsTileDataset",
    "get_dataset_class",
    "register_dataset",
]
