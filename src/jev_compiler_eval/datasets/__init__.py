"""Dataset interfaces and registration."""

from jev_compiler_eval.datasets.base import BaseDataset
from jev_compiler_eval.datasets.mock import MockDataset
from jev_compiler_eval.datasets.registry import get_dataset_class, register_dataset
from jev_compiler_eval.datasets.tenset import TenSetDataset

register_dataset("mock", MockDataset)
register_dataset("tenset", TenSetDataset)

__all__ = [
    "BaseDataset",
    "MockDataset",
    "TenSetDataset",
    "get_dataset_class",
    "register_dataset",
]
