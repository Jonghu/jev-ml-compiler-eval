"""Dataset interfaces and registration."""

from jev_compiler_eval.datasets.base import BaseDataset
from jev_compiler_eval.datasets.mock import MockDataset
from jev_compiler_eval.datasets.registry import get_dataset_class, register_dataset

register_dataset("mock", MockDataset)

__all__ = [
    "BaseDataset",
    "MockDataset",
    "get_dataset_class",
    "register_dataset",
]
