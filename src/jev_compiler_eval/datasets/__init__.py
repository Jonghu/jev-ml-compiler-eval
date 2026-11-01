"""Dataset interfaces and registration."""

from jev_compiler_eval.datasets.base import BaseDataset
from jev_compiler_eval.datasets.registry import get_dataset_class, register_dataset

__all__ = ["BaseDataset", "get_dataset_class", "register_dataset"]
