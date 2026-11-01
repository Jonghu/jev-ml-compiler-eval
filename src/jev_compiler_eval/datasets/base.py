"""Base interface for evaluation datasets."""

from abc import ABC, abstractmethod
from typing import Mapping, Sequence

from jev_compiler_eval.types import EvaluationCase


class BaseDataset(ABC):
    """Common interface implemented by every evaluation dataset."""

    def __init__(self, config: Mapping[str, object]) -> None:
        self.config = dict(config)
        self.validate_config()

    @property
    @abstractmethod
    def dataset_type(self) -> str:
        """Return the dataset type used by experiment configuration."""

    @abstractmethod
    def validate_config(self) -> None:
        """Validate dataset-specific configuration."""

    @abstractmethod
    def load(self) -> Sequence[EvaluationCase]:
        """Load evaluation cases from the configured dataset."""
