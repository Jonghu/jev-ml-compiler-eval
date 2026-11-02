"""Base interface for decision models."""

from abc import ABC, abstractmethod
from typing import Mapping, Sequence

from jev_compiler_eval.types import Candidate, DecisionResult, Workload


class BaseDecisionModel(ABC):
    """Common interface implemented by every decision model."""

    def __init__(self, config: Mapping[str, object]) -> None:
        self.config = dict(config)
        self.validate_config()

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Return the model type used by experiment configuration."""

    @abstractmethod
    def validate_config(self) -> None:
        """Validate model-specific configuration."""

    @abstractmethod
    def rank(
        self,
        workload: Workload,
        candidates: Sequence[Candidate],
    ) -> DecisionResult:
        """Rank candidates for one workload."""

    def close(self) -> None:
        """Release resources held by the model."""
