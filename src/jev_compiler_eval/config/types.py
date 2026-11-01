"""Typed experiment configuration values."""

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class ExperimentMetadata:
    """Metadata and shared settings for one experiment."""

    name: str
    seed: int
    output_dir: str
    description: str | None = None


@dataclass(frozen=True)
class DatasetSpec:
    """Dataset implementation name and implementation-specific settings."""

    dataset_type: str
    config: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateSelectionSpec:
    """Candidate selection strategy and strategy-specific settings."""

    strategy: str
    config: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelSpec:
    """A named model variant and its implementation-specific settings."""

    model_id: str
    model_type: str
    config: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionSpec:
    """Settings controlling experiment execution behavior."""

    fail_fast: bool


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete, structurally validated experiment configuration."""

    schema_version: int
    experiment: ExperimentMetadata
    dataset: DatasetSpec
    candidate_selection: CandidateSelectionSpec
    models: tuple[ModelSpec, ...]
    metrics: tuple[str, ...]
    execution: ExecutionSpec
