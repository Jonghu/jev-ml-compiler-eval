"""Shared data types for datasets, models, and evaluation."""

from dataclasses import dataclass, field
from math import isfinite
from typing import Mapping


def _require_identifier(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")


@dataclass(frozen=True)
class Workload:
    """A compiler workload presented to a decision model."""

    workload_id: str
    context: Mapping[str, object]

    def __post_init__(self) -> None:
        _require_identifier(self.workload_id, "workload_id")


@dataclass(frozen=True)
class Candidate:
    """A candidate compiler configuration presented to a decision model."""

    candidate_id: str
    configuration: Mapping[str, object]

    def __post_init__(self) -> None:
        _require_identifier(self.candidate_id, "candidate_id")


@dataclass(frozen=True)
class RuntimeRecord:
    """A measured runtime kept separate from model-visible inputs."""

    workload_id: str
    candidate_id: str
    runtime_ms: float

    def __post_init__(self) -> None:
        _require_identifier(self.workload_id, "workload_id")
        _require_identifier(self.candidate_id, "candidate_id")
        if (
            isinstance(self.runtime_ms, bool)
            or not isinstance(self.runtime_ms, (int, float))
            or not isfinite(self.runtime_ms)
            or self.runtime_ms <= 0
        ):
            raise ValueError("runtime_ms must be a finite positive number")


@dataclass(frozen=True)
class EvaluationCase:
    """A workload, its candidates, and hidden measured runtimes."""

    workload: Workload
    candidates: tuple[Candidate, ...]
    runtimes: tuple[RuntimeRecord, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "runtimes", tuple(self.runtimes))

        candidate_ids = [candidate.candidate_id for candidate in self.candidates]
        if not candidate_ids:
            raise ValueError("an evaluation case must contain at least one candidate")
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate IDs must be unique within an evaluation case")

        runtime_candidate_ids: list[str] = []
        for runtime in self.runtimes:
            if runtime.workload_id != self.workload.workload_id:
                raise ValueError(
                    "runtime workload_id must match the evaluation workload"
                )
            runtime_candidate_ids.append(runtime.candidate_id)

        if len(runtime_candidate_ids) != len(set(runtime_candidate_ids)):
            raise ValueError("runtime candidate IDs must be unique")
        if set(runtime_candidate_ids) != set(candidate_ids):
            raise ValueError("runtimes must match the evaluation candidates exactly")


@dataclass(frozen=True)
class DecisionResult:
    """A model-produced candidate ranking and decision metadata."""

    ranked_candidate_ids: tuple[str, ...]
    decision_latency_ms: float
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "ranked_candidate_ids", tuple(self.ranked_candidate_ids)
        )
        if not self.ranked_candidate_ids:
            raise ValueError("a decision result must contain at least one candidate")
        if len(self.ranked_candidate_ids) != len(set(self.ranked_candidate_ids)):
            raise ValueError("ranked candidate IDs must be unique")
        for candidate_id in self.ranked_candidate_ids:
            _require_identifier(candidate_id, "ranked candidate ID")
        if (
            isinstance(self.decision_latency_ms, bool)
            or not isinstance(self.decision_latency_ms, (int, float))
            or not isfinite(self.decision_latency_ms)
            or self.decision_latency_ms < 0
        ):
            raise ValueError("decision_latency_ms must be a finite non-negative number")

    @property
    def selected_candidate_id(self) -> str:
        """Return the highest-ranked candidate ID."""

        return self.ranked_candidate_ids[0]
