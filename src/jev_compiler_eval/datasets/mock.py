"""Committed mock dataset used for examples and smoke tests."""

import json
from collections.abc import Mapping
from json import JSONDecodeError
from pathlib import Path
from typing import cast

from jev_compiler_eval.datasets.base import BaseDataset
from jev_compiler_eval.types import (
    Candidate,
    EvaluationCase,
    RuntimeRecord,
    Workload,
)


def _mapping(value: object, location: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"{location} keys must be strings")
    return dict(value)


def _list(value: object, location: str) -> list[object]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{location} must be a non-empty list")
    return value


def _string(value: object, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a non-empty string")
    return value


def _exact_keys(
    value: Mapping[str, object],
    expected: set[str],
    location: str,
) -> None:
    missing = expected - value.keys()
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"{location} is missing required field(s): {names}")
    unknown = value.keys() - expected
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"{location} contains unknown field(s): {names}")


class MockDataset(BaseDataset):
    """Load the small JSON dataset committed with the repository."""

    @property
    def dataset_type(self) -> str:
        return "mock"

    def validate_config(self) -> None:
        if set(self.config) != {"path"}:
            raise ValueError('mock dataset config must contain only "path"')
        path = self.config["path"]
        if not isinstance(path, str) or not path.strip():
            raise ValueError("mock dataset path must be a non-empty string")

    def load(self) -> tuple[EvaluationCase, ...]:
        path = Path(cast(str, self.config["path"]))
        try:
            raw_dataset = json.loads(path.read_text(encoding="utf-8"))
        except JSONDecodeError as error:
            raise ValueError(f'failed to parse mock dataset "{path}"') from error

        root = _mapping(raw_dataset, "mock dataset root")
        _exact_keys(root, {"dataset", "workloads"}, "mock dataset root")
        if root["dataset"] != "mock":
            raise ValueError('mock dataset root field "dataset" must equal "mock"')

        workloads = _list(root["workloads"], "mock dataset workloads")
        return tuple(
            self._parse_workload(raw_workload, index)
            for index, raw_workload in enumerate(workloads)
        )

    def _parse_workload(self, value: object, index: int) -> EvaluationCase:
        location = f"workloads[{index}]"
        data = _mapping(value, location)
        _exact_keys(
            data,
            {"workload_id", "context", "candidates"},
            location,
        )
        workload_id = _string(data["workload_id"], f"{location}.workload_id")
        workload = Workload(
            workload_id=workload_id,
            context=_mapping(data["context"], f"{location}.context"),
        )

        candidates: list[Candidate] = []
        runtimes: list[RuntimeRecord] = []
        raw_candidates = _list(data["candidates"], f"{location}.candidates")
        for candidate_index, raw_candidate in enumerate(raw_candidates):
            candidate_location = f"{location}.candidates[{candidate_index}]"
            candidate_data = _mapping(raw_candidate, candidate_location)
            _exact_keys(
                candidate_data,
                {"candidate_id", "configuration", "runtime_ms"},
                candidate_location,
            )
            candidate_id = _string(
                candidate_data["candidate_id"],
                f"{candidate_location}.candidate_id",
            )
            candidates.append(
                Candidate(
                    candidate_id=candidate_id,
                    configuration=_mapping(
                        candidate_data["configuration"],
                        f"{candidate_location}.configuration",
                    ),
                )
            )
            runtimes.append(
                RuntimeRecord(
                    workload_id=workload_id,
                    candidate_id=candidate_id,
                    runtime_ms=candidate_data["runtime_ms"],  # type: ignore[arg-type]
                )
            )

        return EvaluationCase(
            workload=workload,
            candidates=tuple(candidates),
            runtimes=tuple(runtimes),
        )
