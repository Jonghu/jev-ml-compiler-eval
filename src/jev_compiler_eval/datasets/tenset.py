"""TVM auto-scheduler measurements and label-free TenSet decision cases.

The JSON reader deliberately does not depend on TVM. Original MeasureInput values
are retained beside labels for the official TVM baseline, never in Jev inputs.
"""

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from math import isfinite
from pathlib import Path
from statistics import mean

from jev_compiler_eval.datasets.base import BaseDataset
from jev_compiler_eval.types import Candidate, EvaluationCase, RuntimeRecord, Workload

SCHEMA_VERSION = 1
TRANSFORMATIONS = {
    "AN": "annotate",
    "FU": "fuse",
    "PR": "pragma",
    "RE": "reorder",
    "SP": "split",
    "FSP": "follow_split",
    "FFSP": "follow_fused_split",
    "SA": "storage_align",
    "CA": "compute_at",
    "CI": "compute_inline",
    "CR": "compute_root",
    "CHR": "cache_read",
    "CHW": "cache_write",
    "RF": "rfactor",
}
_CONTEXT_FIELDS = {
    "format",
    "workload_key",
    "target",
    "operation",
    "tensors",
    "target_host",
    "hardware_params",
    "layout_rewrite_option",
    "task_input_names",
    "compute_dag",
    "loop_structure",
}
_CONFIG_FIELDS = {"initial_state", "transformations"}
_METADATA_FIELDS = {"operation", "tensors", "compute_dag", "loop_structure"}


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: object) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _reject_label_keys(value: object) -> None:
    """Reject accidental measurement metadata nested inside visible structures."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            if (
                not isinstance(key, str)
                or any(
                    word in key.lower()
                    for word in (
                        "runtime",
                        "latency",
                        "measurement",
                        "timestamp",
                        "label",
                    )
                )
                or key.lower()
                in {"r", "cost", "costs", "record_input", "record_version"}
            ):
                raise ValueError("label-bearing field in model-visible TenSet data")
            _reject_label_keys(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_label_keys(child)


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def workload_split(
    workload_key: str,
    *,
    seed: int = 42,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.2,
) -> str:
    """Assign a tensor program to a stable split, independent of all labels.

    This is a project-defined workload split, not an official TenSet split. The
    target is intentionally excluded so a tensor program cannot cross splits.
    """
    for value in (validation_fraction, test_fraction):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("split fractions must be finite numbers")
        if not isfinite(value) or value < 0:
            raise ValueError("split fractions must be finite non-negative numbers")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be < 1")
    value = int(_digest([seed, workload_key]), 16) / (1 << 256)
    if value < test_fraction:
        return "test"
    if value < test_fraction + validation_fraction:
        return "validation"
    return "train"


def _context(task: list) -> dict[str, object]:
    workload_key, target = task[:2]
    try:
        key = json.loads(workload_key)
    except (json.JSONDecodeError, TypeError):
        key = [workload_key]
    first = key[0] if isinstance(key, list) and key else "unknown"
    # Native compute-DAG hashes do not encode an operation name. Do not guess it.
    opaque = (
        isinstance(first, str)
        and len(first) == 32
        and all(character in "0123456789abcdef" for character in first.lower())
    )
    operation = "unknown" if opaque else str(first)
    tensors = []
    if isinstance(key, list):
        for argument in key[1:]:
            if (
                isinstance(argument, list)
                and argument
                and all(
                    isinstance(dim, int) and not isinstance(dim, bool)
                    for dim in argument
                )
            ):
                tensors.append({"shape": argument})
    context: dict[str, object] = {
        "format": "tvm-auto-scheduler-json",
        "workload_key": workload_key,
        "target": target,
        "operation": operation,
        "tensors": tensors,
    }
    # Save task metadata by position, matching TVM's JSON MeasureInput format.
    if len(task) > 2:
        context["hardware_params"] = task[2]
    if len(task) > 3:
        context["target_host"] = task[3]
    if len(task) > 4:
        context["layout_rewrite_option"] = task[4]
    if len(task) > 5:
        context["task_input_names"] = task[5]
    return context


def _load_metadata(path: str | Path | None) -> dict:
    if path is None:
        return {}
    metadata = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("metadata must map workload keys to label-free descriptions")
    for key, description in metadata.items():
        if not isinstance(key, str) or not isinstance(description, dict):
            raise ValueError("metadata must map workload keys to mappings")
        if set(description) - _METADATA_FIELDS:
            raise ValueError("unsupported or label-bearing workload metadata fields")
        _reject_label_keys(description)
        _json(description)
    return metadata


def _parse_record(record: object) -> tuple[list, float, object] | None:
    if not isinstance(record, dict) or not {"i", "r"} <= record.keys():
        raise ValueError("measurement must contain TVM 'i' and 'r' fields")
    measure_input, result = record["i"], record["r"]
    if (
        not isinstance(measure_input, list)
        or len(measure_input) != 2
        or not isinstance(measure_input[0], list)
        or len(measure_input[0]) < 2
        or not isinstance(measure_input[1], list)
        or len(measure_input[1]) != 2
        or not isinstance(result, list)
        or len(result) < 2
    ):
        raise ValueError("invalid TVM MeasureInput or MeasureResult structure")
    task, state = measure_input
    if not all(isinstance(value, str) and value for value in task[:2]):
        raise ValueError("workload_key and target must be non-empty strings")
    if not isinstance(state[1], list) or not all(
        isinstance(step, list) and step and isinstance(step[0], str)
        for step in state[1]
    ):
        raise ValueError("schedule transformation steps must be lists")
    if result[1] != 0:
        return None
    costs = result[0]
    if not isinstance(costs, list) or not costs:
        return None
    if any(
        isinstance(cost, bool)
        or not isinstance(cost, (int, float))
        or not isfinite(cost)
        or not 0 < cost < 1e10
        for cost in costs
    ):
        return None
    return measure_input, mean(costs) * 1000, record.get("v", "v0.6")


def prepare_records(
    paths: Sequence[Path | str],
    *,
    target: str,
    seed: int = 42,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.2,
    min_candidates: int = 2,
    workload_limit: int | None = None,
    metadata_path: Path | str | None = None,
) -> dict[str, object]:
    """Extract valid unique schedules and a reproducible workload split.

    Each repeated record contributes its mean measurement equally to the final
    candidate runtime. Selection and IDs never depend on measurement values.
    Files are streamed line by line; selected measurements remain in memory.
    """
    if not isinstance(target, str) or not target.strip():
        raise ValueError("target must be a non-empty exact TVM target string")
    _integer(seed, "seed")
    _integer(min_candidates, "min_candidates", 1)
    if workload_limit is not None:
        _integer(workload_limit, "workload_limit", 1)
    workload_split(
        "validate",
        seed=seed,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )
    metadata = _load_metadata(metadata_path)
    groups: dict[str, dict] = {}
    sources = []
    counts: Counter = Counter()
    for source in sorted({Path(path).resolve() for path in paths}):
        digest = hashlib.sha256()
        with source.open("rb") as stream:
            for line_number, line in enumerate(stream, 1):
                digest.update(line)
                if not line.strip() or line.lstrip().startswith(b"#"):
                    continue
                counts["records_seen"] += 1
                try:
                    parsed = _parse_record(json.loads(line))
                except (ValueError, TypeError) as error:
                    raise ValueError(f"{source}:{line_number}: {error}") from error
                if parsed is None:
                    counts["invalid_measurements"] += 1
                    continue
                measure_input, runtime_ms, version = parsed
                task, state = measure_input
                if task[1] != target:
                    counts["other_targets"] += 1
                    continue
                workload_id = "tenset-" + _digest([task[0], target])[:24]
                candidate_id = "schedule-" + _digest(state)[:24]
                context = _context(task)
                context.update(metadata.get(task[0], {}))
                group = groups.setdefault(
                    workload_id,
                    {
                        "workload_id": workload_id,
                        "split": workload_split(
                            task[0],
                            seed=seed,
                            validation_fraction=validation_fraction,
                            test_fraction=test_fraction,
                        ),
                        "context": context,
                        "candidates": {},
                    },
                )
                if group["context"] != context:
                    raise ValueError("conflicting task metadata for one workload")
                candidate = group["candidates"].setdefault(
                    candidate_id,
                    {
                        "candidate_id": candidate_id,
                        "configuration": {
                            "initial_state": state[0],
                            "transformations": [
                                {
                                    "code": step[0],
                                    "name": TRANSFORMATIONS.get(step[0], "unknown"),
                                    "arguments": step[1:],
                                }
                                for step in state[1]
                            ],
                        },
                        "record_input": measure_input,
                        "record_version": version,
                        "measurements": [],
                    },
                )
                candidate["measurements"].append(runtime_ms)
                counts["valid_measurements"] += 1
        sources.append({"path": str(source), "sha256": digest.hexdigest()})
    workloads = []
    for workload_id in sorted(groups, key=lambda key: _digest([seed, key])):
        group = groups[workload_id]
        if len(group["candidates"]) < min_candidates:
            counts["workloads_below_min_candidates"] += 1
            continue
        candidates = []
        for candidate_id in sorted(group["candidates"]):
            candidate = group["candidates"][candidate_id]
            candidate["runtime_ms"] = mean(candidate.pop("measurements"))
            candidates.append(candidate)
        group["candidates"] = candidates
        workloads.append(group)
    if workload_limit is not None:
        workloads = workloads[:workload_limit]
    if not workloads:
        raise ValueError("no workloads meet the target and minimum candidate count")
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": "tenset",
        "provenance": {
            "sources": sources,
            "target": target,
            "seed": seed,
            "validation_fraction": validation_fraction,
            "test_fraction": test_fraction,
            "min_candidates": min_candidates,
            "workload_limit": workload_limit,
            "counts": dict(counts),
            "split_counts": dict(Counter(workload["split"] for workload in workloads)),
            "runtime_aggregation": "mean of per-record mean costs; seconds to ms",
            "metadata_sha256": _file_sha256(metadata_path) if metadata_path else None,
        },
        "workloads": workloads,
    }


def load_prepared(path: str | Path) -> dict:
    """Load a prepared TenSet artifact, retaining hidden baseline inputs."""
    try:
        root = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("failed to parse prepared TenSet artifact") from error
    if not isinstance(root, dict) or root.get("dataset") != "tenset":
        raise ValueError("prepared artifact must declare dataset=tenset")
    if root.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported prepared TenSet schema_version")
    if not isinstance(root.get("workloads"), list) or not root["workloads"]:
        raise ValueError("prepared TenSet workloads must be a non-empty list")
    return root


def serialized_input_bytes(workload: Workload, candidates: Sequence[Candidate]) -> int:
    """Measure the actual generic Jev state/question JSON, excluding SDK framing.

    The configurable byte cap is a local experiment policy, not a claim about
    TypeSafe's service limits. No truncation or label-based pruning is performed.
    """
    from jev_compiler_eval.models.jev import (
        DECISION_INSTRUCTIONS,
        DECISION_QUESTION_ID,
    )

    payload = {
        "state": {"workload_id": workload.workload_id, "context": workload.context},
        "questions": {
            DECISION_QUESTION_ID: {
                "type": "choice",
                "instructions": DECISION_INSTRUCTIONS,
                "criteria": {
                    candidate.candidate_id: candidate.configuration
                    for candidate in candidates
                },
            }
        },
    }
    return len(_json(payload).encode("utf-8"))


class TenSetDataset(BaseDataset):
    """Read a prepared artifact and sample identical label-independent cases."""

    @property
    def dataset_type(self) -> str:
        return "tenset"

    def validate_config(self) -> None:
        allowed = {
            "path",
            "split",
            "candidate_limit",
            "seed",
            "max_input_bytes",
            "max_workloads",
            "require_program_metadata",
        }
        if "path" not in self.config or set(self.config) - allowed:
            raise ValueError(
                "TenSet config requires path and supported sampling fields"
            )
        if not isinstance(self.config["path"], str) or not self.config["path"].strip():
            raise ValueError("TenSet path must be a non-empty string")
        if self.config.get("split", "test") not in {
            "train",
            "validation",
            "test",
            "all",
        }:
            raise ValueError("TenSet split must be train, validation, test, or all")
        _integer(self.config.get("seed", 42), "seed")
        if not isinstance(self.config.get("require_program_metadata", False), bool):
            raise ValueError("require_program_metadata must be a boolean")
        for field, default in (
            ("candidate_limit", 32),
            ("max_input_bytes", 262144),
            ("max_workloads", None),
        ):
            value = self.config.get(field, default)
            if value is not None:
                _integer(value, field, 1)

    def load(self) -> tuple[EvaluationCase, ...]:
        root = load_prepared(self.config["path"])
        split = self.config.get("split", "test")
        seed = self.config.get("seed", 42)
        limit = self.config.get("candidate_limit", 32)
        byte_limit = self.config.get("max_input_bytes", 262144)
        max_workloads = self.config.get("max_workloads")
        seen = set()
        selected = []
        for data in root["workloads"]:
            if not isinstance(data, Mapping) or data.get("split") not in {
                "train",
                "validation",
                "test",
            }:
                raise ValueError("invalid prepared TenSet workload or split")
            workload_id = data.get("workload_id")
            if (
                not isinstance(workload_id, str)
                or not workload_id.strip()
                or workload_id in seen
            ):
                raise ValueError("TenSet workload IDs must be unique strings")
            seen.add(workload_id)
            if split == "all" or data["split"] == split:
                selected.append(data)
        selected.sort(key=lambda data: _digest([seed, data["workload_id"]]))
        if max_workloads is not None:
            selected = selected[:max_workloads]
        cases = []
        for data in selected:
            workload_id = data["workload_id"]
            context = data.get("context")
            if not isinstance(context, Mapping) or set(context) - _CONTEXT_FIELDS:
                raise ValueError("unsupported or label-bearing TenSet context fields")
            _reject_label_keys(context)
            if self.config.get("require_program_metadata", False):
                tensors = context.get("tensors")
                if (
                    context.get("operation") in (None, "", "unknown")
                    or not isinstance(tensors, list)
                    or not tensors
                    or any(
                        not isinstance(tensor, Mapping)
                        or "shape" not in tensor
                        or not tensor.get("dtype")
                        for tensor in tensors
                    )
                    or not context.get("compute_dag")
                    or not isinstance(context.get("loop_structure"), list)
                ):
                    raise ValueError(
                        f"{workload_id}: complete program metadata is required; "
                        "export metadata and enrich the prepared artifact first"
                    )
            workload = Workload(workload_id, context)
            entries = data.get("candidates")
            if not isinstance(entries, list) or not entries:
                raise ValueError("TenSet candidates must be a non-empty list")
            candidate_ids = []
            for entry in entries:
                if not isinstance(entry, Mapping):
                    raise ValueError("TenSet candidate entries must be mappings")
                candidate_id = entry.get("candidate_id")
                if not isinstance(candidate_id, str) or not candidate_id.strip():
                    raise ValueError("TenSet candidate IDs must be non-empty strings")
                candidate_ids.append(candidate_id)
            if len(candidate_ids) != len(set(candidate_ids)):
                raise ValueError("TenSet candidate IDs must be unique")
            ordered = sorted(
                entries,
                key=lambda entry: _digest([seed, workload_id, entry["candidate_id"]]),
            )
            if limit is not None:
                ordered = ordered[:limit]
            candidates, runtimes = [], []
            for entry in ordered:
                configuration = entry.get("configuration")
                if not isinstance(configuration, Mapping) or (
                    set(configuration) != _CONFIG_FIELDS
                ):
                    raise ValueError(
                        "unsupported TenSet candidate configuration fields"
                    )
                _reject_label_keys(configuration)
                candidates.append(Candidate(entry["candidate_id"], configuration))
                runtimes.append(
                    RuntimeRecord(
                        workload_id,
                        entry["candidate_id"],
                        entry["runtime_ms"],
                    )
                )
            case = EvaluationCase(workload, tuple(candidates), tuple(runtimes))
            size = serialized_input_bytes(workload, candidates)
            if byte_limit is not None and size > byte_limit:
                raise ValueError(
                    f"{workload_id}: serialized input {size} bytes exceeds local "
                    f"max_input_bytes={byte_limit}; reduce candidate_limit explicitly"
                )
            cases.append(case)
        if not cases:
            raise ValueError(
                f"TenSet prepared artifact has no workloads in split={split}"
            )
        self._manifest = {
            "prepared_file_sha256": _file_sha256(self.config["path"]),
            "provenance": root.get("provenance", {}),
            "split": split,
            "seed": seed,
            "candidate_limit": limit,
            "max_workloads": max_workloads,
            "max_input_bytes": byte_limit,
            "require_program_metadata": self.config.get(
                "require_program_metadata", False
            ),
            "realized_cases": [
                {
                    "workload_id": case.workload.workload_id,
                    "candidate_ids": [
                        candidate.candidate_id for candidate in case.candidates
                    ],
                    "serialized_input_bytes": serialized_input_bytes(
                        case.workload,
                        case.candidates,
                    ),
                }
                for case in cases
            ],
        }
        return tuple(cases)

    def manifest_metadata(self) -> Mapping[str, object]:
        """Return artifact provenance and the realized candidate-set manifest."""
        if not hasattr(self, "_manifest"):
            raise RuntimeError("load the TenSet dataset before requesting its manifest")
        return self._manifest
