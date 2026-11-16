"""Run and resume a TenSet collection with one workload in memory at a time."""

import csv
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from jev_compiler_eval.config import load_experiment_config
from jev_compiler_eval.datasets.tenset import TenSetDataset
from jev_compiler_eval.datasets.tenset_collection import (
    load_collection_index,
    load_collection_shard,
)
from jev_compiler_eval.evaluation.runner import (
    ExperimentRunResult,
    _git_commit,
    _model_input,
    _package_versions,
    run_experiment,
)
from jev_compiler_eval.metrics import resolve_metrics
from jev_compiler_eval.models import get_model_class
from jev_compiler_eval.results import ModelSummary, write_results
from jev_compiler_eval.types import DecisionResult, EvaluationCase

BUNDLE_FILES = (
    "experiment.yaml",
    "predictions.jsonl",
    "metrics.json",
    "summary.csv",
    "manifest.json",
)
STATE_FILE = "collection-state.json"
RECEIPT_FILE = "collection-completed.json"


@dataclass(frozen=True)
class CollectionPreflight:
    """Offline validation counts; byte counts are not service token guarantees."""

    collection_path: str
    collection_sha256: str
    available_workloads: int
    selected_workloads: int
    total_candidates: int
    min_candidates: int
    max_candidates: int
    max_serialized_input_bytes: int
    output_dir: str
    api_requests_made: int = 0
    model_plans: Mapping[str, object] = field(default_factory=dict)


@dataclass
class _Plan:
    config: Any
    raw: dict
    config_yaml: str
    index_path: Path
    index: dict
    entries: list[dict]
    output_dir: Path
    identity: dict


def _hash_file(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _implementation_fingerprints() -> dict[str, str]:
    """Freeze installed implementation files, including uncommitted source edits."""

    package_root = Path(__file__).resolve().parents[1]
    return {
        path.relative_to(package_root).as_posix(): _hash_file(path)
        for path in sorted(package_root.rglob("*.py"))
    }


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False, prefix=".json-"
    ) as file:
        temporary = Path(file.name)
        try:
            json.dump(value, file, indent=2, allow_nan=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def _load_plan(
    path: str | Path,
    *,
    workload_limit: int | None,
    output_dir: str | Path | None,
) -> _Plan:
    source = Path(path)
    config = load_experiment_config(source)
    if config.dataset.dataset_type != "tenset":
        raise ValueError("collection execution requires dataset.type=tenset")
    split = config.dataset.config.get("split", "test")
    if split not in {"test", "validation"}:
        raise ValueError("collection execution requires a held-out project split")
    if not config.execution.fail_fast:
        raise ValueError("collection execution requires execution.fail_fast=true")
    if (
        config.candidate_selection.strategy != "all"
        or config.candidate_selection.config
    ):
        raise ValueError(
            "collection execution requires candidate_selection.strategy=all"
        )
    if workload_limit is not None and (
        isinstance(workload_limit, bool)
        or not isinstance(workload_limit, int)
        or workload_limit < 1
    ):
        raise ValueError("workload_limit must be a positive integer")
    resolve_metrics(config.metrics)
    for spec in config.models:
        # Validation must not initialize SDK clients, workers, or rate limiters.
        model_class = get_model_class(spec.model_type)
        probe = model_class.__new__(model_class)
        probe.config = dict(spec.config)
        probe.validate_config()
    if any(spec.model_type == "jev-tournament" for spec in config.models) and any(
        name.startswith("pairwise_accuracy") or name == "kendall_tau"
        for name in config.metrics
    ):
        raise ValueError("tournament results do not support global ranking metrics")
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    index_path = Path(config.dataset.config["path"]).resolve()
    index = load_collection_index(index_path)
    if index.get("status") != "complete":
        raise ValueError("collection index must be complete before execution")
    entries = index["shards"]
    if not entries or any(entry["split"] != split for entry in entries):
        raise ValueError("collection shards must match the configured held-out split")
    limit = (
        workload_limit
        if workload_limit is not None
        else config.dataset.config.get("max_workloads")
    )
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("max_workloads must be a positive integer or null")
        if limit > len(entries):
            raise ValueError("requested workload count exceeds the complete collection")
        entries = entries[:limit]
        raw["dataset"]["config"]["max_workloads"] = limit
    destination = Path(output_dir or config.experiment.output_dir).resolve()
    raw["dataset"]["config"]["path"] = str(index_path)
    raw["experiment"]["output_dir"] = str(destination)
    config_yaml = yaml.safe_dump(raw, sort_keys=False)
    identity = {
        "schema_version": 1,
        "collection_index_sha256": _hash_file(index_path),
        "effective_config_sha256": hashlib.sha256(config_yaml.encode()).hexdigest(),
        "selected_workloads": len(entries),
        "git_commit": _git_commit(),
        "implementation_sources": _implementation_fingerprints(),
        "python_version": platform.python_version(),
        "package_versions": _package_versions(),
    }
    return _Plan(
        config, raw, config_yaml, index_path, index, entries, destination, identity
    )


def _child_config(plan: _Plan, position: int, entry: dict) -> tuple[dict, Path, str]:
    value = json.loads(json.dumps(plan.raw))
    shard = (plan.index_path.parent / entry["path"]).resolve()
    value["dataset"]["config"].update({"path": str(shard), "max_workloads": 1})
    value["experiment"].update(
        {
            "name": f"{plan.config.experiment.name}-shard-{position:06d}",
            "output_dir": str(plan.output_dir / "workloads" / f"{position:06d}"),
        }
    )
    return value, shard, yaml.safe_dump(value, sort_keys=False)


def _load_case(plan: _Plan, position: int, entry: dict) -> tuple[EvaluationCase, dict]:
    raw, shard, _ = _child_config(plan, position, entry)
    load_collection_shard(plan.index_path, entry)
    dataset = TenSetDataset(raw["dataset"]["config"])
    dataset.config["max_workloads"] = None
    cases = dataset.load()
    if len(cases) != 1 or cases[0].workload.workload_id != entry["workload_id"]:
        raise ValueError("collection shard must contain exactly its indexed workload")
    limit = plan.config.dataset.config.get("candidate_limit", 32)
    expected = entry["candidate_count"]
    if limit is not None:
        expected = min(limit, expected)
    if len(cases[0].candidates) != expected:
        raise ValueError("collection candidate count disagrees with the shard index")
    return cases[0], dict(dataset.manifest_metadata())


def preflight_tenset_collection(
    config_path: str | Path,
    *,
    workload_limit: int | None = None,
    output_dir: str | Path | None = None,
) -> CollectionPreflight:
    """Verify all selected shards and model settings without clients or API calls."""

    plan = _load_plan(config_path, workload_limit=workload_limit, output_dir=output_dir)
    total = 0
    minimum = None
    maximum = max_bytes = 0
    plans = {}
    for position, entry in enumerate(plan.entries):
        case, metadata = _load_case(plan, position, entry)
        count = len(case.candidates)
        total += count
        minimum = count if minimum is None else min(minimum, count)
        maximum = max(maximum, count)
        max_bytes = max(
            max_bytes, metadata["realized_cases"][0]["serialized_input_bytes"]
        )
        for spec in plan.config.models:
            if spec.model_type != "jev-tournament":
                continue
            model_plan = get_model_class(spec.model_type).plan(
                case.workload, case.candidates, spec.config
            )
            accumulated = plans.setdefault(
                spec.model_id,
                {
                    "first_round_requests": 0,
                    "first_round_questions": 0,
                    "estimated_first_round_input_tokens": 0,
                    "max_estimated_state_question_tokens": 0,
                    "max_estimated_request_tokens": 0,
                    "warnings": model_plan["warnings"],
                },
            )
            for name in (
                "first_round_requests",
                "first_round_questions",
                "estimated_first_round_input_tokens",
            ):
                accumulated[name] += model_plan[name]
            for name in (
                "max_estimated_state_question_tokens",
                "max_estimated_request_tokens",
            ):
                accumulated[name] = max(accumulated[name], model_plan[name])
    return CollectionPreflight(
        str(plan.index_path),
        plan.identity["collection_index_sha256"],
        len(plan.index["shards"]),
        len(plan.entries),
        total,
        minimum or 0,
        maximum,
        max_bytes,
        str(plan.output_dir),
        model_plans=plans,
    )


def _validate_bundle(
    directory: Path,
    *,
    plan: _Plan,
    entry: dict,
    case: EvaluationCase,
    config_yaml: str,
) -> list[dict]:
    if not all((directory / name).is_file() for name in BUNDLE_FILES):
        raise ValueError("completed child bundle is missing canonical result files")
    if (directory / "experiment.yaml").read_text(encoding="utf-8") != config_yaml:
        raise ValueError("saved child configuration does not match this collection run")
    manifest = _json(directory / "manifest.json")
    expected_models = {spec.model_id: spec.model_type for spec in plan.config.models}
    if (
        manifest.get("dataset_type") != "tenset"
        or manifest.get("model_ids") != list(expected_models)
        or manifest.get("seed") != plan.config.experiment.seed
        or manifest.get("dataset_provenance", {}).get("prepared_file_sha256")
        != entry["sha256"]
    ):
        raise ValueError(
            "child manifest does not match the configured models and shard"
        )
    rows = []
    with (directory / "predictions.jsonl").open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
                if len(rows) > len(expected_models):
                    raise ValueError("child contains more than one workload per model")
    if len(rows) != len(expected_models) or {row["model_id"] for row in rows} != set(
        expected_models
    ):
        raise ValueError("child must contain one prediction per configured model")
    ids = [candidate.candidate_id for candidate in case.candidates]
    times = {runtime.candidate_id: runtime.runtime_ms for runtime in case.runtimes}
    model_input = _model_input(case)
    input_hash = hashlib.sha256(model_input).hexdigest()
    functions = resolve_metrics(plan.config.metrics)
    summaries = _json(directory / "metrics.json").get("models", [])
    if len(summaries) != len(expected_models):
        raise ValueError("child summaries do not match the configured models")
    summary_by_model = {summary["model_id"]: summary for summary in summaries}
    if set(summary_by_model) != set(expected_models):
        raise ValueError("child summary model IDs do not match configured models")
    for row in rows:
        if (
            row["workload_id"] != entry["workload_id"]
            or row["model_type"] != expected_models[row["model_id"]]
            or row["candidate_ids"] != ids
            or row.get("runtime_by_candidate_ms") != times
            or row.get("model_input_sha256") != input_hash
            or row.get("model_input_bytes") != len(model_input)
        ):
            raise ValueError("child predictions disagree with the indexed workload")
        decision = DecisionResult(
            tuple(row["predicted_ranking"]),
            row["decision_latency_ms"],
            row.get("metadata", {}),
        )
        if set(decision.ranked_candidate_ids) != set(ids):
            raise ValueError("child ranking must contain every candidate exactly once")
        optimum = min(times.values())
        if (
            row["selected_candidate_id"] != decision.selected_candidate_id
            or row["selected_runtime_ms"] != times[decision.selected_candidate_id]
            or row["optimal_runtime_ms"] != optimum
            or set(row["optimal_candidate_ids"])
            != {candidate for candidate, runtime in times.items() if runtime == optimum}
        ):
            raise ValueError("child selected and optimal runtimes disagree with labels")
        expected_metrics = {
            name: function(case, decision) for name, function in functions.items()
        }
        if set(row["metrics"]) != set(expected_metrics) or any(
            not math.isclose(row["metrics"][name], value, rel_tol=1e-12, abs_tol=1e-12)
            for name, value in expected_metrics.items()
        ):
            raise ValueError("child metrics disagree with the saved prediction")
        summary = summary_by_model[row["model_id"]]
        if (
            summary["workloads"] != 1
            or summary["metrics"] != row["metrics"]
            or summary["model_type"] != row["model_type"]
            or summary["mean_decision_latency_ms"] != row["decision_latency_ms"]
        ):
            raise ValueError("child summary disagrees with the per-workload prediction")
    with (directory / "summary.csv").open(encoding="utf-8", newline="") as file:
        csv_rows = list(csv.DictReader(file))
    if len(csv_rows) != len(expected_models) or {
        row["model_id"] for row in csv_rows
    } != set(expected_models):
        raise ValueError("child CSV summary is incomplete")
    for row in csv_rows:
        summary = summary_by_model[row["model_id"]]
        if (
            row["model_type"] != summary["model_type"]
            or int(row["workloads"]) != 1
            or float(row["mean_decision_latency_ms"])
            != summary["mean_decision_latency_ms"]
            or any(float(row[name]) != summary["metrics"][name] for name in functions)
        ):
            raise ValueError("child CSV summary disagrees with per-workload metrics")
    return rows


def _receipt_identity(plan: _Plan, entry: dict, config_yaml: str) -> dict:
    return {
        **plan.identity,
        "shard_sha256": entry["sha256"],
        "workload_id": entry["workload_id"],
        "child_config_sha256": hashlib.sha256(config_yaml.encode()).hexdigest(),
    }


def _completed_child(
    directory: Path, plan: _Plan, entry: dict, case: EvaluationCase, config_yaml: str
) -> bool:
    receipt_path = directory / RECEIPT_FILE
    if receipt_path.exists():
        receipt = _json(receipt_path)
        if receipt.get("identity") != _receipt_identity(plan, entry, config_yaml):
            raise ValueError("completed child belongs to a different collection run")
        if receipt.get("files") != {
            name: _hash_file(directory / name) for name in BUNDLE_FILES
        }:
            raise ValueError("completed child result checksum changed")
        _validate_bundle(
            directory, plan=plan, entry=entry, case=case, config_yaml=config_yaml
        )
        return True
    if not directory.exists() or not any(directory.iterdir()):
        return False
    if all((directory / name).is_file() for name in BUNDLE_FILES):
        # Recover an interruption between canonical output and its completion receipt.
        _validate_bundle(
            directory, plan=plan, entry=entry, case=case, config_yaml=config_yaml
        )
        _write_receipt(directory, plan, entry, config_yaml)
        return True
    # Preserve interrupted output rather than overwriting or silently accepting it.
    incomplete = plan.output_dir / "interrupted"
    incomplete.mkdir(exist_ok=True)
    directory.rename(incomplete / f"{directory.name}-{uuid4().hex}")
    return False


def _write_receipt(directory: Path, plan: _Plan, entry: dict, config_yaml: str) -> None:
    _atomic_json(
        directory / RECEIPT_FILE,
        {
            "identity": _receipt_identity(plan, entry, config_yaml),
            "files": {name: _hash_file(directory / name) for name in BUNDLE_FILES},
        },
    )


def _aggregate(plan: _Plan) -> tuple[ModelSummary, ...]:
    totals = {
        spec.model_id: {name: 0.0 for name in plan.config.metrics}
        for spec in plan.config.models
    }
    latencies = {spec.model_id: 0.0 for spec in plan.config.models}
    realized = []
    children = []
    for position, entry in enumerate(plan.entries):
        directory = plan.output_dir / "workloads" / f"{position:06d}"
        child_manifest = _json(directory / "manifest.json")
        # Keep only paths here; the potentially large candidate-ID lists stream later.
        realized.append(directory / "manifest.json")
        children.append(
            {
                "workload_id": entry["workload_id"],
                "shard_sha256": entry["sha256"],
                "result_path": str(directory.relative_to(plan.output_dir)),
                "completion_sha256": _hash_file(directory / RECEIPT_FILE),
                "git_commit": child_manifest.get("git_commit"),
            }
        )
        with (directory / "predictions.jsonl").open(encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                prediction = json.loads(line)
                for name in plan.config.metrics:
                    totals[prediction["model_id"]][name] += prediction["metrics"][name]
                latencies[prediction["model_id"]] += prediction["decision_latency_ms"]
    count = len(plan.entries)
    summaries = tuple(
        ModelSummary(
            spec.model_id,
            spec.model_type,
            count,
            {name: value / count for name, value in totals[spec.model_id].items()},
            latencies[spec.model_id] / count,
        )
        for spec in plan.config.models
    )
    dataset_provenance = {
        **plan.raw["dataset"]["config"],
        "prepared_file_sha256": plan.identity["collection_index_sha256"],
        "collection_index_sha256": plan.identity["collection_index_sha256"],
        "provenance": plan.index.get("provenance", {}),
        "available_workloads": len(plan.index["shards"]),
        "selected_workloads": count,
        "complete_collection_evaluated": count == len(plan.index["shards"]),
    }
    manifest = {
        "schema_version": plan.config.schema_version,
        "experiment_name": plan.config.experiment.name,
        "dataset_type": "tenset",
        "model_ids": [spec.model_id for spec in plan.config.models],
        "seed": plan.config.experiment.seed,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "git_commit": _git_commit(),
        "package_versions": _package_versions(),
        "collection_execution": {
            "status": "complete",
            "execution_identity": plan.identity,
            "aggregation": (
                "unweighted workload macro mean, independent of candidate count"
            ),
            "child_bundles": children,
        },
        "dataset_provenance": dataset_provenance,
    }
    with tempfile.TemporaryDirectory(prefix=".aggregate-", dir=plan.output_dir) as name:
        stage = Path(name)
        write_results(
            output_dir=stage,
            experiment_yaml=plan.config_yaml,
            manifest={},
            predictions=(),
            summaries=summaries,
            metric_names=plan.config.metrics,
        )
        with (stage / "predictions.jsonl").open("wb") as target:
            for position in range(count):
                child = plan.output_dir / "workloads" / f"{position:06d}"
                with (child / "predictions.jsonl").open("rb") as source:
                    shutil.copyfileobj(source, target)
        # Stream realized cases instead of holding all candidate IDs in RAM.
        with (stage / "manifest.json").open("w", encoding="utf-8") as target:
            target.write("{")
            for index, (key, value) in enumerate(manifest.items()):
                if index:
                    target.write(",")
                target.write(json.dumps(key) + ":")
                if key != "dataset_provenance":
                    json.dump(value, target, allow_nan=False)
                    continue
                target.write(json.dumps(value, allow_nan=False)[:-1])
                target.write(',"realized_cases":[')
                for position, path in enumerate(realized):
                    if position:
                        target.write(",")
                    cases = _json(path)["dataset_provenance"]["realized_cases"]
                    if len(cases) != 1:
                        raise ValueError(
                            "child provenance must describe one realized case"
                        )
                    json.dump(cases[0], target, allow_nan=False)
                target.write("]}")
            target.write("}\n")
        for name in BUNDLE_FILES:
            os.replace(stage / name, plan.output_dir / name)
    return summaries


def run_tenset_collection(
    config_path: str | Path,
    *,
    workload_limit: int | None = None,
    resume: bool = False,
    output_dir: str | Path | None = None,
    progress: Callable[[int, int, str, bool], None] | None = None,
) -> ExperimentRunResult:
    """Run every selected shard; resume verified completed cases without model calls.

    A failed case stops the run and no aggregate bundle is declared complete.
    Restart with identical inputs and ``resume=True``. Completed child bundles
    are verified and reused; unfinished model requests may reuse the API cache.
    """

    plan = _load_plan(config_path, workload_limit=workload_limit, output_dir=output_dir)
    destination = plan.output_dir
    state_path = destination / STATE_FILE
    if destination.exists() and (
        not destination.is_dir() or any(destination.iterdir())
    ):
        if not resume or not state_path.is_file():
            raise FileExistsError(
                "collection output is not empty; use --resume for this run"
            )
        state = _json(state_path)
        if state.get("identity") != plan.identity:
            raise ValueError(
                "resume requires unchanged configuration, collection, and code"
            )
        if state.get("status") == "complete":
            if state.get("files") != {
                name: _hash_file(destination / name) for name in BUNDLE_FILES
            }:
                raise ValueError("completed aggregate checksum changed")
    else:
        destination.mkdir(parents=True, exist_ok=True)
        state = {
            "identity": plan.identity,
            "status": "running",
            "completed_workloads": 0,
        }
        _atomic_json(state_path, state)
    (destination / "configs").mkdir(exist_ok=True)
    (destination / "workloads").mkdir(exist_ok=True)
    model_pool = {}
    try:
        for position, entry in enumerate(plan.entries):
            case, _ = _load_case(plan, position, entry)
            raw, _, child_yaml = _child_config(plan, position, entry)
            child_dir = Path(raw["experiment"]["output_dir"])
            reused = _completed_child(child_dir, plan, entry, case, child_yaml)
            if not reused:
                child_config = destination / "configs" / f"{position:06d}.yaml"
                if child_config.exists():
                    if child_config.read_text(encoding="utf-8") != child_yaml:
                        raise ValueError(
                            "child configuration changed; refusing to overwrite"
                        )
                else:
                    child_config.write_text(child_yaml, encoding="utf-8")
                if not model_pool:
                    for spec in plan.config.models:
                        model_pool[spec.model_id] = get_model_class(spec.model_type)(
                            spec.config
                        )
                run_experiment(child_config, model_pool=model_pool)
                _validate_bundle(
                    child_dir, plan=plan, entry=entry, case=case, config_yaml=child_yaml
                )
                _write_receipt(child_dir, plan, entry, child_yaml)
            if progress is not None:
                progress(position + 1, len(plan.entries), entry["workload_id"], reused)
            if state.get("status") != "complete":
                state.update({"status": "running", "completed_workloads": position + 1})
                state.pop("failure", None)
                _atomic_json(state_path, state)
        if state.get("status") == "complete":
            summaries = tuple(
                ModelSummary(
                    row["model_id"],
                    row["model_type"],
                    row["workloads"],
                    row["metrics"],
                    row["mean_decision_latency_ms"],
                )
                for row in _json(destination / "metrics.json")["models"]
            )
        else:
            summaries = _aggregate(plan)
            state.update(
                {
                    "status": "complete",
                    "files": {
                        name: _hash_file(destination / name) for name in BUNDLE_FILES
                    },
                }
            )
            _atomic_json(state_path, state)
    except BaseException as error:
        if state.get("status") != "complete":
            state.update(
                {
                    "status": "interrupted",
                    "failure": {"error_type": type(error).__name__},
                }
            )
            _atomic_json(state_path, state)
        raise
    finally:
        active_error = sys.exception()
        close_error = None
        for model in model_pool.values():
            try:
                model.close()
            except BaseException as error:
                close_error = close_error or error
        if close_error is not None and active_error is None:
            raise close_error
    return ExperimentRunResult(
        plan.config.experiment.name, "tenset", destination, summaries
    )


def preflight_as_dict(result: CollectionPreflight) -> dict:
    """Return JSON-ready preflight counts for command-line output."""

    return asdict(result)
