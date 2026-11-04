"""Run configured evaluations across datasets and decision models."""

import hashlib
import json
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from statistics import mean
from typing import Mapping

from jev_compiler_eval.config import ExperimentConfig, load_experiment_config
from jev_compiler_eval.datasets import BaseDataset, get_dataset_class
from jev_compiler_eval.metrics import MetricFunction, resolve_metrics
from jev_compiler_eval.models import get_model_class
from jev_compiler_eval.results import ModelSummary, write_results
from jev_compiler_eval.types import DecisionResult, EvaluationCase


@dataclass(frozen=True)
class ExperimentRunResult:
    """Locations and summaries produced by a completed experiment."""

    experiment_name: str
    dataset_type: str
    output_dir: Path
    summaries: tuple[ModelSummary, ...]


def _validate_execution(config: ExperimentConfig, output_dir: Path) -> None:
    if not config.execution.fail_fast:
        raise ValueError("execution.fail_fast=false is not supported")
    if config.candidate_selection.strategy != "all":
        raise ValueError("only candidate_selection.strategy=all is currently supported")
    if config.candidate_selection.config:
        raise ValueError("candidate_selection.strategy=all does not accept settings")
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f'output path "{output_dir}" is not a directory')
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f'output directory "{output_dir}" is not empty')


def _get_dataset(config: ExperimentConfig) -> BaseDataset:
    try:
        dataset_class = get_dataset_class(config.dataset.dataset_type)
    except KeyError as error:
        raise ValueError(
            f'unknown dataset type "{config.dataset.dataset_type}"'
        ) from error
    return dataset_class(config.dataset.config)


def _validate_ranking(case: EvaluationCase, result: DecisionResult) -> None:
    candidate_ids = {candidate.candidate_id for candidate in case.candidates}
    ranked_ids = set(result.ranked_candidate_ids)
    if ranked_ids != candidate_ids or len(result.ranked_candidate_ids) != len(
        case.candidates
    ):
        raise ValueError(
            "model ranking must contain every evaluation candidate exactly once"
        )


def _git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def _package_versions() -> dict[str, str]:
    """Record the installed packages relevant to reproducible evaluations."""

    versions = {}
    for package in ("jev-ml-compiler-eval", "typesafe-sdk", "numpy", "matplotlib"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            continue
    return versions


def _model_input(case: EvaluationCase) -> bytes:
    """Fingerprint model-visible data independently of hidden runtime labels."""

    payload = {
        "workload_id": case.workload.workload_id,
        "context": dict(case.workload.context),
        "candidates": {
            candidate.candidate_id: dict(candidate.configuration)
            for candidate in case.candidates
        },
    }
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _evaluate_model(
    *,
    model_id: str,
    model_type: str,
    model_config: Mapping[str, object],
    cases: tuple[EvaluationCase, ...],
    metrics: Mapping[str, MetricFunction],
) -> tuple[list[dict[str, object]], ModelSummary]:
    try:
        model_class = get_model_class(model_type)
    except KeyError as error:
        raise ValueError(f'unknown model type "{model_type}"') from error
    model = model_class(model_config)

    predictions: list[dict[str, object]] = []
    metric_values = {name: [] for name in metrics}
    latencies: list[float] = []
    try:
        for case in cases:
            result = model.rank(case.workload, case.candidates)
            _validate_ranking(case, result)

            runtimes = {
                runtime.candidate_id: runtime.runtime_ms for runtime in case.runtimes
            }
            optimal_runtime = min(runtimes.values())
            optimal_candidate_ids = [
                candidate_id
                for candidate_id, runtime in runtimes.items()
                if runtime == optimal_runtime
            ]
            case_metrics = {
                name: metric(case, result) for name, metric in metrics.items()
            }
            for name, value in case_metrics.items():
                metric_values[name].append(value)
            latencies.append(result.decision_latency_ms)
            model_input = _model_input(case)

            predictions.append(
                {
                    "model_id": model_id,
                    "model_type": model_type,
                    "workload_id": case.workload.workload_id,
                    "candidate_ids": [
                        candidate.candidate_id for candidate in case.candidates
                    ],
                    "predicted_ranking": list(result.ranked_candidate_ids),
                    "selected_candidate_id": result.selected_candidate_id,
                    "selected_runtime_ms": runtimes[result.selected_candidate_id],
                    "optimal_candidate_ids": optimal_candidate_ids,
                    "optimal_runtime_ms": optimal_runtime,
                    "runtime_by_candidate_ms": runtimes,
                    "model_input_sha256": hashlib.sha256(model_input).hexdigest(),
                    "model_input_bytes": len(model_input),
                    "metrics": case_metrics,
                    "decision_latency_ms": result.decision_latency_ms,
                    "metadata": dict(result.metadata),
                }
            )
    finally:
        model.close()

    summary = ModelSummary(
        model_id=model_id,
        model_type=model_type,
        workloads=len(cases),
        metrics={name: mean(values) for name, values in metric_values.items()},
        mean_decision_latency_ms=mean(latencies),
    )
    return predictions, summary


def run_experiment(path: str | Path) -> ExperimentRunResult:
    """Run one YAML experiment and write its complete result bundle."""

    config_path = Path(path)
    config = load_experiment_config(config_path)
    output_dir = Path(config.experiment.output_dir)
    _validate_execution(config, output_dir)
    metrics = resolve_metrics(config.metrics)
    dataset = _get_dataset(config)
    cases = tuple(dataset.load())
    if not cases:
        raise ValueError("dataset returned no evaluation cases")

    predictions: list[dict[str, object]] = []
    summaries: list[ModelSummary] = []
    for model_spec in config.models:
        model_predictions, summary = _evaluate_model(
            model_id=model_spec.model_id,
            model_type=model_spec.model_type,
            model_config=model_spec.config,
            cases=cases,
            metrics=metrics,
        )
        predictions.extend(model_predictions)
        summaries.append(summary)

    manifest: dict[str, object] = {
        "schema_version": config.schema_version,
        "experiment_name": config.experiment.name,
        "dataset_type": config.dataset.dataset_type,
        "model_ids": [model.model_id for model in config.models],
        "seed": config.experiment.seed,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "git_commit": _git_commit(),
        "package_versions": _package_versions(),
    }
    dataset_metadata = getattr(dataset, "manifest_metadata", None)
    if dataset_metadata is not None:
        manifest["dataset_provenance"] = dict(dataset_metadata())
    write_results(
        output_dir=output_dir,
        experiment_yaml=config_path.read_text(encoding="utf-8"),
        manifest=manifest,
        predictions=predictions,
        summaries=summaries,
        metric_names=config.metrics,
    )
    return ExperimentRunResult(
        experiment_name=config.experiment.name,
        dataset_type=config.dataset.dataset_type,
        output_dir=output_dir,
        summaries=tuple(summaries),
    )
