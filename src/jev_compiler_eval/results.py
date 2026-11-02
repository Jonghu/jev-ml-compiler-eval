"""Experiment result types and file output."""

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class ModelSummary:
    """Aggregated results for one configured model."""

    model_id: str
    model_type: str
    workloads: int
    metrics: Mapping[str, float]
    mean_decision_latency_ms: float


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_results(
    *,
    output_dir: Path,
    experiment_yaml: str,
    manifest: Mapping[str, object],
    predictions: Sequence[Mapping[str, object]],
    summaries: Sequence[ModelSummary],
    metric_names: Sequence[str],
) -> None:
    """Write a completed experiment result bundle."""

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "experiment.yaml").write_text(experiment_yaml, encoding="utf-8")
    _write_json(output_dir / "manifest.json", manifest)

    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as file:
        for prediction in predictions:
            file.write(json.dumps(prediction) + "\n")

    metrics_payload = {
        "models": [
            {
                "model_id": summary.model_id,
                "model_type": summary.model_type,
                "workloads": summary.workloads,
                "metrics": dict(summary.metrics),
                "mean_decision_latency_ms": summary.mean_decision_latency_ms,
            }
            for summary in summaries
        ]
    }
    _write_json(output_dir / "metrics.json", metrics_payload)

    fieldnames = [
        "model_id",
        "model_type",
        "workloads",
        *metric_names,
        "mean_decision_latency_ms",
    ]
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    "model_id": summary.model_id,
                    "model_type": summary.model_type,
                    "workloads": summary.workloads,
                    **summary.metrics,
                    "mean_decision_latency_ms": summary.mean_decision_latency_ms,
                }
            )
