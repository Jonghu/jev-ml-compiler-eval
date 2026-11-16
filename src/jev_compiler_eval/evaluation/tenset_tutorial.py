"""Evaluate record-level tutorial test pools without claiming a matched XGBoost run.

The ordinary collection runner retains its runtime-based macro summaries. This
wrapper additionally recomputes Top-1/Top-5 from the private official throughput
labels and reports the tutorial's test-record-count weighted aggregation. Original
published test identities and XGBoost artifacts remain unavailable; the published
aggregate is therefore recorded separately as an external reference.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from statistics import mean

from jev_compiler_eval.datasets.tenset_collection import load_collection_shard
from jev_compiler_eval.evaluation import tenset_collection as collection
from jev_compiler_eval.evaluation.runner import ExperimentRunResult

EXPERIMENT_PROTOCOL = "tenset_500file_tutorial_record_test_reconstruction"
ALIGNMENT_STATUS = "unverified_original_tutorial_artifacts"
SUMMARY_JSON = "official-weighted-summary.json"
SUMMARY_CSV = "official-weighted-summary.csv"
REFERENCE_JSON = "external-xgboost-reference.json"
REFERENCE_SOURCE = (
    "https://github.com/tlc-pack/tenset/blob/"
    "35774ed373f9693cc6efcbe83f5703141e8ab0c8/"
    "docs/try_pretrained_models_on_aws_c59x.md#L78"
)


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _plan(config_path, *, workload_limit=None, output_dir=None):
    """Use the shared offline configuration validator, then check the protocol."""

    plan = collection._load_plan(
        config_path, workload_limit=workload_limit, output_dir=output_dir
    )
    provenance = plan.index["provenance"]
    if provenance.get("experiment_protocol") != EXPERIMENT_PROTOCOL:
        raise ValueError("the collection is not a tutorial record-test reconstruction")
    if provenance.get("reference_alignment_status") != ALIGNMENT_STATUS:
        raise ValueError("unsupported or missing tutorial reference-alignment status")
    if plan.config.dataset.config.get("candidate_limit", 32) is not None:
        raise ValueError("tutorial evaluation requires candidate_limit: null")
    if plan.config.dataset.config.get("split", "test") != "test":
        raise ValueError("tutorial evaluation requires its declared test records")
    if not {"top_1_score", "top_5_score"}.issubset(plan.config.metrics):
        raise ValueError("tutorial evaluation requires Top-1 and Top-5 Score")
    if set(plan.config.metrics) - {
        "top_1_score",
        "top_5_score",
        "runtime_regret",
    }:
        raise ValueError("tutorial wrapper supports selection metrics only")
    return plan


def _pool(plan, entry: dict) -> tuple[dict, dict[str, float], dict[str, float]]:
    """Validate private labels and raw record identities without exposing them."""

    document = load_collection_shard(plan.index_path, entry)
    workload = document["workloads"][0]
    labels, runtimes, record_indices = {}, {}, set()
    products = []
    for candidate in workload["candidates"]:
        identity = candidate["candidate_id"]
        if identity in labels:
            raise ValueError("tutorial record candidate IDs must be unique")
        index = candidate.get("source_record_index")
        if type(index) is not int or index < 0 or index in record_indices:
            raise ValueError("tutorial source record indices must be unique integers")
        label = candidate.get("official_throughput")
        runtime = candidate.get("runtime_ms")
        if (
            isinstance(label, bool)
            or not isinstance(label, (float, int))
            or not math.isfinite(label)
            or not 0 < label <= 1
        ):
            raise ValueError("tutorial official throughput labels must be in (0, 1]")
        if (
            isinstance(runtime, bool)
            or not isinstance(runtime, (float, int))
            or not math.isfinite(runtime)
            or runtime <= 0
        ):
            raise ValueError("tutorial record runtimes must be finite and positive")
        record_indices.add(index)
        labels[identity] = float(label)
        runtimes[identity] = float(runtime)
        products.append(float(label) * float(runtime))
    if len(labels) != entry["candidate_count"] or len(labels) < 2:
        raise ValueError("tutorial test pool count differs from its collection index")
    # The official labels are float32 ratios using a full-task oracle. Products
    # may differ by float32 roundoff but must describe one common normalizer.
    if any(not math.isclose(p, products[0], rel_tol=2e-6) for p in products):
        raise ValueError("tutorial throughput labels disagree with record runtimes")
    return workload, labels, runtimes


def preflight_tenset_tutorial(
    config_path: str | Path,
    *,
    workload_limit: int | None = None,
    output_dir: str | Path | None = None,
) -> dict:
    """Validate the private test pools and question packing without API clients."""

    plan = _plan(config_path, workload_limit=workload_limit, output_dir=output_dir)
    total = 0
    for entry in plan.entries:
        _, labels, _ = _pool(plan, entry)
        total += len(labels)
    result = collection.preflight_as_dict(
        collection.preflight_tenset_collection(
            config_path, workload_limit=workload_limit, output_dir=output_dir
        )
    )
    return {
        **result,
        "experiment_protocol": EXPERIMENT_PROTOCOL,
        "reference_alignment_status": ALIGNMENT_STATUS,
        "test_record_count": total,
        "paired_xgboost_comparison": False,
        "published_xgboost_reference_source": REFERENCE_SOURCE,
        "live_execution_requires_allow_external_reference": True,
    }


def _peak_score(labels: Mapping[str, float], ranking: list[str], k: int) -> float:
    """Recompute the official prefix-best formula using exported label values."""

    oracle = max(labels.values())
    best = 0.0
    scores = []
    for identity in ranking[:k]:
        best = max(best, labels[identity])
        scores.append(best / oracle)
    return mean(scores)


def _atomic_text(path: Path, writer: Callable) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, value: dict) -> None:
    def write(stream):
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")

    _atomic_text(path, write)


def summarize_tenset_tutorial(
    config_path: str | Path,
    *,
    workload_limit: int | None = None,
    output_dir: str | Path | None = None,
) -> dict:
    """Publish separate record-weighted and macro selection-score summaries."""

    plan = _plan(config_path, workload_limit=workload_limit, output_dir=output_dir)
    state = _read_json(plan.output_dir / collection.STATE_FILE)
    if state.get("identity") != plan.identity or state.get("status") != "complete":
        raise ValueError("tutorial summary requires its unchanged, completed run")
    expected_files = state.get("files")
    if expected_files != {
        name: _sha256(plan.output_dir / name) for name in collection.BUNDLE_FILES
    }:
        raise ValueError("completed tutorial result checksums differ")
    expected_models = {spec.model_id: spec.model_type for spec in plan.config.models}
    entries = {entry["workload_id"]: entry for entry in plan.entries}
    values = {model: [] for model in expected_models}
    seen = set()
    source = plan.output_dir / "predictions.jsonl"
    with source.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("tutorial predictions must be JSON objects")
            model, workload_id = row.get("model_id"), row.get("workload_id")
            pair = (model, workload_id)
            if (
                model not in expected_models
                or row.get("model_type") != expected_models[model]
                or workload_id not in entries
                or pair in seen
            ):
                raise ValueError("unexpected or duplicate tutorial prediction")
            seen.add(pair)
            _, labels, runtimes = _pool(plan, entries[workload_id])
            candidates, ranking = row.get("candidate_ids"), row.get("predicted_ranking")
            if any(
                not isinstance(order, list)
                or len(order) != len(labels)
                or len(set(order)) != len(order)
                or set(order) != set(labels)
                for order in (candidates, ranking)
            ):
                raise ValueError(
                    "tutorial prediction does not cover its exact test pool"
                )
            selected = ranking[0]
            if (
                row.get("runtime_by_candidate_ms") != runtimes
                or row.get("selected_candidate_id") != selected
                or row.get("selected_runtime_ms") != runtimes[selected]
            ):
                raise ValueError("tutorial prediction labels differ from the test pool")
            values[model].append(
                {
                    "weight": len(labels),
                    "top_1_score": _peak_score(labels, ranking, 1),
                    "top_5_score": _peak_score(labels, ranking, 5),
                }
            )
    expected = {(model, workload) for model in expected_models for workload in entries}
    if seen != expected:
        raise ValueError("tutorial predictions are missing model/workload results")
    models = []
    csv_rows = []
    for model, model_type in expected_models.items():
        rows = values[model]
        total = sum(row["weight"] for row in rows)
        weighted = {
            name: sum(row[name] * row["weight"] for row in rows) / total
            for name in ("top_1_score", "top_5_score")
        }
        macro = {
            name: mean(row[name] for row in rows)
            for name in ("top_1_score", "top_5_score")
        }
        base = {
            "model_id": model,
            "model_type": model_type,
            "workloads": len(rows),
            "test_records": total,
        }
        models.append(
            {**base, "record_weighted_metrics": weighted, "macro_metrics": macro}
        )
        csv_rows.append(
            {
                **base,
                **{f"weighted_{name}": value for name, value in weighted.items()},
                **{f"macro_{name}": value for name, value in macro.items()},
            }
        )
    summary = {
        "schema_version": 1,
        "experiment_protocol": EXPERIMENT_PROTOCOL,
        "reference_alignment_status": ALIGNMENT_STATUS,
        "paired_xgboost_comparison": False,
        "collection_index_sha256": _sha256(plan.index_path),
        "predictions_sha256": _sha256(source),
        "aggregation": "weight each workload by its number of test records",
        "score_definition": "mean prefix-best throughput / max throughput in test pool",
        "arithmetic": "Python float64 recomputation of exported official label values",
        "canonical_metrics_note": (
            "The ordinary metrics.json remains runtime-based and workload-macro. "
            "These label-based scores can differ by floating-point roundoff."
        ),
        "global_tournament_pairwise_accuracy_reported": False,
        "external_reference_file": REFERENCE_JSON,
        "models": models,
    }
    reference = {
        "schema_version": 1,
        "status": "external_published_tutorial_reference_only",
        "source": REFERENCE_SOURCE,
        "reference_alignment_status": ALIGNMENT_STATUS,
        "paired_comparison": False,
        "model": "XGBoost",
        "sampled_files": 500,
        "retained_workloads": 484,
        "training_records": 1689022,
        "test_records": 187679,
        "platform": "platinum-8272",
        "split": "90/10 within-task record split",
        "reported_metrics": {
            "top_1_score": 0.829146,
            "top_5_score": 0.904636,
            "pairwise_accuracy": 0.846439,
        },
        "limitations": [
            "Original inventory, dataset cache and XGBoost artifacts are unavailable.",
            "Equal counts do not prove matching test records or XGBoost predictions.",
            "The published aggregate is not a model prediction in this experiment.",
            "Tournament group rankings do not establish global Pairwise Accuracy.",
            "No XGBoost runtime-regret value is inferred from aggregate scores.",
        ],
    }
    _write_json(plan.output_dir / REFERENCE_JSON, reference)
    _write_json(plan.output_dir / SUMMARY_JSON, summary)

    def write_csv(stream):
        writer = csv.DictWriter(
            stream, fieldnames=list(csv_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    _atomic_text(plan.output_dir / SUMMARY_CSV, write_csv)
    return summary


def run_tenset_tutorial(
    config_path: str | Path,
    *,
    allow_external_reference: bool = False,
    workload_limit: int | None = None,
    resume: bool = False,
    output_dir: str | Path | None = None,
    progress: Callable[[int, int, str, bool], None] | None = None,
) -> ExperimentRunResult:
    """Run a checked reconstruction with explicit external-reference labelling."""

    if allow_external_reference is not True:
        raise ValueError(
            "original tutorial/XGBoost artifacts are unverified; execution requires "
            "--allow-external-reference and remains a Jev-versus-Random reconstruction"
        )
    # Validate every selected private label and workload before starting any model.
    preflight_tenset_tutorial(
        config_path, workload_limit=workload_limit, output_dir=output_dir
    )
    result = collection.run_tenset_collection(
        config_path,
        workload_limit=workload_limit,
        resume=resume,
        output_dir=output_dir,
        progress=progress,
    )
    summarize_tenset_tutorial(
        config_path, workload_limit=workload_limit, output_dir=output_dir
    )
    return result
