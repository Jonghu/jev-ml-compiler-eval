"""Analyze saved TenSet runs without making decisions or issuing API requests."""

import csv
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean

from jev_compiler_eval.metrics import resolve_metrics
from jev_compiler_eval.types import (
    Candidate,
    DecisionResult,
    EvaluationCase,
    RuntimeRecord,
    Workload,
)

METRIC_NAMES = (
    "top_1_score",
    "top_5_score",
    "pairwise_accuracy",
    "pairwise_accuracy_tie_aware",
    "runtime_regret",
)
NUMERIC_COLUMNS = (*METRIC_NAMES, "decision_latency_ms")


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _write_csv(path: Path, rows: Sequence[Mapping], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _group_rows(rows: Sequence[dict], keys: Sequence[str]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    output = []
    for values, members in sorted(groups.items()):
        output.append(
            {
                **dict(zip(keys, values)),
                "decisions": len(members),
                "unique_workloads": len({row["workload_id"] for row in members}),
                **{key: mean(row[key] for row in members) for key in NUMERIC_COLUMNS},
            }
        )
    return output


def _case_from_prediction(prepared: Mapping, prediction: Mapping) -> EvaluationCase:
    workload_id = prediction["workload_id"]
    candidate_ids = prediction["candidate_ids"]
    if (
        not isinstance(candidate_ids, list)
        or not candidate_ids
        or not all(isinstance(item, str) for item in candidate_ids)
        or len(candidate_ids) != len(set(candidate_ids))
    ):
        raise ValueError("prediction candidate IDs must be a nonempty unique list")
    candidate_map = {item["candidate_id"]: item for item in prepared["candidates"]}
    if len(candidate_map) != len(prepared["candidates"]):
        raise ValueError("prepared candidate IDs must be unique")
    if set(candidate_ids) - candidate_map.keys():
        raise ValueError("prediction contains candidates absent from prepared data")
    return EvaluationCase(
        Workload(workload_id, prepared["context"]),
        tuple(
            Candidate(item, candidate_map[item]["configuration"])
            for item in candidate_ids
        ),
        tuple(
            RuntimeRecord(workload_id, item, candidate_map[item]["runtime_ms"])
            for item in candidate_ids
        ),
    )


def _load_rows(
    result_dirs: Sequence[Path], workloads: Mapping, prepared_sha256: str
) -> tuple[list, list]:
    rows = []
    accounting = []
    metric_functions = resolve_metrics(METRIC_NAMES)
    splits = set()
    for run_index, result_dir in enumerate(result_dirs):
        manifest = _read_json(result_dir / "manifest.json")
        if manifest.get("dataset_type") != "tenset":
            raise ValueError(f"result bundle is not a TenSet experiment: {result_dir}")
        dataset_provenance = manifest.get("dataset_provenance", {})
        saved_checksum = dataset_provenance.get("prepared_file_sha256")
        if saved_checksum is not None and saved_checksum != prepared_sha256:
            raise ValueError("result bundle prepared-data checksum does not match")
        split = dataset_provenance.get("split")
        if split is not None:
            splits.add(split)
            if len(splits) > 1:
                raise ValueError("analysis cannot pool different dataset splits")
        run_rows = []
        seen = set()
        candidate_orders = {}
        with (result_dir / "predictions.jsonl").open(encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                prediction = json.loads(line)
                workload_id = prediction["workload_id"]
                key = prediction["model_id"], workload_id
                if key in seen:
                    raise ValueError("duplicate model/workload prediction in bundle")
                seen.add(key)
                if workload_id not in workloads:
                    raise ValueError("prediction workload absent from prepared data")
                if (
                    split not in (None, "all")
                    and workloads[workload_id].get("split") != split
                ):
                    raise ValueError("prediction workload does not belong to run split")
                case = _case_from_prediction(workloads[workload_id], prediction)
                model_input = json.dumps(
                    {
                        "workload_id": workload_id,
                        "context": dict(case.workload.context),
                        "candidates": {
                            item.candidate_id: dict(item.configuration)
                            for item in case.candidates
                        },
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
                if (
                    "model_input_sha256" in prediction
                    and prediction["model_input_sha256"]
                    != hashlib.sha256(model_input).hexdigest()
                ):
                    raise ValueError("saved model input checksum does not match data")
                if "model_input_bytes" in prediction and prediction[
                    "model_input_bytes"
                ] != len(model_input):
                    raise ValueError("saved model input byte count does not match data")
                runtime_map = {
                    item.candidate_id: item.runtime_ms for item in case.runtimes
                }
                if (
                    "runtime_by_candidate_ms" in prediction
                    and prediction["runtime_by_candidate_ms"] != runtime_map
                ):
                    raise ValueError("saved runtimes disagree with prepared data")
                order = tuple(item.candidate_id for item in case.candidates)
                if (
                    workload_id in candidate_orders
                    and order != candidate_orders[workload_id]
                ):
                    raise ValueError(
                        "models must evaluate identical candidate sets and order"
                    )
                candidate_orders[workload_id] = order
                decision = DecisionResult(
                    tuple(prediction["predicted_ranking"]),
                    prediction["decision_latency_ms"],
                    prediction.get("metadata", {}),
                )
                metrics = {
                    name: function(case, decision)
                    for name, function in metric_functions.items()
                }
                row = {
                    "run_index": run_index,
                    "experiment": manifest["experiment_name"],
                    "seed": manifest["seed"],
                    "model_id": prediction["model_id"],
                    "model_type": prediction["model_type"],
                    "workload_id": workload_id,
                    "operation": str(case.workload.context.get("operation", "unknown")),
                    "candidate_count": len(case.candidates),
                    **metrics,
                    "decision_latency_ms": decision.decision_latency_ms,
                }
                rows.append(row)
                run_rows.append(row)
                metadata = prediction.get("metadata", {})
                jev = prediction["model_type"] == "jev"
                cached = metadata.get("cache_hit")
                accounting.append(
                    {
                        **{
                            key: row[key]
                            for key in (
                                "run_index",
                                "experiment",
                                "model_id",
                                "workload_id",
                            )
                        },
                        "cache_hit": cached if jev else "",
                        "live_decisions": int(cached is False) if jev else 0,
                        "cached_decisions": int(cached is True) if jev else 0,
                        "response_input_tokens": metadata.get("input_tokens", ""),
                        "response_output_tokens": metadata.get("output_tokens", ""),
                        "live_input_tokens": metadata.get("input_tokens", "")
                        if jev and cached is False
                        else 0,
                        "live_output_tokens": metadata.get("output_tokens", "")
                        if jev and cached is False
                        else 0,
                        "api_latency_ms": metadata.get("api_latency_ms", ""),
                        "request_id": metadata.get("request_id", ""),
                        "monetary_cost_usd": "",
                    }
                )
        if not run_rows:
            raise ValueError("result bundle contains no predictions")
        models = set(manifest["model_ids"])
        actual_models = {row["model_id"] for row in run_rows}
        if actual_models != models:
            raise ValueError("prediction models do not match manifest model IDs")
        expected = set(candidate_orders)
        for model in models:
            if {
                row["workload_id"] for row in run_rows if row["model_id"] == model
            } != expected:
                raise ValueError("models must evaluate identical workloads")
    return rows, accounting


def _jev_examples(rows: Sequence[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["run_index"], row["workload_id"]].append(row)
    comparisons: dict[tuple, list[dict]] = defaultdict(list)
    for members in grouped.values():
        for jev in (row for row in members if row["model_type"] == "jev"):
            for baseline in (row for row in members if row["model_type"] != "jev"):
                key = jev["model_id"], baseline["model_id"]
                comparisons[key].append(
                    {
                        **{
                            key: jev[key]
                            for key in (
                                "run_index",
                                "experiment",
                                "workload_id",
                                "operation",
                                "candidate_count",
                            )
                        },
                        "jev_model": jev["model_id"],
                        "baseline_model": baseline["model_id"],
                        "jev_top_1_score": jev["top_1_score"],
                        "baseline_top_1_score": baseline["top_1_score"],
                        "top_1_score_delta": jev["top_1_score"]
                        - baseline["top_1_score"],
                        "jev_runtime_regret": jev["runtime_regret"],
                        "baseline_runtime_regret": baseline["runtime_regret"],
                    }
                )
    examples = []
    for members in comparisons.values():
        ordered = sorted(
            members,
            key=lambda row: (
                row["top_1_score_delta"],
                row["workload_id"],
                row["run_index"],
            ),
        )
        examples.extend({**row, "selection": "lowest_delta"} for row in ordered[:5])
        examples.extend({**row, "selection": "highest_delta"} for row in ordered[-5:])
    return examples


def _plot_comparison(
    rows: Sequence[dict], output_dir: Path, plt, *, synthetic: bool = False
) -> None:
    labels = [f"{row['model_id']}\nn={row['candidate_count']}" for row in rows]
    figure, axes = plt.subplots(1, 3, figsize=(max(10, len(rows) * 1.8), 3.8))
    if synthetic:
        figure.suptitle(
            "Synthetic software test — not benchmark results",
            fontsize=14,
            fontweight="bold",
            color="#9d231d",
        )
    for axis, metric, title in zip(
        axes,
        ("top_1_score", "top_5_score", "pairwise_accuracy"),
        ("Top-1 Score", "Top-5 average peak Score", "Pairwise Accuracy"),
    ):
        axis.bar(range(len(rows)), [row[metric] for row in rows])
        axis.set_xticks(range(len(rows)), labels, rotation=30, ha="right")
        axis.set_ylim(0, 1.05)
        axis.set_title(title)
        axis.set_ylabel("Macro mean (higher is better)")
        axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "comparison.png", dpi=180)
    figure.savefig(output_dir / "comparison.svg")
    plt.close(figure)


def analyze_tenset(
    result_dirs: Sequence[Path],
    prepared_path: Path,
    output_dir: Path,
    *,
    create_plots: bool = True,
) -> Path:
    """Validate saved runs and write tables, plots, and accounting diagnostics.

    Metrics are recomputed from prepared runtimes and saved model rankings.
    Each run must compare identical workloads and ordered candidate sets.
    Across runs, tables may pool seeds, and keep candidate counts separate.
    Costs stay blank without an explicit pricing snapshot; no API is called.
    """

    prepared_path, output_dir = Path(prepared_path), Path(output_dir)
    paths = [Path(path) for path in result_dirs]
    if not paths or len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("provide one or more distinct result bundle directories")
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    prepared = _read_json(prepared_path)
    if prepared.get("dataset") != "tenset":
        raise ValueError("prepared data must have dataset=tenset")
    workloads = {row["workload_id"]: row for row in prepared["workloads"]}
    if len(workloads) != len(prepared["workloads"]):
        raise ValueError("prepared workload IDs must be unique")
    prepared_sha256 = hashlib.sha256(prepared_path.read_bytes()).hexdigest()
    rows, accounting = _load_rows(paths, workloads, prepared_sha256)
    plt = None
    if create_plots:
        try:
            import matplotlib

            matplotlib.use("Agg")
            from matplotlib import pyplot as plt
        except ImportError as error:
            raise RuntimeError(
                "plots require the tenset extra: uv sync --extra tenset"
            ) from error
    output_dir.mkdir(parents=True, exist_ok=True)
    row_fields = (
        "run_index",
        "experiment",
        "seed",
        "model_id",
        "model_type",
        "workload_id",
        "operation",
        "candidate_count",
        *NUMERIC_COLUMNS,
    )
    _write_csv(output_dir / "per_workload.csv", rows, row_fields)
    aggregate = []
    for name, keys in (
        ("aggregate", ("model_id", "model_type", "candidate_count")),
        ("by_operation", ("model_id", "model_type", "operation", "candidate_count")),
        ("by_candidate_count", ("model_id", "model_type", "candidate_count")),
        (
            "by_run",
            (
                "run_index",
                "experiment",
                "seed",
                "model_id",
                "model_type",
                "candidate_count",
            ),
        ),
    ):
        grouped = _group_rows(rows, keys)
        _write_csv(
            output_dir / f"{name}.csv",
            grouped,
            (*keys, "decisions", "unique_workloads", *NUMERIC_COLUMNS),
        )
        if name == "aggregate":
            aggregate = grouped
    examples = _jev_examples(rows)
    _write_csv(
        output_dir / "jev_cases.csv",
        examples,
        (
            "selection",
            "run_index",
            "experiment",
            "workload_id",
            "operation",
            "candidate_count",
            "jev_model",
            "baseline_model",
            "jev_top_1_score",
            "baseline_top_1_score",
            "top_1_score_delta",
            "jev_runtime_regret",
            "baseline_runtime_regret",
        ),
    )
    _write_csv(
        output_dir / "decision_accounting.csv",
        accounting,
        (
            "run_index",
            "experiment",
            "model_id",
            "workload_id",
            "cache_hit",
            "live_decisions",
            "cached_decisions",
            "response_input_tokens",
            "response_output_tokens",
            "live_input_tokens",
            "live_output_tokens",
            "api_latency_ms",
            "request_id",
            "monetary_cost_usd",
        ),
    )
    provenance = prepared.get("provenance", {})
    synthetic = provenance.get("synthetic") is True
    summary = {
        "dataset": "tenset",
        "data_status": "synthetic_fixture" if synthetic else "source_records",
        "prepared_sha256": prepared_sha256,
        "provenance": provenance,
        "result_bundles": [str(path) for path in paths],
        "aggregation": "unweighted mean of workload decisions, pooled across runs",
        "paper_aggregation": (
            "TenSet train_model.py weights tasks by candidate count; this equals "
            "the macro mean within each fixed-candidate-count table row"
        ),
        "top_5_definition": "mean prefix-best runtime ratio over first min(5,n) trials",
        "pairwise_definition": (
            "official strict comparisons in original candidate order"
        ),
        "oracle": "minimum runtime among the sampled candidates in each prediction",
        "ranking_ties": "Top-K follows returned order; pairwise uses model score ties",
        "monetary_cost": "unavailable; no pricing snapshot provided",
        "aggregates": aggregate,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    if plt is not None:
        _plot_comparison(aggregate, output_dir, plt, synthetic=synthetic)
    return output_dir
