"""Analyze saved TenSet runs without making decisions or issuing API requests."""

import csv
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from math import isclose
from pathlib import Path
from statistics import mean

import yaml

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


def _group_rows(
    rows: Sequence[dict],
    keys: Sequence[str],
    numeric_columns: Sequence[str] = NUMERIC_COLUMNS,
) -> list[dict]:
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
                **{key: mean(row[key] for row in members) for key in numeric_columns},
            }
        )
    return output


class _CollectionWorkloads(Mapping):
    """Index workload descriptors while retaining at most one decoded shard."""

    def __init__(self, index_path: Path, index: Mapping) -> None:
        self.index_path = index_path
        self.descriptors = {row["workload_id"]: row for row in index["shards"]}
        if len(self.descriptors) != len(index["shards"]):
            raise ValueError("collection workload IDs must be unique")
        self.cached_id = None
        self.cached_workload = None

    def __len__(self):
        return len(self.descriptors)

    def __iter__(self):
        return iter(self.descriptors)

    def __contains__(self, key):
        return key in self.descriptors

    def __getitem__(self, key):
        if key != self.cached_id:
            from jev_compiler_eval.datasets.tenset_collection import (
                load_collection_shard,
            )

            shard = load_collection_shard(self.index_path, self.descriptors[key])
            self.cached_workload = shard["workloads"][0]
            self.cached_id = key
        return self.cached_workload


def _case_from_prediction(
    prepared: Mapping, prediction: Mapping, *, describe_config_fields: bool = False
) -> EvaluationCase:
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
    context = dict(prepared["context"])
    if describe_config_fields:
        from jev_compiler_eval.datasets.tenset_descriptions import (
            tenset_configuration_fields,
        )

        context["configuration_fields"] = tenset_configuration_fields()
    return EvaluationCase(
        Workload(workload_id, context),
        tuple(
            Candidate(item, candidate_map[item]["configuration"])
            for item in candidate_ids
        ),
        tuple(
            RuntimeRecord(workload_id, item, candidate_map[item]["runtime_ms"])
            for item in candidate_ids
        ),
    )


def _requested_metrics(result_dir: Path) -> tuple[str, ...]:
    """Read declared metrics, retaining compatibility with older small bundles."""

    experiment_path = result_dir / "experiment.yaml"
    metrics = None
    if experiment_path.is_file():
        config = yaml.safe_load(experiment_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("saved experiment must be a YAML mapping")
        metrics = config.get("metrics")
        if metrics is None:
            raise ValueError("saved experiment must declare its metrics")
    else:
        metrics_path = result_dir / "metrics.json"
        if metrics_path.is_file():
            saved = _read_json(metrics_path).get("models")
            if not isinstance(saved, list) or not saved:
                raise ValueError("saved metric summaries must contain models")
            sets = []
            for model in saved:
                values = model.get("metrics") if isinstance(model, Mapping) else None
                if not isinstance(values, Mapping) or not values:
                    raise ValueError("saved model metrics must be a nonempty mapping")
                sets.append(set(values))
            if any(values != sets[0] for values in sets):
                raise ValueError("models must declare the same metric set")
            metrics = list(sets[0])
        else:
            with (result_dir / "predictions.jsonl").open(encoding="utf-8") as file:
                for line in file:
                    if line.strip():
                        prediction = json.loads(line)
                        if isinstance(prediction.get("metrics"), Mapping):
                            metrics = list(prediction["metrics"])
                        break
    if metrics is None:
        return METRIC_NAMES
    if (
        not isinstance(metrics, (list, tuple))
        or not metrics
        or any(not isinstance(name, str) for name in metrics)
        or len(metrics) != len(set(metrics))
        or set(metrics) - set(METRIC_NAMES)
    ):
        raise ValueError("analysis requires a nonempty supported TenSet metric set")
    return tuple(name for name in METRIC_NAMES if name in metrics)


def _known_sum(values):
    values = list(values)
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in values
    ):
        return ""
    return sum(values)


def _decision_accounting(row: Mapping, prediction: Mapping) -> dict:
    metadata = prediction.get("metadata", {})
    jev = str(prediction["model_type"]).startswith("jev")
    cached = metadata.get("cache_hit")
    reports = metadata.get("request_reports")
    if reports is not None:
        if (
            not isinstance(reports, list)
            or not reports
            or any(
                not isinstance(report, Mapping)
                or not isinstance(report.get("cache_hit"), bool)
                for report in reports
            )
        ):
            raise ValueError("Jev request reports must declare cache status")
        live = [report for report in reports if not report["cache_hit"]]
        cache_count = len(reports) - len(live)
        accounting = {
            "cache_hit": "mixed" if live and cache_count else not bool(live),
            "live_decisions": int(bool(live)),
            "cached_decisions": int(not live),
            "live_requests": len(live),
            "cached_requests": cache_count,
            "response_input_tokens": _known_sum(
                report.get("input_tokens") for report in reports
            ),
            "response_output_tokens": _known_sum(
                report.get("output_tokens") for report in reports
            ),
            "live_input_tokens": _known_sum(
                report.get("input_tokens") for report in live
            ),
            "live_output_tokens": _known_sum(
                report.get("output_tokens") for report in live
            ),
            "api_latency_ms": _known_sum(
                report.get("api_latency_ms") for report in reports
            ),
            "live_api_latency_ms": _known_sum(
                report.get("api_latency_ms") for report in live
            ),
            "request_id": ";".join(
                str(report["request_id"])
                for report in reports
                if report.get("request_id")
            ),
        }
    else:
        accounting = {
            "cache_hit": cached if jev else "",
            "live_decisions": int(cached is False) if jev else 0,
            "cached_decisions": int(cached is True) if jev else 0,
            "live_requests": int(cached is False) if jev else 0,
            "cached_requests": int(cached is True) if jev else 0,
            "response_input_tokens": metadata.get("input_tokens", ""),
            "response_output_tokens": metadata.get("output_tokens", ""),
            "live_input_tokens": metadata.get("input_tokens", "")
            if jev and cached is False
            else 0,
            "live_output_tokens": metadata.get("output_tokens", "")
            if jev and cached is False
            else 0,
            "api_latency_ms": metadata.get("api_latency_ms", ""),
            "live_api_latency_ms": metadata.get("api_latency_ms", "")
            if jev and cached is False
            else 0,
            "request_id": metadata.get("request_id", ""),
        }
    return {
        **{
            key: row[key]
            for key in ("run_index", "experiment", "model_id", "workload_id")
        },
        **accounting,
        "monetary_cost_usd": "",
    }


def _load_rows(
    result_dirs: Sequence[Path], workloads: Mapping, prepared_sha256: str
) -> tuple[list, list, tuple[str, ...]]:
    rows = []
    accounting = []
    metric_names = None
    splits = set()
    for run_index, result_dir in enumerate(result_dirs):
        run_metrics = _requested_metrics(result_dir)
        if metric_names is None:
            metric_names = run_metrics
        elif run_metrics != metric_names:
            raise ValueError("analysis cannot pool different metric sets")
        metric_functions = resolve_metrics(metric_names)
        manifest = _read_json(result_dir / "manifest.json")
        if manifest.get("dataset_type") != "tenset":
            raise ValueError(f"result bundle is not a TenSet experiment: {result_dir}")
        dataset_provenance = manifest.get("dataset_provenance", {})
        describe_fields = dataset_provenance.get("describe_config_fields", False)
        if not isinstance(describe_fields, bool):
            raise ValueError("describe_config_fields provenance must be a boolean")
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
                prepared_workload = workloads[workload_id]
                if (
                    split not in (None, "all")
                    and prepared_workload.get("split") != split
                ):
                    raise ValueError("prediction workload does not belong to run split")
                case = _case_from_prediction(
                    prepared_workload,
                    prediction,
                    describe_config_fields=describe_fields,
                )
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
                if prediction["model_type"] == "jev-tournament" and any(
                    name.startswith("pairwise_accuracy") for name in metric_names
                ):
                    raise ValueError(
                        "tournament analysis cannot report global Pairwise Accuracy; "
                        "use the selection metrics declared by its experiment"
                    )
                metrics = {
                    name: function(case, decision)
                    for name, function in metric_functions.items()
                }
                saved_metrics = prediction.get("metrics")
                if saved_metrics is not None:
                    if not isinstance(saved_metrics, Mapping) or set(
                        saved_metrics
                    ) != set(metric_names):
                        raise ValueError("prediction metrics disagree with experiment")
                    for name, expected in metrics.items():
                        value = saved_metrics[name]
                        if (
                            isinstance(value, bool)
                            or not isinstance(value, (int, float))
                            or not isclose(value, expected, rel_tol=1e-9, abs_tol=1e-12)
                        ):
                            raise ValueError(
                                "saved metric does not match recomputation"
                            )
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
                accounting.append(_decision_accounting(row, prediction))
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
    return rows, accounting, metric_names or METRIC_NAMES


def _jev_examples(rows: Sequence[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["run_index"], row["workload_id"]].append(row)
    comparisons: dict[tuple, list[dict]] = defaultdict(list)
    for members in grouped.values():
        for jev in (row for row in members if row["model_type"].startswith("jev")):
            if "top_1_score" not in jev or "runtime_regret" not in jev:
                continue
            for baseline in (
                row for row in members if not row["model_type"].startswith("jev")
            ):
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
    rows: Sequence[dict],
    output_dir: Path,
    plt,
    *,
    synthetic: bool = False,
    metric_names: Sequence[str] = METRIC_NAMES,
) -> None:
    labels = [f"{row['model_id']}\n{row['unique_workloads']} workloads" for row in rows]
    plotted = [
        metric
        for metric in ("top_1_score", "top_5_score", "pairwise_accuracy")
        if metric in metric_names
    ]
    if "pairwise_accuracy" not in plotted and "runtime_regret" in metric_names:
        plotted.append("runtime_regret")
    if not plotted:
        plotted = list(metric_names)
    titles = {
        "top_1_score": "Top-1 Score",
        "top_5_score": "Top-5 average peak Score",
        "pairwise_accuracy": "Pairwise Accuracy",
        "pairwise_accuracy_tie_aware": "Tie-aware Pairwise Accuracy",
        "runtime_regret": "Runtime regret",
    }
    figure, axes = plt.subplots(
        1,
        len(plotted),
        figsize=(max(10, len(rows) * 1.8), 3.8),
        squeeze=False,
    )
    if synthetic:
        figure.suptitle(
            "Synthetic software test — not benchmark results",
            fontsize=14,
            fontweight="bold",
            color="#9d231d",
        )
    for axis, metric in zip(axes[0], plotted, strict=True):
        axis.bar(range(len(rows)), [row[metric] for row in rows])
        axis.set_xticks(range(len(rows)), labels, rotation=30, ha="right")
        axis.set_ylim(bottom=0)
        if metric != "runtime_regret":
            axis.set_ylim(top=1.05)
        axis.set_title(titles[metric])
        direction = "lower" if metric == "runtime_regret" else "higher"
        axis.set_ylabel(f"Macro mean ({direction} is better)")
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
    if prepared.get("dataset") == "tenset_collection":
        from jev_compiler_eval.datasets.tenset_collection import load_collection_index

        prepared = load_collection_index(prepared_path)
        workloads = _CollectionWorkloads(prepared_path, prepared)
    elif prepared.get("dataset") == "tenset":
        workloads = {row["workload_id"]: row for row in prepared["workloads"]}
        if len(workloads) != len(prepared["workloads"]):
            raise ValueError("prepared workload IDs must be unique")
    else:
        raise ValueError("prepared data must be a TenSet artifact or collection index")
    prepared_sha256 = hashlib.sha256(prepared_path.read_bytes()).hexdigest()
    rows, accounting, metric_names = _load_rows(paths, workloads, prepared_sha256)
    numeric_columns = (*metric_names, "decision_latency_ms")
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
        *numeric_columns,
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
        grouped = _group_rows(rows, keys, numeric_columns)
        _write_csv(
            output_dir / f"{name}.csv",
            grouped,
            (*keys, "decisions", "unique_workloads", *numeric_columns),
        )
        if name == "aggregate":
            aggregate = grouped
    overall = _group_rows(rows, ("model_id", "model_type"), numeric_columns)
    _write_csv(
        output_dir / "overall.csv",
        overall,
        (
            "model_id",
            "model_type",
            "decisions",
            "unique_workloads",
            *numeric_columns,
        ),
    )
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
            "live_requests",
            "cached_requests",
            "response_input_tokens",
            "response_output_tokens",
            "live_input_tokens",
            "live_output_tokens",
            "api_latency_ms",
            "live_api_latency_ms",
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
        "metrics": list(metric_names),
        "aggregation": "unweighted mean of workload decisions, pooled across runs",
        "paper_aggregation": (
            "TenSet train_model.py weights tasks by candidate count; this equals "
            "the macro mean within each fixed-candidate-count table row"
        ),
        "top_5_definition": "mean prefix-best runtime ratio over first min(5,n) trials",
        "pairwise_definition": "official strict comparisons in original candidate order"
        if "pairwise_accuracy" in metric_names
        else "not reported by this experiment",
        "oracle": "minimum runtime among the candidates in each prediction",
        "ranking_ties": (
            "Top-K follows returned order; reported pairwise metrics use model score "
            "ties. Tournament selection metrics do not assume global probabilities."
        ),
        "monetary_cost": "unavailable; no pricing snapshot provided",
        "aggregates": aggregate,
        "overall": overall,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    if plt is not None:
        _plot_comparison(
            overall,
            output_dir,
            plt,
            synthetic=synthetic,
            metric_names=metric_names,
        )
    return output_dir
