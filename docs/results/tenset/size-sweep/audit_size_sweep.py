"""Audit original exported TenSet size-sweep tables and paired uncertainty.

Use the original extracted upload with analysis/ and size-comparison/ directories.
This script reads tables only, makes no decisions or API calls, and never edits
the source export. It requires NumPy plus the Python standard library.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

import numpy as np

SIZES = (16, 32, 50)
MODELS = {"random-42": "random", "jev-1.14.0-long-context": "jev"}
METRICS = (
    "top_1_score",
    "top_5_score",
    "pairwise_accuracy",
    "pairwise_accuracy_tie_aware",
    "runtime_regret",
)
BOOTSTRAP_METRICS = (
    "top_1_score",
    "top_5_score",
    "pairwise_accuracy",
    "runtime_regret",
)
PREPARED_SHA = "1f949c2c9f788b507c4e395925720c7beaa7063c3efb6ac09ae5ed4855db6275"
RATE = 0.042


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def close(actual, expected, label):
    if not math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"{label} differs: {actual} vs {expected}")


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize_groups(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(str(row[key]) for key in keys)].append(row)
    return {
        key: {
            "decisions": len(members),
            "unique_workloads": len({row["workload_id"] for row in members}),
            **{
                metric: mean(float(row[metric]) for row in members)
                for metric in (*METRICS, "decision_latency_ms")
            },
        }
        for key, members in groups.items()
    }


def audit_aggregate(path, rows, keys):
    expected = summarize_groups(rows, keys)
    observed = {}
    for row in read_csv(path):
        key = tuple(str(row[name]) for name in keys)
        if key in observed:
            raise ValueError(f"duplicate aggregate in {path.name}")
        observed[key] = row
    if set(observed) != set(expected):
        raise ValueError(f"group identity mismatch in {path.name}")
    for key, values in expected.items():
        for name, value in values.items():
            close(observed[key][name], value, f"{path.name}: {key}: {name}")
    return {"groups": len(expected), "all_values_recomputed": True}


def validate_rows(rows):
    if len(rows) != 720:
        raise ValueError("expected 720 per-workload model/budget rows")
    lookup, operations = {}, {}
    runs = defaultdict(set)
    for row in rows:
        budget, model, workload = (
            int(row["candidate_count"]),
            row["model_id"],
            row["workload_id"],
        )
        key = (budget, model, workload)
        if (
            budget not in SIZES
            or model not in MODELS
            or row["model_type"] != MODELS[model]
            or row["seed"] != "42"
            or key in lookup
        ):
            raise ValueError("invalid or duplicate model/budget/workload identity")
        lookup[key] = row
        runs[int(row["run_index"])].add((budget, row["experiment"]))
        if workload in operations and operations[workload] != row["operation"]:
            raise ValueError("operation labels change across model/budget rows")
        operations[workload] = row["operation"]
        for name in (*METRICS, "decision_latency_ms"):
            value = float(row[name])
            if not math.isfinite(value) or value < 0:
                raise ValueError("nonfinite or negative metric/latency")
            if name not in {"runtime_regret", "decision_latency_ms"} and value > 1:
                raise ValueError("normalized score or pairwise accuracy exceeds one")
        close(
            row["top_1_score"],
            1 / (1 + float(row["runtime_regret"])),
            "per-case Top-1/regret identity",
        )
        if float(row["top_5_score"]) + 1e-12 < float(row["top_1_score"]):
            raise ValueError("Top-5 below Top-1")
        count = budget * (budget - 1) // 2
        correct = float(row["pairwise_accuracy"]) * count
        if abs(correct - round(correct)) > 1e-8:
            raise ValueError("strict pairwise accuracy is not an integer-pair ratio")
    ids = sorted(operations)
    if len(ids) != 120:
        raise ValueError("expected exactly 120 distinct workloads")
    expected = {
        (size, model, workload)
        for size in SIZES
        for model in MODELS
        for workload in ids
    }
    if set(lookup) != expected:
        raise ValueError("models/budgets do not cover the same 120 workload IDs")
    if (
        set(runs) != {0, 1, 2}
        or any(len(values) != 1 for values in runs.values())
        or [next(iter(runs[index]))[0] for index in (0, 1, 2)] != list(SIZES)
    ):
        raise ValueError("run indices do not describe one fixed run per budget")
    return lookup, ids, operations


def audit_accounting(path, lookup):
    accounting = read_csv(path)
    if len(accounting) != len(lookup):
        raise ValueError("accounting count differs from per-workload count")
    by_size = {size: [] for size in SIZES}
    seen, request_ids = set(), set()
    for row in accounting:
        size = SIZES[int(row["run_index"])]
        key = (size, row["model_id"], row["workload_id"])
        if key not in lookup or key in seen:
            raise ValueError("accounting identity mismatch or duplicate")
        seen.add(key)
        if row["experiment"] != lookup[key]["experiment"]:
            raise ValueError("accounting experiment identity differs")
        integers = {
            name: int(row[name] or 0)
            for name in (
                "live_decisions",
                "cached_decisions",
                "live_requests",
                "cached_requests",
                "response_input_tokens",
                "response_output_tokens",
                "live_input_tokens",
                "live_output_tokens",
            )
        }
        if any(value < 0 for value in integers.values()):
            raise ValueError("negative usage counter")
        if row["model_id"] == "random-42":
            if any(integers.values()):
                raise ValueError("Random accounting unexpectedly charges API usage")
            continue
        cached = row["cache_hit"] == "True"
        if row["cache_hit"] not in {"True", "False"}:
            raise ValueError("Jev cache status missing")
        if (
            integers["live_requests"] != int(not cached)
            or integers["cached_requests"] != int(cached)
            or integers["live_decisions"] != int(not cached)
            or integers["cached_decisions"] != int(cached)
            or integers["live_input_tokens"]
            != (0 if cached else integers["response_input_tokens"])
            or integers["live_output_tokens"]
            != (0 if cached else integers["response_output_tokens"])
        ):
            raise ValueError("live/cache usage counters are inconsistent")
        request_id = row["request_id"]
        if not request_id or request_id in request_ids:
            raise ValueError("missing or repeated response request ID")
        request_ids.add(request_id)
        by_size[size].append((row, integers))
    if seen != set(lookup):
        raise ValueError("accounting omits model/workload rows")
    results = {}
    for size, members in by_size.items():
        totals = {
            name: sum(item[name] for _, item in members) for name in members[0][1]
        }
        live_latencies = [
            float(row["live_api_latency_ms"])
            for row, item in members
            if item["live_requests"]
        ]
        results[str(size)] = {
            "scope": "new_size_sweep_run"
            if size < 50
            else "reused_historical_50_anchor",
            "workloads": len(members),
            **totals,
            "stored_response_input_estimate_usd": totals["response_input_tokens"]
            * RATE
            / 1e6,
            "logged_live_input_estimate_usd": totals["live_input_tokens"] * RATE / 1e6,
            "mean_live_api_latency_ms": mean(live_latencies),
            "median_live_api_latency_ms": median(live_latencies),
            "mean_decision_latency_ms": mean(
                float(
                    lookup[size, "jev-1.14.0-long-context", row["workload_id"]][
                        "decision_latency_ms"
                    ]
                )
                for row, _ in members
            ),
        }
    new_inputs = sum(results[str(size)]["live_input_tokens"] for size in (16, 32))
    new_outputs = sum(results[str(size)]["live_output_tokens"] for size in (16, 32))
    total_inputs = sum(result["response_input_tokens"] for result in results.values())
    return {
        "pricing_assumption": {
            "usd_per_million_input_tokens": RATE,
            "output_free_assumed": True,
            "invoice_available": False,
        },
        "by_budget": results,
        "new_c16_c32_inputs": new_inputs,
        "new_c16_c32_outputs": new_outputs,
        "new_c16_c32_live_requests": sum(
            results[str(size)]["live_requests"] for size in (16, 32)
        ),
        "new_c16_c32_input_estimate_usd": new_inputs * RATE / 1e6,
        "all_stored_responses_including_historical_anchor_inputs": total_inputs,
        "all_stored_input_estimate_usd": total_inputs * RATE / 1e6,
        "unique_stored_request_ids": len(request_ids),
        "historical_scope_note": (
            "The c50 anchor's 110 live/10 cached status describes its original run, "
            "not new calls made for this sweep. Do not charge c50 again as new usage."
        ),
        "limitations": (
            "Counts are independently summed from exported accounting tables; raw "
            "new c16/c32 responses are unavailable. Estimates cover logged successful "
            "input usage at the stated rate, not invoices or unlogged failed calls."
        ),
    }


def historical_check(path, lookup):
    if path is None:
        return {"supplied": False, "independent_c50_raw_prediction_comparison": False}
    seen = set()
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (50, row["model_id"], row["workload_id"])
            if key not in lookup or key in seen:
                raise ValueError("historical c50 prediction scope differs")
            seen.add(key)
            for name in METRICS:
                close(lookup[key][name], row["metrics"][name], "historical c50 metric")
            close(
                lookup[key]["decision_latency_ms"],
                row["decision_latency_ms"],
                "historical c50 latency",
            )
    if len(seen) != 240:
        raise ValueError(
            "historical c50 file does not cover all 240 model/workload rows"
        )
    return {
        "supplied": True,
        "independent_c50_raw_prediction_comparison": True,
        "rows": len(seen),
        "source_predictions_sha256": sha256(path),
        "all_five_metrics_and_latency_match": True,
        "note": (
            "This verifies c50 continuity only, not raw c16/c32 decisions or labels."
        ),
    }


def bootstrap(lookup, ids, replicates, seed):
    differences = np.array(
        [
            [
                [
                    float(lookup[size, "jev-1.14.0-long-context", workload][metric])
                    - float(lookup[size, "random-42", workload][metric])
                    for metric in BOOTSTRAP_METRICS
                ]
                for workload in ids
            ]
            for size in SIZES
        ]
    )
    rng = np.random.default_rng(seed)
    samples = np.empty((replicates, len(SIZES), len(BOOTSTRAP_METRICS)))
    for start in range(0, replicates, 2000):
        indices = rng.integers(
            0, len(ids), size=(min(2000, replicates - start), len(ids))
        )
        samples[start : start + len(indices)] = (
            differences[:, indices, :].mean(axis=2).transpose(1, 0, 2)
        )
    results, table, paired_rows = {}, [], []
    for budget_index, size in enumerate(SIZES):
        metrics = {}
        for metric_index, metric in enumerate(BOOTSTRAP_METRICS):
            delta = differences[budget_index, :, metric_index]
            lower_is_better = metric == "runtime_regret"
            better = delta < 0 if lower_is_better else delta > 0
            worse = delta > 0 if lower_is_better else delta < 0
            random_mean = mean(
                float(lookup[size, "random-42", workload][metric]) for workload in ids
            )
            jev_mean = mean(
                float(lookup[size, "jev-1.14.0-long-context", workload][metric])
                for workload in ids
            )
            low, high = np.quantile(
                samples[:, budget_index, metric_index], [0.025, 0.975]
            )
            value = {
                "random_mean": random_mean,
                "jev_mean": jev_mean,
                "mean_delta_jev_minus_random": float(delta.mean()),
                "bootstrap_95ci": [float(low), float(high)],
                "median_paired_delta": float(np.median(delta)),
                "jev_better_workloads": int(better.sum()),
                "equal_workloads": int((delta == 0).sum()),
                "jev_worse_workloads": int(worse.sum()),
                "lower_is_better": lower_is_better,
            }
            metrics[metric] = value
            table.append(
                {
                    "candidates": size,
                    "metric": metric,
                    "random_mean": random_mean,
                    "jev_mean": jev_mean,
                    "jev_minus_random": value["mean_delta_jev_minus_random"],
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "jev_better_workloads": value["jev_better_workloads"],
                    "equal_workloads": value["equal_workloads"],
                    "jev_worse_workloads": value["jev_worse_workloads"],
                }
            )
        results[str(size)] = metrics
    contrasts, contrast_table = {}, []
    for index, size in enumerate(SIZES[:2]):
        metrics = {}
        for metric_index, metric in enumerate(BOOTSTRAP_METRICS):
            observed = (
                differences[index, :, metric_index] - differences[2, :, metric_index]
            )
            low, high = np.quantile(
                samples[:, index, metric_index] - samples[:, 2, metric_index],
                [0.025, 0.975],
            )
            metrics[metric] = {
                "contrast": f"(Jev-Random) at {size} minus (Jev-Random) at 50",
                "mean_difference_of_deltas": float(observed.mean()),
                "bootstrap_95ci": [float(low), float(high)],
                "lower_is_better_metric": metric == "runtime_regret",
            }
            contrast_table.append(
                {
                    "contrast": f"{size}-50",
                    "metric": metric,
                    "difference_of_jev_minus_random_deltas": float(observed.mean()),
                    "ci_low": float(low),
                    "ci_high": float(high),
                }
            )
        contrasts[f"{size}-50"] = metrics
    for workload_index, workload in enumerate(ids):
        paired = {
            "workload_id": workload,
            "operation": lookup[16, "random-42", workload]["operation"],
        }
        for budget_index, size in enumerate(SIZES):
            for metric_index, metric in enumerate(BOOTSTRAP_METRICS):
                paired[f"c{size}_{metric}_jev_minus_random"] = float(
                    differences[budget_index, workload_index, metric_index]
                )
        paired_rows.append(paired)
    return results, contrasts, table, contrast_table, paired_rows


def audit(
    input_dir, output_dir, historical_predictions=None, replicates=50000, seed=20261116
):
    input_dir, output_dir = Path(input_dir).resolve(), Path(output_dir).resolve()
    if output_dir == input_dir or output_dir.is_relative_to(input_dir):
        raise ValueError(
            "review outputs must be outside the original exported input folder"
        )
    if replicates < 1000:
        raise ValueError("use at least 1000 exploratory bootstrap resamples")
    analysis = input_dir / "analysis"
    plots = input_dir / "size-comparison"
    if not analysis.is_dir() or not plots.is_dir():
        raise ValueError(
            "input must be the original extraction with analysis/ and size-comparison/"
        )
    source_hashes = {
        str(path.relative_to(input_dir)): sha256(path)
        for path in sorted(input_dir.rglob("*"))
        if path.is_file()
    }
    rows = read_csv(analysis / "per_workload.csv")
    lookup, ids, operations = validate_rows(rows)
    summary = json.loads((analysis / "summary.json").read_text())
    method = json.loads((plots / "size-sweep-plot-method.json").read_text())
    if (
        summary["prepared_sha256"] != PREPARED_SHA
        or method["prepared_sha256"] != PREPARED_SHA
    ):
        raise ValueError(
            "declared prepared checksum differs from the audited E5 cohort"
        )
    hash_fields = {
        "analysis_summary_sha256": analysis / "summary.json",
        "by_candidate_count_sha256": analysis / "by_candidate_count.csv",
        "per_workload_sha256": analysis / "per_workload.csv",
        "plot_data_sha256": plots / "size-sweep-plot-data.csv",
    }
    for key, path in hash_fields.items():
        if method[key] != sha256(path):
            raise ValueError(f"original method checksum mismatch: {key}")
    if (
        method["candidate_budgets"] != list(SIZES)
        or method["workloads_per_budget"] != 120
        or method["overall_pooled_scores_used"] is not False
        or set(summary["metrics"]) != set(METRICS)
    ):
        raise ValueError("plot method scope differs from the controlled sweep")
    checks = {}
    for name, keys in (
        ("aggregate", ("model_id", "model_type", "candidate_count")),
        ("by_candidate_count", ("model_id", "model_type", "candidate_count")),
        ("by_operation", ("model_id", "model_type", "operation", "candidate_count")),
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
        ("overall", ("model_id", "model_type")),
    ):
        checks[name] = audit_aggregate(analysis / f"{name}.csv", rows, keys)
    plot_rows = read_csv(plots / "size-sweep-plot-data.csv")
    expected = summarize_groups(rows, ("candidate_count", "model_id", "model_type"))
    observed = {}
    for row in plot_rows:
        key = (row["candidate_count"], row["model_id"], row["model_type"])
        if key in observed or key not in expected:
            raise ValueError("plot data contains an invalid or duplicate budget/model")
        observed[key] = row
        if int(row["workloads"]) != 120:
            raise ValueError("plot data workload count differs")
        for metric in METRICS:
            close(row[metric], expected[key][metric], f"plot data {key} {metric}")
    if set(observed) != set(expected):
        raise ValueError("plot data omits a budget/model")
    for name, keys in (
        ("aggregates", ("model_id", "model_type", "candidate_count")),
        ("overall", ("model_id", "model_type")),
    ):
        expected = summarize_groups(rows, keys)
        actual = {tuple(str(row[k]) for k in keys): row for row in summary[name]}
        if set(actual) != set(expected):
            raise ValueError(f"summary JSON {name} group mismatch")
        for key, values in expected.items():
            for metric, value in values.items():
                close(actual[key][metric], value, f"summary {name} {metric}")
    accounting = audit_accounting(analysis / "decision_accounting.csv", lookup)
    historical = historical_check(historical_predictions, lookup)
    comparisons, contrasts, table, contrast_table, paired_rows = bootstrap(
        lookup, ids, replicates, seed
    )
    method_note = {
        "replicates": replicates,
        "seed": seed,
        "rng": "NumPy default_rng (PCG64); common index draws across all budgets",
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "workload_order": "lexicographic ascending workload_id",
        "workload_ids": ids,
        "bootstrap_unit": (
            "resample 120 workload pairs with replacement, same indices for 16/32/50"
        ),
        "aggregation": (
            "unweighted workload macro mean within each fixed candidate count"
        ),
        "interval": (
            "percentile 2.5–97.5%, exploratory and unadjusted for multiple comparisons"
        ),
        "contrasts": (
            "within each resample subtract the c50 Jev−Random delta from c16 or c32"
        ),
        "limitations": [
            (
                "Intervals condition on one Random seed and one saved Jev decision "
                "per workload/budget."
            ),
            (
                "True DAG/network-family dependence is not modeled; "
                "operation signatures are not independent families."
            ),
            (
                "Changing pool size also changes the candidate-pool oracle "
                "and Random permutation."
            ),
            (
                "The c50 anchor is historical; provider/time variation "
                "is not a controlled factor."
            ),
            (
                "No fresh API repeatability or out-of-sample generalization "
                "is established."
            ),
        ],
    }
    result = {
        "status": "passed_export_consistency_checks",
        "rows": len(rows),
        "workloads": len(ids),
        "candidate_budgets": list(SIZES),
        "model_rows_per_budget": 120,
        "operation_signatures": len(set(operations.values())),
        "operation_labels_consistent_across_budgets_and_models": True,
        "aggregate_checks": checks,
        "original_method_hash_fields_verified": list(hash_fields),
        "frozen_manifest_checksum_reference": method["manifest_sha256"],
        "frozen_manifest_uploaded": False,
        "historical_anchor_check": historical,
        "per_case_metric_identities_and_strict_pair_count_granularity_verified": True,
        "new_c16_c32_raw_predictions_available": False,
        "source_limitations": [
            (
                "The uploaded export does not include raw c16/c32 predictions, "
                "saved model configurations, API bodies or frozen sweep manifest."
            ),
            (
                "Raw new-run rankings, probabilities, labels, input hashes "
                "and actual candidate prefix nesting cannot be independently "
                "rechecked."
            ),
            (
                "Prepared-data SHA agrees with the historical audited cohort "
                "reference; the actual prepared file is not uploaded."
            ),
            (
                "Scores and usage are audited from derived tables, "
                "not a fresh hardware benchmark or model run."
            ),
            (
                "overall.csv pools 360 decisions per model from only 120 distinct "
                "workloads; it is not a candidate-size comparison."
            ),
        ],
        "operation_signature_counts": dict(
            sorted(Counter(operations.values()).items())
        ),
        "source_sha256": source_hashes,
    }
    if source_hashes != {
        str(path.relative_to(input_dir)): sha256(path)
        for path in sorted(input_dir.rglob("*"))
        if path.is_file()
    }:
        raise ValueError("source export changed during audit")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "audit-summary.json", result)
    write_json(
        output_dir / "paired-statistics.json",
        {
            "method": method_note,
            "by_budget": comparisons,
            "between_budget_contrasts": contrasts,
        },
    )
    write_json(output_dir / "usage-summary.json", accounting)
    write_csv(output_dir / "paired-comparison.csv", table)
    write_csv(output_dir / "between-budget-contrasts.csv", contrast_table)
    write_csv(output_dir / "paired-differences.csv", paired_rows)
    print(
        json.dumps(
            {
                "audit_status": result["status"],
                "rows": len(rows),
                "workloads": len(ids),
                "new_c16_c32_input_estimate_usd": accounting[
                    "new_c16_c32_input_estimate_usd"
                ],
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--historical-predictions", type=Path)
    parser.add_argument("--replicates", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20261116)
    args = parser.parse_args()
    audit(
        args.input_dir,
        args.output_dir,
        historical_predictions=args.historical_predictions,
        replicates=args.replicates,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
