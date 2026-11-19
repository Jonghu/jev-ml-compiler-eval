#!/usr/bin/env python3
"""Explore DAG-hash cluster sensitivity using saved TenSet sweep scores.

Requires Python 3.10+ and NumPy only. Does not call model APIs or load credentials.
The family map is the historical paired-workloads.csv, used only for its IDs and
dag_hash_family column. Hash groups are not verified independent network families.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

SIZES = (16, 32, 50)
MODELS = ("random-42", "jev-1.14.0-long-context")
METRICS = (
    "top_1_score",
    "top_5_score",
    "pairwise_accuracy",
    "runtime_regret",
)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def calculate(per_workload, family_map, output_dir, replicates, seed):
    source_hashes = {
        "per_workload": sha256(per_workload),
        "family_map": sha256(family_map),
    }
    rows = read_csv(per_workload)
    lookup = {}
    for row in rows:
        key = (int(row["candidate_count"]), row["model_id"], row["workload_id"])
        if key in lookup:
            raise ValueError(f"duplicate score row: {key}")
        lookup[key] = row
    ids = sorted({key[2] for key in lookup})
    expected = {(n, model, wid) for n in SIZES for model in MODELS for wid in ids}
    if len(ids) != 120 or len(rows) != 720 or set(lookup) != expected:
        raise ValueError(
            "requires exactly 120 matched workloads x 3 budgets x 2 models"
        )
    mapping = {}
    for row in read_csv(family_map):
        wid, family = row["workload_id"], row["dag_hash_family"]
        if wid in mapping or not family:
            raise ValueError(
                "family map requires unique workload IDs and nonempty groups"
            )
        mapping[wid] = family
    groups = sorted(set(mapping.values()))
    if set(mapping) != set(ids) or len(groups) != 56:
        raise ValueError(
            "family map must match all 120 workload IDs and 56 hash groups"
        )
    differences = np.array(
        [
            [
                [
                    float(lookup[n, MODELS[1], wid][metric])
                    - float(lookup[n, MODELS[0], wid][metric])
                    for metric in METRICS
                ]
                for n in SIZES
            ]
            for wid in ids
        ]
    )
    if not np.isfinite(differences).all():
        raise ValueError("non-finite metric differences")
    members = [[i for i, wid in enumerate(ids) if mapping[wid] == g] for g in groups]
    counts = np.array([len(indices) for indices in members])
    sums = np.array([differences[indices].sum(axis=0) for indices in members])
    points = differences.mean(axis=0)
    samples = np.empty((replicates, len(SIZES), len(METRICS)))
    denominator_range = [len(ids), len(ids)]
    rng = np.random.default_rng(seed)
    for start in range(0, replicates, 2000):
        indices = rng.integers(
            0, len(groups), size=(min(2000, replicates - start), len(groups))
        )
        denominator = counts[indices].sum(axis=1)
        denominator_range[0] = min(denominator_range[0], int(denominator.min()))
        denominator_range[1] = max(denominator_range[1], int(denominator.max()))
        samples[start : start + len(indices)] = (
            sums[indices].sum(axis=1) / denominator[:, None, None]
        )
    results, table = {}, []
    for index, size in enumerate(SIZES):
        metrics = {}
        for metric_index, metric in enumerate(METRICS):
            low, high = np.quantile(samples[:, index, metric_index], [0.025, 0.975])
            value = {
                "mean_delta_jev_minus_random": float(points[index, metric_index]),
                "bootstrap_95ci": [float(low), float(high)],
                "lower_is_better": metric == "runtime_regret",
            }
            metrics[metric] = value
            table.append(
                {
                    "candidates": size,
                    "metric": metric,
                    "jev_minus_random": value["mean_delta_jev_minus_random"],
                    "ci_low": float(low),
                    "ci_high": float(high),
                }
            )
        results[str(size)] = metrics
    contrasts, contrast_table = {}, []
    for index, size in enumerate(SIZES[:2]):
        metrics = {}
        for metric_index, metric in enumerate(METRICS):
            delta = points[index, metric_index] - points[2, metric_index]
            low, high = np.quantile(
                samples[:, index, metric_index] - samples[:, 2, metric_index],
                [0.025, 0.975],
            )
            metrics[metric] = {
                "difference_of_jev_minus_random_deltas": float(delta),
                "bootstrap_95ci": [float(low), float(high)],
            }
            contrast_table.append(
                {
                    "contrast": f"{size}-50",
                    "metric": metric,
                    "difference_of_jev_minus_random_deltas": float(delta),
                    "ci_low": float(low),
                    "ci_high": float(high),
                }
            )
        contrasts[f"{size}-50"] = metrics
    result = {
        "method": (
            "Exploratory percentile cluster bootstrap: resample 56 leading "
            "DAG-hash groups with replacement; retain all member workloads; "
            "divide summed paired metric differences by summed workload counts. "
            "Use the same group-index draws across all three budgets."
        ),
        "replicates": replicates,
        "seed": seed,
        "rng": "NumPy default_rng (PCG64)",
        "numpy_version": np.__version__,
        "workloads": len(ids),
        "clusters": len(groups),
        "largest_cluster_workloads": int(counts.max()),
        "bootstrap_replicate_workload_count_range": denominator_range,
        "workload_order": "lexicographic ascending workload_id",
        "cluster_order": "lexicographic ascending dag_hash_family",
        "source_sha256": source_hashes,
        "source_names": {
            "per_workload": per_workload.name,
            "family_map": family_map.name,
        },
        "workload_to_hash_group": dict(sorted(mapping.items())),
        "group_workload_counts": dict(sorted(Counter(mapping.values()).items())),
        "limitations": [
            "Intervals are exploratory and unadjusted for multiple comparisons.",
            (
                "Leading-hash groups are not verified independent program or "
                "network families; this is a sensitivity analysis only."
            ),
            (
                "Conditions on one Random seed and one saved Jev outcome per "
                "workload/budget; c50 is a historical reused anchor."
            ),
            (
                "New c16/c32 raw predictions, labels, candidate pools and saved "
                "configurations were not uploaded for independent verification."
            ),
            (
                "Candidate-pool optima change with budget; normalized-score "
                "changes do not directly measure selected-runtime changes."
            ),
        ],
        "per_budget": results,
        "between_budget_contrasts": contrasts,
    }
    if source_hashes != {
        "per_workload": sha256(per_workload),
        "family_map": sha256(family_map),
    }:
        raise ValueError("source inputs changed during calculation")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "family-cluster-sensitivity.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    save_csv(output_dir / "family-cluster-comparison.csv", table)
    save_csv(output_dir / "family-cluster-between-budget-contrasts.csv", contrast_table)
    print(json.dumps({"per_budget": results, "contrasts": contrasts}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-workload", type=Path, required=True)
    parser.add_argument("--family-map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20261116)
    args = parser.parse_args()
    if args.replicates < 1000:
        parser.error("--replicates must be at least 1000")
    if args.output_dir.resolve() in {
        args.per_workload.resolve().parent,
        args.family_map.resolve().parent,
    }:
        parser.error("--output-dir must differ from input file directories")
    calculate(
        args.per_workload,
        args.family_map,
        args.output_dir,
        args.replicates,
        args.seed,
    )


if __name__ == "__main__":
    main()
