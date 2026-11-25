"""Summarize the TpuGraphs context-engineering ablation (A0-A5) into tables.

Reads every ``predictions.jsonl`` below ``<root>/runs``, groups repeated models
(``<variant>-rep<n>``) into variants, and adds offline heuristic baselines on
the same kernels. Writes ``<root>/summary/``:

- ``variants.csv``: per variant, the mean and standard deviation over repeats
  of each run-level metric, plus pick diagnostics, tokens, and latency;
- ``runs.csv``: the same metrics per run (one variant repeat);
- ``per_kernel.csv``: every pick of every run, with its Top-1 error;
- ``baselines.csv``: offline heuristics on the same kernels;
- ``pairs.csv``: paired per-kernel comparisons of variants (sign test);
- ``summary.md``: the variant and baseline tables in Markdown.
"""

import argparse
import csv
import json
import math
import re
import statistics
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from jev_compiler_eval.datasets.tpugraphs import TpuGraphsTileDataset
from jev_compiler_eval.types import EvaluationCase

METRICS = (
    "top_1_error",
    "top_1_accuracy",
    "top_5_error",
    "top_10_error",
    "kendall_tau",
)
SIZE_FIELDS = ("output_bounds", "kernel_bounds")
DEFAULT_CANDIDATE_ID = "config_0"
# (baseline variant, compared variant) pairs of the A0-A5 plan.
PAIRS = (
    ("a0-naive", "a1-config-fields"),
    ("a0-naive", "a2-graph-fields"),
    ("a0-naive", "a3-both-fields"),
    ("a1-config-fields", "a3-both-fields"),
    ("a2-graph-fields", "a3-both-fields"),
    ("a3-both-fields", "a4-size-ascending"),
    ("a3-both-fields", "a4-size-descending"),
    ("a4-size-ascending", "a4-size-descending"),
    ("a3-both-fields", "a5-keep-largest-10"),
)
_REPEAT = re.compile(r"^(?P<variant>.+)-rep(?P<repeat>\d+)$")


def split_model_id(model_id: str) -> tuple[str, int | None]:
    """Return the variant name and repeat number of a model ID."""

    match = _REPEAT.match(model_id)
    if match is None:
        return model_id, None
    return match["variant"], int(match["repeat"])


def tile_size(configuration: Mapping[str, object]) -> int:
    """Return the product of the output and kernel bound entries."""

    size = 1
    for name in SIZE_FIELDS:
        for value in configuration.get(name, []):
            size *= int(value)
    return size


def mean_sd(values: Sequence[float]) -> tuple[float, float]:
    """Return the mean and sample standard deviation (zero for one value)."""

    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, sd


def read_runs(runs_dir: Path) -> dict[str, list[dict]]:
    """Return prediction records keyed by model ID, from every run bundle."""

    records: dict[str, list[dict]] = {}
    for path in sorted(runs_dir.glob("*/predictions.jsonl")):
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    record = json.loads(line)
                    records.setdefault(record["model_id"], []).append(record)
    return records


def run_summary(records: Sequence[dict], cases: Mapping[str, EvaluationCase]) -> dict:
    """Return run-level metrics and pick diagnostics of one model."""

    summary: dict[str, float] = {"kernels": len(records)}
    for metric in METRICS:
        summary[metric] = statistics.fmean(r["metrics"][metric] for r in records)
    top_1 = [r["metrics"]["top_1_error"] for r in records]
    summary["top_1_error_median"] = statistics.median(top_1)
    smallest = largest = default = 0
    for record in records:
        case = cases[record["workload_id"]]
        sizes = {c.candidate_id: tile_size(c.configuration) for c in case.candidates}
        picked = sizes[record["selected_candidate_id"]]
        smallest += picked == min(sizes.values())
        largest += picked == max(sizes.values())
        default += record["selected_candidate_id"] == DEFAULT_CANDIDATE_ID
    summary["picks_smallest_tile"] = smallest
    summary["picks_largest_tile"] = largest
    summary["picks_config_0"] = default
    metadata = [r.get("metadata", {}) for r in records]
    summary["input_tokens"] = sum(m.get("input_tokens", 0) or 0 for m in metadata)
    summary["output_tokens"] = sum(m.get("output_tokens", 0) or 0 for m in metadata)
    summary["live_requests"] = sum(m.get("live_requests", 0) or 0 for m in metadata)
    summary["mean_api_latency_ms"] = statistics.fmean(
        m.get("api_latency_ms", 0.0) or 0.0 for m in metadata
    )
    latencies = [r["decision_latency_ms"] for r in records]
    summary["mean_decision_latency_ms"] = statistics.fmean(latencies)
    summary["median_decision_latency_ms"] = statistics.median(latencies)
    summary["max_decision_latency_ms"] = max(latencies)
    return summary


def sign_test(wins: int, losses: int) -> float:
    """Return the two-sided exact sign-test p-value, ignoring ties."""

    trials = wins + losses
    if trials == 0:
        return 1.0
    tail = sum(math.comb(trials, k) for k in range(min(wins, losses) + 1))
    return min(1.0, 2 * tail / 2**trials)


def paired_comparison(
    left: Mapping[str, float], right: Mapping[str, float]
) -> dict[str, float]:
    """Compare per-kernel errors; "better" means the right variant is lower."""

    shared = sorted(set(left) & set(right))
    better = sum(right[w] < left[w] for w in shared)
    worse = sum(right[w] > left[w] for w in shared)
    return {
        "kernels": len(shared),
        "right_better": better,
        "right_worse": worse,
        "ties": len(shared) - better - worse,
        "mean_difference": statistics.fmean(right[w] - left[w] for w in shared),
        "sign_test_p": sign_test(better, worse),
    }


def pick_agreement(runs: Iterable[Sequence[dict]]) -> float:
    """Return the share of kernels where every repeat picked the same candidate."""

    picks: dict[str, set[str]] = {}
    for records in runs:
        for record in records:
            picks.setdefault(record["workload_id"], set()).add(
                record["selected_candidate_id"]
            )
    if not picks:
        return math.nan
    return sum(len(chosen) == 1 for chosen in picks.values()) / len(picks)


def _pick_scores(
    runtime: Mapping[str, float], best: float, pool: Sequence[str]
) -> tuple[float, float, float, float]:
    """Return expected-random error, oracle error, and their optimal shares."""

    optimal = [cid for cid in pool if runtime[cid] == best]
    expected = statistics.fmean(runtime[cid] for cid in pool) / best - 1
    oracle = min(runtime[cid] for cid in pool) / best - 1
    return expected, len(optimal) / len(pool), oracle, float(bool(optimal))


def heuristic_baselines(
    cases: Iterable[EvaluationCase],
    keep_largest: int,
    shown: Mapping[str, Sequence[str]] | None = None,
) -> list[dict[str, object]]:
    """Return offline Top-1 errors of simple rules on the given kernels.

    Each row holds per-kernel (error, probability of picking an optimum).
    "Expected" rows average a uniform random pick over the pool. When
    ``shown`` maps workload IDs to the candidates a filtered run actually
    showed Jev, the filter rows use exactly those pools; otherwise the pool is
    recomputed with equal sizes kept in dataset order.
    """

    filter_random = f"largest-{keep_largest} filter + random pick (expected)"
    filter_oracle = f"largest-{keep_largest} filter + oracle pick"
    rows: dict[str, list[tuple[float, float]]] = {
        "compiler default (config_0)": [],
        "largest tile": [],
        filter_random: [],
        filter_oracle: [],
        "random pick (expected)": [],
    }
    for case in cases:
        runtime = {r.candidate_id: r.runtime_ms for r in case.runtimes}
        best = min(runtime.values())
        sizes = {c.candidate_id: tile_size(c.configuration) for c in case.candidates}
        order = [c.candidate_id for c in case.candidates]
        by_size = sorted(order, key=lambda cid: -sizes[cid])
        if shown is not None and case.workload.workload_id in shown:
            kept = list(shown[case.workload.workload_id])
        else:
            kept = by_size[:keep_largest]
        default = runtime[DEFAULT_CANDIDATE_ID]
        rows["compiler default (config_0)"].append(
            (default / best - 1, float(default == best))
        )
        largest = runtime[by_size[0]]
        rows["largest tile"].append((largest / best - 1, float(largest == best)))
        expected, expected_share, oracle, oracle_share = _pick_scores(
            runtime, best, kept
        )
        rows[filter_random].append((expected, expected_share))
        rows[filter_oracle].append((oracle, oracle_share))
        expected, expected_share, _, _ = _pick_scores(runtime, best, order)
        rows["random pick (expected)"].append((expected, expected_share))
    return [
        {
            "baseline": name,
            "kernels": len(values),
            "top_1_error": statistics.fmean(error for error, _ in values),
            "top_1_error_median": statistics.median(error for error, _ in values),
            "optimal_share": statistics.fmean(share for _, share in values),
        }
        for name, values in rows.items()
    ]


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _percent(value: float) -> str:
    return f"{100 * value:.1f}"


def summarize(root: Path, dataset_dir: Path, keep_largest: int) -> Path:
    """Write the ablation summary tables and return the summary directory."""

    records = read_runs(root / "runs")
    if not records:
        raise ValueError(f'no predictions.jsonl found below "{root / "runs"}"')
    dataset = TpuGraphsTileDataset({"path": str(dataset_dir), "max_candidates": 50})
    cases = {case.workload.workload_id: case for case in dataset.load()}

    runs: list[dict[str, object]] = []
    per_kernel: list[dict[str, object]] = []
    by_variant: dict[str, list[tuple[int | None, list[dict]]]] = {}
    for model_id, model_records in records.items():
        variant, repeat = split_model_id(model_id)
        by_variant.setdefault(variant, []).append((repeat, model_records))
        runs.append(
            {"variant": variant, "repeat": repeat, "model_id": model_id}
            | run_summary(model_records, cases)
        )
        for record in model_records:
            per_kernel.append(
                {
                    "variant": variant,
                    "repeat": repeat,
                    "workload_id": record["workload_id"],
                    "candidates": len(record["candidate_ids"]),
                    "selected_candidate_id": record["selected_candidate_id"],
                    "optimal_candidate_ids": " ".join(record["optimal_candidate_ids"]),
                    "top_1_error": record["metrics"]["top_1_error"],
                }
            )

    variants: list[dict[str, object]] = []
    for variant, repeats in by_variant.items():
        summaries = [run for run in runs if run["variant"] == variant]
        row: dict[str, object] = {"variant": variant, "repeats": len(repeats)}
        for key in (
            *METRICS,
            "top_1_error_median",
            "picks_smallest_tile",
            "picks_largest_tile",
            "picks_config_0",
            "mean_api_latency_ms",
            "median_decision_latency_ms",
        ):
            mean, sd = mean_sd([float(s[key]) for s in summaries])
            row[f"{key}_mean"] = mean
            row[f"{key}_sd"] = sd
        row["pick_agreement_across_repeats"] = pick_agreement(r for _, r in repeats)
        row["input_tokens_total"] = sum(int(s["input_tokens"]) for s in summaries)
        variants.append(row)
    variants.sort(key=lambda row: str(row["variant"]))
    # Filter baselines use the pools the filtered Jev run actually showed.
    shown = {
        record["workload_id"]: record["metadata"]["presented_candidate_ids"]
        for model_id, model_records in records.items()
        if split_model_id(model_id)[0].startswith(f"a5-keep-largest-{keep_largest}")
        for record in model_records
    }
    baselines = heuristic_baselines(cases.values(), keep_largest, shown or None)

    # Per-kernel Top-1 error averaged over the repeats of each variant.
    kernel_error: dict[str, dict[str, float]] = {}
    for variant, repeats in by_variant.items():
        values: dict[str, list[float]] = {}
        for _, model_records in repeats:
            for record in model_records:
                values.setdefault(record["workload_id"], []).append(
                    record["metrics"]["top_1_error"]
                )
        kernel_error[variant] = {w: statistics.fmean(v) for w, v in values.items()}
    pairs = [
        {"left": left, "right": right}
        | paired_comparison(kernel_error[left], kernel_error[right])
        for left, right in PAIRS
        if left in kernel_error and right in kernel_error
    ]

    out = root / "summary"
    out.mkdir(parents=True, exist_ok=True)
    if pairs:
        _write_csv(out / "pairs.csv", pairs)
    _write_csv(out / "variants.csv", variants)
    _write_csv(out / "runs.csv", runs)
    _write_csv(out / "per_kernel.csv", per_kernel)
    _write_csv(out / "baselines.csv", baselines)

    lines = [
        "# TpuGraphs ablation summary",
        "",
        "Errors are percentages, mean over kernels; ± is the standard deviation "
        "over repeats.",
        "",
        "| Variant | Repeats | Top-1 err | Top-1 median | Top-1 acc | Top-5 err "
        "| Kendall τ | Smallest-tile picks | Largest-tile picks | Same pick in "
        "all repeats |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in variants:
        lines.append(
            f"| {row['variant']} | {row['repeats']} "
            f"| {_percent(row['top_1_error_mean'])} "
            f"± {_percent(row['top_1_error_sd'])} "
            f"| {_percent(row['top_1_error_median_mean'])} "
            f"| {row['top_1_accuracy_mean']:.3f} "
            f"| {_percent(row['top_5_error_mean'])} "
            f"| {row['kendall_tau_mean']:+.3f} ± {row['kendall_tau_sd']:.3f} "
            f"| {row['picks_smallest_tile_mean']:.1f} "
            f"| {row['picks_largest_tile_mean']:.1f} "
            f"| {row['pick_agreement_across_repeats']:.2f} |"
        )
    lines += [
        "",
        "Offline heuristics on the same kernels (no model):",
        "",
        "| Baseline | Kernels | Top-1 err | Top-1 median | P(optimal pick) |",
        "|---|---|---|---|---|",
    ]
    for row in baselines:
        lines.append(
            f"| {row['baseline']} | {row['kernels']} "
            f"| {_percent(row['top_1_error'])} "
            f"| {_percent(row['top_1_error_median'])} "
            f"| {row['optimal_share']:.3f} |"
        )
    lines += [
        "",
        "Paired per-kernel comparisons (Top-1 error averaged over repeats; "
        '"better" means the right variant has lower error):',
        "",
        "| Left | Right | Better | Worse | Ties | Mean change (pp) | Sign-test p |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in pairs:
        lines.append(
            f"| {row['left']} | {row['right']} | {row['right_better']} "
            f"| {row['right_worse']} | {row['ties']} "
            f"| {100 * row['mean_difference']:+.1f} | {row['sign_test_p']:.3g} |"
        )
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("results/tpugraphs/ablation-small")
    )
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/tpugraphs/npz/tile/xla/valid")
    )
    parser.add_argument("--keep-largest", type=int, default=10)
    args = parser.parse_args()
    out = summarize(args.root, args.dataset, args.keep_largest)
    print(f"Ablation summary saved to {out}")


if __name__ == "__main__":
    main()
