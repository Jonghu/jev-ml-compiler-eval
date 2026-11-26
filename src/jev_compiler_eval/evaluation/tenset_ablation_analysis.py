"""Offline summaries and explicit raw-result archives for TenSet ablations.

The endpoint scores below are deliberately separate from TenSet's mean-prefix
Top-5 score. Filtering always retains the original candidate-pool denominator;
the unshown tail is never treated as a learned ranking.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import zipfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from random import Random
from statistics import mean, median, stdev

PAIRS = (
    ("a0", "a1"),
    ("a0", "a2"),
    ("a0", "a3"),
    ("a1", "a3"),
    ("a2", "a3"),
    ("a3", "a4-asc"),
    ("a3", "a4-desc"),
    ("a4-asc", "a4-desc"),
    ("a3", "a5"),
)
METRICS = (
    "top_1_score",
    "top_5_score",
    "endpoint_top_1_score",
    "endpoint_top_5_score",
    "top_1_error",
    "top_5_error",
    "runtime_regret",
    "top_1_accuracy",
    "pairwise_accuracy",
    "pairwise_accuracy_tie_aware",
    "paper_network_top_1_score",
    "paper_network_top_5_score",
    "pilot_network_top_1_score",
    "pilot_network_top_5_score",
)


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _number(value: object, name: str, *, positive: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or (value <= 0 if positive else value < 0)
    ):
        kind = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be a finite {kind} number")
    return float(value)


def _ids(value: object, name: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item.strip() for item in value)
        or len(value) != len(set(value))
    ):
        raise ValueError(f"{name} must contain distinct nonempty candidate IDs")
    return value


def _validate_row(row: dict) -> dict:
    for key in ("phase", "variant", "workload_id"):
        if not isinstance(row.get(key), str) or not row[key]:
            raise ValueError(f"prediction {key} must be a nonempty string")
    if type(row.get("repeat")) is not int or row["repeat"] < 1:
        raise ValueError("prediction repeat must be a positive integer")
    candidates = _ids(row.get("candidate_ids"), "candidate_ids")
    shown = _ids(row.get("shown_candidate_ids"), "shown_candidate_ids")
    ranking = _ids(row.get("ranking"), "ranking")
    if set(ranking) != set(candidates) or not set(shown) <= set(candidates):
        raise ValueError("ranking and shown candidates must match the original pool")
    if set(ranking[: len(shown)]) != set(shown):
        raise ValueError("unshown candidates must rank after all shown candidates")
    if len(shown) < min(5, len(candidates)):
        raise ValueError("Top-5 evaluation cannot be filled by unshown candidates")
    raw = row.get("runtimes_ms")
    if not isinstance(raw, dict) or set(raw) != set(candidates):
        raise ValueError("runtimes_ms must match the original candidate pool")
    row = dict(row)
    row["runtimes_ms"] = {
        key: _number(value, "runtime", positive=True) for key, value in raw.items()
    }
    row["task_weight"] = _number(
        row.get("task_weight", 1), "task_weight", positive=True
    )
    row["decision_latency_ms"] = _number(
        row.get("decision_latency_ms", 0), "decision latency"
    )
    if "paper_normalizer_ms" in row:
        row["paper_normalizer_ms"] = _number(
            row["paper_normalizer_ms"], "paper normalizer", positive=True
        )
    if not isinstance(row.get("request_reports", []), list):
        raise ValueError("request_reports must be a list")
    row["metrics"] = _row_metrics(row)
    return row


def _reference(row: Mapping) -> float:
    return row.get("paper_normalizer_ms", min(row["runtimes_ms"].values()))


def _row_metrics(row: Mapping) -> dict[str, float]:
    runtimes, ranking = row["runtimes_ms"], row["ranking"]
    reference = _reference(row)
    prefixes = []
    best = math.inf
    for candidate in ranking[:5]:
        best = min(best, runtimes[candidate])
        prefixes.append(reference / best)
    top1 = reference / runtimes[ranking[0]]
    metrics = {
        "top_1_score": top1,
        "top_5_score": mean(prefixes),
        "endpoint_top_1_score": top1,
        "endpoint_top_5_score": prefixes[-1],
        "top_1_error": runtimes[ranking[0]] / reference - 1,
        "top_5_error": best / reference - 1,
        "runtime_regret": runtimes[ranking[0]] / reference - 1,
        "top_1_accuracy": float(runtimes[ranking[0]] == min(runtimes.values())),
    }
    # Probability ties are meaningful only inside the model's actual question.
    # Do not derive pairwise quality from a heuristic filtered tail.
    if len(row["shown_candidate_ids"]) == len(row["candidate_ids"]):
        saved = row.get("metrics", {})
        if isinstance(saved, Mapping):
            for key in ("pairwise_accuracy", "pairwise_accuracy_tie_aware"):
                if key in saved:
                    value = _number(saved[key], key)
                    if value > 1:
                        raise ValueError(
                            "pairwise metrics must be between zero and one"
                        )
                    metrics[key] = value
    return metrics


def expected_random_metrics(
    runtimes_ms: Mapping[str, float],
    shown_candidate_ids: Sequence[str] | None = None,
    *,
    reference_ms: float | None = None,
) -> dict[str, float]:
    """Exact uniform-without-replacement expectations on a fixed visible pool.

    The reference remains the original pool minimum (or the supplied private
    paper normalizer). This is not a draw and uses no model/API call.
    """
    runtimes = {
        key: _number(value, "runtime", positive=True)
        for key, value in runtimes_ms.items()
    }
    shown = (
        list(shown_candidate_ids) if shown_candidate_ids is not None else list(runtimes)
    )
    if (
        not runtimes
        or not shown
        or len(shown) != len(set(shown))
        or not set(shown) <= set(runtimes)
    ):
        raise ValueError(
            "random baseline requires a nonempty subset of the original pool"
        )
    reference = (
        min(runtimes.values())
        if reference_ms is None
        else _number(reference_ms, "reference", positive=True)
    )
    values = sorted(runtimes[candidate] for candidate in shown)
    count = len(values)

    def expected_best(k: int, *, score: bool) -> float:
        k = min(k, count)
        denominator = math.comb(count, k)
        return sum(
            (reference / value if score else value / reference - 1)
            * (math.comb(count - index - 1, k - 1) / denominator)
            for index, value in enumerate(values[: count - k + 1])
        )

    return {
        "top_1_score": expected_best(1, score=True),
        "top_5_score": mean(
            expected_best(k, score=True) for k in range(1, min(5, count) + 1)
        ),
        "endpoint_top_1_score": expected_best(1, score=True),
        "endpoint_top_5_score": expected_best(5, score=True),
        "top_1_error": expected_best(1, score=False),
        "top_5_error": expected_best(5, score=False),
        "runtime_regret": expected_best(1, score=False),
        "top_1_accuracy": sum(value == min(runtimes.values()) for value in values)
        / count,
        "retained_optimum": float(values[0] == min(runtimes.values())),
        "oracle_top_1_score": reference / values[0],
        "oracle_top_1_error": values[0] / reference - 1,
    }


def _usage(row: Mapping) -> dict:
    result = {
        key: 0
        for key in (
            "live_requests",
            "cached_requests",
            "live_input_tokens",
            "cached_input_tokens",
            "live_output_tokens",
            "cached_output_tokens",
            "unknown_input_usage_requests",
        )
    }
    for report in row.get("request_reports", []):
        if not isinstance(report, Mapping) or type(report.get("cache_hit")) is not bool:
            raise ValueError("request reports require a boolean cache_hit")
        kind = "cached" if report["cache_hit"] else "live"
        result[f"{kind}_requests"] += 1
        for token in ("input", "output"):
            value = report.get(f"{token}_tokens")
            if value is None:
                if token == "input":
                    result["unknown_input_usage_requests"] += 1
                continue
            if type(value) is not int or value < 0:
                raise ValueError(
                    "request usage must contain nonnegative integer tokens"
                )
            result[f"{kind}_{token}_tokens"] += value
    return result


def _write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _local_file(root: Path, path: Path) -> Path:
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError("result paths must remain inside the local study")
    return path


def _audit_prediction(root: Path, study: Mapping, raw: Mapping) -> None:
    """Bind scientific rows to frozen labels/inputs and immutable local receipts.

    Archives may be reviewed on another computer, so no original absolute Mac
    data path or matching current implementation tree is needed for this check.
    """
    if "identity" not in study:
        return  # Synthetic unit fixtures have no execution identity.
    phase, variant = raw["phase"], raw["variant"]
    settings = study.get("phases", {}).get(phase, {})
    descriptors = {case["workload_id"]: case for case in settings.get("cases", [])}
    descriptor = descriptors.get(raw["workload_id"])
    if descriptor is None or variant not in settings.get("variants", []):
        raise ValueError("prediction is absent from the frozen study inventory")
    ids = descriptor["candidate_ids"]
    plan = descriptor.get("plans", {}).get(variant, {})
    if (
        raw["candidate_ids"] != ids
        or _fingerprint(
            [[candidate, raw["runtimes_ms"][candidate]] for candidate in ids]
        )
        != descriptor.get("labels_sha256")
        or raw["shown_candidate_ids"] != plan.get("shown_candidate_ids")
        or raw.get("model_input_sha256") != plan.get("model_input_sha256")
        or any(
            raw.get(key) != descriptor.get(key)
            for key in ("task_weight", "runtime_basis", "paper_normalizer_ms")
        )
    ):
        raise ValueError(
            "prediction labels, private fields or inputs differ from study"
        )
    receipt_path = _local_file(
        root,
        root
        / "receipts"
        / phase
        / variant
        / f"rep{raw['repeat']}"
        / f"{_fingerprint(raw['workload_id'])}.json",
    )
    if not receipt_path.is_file():
        raise ValueError("scientific prediction has no immutable completion receipt")
    receipt = _json(receipt_path)
    if (
        receipt.get("study_identity") != study["identity"]
        or receipt.get("prediction_sha256") != _fingerprint(raw)
        or receipt.get("prediction") != raw
    ):
        raise ValueError("prediction disagrees with its immutable completion receipt")


def _load_rows(
    root: Path, study: Mapping, phase: str | None, allow_incomplete: bool
) -> list[dict]:
    rows, fingerprints = [], {}

    def append(raw: object, *, receipt_only: bool = False) -> None:
        if not isinstance(raw, dict):
            raise ValueError("predictions must be JSON objects")
        if phase is not None and raw.get("phase") != phase:
            return
        row = _validate_row(raw)
        key = row["phase"], row["variant"], row["repeat"], row["workload_id"]
        digest = _fingerprint(raw)
        if key in fingerprints:
            if receipt_only and fingerprints[key] == digest:
                return
            raise ValueError("duplicate or conflicting ablation predictions")
        _audit_prediction(root, study, raw)
        fingerprints[key] = digest
        rows.append(row)

    for path in sorted(root.rglob("predictions.jsonl")):
        _local_file(root, path)
        content = path.read_bytes()
        raw_rows = [
            json.loads(line)
            for line in content.decode("utf-8").splitlines()
            if line.strip()
        ]
        selected = [
            row
            for row in raw_rows
            if isinstance(row, dict) and (phase is None or row.get("phase") == phase)
        ]
        if not selected:
            continue
        if "identity" in study:
            run_path = _local_file(root, path.parent / "run.json")
            if not run_path.is_file():
                raise ValueError("scientific run is missing run.json")
            run = _json(run_path)
            if (
                run.get("study_identity") != study["identity"]
                or run.get("predictions_sha256") != hashlib.sha256(content).hexdigest()
                or run.get("workloads") != len(raw_rows)
                or any(
                    any(
                        run.get(key) != row.get(key)
                        for key in ("phase", "variant", "repeat")
                    )
                    for row in selected
                )
            ):
                raise ValueError("run identity or predictions checksum differs")
        for raw in raw_rows:
            append(raw)
    if allow_incomplete:
        for path in sorted((root / "receipts").rglob("*.json")):
            _local_file(root, path)
            receipt = _json(path)
            raw = receipt.get("prediction")
            if not isinstance(raw, dict):
                raise ValueError("completion receipt is missing its prediction")
            if receipt.get("prediction_sha256") != _fingerprint(raw):
                raise ValueError("completion receipt prediction checksum differs")
            append(raw, receipt_only=True)
    return rows


def _coverage(
    rows: Sequence[dict], study: Mapping, allow_incomplete: bool
) -> list[str]:
    warnings = []
    repeats = study.get("repeats", 3)
    if type(repeats) is not int or repeats < 1:
        repeats = 3
    by_variant = defaultdict(dict)
    for row in rows:
        by_variant[row["phase"], row["variant"]].setdefault(row["repeat"], set()).add(
            row["workload_id"]
        )
    for (phase, variant), runs in sorted(by_variant.items()):
        if (
            set(runs) != set(range(1, repeats + 1))
            or len({frozenset(values) for values in runs.values()}) != 1
        ):
            warnings.append(
                f"{phase}/{variant}: missing repeats or unequal workload coverage"
            )
    selected_phases = {row["phase"] for row in rows}
    declared_phases = study.get("phases", {})
    for phase in selected_phases:
        settings = declared_phases.get(phase, {})
        if declared_phases and phase not in declared_phases:
            raise ValueError("prediction phase is absent from the frozen study")
        expected_variants = set(settings.get("variants", []))
        observed_variants = {variant for name, variant in by_variant if name == phase}
        if expected_variants and observed_variants - expected_variants:
            raise ValueError("prediction variant is absent from the frozen study")
        if expected_variants - observed_variants:
            missing = sorted(expected_variants - observed_variants)
            warnings.append(f"{phase}: missing variants {missing}")
        expected_cases = {
            case["workload_id"]: case for case in settings.get("cases", [])
        }
        for variant in observed_variants:
            observed_cases = {
                row["workload_id"]
                for row in rows
                if row["phase"] == phase and row["variant"] == variant
            }
            if expected_cases and observed_cases - set(expected_cases):
                raise ValueError("prediction workload is absent from the frozen study")
            if expected_cases and observed_cases != set(expected_cases):
                warnings.append(
                    f"{phase}/{variant}: incomplete frozen workload inventory"
                )
        for row in rows:
            if row["phase"] != phase or not expected_cases:
                continue
            expected = expected_cases[row["workload_id"]]
            if (
                expected.get("candidate_ids", row["candidate_ids"])
                != row["candidate_ids"]
            ):
                raise ValueError(
                    "prediction candidate pool differs from frozen inventory"
                )
            if expected.get("task_weight", row["task_weight"]) != row["task_weight"]:
                raise ValueError("prediction task weight differs from frozen inventory")
            if expected.get(
                "paper_normalizer_ms", row.get("paper_normalizer_ms")
            ) != row.get("paper_normalizer_ms"):
                raise ValueError("paper normalizer differs from frozen inventory")
    # Separate phases may legitimately have different inventories, but comparisons
    # within a phase must cover the same cases, labels, and source candidate order.
    pools = {}
    for row in rows:
        key = row["phase"], row["workload_id"]
        value = (
            row["candidate_ids"],
            row["runtimes_ms"],
            row["task_weight"],
            row.get("paper_normalizer_ms"),
        )
        if key in pools and pools[key] != value:
            raise ValueError(
                "ablation conditions disagree on original pools, labels or weights"
            )
        pools[key] = value
    if warnings and not allow_incomplete:
        raise ValueError(
            "; ".join(warnings) + "; use allow_incomplete only for provisional output"
        )
    return warnings


def _runs(rows: Sequence[dict], study: Mapping) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["phase"], row["variant"], row["repeat"]].append(row)
    result = []
    for (phase, variant, repeat), members in sorted(groups.items()):
        values = {
            "phase": phase,
            "variant": variant,
            "repeat": repeat,
            "workloads": len(members),
            "mean_decision_latency_ms": mean(
                row["decision_latency_ms"] for row in members
            ),
            "median_decision_latency_ms": median(
                row["decision_latency_ms"] for row in members
            ),
        }
        for metric in METRICS:
            scores = [
                row["metrics"][metric] for row in members if metric in row["metrics"]
            ]
            if len(scores) == len(members):
                values[metric] = mean(scores)
        if all("paper_normalizer_ms" in row for row in members):
            expected = {
                case["workload_id"]
                for case in study.get("phases", {}).get(phase, {}).get("cases", [])
            }
            complete = (
                bool(expected) and {row["workload_id"] for row in members} == expected
            )
            network_prefix = "paper_network" if complete else "pilot_network"
            numerator = sum(
                row["task_weight"] * row["paper_normalizer_ms"] for row in members
            )
            for k in (1, 5):
                denominator = sum(
                    row["task_weight"]
                    * min(
                        row["runtimes_ms"][candidate]
                        for candidate in row["ranking"][:k]
                    )
                    for row in members
                )
                values[f"{network_prefix}_top_{k}_score"] = numerator / denominator
            values["runtime_basis"] = "paper_f32_recovered_throughput_floor"
            values["complete_network_coverage"] = complete
            values["network_task_weight"] = sum(row["task_weight"] for row in members)
        usage = [_usage(row) for row in members]
        values.update({key: sum(row[key] for row in usage) for key in usage[0]})
        live_latencies = [
            _number(report["api_latency_ms"], "API latency")
            for row in members
            for report in row.get("request_reports", [])
            if not report["cache_hit"] and report.get("api_latency_ms") is not None
        ]
        if live_latencies:
            values["mean_live_api_latency_ms"] = mean(live_latencies)
            values["median_live_api_latency_ms"] = median(live_latencies)
            values["max_live_api_latency_ms"] = max(live_latencies)
        result.append(values)
    return result


def _variants(rows: Sequence[dict], runs: Sequence[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in runs:
        groups[row["phase"], row["variant"]].append(row)
    result = []
    for (phase, variant), members in sorted(groups.items()):
        cases = defaultdict(list)
        for row in rows:
            if (row["phase"], row["variant"]) == (phase, variant):
                cases[row["workload_id"]].append(row["ranking"][0])
        values = {
            "phase": phase,
            "variant": variant,
            "repeats": len(members),
            "workloads": len(cases),
            "same_pick_share": mean(len(set(picks)) == 1 for picks in cases.values()),
        }
        for metric in (
            *METRICS,
            "median_decision_latency_ms",
            "mean_live_api_latency_ms",
            "median_live_api_latency_ms",
        ):
            scores = [row[metric] for row in members if metric in row]
            if len(scores) == len(members):
                values[f"{metric}_mean"] = mean(scores)
                values[f"{metric}_std"] = stdev(scores) if len(scores) > 1 else 0
        result.append(values)
    return result


def _paired(rows: Sequence[dict]) -> list[dict]:
    phases = sorted({row["phase"] for row in rows})
    values = defaultdict(list)
    for row in rows:
        for metric, score in row["metrics"].items():
            values[row["phase"], row["variant"], row["workload_id"], metric].append(
                score
            )
    output = []
    for phase in phases:
        for left, right in PAIRS:
            left_ids = {
                row["workload_id"]
                for row in rows
                if row["phase"] == phase and row["variant"] == left
            }
            right_ids = {
                row["workload_id"]
                for row in rows
                if row["phase"] == phase and row["variant"] == right
            }
            if not left_ids or left_ids != right_ids:
                continue
            for metric in METRICS:
                differences = []
                for workload in sorted(left_ids):
                    a = values[phase, left, workload, metric]
                    b = values[phase, right, workload, metric]
                    if not a or not b:
                        break
                    differences.append(mean(b) - mean(a))
                if len(differences) != len(left_ids):
                    continue
                higher = "score" in metric or "accuracy" in metric
                better = sum(
                    value > 0 if higher else value < 0 for value in differences
                )
                worse = sum(value < 0 if higher else value > 0 for value in differences)
                n = better + worse
                p = (
                    min(
                        1.0,
                        2
                        * sum(math.comb(n, i) for i in range(min(better, worse) + 1))
                        / 2**n,
                    )
                    if n
                    else 1.0
                )
                output.append(
                    {
                        "phase": phase,
                        "left": left,
                        "right": right,
                        "metric": metric,
                        "workloads": len(differences),
                        "mean_right_minus_left": mean(differences),
                        "right_better": better,
                        "right_worse": worse,
                        "ties": len(differences) - n,
                        "two_sided_sign_test_p": p,
                        "interpretation": (
                            "exploratory, unadjusted; repeats averaged within workload"
                        ),
                    }
                )
    return output


def _baselines(rows: Sequence[dict]) -> list[dict]:
    output, seen = [], set()
    for row in rows:
        for name, shown in (
            ("uniform_random", row["candidate_ids"]),
            ("filter_plus_uniform_random", row["shown_candidate_ids"]),
        ):
            key = row["phase"], row["variant"], row["workload_id"], name
            if key in seen:
                continue
            seen.add(key)
            output.append(
                {
                    "phase": row["phase"],
                    "variant": row["variant"],
                    "workload_id": row["workload_id"],
                    "baseline": name,
                    "original_candidates": len(row["candidate_ids"]),
                    "shown_candidates": len(shown),
                    **expected_random_metrics(
                        row["runtimes_ms"], shown, reference_ms=_reference(row)
                    ),
                }
            )
    return output


def _baseline_aggregates(rows: Sequence[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["phase"], row["variant"], row["baseline"]].append(row)
    output = []
    numeric = (*METRICS, "retained_optimum", "oracle_top_1_score", "oracle_top_1_error")
    for (phase, variant, baseline), members in sorted(groups.items()):
        output.append(
            {
                "phase": phase,
                "variant": variant,
                "baseline": baseline,
                "workloads": len(members),
                **{
                    key: mean(row[key] for row in members)
                    for key in numeric
                    if all(key in row for row in members)
                },
                "aggregation": (
                    "macro workload mean; expectations are not network ratios"
                ),
            }
        )
    return output


def _random_rows(rows: Sequence[dict], seed: int) -> list[dict]:
    """Reproduce the registered RandomModel's single seeded ranking offline."""
    output, seen = [], set()
    for row in rows:
        key = row["phase"], row["workload_id"]
        if key in seen:
            continue
        seen.add(key)
        ranking = list(row["candidate_ids"])
        payload = f"{seed}\0{row['workload_id']}".encode()
        Random(int.from_bytes(hashlib.sha256(payload).digest())).shuffle(ranking)
        output.append(
            _validate_row(
                {
                    **row,
                    "variant": f"random-{seed}",
                    "repeat": 1,
                    "shown_candidate_ids": row["candidate_ids"],
                    "ranking": ranking,
                    "metrics": {},
                    "request_reports": [],
                    "decision_latency_ms": 0,
                }
            )
        )
    return output


def _plot(
    output: Path, variants: Sequence[Mapping], random_runs: Sequence[Mapping]
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "plots need the tenset extra; use --no-plots or uv sync --extra tenset"
        ) from error
    for phase in sorted({row["phase"] for row in variants}):
        members = [row for row in variants if row["phase"] == phase]
        network = all("paper_network_top_1_score_mean" in row for row in members)
        metric_names = (
            ("paper_network_top_1_score", "paper_network_top_5_score")
            if network
            else ("top_1_score", "endpoint_top_5_score")
        )
        labels = (
            ("Weighted network Top-1 score", "Weighted network Top-5 score")
            if network
            else ("Top-1 score", "Endpoint best-of-5 score")
        )
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
        for axis, metric, label in zip(
            axes,
            metric_names,
            labels,
        ):
            means = [row[f"{metric}_mean"] for row in members]
            deviations = [row[f"{metric}_std"] for row in members]
            axis.bar(
                [row["variant"] for row in members],
                means,
                yerr=deviations,
                color="#E84B36",
                capsize=3,
            )
            axis.set_ylabel(label)
            axis.tick_params(axis="x", rotation=30)
            lower = min(0, *(value - sd for value, sd in zip(means, deviations)))
            upper = max(value + sd for value, sd in zip(means, deviations))
            random_reference = next(
                (row for row in random_runs if row["phase"] == phase), None
            )
            if random_reference is not None and metric in random_reference:
                lower = min(lower, random_reference[metric])
                upper = max(upper, random_reference[metric])
                axis.axhline(
                    random_reference[metric],
                    color="#13294B",
                    linestyle="--",
                    label="Random (one seeded draw)",
                )
                axis.legend(fontsize=8)
            # Setting limits before axhline disables autoscaling, which can hide
            # a stronger Random baseline above every Jev bar. Include the entire
            # bar/error range and reference before choosing explicit limits.
            span = upper - lower or 1.0
            axis.set_ylim(
                bottom=lower - 0.05 * span if lower < 0 else 0,
                top=upper + 0.1 * span,
            )
        fig.suptitle(f"TenSet context ablation: {phase} (repeat mean ± sample SD)")
        fig.savefig(output / f"{phase}-comparison.png", dpi=180)
        fig.savefig(output / f"{phase}-comparison.svg")
        plt.close(fig)


def summarize_ablation(
    root: Path | str,
    *,
    output_dir: Path | str | None = None,
    phase: str | None = None,
    create_plots: bool = True,
    overwrite: bool = False,
    allow_incomplete: bool = False,
) -> dict:
    """Validate saved rows and export repeat, paired, baseline and usage tables."""
    root = Path(root).resolve()
    study = _json(root / "study.json")
    if "identity" in study:
        identity_payload = {
            key: value for key, value in study.items() if key != "identity"
        }
        digest = hashlib.sha256(
            json.dumps(
                identity_payload, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()
        if study["identity"] != digest:
            raise ValueError("frozen study identity does not match its contents")
    rows = _load_rows(root, study, phase, allow_incomplete)
    if not rows:
        raise ValueError("no completed ablation predictions found")
    warnings = _coverage(rows, study, allow_incomplete)
    output = Path(output_dir).resolve() if output_dir is not None else root / "summary"
    if output == root or output.name in {"runs", "jev-cache"}:
        raise ValueError("summary output must be separate from raw results")
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError("summary output is nonempty; use overwrite explicitly")
    runs = _runs(rows, study)
    variants = _variants(rows, runs)
    baselines = _baselines(rows)
    baseline_aggregates = _baseline_aggregates(baselines)
    random_rows = _random_rows(rows, study.get("seed", 42))
    random_runs = _runs(random_rows, study)
    paired = _paired(rows)
    per_workload = [
        {
            "phase": row["phase"],
            "variant": row["variant"],
            "repeat": row["repeat"],
            "workload_id": row["workload_id"],
            "task_weight": row["task_weight"],
            "candidate_count": len(row["candidate_ids"]),
            "shown_count": len(row["shown_candidate_ids"]),
            "selected_candidate_id": row["ranking"][0],
            "decision_latency_ms": row["decision_latency_ms"],
            **row["metrics"],
        }
        for row in rows
    ]
    usage = [_usage(row) for row in rows]
    input_price = _number(
        study.get("pricing", {}).get("input_per_million_usd", 0.042),
        "assumed input price",
    )
    summary = {
        "protocol": study.get("protocol"),
        "study_sha256": hashlib.sha256((root / "study.json").read_bytes()).hexdigest(),
        "phase_filter": phase,
        "decisions": len(rows),
        "runs": len(runs),
        "variants": len(variants),
        "warnings": warnings,
        "unreported_phases": sorted(
            set(study.get("phases", {})) - {row["phase"] for row in rows}
        ),
        "output_dir": str(output),
        "usage": {key: sum(row[key] for row in usage) for key in usage[0]},
        "input_cost_estimate": {
            "usd_per_million_input_tokens": input_price,
            "known_live_input_cost_usd": (
                sum(row["live_input_tokens"] for row in usage) * input_price / 1e6
            ),
            "output_cost_included": False,
            "retries_not_recorded_in_reports_included": False,
            "complete_input_usage": all(
                row["unknown_input_usage_requests"] == 0 for row in usage
            ),
        },
        "metric_notes": {
            "top_5_score": "mean of normalized best runtimes over prefixes 1..min(5,n)",
            "endpoint_top_5_score": (
                "original-pool reference / fastest runtime in the first five picks"
            ),
            "paper_network_scores": (
                "ratio of sums of task-frequency-weighted private normalizers and "
                "recovered selected latencies; separate from macro scores"
            ),
            "pilot_network_scores": (
                "partial-inventory ratios are named pilot_network; "
                "they are not complete paper-network results"
            ),
            "filtered_pairwise_metrics": (
                "omitted; unshown tails are not model rankings"
            ),
            "repeat_std": (
                "sample standard deviation of run aggregates, not a confidence interval"
            ),
            "paired_tests": (
                "exploratory unadjusted sign tests of repeat-averaged "
                "within-workload differences"
            ),
            "cost": (
                "recorded token counts only; cached tokens are historical, "
                "not newly charged"
            ),
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, data in (
        ("runs.csv", runs),
        ("variants.csv", variants),
        ("per_workload.csv", per_workload),
        ("baselines.csv", baselines),
        ("baseline-aggregates.csv", baseline_aggregates),
        ("random-reference.csv", random_runs),
        (
            "random-per-workload.csv",
            [
                {
                    "phase": row["phase"],
                    "workload_id": row["workload_id"],
                    "ranking": json.dumps(row["ranking"]),
                    **row["metrics"],
                }
                for row in random_rows
            ],
        ),
        ("paired.csv", paired),
    ):
        _write_csv(output / name, data)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# TenSet context ablation",
        "",
        "Mean ± sample SD across fresh repeats; Top-5 below is an endpoint score.",
        "",
        "| Phase | Variant | Repeats | Workloads | Top-1 score | "
        "Endpoint Top-5 score | Same pick |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in variants:
        lines.append(
            f"| {row['phase']} | {row['variant']} | {row['repeats']} | "
            f"{row['workloads']} | {row['top_1_score_mean']:.4f} ± "
            f"{row['top_1_score_std']:.4f} | "
            f"{row['endpoint_top_5_score_mean']:.4f} ± "
            f"{row['endpoint_top_5_score_std']:.4f} | "
            f"{row['same_pick_share']:.1%} |"
        )
    lines.extend(
        [
            "",
            "See baselines.csv for exact full-pool Random, filtered-pool Random and "
            "filter-oracle scores. All retain the original denominator.",
            "random-reference.csv is one reproducible seeded Random draw, "
            "including the weighted network score where the paper fields exist. "
            "It has no repeated-model variation estimate.",
            "",
            "Top-5 prefix scores are retained separately in the CSV files. "
            "Filtered tails have no reported pairwise quality. "
            "Paired tests are exploratory and unadjusted.",
        ]
    )
    network_variants = [
        row for row in variants if "paper_network_top_1_score_mean" in row
    ]
    if network_variants:
        lines.extend(
            [
                "",
                "## Full-network weighted endpoints",
                "",
                "These ratios use frequency-weighted sums of private paper normalizers "
                "and recovered selected latencies, separately from macro scores.",
                "",
                "| Variant | Network Top-1 score | Network Top-5 score |",
                "|---|---:|---:|",
            ]
        )
        for row in network_variants:
            lines.append(
                f"| {row['variant']} | "
                f"{row['paper_network_top_1_score_mean']:.4f} ± "
                f"{row['paper_network_top_1_score_std']:.4f} | "
                f"{row['paper_network_top_5_score_mean']:.4f} ± "
                f"{row['paper_network_top_5_score_std']:.4f} |"
            )
        for row in random_runs:
            if "paper_network_top_1_score" in row:
                lines.append(
                    f"| {row['variant']} (one draw) | "
                    f"{row['paper_network_top_1_score']:.4f} | "
                    f"{row['paper_network_top_5_score']:.4f} |"
                )
    if warnings:
        lines.extend(["", "PROVISIONAL: " + "; ".join(warnings)])
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if create_plots:
        _plot(output, variants, random_runs)
    return summary


_SECRET_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "access_token",
    "refresh_token",
    "password",
    "client_secret",
    "typesafe_api_key",
}


def _check_secrets(value: object) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if isinstance(key, str) and key.lower().replace("-", "_") in _SECRET_KEYS:
                raise ValueError("refusing to archive credential-bearing data")
            _check_secrets(child)
    elif isinstance(value, list):
        for child in value:
            _check_secrets(child)


def package_ablation(root: Path | str, destination: Path | str) -> Path:
    """Archive only study results/configs/cache, rejecting secrets and symlinks."""
    root, destination = Path(root).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("archive already exists")
    if not (root / "study.json").is_file():
        raise ValueError("study.json is required for raw-result sharing")
    candidates = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        allowed = (
            relative.as_posix() == "study.json"
            or relative.as_posix() == "config.json"
            or relative.as_posix() == "README.md"
            or (
                relative.parts[0] in {"runs", "receipts", "summary"}
                and path.suffix
                in {".json", ".jsonl", ".yaml", ".csv", ".md", ".svg", ".png"}
            )
            or (
                "jev-cache" in relative.parts
                and bool(re.fullmatch(r"[0-9a-f]{64}\.json", path.name))
            )
        )
        if not allowed or any(part.startswith(".") for part in relative.parts):
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("archive paths must remain inside the study")
        if path.suffix == ".json":
            _check_secrets(json.loads(path.read_text(encoding="utf-8")))
        elif path.suffix == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    _check_secrets(json.loads(line))
        elif path.suffix == ".yaml":
            import yaml

            _check_secrets(yaml.safe_load(path.read_text(encoding="utf-8")))
        candidates.append((path, relative))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, relative in candidates:
            archive.write(path, arcname=relative.as_posix())
        archive.writestr(
            "SHARING-NOTES.md",
            "# TenSet raw ablation results\n\n"
            "Contains private evaluation labels and raw model requests/responses "
            "for team review. No environment file is included. Cached responses "
            "are historical usage, not fresh repeat calls.\n",
        )
    return destination
