"""Reconstruct TenSet's weighted network endpoint metric without TVM.

This is the metric from the pinned ``eval_model_on_dataset.py`` evaluator. It
is distinct from this project's macro/prefix-average Top-K scheduling scores.
Labels and task minima originate as float32 in C++; ``feature.unpack_feature``
unpacks them to Python floats and NumPy's default float64 arrays. Consequently
the latency recovery and weighted network accumulation below use float64.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import stat
import struct
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, urlsplit

SOURCE_COMMIT = "35774ed373f9693cc6efcbe83f5703141e8ab0c8"
SOURCE_URL = (
    "https://github.com/tlc-pack/tenset/blob/"
    f"{SOURCE_COMMIT}/scripts/eval_model_on_dataset.py"
)
PROTOCOL = "tenset_official_weighted_network_endpoint_v1"
THROUGHPUT_FLOOR = 1e-5
SUMMARY_JSON = "paper-network-summary.json"
SUMMARY_CSV = "paper-network-summary.csv"
TASK_CSV = "paper-task-decompositions.csv"
REFERENCE_CSV = "paper-reference-comparison.csv"
REFERENCE_MD = "paper-reference-comparison.md"


@dataclass(frozen=True)
class PaperTaskPool:
    """One private evaluation pool and its network extraction frequency.

    ``full_task_min_latency_seconds`` is the full source-task normalizer, not
    the minimum latency among the model's chosen candidates. Both that value
    and ``labels_by_candidate`` are evaluator inputs and must never be added
    to Jev's state, instructions or candidate configurations.
    """

    workload_id: str
    workload_key: str
    target: str
    weight: int
    full_task_min_latency_seconds: float
    labels_by_candidate: Mapping[str, float]


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _weight(value: object) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError("network task weights must be positive integers")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError("network task weights must be finite")
    return value


def _float32(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite and representable as float32")
    try:
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite and representable as float32")
        result = struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, struct.error) as error:
        raise ValueError(f"{name} must be representable as float32") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite in float32")
    return result


def _top_ks(values: Sequence[int]) -> tuple[int, ...]:
    if (
        not isinstance(values, Sequence)
        or isinstance(values, (str, bytes))
        or not values
        or any(type(value) is not int or value < 1 for value in values)
        or len(values) != len(set(values))
    ):
        raise ValueError("top_ks must contain distinct positive integers")
    return tuple(values)


def _validated_pools(
    pools: Sequence[PaperTaskPool],
    expected_task_weights: Mapping[str, int],
    target: str,
    pilot: bool,
) -> tuple[dict[str, tuple[PaperTaskPool, float, dict[str, float]]], dict[str, int]]:
    if type(pilot) is not bool:
        raise ValueError("pilot must be a boolean")
    _string(target, "target")
    if not isinstance(expected_task_weights, Mapping) or not expected_task_weights:
        raise ValueError("expected_task_weights must contain the complete network")
    expected = {
        _string(identity, "expected workload ID"): _weight(weight)
        for identity, weight in expected_task_weights.items()
    }
    if not isinstance(pools, Sequence) or not pools:
        raise ValueError("network scoring requires at least one task pool")
    result = {}
    workload_keys = set()
    for pool in pools:
        if not isinstance(pool, PaperTaskPool):
            raise ValueError("network task pools must be PaperTaskPool values")
        identity = _string(pool.workload_id, "workload ID")
        key = _string(pool.workload_key, "workload key")
        if identity in result or key in workload_keys:
            raise ValueError("network task pools must have unique IDs and keys")
        if pool.target != target:
            raise ValueError("every network task must use the exact requested target")
        if identity not in expected or _weight(pool.weight) != expected[identity]:
            raise ValueError("task pool weight disagrees with the network inventory")
        minimum = _float32(
            pool.full_task_min_latency_seconds, "full-task minimum latency"
        )
        if minimum <= 0:
            raise ValueError("full-task minimum latency must be positive")
        if (
            not isinstance(pool.labels_by_candidate, Mapping)
            or not pool.labels_by_candidate
        ):
            raise ValueError("each task must contain its nonempty candidate label pool")
        labels = {}
        for candidate, label in pool.labels_by_candidate.items():
            candidate = _string(candidate, "candidate ID")
            value = _float32(label, "normalized throughput label")
            if value > 1:
                raise ValueError("full-task normalized throughput cannot exceed one")
            labels[candidate] = value
        workload_keys.add(key)
        result[identity] = pool, minimum, labels
    if not pilot and set(result) != set(expected):
        raise ValueError("full-network scoring requires every expected task pool")
    return result, expected


def score_paper_network(
    pools: Sequence[PaperTaskPool],
    rankings_by_workload: Mapping[str, Sequence[str]],
    *,
    network_key: object,
    target: str,
    expected_task_weights: Mapping[str, int],
    pilot: bool = False,
    top_ks: Sequence[int] = (1, 5),
) -> dict:
    """Compute weighted network latency ratios for complete task rankings.

    The required inventory binds pool coverage and frequency weights. A pilot
    must be explicitly marked and produces ``pilot_network_*`` subtotals,
    never metrics labelled as a complete network evaluation. Negative or zero
    labels are retained and floored exactly as in the reference evaluator.
    """

    task_pools, expected = _validated_pools(pools, expected_task_weights, target, pilot)
    ks = _top_ks(top_ks)
    if not isinstance(rankings_by_workload, Mapping) or set(
        rankings_by_workload
    ) != set(task_pools):
        raise ValueError("rankings must cover every evaluated task exactly once")
    try:
        network = json.loads(json.dumps(network_key, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("network_key must be finite JSON data") from error
    if network in (None, "", [], {}):
        raise ValueError("network_key must identify the evaluated network")
    numerator = 0.0
    denominators = {k: 0.0 for k in ks}
    decompositions = []
    # The official network dictionary follows extraction order. Preserve the
    # required inventory's order rather than order tasks by their latency.
    for identity in expected:
        if identity not in task_pools:
            continue  # Only reachable for an explicitly labelled pilot subset.
        pool, minimum, labels = task_pools[identity]
        ranking = rankings_by_workload[identity]
        if (
            not isinstance(ranking, Sequence)
            or isinstance(ranking, (str, bytes))
            or len(ranking) != len(labels)
            or any(not isinstance(item, str) for item in ranking)
            or len(set(ranking)) != len(ranking)
            or set(ranking) != set(labels)
        ):
            raise ValueError("each ranking must cover its exact candidate label pool")
        weighted_oracle = minimum * pool.weight
        numerator += weighted_oracle
        for k in ks:
            prefix = ranking[:k]
            # float32-origin values are unpacked into float64 by the official
            # Python feature reader; do not round this division back to float32.
            selected = min(
                prefix,
                key=lambda candidate: minimum
                / max(labels[candidate], THROUGHPUT_FLOOR),
            )
            label = labels[selected]
            latency = minimum / max(label, THROUGHPUT_FLOOR)
            contribution = latency * pool.weight
            denominators[k] += contribution
            decompositions.append(
                {
                    "workload_id": identity,
                    "workload_key": pool.workload_key,
                    "target": target,
                    "task_weight": pool.weight,
                    "candidate_count": len(labels),
                    "top_k": k,
                    "evaluated_prefix_count": len(prefix),
                    "best_candidate_in_prefix": selected,
                    "selected_normalized_throughput_float32": label,
                    "effective_normalized_throughput": max(label, THROUGHPUT_FLOOR),
                    "full_task_min_latency_seconds_float32": minimum,
                    "weighted_oracle_latency_seconds": weighted_oracle,
                    "best_of_k_latency_seconds": latency,
                    "weighted_selected_latency_seconds": contribution,
                }
            )
    if (
        not math.isfinite(numerator)
        or numerator <= 0
        or any(
            not math.isfinite(value) or value <= 0 for value in denominators.values()
        )
    ):
        raise ValueError("weighted network latencies must remain finite and positive")
    scope = "pilot_subset" if pilot else "full_network"
    prefix = "pilot_network" if pilot else "paper_network"
    return {
        "schema_version": 1,
        "metric_protocol": PROTOCOL,
        "source_commit": SOURCE_COMMIT,
        "source_evaluator": SOURCE_URL,
        "network_key": network,
        "target": target,
        "scope": scope,
        "coverage": {
            "expected_distinct_tasks": len(expected),
            "evaluated_distinct_tasks": len(task_pools),
            "expected_total_task_weight": sum(expected.values()),
            "evaluated_total_task_weight": sum(
                expected[identity] for identity in task_pools
            ),
            "complete_network_coverage": set(task_pools) == set(expected),
        },
        "top_ks": list(ks),
        "metrics": {
            f"{prefix}_best_of_{k}_score": numerator / denominators[k] for k in ks
        },
        "best_weighted_latency_seconds": numerator,
        "selected_weighted_latency_seconds_by_k": {
            str(k): value for k, value in denominators.items()
        },
        "score_definition": (
            "sum(task_weight * full_task_min_latency) / "
            "sum(task_weight * min(full_task_min_latency / "
            "max(float32_origin_label, 1e-5) for candidates in first K))"
        ),
        "label_arithmetic": (
            "float32-origin full-task minima and labels; float64 latency recovery "
            "and ordered weighted accumulation, matching feature.unpack_feature"
        ),
        "throughput_floor": THROUGHPUT_FLOOR,
        "task_decompositions": decompositions,
    }


def compare_paper_network(
    pools: Sequence[PaperTaskPool],
    model_rankings: Mapping[str, Mapping[str, Sequence[str]]],
    *,
    network_key: object,
    target: str,
    expected_task_weights: Mapping[str, int],
    model_types: Mapping[str, str] | None = None,
    pilot: bool = False,
    top_ks: Sequence[int] = (1, 5),
) -> dict:
    """Score model rankings on identical pools and retain paired Random results."""

    if not isinstance(model_rankings, Mapping) or not model_rankings:
        raise ValueError("provide at least one model's network rankings")
    if model_types is not None and set(model_types) != set(model_rankings):
        raise ValueError("model_types must describe every model exactly once")
    models = []
    for model_id, rankings in model_rankings.items():
        _string(model_id, "model ID")
        model_type = (
            _string(model_types[model_id], "model type")
            if model_types is not None
            else "unspecified"
        )
        result = score_paper_network(
            pools,
            rankings,
            network_key=network_key,
            target=target,
            expected_task_weights=expected_task_weights,
            pilot=pilot,
            top_ks=top_ks,
        )
        models.append({"model_id": model_id, "model_type": model_type, **result})
    first = models[0]
    comparisons = []
    for baseline in (row for row in models if row["model_type"] == "random"):
        for model in (row for row in models if row["model_type"] != "random"):
            for metric, score in model["metrics"].items():
                reference = baseline["metrics"][metric]
                comparisons.append(
                    {
                        "model_id": model["model_id"],
                        "random_model_id": baseline["model_id"],
                        "metric": metric,
                        "model_score": score,
                        "random_score": reference,
                        "score_delta": score - reference,
                    }
                )
    return {
        key: first[key]
        for key in (
            "schema_version",
            "metric_protocol",
            "source_commit",
            "source_evaluator",
            "network_key",
            "target",
            "scope",
            "coverage",
            "top_ks",
            "score_definition",
            "label_arithmetic",
            "throughput_floor",
        )
    } | {
        "models": models,
        "paired_random_comparisons": comparisons,
        "reference_status": (
            "official network evaluator reconstructed; no source-paper model "
            "checkpoint or table result has been reproduced by these metrics"
        ),
    }


def _csv_text(rows: Sequence[Mapping], fieldnames: Sequence[str]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _reference_score(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError("reference comparison scores must be finite values in [0, 1]")
    return float(value)


def _reference_comparison_contents(comparison: Mapping, models: list) -> dict[str, str]:
    reference = comparison.get("external_reference")
    if reference is None:
        return {}
    coverage = comparison.get("coverage")
    if (
        comparison.get("scope") != "full_network"
        or not isinstance(coverage, Mapping)
        or coverage.get("complete_network_coverage") is not True
    ):
        raise ValueError(
            "published reference comparison requires complete full network"
        )
    if not isinstance(reference, Mapping):
        raise ValueError("external_reference must be a published model mapping")
    name = _string(reference.get("model"), "published model descriptor")
    if any(ord(character) < 32 or ord(character) == 127 for character in name):
        raise ValueError("published model descriptor must be a single printable line")
    source = _string(reference.get("source_url"), "published source URL")
    parsed = urlsplit(source)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or any(character.isspace() or ord(character) < 32 for character in source)
    ):
        raise ValueError(
            "published source URL must be an HTTP(S) URL without credentials"
        )
    jev = [
        model for model in models if str(model.get("model_type", "")).startswith("jev")
    ]
    random = [model for model in models if model.get("model_type") == "random"]
    if len(models) != 2 or len(jev) != 1 or len(random) != 1:
        raise ValueError(
            "reference comparison requires one local Jev and one Random model"
        )
    score_names = ("paper_network_best_of_1_score", "paper_network_best_of_5_score")
    rows = []
    for label, model in (("Jev", jev[0]), ("Random", random[0])):
        if any(metric not in model["metrics"] for metric in score_names):
            raise ValueError("reference comparison requires both local endpoint scores")
        rows.append(
            {
                "Model": label,
                "Evidenceclass": "local paired measurement",
                **{
                    metric: _reference_score(model["metrics"][metric])
                    for metric in score_names
                },
                "same_as_local_candidate_pool_verified": True,
                "source_URL": "predictions.jsonl",
            }
        )
    rows.append(
        {
            "Model": name,
            "Evidenceclass": "published reference (unpaired)",
            score_names[0]: _reference_score(reference.get("top_1")),
            score_names[1]: _reference_score(reference.get("top_5")),
            "same_as_local_candidate_pool_verified": False,
            "source_URL": source,
        }
    )
    lines = [
        "# Weighted network endpoint scores",
        "",
        "| Model | Evidence class | Top-1 network score | Top-5 network score | "
        "Same local candidate pool verified | Source |",
        "|---|---|---:|---:|---|---|",
    ]
    for row in rows:
        model_name = row["Model"].replace("\\", "\\\\").replace("|", "\\|")
        local = row["same_as_local_candidate_pool_verified"]
        url = quote(row["source_URL"], safe=":/?#[]@!$&'*+,;=%")
        link = f"[{'this experiment' if local else 'paper'}]({url})"
        lines.append(
            f"| {model_name} | {row['Evidenceclass']} | "
            f"{row[score_names[0]]:.4f} | {row[score_names[1]]:.4f} | "
            f"{'Yes' if local else 'No'} | {link} |"
        )
    lines.extend(
        [
            "",
            "Scores use the weighted network best-of-K endpoint definition and are "
            "rounded to four decimal places here; CSV retains full precision.",
            "",
            "The published model is an external reference. Its original paper test "
            "inventory and trained checkpoint remain unverified by this run. Only "
            "the local Jev-versus-Random comparison is paired on identical candidate "
            "pools; the external row is not a locally executed model and carries "
            "no paired significance claim.",
            "",
        ]
    )
    return {
        REFERENCE_CSV: _csv_text(rows, list(rows[0])),
        REFERENCE_MD: "\n".join(lines),
    }


def _existing_export_matches(path: Path, content: str) -> bool:
    """Accept only an existing regular, nonsymlink file with identical bytes."""

    if path.is_symlink():
        raise FileExistsError(f"network metric export differs: {path}")
    flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        return False
    except OSError as error:
        raise FileExistsError(f"network metric export differs: {path}") from error
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise FileExistsError(f"network metric export differs: {path}")
    with os.fdopen(descriptor, "rb") as stream:
        if stream.read() != content.encode("utf-8"):
            raise FileExistsError(f"network metric export differs: {path}")
    return True


def export_paper_network_scores(output_dir: str | Path, comparison: Mapping) -> dict:
    """Write separate endpoint exports, idempotently preserving existing files.

    A partial previous publication can be resumed only when every existing
    export matches the recomputed content exactly. Different results are never
    overwritten. This function makes no model or network requests.
    """

    destination = Path(output_dir)
    if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
        raise ValueError("network metric output must be a directory, not a symlink")
    models = comparison.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("network comparison exports require scored models")
    summary_rows, task_rows = [], []
    metric_names = list(models[0]["metrics"])
    for model in models:
        if list(model["metrics"]) != metric_names:
            raise ValueError("exported models must report identical metric names")
        summary_rows.append(
            {
                "model_id": model["model_id"],
                "model_type": model["model_type"],
                "scope": comparison["scope"],
                "network_key": json.dumps(comparison["network_key"]),
                "target": comparison["target"],
                "distinct_tasks": comparison["coverage"]["evaluated_distinct_tasks"],
                "total_task_weight": comparison["coverage"][
                    "evaluated_total_task_weight"
                ],
                "complete_network_coverage": comparison["coverage"][
                    "complete_network_coverage"
                ],
                **model["metrics"],
                "best_weighted_latency_seconds": model["best_weighted_latency_seconds"],
                **{
                    f"selected_weighted_latency_seconds_k{k}": model[
                        "selected_weighted_latency_seconds_by_k"
                    ][str(k)]
                    for k in comparison["top_ks"]
                },
            }
        )
        task_rows.extend(
            {
                "model_id": model["model_id"],
                "model_type": model["model_type"],
                "scope": comparison["scope"],
                **task,
            }
            for task in model["task_decompositions"]
        )
    contents = {
        SUMMARY_JSON: json.dumps(comparison, indent=2, allow_nan=False) + "\n",
        SUMMARY_CSV: _csv_text(summary_rows, list(summary_rows[0])),
        TASK_CSV: _csv_text(task_rows, list(task_rows[0])),
        **_reference_comparison_contents(comparison, models),
    }
    for name, content in contents.items():
        _existing_export_matches(destination / name, content)
    destination.mkdir(parents=True, exist_ok=True)
    for name, content in contents.items():
        path = destination / name
        if _existing_export_matches(path, content):
            continue
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination, delete=False
        ) as stream:
            temporary = Path(stream.name)
            try:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        try:
            try:
                # Both files are in the same directory/filesystem. Linking is
                # atomic and, unlike replace(), cannot overwrite a late file.
                os.link(temporary, path)
            except FileExistsError:
                if not _existing_export_matches(path, content):
                    raise FileExistsError(
                        f"network metric export changed during publication: {path}"
                    ) from None
        finally:
            temporary.unlink(missing_ok=True)
    return {name: destination / name for name in contents}
