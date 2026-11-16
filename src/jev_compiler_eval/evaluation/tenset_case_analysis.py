"""Offline operation breakdowns and concrete Jev-versus-Random schedule cases."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from math import isclose, isfinite
from pathlib import Path
from statistics import mean

import yaml

from jev_compiler_eval.datasets.tenset import (
    TenSetDataset,
    _digest,
    _reject_label_keys,
)
from jev_compiler_eval.datasets.tenset_collection import load_collection_index
from jev_compiler_eval.evaluation.tenset_analysis import (
    _CollectionWorkloads,
    _load_rows,
    _read_json,
)

SCORES = ("top_1_score", "top_5_score")


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _csv(path: Path, rows: Sequence[Mapping], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _family(operation: str) -> str:
    """Coarse, explicitly name-based grouping of actual operation metadata."""
    value = operation.lower()
    if value in {"", "unknown", "none"}:
        return "unknown"
    for group, names in (
        ("convolution", ("conv",)),
        ("batch_matmul", ("batch_matmul", "batchmatmul")),
        ("matmul_or_dense", ("matmul", "dense", "gemm")),
        ("reduction", ("reduce", "sum", "softmax", "mean", "pool")),
        ("elementwise", ("relu", "add", "multiply", "divide", "elementwise")),
    ):
        if any(name in value for name in names):
            return group
    return "other"


def _metadata(workload: Mapping) -> dict:
    context = workload["context"]
    _reject_label_keys(context)
    operation = context.get("operation")
    operation = operation if isinstance(operation, str) and operation else "unknown"
    tensors = context.get("tensors")
    tensors = tensors if isinstance(tensors, list) else []
    workload_key = context.get("workload_key")
    workload_key = workload_key if isinstance(workload_key, str) else ""
    family_hash = ""
    try:
        decoded = json.loads(workload_key)
    except (ValueError, TypeError):
        decoded = None
    if (
        isinstance(decoded, list)
        and decoded
        and isinstance(decoded[0], str)
        and re.fullmatch(r"[0-9a-fA-F]{32}", decoded[0])
    ):
        family_hash = decoded[0].lower()
    complete = (
        operation != "unknown"
        and bool(tensors)
        and all(
            isinstance(t, Mapping) and "shape" in t and t.get("dtype") for t in tensors
        )
        and bool(context.get("compute_dag"))
        and isinstance(context.get("loop_structure"), list)
    )
    return {
        "operation": operation,
        "operation_family": _family(operation),
        "workload_key": workload_key,
        "dag_family_hash": family_hash,
        "target": str(context.get("target", "unknown")),
        "tensors": tensors,
        "metadata_complete": bool(complete),
    }


def _load_scope(result_dir: Path, dataset: Path):
    root = _read_json(dataset)
    collection = root.get("dataset") == "tenset_collection"
    if collection:
        root = load_collection_index(dataset)
        workloads = _CollectionWorkloads(dataset, root)
    elif root.get("dataset") == "tenset":
        entries = root.get("workloads")
        if not isinstance(entries, list) or not entries:
            raise ValueError("prepared TenSet workloads must be a nonempty list")
        workloads = {entry["workload_id"]: entry for entry in entries}
        if len(workloads) != len(entries):
            raise ValueError("prepared workload IDs must be unique")
    else:
        raise ValueError("dataset must be a TenSet artifact or collection index")
    saved = yaml.safe_load((result_dir / "experiment.yaml").read_text())
    if not isinstance(saved, dict) or saved.get("dataset", {}).get("type") != "tenset":
        raise ValueError("saved experiment must describe a TenSet dataset")
    settings = dict(saved["dataset"]["config"])
    settings["path"] = str(dataset)
    TenSetDataset(settings).validate_config()
    split, seed = settings.get("split", "test"), settings.get("seed", 42)
    if collection:
        identifiers = [
            entry["workload_id"]
            for entry in root["shards"]
            if split == "all" or entry["split"] == split
        ]
    else:
        identifiers = [
            key
            for key in workloads
            if split == "all" or workloads[key].get("split") == split
        ]
        identifiers.sort(key=lambda key: _digest([seed, key]))
    limit = settings.get("max_workloads")
    if limit is not None:
        identifiers = identifiers[:limit]
    if not identifiers:
        raise ValueError("saved experiment selects no workloads")
    model_types = {entry["id"]: entry["type"] for entry in saved.get("models", [])}
    if not model_types or len(model_types) != len(saved["models"]):
        raise ValueError("saved experiment must declare unique model IDs")
    return root, workloads, settings, identifiers, model_types


def _expected_candidates(workload: Mapping, settings: Mapping) -> list[str]:
    entries = workload["candidates"]
    identifiers = [candidate["candidate_id"] for candidate in entries]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("prepared candidate IDs must be unique")
    identifiers.sort(
        key=lambda key: _digest(
            [settings.get("seed", 42), workload["workload_id"], key]
        )
    )
    limit = settings.get("candidate_limit", 32)
    return identifiers if limit is None else identifiers[:limit]


def _scores(pool: Mapping[str, Mapping], ranking: list[str]) -> tuple[dict, str]:
    """Use exported private reference labels when present, otherwise runtimes."""
    present = ["official_throughput" in candidate for candidate in pool.values()]
    if any(present) and not all(present):
        raise ValueError("official throughput labels must cover the entire test pool")
    if all(present):
        labels = {
            key: candidate["official_throughput"] for key, candidate in pool.items()
        }
        if any(
            isinstance(label, bool)
            or not isinstance(label, (int, float))
            or not isfinite(label)
            or not 0 < label <= 1
            for label in labels.values()
        ):
            raise ValueError("official throughput labels must be finite and in (0, 1]")
        products = [
            labels[key] * candidate["runtime_ms"] for key, candidate in pool.items()
        ]
        if any(not isclose(value, products[0], rel_tol=2e-6) for value in products):
            raise ValueError("official labels disagree with prepared runtime ratios")
        basis = "private_official_throughput"
    else:
        labels = {key: 1 / candidate["runtime_ms"] for key, candidate in pool.items()}
        basis = "prepared_runtime_ratio"
    oracle = max(labels.values())
    best = 0.0
    curve = []
    for key in ranking[:5]:
        best = max(best, labels[key])
        curve.append(best / oracle)
    return {"top_1_score": curve[0], "top_5_score": mean(curve)}, basis


def _failure(candidate: Mapping) -> dict:
    error = candidate.get("source_error_code")
    if error is not None and type(error) is not int:
        raise ValueError("source error code must be an integer")
    sentinel = isclose(candidate["runtime_ms"], 1e13, rel_tol=1e-6)
    return {
        "source_error_code": error,
        "suspected_failure_sentinel": sentinel,
        "is_failure": error != 0 if error is not None else sentinel,
        "failure_status_basis": "source_error_code"
        if error is not None
        else "sentinel_heuristic",
    }


def _group(rows: Sequence[dict], keys: Sequence[str]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    output = []
    for values, members in sorted(groups.items()):
        total = sum(row["candidate_count"] for row in members)
        output.append(
            {
                **dict(zip(keys, values)),
                "workloads": len(members),
                "test_records": total,
                "metadata_complete_workloads": sum(
                    row["metadata_complete"] for row in members
                ),
                "selected_failure_records": sum(
                    row["selected_failure"] for row in members
                ),
                **{
                    f"macro_{name}": mean(row[name] for row in members)
                    for name in SCORES
                },
                **{
                    f"weighted_{name}": sum(
                        row[name] * row["candidate_count"] for row in members
                    )
                    / total
                    for name in SCORES
                },
            }
        )
    return output


def _choice(candidate: Mapping) -> dict:
    return {
        "candidate_id": candidate["candidate_id"],
        "recorded_cost_ms": candidate["runtime_ms"],
        **_failure(candidate),
        "configuration": candidate["configuration"],
    }


def _case_studies(workloads: Mapping, details: Mapping, examples: int) -> list[dict]:
    paired = defaultdict(dict)
    for row in details.values():
        paired[row["workload_id"]][row["model_id"]] = row
    comparisons = defaultdict(list)
    for workload, members in paired.items():
        random_rows = [row for row in members.values() if row["model_type"] == "random"]
        if len(random_rows) != 1:
            raise ValueError("case analysis requires exactly one Random baseline")
        baseline = random_rows[0]
        for row in members.values():
            if row["model_type"].startswith("jev"):
                comparisons[row["model_id"]].append(
                    {
                        "workload_id": workload,
                        "jev": row,
                        "random": baseline,
                        "delta": row["top_1_score"] - baseline["top_1_score"],
                    }
                )
    selected = []
    for comparisons_for_model in comparisons.values():
        for outcome, sign in (("success", 1), ("failure", -1)):
            ordered = sorted(
                (case for case in comparisons_for_model if sign * case["delta"] > 0),
                key=lambda case: (-sign * case["delta"], case["workload_id"]),
            )
            for case in ordered[:examples]:
                workload = workloads[case["workload_id"]]
                candidates = {c["candidate_id"]: c for c in workload["candidates"]}
                pool = case["jev"]["candidate_ids"]
                optimum = min(
                    pool, key=lambda key: (candidates[key]["runtime_ms"], key)
                )
                context = workload["context"]
                selected.append(
                    {
                        "outcome": outcome,
                        "selection_rule": (
                            "post-hoc largest positive/negative Top-1 differences"
                        ),
                        "workload_id": case["workload_id"],
                        "jev_model": case["jev"]["model_id"],
                        "random_model": case["random"]["model_id"],
                        "top_1_score_delta": case["delta"],
                        "candidate_count": len(pool),
                        "score_basis": case["jev"]["score_basis"],
                        "context": {
                            key: context[key]
                            for key in (
                                "operation",
                                "workload_key",
                                "target",
                                "tensors",
                                "compute_dag",
                                "loop_structure",
                            )
                            if key in context
                        },
                        "jev": _choice(
                            candidates[case["jev"]["selected_candidate_id"]]
                        ),
                        "random": _choice(
                            candidates[case["random"]["selected_candidate_id"]]
                        ),
                        "oracle": _choice(candidates[optimum]),
                        "interpretation": (
                            "Describes observed schedules and costs. Differences "
                            "do not establish why Jev chose them or cause changes."
                        ),
                    }
                )
    return selected


def _case_markdown(cases: Sequence[dict]) -> str:
    sections = [
        "# Jev-versus-Random schedule case studies",
        "",
        (
            "Cases are post-hoc illustrations. All workloads remain in the "
            "aggregate metrics."
        ),
        (
            "Recorded failure-sentinel costs are placeholders, "
            "not physical execution latencies."
        ),
        "",
    ]
    for case in cases:
        sections.extend(
            [
                f"## {case['outcome'].title()}: {case['workload_id']}",
                "",
                f"Operation: {case['context'].get('operation', 'unknown')}. "
                f"Candidates: {case['candidate_count']}. "
                "Top-1 difference, Jev minus Random: "
                f"{case['top_1_score_delta']:+.6f}.",
                "",
                "Tensor shapes: "
                + json.dumps(case["context"].get("tensors", []))
                + ".",
                "",
                "| Choice | Candidate | Cost (ms) | Failure flag | Transformations |",
                "|---|---|---:|---|---|",
            ]
        )
        for name in ("jev", "random", "oracle"):
            choice = case[name]
            counts = Counter(
                step.get("code", "unknown")
                for step in choice["configuration"].get("transformations", [])
            )
            transforms = ", ".join(
                f"{key} x {value}" for key, value in sorted(counts.items())
            )
            sections.append(
                f"| {name.title()} | {choice['candidate_id']} | "
                f"{choice['recorded_cost_ms']:.9g} | "
                f"{choice['is_failure']} | {transforms or 'none recorded'} |"
            )
        sections.extend(
            [
                "",
                "Complete transformation arguments and program context "
                "are in case_studies.json.",
                "",
                case["interpretation"],
                "",
            ]
        )
    return "\n".join(sections)


def analyze_tenset_cases(
    result_dir: str | Path,
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    examples_per_outcome: int = 3,
) -> Path:
    """Validate the frozen pool, join local metadata and export offline analyses."""
    result_dir, dataset, destination = (
        Path(result_dir),
        Path(dataset_path),
        Path(output_dir),
    )
    if type(examples_per_outcome) is not int or examples_per_outcome < 1:
        raise ValueError("examples_per_outcome must be a positive integer")
    if destination.exists() and (
        not destination.is_dir() or any(destination.iterdir())
    ):
        raise FileExistsError(f"output directory is not empty: {destination}")
    _, workloads, settings, identifiers, model_types = _load_scope(result_dir, dataset)
    prepared_sha256 = _sha256(dataset)
    # Shared checks recompute all declared metrics and validate prepared/input
    # fingerprints, runtime maps, model sets and paired candidate ordering.
    canonical, _, metric_names = _load_rows([result_dir], workloads, prepared_sha256)
    if not set(SCORES).issubset(metric_names):
        raise ValueError("case analysis requires Top-1 and Top-5 Score")
    canonical_map = {(row["model_id"], row["workload_id"]): row for row in canonical}
    manifest = _read_json(result_dir / "manifest.json")
    provenance = manifest.get("dataset_provenance", {})
    if provenance.get("prepared_file_sha256") != prepared_sha256:
        raise ValueError("case analysis requires a matching prepared-data fingerprint")
    for field, default in {
        "split": "test",
        "seed": 42,
        "candidate_limit": 32,
        "max_workloads": None,
        "describe_config_fields": False,
    }.items():
        if provenance.get(field, default) != settings.get(field, default):
            raise ValueError("saved experiment and manifest dataset settings differ")
    actual_workloads = {row["workload_id"] for row in canonical}
    if actual_workloads != set(identifiers):
        raise ValueError(
            "predictions do not cover the entire configured workload cohort"
        )
    models = set(manifest["model_ids"])
    if models != set(model_types):
        raise ValueError("saved experiment model IDs differ from the manifest")
    if len(canonical) != len(models) * len(identifiers):
        raise ValueError("the result bundle is missing model/workload predictions")
    rows, details = [], {}
    with (result_dir / "predictions.jsonl").open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            prediction = json.loads(line)
            if prediction["model_type"] != model_types[prediction["model_id"]]:
                raise ValueError("prediction model type differs from saved experiment")
            key = prediction["model_id"], prediction["workload_id"]
            workload = workloads[prediction["workload_id"]]
            expected = _expected_candidates(workload, settings)
            if prediction["candidate_ids"] != expected:
                raise ValueError(
                    "prediction candidate pool differs from configured frozen pool"
                )
            if any(
                name not in prediction
                for name in (
                    "model_input_sha256",
                    "model_input_bytes",
                    "runtime_by_candidate_ms",
                )
            ):
                raise ValueError(
                    "case analysis requires saved model-input and runtime fingerprints"
                )
            ranking = prediction["predicted_ranking"]
            if prediction.get("selected_candidate_id") != ranking[0]:
                raise ValueError("selected candidate differs from saved ranking")
            candidates = {
                entry["candidate_id"]: entry for entry in workload["candidates"]
            }
            pool = {identity: candidates[identity] for identity in expected}
            for candidate in pool.values():
                _reject_label_keys(candidate["configuration"])
            selected = candidates[ranking[0]]
            if prediction.get("selected_runtime_ms") != selected["runtime_ms"]:
                raise ValueError("selected runtime differs from prepared data")
            values, basis = _scores(pool, ranking)
            metadata = _metadata(workload)
            row = {
                "model_id": prediction["model_id"],
                "model_type": prediction["model_type"],
                "workload_id": prediction["workload_id"],
                **{
                    key: metadata[key]
                    for key in (
                        "operation",
                        "operation_family",
                        "workload_key",
                        "dag_family_hash",
                        "target",
                        "metadata_complete",
                    )
                },
                "tensor_shapes": json.dumps(metadata["tensors"], separators=(",", ":")),
                "candidate_count": len(expected),
                "score_basis": basis,
                **values,
                "runtime_top_1_score": canonical_map[key]["top_1_score"],
                "runtime_top_5_score": canonical_map[key]["top_5_score"],
                "selected_candidate_id": ranking[0],
                "selected_recorded_cost_ms": selected["runtime_ms"],
                "selected_failure": _failure(selected)["is_failure"],
            }
            rows.append(row)
            details[key] = {**row, "candidate_ids": expected}
    examples = _case_studies(workloads, details, examples_per_outcome)
    groups = {
        "by_operation": ("model_id", "model_type", "operation", "score_basis"),
        "by_operation_family": (
            "model_id",
            "model_type",
            "operation_family",
            "score_basis",
        ),
        "by_candidate_count": (
            "model_id",
            "model_type",
            "candidate_count",
            "score_basis",
        ),
        "overall": ("model_id", "model_type", "score_basis"),
    }
    destination.mkdir(parents=True, exist_ok=True)
    _csv(destination / "per_workload.csv", rows, tuple(rows[0]))
    for name, keys in groups.items():
        grouped = _group(rows, keys)
        _csv(destination / f"{name}.csv", grouped, tuple(grouped[0]))
    _json(destination / "case_studies.json", examples)
    (destination / "case_studies.md").write_text(_case_markdown(examples))
    _json(
        destination / "analysis-manifest.json",
        {
            "schema_version": 1,
            "scope": "offline Jev-versus-Random metadata and case analysis",
            "api_requests_made": 0,
            "dataset_sha256": prepared_sha256,
            "predictions_sha256": _sha256(result_dir / "predictions.jsonl"),
            "manifest_sha256": _sha256(result_dir / "manifest.json"),
            "experiment_sha256": _sha256(result_dir / "experiment.yaml"),
            "workloads": len(identifiers),
            "prediction_rows": len(rows),
            "case_studies": len(examples),
            "score_basis": sorted({row["score_basis"] for row in rows}),
            "aggregation": {
                "macro": "equal weight per workload",
                "record_weighted": "weight by number of evaluated test candidates",
            },
            "operation_family_method": (
                "Coarse substring rules on actual operation metadata, not workload "
                "IDs. Fused programs retain the literal operation signature."
            ),
            "dag_family_hash_method": (
                "First workload_key element when it is a 32-character hex string. "
                "No family is inferred from an opaque workload_id."
            ),
            "candidate_size_comparison": (
                "Descriptive across naturally different workloads; "
                "not a controlled size ablation."
            ),
            "case_selection": (
                "post-hoc largest positive and negative paired Top-1 differences"
            ),
            "failure_costs": (
                "Explicit source error code where available, otherwise a 1e13-ms "
                "sentinel heuristic. Such costs are not physical runtimes."
            ),
            "global_tournament_pairwise_accuracy": "not computed",
            "missing_metadata": "retained and labelled unknown/incomplete",
        },
    )
    return destination
