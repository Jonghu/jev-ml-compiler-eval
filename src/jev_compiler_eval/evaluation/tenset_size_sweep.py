"""Freeze an offline, controlled candidate-size sweep over the audited E5 cohort.

Preparation writes configurations and an input manifest only. It never creates
an API client, changes the prepared data, or edits an existing result bundle.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import shlex
import tempfile
from collections.abc import Callable
from pathlib import Path
from statistics import mean

import yaml

from jev_compiler_eval.config import load_experiment_config
from jev_compiler_eval.datasets.tenset import (
    TenSetDataset,
    load_prepared,
    workload_split,
)
from jev_compiler_eval.evaluation.runner import (
    _model_input,
    _package_versions,
    run_experiment,
)
from jev_compiler_eval.metrics import resolve_metrics
from jev_compiler_eval.models.random import RandomModel
from jev_compiler_eval.types import DecisionResult

AUDITED_PREPARED_SHA256 = (
    "1f949c2c9f788b507c4e395925720c7beaa7063c3efb6ac09ae5ed4855db6275"
)
TARGET = "llvm -keys=cpu -link-params=0 -mcpu=core-avx2 -model=e5-2673"
COHORT_SIZE = 120
SEED = 42
METRICS = (
    "top_1_score",
    "top_5_score",
    "pairwise_accuracy",
    "pairwise_accuracy_tie_aware",
    "runtime_regret",
)
MODEL = "jev-1.14.0"


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def _critical_sources() -> dict[str, str]:
    """Freeze the decision, data, cache and metric code used by this sweep."""
    package = Path(__file__).resolve().parents[1]
    return {
        name: _sha256(package / name)
        for name in (
            "evaluation/tenset_size_sweep.py",
            "evaluation/runner.py",
            "datasets/tenset.py",
            "models/jev.py",
            "models/random.py",
            "models/base.py",
            "metrics.py",
            "types.py",
            "jev/client.py",
            "config/loader.py",
            "config/types.py",
        )
    }


def _dataset_config(path: Path, size: int | None) -> dict:
    return {
        "path": str(path),
        "split": "test",
        "candidate_limit": size,
        "max_workloads": COHORT_SIZE,
        "seed": SEED,
        "max_input_bytes": 262144 if size is not None else None,
        "require_program_metadata": True,
        "describe_config_fields": False,
    }


def _load_frozen_cohort(path: Path, expected_sha256: str) -> tuple[dict, dict, str]:
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise ValueError(
            "expected prepared SHA-256 must be 64 lowercase hex characters"
        )
    checksum = _sha256(path)
    if checksum != expected_sha256:
        raise ValueError("prepared checksum differs from the explicitly frozen input")
    document = load_prepared(path)
    if len(document["workloads"]) != COHORT_SIZE:
        raise ValueError(f"size sweep requires exactly {COHORT_SIZE} frozen workloads")
    for row in document["workloads"]:
        context = row.get("context", {})
        if (
            row.get("split") != "test"
            or context.get("target") != TARGET
            or not isinstance(context.get("workload_key"), str)
            or workload_split(context["workload_key"], seed=SEED) != "test"
        ):
            raise ValueError(
                "size sweep requires the E5 target and seed-42 test cohort"
            )
        if context.get("configuration_fields") is not None:
            raise ValueError(
                "the frozen size-sweep baseline must have no field definitions"
            )
        candidates = row.get("candidates", [])
        if len(candidates) < 50:
            raise ValueError("every frozen workload needs at least 50 candidates")
        for candidate in candidates:
            runtime = candidate.get("runtime_ms")
            if (
                type(runtime) not in (int, float)
                or not math.isfinite(runtime)
                or not 0 < runtime < 1e13
                or candidate.get("source_error_code", 0) != 0
                or "official_throughput" in candidate
            ):
                raise ValueError("size sweep requires valid-only measured schedules")
    complete = TenSetDataset(_dataset_config(path, None)).load()
    if len(complete) != COHORT_SIZE:
        raise ValueError("the frozen dataset did not yield the complete cohort")
    return document, {case.workload.workload_id: case for case in complete}, checksum


def _configuration(path: Path, size: int, results_root: Path) -> dict:
    return {
        "schema_version": 1,
        "experiment": {
            "name": f"tenset-e5-size-sweep-w{COHORT_SIZE}-c{size}-s42-long-context",
            "description": (
                "Controlled nested candidate pool; Jev versus Random; fixed "
                "workloads, target, metadata, ordering, and long-context mode"
            ),
            "seed": SEED,
            "output_dir": str(results_root / f"c{size}"),
        },
        "dataset": {"type": "tenset", "config": _dataset_config(path, size)},
        "candidate_selection": {"strategy": "all"},
        "models": [
            {"id": "random-42", "type": "random", "config": {"seed": SEED}},
            {
                "id": "jev-1.14.0-long-context",
                "type": "jev",
                "config": {
                    "model": MODEL,
                    "cache_dir": ".cache/jev-tenset-size-sweep-v3-long-context",
                    "max_retries": 0,
                    "request_timeout_seconds": 120.0,
                    "long_context": True,
                },
            },
        ],
        "metrics": list(METRICS),
        "execution": {"fail_fast": True},
    }


def _validate_baseline(directory: Path, checksum: str, cases: tuple) -> dict:
    """Verify an existing 50-schedule bundle before referencing it as an anchor."""
    config_path = directory / "experiment.yaml"
    config = load_experiment_config(config_path)
    data = config.dataset.config
    if (
        config.dataset.dataset_type != "tenset"
        or config.experiment.seed != SEED
        or data.get("split", "test") != "test"
        or data.get("candidate_limit", 32) != 50
        or data.get("max_workloads") != COHORT_SIZE
        or data.get("seed", 42) != SEED
        or data.get("require_program_metadata") is not True
        or data.get("describe_config_fields", False) is not False
        or set(config.metrics) != set(METRICS)
        or config.candidate_selection.strategy != "all"
        or config.candidate_selection.config
    ):
        raise ValueError("50-schedule baseline uses a different dataset protocol")
    model_types = {model.model_type: model for model in config.models}
    if len(config.models) != 2 or set(model_types) != {"random", "jev"}:
        raise ValueError("baseline must contain exactly Random and single-question Jev")
    random_spec, jev_spec = model_types["random"], model_types["jev"]
    settings = jev_spec.config
    if (
        random_spec.config != {"seed": SEED}
        or random_spec.model_id != "random-42"
        or jev_spec.model_id != "jev-1.14.0-long-context"
        or settings.get("model") != MODEL
        or settings.get("long_context", False) is not True
        or settings.get("include_context", True) is not True
        or settings.get("candidate_order", "original") != "original"
        or "order_seed" in settings
    ):
        raise ValueError("baseline model, context, presentation, or seed differs")
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("dataset_type") != "tenset"
        or manifest.get("seed") != SEED
        or manifest.get("dataset_provenance", {}).get("prepared_file_sha256")
        != checksum
        or set(manifest.get("model_ids", []))
        != {random_spec.model_id, jev_spec.model_id}
    ):
        raise ValueError("baseline manifest does not bind the frozen data and models")
    lookup = {case.workload.workload_id: case for case in cases}
    specifications = {model.model_id: model.model_type for model in config.models}
    seen = set()
    metrics = resolve_metrics(METRICS)
    random_model = RandomModel({"seed": SEED})
    predictions_path = directory / "predictions.jsonl"
    with predictions_path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            identity = (row.get("model_id"), row.get("workload_id"))
            if (
                identity in seen
                or identity[0] not in specifications
                or identity[1] not in lookup
                or row.get("model_type") != specifications[identity[0]]
            ):
                raise ValueError("baseline has missing, duplicate, or unexpected rows")
            seen.add(identity)
            case = lookup[identity[1]]
            candidate_ids = [candidate.candidate_id for candidate in case.candidates]
            runtimes = {
                record.candidate_id: record.runtime_ms for record in case.runtimes
            }
            if (
                row.get("candidate_ids") != candidate_ids
                or row.get("runtime_by_candidate_ms") != runtimes
                or row.get("model_input_sha256")
                != hashlib.sha256(_model_input(case)).hexdigest()
                or row.get("model_input_bytes") != len(_model_input(case))
            ):
                raise ValueError(
                    "baseline candidate pool, runtimes, or input hash differs"
                )
            ranking = row.get("predicted_ranking")
            if (
                not isinstance(ranking, list)
                or len(ranking) != len(candidate_ids)
                or set(ranking) != set(candidate_ids)
            ):
                raise ValueError("baseline ranking does not cover the frozen pool")
            metadata = row.get("metadata", {})
            result = DecisionResult(
                tuple(ranking), row.get("decision_latency_ms"), metadata
            )
            if (
                row.get("selected_candidate_id") != result.selected_candidate_id
                or row.get("selected_runtime_ms")
                != runtimes[result.selected_candidate_id]
            ):
                raise ValueError("baseline selected outcome differs from its ranking")
            if row["model_type"] == "random":
                if (
                    metadata != {"seed": SEED}
                    or result.ranked_candidate_ids
                    != random_model.rank(
                        case.workload, case.candidates
                    ).ranked_candidate_ids
                ):
                    raise ValueError(
                        "baseline Random ranking does not reproduce seed 42"
                    )
            elif (
                metadata.get("requested_model") != MODEL
                or metadata.get("resolved_model") != MODEL
                or metadata.get("long_context") is not True
                or metadata.get("include_context", True) is not True
                or metadata.get("candidate_order", "original") != "original"
            ):
                raise ValueError(
                    "baseline response metadata does not match Jev settings"
                )
            saved = row.get("metrics", {})
            if set(saved) != set(METRICS) or any(
                type(saved[name]) not in (int, float)
                or not math.isfinite(saved[name])
                or not math.isclose(
                    saved[name], function(case, result), rel_tol=1e-10, abs_tol=1e-12
                )
                for name, function in metrics.items()
            ):
                raise ValueError(
                    "baseline metrics do not recompute from its predictions"
                )
    expected = {(model, workload) for model in specifications for workload in lookup}
    if seen != expected:
        raise ValueError("baseline is missing frozen model/workload predictions")
    return {
        "status": "verified_existing_50_schedule_anchor",
        "results_dir": str(directory),
        "files": {
            name: _sha256(directory / name)
            for name in ("experiment.yaml", "manifest.json", "predictions.jsonl")
        },
        "note": (
            "An archived response is an anchor, not a fresh API repeat. "
            "This file was not modified or counted as a new run."
        ),
    }


def _publish(path: Path, data: bytes) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != data:
            raise FileExistsError(
                f"refusing to overwrite different sweep artifact: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".size-sweep-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def prepare_tenset_size_sweep(
    prepared_path: str | Path = "data/tenset/prepared-w120-c50-s42-metadata.json",
    *,
    output_dir: str | Path = "experiments/tenset/local/size-sweep",
    results_root: str | Path = "results/tenset/size-sweep",
    sizes: tuple[int, ...] = (16, 32, 50),
    expected_prepared_sha256: str = AUDITED_PREPARED_SHA256,
    baseline_results: str | Path | None = None,
) -> Path:
    """Validate and freeze nested pools and commands without issuing decisions."""
    if (
        not isinstance(sizes, (tuple, list))
        or len(sizes) < 2
        or any(type(size) is not int or not 2 <= size <= 50 for size in sizes)
        or list(sizes) != sorted(set(sizes))
        or sizes[-1] != 50
    ):
        raise ValueError(
            "sizes must be increasing distinct integers in [2, 50], ending at 50"
        )
    prepared = Path(prepared_path).resolve()
    destination, results_root = (
        Path(output_dir).absolute(),
        Path(results_root).absolute(),
    )
    manifest_path = destination / "manifest.json"
    if destination.is_symlink():
        raise ValueError("sweep output directory must not be a symlink")
    _, complete, checksum = _load_frozen_cohort(prepared, expected_prepared_sha256)
    pools = {
        size: TenSetDataset(_dataset_config(prepared, size)).load() for size in sizes
    }
    for size, cases in pools.items():
        if [case.workload.workload_id for case in cases] != list(complete):
            raise ValueError("candidate budgets changed the frozen workload cohort")
        for case in cases:
            full = complete[case.workload.workload_id]
            if (
                len(case.candidates) != size
                or tuple(case.candidates) != tuple(full.candidates[:size])
                or tuple(case.runtimes) != tuple(full.runtimes[:size])
            ):
                raise ValueError("candidate budgets are not nested prefixes")
    anchor = (
        _validate_baseline(Path(baseline_results).resolve(), checksum, pools[50])
        if baseline_results is not None
        else None
    )
    pending = {}
    budgets = []
    for size in sizes:
        entry = {
            "candidates_per_workload": size,
            "workloads": [
                {
                    "workload_id": case.workload.workload_id,
                    "candidate_ids": [
                        candidate.candidate_id for candidate in case.candidates
                    ],
                    "model_input_sha256": hashlib.sha256(
                        _model_input(case)
                    ).hexdigest(),
                }
                for case in pools[size]
            ],
        }
        if size == 50 and anchor:
            entry.update({"action": "reuse_verified_anchor", **anchor})
        else:
            config = _configuration(prepared, size, results_root)
            path = destination / f"jev-vs-random-w{COHORT_SIZE}-c{size}-s42.yaml"
            pending[path] = yaml.safe_dump(config, sort_keys=False).encode()
            member_command = (
                "uv run python scripts/run_tenset_size_sweep.py --manifest "
                + shlex.quote(str(manifest_path))
                + f" --candidate-count {size}"
            )
            entry.update(
                {
                    "action": "new_run_prepared_not_executed",
                    "config_path": str(path),
                    "config_sha256": hashlib.sha256(pending[path]).hexdigest(),
                    "results_dir": config["experiment"]["output_dir"],
                    "preflight_command": member_command + " --preflight-only",
                    "run_command": member_command,
                }
            )
        budgets.append(entry)
    if _sha256(prepared) != checksum:
        raise ValueError("prepared file changed during size-sweep validation")
    result_paths = " ".join(shlex.quote(row["results_dir"]) for row in budgets)
    manifest = {
        "schema_version": 1,
        "experiment_protocol": "tenset_controlled_candidate_size_sweep",
        "status": "prepared_no_api_calls",
        "prepared_path": str(prepared),
        "prepared_sha256": checksum,
        "results_root": str(results_root),
        "preparation_module_sha256": _sha256(Path(__file__)),
        "implementation_sources": _critical_sources(),
        "python_version": platform.python_version(),
        "package_versions": _package_versions(),
        "target": TARGET,
        "seed": SEED,
        "model": MODEL,
        "long_context": True,
        "field_descriptions": False,
        "candidate_presentation": "original seeded-hash dataset order",
        "metrics": list(METRICS),
        "budgets": budgets,
        "analysis_command": (
            "uv run python scripts/analyze_tenset.py --results "
            + result_paths
            + " --dataset "
            + shlex.quote(str(prepared))
            + " --output-dir "
            + shlex.quote(str(results_root / "analysis"))
        ),
        "analysis_note": (
            "Use by_candidate_count.csv for size comparisons. overall.csv pools "
            "budgets and is not a controlled estimate of a candidate-size effect."
        ),
        "plot_command": (
            "uv run python scripts/plot_tenset_size_sweep.py --manifest "
            + shlex.quote(str(manifest_path))
            + " --analysis-dir "
            + shlex.quote(str(results_root / "analysis"))
            + " --output-dir "
            + shlex.quote(str(results_root / "size-comparison"))
        ),
        "repeatability_note": (
            "Fixed-seed/cached replay is reproducible but is not a fresh Jev "
            "replication. Model provider variation is not controlled by this sweep."
        ),
    }
    pending[manifest_path] = _json(manifest)
    for path, data in pending.items():
        if path.exists() and path.read_bytes() != data:
            raise FileExistsError(
                f"refusing to overwrite different sweep artifact: {path}"
            )
    for path, data in pending.items():
        _publish(path, data)
    return manifest_path


def validate_tenset_size_sweep(manifest_path: str | Path, candidate_count: int) -> dict:
    """Recheck frozen data, every pool, and config bytes before any live execution."""
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != 1
        or manifest.get("experiment_protocol")
        != "tenset_controlled_candidate_size_sweep"
        or manifest.get("status") != "prepared_no_api_calls"
        or manifest.get("preparation_module_sha256") != _sha256(Path(__file__))
        or manifest.get("implementation_sources") != _critical_sources()
        or manifest.get("python_version") != platform.python_version()
        or manifest.get("package_versions") != _package_versions()
        or manifest.get("seed") != SEED
        or manifest.get("target") != TARGET
        or manifest.get("model") != MODEL
        or manifest.get("long_context") is not True
        or manifest.get("field_descriptions") is not False
        or manifest.get("metrics") != list(METRICS)
    ):
        raise ValueError("size-sweep manifest or preparation implementation changed")
    budgets = manifest.get("budgets")
    if not isinstance(budgets, list) or len(budgets) < 2:
        raise ValueError("size-sweep manifest has no candidate budgets")
    sizes = [entry.get("candidates_per_workload") for entry in budgets]
    if (
        any(type(size) is not int or not 2 <= size <= 50 for size in sizes)
        or sizes != sorted(set(sizes))
        or sizes[-1] != 50
        or type(candidate_count) is not int
        or candidate_count not in sizes
    ):
        raise ValueError("candidate count is not one of the frozen size-sweep budgets")
    prepared = Path(manifest["prepared_path"])
    _, complete, checksum = _load_frozen_cohort(prepared, manifest["prepared_sha256"])
    chosen = None
    for entry, size in zip(budgets, sizes, strict=True):
        cases = TenSetDataset(_dataset_config(prepared, size)).load()
        expected = [
            {
                "workload_id": case.workload.workload_id,
                "candidate_ids": [
                    candidate.candidate_id for candidate in case.candidates
                ],
                "model_input_sha256": hashlib.sha256(_model_input(case)).hexdigest(),
            }
            for case in cases
        ]
        if [case.workload.workload_id for case in cases] != list(complete) or entry.get(
            "workloads"
        ) != expected:
            raise ValueError(
                "frozen size-sweep case IDs, candidate prefixes, or inputs differ"
            )
        if entry.get("action") == "reuse_verified_anchor":
            if size != 50:
                raise ValueError(
                    "only the 50-schedule anchor may reuse existing results"
                )
            verified = _validate_baseline(Path(entry["results_dir"]), checksum, cases)
            if any(entry.get(key) != value for key, value in verified.items()):
                raise ValueError(
                    "the reused 50-schedule result changed after preparation"
                )
        elif entry.get("action") == "new_run_prepared_not_executed":
            config_path = Path(entry["config_path"])
            if _sha256(config_path) != entry.get("config_sha256"):
                raise ValueError("frozen size-sweep configuration bytes changed")
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            if config != _configuration(prepared, size, Path(manifest["results_root"])):
                raise ValueError(
                    "size-sweep configuration changes a controlled setting"
                )
        else:
            raise ValueError("unknown size-sweep member action")
        if size == candidate_count:
            chosen = {
                "candidate_count": size,
                "workloads": len(cases),
                "prepared_sha256": checksum,
                "action": entry["action"],
                "config_path": entry.get("config_path"),
                "output_dir": entry["results_dir"],
                "max_model_input_bytes": max(len(_model_input(case)) for case in cases),
                "api_requests_made": 0,
            }
    if _sha256(prepared) != checksum:
        raise ValueError("prepared file changed during size-sweep execution validation")
    assert chosen is not None
    return chosen


def run_tenset_size_sweep_member(
    manifest_path: str | Path,
    candidate_count: int,
    *,
    preflight_only: bool = False,
    load_environment: Callable[[], None] | None = None,
):
    """Execute one explicitly selected member after all offline identity guards."""
    if type(preflight_only) is not bool:
        raise ValueError("preflight_only must be a boolean")
    member = validate_tenset_size_sweep(manifest_path, candidate_count)
    if member["action"] == "reuse_verified_anchor":
        return {**member, "status": "verified_existing_anchor_no_live_execution"}
    if preflight_only:
        return {**member, "status": "offline_preflight_passed"}
    if load_environment is not None:
        load_environment()
        # Credential loading must not hide an intervening data/configuration edit.
        member = validate_tenset_size_sweep(manifest_path, candidate_count)
    return run_experiment(member["config_path"])


def plot_tenset_size_sweep(
    manifest_path: str | Path,
    analysis_dir: str | Path,
    output_dir: str | Path,
) -> Path:
    """Plot budget-specific means only after checking the same paired cohort."""
    manifest_path = Path(manifest_path).resolve()
    plan = json.loads(manifest_path.read_text(encoding="utf-8"))
    budgets = plan.get("budgets", [])
    if not budgets:
        raise ValueError("size-sweep plot requires a frozen budget manifest")
    validate_tenset_size_sweep(manifest_path, budgets[0].get("candidates_per_workload"))
    analysis_dir, output_dir = Path(analysis_dir), Path(output_dir)
    summary_path = analysis_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("prepared_sha256") != plan["prepared_sha256"]
        or set(summary.get("metrics", [])) != set(METRICS)
        or {str(Path(path).resolve()) for path in summary.get("result_bundles", [])}
        != {entry["results_dir"] for entry in budgets}
        or len(summary.get("result_bundles", [])) != len(budgets)
    ):
        raise ValueError(
            "analysis does not describe the frozen data, budgets and metrics"
        )
    model_types = {"random-42": "random", "jev-1.14.0-long-context": "jev"}
    expected = {
        (entry["candidates_per_workload"], model_id, row["workload_id"])
        for entry in budgets
        for model_id in model_types
        for row in entry["workloads"]
    }
    per_workload_path = analysis_dir / "per_workload.csv"
    with per_workload_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    groups = {}
    seen = set()
    for row in rows:
        try:
            size = int(row["candidate_count"])
            identity = (size, row["model_id"], row["workload_id"])
            if (
                identity not in expected
                or identity in seen
                or row["model_type"] != model_types.get(row["model_id"])
                or int(row["seed"]) != SEED
            ):
                raise ValueError("unexpected, duplicated, or unpaired workload")
            scores = {name: float(row[name]) for name in METRICS}
            if any(
                not math.isfinite(value)
                or value < 0
                or (name != "runtime_regret" and value > 1)
                for name, value in scores.items()
            ):
                raise ValueError("invalid metric value")
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid size-sweep analysis row: {error}") from error
        seen.add(identity)
        groups.setdefault((size, row["model_id"]), []).append(scores)
    if seen != expected:
        raise ValueError("size-sweep analysis is missing a model, budget or workload")
    plot_rows = []
    for (size, model_id), members in sorted(groups.items()):
        plot_rows.append(
            {
                "candidate_count": size,
                "model_id": model_id,
                "model_type": model_types[model_id],
                "workloads": len(members),
                **{name: mean(row[name] for row in members) for name in METRICS},
            }
        )
    aggregate_path = analysis_dir / "by_candidate_count.csv"
    with aggregate_path.open(encoding="utf-8", newline="") as stream:
        aggregates = list(csv.DictReader(stream))
    indexed = {}
    for row in aggregates:
        key = (int(row["candidate_count"]), row["model_id"])
        if key in indexed:
            raise ValueError("duplicate candidate-count aggregate")
        indexed[key] = row
    if set(indexed) != set(groups):
        raise ValueError("candidate-count aggregates do not cover the frozen budgets")
    for expected_row in plot_rows:
        row = indexed[expected_row["candidate_count"], expected_row["model_id"]]
        if (
            row["model_type"] != expected_row["model_type"]
            or int(row["decisions"]) != COHORT_SIZE
            or int(row["unique_workloads"]) != COHORT_SIZE
            or any(
                not math.isclose(float(row[name]), expected_row[name], abs_tol=1e-12)
                for name in METRICS
            )
        ):
            raise ValueError(
                "candidate-count means disagree with paired per-workload data"
            )
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("plots require uv sync --locked --extra tenset") from error
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = {
        "top_1_score": "Top-1 Score",
        "top_5_score": "Top-5 Score",
        "pairwise_accuracy": "Pairwise Accuracy",
        "pairwise_accuracy_tie_aware": "Tie-aware Pairwise Accuracy",
        "runtime_regret": "Runtime regret",
    }
    sizes = [entry["candidates_per_workload"] for entry in budgets]
    figure, axes = plt.subplots(2, 3, figsize=(13, 7.4))
    for axis, metric in zip(axes.flat, METRICS):
        for model, color, label in (
            ("random-42", "#7E8B99", "Random"),
            ("jev-1.14.0-long-context", "#087F8C", "Jev"),
        ):
            values = [
                next(
                    row[metric]
                    for row in plot_rows
                    if row["model_id"] == model and row["candidate_count"] == size
                )
                for size in sizes
            ]
            axis.plot(sizes, values, marker="o", color=color, label=label)
        axis.set_title(labels[metric])
        axis.set_xlabel("Candidates per workload")
        axis.set_xticks(sizes)
        axis.set_ylabel(
            "Lower is better" if metric == "runtime_regret" else "Higher is better"
        )
        axis.set_ylim(bottom=0)
        if metric != "runtime_regret":
            axis.set_ylim(0, 1)
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    axes.flat[-1].axis("off")
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="upper right", frameon=False)
    synthetic = summary.get("data_status") == "synthetic_fixture"
    heading = "SYNTHETIC FIXTURE — not benchmark results\n" if synthetic else ""
    figure.suptitle(
        heading + "TenSet controlled candidate-size comparison",
        fontsize=15,
    )
    figure.text(
        0.5,
        0.025,
        f"Same {COHORT_SIZE} E5 workloads; seed 42; Jev 1.14.0; long context; "
        "no field definitions.\nEach budget has its own candidate-pool oracle; "
        "curves use per-budget means, not overall pooled scores.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.08, 1, 0.92))
    for suffix in ("png", "svg"):
        figure.savefig(
            output_dir / f"size-sweep-metrics.{suffix}",
            dpi=200,
            metadata={"Date": None} if suffix == "svg" else {},
        )
    plt.close(figure)
    data_path = output_dir / "size-sweep-plot-data.csv"
    with data_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(plot_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(plot_rows)
    method = {
        "data_status": "synthetic_fixture" if synthetic else "source_records",
        "scope": (
            "controlled nested candidate pools over the same frozen workload cohort"
        ),
        "aggregation": "workload macro mean within each candidate budget",
        "workloads_per_budget": COHORT_SIZE,
        "candidate_budgets": sizes,
        "prepared_sha256": plan["prepared_sha256"],
        "manifest_sha256": _sha256(manifest_path),
        "analysis_summary_sha256": _sha256(summary_path),
        "per_workload_sha256": _sha256(per_workload_path),
        "by_candidate_count_sha256": _sha256(aggregate_path),
        "plot_data_sha256": _sha256(data_path),
        "overall_pooled_scores_used": False,
        "note": (
            "Fixed archived/cached responses do not measure fresh API repeatability."
        ),
    }
    (output_dir / "size-sweep-plot-method.json").write_bytes(_json(method))
    return output_dir
