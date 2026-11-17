"""Offline-first execution of reconstructed ResNet-50 network-score evaluation.

The shared collection runner owns request caching, immutable completion receipts,
and resumable execution. Its ordinary prefix-average selection metrics are
intermediate diagnostics; the separate paper endpoint/network metrics are the
reported outcomes for this protocol.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from pathlib import Path

from jev_compiler_eval.datasets.tenset_paper import (
    load_paper_index,
    load_paper_shard,
)
from jev_compiler_eval.evaluation import tenset_collection as collection
from jev_compiler_eval.evaluation.runner import ExperimentRunResult, _model_input
from jev_compiler_eval.evaluation.tenset_paper_metrics import (
    PaperTaskPool,
    compare_paper_network,
    export_paper_network_scores,
)
from jev_compiler_eval.models.base import BaseDecisionModel
from jev_compiler_eval.models.jev_tournament import (
    JevTournamentModel,
    _compact_json_length,
)

EXPERIMENT_PROTOCOL = "tenset_paper_resnet50_full_record_pool"
ALIGNMENT_STATUS = "unverified_original_paper_artifacts"
PROJECT_SEED = 0
SUMMARY_JSON = "paper-network-summary.json"
SUMMARY_CSV = "paper-network-summary.csv"
NETWORK_KEY = ["resnet_50", [[1, 3, 224, 224]]]
RANDOM_ID = "random-0"
JEV_ID = "jev-1.14.0-paper-resnet50-tournament-fields-long-context"
INPUT_PRICE_PER_MILLION = 0.042
EXTERNAL_REFERENCE = {
    "model": "MLP + Ranking loss",
    "top_1": 0.8823,
    "top_5": 0.9456,
    "paper_location": "Section5.2, Table4, ResNet50 column, printed page7",
    "source_url": (
        "https://datasets-benchmarks-proceedings.neurips.cc/paper_files/"
        "paper/2021/file/a684eceee76fc522773286a895bc8436-Paper-round1.pdf#page=7"
    ),
    "comparison_type": "external published reference only, not a paired baseline",
    "original_test_inventory_verified": False,
    "original_checkpoint_verified": False,
}


def _hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _plan(config_path, *, output_dir=None):
    plan = collection._load_plan(
        config_path, workload_limit=None, output_dir=output_dir
    )
    checked = load_paper_index(plan.index_path)
    if checked != plan.index:
        raise ValueError("paper collection index changed while loading configuration")
    provenance = checked["provenance"]
    if provenance.get("experiment_protocol") != EXPERIMENT_PROTOCOL:
        raise ValueError("the collection is not a paper ResNet-50 full-record pool")
    if provenance.get("reference_alignment_status") != ALIGNMENT_STATUS:
        raise ValueError("unsupported paper reference-alignment status")
    if provenance.get("network_key") != NETWORK_KEY:
        raise ValueError("paper runner requires ResNet-50 batch1 input224")
    if provenance.get("platform") != "platinum-8272":
        raise ValueError("paper runner requires the Platinum-8272 target platform")
    settings = plan.config.dataset.config
    required = {
        "candidate_limit": None,
        "seed": PROJECT_SEED,
        "split": "test",
        "require_program_metadata": True,
        "describe_config_fields": True,
        "max_input_bytes": None,
    }
    if any(settings.get(key, object()) != value for key, value in required.items()):
        raise ValueError(
            "paper config requires complete test pools, seed0, program metadata, "
            "field descriptions and no local byte truncation"
        )
    if plan.config.experiment.seed != PROJECT_SEED:
        raise ValueError("paper experiment seed must be the project seed0")
    if plan.config.metrics != ("top_1_score", "top_5_score"):
        raise ValueError("paper collection intermediates require Top-1/Top-5 only")
    specs = {spec.model_id: spec for spec in plan.config.models}
    if set(specs) != {RANDOM_ID, JEV_ID}:
        raise ValueError("paper experiment requires its pinned Jev and Random models")
    if specs[RANDOM_ID].model_type != "random" or dict(specs[RANDOM_ID].config) != {
        "seed": PROJECT_SEED
    }:
        raise ValueError("paper Random model must use the project seed0")
    pinned = {
        "model": "jev-1.14.0",
        "long_context": True,
        "order_seed": PROJECT_SEED,
        "candidate_ids": "opaque",
        "max_options": 255,
        "advance_per_question": 5,
    }
    jev = specs[JEV_ID]
    if jev.model_type != "jev-tournament" or any(
        jev.config.get(key) != value for key, value in pinned.items()
    ):
        raise ValueError("paper Jev model must retain its pinned tournament settings")
    if jev.config.get("include_context", True) is not True:
        raise ValueError("paper Jev evaluation requires the workload context")
    if plan.output_dir.is_relative_to(plan.index_path.parent):
        raise ValueError("result output must be outside the immutable paper collection")
    cache_path = Path(jev.config["cache_dir"]).resolve()
    if cache_path.is_relative_to(plan.index_path.parent):
        raise ValueError("request cache must be outside the immutable paper collection")
    state_path = plan.output_dir / collection.STATE_FILE
    if plan.output_dir.exists() and not plan.output_dir.is_dir():
        raise ValueError("paper result output must be a directory")
    if plan.output_dir.exists() and any(plan.output_dir.iterdir()):
        if not state_path.is_file():
            raise ValueError("nonempty paper output lacks its collection state")
        state = _read_json(state_path)
        if state.get("identity") != plan.identity:
            raise ValueError(
                "paper output requires unchanged configuration, data and code"
            )
        if state.get("status") == "complete" and state.get("files") != {
            name: _hash(plan.output_dir / name) for name in collection.BUNDLE_FILES
        }:
            raise ValueError("completed paper result checksums differ")
    return plan


def _pool(plan, entry):
    document = load_paper_shard(plan.index_path, entry)
    workload = document["workloads"][0]
    return PaperTaskPool(
        workload_id=entry["workload_id"],
        workload_key=entry["workload_key"],
        target=workload["context"]["target"],
        weight=entry["task_weight"],
        full_task_min_latency_seconds=entry["full_task_min_latency_seconds"],
        labels_by_candidate={
            row["candidate_id"]: row["official_throughput"]
            for row in workload["candidates"]
        },
    )


def _scenario_cost(case, config, *, largest):
    """Use the execution planner with hypothetical size-based promotions.

    This estimates a scenario, not a bound across every possible model choice.
    The promotion rule uses serialized configuration length only, never labels.
    """

    planner = JevTournamentModel.__new__(JevTournamentModel)
    BaseDecisionModel.__init__(planner, config)
    state, _, pool, _, configurations = planner._prepare_inputs(
        case.workload, case.candidates
    )
    state_tokens = planner._tokens(_compact_json_length(state))
    rounds, tokens, calls = 0, 0, 0
    while True:
        questions = planner._split(pool, configurations, state_tokens, rounds)
        planner._assert_progress(pool, questions)
        requests = planner._pack(questions, state_tokens)
        calls += len(requests)
        tokens += sum(
            state_tokens + sum(question.estimated_tokens for question in request)
            for request in requests
        )
        rounds += 1
        if len(questions) == 1:
            return {"rounds": rounds, "requests": calls, "input_tokens": tokens}
        survivors = set()
        for question in questions:
            order = sorted(
                question.aliases,
                key=lambda alias: _compact_json_length(configurations[alias]),
                reverse=largest,
            )
            survivors.update(order[: config["advance_per_question"]])
        pool = [alias for alias in pool if alias in survivors]


def _validate_price(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError("input price must be a finite nonnegative USD/million value")
    return float(value)


def preflight_tenset_paper(
    config_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    input_price_per_million: float = INPUT_PRICE_PER_MILLION,
) -> dict:
    """Validate every full-network shard without loading credentials or clients."""

    price = _validate_price(input_price_per_million)
    plan = _plan(config_path, output_dir=output_dir)
    expected_weights = {
        entry["workload_id"]: entry["task_weight"] for entry in plan.index["shards"]
    }
    selected = {entry["workload_id"] for entry in plan.entries}
    pools, rankings, per_task = [], {}, []
    total_candidates = 0
    first = {
        "requests": 0,
        "questions": 0,
        "input_tokens": 0,
        "max_estimated_request_tokens": 0,
    }
    scenarios = {
        name: {"requests": 0, "input_tokens": 0, "max_rounds_per_task": 0}
        for name in (
            "smallest_configuration_promotions",
            "largest_configuration_promotions",
        )
    }
    jev_config = next(
        spec.config for spec in plan.config.models if spec.model_id == JEV_ID
    )
    # Even a pilot validates the entire network inventory and every private pool.
    # Plan each selected case immediately, retaining only scalar plans and the
    # compact private label pools needed for network-wide metric validation.
    for position, entry in enumerate(plan.index["shards"]):
        pool = _pool(plan, entry)
        case, _ = collection._load_case(plan, position, entry)
        total_candidates += len(pool.labels_by_candidate)
        if entry["workload_id"] in selected:
            pools.append(pool)
            rankings[pool.workload_id] = list(pool.labels_by_candidate)
            packing = JevTournamentModel.plan(
                case.workload, case.candidates, jev_config
            )
            task = {
                "workload_id": pool.workload_id,
                "task_weight": pool.weight,
                "candidate_count": len(case.candidates),
                "model_input_sha256": hashlib.sha256(_model_input(case)).hexdigest(),
                "shard_sha256": entry["sha256"],
                "first_round": {
                    name: packing[name]
                    for name in (
                        "first_round_requests",
                        "first_round_questions",
                        "estimated_first_round_input_tokens",
                    )
                },
                "size_promotion_scenarios": {},
            }
            per_task.append(task)
            first["requests"] += packing["first_round_requests"]
            first["questions"] += packing["first_round_questions"]
            first["input_tokens"] += packing["estimated_first_round_input_tokens"]
            first["max_estimated_request_tokens"] = max(
                first["max_estimated_request_tokens"],
                packing["max_estimated_request_tokens"],
            )
            for name, largest in (
                ("smallest_configuration_promotions", False),
                ("largest_configuration_promotions", True),
            ):
                scenario = _scenario_cost(case, jev_config, largest=largest)
                task["size_promotion_scenarios"][name] = scenario
                scenarios[name]["requests"] += scenario["requests"]
                scenarios[name]["input_tokens"] += scenario["input_tokens"]
                scenarios[name]["max_rounds_per_task"] = max(
                    scenarios[name]["max_rounds_per_task"], scenario["rounds"]
                )
        # Do not retain graph/configuration objects until the next load. A large
        # network should require memory for only one serialized task at a time.
        del case
    pilot = plan.config.dataset.config.get("max_workloads") is not None or len(
        plan.entries
    ) != len(plan.index["shards"])
    # Exercise the same metric validator before any paid request. The dummy
    # ranking preserves source order and does not select candidates using labels.
    contract = compare_paper_network(
        pools,
        {"offline-contract-check": rankings},
        network_key=NETWORK_KEY,
        target=plan.index["provenance"]["target"],
        expected_task_weights=expected_weights,
        pilot=pilot,
    )
    for scenario in scenarios.values():
        scenario["remaining_input_tokens_after_first_round"] = (
            scenario["input_tokens"] - first["input_tokens"]
        )
        scenario["estimated_input_cost_usd"] = scenario["input_tokens"] * price / 1e6
        scenario["estimated_remaining_input_cost_usd"] = (
            scenario["remaining_input_tokens_after_first_round"] * price / 1e6
        )
    selected_candidates = sum(task["candidate_count"] for task in per_task)
    return {
        "experiment_protocol": EXPERIMENT_PROTOCOL,
        "reference_alignment_status": ALIGNMENT_STATUS,
        "scoring_alignment": "official pinned network evaluator reconstruction",
        "scope": "pilot_subset" if pilot else "full_network",
        "comparison_scope": "Jev versus Random on identical complete task pools",
        "network_key": NETWORK_KEY,
        "target": plan.index["provenance"]["target"],
        "project_seed": PROJECT_SEED,
        "project_seed_is_verified_paper_seed": False,
        "collection_path": str(plan.index_path),
        "collection_sha256": plan.identity["collection_index_sha256"],
        "effective_config_sha256": plan.identity["effective_config_sha256"],
        "execution_identity": plan.identity,
        "full_network_tasks": len(expected_weights),
        "full_network_total_candidates": total_candidates,
        "selected_tasks": len(plan.entries),
        "selected_candidates": selected_candidates,
        "selected_task_weight": sum(task["task_weight"] for task in per_task),
        "full_network_task_weight": sum(expected_weights.values()),
        "selected_tasks_manifest": per_task,
        "metric_contract_validated": True,
        "metric_contract_scope": contract.get("scope"),
        "intermediate_metrics_note": (
            "Collection top_1_score/top_5_score use prefix-average task metrics; "
            "paper-network-summary uses distinct endpoint/network metrics."
        ),
        "model_plans": {JEV_ID: {"first_round": first, "scenarios": scenarios}},
        "cost_assumptions": {
            "usd_per_million_input_tokens": price,
            "input_only": True,
            "output_cost_included": False,
            "cache_savings_assumed": False,
            "retries_included": False,
            "estimated_first_round_input_cost_usd": first["input_tokens"] * price / 1e6,
            "scenario_total_input_cost_range_usd": [
                min(row["estimated_input_cost_usd"] for row in scenarios.values()),
                max(row["estimated_input_cost_usd"] for row in scenarios.values()),
            ],
            "scenario_remaining_input_cost_range_usd": [
                min(
                    row["estimated_remaining_input_cost_usd"]
                    for row in scenarios.values()
                ),
                max(
                    row["estimated_remaining_input_cost_usd"]
                    for row in scenarios.values()
                ),
            ],
        },
        "warnings": [
            "Character estimates do not guarantee actual token counts or acceptance.",
            (
                "Later-round size-based promotion scenarios are estimates, "
                "not exhaustive bounds on choices or cost."
            ),
            "A price estimate does not authorize API calls; use --execute explicitly.",
            (
                "Original paper test inventory and trained checkpoint have not "
                "been independently matched."
            ),
        ],
        "external_reference": None if pilot else dict(EXTERNAL_REFERENCE),
        "output_dir": str(plan.output_dir),
        "api_requests_made": 0,
    }


def summarize_tenset_paper(config_path, *, output_dir=None):
    """Recompute endpoint network scores from a verified completed collection."""

    plan = _plan(config_path, output_dir=output_dir)
    state = _read_json(plan.output_dir / collection.STATE_FILE)
    if state.get("identity") != plan.identity or state.get("status") != "complete":
        raise ValueError("paper summary requires the unchanged completed collection")
    if state.get("files") != {
        name: _hash(plan.output_dir / name) for name in collection.BUNDLE_FILES
    }:
        raise ValueError("completed paper result checksums differ")
    entries = {entry["workload_id"]: entry for entry in plan.entries}
    specs = {spec.model_id: spec.model_type for spec in plan.config.models}
    rankings = {model: {} for model in specs}
    pools = {key: _pool(plan, entry) for key, entry in entries.items()}
    source = plan.output_dir / "predictions.jsonl"
    with source.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            model, workload_id = row.get("model_id"), row.get("workload_id")
            if (
                model not in specs
                or row.get("model_type") != specs[model]
                or workload_id not in entries
                or workload_id in rankings[model]
            ):
                raise ValueError("unexpected or duplicate paper prediction")
            entry = entries[workload_id]
            position = list(entries).index(workload_id)
            case, _ = collection._load_case(plan, position, entry)
            ids = [candidate.candidate_id for candidate in case.candidates]
            runtimes = {
                record.candidate_id: record.runtime_ms for record in case.runtimes
            }
            if (
                row.get("candidate_ids") != ids
                or row.get("runtime_by_candidate_ms") != runtimes
                or row.get("model_input_sha256")
                != hashlib.sha256(_model_input(case)).hexdigest()
            ):
                raise ValueError(
                    "paper prediction differs from its frozen complete pool"
                )
            rankings[model][workload_id] = row.get("predicted_ranking")
    pilot = plan.config.dataset.config.get("max_workloads") is not None or len(
        entries
    ) != len(plan.index["shards"])
    summary = compare_paper_network(
        list(pools.values()),
        rankings,
        network_key=NETWORK_KEY,
        target=plan.index["provenance"]["target"],
        expected_task_weights={
            entry["workload_id"]: entry["task_weight"] for entry in plan.index["shards"]
        },
        pilot=pilot,
        model_types=specs,
    )
    summary.update(
        {
            "experiment_protocol": EXPERIMENT_PROTOCOL,
            "reference_alignment_status": ALIGNMENT_STATUS,
            "scoring_alignment": "official pinned network evaluator reconstruction",
            "collection_index_sha256": _hash(plan.index_path),
            "predictions_sha256": _hash(source),
            "effective_config_sha256": plan.identity["effective_config_sha256"],
            "project_seed": PROJECT_SEED,
            "project_seed_is_verified_paper_seed": False,
            "external_reference": None if pilot else dict(EXTERNAL_REFERENCE),
            "intermediate_metrics_note": (
                "Canonical collection metrics are prefix-average diagnostics. "
                "Use the separately named endpoint network scores here."
            ),
        }
    )
    export_paper_network_scores(plan.output_dir, summary)
    return summary


def run_tenset_paper(
    config_path: str | Path,
    *,
    execute: bool = False,
    resume: bool = False,
    output_dir: str | Path | None = None,
    progress: Callable[[int, int, str, bool], None] | None = None,
    expected_identity: Mapping[str, object] | None = None,
) -> ExperimentRunResult:
    """Run the checked shared collection only after explicit execution choice."""

    if execute is not True:
        raise ValueError("live paper execution requires execute=True / --execute")
    preflight = preflight_tenset_paper(config_path, output_dir=output_dir)
    identity = preflight["execution_identity"]
    if expected_identity is not None and (
        not isinstance(expected_identity, Mapping)
        or identity != dict(expected_identity)
    ):
        raise ValueError(
            "paper execution differs from its validated preflight identity"
        )
    result = collection.run_tenset_collection(
        config_path,
        resume=resume,
        output_dir=output_dir,
        progress=progress,
        expected_identity=identity,
    )
    summarize_tenset_paper(config_path, output_dir=output_dir)
    return result
