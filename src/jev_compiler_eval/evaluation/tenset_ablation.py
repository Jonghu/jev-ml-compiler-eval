"""Offline-first, frozen TenSet context ablations and filtered network follow-up.

The new protocol deliberately leaves historical experiments and shared models
unchanged. Every model decision is a single Choice question. Complete pools and
private labels remain available to scoring after structural diversity pruning.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from random import Random

import yaml

from jev_compiler_eval.datasets.tenset import TenSetDataset, _reject_label_keys
from jev_compiler_eval.datasets.tenset_ablation_descriptions import ablation_context
from jev_compiler_eval.datasets.tenset_collection import _publish, _read_json, _sha256
from jev_compiler_eval.datasets.tenset_paper import load_paper_index, load_paper_shard
from jev_compiler_eval.evaluation.tenset_ablation_features import (
    FEATURE_VERSION,
    feature_signature,
    select_diverse_candidates,
)
from jev_compiler_eval.evaluation.tenset_paper_metrics import (
    THROUGHPUT_FLOOR,
    _float32,
)
from jev_compiler_eval.metrics import (
    best_of_top_k_score,
    pairwise_accuracy,
    pairwise_accuracy_tie_aware,
    runtime_regret,
    top_1_score,
    top_5_score,
)
from jev_compiler_eval.models.jev_tournament import JevTournamentModel, RateLimiter
from jev_compiler_eval.models.random import RandomModel
from jev_compiler_eval.types import (
    Candidate,
    DecisionResult,
    EvaluationCase,
    RuntimeRecord,
    Workload,
)

PROTOCOL = "tenset_context_ablation_v1"
VARIANTS = ("a0", "a1", "a2", "a3", "a4-asc", "a4-desc", "a5")
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _positive(value: object, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


def _integer(value: object, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _publish_text(path: Path, content: str) -> None:
    """Publish without replacement, including after an interrupted earlier write."""
    data = content.encode()
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != data:
            raise FileExistsError(f"completed artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".ablation-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _sources() -> dict[str, str]:
    package = Path(__file__).resolve().parents[1]
    names = (
        "evaluation/tenset_ablation.py",
        "evaluation/tenset_ablation_features.py",
        "datasets/tenset_ablation_descriptions.py",
        "datasets/tenset_descriptions.py",
        "datasets/tenset.py",
        "datasets/tenset_paper.py",
        "datasets/tenset_collection.py",
        "evaluation/tenset_paper_metrics.py",
        "models/jev_tournament.py",
        "models/jev.py",
        "models/random.py",
        "models/base.py",
        "jev/client.py",
        "metrics.py",
        "types.py",
    )
    return {
        **{name: _sha256(package / name) for name in names},
        "uv.lock": _sha256(Path(__file__).resolve().parents[3] / "uv.lock"),
    }


class AblationChoiceModel(JevTournamentModel):
    """Use the shared planner/client with an explicitly frozen presentation order."""

    def _prepare_inputs(self, workload, candidates):
        if len(candidates) < 2:
            raise ValueError("ablation questions require at least two candidates")
        presented = list(candidates)
        identifiers = [candidate.candidate_id for candidate in presented]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("ablation candidate IDs must be unique")
        width = len(str(len(presented)))
        aliases = [f"c{number + 1:0{width}d}" for number in range(len(presented))]
        state = {"workload_id": workload.workload_id, "context": workload.context}
        _reject_label_keys(workload.context)
        for candidate in presented:
            _reject_label_keys(candidate.configuration)
        return (
            state,
            presented,
            aliases,
            dict(zip(aliases, identifiers, strict=True)),
            {
                alias: candidate.configuration
                for alias, candidate in zip(aliases, presented, strict=True)
            },
        )


def _model_config(study: Mapping, phase: str, repeat: int) -> dict:
    return {
        **study["model"],
        "order_seed": study["seed"],
        "candidate_ids": "opaque",
        "cache_dir": str(
            Path(study["output_dir"]) / "jev-cache" / phase / f"rep{repeat}"
        ),
    }


def _phase_cases(study: Mapping, phase: str) -> Iterator[tuple[EvaluationCase, dict]]:
    settings = study["phases"][phase]
    path = Path(settings["path"])
    if _sha256(path) != settings["expected_sha256"]:
        raise ValueError(f"{phase} source checksum changed; prepare a separate study")
    if phase == "small":
        cases = TenSetDataset(
            {
                "path": str(path),
                "split": "test",
                "seed": study["seed"],
                "candidate_limit": settings["candidate_limit"],
                "max_workloads": None,
                "max_input_bytes": None,
                "require_program_metadata": True,
                "describe_config_fields": False,
            }
        ).load()
        if len(cases) != settings["workloads"]:
            raise ValueError("small phase must contain its exact frozen workload count")
        for case in cases:
            if len(case.candidates) != settings["candidate_limit"]:
                raise ValueError(
                    "small phase needs the same candidate count per workload"
                )
            yield case, {"task_weight": 1, "runtime_basis": "measured_milliseconds"}
        return
    index = load_paper_index(path)
    if len(index["shards"]) != settings["workloads"]:
        raise ValueError("full phase must contain every expected ResNet-50 task")
    for entry in index["shards"]:
        document = load_paper_shard(path, entry)
        row = document["workloads"][0]
        normalizer = _float32(entry["full_task_min_latency_seconds"], "task minimum")
        candidates = tuple(
            Candidate(item["candidate_id"], item["configuration"])
            for item in row["candidates"]
        )
        runtimes = tuple(
            RuntimeRecord(
                row["workload_id"],
                item["candidate_id"],
                normalizer
                / max(
                    _float32(item["official_throughput"], "throughput"),
                    THROUGHPUT_FLOOR,
                )
                * 1000,
            )
            for item in row["candidates"]
        )
        yield (
            EvaluationCase(
                Workload(row["workload_id"], row["context"]), candidates, runtimes
            ),
            {
                "task_weight": entry["task_weight"],
                "paper_normalizer_ms": normalizer * 1000,
                "runtime_basis": "paper_f32_recovered_throughput_floor",
                "source_shard_sha256": entry["sha256"],
                "workload_key": entry["workload_key"],
            },
        )


def _descriptor(case: EvaluationCase, extra: Mapping) -> dict:
    return {
        "workload_id": case.workload.workload_id,
        "candidate_ids": [candidate.candidate_id for candidate in case.candidates],
        "input_sha256": _fingerprint(
            {
                "context": case.workload.context,
                "candidates": [
                    [candidate.candidate_id, candidate.configuration]
                    for candidate in case.candidates
                ],
            }
        ),
        "labels_sha256": _fingerprint(
            [[record.candidate_id, record.runtime_ms] for record in case.runtimes]
        ),
        **extra,
    }


def _presentation(case: EvaluationCase, variant: str, keep: int, seed: int):
    context = ablation_context(
        case.workload.context,
        candidate_definitions=variant in ("a1", "a3", "a4-asc", "a4-desc", "a5"),
        graph_specification=variant in ("a2", "a3", "a4-asc", "a4-desc", "a5"),
    )
    presented = list(case.candidates)
    Random(f"{seed}:{case.workload.workload_id}").shuffle(presented)
    if variant in ("a4-asc", "a4-desc"):
        presented.sort(key=feature_signature, reverse=variant == "a4-desc")
    elif variant == "a5":
        presented = select_diverse_candidates(
            presented, keep, seed, case.workload.workload_id
        )
    return Workload(case.workload.workload_id, context), tuple(presented)


def _question_plan(
    study: Mapping, phase: str, variant: str, case: EvaluationCase, expected=None
):
    workload, shown = _presentation(
        case, variant, study["phases"][phase]["keep"], study["seed"]
    )
    plan = AblationChoiceModel.plan(workload, shown, _model_config(study, phase, 1))
    if plan["first_round_questions"] != 1 or plan["first_round_requests"] != 1:
        raise ValueError(
            f"{phase}/{variant}/{workload.workload_id} needs multiple questions; "
            "this controlled study requires one question. Revise budgets or prepare "
            "a separately documented smaller common cohort before any paid calls."
        )
    if expected is not None and _input_plan(workload, shown, plan) != expected:
        raise ValueError(
            "model presentation or token estimate differs from frozen plan"
        )
    return workload, shown, plan


def _input_plan(workload, shown, plan):
    return {
        "shown_candidate_ids": [candidate.candidate_id for candidate in shown],
        "estimated_input_tokens": plan["estimated_first_round_input_tokens"],
        "model_input_sha256": _fingerprint(
            [workload.context, [[c.candidate_id, c.configuration] for c in shown]]
        ),
    }


def _read_settings(path: Path) -> dict:
    settings = yaml.safe_load(path.read_text())
    if (
        not isinstance(settings, dict)
        or set(settings) != {"schema_version", "study", "model", "pricing", "phases"}
        or settings["schema_version"] != 1
    ):
        raise ValueError("invalid ablation study configuration")
    header = settings["study"]
    if set(header) != {"name", "seed", "repeats", "output_dir"}:
        raise ValueError("invalid study header")
    if (
        type(header["seed"]) is not int
        or header["seed"] != 42
        or type(header["repeats"]) is not int
        or header["repeats"] != 3
    ):
        raise ValueError("the ablation protocol fixes seed 42 and three repeats")
    if not isinstance(header["name"], str) or not header["name"].strip():
        raise ValueError("study name must not be empty")
    if (
        settings["model"].get("model") != "jev-1.14.0"
        or settings["model"].get("long_context") is not True
    ):
        raise ValueError("TenSet ablation fixes Jev 1.14.0 with long context")
    model_keys = {
        "model",
        "long_context",
        "max_retries",
        "request_timeout_seconds",
        "max_options",
        "advance_per_question",
        "max_request_tokens",
        "max_state_question_tokens",
        "chars_per_token",
        "max_requests_per_minute",
        "max_tokens_per_second",
    }
    if set(settings["model"]) != model_keys:
        raise ValueError(
            "model settings must retain the ablation's context and ID rules"
        )
    if set(settings["pricing"]) != {"input_per_million_usd"}:
        raise ValueError("invalid pricing configuration")
    _positive(settings["pricing"]["input_per_million_usd"], "input price")
    if set(settings["phases"]) != {"small", "full"}:
        raise ValueError("both small and full phases are required")
    for phase, value in settings["phases"].items():
        required = {"path", "workloads", "candidate_limit", "keep", "variants"}
        if phase == "small":
            required.add("expected_sha256")
        if set(value) != required:
            raise ValueError(f"invalid {phase} phase settings")
        _integer(value["workloads"], "workloads")
        if _integer(value["keep"], "keep") < 5:
            raise ValueError("filter keep must be at least five for Top-5 scoring")
        expected_variants = list(VARIANTS) if phase == "small" else ["a5"]
        if value["variants"] != expected_variants:
            raise ValueError(
                f"{phase} must retain the protocol's complete variant inventory"
            )
        if phase == "small":
            _integer(value["candidate_limit"], "candidate_limit")
        elif value["candidate_limit"] is not None:
            raise ValueError("full phase must score complete source candidate pools")
    return settings


def prepare_ablation(config_path: str | Path) -> Path:
    """Freeze both inputs and all request plans without loading .env or a client."""
    config_path = Path(config_path).resolve()
    settings = _read_settings(config_path)
    header = settings["study"]
    root = (PROJECT_ROOT / header["output_dir"]).resolve()
    study = {
        "protocol": PROTOCOL,
        "name": header["name"],
        "seed": header["seed"],
        "repeats": 3,
        "output_dir": str(root),
        "model": settings["model"],
        "pricing": settings["pricing"],
        "feature_version": FEATURE_VERSION,
        "implementation": _sources(),
        "phases": {},
    }
    for phase, value in settings["phases"].items():
        source = (PROJECT_ROOT / value["path"]).resolve()
        if root == source or root.is_relative_to(source.parent):
            raise ValueError(
                "study results must be outside the source dataset directory"
            )
        if not source.is_file():
            raise FileNotFoundError(
                f"{phase} dataset is missing: {source}; prepare it first"
            )
        expected = value.get("expected_sha256", _sha256(source))
        study["phases"][phase] = {
            **value,
            "path": str(source),
            "expected_sha256": expected,
            "cases": [],
        }
        for case, extra in _phase_cases(study, phase):
            descriptor = _descriptor(case, extra)
            descriptor["plans"] = {}
            for variant in value["variants"]:
                workload, shown, plan = _question_plan(study, phase, variant, case)
                descriptor["plans"][variant] = _input_plan(workload, shown, plan)
            study["phases"][phase]["cases"].append(descriptor)
    study["identity"] = _fingerprint(study)
    path = root / "study.json"
    _publish(path, study)
    _publish(root / "config.json", settings)
    _publish_text(
        root / "README.md",
        "# TenSet context ablation\n\n"
        "Preparation is offline and does not establish completed experiments.\n\n"
        "- study.json/config.json: frozen source, code, variant and cost settings.\n"
        "- receipts/: immutable per-decision results, including partial pilots.\n"
        "- runs/: complete predictions and checksummed run metadata.\n"
        "- jev-cache/: raw requests/responses; separate phase/repeat directories.\n"
        "- summary/: offline metrics, paired comparisons, baselines and plots.\n\n"
        "Both the seven-condition small study and the filtered full ResNet-50 "
        "follow-up require three repeats. Structural diversity filtering is "
        "runtime-blind and differs from TPU largest-tile pruning. Private runtime "
        "labels are retained for scoring only. Do not include an environment file "
        "when sharing. See docs/run-tenset-context-ablation.md "
        "in the code repository.\n",
    )
    return path


def load_study(path: str | Path) -> dict:
    study = _read_json(Path(path))
    identity = study.get("identity")
    if identity != _fingerprint(
        {key: value for key, value in study.items() if key != "identity"}
    ):
        raise ValueError("frozen study identity changed")
    if study.get("protocol") != PROTOCOL or study.get("implementation") != _sources():
        raise ValueError(
            "study code differs; prepare a new output directory with current code"
        )
    if Path(path).resolve() != Path(study["output_dir"]) / "study.json":
        raise ValueError("study path differs from its frozen output directory")
    return study


def _selection(study, phase, variants, repeats, max_workloads):
    phases = ("small", "full") if phase == "all" else (phase,)
    if any(item not in study["phases"] for item in phases):
        raise ValueError("phase must be small, full or all")
    selected_repeats = tuple(repeats) if repeats is not None else (1, 2, 3)
    if (
        not selected_repeats
        or len(set(selected_repeats)) != len(selected_repeats)
        or any(type(n) is not int or n not in (1, 2, 3) for n in selected_repeats)
    ):
        raise ValueError("repeats must be distinct values from 1, 2, 3")
    if max_workloads is not None:
        _integer(max_workloads, "max_workloads")
    result = []
    for name in phases:
        available = study["phases"][name]["variants"]
        selected_variants = (
            tuple(variants) if variants is not None else tuple(available)
        )
        if (
            not selected_variants
            or len(set(selected_variants)) != len(selected_variants)
            or set(selected_variants) - set(available)
        ):
            raise ValueError(f"invalid variants for {name}: {selected_variants}")
        cases = study["phases"][name]["cases"]
        if max_workloads is not None:
            cases = cases[:max_workloads]
        for variant in selected_variants:
            for repeat in selected_repeats:
                for case in cases:
                    result.append((name, variant, repeat, case))
    return result


def _receipt_path(study, phase, variant, repeat, workload):
    # Workload identifiers come from the verified source; use a digest as filename.
    return (
        Path(study["output_dir"])
        / "receipts"
        / phase
        / variant
        / f"rep{repeat}"
        / f"{_fingerprint(workload)}.json"
    )


def _receipt(study, phase, variant, repeat, descriptor):
    path = _receipt_path(study, phase, variant, repeat, descriptor["workload_id"])
    if not path.exists():
        return None
    receipt = _read_json(path)
    row = receipt.get("prediction")
    if (
        receipt.get("study_identity") != study["identity"]
        or not isinstance(row, dict)
        or receipt.get("prediction_sha256") != _fingerprint(row)
    ):
        raise ValueError(f"completion receipt was changed: {path}")
    expected = {
        "phase": phase,
        "variant": variant,
        "repeat": repeat,
        "workload_id": descriptor["workload_id"],
        "candidate_ids": descriptor["candidate_ids"],
    }
    if any(row.get(key) != value for key, value in expected.items()):
        raise ValueError(f"completion receipt identity differs: {path}")
    if (
        row.get("shown_candidate_ids")
        != descriptor["plans"][variant]["shown_candidate_ids"]
        or row.get("model_input_sha256")
        != descriptor["plans"][variant]["model_input_sha256"]
    ):
        raise ValueError(f"completion receipt input differs: {path}")
    ids = descriptor["candidate_ids"]
    runtimes = row.get("runtimes_ms", {})
    ranking = row.get("ranking", [])
    shown = row["shown_candidate_ids"]
    if (
        set(runtimes) != set(ids)
        or any(
            type(value) not in (int, float) or not math.isfinite(value) or value <= 0
            for value in runtimes.values()
        )
        or _fingerprint([[identity, runtimes[identity]] for identity in ids])
        != descriptor["labels_sha256"]
        or len(ranking) != len(ids)
        or set(ranking) != set(ids)
        or set(ranking[: len(shown)]) != set(shown)
        or any(
            row.get(key) != descriptor.get(key)
            for key in ("task_weight", "runtime_basis", "paper_normalizer_ms")
        )
    ):
        raise ValueError(f"completion receipt labels or ranking differ: {path}")
    return row


def preflight_ablation(
    study_path,
    *,
    phase="all",
    variants=None,
    repeats=None,
    max_workloads=None,
    calibrate_from_pilot=False,
):
    """Validate every selected source and receipt, then estimate remaining inputs."""
    study = load_study(study_path)
    selected = _selection(study, phase, variants, repeats, max_workloads)
    # Re-read actual private inputs, not just frozen byte estimates. This is
    # intentionally offline; input changes fail before environment/client loading.
    for name in dict.fromkeys(item[0] for item in selected):
        actual = []
        planned = {row["workload_id"]: row for row in study["phases"][name]["cases"]}
        requested = {
            (variant, descriptor["workload_id"])
            for phase_name, variant, _, descriptor in selected
            if phase_name == name
        }
        for case, extra in _phase_cases(study, name):
            actual.append(_descriptor(case, extra))
            for variant in study["phases"][name]["variants"]:
                if (variant, case.workload.workload_id) in requested:
                    frozen = planned.get(case.workload.workload_id)
                    if frozen is None:
                        raise ValueError("source workload inventory changed")
                    _question_plan(study, name, variant, case, frozen["plans"][variant])
        expected = [
            {key: value for key, value in row.items() if key != "plans"}
            for row in study["phases"][name]["cases"]
        ]
        if actual != expected:
            raise ValueError(f"{name} workload/candidate/label inventory changed")
    total, remaining, completed = 0, 0, 0
    usage_actual, usage_estimated, live_input = 0, 0, 0
    by_phase = {}
    by_variant = {}
    for name, variant, repeat, descriptor in selected:
        estimate = descriptor["plans"][variant]["estimated_input_tokens"]
        total += estimate
        row = _receipt(study, name, variant, repeat, descriptor)
        statistics = by_phase.setdefault(
            name,
            {
                "decisions": 0,
                "completed_decisions": 0,
                "remaining_estimated_input_tokens": 0,
            },
        )
        statistics["decisions"] += 1
        condition = by_variant.setdefault(
            f"{name}/{variant}",
            {
                "decisions": 0,
                "completed_decisions": 0,
                "estimated_input_tokens": 0,
                "remaining_estimated_input_tokens": 0,
            },
        )
        condition["decisions"] += 1
        condition["estimated_input_tokens"] += estimate
        if row is None:
            remaining += estimate
            statistics["remaining_estimated_input_tokens"] += estimate
            condition["remaining_estimated_input_tokens"] += estimate
        else:
            completed += 1
            statistics["completed_decisions"] += 1
            condition["completed_decisions"] += 1
            for report in row["request_reports"]:
                value = report.get("input_tokens")
                if not report.get("cache_hit") and type(value) is int and value >= 0:
                    live_input += value
                    usage_actual += value
                    usage_estimated += report["estimated_input_tokens"]
    price = study["pricing"]["input_per_million_usd"]
    for group in (*by_phase.values(), *by_variant.values()):
        group["estimated_remaining_input_cost_usd"] = (
            group["remaining_estimated_input_tokens"] * price / 1e6
        )
    result = {
        "study_identity": study["identity"],
        "protocol": PROTOCOL,
        "api_requests_made": 0,
        "input_only": True,
        "selected_decisions": len(selected),
        "completed_decisions": completed,
        "remaining_requests_no_cache_assumed": len(selected) - completed,
        "selected_scope_complete": completed == len(selected),
        "estimated_total_input_tokens": total,
        "estimated_remaining_input_tokens": remaining,
        "estimated_total_input_cost_usd": total * price / 1e6,
        "estimated_remaining_input_cost_usd": remaining * price / 1e6,
        "recorded_live_input_cost_usd": live_input * price / 1e6,
        "input_price_per_million_usd": price,
        "by_phase": by_phase,
        "by_variant": by_variant,
        "warnings": [
            "Character estimates are not exact tokenizer counts or invoice guarantees.",
            "Output fees and failed/retried requests are excluded; "
            "confirm account pricing.",
            "No cache savings are assumed for incomplete decisions; "
            "repeats have separate caches.",
            "Structural diversity differs from TPU largest-tile pruning "
            "and is not a performance predictor.",
        ],
    }
    if calibrate_from_pilot:
        # Completed requests outside the current phase/repeat selection are also
        # useful for calibration, but never treated as free future repeats.
        if not usage_estimated:
            for item in _selection(study, "all", None, None, None):
                row = _receipt(study, *item)
                if row:
                    for report in row["request_reports"]:
                        value = report.get("input_tokens")
                        if (
                            not report.get("cache_hit")
                            and type(value) is int
                            and value >= 0
                        ):
                            usage_actual += value
                            usage_estimated += report["estimated_input_tokens"]
        ratio = usage_actual / usage_estimated if usage_estimated else None
        result["pilot_calibration"] = {
            "actual_to_estimated_token_ratio": ratio,
            "projected_remaining_input_cost_usd": None
            if ratio is None
            else remaining * ratio * price / 1e6,
            "note": "Observed pilot average only; workload and context "
            "composition can differ.",
        }
    return result


def _metrics(case, result, *, filtered):
    metrics = {
        "top_1_score": top_1_score(case, result),
        "top_5_score": top_5_score(case, result),
        "runtime_regret": runtime_regret(case, result),
        "endpoint_top_1_score": best_of_top_k_score(case, result, 1),
        "endpoint_top_5_score": best_of_top_k_score(case, result, 5),
    }
    metrics["top_1_error"] = 1 / metrics["endpoint_top_1_score"] - 1
    metrics["top_5_error"] = 1 / metrics["endpoint_top_5_score"] - 1
    if not filtered:
        probabilities = result.metadata.get("last_question_probabilities")
        if probabilities is None:
            raise ValueError("single-question pairwise metrics require probabilities")
        # Top-K retains choice-first order. Pairwise diagnostics preserve
        # the actual rounded probabilities, including their ties.
        probability_ranking = tuple(
            sorted(result.ranked_candidate_ids, key=lambda key: -probabilities[key])
        )
        probability_result = DecisionResult(
            probability_ranking,
            result.decision_latency_ms,
            {"probabilities": probabilities},
        )
        metrics["pairwise_accuracy"] = pairwise_accuracy(case, probability_result)
        metrics["pairwise_accuracy_tie_aware"] = pairwise_accuracy_tie_aware(
            case, probability_result
        )
    return metrics


def _prediction(study, phase, variant, repeat, case, extra, descriptor, result, shown):
    shown_ids = [candidate.candidate_id for candidate in shown]
    # The hidden tail is a completion convention, not a model prediction.
    hidden = [
        candidate.candidate_id
        for candidate in case.candidates
        if candidate.candidate_id not in set(shown_ids)
    ]
    ranking = (*result.ranked_candidate_ids, *hidden)
    if set(result.ranked_candidate_ids) != set(shown_ids):
        raise ValueError("Jev ranking does not match its shown candidate set")
    complete = DecisionResult(ranking, result.decision_latency_ms, result.metadata)
    return {
        "phase": phase,
        "variant": variant,
        "repeat": repeat,
        "workload_id": case.workload.workload_id,
        "candidate_ids": descriptor["candidate_ids"],
        "shown_candidate_ids": shown_ids,
        "ranking": list(ranking),
        "runtimes_ms": {r.candidate_id: r.runtime_ms for r in case.runtimes},
        "model_input_sha256": descriptor["plans"][variant]["model_input_sha256"],
        "decision_latency_ms": result.decision_latency_ms,
        "request_reports": result.metadata["request_reports"],
        "metadata": {**result.metadata, "hidden_tail": "source_order_not_predicted"},
        "metrics": _metrics(case, complete, filtered=bool(hidden)),
        "random_ranking": list(
            RandomModel({"seed": study["seed"]})
            .rank(case.workload, case.candidates)
            .ranked_candidate_ids
        ),
        **extra,
    }


def _publish_runs(study, selection):
    root = Path(study["output_dir"])
    for phase, variant, repeat in dict.fromkeys(item[:3] for item in selection):
        descriptors = study["phases"][phase]["cases"]
        rows = [
            _receipt(study, phase, variant, repeat, descriptor)
            for descriptor in descriptors
        ]
        if any(row is None for row in rows):
            continue  # Pilot receipts are resumable but not called a complete run.
        directory = root / "runs" / phase / variant / f"rep{repeat}"
        content = "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows)
        _publish(
            directory / "run.json",
            {
                "study_identity": study["identity"],
                "phase": phase,
                "variant": variant,
                "repeat": repeat,
                "workloads": len(rows),
                "model": _model_config(study, phase, repeat),
                "predictions_sha256": hashlib.sha256(content.encode()).hexdigest(),
            },
        )
        _publish_text(directory / "predictions.jsonl", content)


def _run_ablation_locked(
    study_path,
    *,
    execute=False,
    phase="all",
    variants=None,
    repeats=None,
    max_workloads=None,
    resume=False,
    budget_usd=None,
    load_environment: Callable | None = None,
    progress: Callable | None = None,
):
    """Execute only after offline validation and an explicit input budget."""
    if execute is not True:
        raise ValueError("live execution requires --execute explicitly")
    budget = _positive(budget_usd, "budget_usd")
    plan = preflight_ablation(
        study_path,
        phase=phase,
        variants=variants,
        repeats=repeats,
        max_workloads=max_workloads,
    )
    if plan["estimated_remaining_input_cost_usd"] > budget:
        raise ValueError(
            "estimated remaining input cost exceeds --budget-usd; no client was created"
        )
    if plan["completed_decisions"] and not resume:
        raise FileExistsError(
            "selected results already exist; use --resume to verify and reuse them"
        )
    study = load_study(study_path)
    selection = _selection(study, phase, variants, repeats, max_workloads)
    if not plan["remaining_requests_no_cache_assumed"]:
        _publish_runs(study, selection)
        return plan
    if load_environment:
        load_environment()
    spent = 0.0
    done = 0
    models = {}
    limiter = RateLimiter(
        study["model"]["max_requests_per_minute"],
        study["model"]["max_tokens_per_second"],
    )
    try:
        return _execute_selection(
            study,
            selection,
            models,
            limiter,
            budget,
            spent,
            done,
            progress,
            study_path,
            phase,
            variants,
            repeats,
            max_workloads,
        )
    finally:
        for model in models.values():
            model.close()


def _execute_selection(
    study,
    selection,
    models,
    limiter,
    budget,
    spent,
    done,
    progress,
    study_path,
    phase,
    variants,
    repeats,
    max_workloads,
):
    live_requests = 0
    cached_requests = 0
    for name in dict.fromkeys(item[0] for item in selection):
        groups = {}
        for item in selection:
            if item[0] == name:
                groups.setdefault(item[3]["workload_id"], []).append(item)
        descriptors = {
            row["workload_id"]: row for row in study["phases"][name]["cases"]
        }
        for case, extra in _phase_cases(study, name):
            expected = descriptors[case.workload.workload_id]
            if _descriptor(case, extra) != {
                key: value for key, value in expected.items() if key != "plans"
            }:
                raise ValueError("source case changed after offline validation")
            for _, variant, repeat, _ in groups.get(case.workload.workload_id, []):
                descriptor = descriptors[case.workload.workload_id]
                reused = _receipt(study, name, variant, repeat, descriptor)
                if reused is None:
                    workload, shown, packing = _question_plan(
                        study, name, variant, case, descriptor["plans"][variant]
                    )
                    estimated_cost = (
                        packing["estimated_first_round_input_tokens"]
                        * study["pricing"]["input_per_million_usd"]
                        / 1e6
                    )
                    if spent + estimated_cost > budget:
                        raise ValueError(
                            "input budget exhausted between decisions; "
                            "completed receipts are retained"
                        )
                    key = name, repeat
                    if key not in models:
                        models[key] = AblationChoiceModel(
                            _model_config(study, name, repeat)
                        )
                        models[key]._limiter = limiter
                    result = models[key].rank(workload, shown)
                    row = _prediction(
                        study,
                        name,
                        variant,
                        repeat,
                        case,
                        extra,
                        descriptor,
                        result,
                        shown,
                    )
                    _publish(
                        _receipt_path(
                            study, name, variant, repeat, case.workload.workload_id
                        ),
                        {
                            "study_identity": study["identity"],
                            "prediction_sha256": _fingerprint(row),
                            "prediction": row,
                        },
                    )
                    live_requests += sum(
                        not report["cache_hit"] for report in row["request_reports"]
                    )
                    cached_requests += sum(
                        report["cache_hit"] for report in row["request_reports"]
                    )
                    spent += (
                        sum(
                            report["input_tokens"]
                            if type(report.get("input_tokens")) is int
                            else report["estimated_input_tokens"]
                            for report in row["request_reports"]
                            if not report["cache_hit"]
                        )
                        * study["pricing"]["input_per_million_usd"]
                        / 1e6
                    )
                done += 1
                if progress:
                    progress(
                        done,
                        len(selection),
                        name,
                        variant,
                        repeat,
                        case.workload.workload_id,
                        reused is not None,
                    )
    _publish_runs(study, selection)
    result = preflight_ablation(
        study_path,
        phase=phase,
        variants=variants,
        repeats=repeats,
        max_workloads=max_workloads,
        calibrate_from_pilot=True,
    )
    result["api_requests_made"] = live_requests
    result["cached_requests_reused_this_invocation"] = cached_requests
    result["status"] = (
        "pilot_selection_complete"
        if max_workloads is not None
        else "selected_runs_complete"
    )
    return result


def run_ablation(study_path, **kwargs):
    """Prevent overlapping executions from sending duplicate paid requests."""
    if kwargs.get("execute") is not True:
        raise ValueError("live execution requires --execute explicitly")
    study = load_study(study_path)
    with (Path(study["output_dir"]) / ".execution.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "another ablation execution holds the study lock"
            ) from error
        try:
            return _run_ablation_locked(study_path, **kwargs)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
