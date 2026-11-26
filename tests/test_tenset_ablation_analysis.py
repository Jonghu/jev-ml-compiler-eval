"""Independent arithmetic and provenance checks for offline ablation summaries."""

import copy
import csv
import hashlib
import itertools
import json
import math
import zipfile
from statistics import mean

import pytest

from jev_compiler_eval.evaluation.tenset_ablation_analysis import (
    expected_random_metrics,
    package_ablation,
    summarize_ablation,
)
from jev_compiler_eval.models.random import RandomModel
from jev_compiler_eval.types import Candidate, Workload


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def make_study(tmp_path, *, phase="small", variants=("a0", "a1", "a5")):
    root = tmp_path / "study"
    root.mkdir()
    ids = list("abcdef")
    study = {
        "protocol": "synthetic_ablation",
        "repeats": 3,
        "phases": {
            phase: {
                "variants": list(variants),
                "cases": [{"workload_id": "w", "candidate_ids": ids, "task_weight": 1}],
            }
        },
    }
    (root / "study.json").write_text(json.dumps(study))
    rows = []
    for variant in variants:
        for repeat in (1, 2, 3):
            ranking = list("fedcba") if variant == "a0" else ids
            shown = ids
            if variant == "a5":
                shown, ranking = list("bcdef"), list("bcdefa")
            rows.append(
                {
                    "phase": phase,
                    "variant": variant,
                    "repeat": repeat,
                    "workload_id": "w",
                    "candidate_ids": ids,
                    "shown_candidate_ids": shown,
                    "ranking": ranking,
                    "runtimes_ms": dict(zip(ids, (1, 2, 4, 8, 16, 32))),
                    "task_weight": 1,
                    "metrics": {"pairwise_accuracy": 0.75},
                    "decision_latency_ms": repeat * 10,
                    "request_reports": [
                        {
                            "cache_hit": repeat == 3,
                            "input_tokens": 100,
                            "output_tokens": 20,
                        }
                    ],
                }
            )
    save_rows(root, rows)
    return root, study, rows


def save_rows(root, rows):
    for path in root.rglob("predictions.jsonl"):
        path.unlink()
    for row in rows:
        run = root / "runs" / row["phase"] / row["variant"] / f"rep{row['repeat']}"
        run.mkdir(parents=True, exist_ok=True)
        with (run / "predictions.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def seal_study(root, study, rows):
    """Write the same frozen-input / immutable-receipt contract as the runner."""
    for phase, settings in study["phases"].items():
        for case in settings["cases"]:
            matching = [
                row
                for row in rows
                if row["phase"] == phase and row["workload_id"] == case["workload_id"]
            ]
            if not matching:
                continue
            first = matching[0]
            case["labels_sha256"] = fingerprint(
                [
                    [candidate, first["runtimes_ms"][candidate]]
                    for candidate in case["candidate_ids"]
                ]
            )
            case["runtime_basis"] = first.get("runtime_basis", "measured_milliseconds")
            case["plans"] = {}
            for row in matching:
                row["runtime_basis"] = case["runtime_basis"]
                row["model_input_sha256"] = fingerprint(
                    [phase, row["variant"], row["workload_id"]]
                )
                case["plans"][row["variant"]] = {
                    "shown_candidate_ids": row["shown_candidate_ids"],
                    "model_input_sha256": row["model_input_sha256"],
                }
    study["identity"] = fingerprint(
        {key: value for key, value in study.items() if key != "identity"}
    )
    (root / "study.json").write_text(json.dumps(study))
    (root / "config.json").write_text('{"schema_version":1}')
    save_rows(root, rows)
    for row in rows:
        receipt = (
            root
            / "receipts"
            / row["phase"]
            / row["variant"]
            / f"rep{row['repeat']}"
            / f"{fingerprint(row['workload_id'])}.json"
        )
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(
            json.dumps(
                {
                    "study_identity": study["identity"],
                    "prediction_sha256": fingerprint(row),
                    "prediction": row,
                }
            )
        )
    for path in root.rglob("predictions.jsonl"):
        members = [json.loads(line) for line in path.read_text().splitlines()]
        first = members[0]
        (path.parent / "run.json").write_text(
            json.dumps(
                {
                    "study_identity": study["identity"],
                    "phase": first["phase"],
                    "variant": first["variant"],
                    "repeat": first["repeat"],
                    "workloads": len(members),
                    "predictions_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        )


def test_exact_random_baselines_agree_with_enumerated_without_replacement_draws():
    runtimes = dict(zip("abcdef", (1, 2, 4, 8, 16, 32)))
    result = expected_random_metrics(runtimes, list("bcdef"))
    draws = list(itertools.permutations("bcdef"))
    prefix_scores, endpoint_scores = [], []
    for draw in draws:
        scores = [
            1 / min(runtimes[candidate] for candidate in draw[:k]) for k in range(1, 6)
        ]
        prefix_scores.append(mean(scores))
        endpoint_scores.append(scores[-1])
    assert result["top_1_score"] == pytest.approx(
        mean(1 / runtimes[draw[0]] for draw in draws)
    )
    assert result["top_5_score"] == pytest.approx(mean(prefix_scores))
    assert result["endpoint_top_5_score"] == pytest.approx(mean(endpoint_scores))
    assert result["runtime_regret"] == pytest.approx(
        mean(runtimes[draw[0]] - 1 for draw in draws)
    )
    assert result["retained_optimum"] == 0
    assert result["oracle_top_1_score"] == 0.5
    assert result["oracle_top_1_error"] == 1


def test_summary_separates_endpoint_prefix_and_cached_usage(tmp_path):
    root, _, _ = make_study(tmp_path)
    result = summarize_ablation(root, create_plots=False)
    assert result["decisions"] == 9
    assert result["usage"]["live_input_tokens"] == 600
    assert result["usage"]["cached_input_tokens"] == 300
    assert result["usage"]["live_requests"] == 6
    assert result["usage"]["cached_requests"] == 3
    runs = read_csv(root / "summary" / "runs.csv")
    first = next(row for row in runs if row["variant"] == "a0")
    assert float(first["top_5_score"]) == pytest.approx(
        mean((1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2))
    )
    assert float(first["endpoint_top_5_score"]) == 0.5
    filtered = next(row for row in runs if row["variant"] == "a5")
    assert filtered["pairwise_accuracy"] == ""
    baselines = read_csv(root / "summary" / "baselines.csv")
    retained = next(
        row
        for row in baselines
        if row["variant"] == "a5" and row["baseline"] == "filter_plus_uniform_random"
    )
    assert float(retained["oracle_top_1_score"]) == 0.5
    assert "original denominator" in (root / "summary" / "summary.md").read_text()


def test_repeat_spread_and_paired_comparisons_average_within_workload(tmp_path):
    root, _, rows = make_study(tmp_path)
    rows[4]["ranking"] = list("bacdef")
    save_rows(root, rows)
    summarize_ablation(root, create_plots=False)
    variants = {
        row["variant"]: row for row in read_csv(root / "summary" / "variants.csv")
    }
    assert float(variants["a1"]["same_pick_share"]) == 0
    assert float(variants["a1"]["top_1_score_mean"]) == pytest.approx(5 / 6)
    assert float(variants["a1"]["top_1_score_std"]) == pytest.approx(math.sqrt(1 / 12))
    paired = read_csv(root / "summary" / "paired.csv")
    difference = next(
        row
        for row in paired
        if row["left"] == "a0"
        and row["right"] == "a1"
        and row["metric"] == "top_1_score"
    )
    assert float(difference["mean_right_minus_left"]) == pytest.approx(5 / 6 - 1 / 32)
    assert difference["right_better"] == "1"


def test_full_network_uses_frequency_weighted_ratio_of_latencies(tmp_path):
    root, study, rows = make_study(tmp_path, phase="full", variants=("a5",))
    first = study["phases"]["full"]["cases"][0]
    first.update(task_weight=9, paper_normalizer_ms=1)
    second = copy.deepcopy(first)
    second.update(workload_id="v", task_weight=1, paper_normalizer_ms=9)
    study["phases"]["full"]["cases"].append(second)
    (root / "study.json").write_text(json.dumps(study))
    new_rows = []
    for row in rows:
        row.update(
            shown_candidate_ids=list("abcdef"),
            ranking=list("cbadef"),
            task_weight=9,
            paper_normalizer_ms=1,
        )
        other = copy.deepcopy(row)
        other.update(
            workload_id="v",
            ranking=list("abcdef"),
            task_weight=1,
            paper_normalizer_ms=9,
            runtimes_ms={key: value * 9 for key, value in row["runtimes_ms"].items()},
        )
        new_rows.extend((row, other))
    save_rows(root, new_rows)
    summarize_ablation(root, create_plots=False)
    runs = read_csv(root / "summary" / "runs.csv")
    assert float(runs[0]["top_1_score"]) == 0.625
    assert float(runs[0]["paper_network_top_1_score"]) == 0.4
    assert float(runs[0]["paper_network_top_5_score"]) == 1
    assert runs[0]["runtime_basis"] == "paper_f32_recovered_throughput_floor"


@pytest.mark.parametrize(
    "mutation",
    (
        "labels",
        "candidate_pool",
        "normalizer",
        "hidden_prefix",
        "duplicate",
        "missing_repeat",
    ),
)
def test_invalid_or_unpaired_rows_are_rejected_before_creating_summary(
    tmp_path, mutation
):
    root, _, rows = make_study(tmp_path)
    if mutation == "labels":
        rows[1]["runtimes_ms"]["a"] = 99
    elif mutation == "candidate_pool":
        rows[0]["candidate_ids"] = list("bacdef")
    elif mutation == "normalizer":
        rows[1]["paper_normalizer_ms"] = 2
    elif mutation == "hidden_prefix":
        rows[-1]["ranking"] = list("abcdef")
    elif mutation == "duplicate":
        rows.append(copy.deepcopy(rows[-1]))
    else:
        rows.pop()
    save_rows(root, rows)
    with pytest.raises(ValueError):
        summarize_ablation(root, create_plots=False)
    assert not (root / "summary").exists()


def test_incomplete_summary_requires_explicit_flag_and_marks_provisional(tmp_path):
    root, _, rows = make_study(tmp_path)
    save_rows(root, rows[:2])
    result = summarize_ablation(root, create_plots=False, allow_incomplete=True)
    assert result["warnings"]
    assert "PROVISIONAL" in (root / "summary" / "summary.md").read_text()


def test_archive_includes_raw_results_and_cache_but_excludes_environment_files(
    tmp_path,
):
    root, _, _ = make_study(tmp_path)
    cache = root / "jev-cache" / "small" / "rep1"
    cache.mkdir(parents=True)
    (cache / ("a" * 64 + ".json")).write_text('{"request":{"state":{}},"response":{}}')
    (root / ".env").write_text("TYPESAFE_API_KEY=not-for-sharing")
    (root / "runs" / ".env").write_text("TYPESAFE_API_KEY=not-for-sharing")
    archive = package_ablation(root, tmp_path / "share.zip")
    with zipfile.ZipFile(archive) as saved:
        assert "study.json" in saved.namelist()
        assert any(name.startswith("runs/") for name in saved.namelist())
        assert any(name.startswith("jev-cache/") for name in saved.namelist())
        assert all(".env" not in name for name in saved.namelist())
    with pytest.raises(FileExistsError):
        package_ablation(root, archive)


def test_archive_refuses_credential_fields_and_external_symlinks(tmp_path):
    root, _, _ = make_study(tmp_path)
    destination = tmp_path / "share.zip"
    secret = root / "runs" / "sensitive.json"
    secret.write_text('{"authorization":"secret"}')
    with pytest.raises(ValueError, match="credential"):
        package_ablation(root, destination)
    assert not destination.exists()
    secret.unlink()
    external = tmp_path / "external.json"
    external.write_text("{}")
    secret.symlink_to(external)
    with pytest.raises(ValueError, match="inside"):
        package_ablation(root, destination)
    assert not destination.exists()


def test_seeded_random_reference_matches_registered_model(tmp_path):
    root, _, rows = make_study(tmp_path)
    result = summarize_ablation(root, create_plots=False)
    reference = read_csv(root / "summary" / "random-per-workload.csv")[0]
    prediction = RandomModel({"seed": 42}).rank(
        Workload("w", {}),
        [Candidate(candidate, {}) for candidate in rows[0]["candidate_ids"]],
    )
    assert json.loads(reference["ranking"]) == list(prediction.ranked_candidate_ids)
    assert result["input_cost_estimate"]["known_live_input_cost_usd"] == pytest.approx(
        600 * 0.042 / 1e6
    )


def test_frozen_study_hash_cannot_be_rewritten_without_new_identity(tmp_path):
    root, study, _ = make_study(tmp_path)
    study["identity"] = hashlib.sha256(
        json.dumps(study, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    study["seed"] = 99
    (root / "study.json").write_text(json.dumps(study))
    with pytest.raises(ValueError, match="identity"):
        summarize_ablation(root, create_plots=False)
    assert not (root / "summary").exists()


def test_plots_include_random_reference_and_render_without_api_calls(tmp_path):
    pytest.importorskip("matplotlib")
    root, _, _ = make_study(tmp_path)
    summarize_ablation(root, create_plots=True)
    assert (root / "summary" / "small-comparison.png").stat().st_size > 1000
    svg = (root / "summary" / "small-comparison.svg").read_text()
    assert "Random (one seeded draw)" in svg


def test_scientific_bundle_checks_runs_and_receipts_after_relocation(tmp_path):
    root, study, rows = make_study(tmp_path)
    study["output_dir"] = "/an/original/mac/path/that/is/not/available"
    seal_study(root, study, rows)
    result = summarize_ablation(root, create_plots=False)
    assert result["decisions"] == 9
    archive = package_ablation(root, tmp_path / "audit.zip")
    with zipfile.ZipFile(archive) as saved:
        assert "config.json" in saved.namelist()
        assert any(name.startswith("receipts/") for name in saved.namelist())


@pytest.mark.parametrize(
    "mutation", ("run_hash", "labels", "receipt", "missing_receipt", "model_input")
)
def test_scientific_result_tampering_is_rejected(tmp_path, mutation):
    root, study, rows = make_study(tmp_path)
    seal_study(root, study, rows)
    row = rows[0]
    prediction_path = root / "runs" / "small" / "a0" / "rep1" / "predictions.jsonl"
    receipt_path = (
        root / "receipts" / "small" / "a0" / "rep1" / f"{fingerprint('w')}.json"
    )
    if mutation == "run_hash":
        prediction_path.write_text(prediction_path.read_text() + "\n")
    elif mutation == "missing_receipt":
        receipt_path.unlink()
    elif mutation == "receipt":
        receipt = json.loads(receipt_path.read_text())
        receipt["prediction"]["decision_latency_ms"] = 999
        receipt_path.write_text(json.dumps(receipt))
    else:
        if mutation == "labels":
            row["runtimes_ms"]["a"] = 999
        else:
            row["model_input_sha256"] = "different-model-input"
        prediction_path.write_text(json.dumps(row) + "\n")
        receipt_path.write_text(
            json.dumps(
                {
                    "study_identity": study["identity"],
                    "prediction_sha256": fingerprint(row),
                    "prediction": row,
                }
            )
        )
        run = json.loads((prediction_path.parent / "run.json").read_text())
        run["predictions_sha256"] = hashlib.sha256(
            prediction_path.read_bytes()
        ).hexdigest()
        (prediction_path.parent / "run.json").write_text(json.dumps(run))
    with pytest.raises(ValueError):
        summarize_ablation(root, create_plots=False)
    assert not (root / "summary").exists()


def test_receipt_only_pilot_is_provisional_and_not_a_complete_network_score(tmp_path):
    root, study, rows = make_study(tmp_path, phase="full", variants=("a5",))
    for row in rows:
        row["paper_normalizer_ms"] = 1
        row["runtime_basis"] = "paper_f32_recovered_throughput_floor"
    study["phases"]["full"]["cases"][0]["paper_normalizer_ms"] = 1
    study["phases"]["full"]["cases"].append(
        {
            "workload_id": "unrun",
            "candidate_ids": list("abcdef"),
            "task_weight": 1,
            "paper_normalizer_ms": 1,
        }
    )
    seal_study(root, study, rows)
    for path in root.rglob("predictions.jsonl"):
        path.unlink()
    with pytest.raises(ValueError, match="no completed"):
        summarize_ablation(root, create_plots=False)
    result = summarize_ablation(root, create_plots=False, allow_incomplete=True)
    assert result["warnings"]
    runs = read_csv(root / "summary" / "runs.csv")
    assert "paper_network_top_1_score" not in runs[0]
    assert float(runs[0]["pilot_network_top_1_score"]) == 0.5
    assert runs[0]["complete_network_coverage"] == "False"
