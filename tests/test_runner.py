import csv
import json
from pathlib import Path

import pytest

from jev_compiler_eval.evaluation import run_experiment

PROJECT_ROOT = Path(__file__).parent.parent
MOCK_DATASET_PATH = PROJECT_ROOT / "examples" / "mock_dataset.json"


def experiment_yaml(
    output_dir: Path,
    *,
    dataset_type: str = "mock",
    model_type: str = "random",
    metric: str = "runtime_regret",
    strategy: str = "all",
    fail_fast: bool = True,
) -> str:
    return f"""
schema_version: 1
experiment:
  name: test-experiment
  seed: 42
  output_dir: {output_dir}
dataset:
  type: {dataset_type}
  config:
    path: {MOCK_DATASET_PATH}
candidate_selection:
  strategy: {strategy}
models:
  - id: random-42
    type: {model_type}
    config:
      seed: 42
  - id: random-7
    type: {model_type}
    config:
      seed: 7
metrics:
  - {metric}
  - top_1_accuracy
execution:
  fail_fast: {str(fail_fast).lower()}
"""


def write_experiment(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "experiment.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def test_runs_multiple_models_and_writes_result_bundle(tmp_path: Path) -> None:
    output_dir = tmp_path / "results"
    config_path = write_experiment(tmp_path, experiment_yaml(output_dir))

    result = run_experiment(config_path)

    assert result.output_dir == output_dir
    assert tuple(summary.model_id for summary in result.summaries) == (
        "random-42",
        "random-7",
    )
    assert {path.name for path in output_dir.iterdir()} == {
        "experiment.yaml",
        "manifest.json",
        "predictions.jsonl",
        "metrics.json",
        "summary.csv",
    }

    predictions = [
        json.loads(line)
        for line in (output_dir / "predictions.jsonl").read_text().splitlines()
    ]
    assert len(predictions) == 4
    first_model_candidates = [
        prediction["candidate_ids"]
        for prediction in predictions
        if prediction["model_id"] == "random-42"
    ]
    second_model_candidates = [
        prediction["candidate_ids"]
        for prediction in predictions
        if prediction["model_id"] == "random-7"
    ]
    assert first_model_candidates == second_model_candidates

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["experiment_name"] == "test-experiment"
    assert manifest["model_ids"] == ["random-42", "random-7"]
    assert manifest["git_commit"]

    with (output_dir / "summary.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 2
    assert rows[0]["workloads"] == "2"


def test_refuses_to_overwrite_existing_results(tmp_path: Path) -> None:
    output_dir = tmp_path / "results"
    output_dir.mkdir()
    marker = output_dir / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    config_path = write_experiment(tmp_path, experiment_yaml(output_dir))

    with pytest.raises(FileExistsError, match="output directory .* is not empty"):
        run_experiment(config_path)

    assert marker.read_text(encoding="utf-8") == "keep"


def test_rejects_fail_fast_false(tmp_path: Path) -> None:
    output_dir = tmp_path / "results"
    config_path = write_experiment(
        tmp_path,
        experiment_yaml(output_dir, fail_fast=False),
    )

    with pytest.raises(ValueError, match="fail_fast=false is not supported"):
        run_experiment(config_path)


@pytest.mark.parametrize(
    ("override", "value", "message"),
    [
        ("dataset_type", "missing", 'unknown dataset type "missing"'),
        ("model_type", "missing", 'unknown model type "missing"'),
        ("metric", "missing", 'unknown metric "missing"'),
        (
            "strategy",
            "random_sample",
            "only candidate_selection.strategy=all is currently supported",
        ),
    ],
)
def test_rejects_unsupported_implementations_and_settings(
    tmp_path: Path,
    override: str,
    value: str,
    message: str,
) -> None:
    output_dir = tmp_path / "results"
    config_path = write_experiment(
        tmp_path,
        experiment_yaml(output_dir, **{override: value}),  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match=message):
        run_experiment(config_path)
