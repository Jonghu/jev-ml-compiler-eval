from pathlib import Path

import pytest

from jev_compiler_eval.config import load_experiment_config

VALID_CONFIG = """
schema_version: 1

experiment:
  name: mock-comparison
  description: Compare multiple model variants
  seed: 42
  output_dir: results/mock/comparison

dataset:
  type: future-dataset
  config:
    path: examples/data.json

candidate_selection:
  strategy: random_sample
  count: 32
  seed: 42

models:
  - id: random
    type: future-random
    config:
      seed: 42

  - id: jev-compact
    type: future-jev
    config:
      cache_dir: .cache/jev
      serializer:
        mode: compact
        max_context_tokens: 4096

metrics:
  - runtime_regret
  - top_1_accuracy

execution:
  fail_fast: true
"""


def write_config(tmp_path: Path, content: str = VALID_CONFIG) -> Path:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(content, encoding="utf-8")
    return config_path


def test_loads_structurally_valid_configuration(tmp_path: Path) -> None:
    config = load_experiment_config(write_config(tmp_path))

    assert config.schema_version == 1
    assert config.experiment.name == "mock-comparison"
    assert config.dataset.dataset_type == "future-dataset"
    assert config.candidate_selection.strategy == "random_sample"
    assert config.candidate_selection.config == {"count": 32, "seed": 42}
    assert tuple(model.model_id for model in config.models) == (
        "random",
        "jev-compact",
    )
    assert config.models[1].config["serializer"] == {
        "mode": "compact",
        "max_context_tokens": 4096,
    }
    assert config.metrics == ("runtime_regret", "top_1_accuracy")
    assert config.execution.fail_fast is True


def test_does_not_require_implementations_to_be_registered(tmp_path: Path) -> None:
    config = load_experiment_config(write_config(tmp_path))

    assert config.dataset.dataset_type == "future-dataset"
    assert config.models[0].model_type == "future-random"


def test_rejects_duplicate_model_ids(tmp_path: Path) -> None:
    content = VALID_CONFIG.replace("id: jev-compact", "id: random")

    with pytest.raises(ValueError, match='duplicate model id "random"'):
        load_experiment_config(write_config(tmp_path, content))


def test_rejects_unsupported_schema_version(tmp_path: Path) -> None:
    content = VALID_CONFIG.replace("schema_version: 1", "schema_version: 2")

    with pytest.raises(ValueError, match="unsupported schema_version 2; expected 1"):
        load_experiment_config(write_config(tmp_path, content))


def test_rejects_missing_required_field(tmp_path: Path) -> None:
    content = VALID_CONFIG.replace("  output_dir: results/mock/comparison\n", "")

    with pytest.raises(
        ValueError, match=r"experiment is missing required field\(s\): output_dir"
    ):
        load_experiment_config(write_config(tmp_path, content))


def test_rejects_unknown_structural_field(tmp_path: Path) -> None:
    content = VALID_CONFIG.replace(
        "schema_version: 1", "schema_version: 1\nunknown: true"
    )

    with pytest.raises(
        ValueError, match=r"configuration root contains unknown field\(s\): unknown"
    ):
        load_experiment_config(write_config(tmp_path, content))


def test_rejects_invalid_yaml(tmp_path: Path) -> None:
    content = "schema_version: [1"

    with pytest.raises(ValueError, match="failed to parse YAML configuration"):
        load_experiment_config(write_config(tmp_path, content))
