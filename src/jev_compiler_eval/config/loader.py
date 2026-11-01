"""Load and structurally validate YAML experiment configuration."""

from collections.abc import Mapping
from pathlib import Path

import yaml

from jev_compiler_eval.config.types import (
    CandidateSelectionSpec,
    DatasetSpec,
    ExecutionSpec,
    ExperimentConfig,
    ExperimentMetadata,
    ModelSpec,
)

SCHEMA_VERSION = 1


def _mapping(value: object, location: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"{location} keys must be strings")
    return dict(value)


def _check_keys(
    value: Mapping[str, object],
    *,
    location: str,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    optional = optional or set()
    missing = required - value.keys()
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"{location} is missing required field(s): {names}")

    unknown = value.keys() - required - optional
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"{location} contains unknown field(s): {names}")


def _string(value: object, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a non-empty string")
    return value


def _integer(value: object, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{location} must be an integer")
    return value


def _parse_experiment(value: object) -> ExperimentMetadata:
    data = _mapping(value, "experiment")
    _check_keys(
        data,
        location="experiment",
        required={"name", "seed", "output_dir"},
        optional={"description"},
    )

    description = data.get("description")
    if description is not None and not isinstance(description, str):
        raise ValueError("experiment.description must be a string or null")

    return ExperimentMetadata(
        name=_string(data["name"], "experiment.name"),
        seed=_integer(data["seed"], "experiment.seed"),
        output_dir=_string(data["output_dir"], "experiment.output_dir"),
        description=description,
    )


def _parse_dataset(value: object) -> DatasetSpec:
    data = _mapping(value, "dataset")
    _check_keys(data, location="dataset", required={"type", "config"})
    return DatasetSpec(
        dataset_type=_string(data["type"], "dataset.type"),
        config=_mapping(data["config"], "dataset.config"),
    )


def _parse_candidate_selection(value: object) -> CandidateSelectionSpec:
    data = _mapping(value, "candidate_selection")
    if "strategy" not in data:
        raise ValueError("candidate_selection is missing required field(s): strategy")
    strategy = _string(data.pop("strategy"), "candidate_selection.strategy")
    return CandidateSelectionSpec(strategy=strategy, config=data)


def _parse_models(value: object) -> tuple[ModelSpec, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("models must be a non-empty list")

    models: list[ModelSpec] = []
    model_ids: set[str] = set()
    for index, raw_model in enumerate(value):
        location = f"models[{index}]"
        data = _mapping(raw_model, location)
        _check_keys(data, location=location, required={"id", "type", "config"})
        model_id = _string(data["id"], f"{location}.id")
        if model_id in model_ids:
            raise ValueError(f'duplicate model id "{model_id}"')
        model_ids.add(model_id)
        models.append(
            ModelSpec(
                model_id=model_id,
                model_type=_string(data["type"], f"{location}.type"),
                config=_mapping(data["config"], f"{location}.config"),
            )
        )
    return tuple(models)


def _parse_metrics(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("metrics must be a non-empty list")

    metrics = tuple(
        _string(metric, f"metrics[{index}]") for index, metric in enumerate(value)
    )
    if len(metrics) != len(set(metrics)):
        raise ValueError("metrics must not contain duplicates")
    return metrics


def _parse_execution(value: object) -> ExecutionSpec:
    data = _mapping(value, "execution")
    _check_keys(data, location="execution", required={"fail_fast"})
    fail_fast = data["fail_fast"]
    if not isinstance(fail_fast, bool):
        raise ValueError("execution.fail_fast must be a boolean")
    return ExecutionSpec(fail_fast=fail_fast)


def _parse_config(value: object) -> ExperimentConfig:
    data = _mapping(value, "configuration root")
    _check_keys(
        data,
        location="configuration root",
        required={
            "schema_version",
            "experiment",
            "dataset",
            "candidate_selection",
            "models",
            "metrics",
            "execution",
        },
    )

    schema_version = _integer(data["schema_version"], "schema_version")
    if schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema_version {schema_version}; expected {SCHEMA_VERSION}"
        )

    return ExperimentConfig(
        schema_version=schema_version,
        experiment=_parse_experiment(data["experiment"]),
        dataset=_parse_dataset(data["dataset"]),
        candidate_selection=_parse_candidate_selection(data["candidate_selection"]),
        models=_parse_models(data["models"]),
        metrics=_parse_metrics(data["metrics"]),
        execution=_parse_execution(data["execution"]),
    )


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load and structurally validate an experiment YAML file."""

    config_path = Path(path)
    try:
        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        message = f'failed to parse YAML configuration "{config_path}"'
        raise ValueError(message) from error
    return _parse_config(raw_config)
