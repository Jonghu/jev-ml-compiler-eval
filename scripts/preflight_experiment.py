"""Validate a prepared experiment and estimate tournament packing without API calls."""

import argparse
from pathlib import Path

from jev_compiler_eval.config import load_experiment_config
from jev_compiler_eval.datasets import get_dataset_class
from jev_compiler_eval.metrics import resolve_metrics
from jev_compiler_eval.models import get_model_class


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    resolve_metrics(config.metrics)
    model_classes = {}
    for specification in config.models:
        model_class = get_model_class(specification.model_type)
        validator = model_class.__new__(model_class)
        validator.config = dict(specification.config)
        validator.validate_config()
        model_classes[specification.model_id] = model_class
    dataset = get_dataset_class(config.dataset.dataset_type)(config.dataset.config)
    cases = dataset.load()
    counts = [len(case.candidates) for case in cases]
    print(f"Experiment: {config.experiment.name}")
    print(
        f"Workloads: {len(cases)}; candidates per workload: {min(counts)}–{max(counts)}"
    )
    if hasattr(dataset, "manifest_metadata"):
        sizes = [
            case["serialized_input_bytes"]
            for case in dataset.manifest_metadata()["realized_cases"]
        ]
        print(f"Single-question input bytes: {min(sizes):,}–{max(sizes):,}")
    for specification in config.models:
        model_class = model_classes[specification.model_id]
        print(f"Model: {specification.model_id}")
        if specification.model_type == "jev-tournament":
            if {"pairwise_accuracy", "pairwise_accuracy_tie_aware"} & set(
                config.metrics
            ):
                raise ValueError("tournament profiles must use selection metrics")
            plans = [
                model_class.plan(case.workload, case.candidates, specification.config)
                for case in cases
            ]
            print(
                f"Validated first-round tournament packing for {len(plans)} workloads"
            )
    print("Offline preflight passed. No API client was constructed or request sent.")
    print("Byte counts and tournament token estimates do not guarantee service limits.")


if __name__ == "__main__":
    main()
