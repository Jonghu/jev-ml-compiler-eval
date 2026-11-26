"""Freeze the complete TenSet ablation study without accessing Jev."""

import argparse
import json
from pathlib import Path

from jev_compiler_eval.evaluation.tenset_ablation import (
    preflight_ablation,
    prepare_ablation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/tenset/ablation/study.yaml"),
    )
    args = parser.parse_args()
    path = prepare_ablation(args.config)
    print(f"Frozen study: {path}")
    print(json.dumps(preflight_ablation(path), indent=2))
    print("Offline preparation only; no API client or environment file was loaded.")


if __name__ == "__main__":
    main()
