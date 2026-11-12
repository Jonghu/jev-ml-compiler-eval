"""Train the official TenSet XGBoost model on the prepared training split."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from jev_compiler_eval.models.tenset_xgb import WORKER_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, help="Prepared TenSet JSON")
    parser.add_argument(
        "--checkpoint", required=True, help="New checkpoint destination"
    )
    parser.add_argument(
        "--task-registry", required=True, help="Trusted official all_tasks.pkl"
    )
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Interpreter with the original TenSet TVM fork and XGBoost installed",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for path in (args.prepared, args.task_registry):
        if not Path(path).is_file():
            parser.error(f"input file does not exist: {path}")
    if Path(args.checkpoint).exists() or Path(args.checkpoint + ".json").exists():
        parser.error("checkpoint and provenance destinations must not exist")
    completed = subprocess.run(
        [
            args.python_executable,
            str(WORKER_PATH),
            "train",
            "--prepared",
            args.prepared,
            "--checkpoint",
            args.checkpoint,
            "--task-registry",
            args.task_registry,
            "--seed",
            str(args.seed),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
    )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise SystemExit("TenSet training worker returned invalid JSON") from error
    if completed.returncode or "error" in result:
        raise SystemExit(result.get("error", "TenSet training worker failed"))
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Provenance: {args.checkpoint}.json")
    print(f"Training workloads: {len(result['training_workload_ids'])}")
    print(f"Training candidates: {result['training_candidates']}")


if __name__ == "__main__":
    main()
