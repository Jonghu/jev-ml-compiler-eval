"""Export label-free tensor programs from the trusted TenSet task registry."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from jev_compiler_eval.models.tenset_xgb import WORKER_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-registry", required=True, help="Trusted all_tasks.pkl")
    parser.add_argument("--output", required=True, help="New metadata JSON destination")
    parser.add_argument("--python-executable", default=sys.executable)
    args = parser.parse_args()
    if not Path(args.task_registry).is_file():
        parser.error("task registry input file does not exist")
    if Path(args.output).exists():
        parser.error("metadata destination must not exist")
    completed = subprocess.run(
        [
            args.python_executable,
            str(WORKER_PATH),
            "metadata",
            "--task-registry",
            args.task_registry,
            "--output",
            args.output,
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
    )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise SystemExit("TenSet metadata worker returned invalid JSON") from error
    if completed.returncode or "error" in result:
        raise SystemExit(result.get("error", "TenSet metadata export failed"))
    print(f"Metadata: {result['output']}")
    print(f"Workloads: {result['workloads']}")


if __name__ == "__main__":
    main()
