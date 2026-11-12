"""Subprocess checks for preparing synthetic TVM records; no real timings."""

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_tenset.py"


def raw_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "raw"
    directory = root / "measure_records/e5-2673"
    directory.mkdir(parents=True)
    for index in range(3):
        lines = []
        for factor in (8, 16):
            record = {
                "i": [
                    [json.dumps([f"synthetic_matmul_{index}", [32, 64]]), "llvm"],
                    [[], [["SP", 0, 0, 32, [factor], 1]]],
                ],
                "r": [[0.001 * factor], 0, 0, 0],
                "v": "v0.6",
            }
            lines.append(json.dumps(record))
        (directory / f"workload{index}.json").write_text("\n".join(lines) + "\n")
    return root


def run_prepare(root: Path, output: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--raw-root",
            str(root),
            "--target",
            "llvm",
            "--output",
            str(output),
            "--test-fraction",
            "0",
            "--validation-fraction",
            "0",
            *extra,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_prepare_cli_budget_split_provenance_and_no_overwrite(tmp_path: Path):
    root = raw_fixture(tmp_path)
    output = tmp_path / "prepared.json"
    result = run_prepare(root, output, "--file-limit", "2")
    assert result.returncode == 0, result.stderr
    prepared = json.loads(output.read_text())
    assert len(prepared["workloads"]) == 2
    assert {workload["split"] for workload in prepared["workloads"]} == {"train"}
    provenance = prepared["provenance"]
    assert provenance["source_files_available"] == 3
    assert provenance["source_files_selected"] == 2
    assert provenance["platform"] == "e5-2673"
    assert len(provenance["sources"]) == 2
    before = output.read_bytes()
    repeated = run_prepare(root, output)
    assert repeated.returncode != 0
    assert "already exists" in repeated.stderr
    assert output.read_bytes() == before
    second = tmp_path / "prepared-second.json"
    assert run_prepare(root, second, "--file-limit", "2").returncode == 0
    assert json.loads(second.read_text())["workloads"] == prepared["workloads"]


def test_prepare_requires_real_record_directory(tmp_path: Path):
    output = tmp_path / "prepared.json"
    result = run_prepare(tmp_path, output)
    assert result.returncode != 0
    assert "expected one measure_records" in result.stderr
    assert not output.exists()
