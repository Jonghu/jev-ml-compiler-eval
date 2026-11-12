"""Official TenSet XGBoost adapter with an isolated legacy TVM worker."""

from __future__ import annotations

import hashlib
import json
import queue
import subprocess
import sys
import tempfile
import threading
from collections.abc import Mapping, Sequence
from math import isfinite
from pathlib import Path
from time import perf_counter_ns

from jev_compiler_eval.datasets.tenset import TRANSFORMATIONS
from jev_compiler_eval.models.base import BaseDecisionModel
from jev_compiler_eval.types import Candidate, DecisionResult, Workload

IMPLEMENTATION = "tlc-pack/tenset:auto_scheduler.cost_model.XGBModel"
WORKER_PATH = Path(__file__).with_name("_tenset_tvm_worker.py")


def file_sha256(path: str | Path) -> str:
    """Hash an artifact without loading it entirely into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as artifact:
        for block in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_measure_input(
    context: Mapping[str, object], candidate: Mapping[str, object]
) -> None:
    """Check the private TVM input describes the visible workload and schedule."""

    raw = candidate.get("record_input")
    if (
        not isinstance(raw, list)
        or len(raw) != 2
        or not isinstance(raw[0], list)
        or len(raw[0]) < 2
        or raw[0][:2] != [context.get("workload_key"), context.get("target")]
        or not isinstance(raw[1], list)
        or len(raw[1]) != 2
        or not isinstance(raw[1][1], list)
    ):
        raise ValueError("hidden TVM input does not match the visible workload")
    state = raw[1]
    if any(
        not isinstance(step, list) or not step or not isinstance(step[0], str)
        for step in state[1]
    ):
        raise ValueError("hidden TVM input has invalid transformations")
    configuration = {
        "initial_state": state[0],
        "transformations": [
            {
                "code": step[0],
                "name": TRANSFORMATIONS.get(step[0], "unknown"),
                "arguments": step[1:],
            }
            for step in state[1]
        ],
    }
    if candidate.get("configuration") != configuration:
        raise ValueError("hidden TVM input does not match the visible candidate")


class TenSetXGBModel(BaseDecisionModel):
    """Rank using the paper's per-store features and pack-sum XGBoost model.

    TVM and XGBoost are imported only by a persistent subprocess. The interpreter
    can therefore point at the original TenSet Python environment while the
    shared evaluation runner remains on Python 3.11. Only model-visible inputs
    and synthetic dummy results cross the inference boundary.
    """

    def __init__(self, config: Mapping[str, object]) -> None:
        super().__init__(config)
        self._process: subprocess.Popen[str] | None = None
        self._responses: queue.Queue[str | None] = queue.Queue()
        self._worker_metadata: dict[str, object] = {}
        self._initialization_ms = 0.0
        self._closed = False

        prepared_path = Path(str(self.config["prepared_path"]))
        self._prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
        if self._prepared.get("dataset") != "tenset":
            raise ValueError("XGBoost prepared_path must contain a TenSet dataset")
        provenance_path = Path(
            str(
                self.config.get(
                    "provenance_path", self.config["checkpoint_path"] + ".json"
                )
            )
        )
        self._provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if self._provenance.get("implementation") != IMPLEMENTATION:
            raise ValueError("checkpoint provenance is not the official TenSet XGBoost")
        for field, path in (
            ("checkpoint_sha256", self.config["checkpoint_path"]),
            ("prepared_sha256", prepared_path),
            ("task_registry_sha256", self.config["task_registry_path"]),
        ):
            if self._provenance.get(field) != file_sha256(str(path)):
                raise ValueError(
                    f"XGBoost provenance {field} does not match its artifact"
                )
        self._training_ids = set(self._provenance.get("training_workload_ids", []))
        if not self._training_ids:
            raise ValueError("XGBoost provenance must identify training workloads")
        workloads = self._prepared.get("workloads", [])
        self._workloads = {entry["workload_id"]: entry for entry in workloads}
        if len(self._workloads) != len(workloads):
            raise ValueError("XGBoost prepared workload IDs must be unique")
        expected_training_ids = {
            entry["workload_id"] for entry in workloads if entry["split"] == "train"
        }
        if self._training_ids != expected_training_ids:
            raise ValueError(
                "checkpoint training workloads do not match prepared split"
            )
        self._training_keys = {
            entry["context"]["workload_key"]
            for entry in workloads
            if entry["split"] == "train"
        }
        if any(
            entry["context"]["workload_key"] in self._training_keys
            for entry in workloads
            if entry["split"] != "train"
        ):
            raise ValueError("XGBoost train and held-out workload keys overlap")
        self._stderr = tempfile.TemporaryFile(mode="w+t", encoding="utf-8")

    @property
    def model_type(self) -> str:
        return "tenset_xgb"

    def validate_config(self) -> None:
        required = {"prepared_path", "checkpoint_path", "task_registry_path"}
        optional = {"provenance_path", "python_executable", "worker_timeout_seconds"}
        if (
            not required <= self.config.keys()
            or self.config.keys() - required - optional
        ):
            raise ValueError(
                "tenset_xgb config requires prepared_path, checkpoint_path, "
                "task_registry_path and optional provenance_path, python_executable, "
                "worker_timeout_seconds"
            )
        for name in required | (
            optional & self.config.keys() - {"worker_timeout_seconds"}
        ):
            value = self.config[name]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"tenset_xgb {name} must be a non-empty string")
        timeout = self.config.get("worker_timeout_seconds", 300.0)
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("XGBoost worker timeout must be a finite positive number")

    def _read_response(self) -> dict[str, object]:
        try:
            line = self._responses.get(
                timeout=float(self.config.get("worker_timeout_seconds", 300.0))
            )
        except queue.Empty as error:
            self.close()
            raise TimeoutError("TenSet TVM worker timed out") from error
        if line is None:
            self._stderr.seek(0)
            diagnostic = self._stderr.read()[-4000:]
            raise RuntimeError(f"TenSet TVM worker exited unexpectedly: {diagnostic}")
        try:
            response = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError("TenSet TVM worker returned invalid JSON") from error
        if "error" in response:
            raise RuntimeError(f"TenSet TVM worker: {response['error']}")
        return response

    def _start_worker(self) -> None:
        if self._process is not None:
            return
        started_at = perf_counter_ns()
        self._process = subprocess.Popen(
            [
                str(self.config.get("python_executable", sys.executable)),
                "-u",
                str(WORKER_PATH),
                "serve",
                "--checkpoint",
                str(self.config["checkpoint_path"]),
                "--task-registry",
                str(self.config["task_registry_path"]),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        stdout = self._process.stdout
        assert stdout is not None

        def collect_responses() -> None:
            for line in stdout:
                self._responses.put(line)
            self._responses.put(None)

        threading.Thread(target=collect_responses, daemon=True).start()
        self._worker_metadata = self._read_response()
        if self._worker_metadata.get("status") != "ready":
            raise RuntimeError("TenSet TVM worker did not initialize")
        self._initialization_ms = (perf_counter_ns() - started_at) / 1_000_000

    def rank(
        self, workload: Workload, candidates: Sequence[Candidate]
    ) -> DecisionResult:
        if self._closed:
            raise RuntimeError("TenSet XGBoost model is closed")
        candidate_ids = [candidate.candidate_id for candidate in candidates]
        if not candidate_ids or len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("XGBoost requires nonempty, unique candidate IDs")
        if workload.workload_id in self._training_ids:
            raise ValueError("XGBoost cannot evaluate a training workload")
        entry = self._workloads.get(workload.workload_id)
        if entry is None or entry["context"] != dict(workload.context):
            raise ValueError(
                "evaluation workload does not match prepared XGBoost inputs"
            )
        if entry["split"] not in {"validation", "test"}:
            raise ValueError("XGBoost evaluation requires a held-out workload")
        saved = {
            candidate["candidate_id"]: candidate for candidate in entry["candidates"]
        }
        records = []
        for candidate in candidates:
            original = saved.get(candidate.candidate_id)
            if original is None or original["configuration"] != dict(
                candidate.configuration
            ):
                raise ValueError(
                    "evaluation candidate does not match prepared XGBoost input"
                )
            _validate_measure_input(workload.context, original)
            records.append(
                {"i": original["record_input"], "v": original["record_version"]}
            )
        # Initialization is included in the first decision and recorded separately.
        started_at = perf_counter_ns()
        first_request = self._process is None
        self._start_worker()
        assert self._process is not None and self._process.stdin is not None
        self._process.stdin.write(
            json.dumps({"op": "predict", "records": records}) + "\n"
        )
        self._process.stdin.flush()
        response = self._read_response()
        scores = response.get("scores")
        if not isinstance(scores, list) or len(scores) != len(candidates):
            raise ValueError("XGBoost worker scores must match every candidate")
        for score in scores:
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not isfinite(score)
            ):
                raise ValueError("XGBoost returned a nonfinite candidate score")
        ranking = tuple(
            candidate_ids[index]
            for index in sorted(
                range(len(candidates)),
                key=lambda index: (-scores[index], candidate_ids[index]),
            )
        )
        return DecisionResult(
            ranked_candidate_ids=ranking,
            decision_latency_ms=(perf_counter_ns() - started_at) / 1_000_000,
            metadata={
                "implementation": IMPLEMENTATION,
                "checkpoint_sha256": self._provenance["checkpoint_sha256"],
                "scores": dict(zip(candidate_ids, scores, strict=True)),
                "score_direction": "higher_is_better",
                "worker_initialization_ms": self._initialization_ms
                if first_request
                else 0.0,
                "feature_prediction_ms": response.get("feature_prediction_ms"),
                "versions": self._worker_metadata.get("versions"),
            },
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._process is not None:
            try:
                if self._process.stdin is not None:
                    self._process.stdin.close()
                self._process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                self._process.kill()
                self._process.wait(timeout=5)
            finally:
                if self._process.stdout is not None:
                    self._process.stdout.close()
        self._stderr.close()
