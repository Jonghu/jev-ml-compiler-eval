"""Reproducible random decision baseline."""

import hashlib
import random
from collections.abc import Sequence
from time import perf_counter_ns
from typing import cast

from jev_compiler_eval.models.base import BaseDecisionModel
from jev_compiler_eval.types import Candidate, DecisionResult, Workload


class RandomModel(BaseDecisionModel):
    """Rank candidates randomly using a deterministic workload-specific seed."""

    @property
    def model_type(self) -> str:
        return "random"

    def validate_config(self) -> None:
        if set(self.config) != {"seed"}:
            raise ValueError('random model config must contain only "seed"')
        seed = self.config["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("random model seed must be an integer")

    def rank(
        self,
        workload: Workload,
        candidates: Sequence[Candidate],
    ) -> DecisionResult:
        if not candidates:
            raise ValueError("random model requires at least one candidate")

        candidate_ids = [candidate.candidate_id for candidate in candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("random model candidate IDs must be unique")

        started_at = perf_counter_ns()
        seed = self._workload_seed(workload.workload_id)
        random.Random(seed).shuffle(candidate_ids)
        latency_ms = (perf_counter_ns() - started_at) / 1_000_000

        return DecisionResult(
            ranked_candidate_ids=tuple(candidate_ids),
            decision_latency_ms=latency_ms,
            metadata={"seed": cast(int, self.config["seed"])},
        )

    def _workload_seed(self, workload_id: str) -> int:
        base_seed = cast(int, self.config["seed"])
        payload = f"{base_seed}\0{workload_id}".encode()
        return int.from_bytes(hashlib.sha256(payload).digest())
