"""Jev decision model using generic workload and candidate serialization."""

from collections.abc import Mapping, Sequence
from math import isfinite
from pathlib import Path
from time import perf_counter_ns
from typing import cast

from jev_compiler_eval.jev import JevApiClient, JevApiResult, JevClientConfig
from jev_compiler_eval.models.base import BaseDecisionModel
from jev_compiler_eval.types import Candidate, DecisionResult, Workload

DECISION_QUESTION_ID = "decision"
DECISION_INSTRUCTIONS = (
    "Which candidate is most likely to have the lowest runtime for this workload?"
)


class JevModel(BaseDecisionModel):
    """Rank compiler candidates using TypeSafe Jev Choice probabilities."""

    def __init__(self, config: Mapping[str, object]) -> None:
        super().__init__(config)
        self._client = JevApiClient(
            JevClientConfig(
                model=cast(str, self.config["model"]),
                cache_dir=Path(cast(str, self.config["cache_dir"])),
                max_retries=cast(int, self.config["max_retries"]),
                request_timeout_seconds=cast(
                    float, self.config["request_timeout_seconds"]
                ),
            )
        )

    @property
    def model_type(self) -> str:
        return "jev"

    def validate_config(self) -> None:
        expected = {
            "model",
            "cache_dir",
            "max_retries",
            "request_timeout_seconds",
        }
        if set(self.config) != expected:
            raise ValueError(
                "Jev model config must contain model, cache_dir, max_retries, "
                "and request_timeout_seconds"
            )

        model = self.config["model"]
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Jev model must be a non-empty string")
        cache_dir = self.config["cache_dir"]
        if not isinstance(cache_dir, str) or not cache_dir.strip():
            raise ValueError("Jev cache_dir must be a non-empty string")
        max_retries = self.config["max_retries"]
        if (
            isinstance(max_retries, bool)
            or not isinstance(max_retries, int)
            or max_retries < 0
        ):
            raise ValueError("Jev max_retries must be a non-negative integer")
        timeout = self.config["request_timeout_seconds"]
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("Jev request timeout must be a finite positive number")

    def rank(
        self,
        workload: Workload,
        candidates: Sequence[Candidate],
    ) -> DecisionResult:
        if not candidates:
            raise ValueError("Jev model requires at least one candidate")
        candidate_ids = [candidate.candidate_id for candidate in candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("Jev model candidate IDs must be unique")

        started_at = perf_counter_ns()
        response = self._client.decide(
            state={
                "workload_id": workload.workload_id,
                "context": workload.context,
            },
            questions={
                DECISION_QUESTION_ID: {
                    "type": "choice",
                    "instructions": DECISION_INSTRUCTIONS,
                    "criteria": {
                        candidate.candidate_id: candidate.configuration
                        for candidate in candidates
                    },
                }
            },
        )
        choice, confidence, probabilities = self._parse_choice(
            response,
            candidate_ids,
        )
        ranking = tuple(
            sorted(
                candidate_ids,
                key=lambda candidate_id: (
                    candidate_id == choice,
                    probabilities[candidate_id],
                ),
                reverse=True,
            )
        )
        latency_ms = (perf_counter_ns() - started_at) / 1_000_000
        return DecisionResult(
            ranked_candidate_ids=ranking,
            decision_latency_ms=latency_ms,
            metadata={
                "requested_model": self.config["model"],
                "resolved_model": response.model,
                "probabilities": probabilities,
                "confidence": confidence,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "request_id": response.request_id,
                "cache_hit": response.cache_hit,
                "api_latency_ms": response.request_latency_ms,
            },
        )

    def close(self) -> None:
        """Close the shared Jev API client."""

        self._client.close()

    def _parse_choice(
        self,
        response: JevApiResult,
        candidate_ids: Sequence[str],
    ) -> tuple[str, float, dict[str, float]]:
        raw_answer = response.answers.get(DECISION_QUESTION_ID)
        if raw_answer is None:
            raise ValueError('Jev response is missing the "decision" answer')
        if raw_answer.get("type") != "choice":
            raise ValueError('Jev "decision" answer must have type "choice"')

        choice = raw_answer.get("choice")
        if not isinstance(choice, str) or choice not in candidate_ids:
            raise ValueError("Jev response choice must match an evaluation candidate")
        confidence = raw_answer.get("confidence")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not isfinite(confidence)
            or not 0 <= confidence <= 1
        ):
            raise ValueError("Jev response confidence must be between zero and one")

        raw_probabilities = raw_answer.get("probabilities")
        if not isinstance(raw_probabilities, Mapping):
            raise ValueError("Jev response probabilities must be a mapping")
        if set(raw_probabilities) != set(candidate_ids):
            raise ValueError(
                "Jev response probabilities must match evaluation candidates exactly"
            )

        probabilities: dict[str, float] = {}
        for candidate_id, value in raw_probabilities.items():
            if (
                not isinstance(candidate_id, str)
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError(
                    "Jev response probabilities must be numbers between zero and one"
                )
            probabilities[candidate_id] = float(value)

        if probabilities[choice] != max(probabilities.values()):
            raise ValueError("Jev response choice must have the highest probability")
        return choice, float(confidence), probabilities
