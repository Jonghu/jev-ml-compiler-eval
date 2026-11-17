"""Jev decision model using generic workload and candidate serialization."""

from collections.abc import Mapping, Sequence
from math import isfinite
from pathlib import Path
from random import Random
from time import perf_counter_ns
from typing import cast

from jev_compiler_eval.jev import JevApiClient, JevApiResult, JevClientConfig
from jev_compiler_eval.models.base import BaseDecisionModel
from jev_compiler_eval.types import Candidate, DecisionResult, Workload

DECISION_QUESTION_ID = "decision"
DECISION_INSTRUCTIONS = (
    "Which candidate is most likely to have the lowest runtime for this workload?"
)
REQUIRED_CONFIG_KEYS = frozenset(
    {"model", "cache_dir", "max_retries", "request_timeout_seconds"}
)
OPTIONAL_CONFIG_KEYS = frozenset({"include_context", "candidate_order", "order_seed"})
CANDIDATE_ORDERS = ("original", "shuffled")


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
        keys = set(self.config)
        if not REQUIRED_CONFIG_KEYS <= keys:
            raise ValueError(
                "Jev model config must contain model, cache_dir, max_retries, "
                "and request_timeout_seconds"
            )
        unknown = keys - REQUIRED_CONFIG_KEYS - OPTIONAL_CONFIG_KEYS
        if unknown:
            raise ValueError(f"unknown Jev model config keys: {sorted(unknown)}")

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

        if not isinstance(self.config.get("include_context", True), bool):
            raise ValueError("Jev include_context must be a boolean")
        order = self.config.get("candidate_order", "original")
        if order not in CANDIDATE_ORDERS:
            raise ValueError('Jev candidate_order must be "original" or "shuffled"')
        has_seed = "order_seed" in self.config
        if order == "shuffled" and not has_seed:
            raise ValueError('Jev candidate_order "shuffled" requires order_seed')
        if order == "original" and has_seed:
            raise ValueError('Jev order_seed requires candidate_order "shuffled"')
        if has_seed:
            seed = self.config["order_seed"]
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise ValueError("Jev order_seed must be an integer")

    def _presented_order(
        self,
        workload: Workload,
        candidates: Sequence[Candidate],
    ) -> list[Candidate]:
        """Return the order in which candidates are sent to Jev.

        Shuffling is seeded per workload so that every rerun sends identical
        requests and therefore reuses the response cache.
        """

        presented = list(candidates)
        if self.config.get("candidate_order", "original") == "shuffled":
            seed = self.config["order_seed"]
            Random(f"{seed}:{workload.workload_id}").shuffle(presented)
        return presented

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

        state: dict[str, object] = {"workload_id": workload.workload_id}
        if self.config.get("include_context", True):
            state["context"] = workload.context
        presented = self._presented_order(workload, candidates)

        started_at = perf_counter_ns()
        response = self._client.decide(
            state=state,
            questions={
                DECISION_QUESTION_ID: {
                    "type": "choice",
                    "instructions": DECISION_INSTRUCTIONS,
                    "criteria": {
                        candidate.candidate_id: candidate.configuration
                        for candidate in presented
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
        metadata: dict[str, object] = {}
        if self.config.get("candidate_order", "original") != "original":
            metadata["presented_candidate_ids"] = [
                candidate.candidate_id for candidate in presented
            ]
        return DecisionResult(
            ranked_candidate_ids=ranking,
            decision_latency_ms=latency_ms,
            metadata={
                **metadata,
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
        return parse_choice_answer(
            response.answers.get(DECISION_QUESTION_ID), candidate_ids
        )


def parse_choice_answer(
    raw_answer: Mapping[str, object] | None,
    candidate_ids: Sequence[str],
    question_id: str = DECISION_QUESTION_ID,
) -> tuple[str, float, dict[str, float]]:
    """Validate one Jev Choice answer and return choice, confidence, probabilities."""

    if raw_answer is None:
        raise ValueError(f'Jev response is missing the "{question_id}" answer')
    if raw_answer.get("type") != "choice":
        raise ValueError(f'Jev "{question_id}" answer must have type "choice"')

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
