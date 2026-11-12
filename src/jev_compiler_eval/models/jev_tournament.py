"""Jev decision model that ranks large candidate sets with a Choice tournament.

One Choice question accepts a limited number of options, so a workload with
more candidates is split into balanced groups. Each group is one Choice
question; the top ``advance_per_question`` candidates of every group advance
to the next round until one question holds every remaining candidate. All
questions of a round that fit the token budget share one request, so the
workload state is sent once per request.
"""

import json
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import ceil, isfinite
from pathlib import Path
from random import Random
from typing import cast

from jev_compiler_eval.jev import JevApiClient, JevApiResult, JevClientConfig
from jev_compiler_eval.models.base import BaseDecisionModel
from jev_compiler_eval.models.jev import DECISION_INSTRUCTIONS, parse_choice_answer
from jev_compiler_eval.types import Candidate, DecisionResult, Workload

CANDIDATE_ID_MODES = ("opaque", "original")
REQUIRED_CONFIG_KEYS = frozenset(
    {
        "model",
        "cache_dir",
        "max_retries",
        "request_timeout_seconds",
        "order_seed",
        "candidate_ids",
        "max_options",
        "advance_per_question",
        "max_request_tokens",
        "max_state_question_tokens",
        "chars_per_token",
        "max_requests_per_minute",
        "max_tokens_per_second",
    }
)
OPTIONAL_CONFIG_KEYS = frozenset({"include_context"})
# Fixed allowance for JSON framing and per-question fields in token estimates.
QUESTION_OVERHEAD_TOKENS = 40


def _compact_json_length(value: object) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


class RateLimiter:
    """Client-side limiter for requests per minute and input tokens per second.

    Only live requests count. A reservation made for a request that was then
    served from the cache is released again.
    """

    def __init__(
        self,
        max_requests_per_minute: int,
        max_tokens_per_second: int,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.max_requests_per_minute = max_requests_per_minute
        self.max_tokens_per_second = max_tokens_per_second
        self._clock = clock
        self._sleep = sleep
        self._requests: deque[float] = deque()
        self._tokens: deque[tuple[float, int]] = deque()
        self.waited_seconds = 0.0

    def _prune(self, now: float) -> None:
        while self._requests and now - self._requests[0] >= 60.0:
            self._requests.popleft()
        while self._tokens and now - self._tokens[0][0] >= 1.0:
            self._tokens.popleft()

    def acquire(self, estimated_tokens: int) -> None:
        """Block until one more request of the estimated size fits the limits."""

        while True:
            now = self._clock()
            self._prune(now)
            used = sum(tokens for _, tokens in self._tokens)
            requests_ok = len(self._requests) < self.max_requests_per_minute
            # A request larger than the per-second budget waits for an empty window.
            tokens_ok = not self._tokens or (
                used + estimated_tokens <= self.max_tokens_per_second
            )
            if requests_ok and tokens_ok:
                self._requests.append(now)
                self._tokens.append((now, estimated_tokens))
                return
            waits = []
            if not requests_ok:
                waits.append(60.0 - (now - self._requests[0]))
            if not tokens_ok:
                waits.append(1.0 - (now - self._tokens[0][0]))
            delay = max(min(waits), 0.001)
            self.waited_seconds += delay
            self._sleep(delay)

    def release_last(self) -> None:
        """Undo the most recent reservation, used after a cache hit."""

        if self._requests:
            self._requests.pop()
        if self._tokens:
            self._tokens.pop()


@dataclass
class _Question:
    question_id: str
    aliases: list[str]
    estimated_tokens: int


class JevTournamentModel(BaseDecisionModel):
    """Rank any number of candidates with batched Jev Choice questions."""

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
        self._limiter = RateLimiter(
            cast(int, self.config["max_requests_per_minute"]),
            cast(int, self.config["max_tokens_per_second"]),
        )

    @property
    def model_type(self) -> str:
        return "jev-tournament"

    def validate_config(self) -> None:
        keys = set(self.config)
        missing = REQUIRED_CONFIG_KEYS - keys
        if missing:
            raise ValueError(f"Jev tournament config is missing: {sorted(missing)}")
        unknown = keys - REQUIRED_CONFIG_KEYS - OPTIONAL_CONFIG_KEYS
        if unknown:
            raise ValueError(f"unknown Jev tournament config keys: {sorted(unknown)}")

        for name in ("model", "cache_dir"):
            value = self.config[name]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Jev tournament {name} must be a non-empty string")

        def integer(name: str, minimum: int) -> int:
            value = self.config[name]
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"Jev tournament {name} must be an integer")
            if value < minimum:
                raise ValueError(f"Jev tournament {name} must be at least {minimum}")
            return value

        def positive_number(name: str) -> float:
            value = self.config[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"Jev tournament {name} must be a positive number")
            return float(value)

        integer("max_retries", 0)
        positive_number("request_timeout_seconds")
        integer("order_seed", -(2**63))
        max_options = integer("max_options", 2)
        advance = integer("advance_per_question", 1)
        # With at most max_options per group and at least two groups, advancing
        # at most half of max_options per group always shrinks the field.
        if 2 * advance > max_options:
            raise ValueError(
                "Jev tournament advance_per_question must be at most half of "
                "max_options"
            )
        max_request = integer("max_request_tokens", 1)
        max_state_question = integer("max_state_question_tokens", 1)
        if max_state_question > max_request:
            raise ValueError(
                "Jev tournament max_state_question_tokens must not exceed "
                "max_request_tokens"
            )
        positive_number("chars_per_token")
        integer("max_requests_per_minute", 1)
        integer("max_tokens_per_second", 1)
        if self.config["candidate_ids"] not in CANDIDATE_ID_MODES:
            raise ValueError(
                'Jev tournament candidate_ids must be "opaque" or "original"'
            )
        if not isinstance(self.config.get("include_context", True), bool):
            raise ValueError("Jev tournament include_context must be a boolean")

    def close(self) -> None:
        """Close the shared Jev API client."""

        self._client.close()

    # Token estimates -----------------------------------------------------

    def _tokens(self, characters: int) -> int:
        return ceil(characters / cast(float, self.config["chars_per_token"]))

    def _question_body(
        self,
        aliases: Sequence[str],
        configurations: Mapping[str, Mapping[str, object]],
    ) -> dict[str, object]:
        return {
            "type": "choice",
            "instructions": DECISION_INSTRUCTIONS,
            "criteria": {alias: configurations[alias] for alias in aliases},
        }

    def _split(
        self,
        pool: list[str],
        configurations: Mapping[str, Mapping[str, object]],
        state_tokens: int,
        round_index: int,
    ) -> list[_Question]:
        """Split the pool into the fewest balanced groups that fit every limit."""

        max_options = cast(int, self.config["max_options"])
        limit = cast(int, self.config["max_state_question_tokens"])
        groups = ceil(len(pool) / max_options)
        while True:
            size, extra = divmod(len(pool), groups)
            questions: list[_Question] = []
            start = 0
            for index in range(groups):
                end = start + size + (1 if index < extra else 0)
                aliases = pool[start:end]
                start = end
                body = self._question_body(aliases, configurations)
                tokens = (
                    self._tokens(_compact_json_length(body)) + QUESTION_OVERHEAD_TOKENS
                )
                questions.append(_Question(f"r{round_index}_q{index}", aliases, tokens))
            if all(state_tokens + q.estimated_tokens <= limit for q in questions):
                return questions
            if size < 2:
                raise ValueError(
                    "a Jev Choice question does not fit max_state_question_tokens; "
                    "the workload state is too large"
                )
            groups += 1

    def _pack(
        self, questions: list[_Question], state_tokens: int
    ) -> list[list[_Question]]:
        limit = cast(int, self.config["max_request_tokens"])
        requests: list[list[_Question]] = []
        current: list[_Question] = []
        used = state_tokens
        for question in questions:
            if current and used + question.estimated_tokens > limit:
                requests.append(current)
                current, used = [], state_tokens
            current.append(question)
            used += question.estimated_tokens
        if current:
            requests.append(current)
        return requests

    # Ranking -------------------------------------------------------------

    def rank(
        self,
        workload: Workload,
        candidates: Sequence[Candidate],
    ) -> DecisionResult:
        if len(candidates) < 2:
            raise ValueError("Jev tournament requires at least two candidates")
        original_ids = [candidate.candidate_id for candidate in candidates]
        if len(original_ids) != len(set(original_ids)):
            raise ValueError("Jev tournament candidate IDs must be unique")

        started_at = time.perf_counter_ns()
        presented = list(candidates)
        Random(f"{self.config['order_seed']}:{workload.workload_id}").shuffle(presented)
        if self.config["candidate_ids"] == "opaque":
            width = len(str(len(presented)))
            aliases = [f"c{index + 1:0{width}d}" for index in range(len(presented))]
        else:
            aliases = [candidate.candidate_id for candidate in presented]
        to_original = {
            alias: candidate.candidate_id
            for alias, candidate in zip(aliases, presented, strict=True)
        }
        configurations = {
            alias: candidate.configuration
            for alias, candidate in zip(aliases, presented, strict=True)
        }
        presented_index = {alias: index for index, alias in enumerate(aliases)}

        state: dict[str, object] = {"workload_id": workload.workload_id}
        if self.config.get("include_context", True):
            state["context"] = workload.context
        state_tokens = self._tokens(_compact_json_length(state))

        advance = cast(int, self.config["advance_per_question"])
        pool = list(aliases)
        # (round, position within its question, presented index) per eliminated alias
        eliminated: dict[str, tuple[int, int, int]] = {}
        probabilities: dict[str, float] = {}
        rounds: list[dict[str, object]] = []
        totals = {
            "requests": 0,
            "live_requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "estimated_live_input_tokens": 0,
        }
        api_latency_ms = 0.0
        resolved_models: set[str] = set()
        request_ids: list[str] = []
        round_index = 0
        while True:
            questions = self._split(pool, configurations, state_tokens, round_index)
            packed = self._pack(questions, state_tokens)
            orders: dict[str, list[str]] = {}
            final_confidence = 0.0
            for request_questions in packed:
                estimate = state_tokens + sum(
                    q.estimated_tokens for q in request_questions
                )
                self._limiter.acquire(estimate)
                response = self._client.decide(
                    state=state,
                    questions={
                        q.question_id: self._question_body(q.aliases, configurations)
                        for q in request_questions
                    },
                )
                if response.cache_hit:
                    self._limiter.release_last()
                else:
                    totals["live_requests"] += 1
                    totals["estimated_live_input_tokens"] += estimate
                totals["requests"] += 1
                totals["input_tokens"] += response.input_tokens or 0
                totals["output_tokens"] += response.output_tokens or 0
                api_latency_ms += response.request_latency_ms
                resolved_models.add(response.model)
                if response.request_id:
                    request_ids.append(response.request_id)
                for question in request_questions:
                    orders[question.question_id], confidence = self._answer_order(
                        response, question, presented_index, probabilities
                    )
                    final_confidence = confidence

            rounds.append(
                {
                    "candidates": len(pool),
                    "questions": len(questions),
                    "requests": len(packed),
                    "question_sizes": [len(q.aliases) for q in questions],
                }
            )
            if len(questions) == 1:
                final_order = orders[questions[0].question_id]
                break
            pool = []
            for question in questions:
                order = orders[question.question_id]
                pool.extend(order[:advance])
                for position, alias in enumerate(order[advance:], start=advance):
                    eliminated[alias] = (round_index, position, presented_index[alias])
            # Keep the presented order inside the next round's groups.
            pool.sort(key=presented_index.__getitem__)
            round_index += 1

        losers = sorted(
            eliminated,
            key=lambda alias: (
                -eliminated[alias][0],
                eliminated[alias][1],
                eliminated[alias][2],
            ),
        )
        ranking = tuple(to_original[alias] for alias in [*final_order, *losers])
        latency_ms = (time.perf_counter_ns() - started_at) / 1_000_000
        return DecisionResult(
            ranked_candidate_ids=ranking,
            decision_latency_ms=latency_ms,
            metadata={
                "requested_model": self.config["model"],
                "resolved_models": sorted(resolved_models),
                "candidate_ids": self.config["candidate_ids"],
                "presented_candidate_ids": [c.candidate_id for c in presented],
                "rounds": rounds,
                **totals,
                "api_latency_ms": api_latency_ms,
                "final_confidence": final_confidence,
                "final_probabilities": {
                    to_original[alias]: probabilities[alias] for alias in final_order
                },
                "probabilities": {
                    to_original[alias]: value for alias, value in probabilities.items()
                },
                "request_ids": request_ids,
                "rate_limit_wait_seconds": self._limiter.waited_seconds,
            },
        )

    def _answer_order(
        self,
        response: JevApiResult,
        question: _Question,
        presented_index: Mapping[str, int],
        probabilities: dict[str, float],
    ) -> tuple[list[str], float]:
        """Return a question's aliases ordered best first, and its confidence.

        The chosen candidate comes first, then descending probability; ties
        keep the presented order.
        """

        choice, confidence, answer = parse_choice_answer(
            response.answers.get(question.question_id),
            question.aliases,
            question.question_id,
        )
        # Later rounds overwrite earlier probabilities for advancing candidates.
        probabilities.update(answer)
        order = sorted(
            question.aliases,
            key=lambda alias: (
                alias != choice,
                -answer[alias],
                presented_index[alias],
            ),
        )
        return order, confidence
