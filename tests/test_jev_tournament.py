from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest

import jev_compiler_eval.models.jev_tournament as tournament_module
from jev_compiler_eval.jev import JevApiResult, JevClientConfig
from jev_compiler_eval.models import JevTournamentModel, get_model_class
from jev_compiler_eval.models.jev_tournament import RateLimiter
from jev_compiler_eval.types import Candidate, Workload


class FakeJevClient:
    """Prefers the candidate with the largest "tile" value in every question."""

    instances: list["FakeJevClient"] = []
    cache_hit = False

    def __init__(self, config: JevClientConfig) -> None:
        self.calls: list[dict[str, object]] = []
        self.closed = False
        self.__class__.instances.append(self)

    def decide(
        self,
        *,
        state: object,
        questions: Mapping[str, Mapping[str, object]],
    ) -> JevApiResult:
        self.calls.append({"state": state, "questions": questions})
        answers: dict[str, dict[str, object]] = {}
        for question_id, question in questions.items():
            criteria = cast(Mapping[str, Mapping[str, int]], question["criteria"])
            tiles = {alias: config["tile"] for alias, config in criteria.items()}
            total = sum(tiles.values())
            probabilities = {alias: tile / total for alias, tile in tiles.items()}
            choice = max(tiles, key=tiles.__getitem__)
            answers[question_id] = {
                "type": "choice",
                "choice": choice,
                "confidence": probabilities[choice],
                "probabilities": probabilities,
            }
        return JevApiResult(
            model="jev-1.14.0",
            answers=answers,
            input_tokens=100,
            output_tokens=10,
            request_id=f"request-{len(self.calls)}",
            cache_hit=self.cache_hit,
            request_latency_ms=1.0,
        )

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def fake_jev_client(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeJevClient.instances.clear()
    FakeJevClient.cache_hit = False
    monkeypatch.setattr(tournament_module, "JevApiClient", FakeJevClient)


def config(tmp_path: Path, **overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "model": "jev-1.14.0",
        "cache_dir": str(tmp_path / "cache"),
        "max_retries": 3,
        "request_timeout_seconds": 30.0,
        "order_seed": 42,
        "candidate_ids": "opaque",
        "max_options": 10,
        "advance_per_question": 2,
        "max_request_tokens": 64_000,
        "max_state_question_tokens": 32_000,
        "chars_per_token": 1.4,
        "max_requests_per_minute": 1_000,
        "max_tokens_per_second": 1_000_000,
    }
    values.update(overrides)
    return values


def candidates(count: int) -> tuple[Candidate, ...]:
    return tuple(Candidate(f"config_{i}", {"tile": i + 1}) for i in range(count))


def test_tournament_model_is_registered() -> None:
    assert get_model_class("jev-tournament") is JevTournamentModel


def test_small_workload_uses_one_question(tmp_path: Path) -> None:
    model = JevTournamentModel(config(tmp_path))

    result = model.rank(Workload("w", {"op": "add"}), candidates(6))

    (call,) = FakeJevClient.instances[0].calls
    assert list(cast(Mapping[str, object], call["questions"])) == ["r0_q0"]
    assert result.ranked_candidate_ids == tuple(f"config_{i}" for i in range(5, -1, -1))
    assert result.metadata["rounds"] == [
        {"candidates": 6, "questions": 1, "requests": 1, "question_sizes": [6]}
    ]


def test_large_workload_runs_rounds_and_finds_the_best(tmp_path: Path) -> None:
    model = JevTournamentModel(config(tmp_path))

    result = model.rank(Workload("w", {}), candidates(45))

    # 45 -> 5 groups of 9 -> 10 finalists in one question.
    rounds = cast(list[dict[str, object]], result.metadata["rounds"])
    assert [r["candidates"] for r in rounds] == [45, 10]
    assert rounds[0]["question_sizes"] == [9, 9, 9, 9, 9]
    assert result.ranked_candidate_ids[0] == "config_44"
    assert sorted(result.ranked_candidate_ids) == sorted(
        c.candidate_id for c in candidates(45)
    )
    # Every round-0 question shares one request and one copy of the state.
    assert len(FakeJevClient.instances[0].calls) == 2
    assert result.metadata["requests"] == 2
    assert result.metadata["live_requests"] == 2


def test_finalists_rank_above_eliminated_candidates(tmp_path: Path) -> None:
    model = JevTournamentModel(config(tmp_path))

    result = model.rank(Workload("w", {}), candidates(45))

    finalists = set(cast(Mapping[str, float], result.metadata["final_probabilities"]))
    assert len(finalists) == 10
    assert set(result.ranked_candidate_ids[:10]) == finalists


def test_opaque_ids_hide_original_names_and_order(tmp_path: Path) -> None:
    model = JevTournamentModel(config(tmp_path))

    result = model.rank(Workload("w", {}), candidates(8))

    (call,) = FakeJevClient.instances[0].calls
    questions = cast(Mapping[str, Mapping[str, object]], call["questions"])
    criteria = cast(Mapping[str, object], questions["r0_q0"]["criteria"])
    assert list(criteria) == [f"c{i}" for i in range(1, 9)]
    presented = cast(list[str], result.metadata["presented_candidate_ids"])
    assert presented != [c.candidate_id for c in candidates(8)]
    assert [criteria[f"c{i + 1}"] for i in range(8)] == [
        {"tile": int(cid.split("_")[1]) + 1} for cid in presented
    ]


def test_order_is_reproducible_per_workload(tmp_path: Path) -> None:
    def presented(seed: int, workload_id: str) -> list[str]:
        model = JevTournamentModel(config(tmp_path, order_seed=seed))
        result = model.rank(Workload(workload_id, {}), candidates(12))
        return cast(list[str], result.metadata["presented_candidate_ids"])

    first = presented(42, "w1")
    assert presented(42, "w1") == first
    assert presented(42, "w2") != first
    assert presented(7, "w1") != first


def test_packs_questions_by_request_token_budget(tmp_path: Path) -> None:
    # Each 9-option question is estimated at about 245 tokens, so 600 fit two.
    model = JevTournamentModel(
        config(tmp_path, max_request_tokens=600, max_state_question_tokens=600)
    )

    result = model.rank(Workload("w", {}), candidates(45))

    calls = FakeJevClient.instances[0].calls
    sizes = [len(cast(Mapping[str, object], c["questions"])) for c in calls]
    assert sizes == [2, 2, 1, 1]
    assert result.ranked_candidate_ids[0] == "config_44"


def test_rejects_state_too_large_for_any_question(tmp_path: Path) -> None:
    model = JevTournamentModel(
        config(tmp_path, max_request_tokens=100, max_state_question_tokens=100)
    )

    with pytest.raises(ValueError, match="state is too large"):
        model.rank(Workload("w", {"text": "x" * 1_000}), candidates(4))


def test_can_omit_context(tmp_path: Path) -> None:
    model = JevTournamentModel(config(tmp_path, include_context=False))

    model.rank(Workload("w", {"op": "add"}), candidates(3))

    assert FakeJevClient.instances[0].calls[0]["state"] == {"workload_id": "w"}


def test_cache_hits_do_not_count_as_live(tmp_path: Path) -> None:
    FakeJevClient.cache_hit = True
    model = JevTournamentModel(config(tmp_path))

    result = model.rank(Workload("w", {}), candidates(45))

    assert result.metadata["live_requests"] == 0
    assert result.metadata["requests"] == 2


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"candidate_ids": "named"}, "candidate_ids must be"),
        ({"max_options": 1}, "max_options must be at least 2"),
        ({"advance_per_question": 10}, "at most half of max_options"),
        ({"advance_per_question": 6}, "at most half of max_options"),
        ({"max_state_question_tokens": 70_000}, "must not exceed"),
        ({"chars_per_token": 0}, "chars_per_token must be a positive number"),
        ({"order_seed": True}, "order_seed must be an integer"),
        ({"include_context": "yes"}, "include_context must be a boolean"),
        ({"temperature": 0}, "unknown Jev tournament config keys"),
    ],
)
def test_rejects_invalid_config(
    tmp_path: Path,
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        JevTournamentModel(config(tmp_path, **overrides))


def test_rejects_missing_config(tmp_path: Path) -> None:
    values = config(tmp_path)
    del values["order_seed"]

    with pytest.raises(ValueError, match="missing"):
        JevTournamentModel(values)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def test_rate_limiter_enforces_tokens_per_second() -> None:
    clock = FakeClock()
    limiter = RateLimiter(1_000, 100, clock=clock, sleep=clock.sleep)

    limiter.acquire(60)
    limiter.acquire(60)

    assert clock.sleeps == [pytest.approx(1.0)]


def test_rate_limiter_enforces_requests_per_minute() -> None:
    clock = FakeClock()
    limiter = RateLimiter(2, 1_000_000, clock=clock, sleep=clock.sleep)

    for _ in range(3):
        limiter.acquire(1)

    assert sum(clock.sleeps) == pytest.approx(60.0)


def test_rate_limiter_lets_oversized_requests_through_alone() -> None:
    clock = FakeClock()
    limiter = RateLimiter(1_000, 100, clock=clock, sleep=clock.sleep)

    limiter.acquire(500)
    assert clock.sleeps == []
    limiter.acquire(10)
    assert sum(clock.sleeps) == pytest.approx(1.0)


def test_rate_limiter_release_frees_a_cache_hit_reservation() -> None:
    clock = FakeClock()
    limiter = RateLimiter(1, 100, clock=clock, sleep=clock.sleep)

    limiter.acquire(90)
    limiter.release_last()
    limiter.acquire(90)

    assert clock.sleeps == []
