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
        self.config = config
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
            request_sha256=f"request-hash-{len(self.calls)}",
            cache_path=f"fake-cache/request-hash-{len(self.calls)}.json",
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


def test_option_limit_is_checked_before_client_construction(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at most 255"):
        JevTournamentModel(config(tmp_path, max_options=256))
    assert FakeJevClient.instances == []


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


@pytest.mark.parametrize("long_context", [False, True])
def test_long_context_mode_reaches_client_and_prediction_metadata(
    tmp_path: Path, long_context: bool
) -> None:
    model = JevTournamentModel(config(tmp_path, long_context=long_context))

    result = model.rank(Workload("w", {}), candidates(6))

    assert FakeJevClient.instances[0].config.long_context is long_context
    assert (result.metadata.get("long_context") is True) is long_context


@pytest.mark.parametrize("value", [None, 0, 1, "true", "false", [], {}])
def test_rejects_non_boolean_long_context_before_constructing_client(
    tmp_path: Path, value: object
) -> None:
    with pytest.raises(ValueError, match="long_context.*boolean"):
        JevTournamentModel(config(tmp_path, long_context=value))

    assert FakeJevClient.instances == []


def test_question_probabilities_remain_local_and_tail_is_labeled(
    tmp_path: Path,
) -> None:
    from jev_compiler_eval.metrics import pairwise_accuracy, runtime_regret
    from jev_compiler_eval.types import EvaluationCase, RuntimeRecord

    model = JevTournamentModel(config(tmp_path))
    options = candidates(45)
    workload = Workload("w", {})

    result = model.rank(workload, options)

    assert "probabilities" not in result.metadata
    assert "scores" not in result.metadata
    assert result.metadata["probability_scope"] == "within_question_only"
    assert result.metadata["ranking_scope"] == "finalists_then_elimination_round"
    last = cast(Mapping[str, float], result.metadata["last_question_probabilities"])
    assert set(last) == {candidate.candidate_id for candidate in options}
    # This actual multi-round example has scores that are not globally ordered.
    assert any(
        last[left] < last[right]
        for left, right in zip(
            result.ranked_candidate_ids, result.ranked_candidate_ids[1:]
        )
    )
    finalists = cast(list[str], result.metadata["finalist_candidate_ids"])
    eliminated = cast(list[dict[str, object]], result.metadata["eliminated_candidates"])
    assert result.ranked_candidate_ids[: len(finalists)] == tuple(finalists)
    assert set(finalists).isdisjoint(entry["candidate_id"] for entry in eliminated)
    assert len(finalists) + len(eliminated) == len(options)
    case = EvaluationCase(
        workload,
        options,
        tuple(
            RuntimeRecord(
                "w",
                candidate.candidate_id,
                1 / cast(int, candidate.configuration["tile"]),
            )
            for candidate in options
        ),
    )
    assert runtime_regret(case, result) == 0.0
    # This is explicitly a total-rank diagnostic, without cross-question scores.
    assert 0 <= pairwise_accuracy(case, result) <= 1


def test_reports_preserve_each_round_answer_and_cache_reference(tmp_path: Path) -> None:
    model = JevTournamentModel(config(tmp_path))

    result = model.rank(Workload("w", {}), candidates(45))

    requests = cast(list[dict[str, object]], result.metadata["request_reports"])
    questions = cast(list[dict[str, object]], result.metadata["question_reports"])
    assert len(requests) == 2
    assert len(questions) == 6
    for index, request in enumerate(requests, 1):
        assert request["request_sha256"] == f"request-hash-{index}"
        assert request["cache_path"] == f"fake-cache/request-hash-{index}.json"
        assert request["cache_hit"] is False
        assert request["input_tokens"] == 100
    assert [question["request_index"] for question in questions] == [0] * 5 + [1]
    early = next(
        question
        for question in questions
        if question["round"] == 0 and "config_44" in question["candidate_ids"]
    )
    final = questions[-1]
    early_probabilities = cast(Mapping[str, float], early["probabilities"])
    final_probabilities = cast(Mapping[str, float], final["probabilities"])
    assert early_probabilities["config_44"] != final_probabilities["config_44"]
    assert early["selected_candidate_id"] == "config_44"
    assert final["selected_candidate_id"] == "config_44"
    assert result.metadata["final_probabilities"] == final_probabilities


def test_token_budget_that_prevents_elimination_fails_before_request(
    tmp_path: Path,
) -> None:
    model = JevTournamentModel(
        config(tmp_path, max_request_tokens=180, max_state_question_tokens=180)
    )

    with pytest.raises(ValueError, match="no candidates to eliminate"):
        model.rank(Workload("w", {}), candidates(45))

    assert FakeJevClient.instances[0].calls == []


def test_offline_plan_constructs_no_client_and_matches_first_round(
    tmp_path: Path,
) -> None:
    values = config(
        tmp_path,
        long_context=True,
        max_request_tokens=600,
        max_state_question_tokens=600,
    )
    workload = Workload("w", {})
    options = candidates(45)

    plan = JevTournamentModel.plan(workload, options, values)

    assert FakeJevClient.instances == []
    assert plan["candidate_count"] == 45
    assert plan["first_round_question_sizes"] == [9] * 5
    assert plan["first_round_questions"] == 5
    assert plan["first_round_requests"] == 3
    assert plan["long_context"] is True
    assert plan["max_estimated_state_question_tokens"] <= 600
    assert plan["max_estimated_request_tokens"] <= 600
    model = JevTournamentModel(values)
    result = model.rank(workload, options)
    first = cast(list[dict[str, object]], result.metadata["rounds"])[0]
    assert first["question_sizes"] == plan["first_round_question_sizes"]
    assert first["requests"] == plan["first_round_requests"]


def test_offline_plan_fails_for_oversized_context_without_client(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="state is too large"):
        JevTournamentModel.plan(
            Workload("w", {"text": "x" * 1_000}),
            candidates(4),
            config(tmp_path, max_request_tokens=100, max_state_question_tokens=100),
        )

    assert FakeJevClient.instances == []


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
        ({"long_context": "true"}, "long_context must be a boolean"),
        ({"temperature": 0}, "unknown Jev tournament config keys"),
        ({"candidate_order": "sorted"}, "candidate_order must be one of"),
        ({"candidate_order": "size_ascending"}, "size_fields is required"),
        ({"keep_largest": 3}, "size_fields is required"),
        ({"keep_largest": 1, "size_fields": ["tile"]}, "keep_largest must be at"),
        ({"size_fields": "tile"}, "size_fields must be a non-empty list"),
    ],
)
def test_rejects_invalid_config(
    tmp_path: Path,
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        JevTournamentModel(config(tmp_path, **overrides))


def sized_candidates() -> tuple[Candidate, ...]:
    # Sizes: config_0 = 6, config_1 = 2, config_2 = 12, config_3 = 1, config_4 = 6.
    return (
        Candidate("config_0", {"tile": 1, "bounds": [2, 3]}),
        Candidate("config_1", {"tile": 2, "bounds": [2]}),
        Candidate("config_2", {"tile": 3, "bounds": [3, 4]}),
        Candidate("config_3", {"tile": 4, "bounds": [1, 1]}),
        Candidate("config_4", {"tile": 5, "bounds": [6, 1]}),
    )


def presented_sizes(call: Mapping[str, object]) -> list[int]:
    questions = cast(Mapping[str, Mapping[str, object]], call["questions"])
    criteria = cast(Mapping[str, Mapping[str, object]], questions["r0_q0"]["criteria"])
    sizes = []
    for configuration in criteria.values():
        size = 1
        for value in cast(list[int], configuration.get("bounds", [])):
            size *= value
        sizes.append(size)
    return sizes


@pytest.mark.parametrize(
    ("order", "expected"),
    [("size_ascending", [1, 2, 6, 6, 12]), ("size_descending", [12, 6, 6, 2, 1])],
)
def test_size_orders_sort_presented_candidates(
    tmp_path: Path, order: str, expected: list[int]
) -> None:
    model = JevTournamentModel(
        config(tmp_path, candidate_order=order, size_fields=["bounds"])
    )

    result = model.rank(Workload("w", {}), sized_candidates())

    (call,) = FakeJevClient.instances[0].calls
    assert presented_sizes(call) == expected
    assert result.metadata["candidate_order"] == order
    assert result.metadata["filtered_candidate_ids"] == []


def test_size_order_ties_keep_the_shuffled_order(tmp_path: Path) -> None:
    shuffled = JevTournamentModel(config(tmp_path))
    ordered = JevTournamentModel(
        config(tmp_path, candidate_order="size_ascending", size_fields=["bounds"])
    )

    base = shuffled.rank(Workload("w", {}), sized_candidates())
    result = ordered.rank(Workload("w", {}), sized_candidates())

    base_order = cast(list[str], base.metadata["presented_candidate_ids"])
    tied = [cid for cid in base_order if cid in {"config_0", "config_4"}]
    presented = cast(list[str], result.metadata["presented_candidate_ids"])
    assert presented[2:4] == tied


def test_keep_largest_hides_small_candidates_and_ranks_them_last(
    tmp_path: Path,
) -> None:
    model = JevTournamentModel(config(tmp_path, keep_largest=3, size_fields=["bounds"]))

    result = model.rank(Workload("w", {}), sized_candidates())

    (call,) = FakeJevClient.instances[0].calls
    assert sorted(presented_sizes(call)) == [6, 6, 12]
    assert result.metadata["filtered_candidate_ids"] == ["config_1", "config_3"]
    assert set(result.ranked_candidate_ids[:3]) == {"config_0", "config_2", "config_4"}
    assert result.ranked_candidate_ids[3:] == ("config_1", "config_3")
    presented = set(cast(list[str], result.metadata["presented_candidate_ids"]))
    assert presented == {"config_0", "config_2", "config_4"}


def test_keep_largest_leaves_small_workloads_unchanged(tmp_path: Path) -> None:
    unfiltered = JevTournamentModel(config(tmp_path))
    filtered = JevTournamentModel(
        config(tmp_path, keep_largest=5, size_fields=["bounds"])
    )

    unfiltered.rank(Workload("w", {}), sized_candidates())
    filtered.rank(Workload("w", {}), sized_candidates())

    first, second = FakeJevClient.instances
    assert first.calls == second.calls


def test_offline_plan_counts_filtered_candidates(tmp_path: Path) -> None:
    values = config(tmp_path, keep_largest=3, size_fields=["bounds"])

    plan = JevTournamentModel.plan(Workload("w", {}), sized_candidates(), values)

    assert plan["filtered_candidate_count"] == 2
    assert plan["first_round_question_sizes"] == [3]
    assert FakeJevClient.instances == []


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


def test_reused_model_reports_only_each_workloads_rate_limit_wait(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    model = JevTournamentModel(config(tmp_path))
    model._limiter = RateLimiter(1, 1_000_000, clock=clock, sleep=clock.sleep)

    first = model.rank(Workload("first", {}), candidates(45))
    second = model.rank(Workload("second", {}), candidates(45))

    assert first.metadata["rate_limit_wait_seconds"] == pytest.approx(60)
    assert second.metadata["rate_limit_wait_seconds"] == pytest.approx(120)
    assert model._limiter.waited_seconds == pytest.approx(180)
