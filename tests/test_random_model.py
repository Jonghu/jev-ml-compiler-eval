import pytest

from jev_compiler_eval.models import RandomModel, get_model_class
from jev_compiler_eval.types import Candidate, Workload


def make_candidates(count: int = 6) -> tuple[Candidate, ...]:
    return tuple(
        Candidate(f"candidate-{index}", {"index": index}) for index in range(count)
    )


def test_random_model_is_registered() -> None:
    assert get_model_class("random") is RandomModel


def test_same_seed_and_workload_produce_same_ranking() -> None:
    workload = Workload("workload-1", {})
    candidates = make_candidates()

    first = RandomModel({"seed": 42}).rank(workload, candidates)
    second = RandomModel({"seed": 42}).rank(workload, candidates)

    assert first.ranked_candidate_ids == second.ranked_candidate_ids


def test_ranking_is_independent_of_workload_execution_order() -> None:
    workload_a = Workload("workload-a", {})
    workload_b = Workload("workload-b", {})
    candidates = make_candidates()

    first_model = RandomModel({"seed": 42})
    first_a = first_model.rank(workload_a, candidates)
    first_b = first_model.rank(workload_b, candidates)

    second_model = RandomModel({"seed": 42})
    second_b = second_model.rank(workload_b, candidates)
    second_a = second_model.rank(workload_a, candidates)

    assert first_a.ranked_candidate_ids == second_a.ranked_candidate_ids
    assert first_b.ranked_candidate_ids == second_b.ranked_candidate_ids


def test_ranking_contains_every_candidate_once() -> None:
    candidates = make_candidates()

    result = RandomModel({"seed": 42}).rank(Workload("workload-1", {}), candidates)

    assert set(result.ranked_candidate_ids) == {
        candidate.candidate_id for candidate in candidates
    }
    assert len(result.ranked_candidate_ids) == len(candidates)
    assert result.metadata == {"seed": 42}


def test_random_model_rejects_empty_candidates() -> None:
    with pytest.raises(ValueError, match="requires at least one candidate"):
        RandomModel({"seed": 42}).rank(Workload("workload-1", {}), ())


def test_random_model_rejects_missing_seed() -> None:
    with pytest.raises(
        ValueError, match='random model config must contain only "seed"'
    ):
        RandomModel({})


def test_random_model_rejects_unknown_config_fields() -> None:
    with pytest.raises(
        ValueError, match='random model config must contain only "seed"'
    ):
        RandomModel({"seed": 42, "temperature": 0.0})


def test_random_model_rejects_non_integer_seed() -> None:
    with pytest.raises(ValueError, match="random model seed must be an integer"):
        RandomModel({"seed": True})
