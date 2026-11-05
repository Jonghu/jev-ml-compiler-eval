from math import sqrt

import pytest

from jev_compiler_eval.metrics import (
    kendall_tau,
    resolve_metrics,
    runtime_regret,
    top_1_accuracy,
    top_k_error,
)
from jev_compiler_eval.types import (
    Candidate,
    DecisionResult,
    EvaluationCase,
    RuntimeRecord,
    Workload,
)


def make_case(second_runtime: float = 1.0) -> EvaluationCase:
    return EvaluationCase(
        workload=Workload("workload-1", {}),
        candidates=(Candidate("slow", {}), Candidate("fast", {})),
        runtimes=(
            RuntimeRecord("workload-1", "slow", 2.0),
            RuntimeRecord("workload-1", "fast", second_runtime),
        ),
    )


def make_result(selected: str, other: str) -> DecisionResult:
    return DecisionResult(
        ranked_candidate_ids=(selected, other),
        decision_latency_ms=0.0,
    )


def test_runtime_regret_compares_selected_and_optimal_runtime() -> None:
    regret = runtime_regret(make_case(), make_result("slow", "fast"))

    assert regret == pytest.approx(1.0)


def test_top_1_accuracy_identifies_fastest_candidate() -> None:
    assert top_1_accuracy(make_case(), make_result("fast", "slow")) == 1.0
    assert top_1_accuracy(make_case(), make_result("slow", "fast")) == 0.0


def test_tied_fastest_candidate_is_correct_with_zero_regret() -> None:
    case = make_case(second_runtime=2.0)
    result = make_result("slow", "fast")

    assert top_1_accuracy(case, result) == 1.0
    assert runtime_regret(case, result) == 0.0


def make_ranked_case(runtimes: dict[str, float]) -> EvaluationCase:
    return EvaluationCase(
        workload=Workload("workload-1", {}),
        candidates=tuple(Candidate(candidate_id, {}) for candidate_id in runtimes),
        runtimes=tuple(
            RuntimeRecord("workload-1", candidate_id, runtime)
            for candidate_id, runtime in runtimes.items()
        ),
    )


def make_ranking(*candidate_ids: str) -> DecisionResult:
    return DecisionResult(ranked_candidate_ids=candidate_ids, decision_latency_ms=0.0)


RANKED_RUNTIMES = {"a": 4.0, "b": 2.0, "c": 3.0, "d": 1.0, "e": 5.0}


def test_top_k_error_uses_fastest_candidate_within_top_k() -> None:
    case = make_ranked_case(RANKED_RUNTIMES)
    result = make_ranking("a", "c", "b", "e", "d")

    assert top_k_error(case, result, k=1) == pytest.approx(3.0)
    assert top_k_error(case, result, k=2) == pytest.approx(2.0)
    assert top_k_error(case, result, k=3) == pytest.approx(1.0)
    assert top_k_error(case, result, k=5) == pytest.approx(0.0)


def test_top_k_error_is_zero_when_k_exceeds_candidate_count() -> None:
    case = make_ranked_case(RANKED_RUNTIMES)

    assert top_k_error(case, make_ranking("e", "a", "c", "b", "d"), k=10) == 0.0


def test_top_1_error_matches_runtime_regret() -> None:
    case = make_ranked_case(RANKED_RUNTIMES)
    result = make_ranking("c", "a", "b", "e", "d")

    assert top_k_error(case, result, k=1) == runtime_regret(case, result)


def test_top_k_error_rejects_non_positive_k() -> None:
    case = make_ranked_case(RANKED_RUNTIMES)

    with pytest.raises(ValueError, match="k must be a positive integer"):
        top_k_error(case, make_ranking("a", "b", "c", "d", "e"), k=0)


def test_kendall_tau_is_one_for_perfect_and_minus_one_for_reversed_ranking() -> None:
    case = make_ranked_case(RANKED_RUNTIMES)

    assert kendall_tau(case, make_ranking("d", "b", "c", "a", "e")) == 1.0
    assert kendall_tau(case, make_ranking("e", "a", "c", "b", "d")) == -1.0


def test_kendall_tau_counts_concordant_and_discordant_pairs() -> None:
    case = make_ranked_case({"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0})
    result = make_ranking("b", "a", "c", "d")

    # One discordant pair (b, a) and five concordant pairs out of six.
    assert kendall_tau(case, result) == pytest.approx(4 / 6)


def test_kendall_tau_b_corrects_for_tied_runtimes() -> None:
    case = make_ranked_case({"a": 1.0, "b": 1.0, "c": 2.0})
    result = make_ranking("a", "b", "c")

    # Two concordant pairs, one tied runtime pair: 2 / sqrt(3 * 2).
    assert kendall_tau(case, result) == pytest.approx(2 / sqrt(6))


def test_kendall_tau_is_undefined_without_comparable_pairs() -> None:
    with pytest.raises(ValueError, match="undefined"):
        kendall_tau(make_ranked_case({"a": 1.0}), make_ranking("a"))
    with pytest.raises(ValueError, match="undefined"):
        kendall_tau(make_ranked_case({"a": 1.0, "b": 1.0}), make_ranking("a", "b"))


def test_benchmark_metrics_are_resolvable_by_name() -> None:
    names = ("top_1_error", "top_5_error", "top_10_error", "kendall_tau")
    case = make_ranked_case(RANKED_RUNTIMES)
    result = make_ranking("a", "c", "b", "e", "d")

    metrics = resolve_metrics(names)

    assert tuple(metrics) == names
    assert metrics["top_1_error"](case, result) == pytest.approx(3.0)
    assert metrics["top_5_error"](case, result) == pytest.approx(0.0)
    assert metrics["kendall_tau"](case, result) == pytest.approx(-0.4)
