import pytest

from jev_compiler_eval.metrics import runtime_regret, top_1_accuracy
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
