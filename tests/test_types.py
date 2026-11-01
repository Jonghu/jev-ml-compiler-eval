from dataclasses import fields

import pytest

from jev_compiler_eval.types import (
    Candidate,
    DecisionResult,
    EvaluationCase,
    RuntimeRecord,
    Workload,
)


def make_evaluation_case() -> EvaluationCase:
    return EvaluationCase(
        workload=Workload("workload-1", {"operation": "matmul"}),
        candidates=(
            Candidate("candidate-1", {"tile": 8}),
            Candidate("candidate-2", {"tile": 16}),
        ),
        runtimes=(
            RuntimeRecord("workload-1", "candidate-1", 2.0),
            RuntimeRecord("workload-1", "candidate-2", 1.0),
        ),
    )


def test_evaluation_case_keeps_runtimes_separate_from_model_inputs() -> None:
    evaluation_case = make_evaluation_case()

    assert {field.name for field in fields(Workload)} == {"workload_id", "context"}
    assert {field.name for field in fields(Candidate)} == {
        "candidate_id",
        "configuration",
    }
    assert evaluation_case.runtimes[0].runtime_ms == 2.0


def test_evaluation_case_requires_matching_runtime_records() -> None:
    with pytest.raises(
        ValueError, match="runtimes must match the evaluation candidates exactly"
    ):
        EvaluationCase(
            workload=Workload("workload-1", {}),
            candidates=(Candidate("candidate-1", {}),),
            runtimes=(),
        )


def test_evaluation_case_rejects_duplicate_candidate_ids() -> None:
    with pytest.raises(ValueError, match="candidate IDs must be unique"):
        EvaluationCase(
            workload=Workload("workload-1", {}),
            candidates=(Candidate("candidate-1", {}), Candidate("candidate-1", {})),
            runtimes=(RuntimeRecord("workload-1", "candidate-1", 1.0),),
        )


def test_runtime_must_be_positive() -> None:
    with pytest.raises(ValueError, match="runtime_ms must be a finite positive number"):
        RuntimeRecord("workload-1", "candidate-1", 0.0)


def test_decision_result_exposes_top_ranked_candidate() -> None:
    result = DecisionResult(
        ranked_candidate_ids=("candidate-2", "candidate-1"),
        decision_latency_ms=1.5,
    )

    assert result.selected_candidate_id == "candidate-2"


def test_decision_result_rejects_duplicate_candidate_ids() -> None:
    with pytest.raises(ValueError, match="ranked candidate IDs must be unique"):
        DecisionResult(
            ranked_candidate_ids=("candidate-1", "candidate-1"),
            decision_latency_ms=1.5,
        )
