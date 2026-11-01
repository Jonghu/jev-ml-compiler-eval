from collections.abc import Mapping, Sequence

import pytest

from jev_compiler_eval.datasets import (
    BaseDataset,
    get_dataset_class,
    register_dataset,
)
from jev_compiler_eval.models import (
    BaseDecisionModel,
    get_model_class,
    register_model,
)
from jev_compiler_eval.types import Candidate, DecisionResult, EvaluationCase, Workload


class CompleteModel(BaseDecisionModel):
    @property
    def model_type(self) -> str:
        return "complete-test-model"

    def validate_config(self) -> None:
        pass

    def rank(
        self,
        workload: Workload,
        candidates: Sequence[Candidate],
    ) -> DecisionResult:
        return DecisionResult(
            ranked_candidate_ids=tuple(
                candidate.candidate_id for candidate in candidates
            ),
            decision_latency_ms=0.0,
        )


class CompleteDataset(BaseDataset):
    @property
    def dataset_type(self) -> str:
        return "complete-test-dataset"

    def validate_config(self) -> None:
        pass

    def load(self) -> Sequence[EvaluationCase]:
        return ()


class MissingRankModel(BaseDecisionModel):
    @property
    def model_type(self) -> str:
        return "missing-rank"

    def validate_config(self) -> None:
        pass


class MissingLoadDataset(BaseDataset):
    @property
    def dataset_type(self) -> str:
        return "missing-load"

    def validate_config(self) -> None:
        pass


def test_model_interface_requires_rank_implementation() -> None:
    with pytest.raises(TypeError):
        MissingRankModel({})  # type: ignore[abstract]


def test_dataset_interface_requires_load_implementation() -> None:
    with pytest.raises(TypeError):
        MissingLoadDataset({})  # type: ignore[abstract]


def test_model_receives_only_workload_and_candidates() -> None:
    model = CompleteModel({})
    workload = Workload("workload-1", {})
    candidates = (Candidate("candidate-1", {}),)

    result = model.rank(workload, candidates)

    assert result.ranked_candidate_ids == ("candidate-1",)


def test_model_registry_accepts_model_subclasses() -> None:
    register_model("complete-test-model", CompleteModel)

    assert get_model_class("complete-test-model") is CompleteModel


def test_model_registry_rejects_non_model_classes() -> None:
    with pytest.raises(TypeError, match="must inherit from BaseDecisionModel"):
        register_model("invalid-test-model", dict)  # type: ignore[arg-type]


def test_dataset_registry_accepts_dataset_subclasses() -> None:
    register_dataset("complete-test-dataset", CompleteDataset)

    assert get_dataset_class("complete-test-dataset") is CompleteDataset


def test_dataset_registry_rejects_non_dataset_classes() -> None:
    with pytest.raises(TypeError, match="must inherit from BaseDataset"):
        register_dataset("invalid-test-dataset", Mapping)  # type: ignore[arg-type]
