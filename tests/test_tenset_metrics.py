import pytest

from jev_compiler_eval.metrics import (
    best_of_top_k_score,
    pairwise_accuracy,
    pairwise_accuracy_tie_aware,
    resolve_metrics,
    runtime_regret,
    top_1_score,
    top_5_score,
    top_k_score,
)
from jev_compiler_eval.types import (
    Candidate,
    DecisionResult,
    EvaluationCase,
    RuntimeRecord,
    Workload,
)


def case_with_runtimes(values: tuple[float, ...]) -> EvaluationCase:
    return EvaluationCase(
        Workload("w", {}),
        tuple(Candidate(str(index), {}) for index in range(len(values))),
        tuple(
            RuntimeRecord("w", str(index), value) for index, value in enumerate(values)
        ),
    )


def test_top_k_is_runtime_ratio_and_not_selection_accuracy() -> None:
    case = case_with_runtimes((1, 2, 3, 4, 5, 6))
    result = DecisionResult(("5", "4", "3", "2", "1", "0"), 0)
    assert top_1_score(case, result) == pytest.approx(1 / 6)
    assert top_5_score(case, result) == pytest.approx(
        (1 / 6 + 1 / 5 + 1 / 4 + 1 / 3 + 1 / 2) / 5
    )
    assert best_of_top_k_score(case, result, 5) == 0.5
    assert top_k_score(case, result, 6) == pytest.approx(
        (1 / 6 + 1 / 5 + 1 / 4 + 1 / 3 + 1 / 2 + 1) / 6
    )
    assert top_1_score(case, result) == pytest.approx(
        1 / (runtime_regret(case, result) + 1)
    )


def test_top_five_with_fewer_candidates_uses_entire_set() -> None:
    case = case_with_runtimes((2, 1))
    assert top_5_score(case, DecisionResult(("0", "1"), 0)) == 0.75


@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_invalid_k_is_rejected(k: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        top_k_score(case_with_runtimes((1,)), DecisionResult(("0",), 0), k)


def test_pairwise_accuracy_orders_pairs_and_excludes_measured_ties() -> None:
    case = case_with_runtimes((1, 1, 2, 4))
    assert (
        pairwise_accuracy_tie_aware(case, DecisionResult(("1", "0", "2", "3"), 0)) == 1
    )
    assert (
        pairwise_accuracy_tie_aware(case, DecisionResult(("3", "2", "0", "1"), 0)) == 0
    )
    assert (
        pairwise_accuracy_tie_aware(case, DecisionResult(("0", "2", "1", "3"), 0))
        == 0.8
    )


@pytest.mark.parametrize("field", ["scores", "probabilities"])
def test_prediction_ties_receive_half_credit(field: str) -> None:
    case = case_with_runtimes((1, 2, 3))
    result = DecisionResult(("0", "1", "2"), 0, {field: {"0": 1.0, "1": 1.0, "2": 0.0}})
    assert pairwise_accuracy_tie_aware(case, result) == pytest.approx(2.5 / 3)
    assert pairwise_accuracy(case, result) == pytest.approx(2 / 3)


def test_all_measured_ties_and_single_candidate_have_neutral_accuracy() -> None:
    assert (
        pairwise_accuracy_tie_aware(
            case_with_runtimes((1, 1)), DecisionResult(("0", "1"), 0)
        )
        == 0.5
    )
    assert pairwise_accuracy(case_with_runtimes((1,)), DecisionResult(("0",), 0)) == 0.5


def test_incomplete_rankings_fail_before_computing_score() -> None:
    with pytest.raises(ValueError, match="complete candidate ranking"):
        top_5_score(case_with_runtimes((1, 2)), DecisionResult(("0",), 0))


@pytest.mark.parametrize(
    "scores", [{"0": 1}, {"0": 0.0, "1": 1.0}, {"0": float("nan"), "1": 0}]
)
def test_invalid_or_inconsistent_scores_fail(scores: dict) -> None:
    with pytest.raises(ValueError):
        pairwise_accuracy(
            case_with_runtimes((1, 2)),
            DecisionResult(("0", "1"), 0, {"scores": scores}),
        )


def test_tenset_metric_registration() -> None:
    assert set(
        resolve_metrics(("top_1_score", "top_5_score", "pairwise_accuracy"))
    ) == {
        "top_1_score",
        "top_5_score",
        "pairwise_accuracy",
    }


def test_official_pairwise_metric_preserves_strict_tie_semantics() -> None:
    case = case_with_runtimes((1, 1, 2))
    assert pairwise_accuracy(case, DecisionResult(("0", "1", "2"), 0)) == 2 / 3
    assert pairwise_accuracy(case, DecisionResult(("1", "0", "2"), 0)) == 1


def test_official_metrics_match_numpy_reference_when_available() -> None:
    np = pytest.importorskip("numpy")
    generator = np.random.default_rng(42)
    for _ in range(10):
        runtimes = generator.uniform(0.1, 10, 12)
        predictions = generator.uniform(0, 1, 12)
        labels = 1 / runtimes
        ranking = tuple(str(int(index)) for index in np.argsort(predictions)[::-1])
        result = DecisionResult(
            ranking, 0, {"scores": dict(zip(map(str, range(12)), predictions.tolist()))}
        )
        case = case_with_runtimes(tuple(runtimes.tolist()))
        pair_predictions = (predictions.reshape((-1, 1)) - predictions) > 0
        pair_labels = (labels.reshape((-1, 1)) - labels) > 0
        expected_pairwise = (
            np.triu(
                np.logical_not(np.logical_xor(pair_predictions, pair_labels)), k=1
            ).sum()
            / 66
        )
        expected_peak = np.mean(
            np.maximum.accumulate(labels[np.argsort(predictions)[::-1][:5]])
            / np.max(labels)
        )
        assert pairwise_accuracy(case, result) == pytest.approx(expected_pairwise)
        assert top_5_score(case, result) == pytest.approx(expected_peak)
