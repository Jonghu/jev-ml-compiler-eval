"""Shared evaluation metrics."""

from collections.abc import Callable, Iterable, Mapping
from itertools import combinations
from math import isfinite
from functools import partial
from math import sqrt

import numpy as np

from jev_compiler_eval.types import DecisionResult, EvaluationCase

MetricFunction = Callable[[EvaluationCase, DecisionResult], float]


def _runtime_by_candidate(case: EvaluationCase) -> dict[str, float]:
    return {runtime.candidate_id: runtime.runtime_ms for runtime in case.runtimes}


def runtime_regret(case: EvaluationCase, result: DecisionResult) -> float:
    """Return selected runtime divided by optimal runtime, minus one."""

    runtimes = _runtime_by_candidate(case)
    selected_runtime = runtimes[result.selected_candidate_id]
    optimal_runtime = min(runtimes.values())
    return selected_runtime / optimal_runtime - 1.0


def top_1_accuracy(case: EvaluationCase, result: DecisionResult) -> float:
    """Return one when the selected candidate ties for minimum runtime."""

    runtimes = _runtime_by_candidate(case)
    selected_runtime = runtimes[result.selected_candidate_id]
    return float(selected_runtime == min(runtimes.values()))


def _validate_complete_ranking(
    case: EvaluationCase, result: DecisionResult
) -> dict[str, float]:
    runtimes = _runtime_by_candidate(case)
    if set(result.ranked_candidate_ids) != set(runtimes):
        raise ValueError("metric requires a complete candidate ranking")
    return runtimes


def top_k_score(case: EvaluationCase, result: DecisionResult, k: int) -> float:
    """Return TenSet's mean prefix-best normalized throughput over top k.

    This is the single-task ``peak_score`` used by the TenSet cost-model
    evaluation (``python/tvm/auto_scheduler/cost_model/metric.py`` in
    https://github.com/tlc-pack/tenset). For each prefix j=1..min(k,n), compute
    oracle_runtime / best_runtime_in_prefix_j, then average those ratios.
    It is not the endpoint best-of-k score or a hit rate. The oracle is
    restricted to this case's candidate set. The shared runner macro-averages
    workloads; TenSet's training script instead weights by candidate count.
    """

    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    runtimes = _validate_complete_ranking(case, result)
    optimal = min(runtimes.values())
    best = float("inf")
    scores = []
    for item in result.ranked_candidate_ids[:k]:
        best = min(best, runtimes[item])
        scores.append(optimal / best)
    return sum(scores) / len(scores)


def best_of_top_k_score(case: EvaluationCase, result: DecisionResult, k: int) -> float:
    """Return endpoint best-of-k ratio; this is not TenSet's average peak score."""

    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    runtimes = _validate_complete_ranking(case, result)
    return min(runtimes.values()) / min(
        runtimes[item] for item in result.ranked_candidate_ids[:k]
    )


def top_1_score(case: EvaluationCase, result: DecisionResult) -> float:
    """Return the TenSet Top-1 performance score in (0, 1]."""

    return top_k_score(case, result, 1)


def top_5_score(case: EvaluationCase, result: DecisionResult) -> float:
    """Return the TenSet Top-5 performance score in (0, 1]."""

    return top_k_score(case, result, 5)


def _prediction_scores(
    result: DecisionResult, candidate_ids: set[str]
) -> Mapping[str, float] | None:
    # Jev exposes probabilities; the TVM baseline exposes higher-is-better scores.
    for key in ("scores", "probabilities"):
        values = result.metadata.get(key)
        if values is None:
            continue
        if not isinstance(values, Mapping) or set(values) != candidate_ids:
            raise ValueError(f"{key} must match all evaluation candidate IDs")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
            for value in values.values()
        ):
            raise ValueError(f"{key} must contain finite numeric scores")
        ordered = [values[item] for item in result.ranked_candidate_ids]
        if any(left < right for left, right in zip(ordered, ordered[1:])):
            raise ValueError(f"ranking must agree with higher-is-better {key}")
        return values
    return None


def pairwise_accuracy(case: EvaluationCase, result: DecisionResult) -> float:
    """Match TenSet's ``metric_pairwise_comp_accuracy`` strict comparisons.

    Compare higher predicted score and lower measured runtime for every pair
    in the original candidate order. Strict comparison treats ties as False,
    including measured ties; therefore tied cases can depend on input order.
    Scores or probabilities preserve model ties; otherwise use the ranking.
    The official single-candidate convention is 0.5.
    """

    runtimes = _validate_complete_ranking(case, result)
    scores = _prediction_scores(result, set(runtimes))
    if scores is None:
        scores = {item: -rank for rank, item in enumerate(result.ranked_candidate_ids)}
    candidate_ids = [candidate.candidate_id for candidate in case.candidates]
    comparable = len(candidate_ids) * (len(candidate_ids) - 1) // 2
    if comparable == 0:
        return 0.5
    correct = sum(
        (scores[left] > scores[right]) == (runtimes[left] < runtimes[right])
        for left, right in combinations(candidate_ids, 2)
    )
    return correct / comparable


def pairwise_accuracy_tie_aware(case: EvaluationCase, result: DecisionResult) -> float:
    """Return a separately named, order-independent pairwise diagnostic.

    Predicted ties receive half credit when model scores or probabilities are
    supplied. Otherwise the returned ranking is the prediction. Measured ties
    are excluded; a workload with no comparable pairs returns neutral 0.5.
    These tie conventions are explicit project extensions of pairwise accuracy.
    """

    runtimes = _validate_complete_ranking(case, result)
    scores = _prediction_scores(result, set(runtimes))
    positions = {item: rank for rank, item in enumerate(result.ranked_candidate_ids)}
    correct = 0.0
    comparable = 0
    for left, right in combinations(runtimes, 2):
        if runtimes[left] == runtimes[right]:
            continue
        comparable += 1
        if scores is not None and scores[left] == scores[right]:
            correct += 0.5
        elif (positions[left] < positions[right]) == (runtimes[left] < runtimes[right]):
            correct += 1
    return correct / comparable if comparable else 0.5


def top_k_error(case: EvaluationCase, result: DecisionResult, *, k: int) -> float:
    """Return the TpuGraphs Top-K error for one workload.

    The error is the fastest runtime among the ``k`` highest-ranked candidates
    divided by the optimal runtime, minus one. When a workload has fewer than
    ``k`` candidates, every candidate is considered and the error is zero.
    """

    if k < 1:
        raise ValueError("k must be a positive integer")
    runtimes = _runtime_by_candidate(case)
    best_in_top_k = min(
        runtimes[candidate_id] for candidate_id in result.ranked_candidate_ids[:k]
    )
    return best_in_top_k / min(runtimes.values()) - 1.0


def kendall_tau(case: EvaluationCase, result: DecisionResult) -> float:
    """Return Kendall's tau-b between the predicted ranking and true runtimes.

    Ranking positions and runtimes are compared so that a ranking ordered from
    fastest to slowest scores one. Tau-b accounts for tied runtimes; rankings
    never contain ties.
    """

    runtimes = _runtime_by_candidate(case)
    ranked_runtimes = np.array(
        [runtimes[candidate_id] for candidate_id in result.ranked_candidate_ids],
        dtype=np.float64,
    )
    count = len(ranked_runtimes)
    total_pairs = count * (count - 1) // 2

    score = 0
    tied_runtime_pairs = 0
    for index in range(count - 1):
        differences = np.sign(ranked_runtimes[index + 1 :] - ranked_runtimes[index])
        score += int(differences.sum())
        tied_runtime_pairs += int(np.count_nonzero(differences == 0))

    denominator = sqrt(total_pairs * (total_pairs - tied_runtime_pairs))
    if denominator == 0:
        raise ValueError(
            "Kendall's tau is undefined for fewer than two candidates "
            "or identical runtimes"
        )
    return score / denominator


_METRICS: dict[str, MetricFunction] = {
    "runtime_regret": runtime_regret,
    "top_1_accuracy": top_1_accuracy,
    "top_1_score": top_1_score,
    "top_5_score": top_5_score,
    "pairwise_accuracy": pairwise_accuracy,
    "pairwise_accuracy_tie_aware": pairwise_accuracy_tie_aware,
    "top_1_error": partial(top_k_error, k=1),
    "top_5_error": partial(top_k_error, k=5),
    "top_10_error": partial(top_k_error, k=10),
    "kendall_tau": kendall_tau,
}


def resolve_metrics(names: Iterable[str]) -> dict[str, MetricFunction]:
    """Resolve configured metric names to implementations."""

    resolved: dict[str, MetricFunction] = {}
    for name in names:
        try:
            resolved[name] = _METRICS[name]
        except KeyError as error:
            raise ValueError(f'unknown metric "{name}"') from error
    return resolved
