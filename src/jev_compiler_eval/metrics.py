"""Shared evaluation metrics."""

from collections.abc import Callable, Iterable
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
