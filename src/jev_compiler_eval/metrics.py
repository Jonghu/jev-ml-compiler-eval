"""Shared evaluation metrics."""

from collections.abc import Callable, Iterable

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


_METRICS: dict[str, MetricFunction] = {
    "runtime_regret": runtime_regret,
    "top_1_accuracy": top_1_accuracy,
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
