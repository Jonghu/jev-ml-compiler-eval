"""Explain a TpuGraphs tile result bundle: which picks drive the Top-1 error.

The analysis reads a run's ``predictions.jsonl``, the tile:xla kernels the run
used, and, for Jev tournament runs, the cached Jev requests and answers. It
writes per-kernel tables and plots that show how individual picks contribute
to the mean Top-1 error and why the tournament made them.
"""

import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from jev_compiler_eval.datasets.tpugraphs import TpuGraphsTileDataset
from jev_compiler_eval.types import EvaluationCase

DEFAULT_CANDIDATE_ID = "config_0"
_MAX_CANDIDATES = 1_000_000
# Top-1 slowdown buckets as (label, lower bound inclusive, upper bound exclusive).
SLOWDOWN_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("< 1.1×", 0.0, 1.1),
    ("1.1–2×", 1.1, 2.0),
    ("2–10×", 2.0, 10.0),
    ("≥ 10×", 10.0, float("inf")),
)


@dataclass(frozen=True)
class QuestionTrace:
    """One Choice question of a tournament, mapped back to candidate IDs."""

    round_index: int
    question_id: str
    candidate_ids: tuple[str, ...]
    order: tuple[str, ...]
    probabilities: Mapping[str, float]


def read_predictions(path: Path) -> dict[str, dict[str, dict]]:
    """Return predictions keyed by model ID, then workload ID."""

    predictions: dict[str, dict[str, dict]] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                record = json.loads(line)
                predictions.setdefault(record["model_id"], {})[
                    record["workload_id"]
                ] = record
    return predictions


def program_name(workload_id: str) -> str:
    """Return the program name without the per-kernel hash suffix."""

    program, separator, _ = workload_id.rpartition("_")
    return program if separator else workload_id


def contribution_curve(errors: Iterable[float]) -> tuple[list[float], list[float]]:
    """Return the kernel share and cumulative error share, worst kernels first."""

    ordered = sorted(errors, reverse=True)
    total = sum(ordered)
    if not ordered or total <= 0:
        return [], []
    shares: list[float] = []
    cumulative: list[float] = []
    running = 0.0
    for index, error in enumerate(ordered, start=1):
        running += error
        shares.append(index / len(ordered))
        cumulative.append(running / total)
    return shares, cumulative


def slowdown_buckets(errors: Iterable[float]) -> list[tuple[str, int, float]]:
    """Return each slowdown bucket's kernel count and share of the total error."""

    values = list(errors)
    total = sum(values)
    buckets: list[tuple[str, int, float]] = []
    for label, lower, upper in SLOWDOWN_BUCKETS:
        members = [error for error in values if lower <= 1 + error < upper]
        share = sum(members) / total if total > 0 else 0.0
        buckets.append((label, len(members), share))
    return buckets


def opaque_aliases(presented_ids: Sequence[str]) -> dict[str, str]:
    """Map the tournament's opaque aliases back to the presented candidate IDs."""

    width = len(str(len(presented_ids)))
    return {
        f"c{index + 1:0{width}d}": candidate_id
        for index, candidate_id in enumerate(presented_ids)
    }


def _question_order(
    answer: Mapping[str, object],
    aliases: Sequence[str],
    presented_index: Mapping[str, int],
) -> tuple[str, ...]:
    """Order a question's aliases as the tournament does: choice, then probability."""

    choice = answer["choice"]
    probabilities = answer["probabilities"]
    assert isinstance(choice, str) and isinstance(probabilities, Mapping)
    rest = sorted(
        (alias for alias in aliases if alias != choice),
        key=lambda alias: (-float(probabilities[alias]), presented_index[alias]),
    )
    return (choice, *rest)


def index_cache(cache_dir: Path, request_ids: Iterable[str]) -> dict[str, Path]:
    """Find the cache file of each wanted request ID."""

    wanted = set(request_ids)
    found: dict[str, Path] = {}
    for path in cache_dir.glob("*.json"):
        if not wanted - set(found):
            break
        text = path.read_text(encoding="utf-8")
        for request_id in wanted - set(found):
            if f'"{request_id}"' in text:
                found[request_id] = path
    return found


def trace_tournament(
    record: Mapping, cache_files: Mapping[str, Path]
) -> list[QuestionTrace]:
    """Rebuild every question of one kernel's tournament from the Jev cache."""

    metadata = record["metadata"]
    presented = list(metadata["presented_candidate_ids"])
    to_original = opaque_aliases(presented)
    presented_index = {alias: index for index, alias in enumerate(to_original)}
    traces: list[QuestionTrace] = []
    for request_id in metadata["request_ids"]:
        cached = json.loads(cache_files[request_id].read_text(encoding="utf-8"))
        questions = cached["request"]["questions"]
        answers = cached["response"]["answers"]
        for question_id, question in questions.items():
            aliases = list(question["criteria"])
            answer = answers[question_id]
            order = _question_order(answer, aliases, presented_index)
            traces.append(
                QuestionTrace(
                    round_index=int(question_id.split("_")[0].removeprefix("r")),
                    question_id=question_id,
                    candidate_ids=tuple(to_original[alias] for alias in aliases),
                    order=tuple(to_original[alias] for alias in order),
                    probabilities={
                        to_original[alias]: float(value)
                        for alias, value in answer["probabilities"].items()
                    },
                )
            )
    return traces


def optimum_fate(
    traces: Sequence[QuestionTrace], optimal_ids: Sequence[str], advance: int
) -> str:
    """Describe where the best candidate left the tournament."""

    if not traces:
        return "unknown"
    final_round = max(trace.round_index for trace in traces)
    best_position: tuple[int, int, int, float] | None = None
    for trace in traces:
        for candidate_id in optimal_ids:
            if candidate_id in trace.order:
                position = trace.order.index(candidate_id)
                key = (
                    trace.round_index,
                    -position,
                    len(trace.order),
                    trace.probabilities[candidate_id],
                )
                if best_position is None or key[:2] > best_position[:2]:
                    best_position = key
    if best_position is None:
        return "unknown"
    round_index, negative_position, size, probability = best_position
    position = -negative_position + 1
    if round_index == final_round:
        if position == 1:
            return "picked"
        return f"final, rank {position} of {size} (p={probability:.2f})"
    if position <= advance:
        return f"advanced from round {round_index}"
    return (
        f"eliminated in round {round_index}, rank {position} of {size} "
        f"(p={probability:.2f})"
    )


def _load_cases(dataset_path: Path) -> dict[str, EvaluationCase]:
    dataset = TpuGraphsTileDataset(
        {"path": str(dataset_path), "max_candidates": _MAX_CANDIDATES}
    )
    return {case.workload.workload_id: case for case in dataset.load()}


def _bounds(case: EvaluationCase, candidate_id: str) -> str:
    for candidate in case.candidates:
        if candidate.candidate_id == candidate_id:
            return json.dumps(candidate.configuration, separators=(",", ":"))
    return ""


def _write_csv(path: Path, rows: Sequence[Mapping], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze_tpugraphs(
    predictions_path: Path,
    dataset_path: Path,
    output_dir: Path,
    *,
    jev_model_id: str,
    random_model_id: str,
    cache_dir: Path | None,
    worst_count: int = 15,
    trace_count: int = 3,
    advance: int = 5,
    create_plots: bool = True,
) -> Path:
    """Write per-kernel tables and plots for one TpuGraphs result bundle."""

    predictions = read_predictions(predictions_path)
    jev = predictions[jev_model_id]
    random = predictions[random_model_id]
    cases = _load_cases(dataset_path)
    missing = set(jev) - set(cases)
    if missing:
        raise ValueError(f"{len(missing)} predicted kernels are not in the dataset")

    rows: list[dict] = []
    for workload_id, record in jev.items():
        case = cases[workload_id]
        runtimes = {r.candidate_id: r.runtime_ms for r in case.runtimes}
        optimum = min(runtimes.values())
        rows.append(
            {
                "workload_id": workload_id,
                "program": program_name(workload_id),
                "candidates": len(runtimes),
                "jev_top_1_error": record["metrics"]["top_1_error"],
                "random_top_1_error": random[workload_id]["metrics"]["top_1_error"],
                "default_top_1_error": runtimes[DEFAULT_CANDIDATE_ID] / optimum - 1,
                "jev_pick": record["selected_candidate_id"],
                "jev_pick_bounds": _bounds(case, record["selected_candidate_id"]),
                "optimal": record["optimal_candidate_ids"][0],
                "optimal_bounds": _bounds(case, record["optimal_candidate_ids"][0]),
                "rounds": len(record["metadata"].get("rounds", [])),
                "final_confidence": record["metadata"].get("final_confidence"),
            }
        )
    rows.sort(key=lambda row: row["jev_top_1_error"], reverse=True)

    traces: dict[str, list[QuestionTrace]] = {}
    if cache_dir is not None:
        two_round = [row for row in rows if row["rounds"] > 1]
        request_ids = [
            request_id
            for row in two_round
            for request_id in jev[row["workload_id"]]["metadata"]["request_ids"]
        ]
        cache_files = index_cache(cache_dir, request_ids)
        for row in rows:
            record = jev[row["workload_id"]]
            if all(r in cache_files for r in record["metadata"]["request_ids"]):
                traces[row["workload_id"]] = trace_tournament(record, cache_files)
                row["optimum_fate"] = optimum_fate(
                    traces[row["workload_id"]],
                    record["optimal_candidate_ids"],
                    advance,
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "workload_id",
        "program",
        "candidates",
        "rounds",
        "jev_top_1_error",
        "random_top_1_error",
        "default_top_1_error",
        "final_confidence",
        "jev_pick",
        "jev_pick_bounds",
        "optimal",
        "optimal_bounds",
        "optimum_fate",
    ]
    _write_csv(output_dir / "per_kernel.csv", rows, fields)
    _write_csv(output_dir / "worst_kernels.csv", rows[:worst_count], fields)

    if create_plots:
        _plot_all(
            output_dir,
            rows,
            cases,
            jev,
            random,
            traces,
            worst_count=worst_count,
            trace_count=trace_count,
            advance=advance,
        )
    return output_dir


def _plot_all(
    output_dir: Path,
    rows: Sequence[dict],
    cases: Mapping[str, EvaluationCase],
    jev: Mapping[str, dict],
    random: Mapping[str, dict],
    traces: Mapping[str, list[QuestionTrace]],
    *,
    worst_count: int,
    trace_count: int,
    advance: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    jev_color, random_color, default_color, optimum_color = (
        "#d62728",
        "#7f7f7f",
        "#1f77b4",
        "#2ca02c",
    )

    # 1. Cumulative contribution of kernels to the total Top-1 error.
    figure, axis = plt.subplots(figsize=(7, 4.5))
    for key, label, color in (
        ("jev_top_1_error", "Jev", jev_color),
        ("random_top_1_error", "Random", random_color),
    ):
        shares, cumulative = contribution_curve(row[key] for row in rows)
        axis.plot(
            [0, *[100 * s for s in shares]],
            [0, *[100 * c for c in cumulative]],
            color=color,
            label=label,
        )
    shares, cumulative = contribution_curve(row["jev_top_1_error"] for row in rows)
    top = max(1, round(0.05 * len(rows)))
    axis.axvline(100 * shares[top - 1], color="black", linestyle=":", linewidth=1)
    axis.annotate(
        f"worst 5% of kernels = {100 * cumulative[top - 1]:.0f}% of Jev's error",
        (100 * shares[top - 1], 100 * cumulative[top - 1]),
        xytext=(15, 25),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->"},
    )
    axis.set_xlabel("Kernels, worst first (%)")
    axis.set_ylabel("Share of total Top-1 error (%)")
    axis.set_title("Which kernels drive the mean Top-1 error")
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_dir / "top1_error_contribution.png", dpi=200)
    plt.close(figure)

    # 2. Per-kernel slowdown, Jev versus Random.
    programs = sorted({row["program"] for row in rows})
    colors = plt.get_cmap("tab10")
    figure, axis = plt.subplots(figsize=(7, 6))
    for index, program in enumerate(programs):
        members = [row for row in rows if row["program"] == program]
        axis.scatter(
            [1 + row["random_top_1_error"] for row in members],
            [1 + row["jev_top_1_error"] for row in members],
            s=[8 + row["candidates"] / 60 for row in members],
            color=colors(index),
            alpha=0.6,
            label=program,
            edgecolors="none",
        )
    limit = max(
        1 + max(row["jev_top_1_error"], row["random_top_1_error"]) for row in rows
    )
    axis.plot([1, limit], [1, limit], color="black", linewidth=1)
    axis.axhline(10, color=jev_color, linestyle=":", linewidth=1)
    wins = sum(row["jev_top_1_error"] < row["random_top_1_error"] for row in rows)
    losses = sum(row["jev_top_1_error"] > row["random_top_1_error"] for row in rows)
    axis.text(0.97, 0.03, f"Jev better: {wins}", transform=axis.transAxes, ha="right")
    axis.text(
        0.03, 0.97, f"Random better: {losses}", transform=axis.transAxes, va="top"
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Random slowdown (×, log)")
    axis.set_ylabel("Jev slowdown (×, log)")
    axis.set_title("Top-1 slowdown per kernel (marker size = candidates)")
    axis.legend(fontsize=7, loc="lower right", bbox_to_anchor=(1.0, 0.08))
    axis.grid(alpha=0.3, which="both")
    figure.tight_layout()
    figure.savefig(output_dir / "jev_vs_random_scatter.png", dpi=200)
    plt.close(figure)

    # 3. Every candidate of the worst kernels, with the picks marked.
    worst = rows[:worst_count]
    figure, axis = plt.subplots(figsize=(10, 0.45 * len(worst) + 1.5))
    for y, row in enumerate(worst):
        case = cases[row["workload_id"]]
        runtimes = {r.candidate_id: r.runtime_ms for r in case.runtimes}
        optimum = min(runtimes.values())
        axis.scatter(
            [runtime / optimum for runtime in runtimes.values()],
            [y] * len(runtimes),
            s=4,
            color="#bbbbbb",
            zorder=1,
        )
        random_pick = random[row["workload_id"]]["selected_candidate_id"]
        for candidate_id, color, marker, label in (
            (DEFAULT_CANDIDATE_ID, default_color, "s", "Compiler default"),
            (random_pick, random_color, "x", "Random pick"),
            (row["jev_pick"], jev_color, "o", "Jev pick"),
        ):
            axis.scatter(
                runtimes[candidate_id] / optimum,
                y,
                s=45,
                color=color,
                marker=marker,
                zorder=3,
                label=label if y == 0 else None,
            )
    axis.axvline(1, color=optimum_color, linewidth=1, label="Optimum")
    axis.set_yticks(
        range(len(worst)),
        [
            f"{row['program'][:28]} …{row['workload_id'][-6:]} ({row['candidates']})"
            for row in worst
        ],
        fontsize=7,
    )
    axis.invert_yaxis()
    axis.set_xscale("log")
    axis.set_xlabel("Runtime / optimal runtime (×, log)")
    axis.set_title(f"All candidates of Jev's {len(worst)} worst kernels")
    axis.legend(fontsize=8, loc="lower right")
    axis.grid(alpha=0.3, axis="x", which="both")
    figure.tight_layout()
    figure.savefig(output_dir / "worst_kernels.png", dpi=200)
    plt.close(figure)

    # 5. Kernel counts per slowdown bucket, with each bucket's error share.
    series = [
        ("Jev", jev_color, slowdown_buckets(row["jev_top_1_error"] for row in rows)),
        (
            "Random",
            random_color,
            slowdown_buckets(row["random_top_1_error"] for row in rows),
        ),
    ]
    width = 0.38
    figure, axis = plt.subplots(figsize=(8, 4.8))
    for offset, (label, color, buckets) in zip((-width / 2, width / 2), series):
        positions = [index + offset for index in range(len(buckets))]
        bars = axis.bar(
            positions,
            [count for _, count, _ in buckets],
            width,
            color=color,
            label=label,
        )
        for bar, (_, count, share) in zip(bars, buckets):
            center = bar.get_x() + bar.get_width() / 2
            axis.annotate(
                str(count),
                (center, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
            axis.text(
                center,
                bar.get_height() / 2,
                f"{100 * share:.0f}%\nof\nerror",
                ha="center",
                va="center",
                fontsize=8,
                color="white",
            )
    axis.set_xticks(range(len(SLOWDOWN_BUCKETS)), [b[0] for b in SLOWDOWN_BUCKETS])
    axis.set_xlabel("Top-1 slowdown (picked runtime / optimal runtime)")
    axis.set_ylabel("Kernels")
    axis.set_title(f"Top-1 slowdown per kernel ({len(rows)} kernels)")
    axis.set_ylim(0, 1.25 * max(count for *_, b in series for _, count, _ in b))
    axis.legend()
    axis.grid(alpha=0.3, axis="y")
    figure.tight_layout()
    figure.savefig(output_dir / "slowdown_buckets.png", dpi=200)
    plt.close(figure)

    # 4. Tournament traces of the worst two-round kernels.
    traced = [row for row in rows if row["workload_id"] in traces and row["rounds"] > 1]
    for rank, row in enumerate(traced[:trace_count], start=1):
        _plot_trace(
            plt,
            output_dir / f"tournament_trace_{rank}.png",
            row,
            cases[row["workload_id"]],
            traces[row["workload_id"]],
            jev[row["workload_id"]],
            advance=advance,
            colors=(jev_color, optimum_color, default_color),
        )


def _plot_trace(
    plt,
    path: Path,
    row: Mapping,
    case: EvaluationCase,
    traces: Sequence[QuestionTrace],
    record: Mapping,
    *,
    advance: int,
    colors: tuple[str, str, str],
) -> None:
    jev_color, optimum_color, default_color = colors
    runtimes = {r.candidate_id: r.runtime_ms for r in case.runtimes}
    optimum = min(runtimes.values())
    optimal_ids = set(record["optimal_candidate_ids"])
    final_round = max(trace.round_index for trace in traces)

    figure, (first, final) = plt.subplots(1, 2, figsize=(12, 4.8))
    for axis, round_index in ((first, 0), (final, final_round)):
        for trace in (t for t in traces if t.round_index == round_index):
            advanced = set(trace.order[:advance])
            for candidate_id in trace.candidate_ids:
                slowdown = runtimes[candidate_id] / optimum
                probability = trace.probabilities[candidate_id]
                if candidate_id == trace.order[0] and round_index == final_round:
                    style = {"color": jev_color, "s": 80, "marker": "o", "zorder": 4}
                elif candidate_id in optimal_ids:
                    style = {
                        "color": optimum_color,
                        "s": 160,
                        "marker": "*",
                        "zorder": 5,
                    }
                elif candidate_id == DEFAULT_CANDIDATE_ID:
                    style = {
                        "color": default_color,
                        "s": 50,
                        "marker": "s",
                        "zorder": 3,
                    }
                elif round_index < final_round and candidate_id in advanced:
                    style = {"color": "#ff7f0e", "s": 20, "marker": "o", "zorder": 2}
                else:
                    style = {"color": "#bbbbbb", "s": 8, "marker": "o", "zorder": 1}
                axis.scatter(probability, slowdown, **style)
        axis.set_yscale("log")
        axis.set_xlabel("Jev probability within its question")
        axis.set_ylabel("Runtime / optimal runtime (×, log)")
        axis.grid(alpha=0.3, which="both")
    questions = sum(1 for t in traces if t.round_index == 0)
    first.set_title(
        f"Round 0: {row['candidates']} candidates in {questions} questions\n"
        f"orange = top {advance} advanced"
    )
    finalists = sum(
        len(t.candidate_ids) for t in traces if t.round_index == final_round
    )
    final.set_title(
        f"Final: {finalists} finalists\n"
        f"red = Jev pick ({1 + row['jev_top_1_error']:.1f}× slower)"
    )
    for label, color, marker in (
        ("Optimum", optimum_color, "*"),
        ("Compiler default", default_color, "s"),
        ("Jev pick", jev_color, "o"),
    ):
        final.scatter([], [], color=color, marker=marker, label=label)
    final.legend(fontsize=8)
    figure.suptitle(
        f"{row['workload_id']}\noptimum: {row.get('optimum_fate', 'unknown')}",
        fontsize=9,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)
