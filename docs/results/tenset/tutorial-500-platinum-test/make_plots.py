"""Regenerate reviewed Jev-versus-Random figures from the adjacent exports.

Run from the repository root with:
    uv run --extra tenset python \
        docs/results/tenset/tutorial-500-platinum-test/make_plots.py

No raw measurements, credentials, models or API calls are used.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
COLORS = {"random": "#738293", "jev": "#087f8c", "expected": "#b26d22"}
INK, GRID = "#142b43", "#e1e7ec"


def read_json(name):
    return json.loads((ROOT / name).read_text())


def read_csv(name):
    with (ROOT / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def save(figure, name):
    for extension in ("png", "svg"):
        path = ROOT / f"{name}.{extension}"
        figure.savefig(path, dpi=220, facecolor="white")
        if extension == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines())
                + "\n"
            )
    plt.close(figure)


def style_axis(axis):
    axis.set_axisbelow(True)
    axis.grid(axis="x", color=GRID, linewidth=0.7)
    axis.tick_params(length=0)
    for spine in ("top", "right", "left"):
        axis.spines[spine].set_visible(False)
    axis.spines["bottom"].set_color(GRID)


def plot_comparison():
    metrics = {row["model_type"]: row for row in read_csv("metrics.csv")}
    paired = read_json("paired-statistics.json")
    exact = read_json("exact-random-expectation.json")
    figure, axes = plt.subplots(2, 2, figsize=(12, 7.3))
    figure.subplots_adjust(
        left=0.20, right=0.97, top=0.84, bottom=0.26, hspace=0.66, wspace=0.52
    )
    figure.suptitle(
        "TenSet Platinum: schedule-selection scores and paired differences\n"
        "484 workloads · 187,679 test records · record-weighted means",
        fontsize=16,
        fontweight="bold",
        color=INK,
        y=0.98,
    )
    figure.text(0.015, 0.87, "A", fontsize=14, fontweight="bold", color=INK)
    figure.text(0.015, 0.53, "B", fontsize=14, fontweight="bold", color=INK)
    for column, (key, title) in enumerate(
        (("top_1_score", "Top-1 Score"), ("top_5_score", "Top-5 Score"))
    ):
        axis = axes[0, column]
        values = [
            float(metrics["random"][f"weighted_{key}"]),
            float(metrics["jev-tournament"][f"weighted_{key}"]),
            exact["models"]["expected_random"]["weighted"][key],
        ]
        y = [2, 1, 0]
        bars = axis.barh(
            y,
            values,
            height=0.54,
            color=[COLORS["random"], COLORS["jev"], "white"],
            edgecolor=[COLORS["random"], COLORS["jev"], COLORS["expected"]],
            linewidth=1.4,
        )
        bars[2].set_hatch("//")
        axis.set_yticks(
            y,
            [
                "Observed Random-0",
                "Jev tournament",
                "Exact uniform\nRandom expectation",
            ],
        )
        axis.set_xlim(0, 1)
        axis.set_xticks(np.arange(0, 1.01, 0.2))
        axis.set_title(title, loc="left", fontweight="bold", color=INK, pad=14)
        axis.set_xlabel("Score · higher is better")
        for position, value in zip(y, values, strict=True):
            axis.text(
                value + 0.022,
                position,
                f"{value:.4f}",
                va="center",
                color=INK,
                fontweight="bold",
            )
        style_axis(axis)

        axis = axes[1, column]
        observed = paired["paired_comparisons"][key]
        expected = exact["comparisons"]["jev_minus_expected_random"][key]
        deltas = [
            observed["weighted_mean_delta_jev_minus_random"],
            expected["weighted_mean_delta"],
        ]
        intervals = [
            observed["paired_workload_bootstrap_weighted_delta_95ci"],
            expected["weighted_bootstrap_95ci"],
        ]
        for position, value, interval, color in zip(
            [1, 0], deltas, intervals, [COLORS["jev"], COLORS["expected"]], strict=True
        ):
            axis.errorbar(
                value,
                position,
                xerr=[[value - interval[0]], [interval[1] - value]],
                fmt="o",
                markersize=6.5,
                capsize=4,
                color=color,
                linewidth=1.8,
            )
            axis.text(
                0.095,
                position,
                f"{value:+.4f}",
                va="center",
                ha="right",
                color=INK,
                fontsize=10,
            )
        axis.axvline(0, color="#61717f", linestyle="--", linewidth=1)
        axis.set_yticks([1, 0], ["Jev − Random-0", "Jev − expected\nuniform Random"])
        axis.set_ylim(-0.55, 1.55)
        axis.set_xlim(-0.03, 0.10)
        axis.set_xticks([-0.02, 0, 0.02, 0.04, 0.06, 0.08, 0.10])
        axis.set_title(
            f"{title}: paired difference", loc="left", color=INK, fontsize=12, pad=10
        )
        axis.set_xlabel("Jev minus reference · positive favors Jev")
        style_axis(axis)
    figure.text(
        0.02,
        0.055,
        "A: uploaded official-label weighted scores; expected Random is a separate "
        "analytical reference, not another run.\n"
        "B: exploratory percentile 95% intervals from 50,000 paired workload "
        "resamples; record weights retained.\n"
        "Runtime-map recomputation agrees with the uploaded summary within "
        "5.3e−10. Family dependence is not modeled.",
        fontsize=9.5,
        color="#536576",
        linespacing=1.6,
    )
    save(figure, "comparison")


def plot_candidate_counts():
    rows = read_csv("by-candidate-count.csv")
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 5.0))
    figure.subplots_adjust(left=0.075, right=0.97, top=0.79, bottom=0.25, wspace=0.20)
    figure.suptitle(
        "Natural candidate-count groups: descriptive comparison\n"
        "Workload composition differs between groups",
        fontsize=15,
        color=INK,
        fontweight="bold",
        y=0.98,
    )
    groups = ["6-88", "400"]
    for axis, key, title in zip(
        axes,
        ("weighted_top_1_score", "weighted_top_5_score"),
        ("Top-1 Score", "Top-5 Score"),
        strict=True,
    ):
        x = np.arange(len(groups))
        for model, offset, label in (
            ("random", -0.18, "Random-0"),
            ("jev", 0.18, "Jev"),
        ):
            values = [
                float(
                    next(
                        row[key]
                        for row in rows
                        if row["candidate_count_group"] == group
                        and row["model"] == model
                    )
                )
                for group in groups
            ]
            bars = axis.bar(
                x + offset, values, width=0.34, color=COLORS[model], label=label
            )
            for bar, value in zip(bars, values, strict=True):
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.025,
                    f"{value:.3f}",
                    ha="center",
                    fontsize=10,
                    color=INK,
                )
        axis.set_ylim(0, 1)
        axis.set_xticks(
            x,
            [
                "6–88 candidates\n16 workloads · 479 records",
                "400 candidates\n468 workloads · 187,200 records",
            ],
        )
        axis.set_title(title, loc="left", color=INK, fontweight="bold", pad=10)
        axis.set_axisbelow(True)
        axis.grid(axis="y", color=GRID, linewidth=0.7)
        axis.tick_params(length=0)
        for spine in ("top", "right", "left"):
            axis.spines[spine].set_visible(False)
        axis.spines["bottom"].set_color(GRID)
    axes[0].set_ylabel("Record-weighted score · higher is better")
    axes[1].legend(frameon=False, loc="upper right")
    figure.text(
        0.025,
        0.06,
        "These are observed groups, not a controlled size ablation. Candidate count "
        "is confounded with workload identity;\n"
        "the sample is highly unbalanced. Scores are recomputed from saved runtime "
        "maps, retaining failure-sentinel records.",
        fontsize=9.5,
        color="#536576",
        linespacing=1.6,
    )
    save(figure, "candidate-count-descriptive")


def plot_tournament_attrition():
    diagnostics = read_json("tournament-diagnostics.json")
    values = diagnostics["by_round_count"]["2"]
    count = values["workloads"]
    outcomes = [
        values["all_optima_eliminated_before_final"],
        values["optimum_reached_final_but_not_selected"],
        values["selected_test_pool_optimum"],
    ]
    figure, axis = plt.subplots(figsize=(10.6, 4.8))
    figure.subplots_adjust(left=0.35, right=0.94, top=0.72, bottom=0.24)
    figure.suptitle(
        "Where the test-pool optimum was lost\n"
        f"{outcomes[0]} of {count} two-round workloads "
        f"({100 * outcomes[0] / count:.1f}%) lost it before the final",
        fontsize=15,
        color=INK,
        fontweight="bold",
        y=0.97,
    )
    y = [2, 1, 0]
    axis.barh(
        y, outcomes, height=0.56, color=["#c0784b", COLORS["random"], COLORS["jev"]]
    )
    axis.set_yticks(
        y,
        [
            "Eliminated in round 0",
            "Advanced, but not selected",
            "Selected in the final",
        ],
    )
    axis.set_xlim(0, count * 1.12)
    axis.set_xticks([0, 100, 200, 300, 400])
    axis.set_xlabel("Workloads")
    for position, value in zip(y, outcomes, strict=True):
        axis.text(
            value + 7, position, str(value), va="center", color=INK, fontweight="bold"
        )
    style_axis(axis)
    figure.text(
        0.025,
        0.07,
        "All 484 saved test-pool optima were unique. Of 16 single-question workloads, "
        "three selected the optimum.\n"
        "This attributes observed decision paths; it does not prove how a different "
        "tournament or final choice would perform.",
        fontsize=9.5,
        color="#536576",
        linespacing=1.6,
    )
    save(figure, "tournament-attrition")


def main():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.labelcolor": INK,
            "text.color": INK,
            "svg.fonttype": "none",
        }
    )
    plot_comparison()
    plot_candidate_counts()
    plot_tournament_attrition()
    print(f"Regenerated three PNG/SVG figure pairs in {ROOT}")


if __name__ == "__main__":
    main()
