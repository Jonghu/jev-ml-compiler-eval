from jev_compiler_eval.evaluation.tpugraphs_analysis import (
    QuestionTrace,
    contribution_curve,
    opaque_aliases,
    optimum_fate,
    program_name,
    slowdown_buckets,
)


def test_slowdown_buckets_count_kernels_and_error_share() -> None:
    buckets = slowdown_buckets([0.0, 0.5, 4.0, 15.0])

    assert [(label, count) for label, count, _ in buckets] == [
        ("< 1.1×", 1),
        ("1.1–2×", 1),
        ("2–10×", 1),
        ("≥ 10×", 1),
    ]
    assert [round(share, 4) for *_, share in buckets] == [0.0, 0.0256, 0.2051, 0.7692]


def test_contribution_curve_orders_worst_kernels_first() -> None:
    shares, cumulative = contribution_curve([1.0, 3.0, 0.0, 4.0])

    assert shares == [0.25, 0.5, 0.75, 1.0]
    assert cumulative == [0.5, 0.875, 1.0, 1.0]


def test_contribution_curve_handles_zero_error() -> None:
    assert contribution_curve([0.0, 0.0]) == ([], [])


def test_opaque_aliases_match_the_tournament_padding() -> None:
    aliases = opaque_aliases([f"config_{index}" for index in range(12)])

    assert aliases["c01"] == "config_0"
    assert aliases["c12"] == "config_11"


def test_program_name_drops_the_kernel_hash() -> None:
    assert program_name("resnet50.4x4.fp16_-2a38ea52e2c467f6") == "resnet50.4x4.fp16"


def _trace(round_index: int, order: list[str]) -> QuestionTrace:
    return QuestionTrace(
        round_index=round_index,
        question_id=f"r{round_index}_q0",
        candidate_ids=tuple(order),
        order=tuple(order),
        probabilities={candidate: 0.1 for candidate in order},
    )


def test_optimum_fate_reports_round_zero_elimination() -> None:
    traces = [
        _trace(0, ["a", "b", "best", "c"]),
        _trace(1, ["a", "b"]),
    ]

    fate = optimum_fate(traces, ["best"], advance=2)

    assert fate == "eliminated in round 0, rank 3 of 4 (p=0.10)"


def test_optimum_fate_reports_final_rank_and_pick() -> None:
    traces = [_trace(0, ["a", "best", "c"]), _trace(1, ["a", "best"])]

    assert optimum_fate(traces, ["best"], advance=2) == "final, rank 2 of 2 (p=0.10)"
    assert optimum_fate(traces, ["a"], advance=2) == "picked"
