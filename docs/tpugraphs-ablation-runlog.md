# Run log — RQ2 TpuGraphs context-engineering ablation (A0–A5)

A chronological record of the ablation defined in
[the ablation plan](tpugraphs-ablation-plan.md). Earlier runs R1–R7 and
offline analyses A1–A4 are in [the main run log](runlog.md). Add a new entry
for each run; never edit past results.

Variant names A0–A5 refer to the plan, not to the offline analyses A1–A4 of
the main run log.

## AB1 — A0–A5 on kernels with at most 50 configurations, three repeats

- Date: 2026-11-25 (03:20–06:02 UTC) · Commit: `e48b8c6` plus uncommitted
  changes (`describe_graph_fields` dataset setting; `candidate_order`,
  `keep_largest`, and `size_fields` Jev tournament settings; the summary
  script)
- Configs: `experiments/tpugraphs/ablation/{a0-naive,a1-config-fields,a2-graph-fields,a3-a5-both-fields}.yaml`
- Output: `results/tpugraphs/ablation-small/` (not in Git; layout in its
  `README.md`). Tables: `summary/summary.md`; raw Jev requests and responses:
  `jev-cache/rep{1,2,3}/`.
- Model: `jev-1.14.0` through `jev-tournament`; every kernel is one Choice
  question. All variants are blind: opaque IDs and a shuffled order seeded by
  `42:<workload_id>`.
- Kernels: the same 120 `valid` kernels as R4–R6 (5 programs; 92 from
  Inception-v3).
- Jev calls: 21 runs × 120 kernels; 2,409 live requests. 111 requests were
  served from the same repeat's cache because they were identical to an
  earlier request: the 36 A5 kernels with at most 10 candidates equal A3, and
  one A4-ascending kernel equals A3 in every repeat.
- Tokens: 7,340,406 input (including the 111 reused requests) and 504,141
  output. At the R7 rate (≈ $0.042 per million input tokens) this is at most
  about $0.31.
- Latency: median decision latency 76–98 ms per run. One A3 repeat-2 kernel
  took 3,597 s, and the last experiment ran from 03:22 to 06:02 UTC, because
  requests stalled for about 2.5 hours; cache timestamps show almost no
  writes between 03:22 and 06:00 UTC. No error was raised, and responses are
  unaffected. The cause was not verified (likely host sleep or a network
  interruption). Mean latencies of that experiment are therefore not
  meaningful; use the medians.

### Results

Top-1 error in percent, mean over kernels; ± is the standard deviation over
the three repeats. "Same pick" is the share of kernels where all three
repeats picked the same candidate.

| Variant | Top-1 err | Top-1 median | Top-1 acc | Top-5 err | Kendall τ | Smallest-tile picks | Largest-tile picks | Same pick |
|---|---|---|---|---|---|---|---|---|
| A0 naive | 379.9 ± 21.1 | 270.6 | 0.000 | 26.1 | −0.326 | 95.0 | 1.3 | 0.88 |
| A1 + candidate definitions | **91.8 ± 10.5** | 7.8 | 0.156 | 4.9 | +0.171 | 8.0 | 39.3 | 0.78 |
| A2 + graph specification | 409.6 ± 4.1 | 284.0 | 0.000 | 29.8 | −0.331 | 99.3 | 0.3 | 0.91 |
| A3 + both | 155.3 ± 7.9 | 17.2 | 0.136 | 5.3 | +0.113 | 16.3 | 33.3 | 0.80 |
| A4 A3, size ascending | 237.5 ± 1.6 | 36.5 | 0.067 | 10.4 | −0.048 | 45.3 | 30.7 | 0.80 |
| A4 A3, size descending | **91.8 ± 12.5** | 12.0 | 0.167 | 3.4 | +0.205 | 10.0 | 59.7 | 0.78 |
| A5 A3, 10 largest shown | **34.0 ± 2.8** | 6.6 | 0.167 | 1.7 | +0.441* | 3.0 | 48.3 | 0.78 |
| Random (seed 42, one run) | 76.6 | 26.1 | 0.100 | 10.1 | +0.001 | 9 | 12 | — |

\* A5's Kendall τ is inflated by construction: filtered candidates are ranked
last in size order, which the heuristic already orders well. Compare A5 by
Top-1 metrics only.

Offline heuristics on the same 120 kernels (no model). The filter rows use
exactly the candidate pools that A5 showed Jev.

| Baseline | Top-1 err | Top-1 median | P(optimal pick) |
|---|---|---|---|
| Compiler default (`config_0`) | 2.1 | 0.7 | 0.383 |
| Largest tile | 9.2 | 4.1 | 0.250 |
| Largest-10 filter + random pick (expected) | 37.7 | 25.8 | 0.095 |
| Largest-10 filter + oracle pick | 0.4 | 0.0 | 0.825 |
| Random pick (expected) | 85.0 | 58.8 | 0.073 |

Paired per-kernel comparisons (Top-1 error averaged over repeats; "better"
means the right variant has lower error; two-sided exact sign test, ties
excluded):

| Left | Right | Better | Worse | Ties | Mean change (pp) | p |
|---|---|---|---|---|---|---|
| A0 | A1 | 108 | 3 | 9 | −288.1 | 2e-28 |
| A0 | A2 | 9 | 17 | 94 | +29.8 | 0.17 |
| A0 | A3 | 98 | 6 | 16 | −224.6 | 2e-22 |
| A1 | A3 | 10 | 36 | 74 | +63.5 | 0.0002 |
| A2 | A3 | 102 | 2 | 16 | −254.3 | 5e-28 |
| A3 | A4 ascending | 18 | 50 | 52 | +82.2 | 0.0001 |
| A3 | A4 descending | 49 | 28 | 43 | −63.5 | 0.022 |
| A4 ascending | A4 descending | 68 | 16 | 36 | −145.7 | 9e-09 |
| A3 | A5 | 47 | 13 | 60 | −121.3 | 1e-05 |

A5 against the "filter + random pick" baseline on the same pools, per
kernel: A5 is better in 82 kernels and worse in 38 (53 versus 31 on the 84
kernels the filter changes). The shown pool contains an optimum in 99 of 120
kernels.

### Observations

- **Candidate definitions are the largest effect (A0 → A1).** Top-1 error
  falls from 379.9% to 91.8%, better in 108 of 120 kernels, and the
  smallest-tile picks drop from 95 to 8. This reproduces R4 → R6
  (423.7% → 90.4%) under blinding, so R6's gain came from the field
  definitions rather than from recognizing `config_0` by name.
- **The graph specification alone does nothing (A0 → A2).** 94 of 120 kernels
  are unchanged and the rest split 9 to 17 (p = 0.17). Explaining the graph
  without explaining the tiles leaves the small-tile failure intact (99.3
  smallest-tile picks).
- **The two descriptions are not complementary (A1 → A3).** Adding the graph
  specification to the candidate definitions makes Top-1 error worse
  (91.8% → 155.3%; worse in 36 kernels, better in 10, p = 0.0002), and
  smallest-tile picks double (8.0 → 16.3). Longer context about the graph
  appears to dilute the tile semantics. The 2×2 effect is therefore driven
  almost entirely by the candidate definitions.
- **Jev is strongly position-sensitive (A4).** With the same candidates, size
  ascending gives 237.5% and size descending gives 91.8% (worse in 16
  kernels, better in 68, p = 9e-09). Jev favors candidates shown early:
  ascending order raises smallest-tile picks to 45.3, descending order raises
  largest-tile picks to 59.7. Because the content is identical, the
  146-point gap is a presentation effect, not reasoning about tile sizes.
- **Filtering helps, but mostly through the heuristic (A5).** A5 reaches
  34.0% (median 6.6%), the best Jev result, and beats A3 in 47 kernels
  against 13. The filter alone with a random pick already gives 37.7%; Jev
  improves on it per kernel (82 versus 38) and especially in the median
  (6.6% versus 25.8%), but its mean is only 3.7 points lower.
- **Every Jev variant stays behind the simple heuristics.** The single
  largest tile (9.2%) and the compiler default (2.1%) beat every variant,
  including A5. Within the filtered pool, an oracle would reach 0.4%, so Jev
  leaves most of the filter's headroom unused.
- **Response variation is small relative to the effects.** Repeat standard
  deviations are 1.6–21.1 points, and 78–91% of kernels get the same pick in all
  three repeats. Every difference reported as significant above exceeds the
  repeat spread.

### Limitations

- The 120-kernel subset favors small kernels and Inception-v3; R7 showed that
  larger kernels behave differently.
- A4 tests two orders only; position effects inside A3's shuffled order are
  not separated per kernel.
- The graph specification text was written for this project, not taken from
  an official definition. A different wording might behave differently.
- A5's filter comes from a heuristic derived from training labels (main run
  log, A4), and K = 10 was chosen for this subset.

## Findings so far

1. Explaining what the candidate fields mean is the only description that
   helps, and it helps a lot (A1). This holds without candidate names or
   order cues.
2. Explaining the graph does not help on its own and hurts when added to the
   candidate definitions (A2, A3).
3. Presentation order changes Top-1 error by about 146 points with identical
   content (A4): Jev prefers early candidates.
4. A large-tile filter gives the best Jev result (A5, 34.0%), but most of the
   gain comes from the filter; Jev adds a modest per-kernel improvement over
   a random pick inside the filtered pool.
5. No variant approaches the largest-tile rule (9.2%) or the compiler default
   (2.1%) on these kernels.
