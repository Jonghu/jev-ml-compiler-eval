# Run log — RQ2 TpuGraphs `tile:xla`

A chronological record of every evaluation run and offline analysis for the
TpuGraphs workstream. Add a new entry for each run; never edit past results.

## Status note — 2026-11-19

This is a chronological record. Earlier setup summaries and findings describe
what was known at their respective runs. R7 subsequently completed the full
676-kernel labelled validation scope with all distinct configurations. The
TpuGraphs implementation and this log are merged into `origin/main` through
PR #1 (`fbd67c1`); the current revision is integrating the later TenSet work,
with final integration validation still pending.

The local empirical models are Jev and Random. Paper GNN values are external
references; no local GraphSAGE training/checkpoint evaluation is implemented.
The compiler default is an offline project reference, not a registered model.
Those optional reproductions are not required blockers when the report clearly
labels the accepted external references. See the current
[project status](project-status.md) for report completion work.

## Setup shared by all runs

| Item | Value |
|---|---|
| Dataset | TpuGraphs `tile:xla`, official `valid` split (7 programs, 676 kernels) |
| Unit of evaluation | One kernel = one workload; its distinct tile configurations are the candidates |
| Ground truth | Measured, normalized runtimes, hidden from every model |
| Metrics | Top-1 accuracy, Top-1/5/10 error, Kendall's tau-b ([configuration](configuration.md#metrics)) |
| Jev question | "Which candidate is most likely to have the lowest runtime for this workload?" |

### Baselines

| Baseline | What it does | Status |
|---|---|---|
| Random | Uniformly random ranking, seed 42 | Implemented |
| Compiler default | Always ranks `config_0` (the XLA default configuration) first | Offline analysis only; not a model yet |
| GNN (official GraphSAGE) | Learned runtime model trained on the `train` split | Paper values below; reproduction undecided |

Lower Top-K error is better; higher Kendall's tau is better (1 = perfect
order, 0 = random, −1 = reversed). Errors in this log are fractions
(0.105 = 10.5%).

### Paper reference values

TpuGraphs paper (arXiv:2308.13490), Table 2, best tile model (EarlySAGE,
ListMLE), median of three training runs, evaluated on **all kernels with all
configurations**. Kendall's tau is not reported for the tile collection.

| Split | Top-1 err | Top-5 err | Top-10 err |
|---|---|---|---|
| valid | 0.105 | 0.030 | 0.018 |
| test | 0.091 | 0.042 | 0.028 |

None of the runs below are directly comparable with these values yet,
because they use a subset of kernels with at most 50 candidates. A comparable
run needs all 676 `valid` kernels with all configurations.

## Runs

### R1 — Random smoke test, candidates ≤ 50

- Date: 2026-11-05 · Commit: `713ce3f` plus uncommitted metric changes
- Config: `experiments/tpugraphs/tile-valid-random-smoke-test.yaml`
- Output: `results/tpugraphs/tile-valid-random-smoke-test/` (not in Git)
- Live Jev calls: none

| Model | Kernels | Top-1 acc | Top-1 err | Top-5 err | Top-10 err | Kendall τ |
|---|---|---|---|---|---|---|
| Random | 120 | 0.100 | 0.766 | 0.101 | 0.042 | 0.001 |

### A1 — Offline baseline analysis (no experiment run)

Computed directly from the dataset to calibrate later Jev results.

| Model | Subset | Kernels | Top-1 acc | Top-1 err | Top-5 err | Top-10 err | Kendall τ |
|---|---|---|---|---|---|---|---|
| Compiler default | candidates ≤ 50 | 120 | 0.383 | 0.021 | — | — | — |
| Compiler default | all | 676 | 0.330 | 0.024 | — | — | — |
| Random | all | 676 | — | 2.684 | 0.576 | 0.351 | −0.001 |

The compiler default is optimal in 223 of 676 kernels, within 1% of optimal
in 389, and more than 10% slower than optimal in 34 (worst: 51.5%).

### R2 — Jev connection test, candidates ≤ 5

- Date: 2026-11-05 · Commit: `031371e`
- Config: temporary copy of `tile-valid-random-vs-jev.yaml` with
  `max_candidates: 5` (not committed)
- Model: `jev-1.14.0` (resolved `jev-1.14.0`)
- Live Jev calls: 5 · Input/output tokens: 12,580 / 284 · Cost ≈ $0.0005
- Mean API latency: 140 ms

| Model | Kernels | Top-1 acc | Top-1 err | Top-5 err | Top-10 err | Kendall τ |
|---|---|---|---|---|---|---|
| Compiler default | 5 | 0.800 | 0.059 | — | — | — |
| Random | 5 | 0.400 | 0.108 | 0.000 | 0.000 | 0.227 |
| Jev | 5 | 0.000 | 3.123 | 0.000 | 0.000 | −0.507 |

Top-5 and Top-10 errors are zero by definition because every kernel has at
most five candidates.

Observations:

- In all five kernels Jev ranked the smallest tile first, and the smallest
  tile was among the slowest candidates.
- Jev gave `config_0` low probability (0.06–0.16) although it was optimal in
  four of the five kernels.
- Its top choice was `config_1` in four kernels, so a position bias cannot yet
  be separated from a tile-size bias.

### R3 — Jev connection test rerun, candidates ≤ 5

- Date: 2026-11-05 · Commit: `031371e` plus uncommitted cache schema v2
  changes
- Same configuration and kernels as R2; the v1 cache was deleted, so all
  requests were live
- Live Jev calls: 5 · Input/output tokens: 12,580 / 284 · Cost ≈ $0.0005
- Mean API latency: 108 ms

| Model | Kernels | Top-1 acc | Top-1 err | Top-5 err | Top-10 err | Kendall τ |
|---|---|---|---|---|---|---|
| Jev | 5 | 0.000 | 3.123 | 0.000 | 0.000 | −0.427 |

Observations:

- Jev responses are not deterministic. Identical requests returned
  probabilities that differed by up to 0.08, which changed one lower-ranked
  order and Kendall's tau (−0.507 → −0.427). Top-1 choices were unchanged.
- Probabilities are returned with two decimals, so ties are frequent. Ties
  currently keep the original candidate order.
- Cached responses make reruns reproducible, so the cache is part of the
  experimental record.

### R4 — Random versus Jev, candidates ≤ 50 (first full Jev baseline)

- Date: 2026-11-05 · Commit: `957d15a` (only untracked docs changed)
- Config: `experiments/tpugraphs/tile-valid-random-vs-jev.yaml`
- Output: `results/tpugraphs/tile-valid-random-vs-jev/` (not in Git)
- Model: `jev-1.14.0` (resolved `jev-1.14.0` for all 120)
- Jev calls: 115 live, 5 cached from R3 · Input/output tokens: 321,670 /
  27,390 · Cost ≈ $0.014
- Mean live API latency: 100 ms (max 248 ms)
- Kernels: 120 from 5 programs; 92 (77%) are from Inception-v3

| Model | Kernels | Top-1 acc | Top-1 err | Top-5 err | Top-10 err | Kendall τ |
|---|---|---|---|---|---|---|
| Compiler default (offline) | 120 | 0.383 | 0.021 | — | — | — |
| Random | 120 | 0.100 | 0.766 | 0.101 | 0.042 | 0.001 |
| Jev | 120 | 0.008 | 4.237 | 0.145 | 0.009 | −0.269 |

Per program (mean Top-1 error):

| Program | Kernels | Compiler default | Random | Jev | Jev τ |
|---|---|---|---|---|---|
| `inception_v3_batch_128_train` | 92 | 0.016 | 0.873 | 4.027 | −0.255 |
| `bert_pretraining.4x4.fp16` | 12 | 0.035 | 0.483 | 2.887 | −0.293 |
| `resnet50.4x4.fp16` | 8 | 0.026 | 0.291 | 9.985 | −0.326 |
| `tf2_bert_pretrain_dynamic_batch_size` | 5 | 0.050 | 0.421 | 1.489 | −0.410 |
| `mlperf_bert_batch_24_2x2` | 3 | 0.050 | 0.464 | 5.338 | −0.239 |

Diagnostics ("tile size" is the product of the `output_bounds` and
`kernel_bounds` entries, a proxy for tile volume):

| Check | Value |
|---|---|
| Jev picks the smallest tile | 96 of 120 kernels |
| Mean size percentile of Jev's pick (0 = smallest) | 0.05 |
| Kendall τ between Jev rank position and tile size | +0.29 (Jev ranks small tiles first) |
| Kendall τ between tile size and true runtime | −0.59 (larger tiles are faster) |
| Jev's top choice | `config_1` in 52, `config_2` in 24 kernels |
| Jev ranks `config_0` first | 3 of 120 kernels |
| Mean relative position of `config_0` in Jev's ranking | 0.23 (0 = top, 1 = bottom) |
| Jev beats the compiler default (Top-1) | 0 of 120 kernels |
| Jev beats Random (Top-1) | 7 of 120 kernels |
| Kernels with tied Jev probabilities | 111 of 120 |
| Mean Jev confidence / mean maximum probability | 0.372 / 0.415 |

Observations:

- Jev is worse than Random on Top-1 error (median 2.84, worst 23.4) and
  ranks in roughly the reverse runtime order (τ = −0.27).
- The error is systematic: Jev prefers small tiles, while larger tiles are
  faster in this data.
- Top-10 error (0.009) is better than Random's, because Jev usually places
  `config_0`, often optimal, in the upper part of its ranking without ranking
  it first.
- Jev's top choice is `config_1` in 43% of kernels, so position bias remains
  possible alongside the tile-size bias.

### A2 — Offline analysis of R4 Jev failures (no experiment run)

Computed from the R4 predictions and cached Jev responses. Tile size is the
product of the `output_bounds` and `kernel_bounds` entries.

Simple selection rules on the same 120 kernels (Top-1 error):

| Rule | Mean | Median | Optimal kernels |
|---|---|---|---|
| Jev (R4) | 4.237 | 2.840 | 1 |
| Smallest tile | 4.510 | 3.076 | 0 |
| Jev reversed (lowest probability) | 0.265 | 0.090 | 12 |
| Largest tile | 0.092 | 0.041 | 30 |
| Compiler default | 0.021 | 0.007 | 46 |

Observations:

- Jev's ranking is informative but inverted: reversing it beats Random, yet a
  one-line "largest tile" rule beats both.
- In R4, candidate position and tile size are correlated (median within-kernel
  τ = +0.52), so the R4 position bias could not be separated from size bias.
  In the 64 kernels where `config_1` is not the smallest tile, Jev picked the
  smallest tile 48 times and `config_1` 6 times.
- 87 of 120 kernels expose only `output_bounds`. Field meaning and units are
  never explained; the values appear to be in hardware-tile units, not
  elements (for example, default `[1, 1, 80, 3]` for shape `[1, 1, 1280, 384]`).
- `config_0` is usually a large tile (median size percentile 0.92), yet Jev
  ranks it near the top (median rank percentile 0.19).
- 5 groups of kernels (11 kernels) share identical candidate sets but differ in
  graph. Jev picked the same candidate within every group, while the true best
  differed within every group.
- Confidence is not calibrated: Spearman(confidence, Top-1 error) = +0.18, and
  33 of 35 picks with confidence ≥ 0.5 are the smallest tile.

### R5 — Jev ablation: no graph context, shuffled candidate order

- Date: 2026-11-11 · Commit: `078077c` plus uncommitted changes (optional
  `include_context`, `candidate_order`, and `order_seed` Jev settings; cache
  schema v3)
- Config: `experiments/tpugraphs/tile-valid-jev-context-order-ablation.yaml`
- Output: `results/tpugraphs/tile-valid-jev-context-order-ablation/` (not in
  Git)
- Model: `jev-1.14.0` (resolved `jev-1.14.0` for all 240)
- Same 120 kernels and serialization as R4; each variant changes one factor:
  - `no-context`: state contains only `workload_id`
  - `shuffled`: candidates sent in a random order seeded by `42:<workload_id>`
- Jev calls: 240 live · Input/output tokens: 447,101 / 54,780 · Cost ≈ $0.019
- Mean API latency: 84 ms (no-context), 89 ms (shuffled)

Before this run the cache key hashed requests with sorted keys, so a shuffled
request would have reused the R4 response. Cache schema v3 hashes the request
in sent order. The 120 v2 entries were re-keyed from their stored requests,
and R4 still reproduces from cache (120 of 120 hits).

| Model | Kernels | Top-1 acc | Top-1 err | Top-5 err | Top-10 err | Kendall τ |
|---|---|---|---|---|---|---|
| Jev (R4, reference) | 120 | 0.008 | 4.237 | 0.145 | 0.009 | −0.269 |
| Jev no-context | 120 | 0.008 | 4.451 | 0.183 | 0.011 | −0.373 |
| Jev shuffled | 120 | 0.017 | 3.689 | 0.057 | 0.009 | −0.277 |

Comparison with R4:

| Check | R4 | No-context | Shuffled |
|---|---|---|---|
| Picks the smallest tile | 96 | 107 | 96 |
| Same pick as R4 | — | 94 | 89 |
| Median τ between probabilities and R4 probabilities | — | 0.67 | 0.60 |
| Median rank percentile of `config_0` | 0.19 | 0.18 | 0.13 |
| Kernels with tied probabilities | 111 | 108 | 109 |

Pooled within-kernel regression of standardized probability on size rank and
presented position rank:

| Run | Size coefficient | Position coefficient | Median τ(position, size) |
|---|---|---|---|
| R4 | −0.893 | −0.526 | +0.52 |
| Shuffled | −1.023 | +0.204 | +0.02 |

Observations:

- The small-tile preference is robust to candidate order: shuffling leaves 96
  smallest-tile picks and the size coefficient unchanged. The apparent R4
  position bias was largely size bias, because position and size are
  correlated in the original order.
- `config_0` stays near the top after shuffling, including in the 117 kernels
  where it is not presented first. Its high rank is not a position effect; the
  ID name or its configuration remains a candidate explanation.
- Graph context has a small effect: removing it changes 26 picks (17 worse, 9
  better) and moves Jev further toward the smallest tile (96 → 107).
- The response noise floor is unknown. R3 showed nondeterminism, so some of
  the 26 and 31 changed picks may be noise; the Top-1 error difference between
  R4 and shuffled (4.24 versus 3.69) is not yet significant.

### R6 — Jev with configuration field descriptions

- Date: 2026-11-11 · Commit: `9504919` plus uncommitted changes (optional
  `describe_config_fields` TpuGraphs dataset setting)
- Config: `experiments/tpugraphs/tile-valid-jev-field-descriptions.yaml`
- Output: `results/tpugraphs/tile-valid-jev-field-descriptions/` (not in Git)
- Model: `jev-1.14.0` (resolved `jev-1.14.0` for all 120)
- Same 120 kernels, candidate order, and Jev settings as R4. The only change
  is a `configuration_fields` object in the context (about 390 tokens) that
  restates the official README definitions of `output_bounds`,
  `kernel_bounds`, and `input_bounds`, plus the tile size selection task
  sentence from the TpuGraphs paper. It states neither units nor which tile
  sizes tend to be faster.
- Jev calls: 120 live · Input/output tokens: 338,950 / 27,422 · Cost ≈ $0.014
- Mean API latency: 98 ms (max 204 ms)

| Model | Kernels | Top-1 acc | Top-1 err | Top-5 err | Top-10 err | Kendall τ |
|---|---|---|---|---|---|---|
| Compiler default (offline) | 120 | 0.383 | 0.021 | — | — | — |
| Random (R4) | 120 | 0.100 | 0.766 | 0.101 | 0.042 | 0.001 |
| Jev (R4) | 120 | 0.008 | 4.237 | 0.145 | 0.009 | −0.269 |
| Jev with field descriptions | 120 | 0.158 | 0.904 | 0.012 | 0.005 | +0.143 |

| Check | R4 | R6 |
|---|---|---|
| Picks the smallest tile | 96 | 11 |
| Picks the largest tile | 2 | 40 |
| Picks `config_0` | 3 | 41 |
| Optimal pick | 1 | 19 |
| Mean size percentile of the pick | 0.05 | 0.74 |
| Median rank percentile of `config_0` | 0.19 | 0.04 |
| Size coefficient (pooled regression) | −0.893 | +1.483 |
| Position coefficient (pooled regression) | −0.526 | −0.949 |
| Mean confidence | 0.372 | 0.153 |
| Spearman(confidence, Top-1 error) | +0.18 | −0.20 |
| Kernels with tied probabilities | 111 | 107 |

Per kernel, R6 beats R4 in 103 and loses in 2, beats Random in 79 and loses
in 31, and beats the compiler default in 7, ties in 41, and loses in 72.

Per program (mean Top-1 error):

| Program | Kernels | Jev R4 | Jev R6 | Jev R6 τ |
|---|---|---|---|---|
| `inception_v3_batch_128_train` | 92 | 4.027 | 0.546 | +0.166 |
| `bert_pretraining.4x4.fp16` | 12 | 2.887 | 0.283 | +0.176 |
| `resnet50.4x4.fp16` | 8 | 9.985 | 6.635 | +0.034 |
| `tf2_bert_pretrain_dynamic_batch_size` | 5 | 1.489 | 0.257 | −0.070 |
| `mlperf_bert_batch_24_2x2` | 3 | 5.338 | 0.180 | −0.066 |

Observations:

- Describing the fields reverses the size preference: Jev now favors large
  tiles (size coefficient −0.89 → +1.48), and Top-1 error drops from 4.24 to
  0.90 (median 0.063). The change is far larger than the R3/R5 variation, so
  it is unlikely to be noise. The small-tile preference in R4 came from the
  missing field semantics.
- The mean Top-1 error is still above Random's because of outliers (maximum
  13.6; `resnet50.4x4.fp16` mean 6.6). Per kernel, Jev beats Random in 79 of
  120.
- Jev is still far from the compiler default (0.90 versus 0.021). Its best
  case is matching the default: 41 of its 120 picks are `config_0`.
- Jev now ranks `config_0` first or near the top in most kernels (median rank
  percentile 0.04), and the position coefficient grew. Because `config_0` is
  first in the original order and usually large, this run cannot separate a
  size preference from a preference for the first or default-looking
  candidate.
- The descriptions are a bundle. The tile size selection sentence ("maximize
  hardware compute efficiency while ... data fit in local memory") may drive
  most of the effect, and it can be read as favoring larger tiles. Which part
  matters is not yet tested.
- Confidence is lower on average and now weakly calibrated in the right
  direction.

### A3 — R6 versus the paper reference values (no experiment run)

Compares R6 with the TpuGraphs paper (arXiv:2308.13490). The values are
**not directly comparable**: R6 covers 120 of 676 `valid` kernels with at most
50 candidates each, while the paper evaluates all kernels with all
configurations (median 593, maximum 9,988 per kernel). Errors here are
percentages to match the paper.

Paper model: a GNN performance model trained on the `train` split (5,709
kernels) with measured runtimes. Opcode embeddings and node features feed a
GraphSAGE or GCN. In the early-join variant, the 24 tile configuration
features are copied onto every node. Pooled node embeddings pass through a
feedforward layer to one score per configuration, and configurations are
ranked by that score. Training uses ListMLE or MSE, and the paper reports the
median of three training runs. Jev is not trained on TpuGraphs.

Overall (`valid`, mean over kernels):

| Model | Scope | Top-1 err | Top-1 slowdown | Top-5 err | Top-10 err |
|---|---|---|---|---|---|
| Paper EarlySAGE, ListMLE (best; Table 2) | 676 kernels, all configs | 10.5 | 1.11× | 3.0 | 1.8 |
| Paper EarlyGCN, ListMLE (Table 8) | 676, all | 11.5 | 1.12× | 3.6 | 2.4 |
| Paper MLP, ListMLE (Table 8) | 676, all | 84.5 | 1.85× | 39.7 | 25.6 |
| Jev R6 | 120, candidates ≤ 50 | 90.4 (median 6.3) | 1.90× | 1.2 | 0.5 |
| Random (R4) | 120, candidates ≤ 50 | 76.6 | 1.77× | 10.1 | 4.2 |
| Random (A1) | 676, all | 268.4 | 3.68× | 57.6 | 35.1 |

Top-1 slowdown is 1 + Top-1 err / 100.

Per program (Top-1 / Top-5 err; paper Table 9, ListMLE EarlySAGE):

| Program | Paper (kernels) | Jev R6 (kernels) | Random, R6 subset |
|---|---|---|---|
| `inception_v3_batch_128_train` | 6 / 1 (264) | 54.6 / 1.0 (92) | 87.3 / 12.3 |
| `bert_pretraining.4x4.fp16` | 6 / 1 (56) | 28.3 / 1.4 (12) | 48.3 / 2.2 |
| `resnet50.4x4.fp16` | 15 / 4 (103) | 663.5 / 1.3 (8) | 29.1 / 1.7 |
| `tf2_bert_pretrain_dynamic_batch_size` | 5 / 2 (60) | 25.7 / 5.0 (5) | 42.1 / 5.5 |
| `mlperf_bert_batch_24_2x2` | 15 / 4 (75) | 18.0 / 2.6 (3) | 46.4 / 1.8 |
| `resnet_v1_50_official_batch_128_bf16` | 13 / 4 (108) | not in subset | — |
| `unet_3d.4x4.bf16` | 48 / 6 (10) | not in subset | — |

Observations:

- Top-1: Jev R6 (90.4) is far behind the best paper model (10.5), near the
  graph-free paper MLP (84.5), and behind Random on the same subset (76.6).
  It is behind the paper in every program it covers.
- Top-5 and Top-10: R6 looks better than the paper, but the subset makes these
  metrics much easier. Random's Top-10 error is 4.2 on the subset and 35.1 on
  all 676 kernels, because ten picks cover at least a fifth of every subset
  kernel's candidates.
- Moving to all 676 kernels is expected to make every Jev metric worse, since
  kernels with many candidates are harder. This is an expectation, not a
  measurement.
- The compiler default (always `config_0`) is not a paper baseline. It is a
  project reference for interpretation only. Its Top-1 error on all 676
  kernels (2.4, A1) is below the paper's 10.5; the reason has not been
  checked, and a Jev result that beats the paper mainly by picking `config_0`
  would not show model ability.

### R7 — Full `valid` split (676 kernels, all configurations), blind tournament

- Date: 2026-11-11 · Commit: `bed9521` plus uncommitted changes (new
  `jev-tournament` model, `parse_choice_answer` refactor)
- Config: `experiments/tpugraphs/tile-valid-all-jev-tournament.yaml`
- Output: `results/tpugraphs/tile-valid-all-jev-tournament/` (not in Git)
- Model: `jev-1.14.0` (resolved `jev-1.14.0` for all requests)
- Scope: all 676 `valid` kernels and all 1,038,376 distinct configurations,
  matching the paper's evaluation scope. Identical configuration rows are
  merged (see `tpugraphs.md`), which the paper does not do.
- Input: R6 serialization with field descriptions. Candidates are renamed
  to neutral IDs (`c001`, ...) in a seeded shuffled order, so Jev cannot
  recognize `config_0`.
- Tournament: Choice questions of at most 255 options. Each question's top 5
  advance to one final question. Questions of a round share requests of at
  most 56k estimated tokens, with at most 28k for the state plus one question.
  263 kernels need one question; 413 need two rounds.
- Rate limits (client side): 960 requests/min, 200k input tokens/s; no
  waiting was needed, and no 429 errors occurred.
- A live probe of the two largest kernels (16 requests, cached and reused)
  confirmed 255-option questions and requests of about 50k tokens.
- Jev calls: 1,819 requests (1,803 live), 4,866 questions · Input/output
  tokens: 54,225,110 / 11,637,846 · Cost ≈ $2.28 (output is unbilled)
  - The 16 cached requests are the probe's (699,457 input tokens, ≈ $0.03,
    paid once before the run). The run itself sent 53,525,653 input tokens,
    ≈ $2.25, or about $0.0033 per kernel.
- Time: about 7.5 min wall clock for the run (first new cache entry 23:06
  local to manifest 05:13:56 UTC). Total API time 6.9 min; requests are sent
  sequentially. Decision latency per kernel: mean 0.65 s, maximum 4.0 s.

Results (percent, mean over kernels; paper values from Table 2 and Table 8):

| Model | Top-1 err | Top-1 slowdown | Top-1 median | Top-5 err | Top-10 err | Kendall τ |
|---|---|---|---|---|---|---|
| Paper EarlySAGE, ListMLE | 10.5 | 1.11× | — | 3.0 | 1.8 | not reported |
| Paper MLP, ListMLE | 84.5 | 1.85× | — | 39.7 | 25.6 | not reported |
| **Jev tournament, blind** | **413.6** | **5.14×** | 80.3 | **48.9** | **24.2** | +0.079 |
| Random | 268.4 | 3.68× | 111.7 | 57.6 | 35.1 | −0.001 |
| Compiler default (reference) | 2.4 | 1.02× | 0.6 | — | — | — |

Errors follow the paper's Eq. 1 (Top-K E %, mean over kernels). Top-1
slowdown is 1 + Top-1 err / 100: how many times slower than the optimum the
first pick is on average. Paper values are the median of three training
runs; the Jev and Random values come from a single run.

Kendall τ is shown for reference only: the tournament ranking is complete
only among the finalists, and the paper does not report τ for tiles.

Per program (Top-1 / Top-5 err %; paper Table 9, ListMLE EarlySAGE):

| Program | Kernels | Paper | Jev | Random Top-1 |
|---|---|---|---|---|
| `inception_v3_batch_128_train` | 264 | 6 / 1 | 368.9 / 48.0 | 282.7 |
| `resnet_v1_50_official_batch_128_bf16` | 108 | 13 / 4 | 496.2 / 48.2 | 342.9 |
| `resnet50.4x4.fp16` | 103 | 15 / 4 | 411.2 / 56.7 | 266.9 |
| `mlperf_bert_batch_24_2x2` | 75 | 15 / 4 | 549.0 / 76.6 | 205.5 |
| `tf2_bert_pretrain_dynamic_batch_size` | 60 | 5 / 2 | 353.3 / 32.9 | 220.4 |
| `bert_pretraining.4x4.fp16` | 56 | 6 / 1 | 415.7 / 26.3 | 189.4 |
| `unet_3d.4x4.bf16` | 10 | 48 / 6 | 64.1 / 15.4 | 305.9 |

By candidate count (Top-1 err %):

| Candidates | Kernels | Jev mean | Jev median | Jev Top-5 | Random mean |
|---|---|---|---|---|---|
| ≤ 50 | 120 | 107.2 | 8.5 | 5.2 | 76.6 |
| 51–255 | 143 | 539.4 | 98.7 | 29.4 | 189.4 |
| 256–1,000 | 152 | 519.3 | 153.6 | 66.1 | 235.4 |
| 1,001–9,988 | 261 | 424.1 | 96.3 | 69.8 | 419.1 |

Single-question quality (kernels with one question; random = chance level):

| Options | Questions | Best picked | Random | Best in top 5 | Random | Zero-probability share | Distinct probability values |
|---|---|---|---|---|---|---|---|
| ≤ 50 | 120 | 0.158 | 0.073 | 0.483 | 0.353 | 0.00 | 7.7 |
| 51–150 | 63 | 0.016 | 0.011 | 0.111 | 0.057 | 0.29 | 6.3 |
| 151–255 | 80 | 0.050 | 0.005 | 0.150 | 0.027 | 0.69 | 5.8 |

Where the best configuration is lost (413 two-round kernels):

| Outcome | Kernels |
|---|---|
| Eliminated in round 0 | 345 (84%) |
| Reached the final but not picked | 63 |
| Picked | 5 |

If the final round had picked perfectly among its finalists, the mean Top-1
error on these kernels would be 17.4% instead of the actual 459.2%.

Other checks: Jev picks the smallest tile in 76 kernels and the largest in
62 (mean size percentile of the pick 0.61, versus 0.74 in R6). It picks
`config_0` in 40 kernels without seeing its name. Per kernel, it beats Random
in 371 and loses in 293, and beats the compiler default in 37. The worst 5%
of kernels (errors up to 8,480%) account for 43% of the total Top-1 error.
Mean final confidence is 0.127; Spearman(confidence, Top-1 error) = −0.11.

Observations:

- On the paper's evaluation scope, Jev is far behind the paper's GNN on every
  metric: Top-1 413.6% versus 10.5%, Top-5 48.9% versus 3.0%, and Top-10
  24.2% versus 1.8%. It is behind the paper in every program.
- Jev is worse than Random on mean Top-1 error, because a few catastrophic
  picks dominate the mean. It is better than Random on median Top-1, Top-5,
  Top-10, and per-kernel wins (371 versus 293), so it carries a weak signal.
- The signal fades as questions grow. With more than 50 options, Jev returns
  only about six distinct probability values per question, and most options
  get 0.00 (69% at 151–255 options). Which candidates advance is then decided
  mostly by tie order, which is the shuffled presentation order.
- The tournament loses the best configuration in round 0 for 84% of
  two-round kernels. The final round is also weak: picking the best finalist
  would give 17.4%, but Jev's final pick gives 459.2%.
- Blinding costs little on small kernels: on the same 120 kernels with at
  most 50 candidates, mean Top-1 is 107.2% blind versus 90.4% in R6
  (median 8.5% versus 6.3%).
- Prediction metadata keeps each candidate's probability from its last
  question only, so round-0 probabilities of finalists are not stored there.
  The complete per-question answers are in the Jev cache.

### A4 — Rationale for a large-tile candidate filter (no experiment run)

- Date: 2026-11-24 · Commit: `fbd67c1` plus staged R7 analysis code
- Live Jev calls: none. Computed from the dataset with ground-truth runtimes.
- Purpose: justify, before any ablation run, a candidate filtering step that
  keeps the largest tiles. A2 found that large tiles are fast, but A2 looked
  at `valid` labels, so a filter justified only by A2 would be tuned on the
  evaluation split. This entry checks the same pattern on the `train` split,
  which no evaluated model sees, and records the domain rationale.

Tile size is the product of the `output_bounds` and `kernel_bounds` entries
(as in A2). The filter keeps the K largest candidates of each kernel; ties
keep the original candidate order. Kernels with at most K candidates are
unchanged. Spearman correlation is computed per kernel with average ranks for
ties.

Does tile size predict runtime?

| Check | `train` (5,709 kernels) | `valid` (676 kernels) |
|---|---|---|
| Median within-kernel Spearman(tile size, runtime) | −0.803 | −0.819 |
| Kernels with a negative correlation | 98.3% | 99.3% |
| Median size percentile of the optimum (1 = largest) | 0.94 | 0.95 |
| Optimum in the larger half of the candidates | 90.0% | 91.3% |

Is "always pick the largest tile" enough? (Top-1 error, %)

| Rule | `train` mean (median) | `train` optimal | `valid` mean (median) | `valid` optimal |
|---|---|---|---|---|
| Pick the largest tile | 23.2 (11.7) | 7.0% | 20.6 (7.9) | 16.1% |

Largest-K filter (Top-1 error, %; "oracle" picks the best kept candidate,
"Random" is the expected error of a uniform pick among kept candidates):

| K | `train` best kept | `train` oracle | `train` Random | `valid` best kept | `valid` oracle | `valid` Random |
|---|---|---|---|---|---|---|
| 25 | 47.0% | 3.3 | 32.7 | 60.5% | 3.6 | 42.2 |
| 50 | 60.5% | 2.0 | 41.3 | 70.3% | 2.6 | 54.4 |
| 100 | 73.9% | 1.1 | 56.8 | 80.0% | 1.6 | 70.4 |

For comparison on `valid`: unfiltered Random 268.4% (A1), a random 50-candidate
subsample with a uniform pick 272.6%, Jev R7 413.6%, and the paper's best GNN
10.5%.

Rationale:

- **Data.** On `train`, larger tiles are faster in almost every kernel, and the
  optimum is usually among the largest tiles. The `valid` pattern matches, so
  the filter is not an artifact of the evaluation split.
- **Domain.** The TpuGraphs paper describes tile size selection as choosing an
  output tile that maximizes hardware compute efficiency while the data still
  fit in local memory (the sentence used in R6). Larger tiles raise compute
  efficiency until the memory limit binds, which predicts the pattern above.
- **Why a filter rather than a rule.** The largest tile is optimal in only
  7.0% of `train` kernels. The optimum lies just below the largest sizes, where
  the memory limit and tile shape matter. A filter keeps that region and leaves
  the final choice to the decision model.
- **Why K = 50 (proposed).** R7 shows that Jev's Choice probabilities keep
  resolution only up to about 50 options. With K = 50 every kernel needs one
  question and no tournament, and the oracle error on `train` is 2.0%, so the
  filter leaves enough headroom.

Consequences for reporting a filtered Jev run:

- The filter is a hand-crafted heuristic and carries most of the gain on its
  own. A filtered Jev run must be reported with the same filter followed by a
  Random pick and by the largest-tile pick. Jev's contribution is its
  improvement over those baselines, not over unfiltered Jev.
- K was chosen from `train` statistics and the R7 option limit, not from the
  `valid` results in this table.

## Findings so far

1. The compiler default is a far stronger reference than Random. It is a
   project reference for interpretation, not the research comparison, which
   is the paper's learned models (see finding 7).
2. With the current serialization and question, Jev performs worse than
   Random on 120 kernels (Top-1 error 4.24 versus 0.77) because it
   systematically prefers small tiles, which are slow in this data.
3. Cost is negligible (≈ $0.0001 per kernel at this size). The Jev input
   limit, not cost, constrains how many kernels can be evaluated.
4. These results are not comparable with the paper values until all kernels
   and all configurations are evaluated.
5. The small-tile preference is the dominant failure. It survives shuffled
   candidate order and removal of the graph context (R5), and the
   configuration fields reach Jev without meaning or units (A2).
6. Describing the configuration fields fixes the direction of that
   preference (R6: Top-1 error 4.24 → 0.90, Kendall τ −0.27 → +0.14), but Jev
   remains far behind the compiler default and leans heavily on `config_0`.
7. The research comparison is against the paper's learned models, not the
   compiler default (A3). On Top-1, Jev R6 is at the level of the paper's
   graph-free MLP and far behind its best GNN, even on an easier subset.
8. On the paper's full `valid` scope (R7), blind Jev with a Choice tournament
   is far behind the paper's GNN (Top-1/5/10 error 413.6/48.9/24.2% versus
   10.5/3.0/1.8%) and worse than Random on mean Top-1. Choice probabilities
   lose resolution beyond about 50 options, so large questions rank mostly by
   tie order.

## TODO — reconciled 2026-11-19

Completed implementation and experiment items:

- [x] Commit the cache changes and this chronological run log.
- [x] Keep generated raw bundles and caches outside Git under the repository's
  generated-data policy; publish only reviewed report outputs.
- [x] Complete the 120-kernel Random-versus-Jev baseline (R4).
- [x] Implement and run candidate grouping, request batching, and a full
  676-kernel blind tournament (R7).
- [x] Define deterministic question-local tie handling and record presented order.
- [x] Run no-context and shuffled-order variants (R5).
- [x] Run the field-description variant (R6).
- [x] Evaluate neutral shuffled candidate identifiers in R7, while preserving
  the distinction between that combined change and an ID-only causal test.
- [x] Report per-program, candidate-count, elimination, probability-tie, and
  outlier summaries in R4–R7.
- [x] Record the two-largest-kernel probe and budgets that completed all R7 cases.
- [x] Retain paper GNN numbers as explicitly external references.

Remaining report work and optional extensions:

- [ ] Incorporate the RQ2 results, figures, limitations, and source references
  into the final report and cross-dataset comparison.
- [ ] Verify the assumed input-token price against a dated provider or billing
  record before treating estimates as monetary measurements.
- [ ] Optional: inspect concrete graph/configuration explanations for the
  largest R6/R7 failures in addition to the completed aggregate diagnostics.
- [ ] Optional: separate R6 field definitions from its tile-selection sentence,
  and isolate field, identifier, and order effects on fixed candidate pools.
- [ ] Optional: extend identical-request repetitions beyond the five-kernel
  R2/R3 pilot and investigate confidence calibration.
- [ ] Optional: implement the compiler default as a reusable decision model.
- [ ] Optional: reproduce a local GraphSAGE baseline and integrate its rankings
  only if the team elects a paired learned-model comparison.

R1–R7 are preserved above; this reconciliation does not alter their reported
measurements or claim that optional follow-ups have been run.
