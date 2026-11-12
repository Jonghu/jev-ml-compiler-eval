# Run log — RQ2 TpuGraphs `tile:xla`

A chronological record of every evaluation run and offline analysis for the
TpuGraphs workstream. Add a new entry for each run; never edit past results.

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

## Findings so far

1. The compiler default is a far stronger reference than Random. Jev must be
   compared against it, not only against Random.
2. With the current serialization and question, Jev performs worse than
   Random on 120 kernels (Top-1 error 4.24 versus 0.77) because it
   systematically prefers small tiles, which are slow in this data.
3. Cost is negligible (≈ $0.0001 per kernel at this size). The Jev input
   limit, not cost, constrains how many kernels can be evaluated.
4. These results are not comparable with the paper values until all kernels
   and all configurations are evaluated.

## TODO

- [x] Commit the cache schema v2 change (`957d15a`).
- [ ] Commit this run log.
- [ ] Decide as a team whether result bundles and `.cache/jev` are committed.
- [x] Run the 120-kernel Random versus Jev baseline (R4).
- [ ] Decide whether GNN values are cited from the paper or reproduced.
- [ ] Design candidate batching so all 676 kernels with all configurations
      can be evaluated, as in the paper.
- [ ] Add the compiler default as a decision model.
- [ ] Decide how probability ties are broken in Jev rankings (ties in 111 of
      120 kernels in R4).
- [ ] Test position bias by shuffling candidate order.
- [ ] Test prompt and serialization variants that explain what tile fields
      mean; keep R4 as the unchanged baseline.
- [ ] Measure response variance by repeating identical requests.
- [ ] Confirm the meaning of Jev `confidence` versus the maximum probability.
- [ ] Verify the Jev context limit and choose candidate sampling to cover
      more than 120 kernels.
- [ ] Set up the official GNN baseline and integrate it through
      `BaseDecisionModel`.
- [ ] Report results per program, since Inception-v3 is 39% of `valid`.
