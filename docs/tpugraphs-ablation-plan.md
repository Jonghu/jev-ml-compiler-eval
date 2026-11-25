# TpuGraphs context-engineering ablation plan (A0–A5)

This plan defines the context-engineering ablation for RQ2 on TpuGraphs
`tile:xla`. It changes one part of the Jev input at a time and measures the
effect on tile configuration choices. Results are recorded in
[the ablation run log](tpugraphs-ablation-runlog.md); earlier runs R1–R7 and
analyses A1–A4 are in [the main run log](runlog.md).

## Question

Which parts of the model-visible context help Jev pick fast tile
configurations, and how much of any gain comes from Jev rather than from the
context itself?

## Scope

- Kernels: the 120 `valid` kernels with at most 50 distinct configurations
  (5 programs; 92 from Inception-v3). Every kernel fits one Choice question,
  so no tournament rounds confound the comparison.
- This subset is easier than the full split (R7). Results are compared across
  variants, not with the paper's full-split values.
- Model: `jev-1.14.0` through the `jev-tournament` model, which issues one
  question per kernel at this size.

## Conditions held fixed in every variant

| Setting | Value | Reason |
|---|---|---|
| Candidate IDs | Opaque (`c01`, `c02`, ...) | R6 could not separate field effects from recognizing `config_0` |
| Base order | Shuffled, seed `42:<workload_id>` | R5 showed that the original order correlates with tile size |
| Question | "Which candidate is most likely to have the lowest runtime for this workload?" | Unchanged since R2 |
| Graph serialization | Unchanged from R4 ([tpugraphs.md](tpugraphs.md)) | |
| Repeats | 3 per variant, one Jev cache directory per repeat | R3 showed nondeterministic responses |

A0 therefore differs from R4: R4 used original IDs and order. A1 differs from
R6 in the same way.

## Variants

| ID | Context | Presented candidates | Config |
|---|---|---|---|
| A0 naive | graph + candidates | all, shuffled | `ablation/a0-naive.yaml` |
| A1 | A0 + candidate feature definitions | all, shuffled | `ablation/a1-config-fields.yaml` |
| A2 | A0 + graph field specification | all, shuffled | `ablation/a2-graph-fields.yaml` |
| A3 | A0 + both | all, shuffled | `ablation/a3-a5-both-fields.yaml` |
| A4-asc | A3 | all, sorted by tile size, smallest first | same as A3 |
| A4-desc | A3 | all, sorted by tile size, largest first | same as A3 |
| A5 | A3 | only the 10 largest tiles, shuffled | same as A3 |

Configs live in `experiments/tpugraphs/`. Tile size is the product of the
`output_bounds` and `kernel_bounds` entries, as in A2 and A4 of the main run
log.

### What each part adds

- **Candidate feature definitions** (`describe_config_fields`, unchanged since
  R6): official README definitions of `output_bounds`, `kernel_bounds`, and
  `input_bounds`, and the paper's sentence on tile size selection. No units
  and no statement that larger tiles are faster.
- **Graph field specification** (`describe_graph_fields`, new): the meaning of
  each serialized node field (`opcode`, `dtype`, `shape`,
  `layout_minor_to_major`, `operands`, `is_root`, `parameter_number`,
  `convolution`, and remaining HLO attributes). It describes the graph only,
  not the tiles, and states no performance preference. The text is
  `GRAPH_FIELD_DESCRIPTIONS` in `src/jev_compiler_eval/datasets/tpugraphs.py`.
- **Candidate ordering** (`candidate_order` of `jev-tournament`, new): sorts
  the shuffled order by tile size; equal sizes keep the shuffled order. Opaque
  IDs follow the sorted order.
- **Candidate filtering** (`keep_largest` of `jev-tournament`, new): shows Jev
  only the 10 largest tiles. Hidden candidates rank after every shown
  candidate, largest first, so metrics still cover every candidate. Kernels
  with at most 10 candidates (36 of 120) are unchanged, so their A5 request
  equals the A3 request and reuses the A3 response of the same repeat.

## Comparisons

| Comparison | Tests |
|---|---|
| A0 vs A1 | Do tile field meanings help? (R4 → R6 suggested yes, without blinding.) |
| A0 vs A2 | Does explaining the graph help without explaining the tiles? |
| A0 vs A3, and A1/A2 vs A3 | Are the two descriptions complementary (2×2 factorial)? |
| A3 vs A4-asc vs A4-desc | Position sensitivity. If Jev reads the content, all three agree; a gap between ascending and descending is a position effect. |
| A3 vs A5, against the heuristic baselines | Does a smaller, pre-filtered set help Jev, beyond what the filter alone gives? |

## Why a large-tile filter, and why K = 10

The full rationale and numbers are in A4 of the main run log. In brief:

- **Data**: on the `train` split, which no evaluated model sees, larger tiles
  are faster in 98.3% of kernels (median within-kernel Spearman −0.80), and
  the optimum lies in the larger half of the candidates in 90.0% of kernels.
- **Domain**: the paper describes tile size selection as maximizing compute
  efficiency while data fit in local memory, which predicts the same pattern.
- **Filter rather than rule**: the single largest tile is optimal in only
  7.0% of `train` kernels, so the filter keeps the large-tile region and leaves
  the final choice to the model.
- **K = 10 for this subset**: K = 50 would filter nothing here. On the 432
  `train` kernels with at most 50 candidates, keeping 10 retains the optimum
  in 78.2% of kernels, with an oracle Top-1 error of 0.9%. On the 120 `valid`
  kernels it filters 84 kernels.

## Required baselines

Reported next to every variant, on the same 120 kernels:

| Baseline | Source |
|---|---|
| Random (seed 42) | `random` model in the A0 run |
| Compiler default (`config_0`) | offline, from labels |
| Largest tile | offline heuristic |
| Largest-10 filter + random pick (expected) | offline; the reference for A5 |
| Largest-10 filter + oracle pick | offline upper bound for A5 |

The filter is a hand-crafted heuristic. Jev's contribution in A5 is its
improvement over the "filter + random pick" baseline, not over A3.

## Metrics and reporting

- Per run: Top-1 error (mean and median), Top-1 accuracy, Top-5 and Top-10
  error, Kendall τ, as in R1–R7.
- Per variant: mean ± standard deviation over the three repeats, and the share
  of kernels where all repeats picked the same candidate.
- Diagnostics: smallest-tile and largest-tile picks (the R4 failure mode),
  `config_0` picks, tokens, API latency, and decision latency.
- A difference between variants is called meaningful only if it exceeds the
  repeat-to-repeat spread.

## Output layout and team sharing

Everything for this ablation is below `results/tpugraphs/ablation-small/`
(ignored by Git; shared as an archive):

```text
results/tpugraphs/ablation-small/
├── README.md                     # layout and how to read the files
├── runs/<experiment>/            # runner bundles: experiment.yaml, manifest.json,
│                                 # predictions.jsonl, metrics.json, summary.csv
├── jev-cache/rep<n>/             # raw Jev requests and responses, one JSON per request
└── summary/                      # variants.csv, runs.csv, per_kernel.csv,
                                  # baselines.csv, summary.md
```

Each prediction's `metadata.request_reports[].cache_path` points to the raw
request and response it used.

## Commands

```bash
uv run python scripts/run_experiment.py --config experiments/tpugraphs/ablation/a0-naive.yaml
uv run python scripts/run_experiment.py --config experiments/tpugraphs/ablation/a1-config-fields.yaml
uv run python scripts/run_experiment.py --config experiments/tpugraphs/ablation/a2-graph-fields.yaml
uv run python scripts/run_experiment.py --config experiments/tpugraphs/ablation/a3-a5-both-fields.yaml
uv run python scripts/summarize_tpugraphs_ablation.py
```

## Budget

21 Jev runs (7 variants × 3 repeats) of 120 requests each: 2,520 requests.
The offline planner estimates 9.3M input tokens, an overestimate because it
assumes 1.4 characters per token (R4 measured about 1.53). At the R7 rate
(≈ $0.042 per million input tokens) the run costs at most about $0.40.

## Limitations

- The subset favors small kernels and Inception-v3; conclusions may not carry
  over to kernels with hundreds of candidates (R7).
- Three repeats give a rough spread, not a confidence interval.
- The graph specification text was written for this project from the README
  feature names and HLO semantics; it is not an official definition.
- A5 depends on a heuristic derived from the dataset's training labels.
