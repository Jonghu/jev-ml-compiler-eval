# Run the TenSet context-engineering ablation

This study extends the completed TenSet experiments with controlled context,
ordering, and filtering comparisons. It follows the TpuGraphs ablation's
seven-condition, three-repeat design and includes the full-benchmark follow-up.
Both stages are part of the study. No new benchmark results are claimed by
preparing these commands.

The first step is a free preflight on the actual prepared data on your Mac.
Do that before adding experiment credits: the earlier $30 allowance was a
conservative estimate based partly on an unfiltered full-pool tournament.
This study's filtered full stage sends substantially fewer candidates.

## Study design

The canonical configuration is
[`experiments/tenset/ablation/study.yaml`](../experiments/tenset/ablation/study.yaml).

| Stage | Original evaluation pools | Conditions | Repeats | Candidates shown to Jev |
|---|---|---|---:|---|
| Small | Frozen E5-2673 test cohort: 120 workloads, 50 sampled schedules each | A0, A1, A2, A3, A4 ascending, A4 descending, A5 | 3 | All 50, or 10 for A5 |
| Full | ResNet-50 on Platinum-8272: all 27 task pools and 100,126 source records | A5 | 3 | At most 50 per task |

The E5 cohort uses sampled schedules. TpuGraphs' small cohort instead contains
kernels that naturally have at most 50 configurations. The studies match the
controlled-factor and repeat design, rather than having identical sampling
rules or hardware.

| Variant | Model-visible context | Presentation and selection |
|---|---|---|
| A0 | Original program graph and candidate schedules; no appended field guides | Seeded shuffled order |
| A1 | A0 plus candidate transformation definitions | Same candidates and order as A0 |
| A2 | A0 plus graph-field specification | Same candidates and order as A0 |
| A3 | A0 plus both guides | Same candidates and order as A0 |
| A4 ascending | A3 | Sort by a structural schedule-feature signature, ascending |
| A4 descending | A3 | Sort by the same signature, descending |
| A5 | A3 | Retain structurally diverse schedules; present them in the base shuffled order |

The three requested main comparisons are **A0 versus A1, A0 versus A2,
and A0 versus A3**. A1/A2 versus A3 also tests whether the descriptions are
complementary. A3 versus the two A4 conditions tests presentation sensitivity.

Graph specification explains the existing graph representation; it does not
add a different program, measured runtimes, or performance advice. Candidate
definitions explain schedule syntax independently of that graph guide.

### TenSet-specific ordering and filtering

TenSet candidates contain ordered scheduling transformations, rather than the
top-level output/kernel tile dimensions used by TpuGraphs. Copying TPU's
`keep_largest` rule would therefore be meaningless.

This study uses a fixed **structural diversity** policy. Features describe the
schedule's transformation sequence. A5 uses deterministic farthest-first
selection over normalized features to preserve different schedule structures.
It keeps 10 of the 50 schedules in the small study and at most 50 schedules in
each complete full-stage task pool. Neither feature extraction nor selection
uses runtime labels. No model is trained, and the filter is not claimed to
predict which schedules are fastest.

The A4 signature orders the same structural features in opposite directions.
It is an ordering control, not a learned performance ranking. This differs
from TPU's ascending/descending tile-size experiment and must be described
that way in the report.

The analysis reports Random inside the selected pool and an oracle bound in
addition to full-pool Random. These baselines distinguish Jev's contribution
from the selection policy. The oracle reads labels only during offline scoring.
All performance ratios retain the original evaluation pool as the reference;
filtering does not redefine the optimum to make scores look better.

### Fixed settings and repeat isolation

- Jev model: `jev-1.14.0`, with the TenSet long-context mode retained.
- Same question, program representation, target, and original candidate pool
  across the relevant conditions.
- Opaque candidate aliases and a base presentation shuffle seeded by workload
  ID and study seed 42.
- Three fresh repeats with the same presentation order. Separate cache
  directories across repeats measure response variation. Identical requests
  within one repeat may reuse a response.
- Every case must fit **one Choice question**. Preflight fails if token
  budgets would introduce multiple groups; it does not silently truncate data
  or change the experiment into a tournament.
- Prepared input identities, variant settings, and implementation fingerprints
  are frozen before paid execution.

Do not substitute the old E5 or ResNet predictions for these fresh conditions.
Their aliases, guides, order, seed, or ranking protocol differ from this study.

## 1. Check the prepared data

For the first checkout, run from the Mac repository. If this local branch
already exists, use `git switch feat/tenset-context-ablation` instead of the
`git switch --create` line.

```bash
cd "$HOME/Projects/jev-ml-compiler-eval"
git fetch origin feat/tenset-context-ablation
git switch --create feat/tenset-context-ablation FETCH_HEAD
git pull --ff-only origin feat/tenset-context-ablation
uv sync --locked --extra tenset

ls data/tenset/prepared-w120-c50-s42-metadata.json
ls data/tenset/paper-resnet50-platinum/index.json
```

The small stage requires the exact frozen E5 artifact whose SHA-256 is declared
in the study configuration. The full stage uses the existing prepared
ResNet-50 index and its validated task shards. A downloaded CPU ZIP alone is
not a substitute for either prepared artifact. See the
[TenSet run guide](run-tenset-on-macos.md) and
[ResNet-50 preparation guide](run-tenset-paper-resnet50.md) if preparation is
missing.

The full stage evaluates all task pools belonging to this ResNet-50 benchmark;
it does not evaluate every workload in the entire TenSet CPU archive.

## 2. Freeze the study and estimate the complete cost for free

```bash
uv run python scripts/prepare_tenset_ablation.py \
  --config experiments/tenset/ablation/study.yaml

uv run python scripts/run_tenset_ablation.py \
  --study results/tenset/context-ablation/study.json \
  --preflight-only \
  | tee results/tenset/context-ablation/preflight.txt
```

Preparation and preflight stop before loading `.env` or constructing an API
client. They validate both stages, freeze the source inputs, and estimate
request/token counts using the same presentation, selection, and packing logic
as execution. The small stage has at most 2,520 logical decisions
(7 conditions × 3 repeats × 120 workloads); the full stage adds 81
(1 condition × 3 repeats × 27 tasks).

Read the input-token and input-cost estimates for each condition and stage,
then the total. `by_phase` separates the small and full stages; `by_variant`
separates conditions. The main budget fields are
`estimated_total_input_cost_usd` and `estimated_remaining_input_cost_usd`.
`api_requests_made: 0` confirms this was an offline preflight.

The configuration's default rate is **$0.042 per million input tokens**, an
assumption from the previous experiments. Verify the rate for your account and
change it before freezing a new study if necessary.

Preflight is an estimate, not an invoice: tokenization can differ from the
character-based estimate, and output fees and retries are not included in the
input-only forecast. A small paid pilot provides measured usage for calibration.
Preflight assumes no cache savings for decisions that have not completed.

## 3. Run a small paid pilot

Only this section starts paid execution. Keep your replacement API key in the
repository's ignored `.env` file as `TYPESAFE_API_KEY`; never paste it into the
study configuration or results.

If the account has enough existing credits for the pilot, use them first.
Otherwise add only the provider's minimum needed to fund the pilot. This
command runs the first 10 workloads of all seven small-stage conditions for
repeat 1, at most 70 decisions:

```bash
uv run python scripts/run_tenset_ablation.py \
  --study results/tenset/context-ablation/study.json \
  --phase small \
  --repeats 1 \
  --max-workloads 10 \
  --execute \
  --budget-usd 1
```

The dollar value is a client-side input-cost allowance checked against the
planned remaining work. It is not a provider billing cap. The pilot is part of
repeat 1, rather than a fourth repeat, and its verified completed cases can be
reused when continuing the study.

An optional offline summary can inspect the pilot's individual results:

```bash
uv run python scripts/summarize_tenset_ablation.py \
  --root results/tenset/context-ablation \
  --phase small \
  --allow-incomplete \
  --output-dir results/tenset/context-ablation/pilot-summary
```

This uses immutable completed-case receipts even before a whole run has
published `predictions.jsonl`. Keep pilot summaries in their own directory so
they do not occupy the final `summary/` directory. To intentionally regenerate
an existing summary directory, add `--overwrite`.

## 4. Calibrate before funding and running the rest

```bash
uv run python scripts/run_tenset_ablation.py \
  --study results/tenset/context-ablation/study.json \
  --preflight-only \
  --calibrate-from-pilot \
  | tee results/tenset/context-ablation/preflight-calibrated.txt
```

This remains offline. It reads saved successful response usage, compares actual
tokens with their estimates, and projects the remaining work. A pilot may not
represent the largest full-stage graphs, so retain an allowance when funding
the account. Do not count a cached response as a fresh charge or a fresh repeat.
`pilot_calibration.actual_to_estimated_token_ratio` and
`pilot_calibration.projected_remaining_input_cost_usd` contain the calibration;
they are null if no measured live usage is available.

Use this estimate to decide how much credit to add. The earlier $30 figure is
a conservative planning allowance, not a requirement to deposit $30 before
preflight.

## 5. Complete both stages

The example below permits up to $10 of planned remaining input usage. Replace
that allowance with the amount justified by your preflight. If the estimate
does not fit the allowance, the runner refuses to proceed; increase it only
after reviewing the estimate.

```bash
uv run python scripts/run_tenset_ablation.py \
  --study results/tenset/context-ablation/study.json \
  --phase all \
  --execute \
  --resume \
  --budget-usd 10
```

`--resume` verifies completed cases and continues the remaining cases. It does
not accept changed input data or code as the same study. Use a new study/output
directory for a changed protocol. Keep the Mac awake and connected during
execution; a sleep interval distorts wall-clock latency measurements.

To run the stages separately, use `--phase small` and then `--phase full`, with
the same explicit execution, resume, and reviewed budget options. The study is
complete only when all seven small-stage conditions and the full-stage A5
condition have three completed repeats.

Run the unrestricted offline preflight again after execution. With the default
`--phase all` and no variant, repeat, or workload restriction,
`selected_scope_complete: true` and zero remaining requests confirm coverage of
the complete frozen study. The same flag on a restricted pilot command refers
only to that pilot selection.

## 6. Summarize and prepare the raw-results archive

```bash
uv run python scripts/summarize_tenset_ablation.py \
  --root results/tenset/context-ablation \
  --archive results/tenset/tenset-context-ablation-raw.zip
```

The summary includes per-workload and aggregate results, three-repeat means and
standard deviations, consistency of Top-1 picks, paired condition differences,
filter baselines, plots, and recorded token/latency accounting. Preserve the
frozen study, run descriptions, predictions, and raw request/response cache for
Jonghu. The archive builder includes only experiment artifacts and excludes
credentials. Sending the archive to another person is a separate manual step.

Before interpreting a partial pilot, use the summary tool's explicit
`--allow-incomplete` option. A partial summary does not establish completion of
the study or the full benchmark.

## Metrics and interpretation

For the small study, retain the existing TenSet macro Top-1 and Top-5 Scores.
Its Top-5 Score is the average of prefix-best performance ratios, rather than
the endpoint best-of-five ratio. The summary names endpoint scores separately
and also reports runtime regret and Top-K error.

For the full stage, the weighted network endpoint score uses the original
27 task pools, their occurrence weights, and the pinned evaluator's float32
throughput labels and throughput floor. It is distinct from the small-stage
macro/prefix score and should not be pooled with it. Filtered candidates remain
in the original reference inventory.

Pairwise accuracy is informative only when Jev ranks the whole question's
candidate set. A filtered tail is artificial, so do not report its pairwise
accuracy as Jev's global ranking quality. Report filter retention/oracle
coverage and compare against Random on exactly the retained pool.

For an unfiltered single question, both strict and tie-aware pairwise accuracy
use Jev's actual probabilities. Rounded equal probabilities remain ties rather
than acquiring an artificial preference from their display order. Top-K scores
still evaluate Jev's choice-first ranking, followed by probability and
presentation-order tie breaking.

Three repeats describe response variation on a fixed cohort. They do not
constitute three independent datasets, and repeating the same seed does not
guarantee identical responses. Report all conditions, including negative or
inconclusive results, and identify the previously inspected E5 cohort as an
exploratory evaluation.
