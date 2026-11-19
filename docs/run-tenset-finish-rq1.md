# Finish RQ1: Jev versus Random

The main-paper-aligned ResNet-50 run is **completed and audited**. It contains
27 tasks, 56 occurrences, and 100,126 schedule records. Jev's network endpoint
Top-1/Top-5 are 0.097724/0.274583 versus paired Random's 0.091462/0.377121.
Use the [audited results](results/tenset/paper-resnet50/README.md),
[paper reproduction guide](run-tenset-paper-resnet50.md), and
[protocol](tenset-paper-protocol.md). The run was executed at
`f6a2a49f6b2d0dc8b3ae2e9f87ca6b3250d929ea`; no new paid run is needed to
prepare its final report.

The paired research comparison is **Jev versus Random**. The paper's MLP +
Ranking loss values, 0.8823/0.9456, are a separate external reference rather
than a locally trained paired baseline. Matching its checkpoint or training
another specialized model is outside scope. Keep all three completed studies
and their metrics separate:

| Study | Purpose | Metrics to report |
|---|---|---|
| ResNet-50 on Platinum-8272, 27 tasks/56 occurrences/100,126 records | Primary reconstruction of one main-paper network column | Task-frequency-weighted best-of-K network endpoint scores |
| E5-2673, 120 workloads at 16/32/50 valid schedules | Single-question ranking and completed candidate-size comparison | Top-1, Top-5, Pairwise Accuracy, runtime regret |
| Platinum-8272, 484 workloads and 187,679 test records | Larger tournament selection study | Record-weighted Top-1/Top-5, failure and elimination diagnostics |

The tournament's eliminated tail uses a deterministic elimination order, not a
global learned ranking. Its retained failed-measurement costs are placeholders,
not physical execution latencies. Do not manufacture tournament Pairwise
Accuracy or interpret a mean sentinel-based regret as a measured slowdown.

The [RQ1 report](rq1-report.md), [run log](tenset-runlog.md),
[primary ResNet-50 results](results/tenset/paper-resnet50/README.md), and
[reviewed Platinum results](results/tenset/tutorial-500-platinum-test/README.md)
contain the completed numerical results. The user has now completed the 16/32
conditions and uploaded their analysis together with the reused 50-candidate
anchor. The [size-study results](results/tenset/size-sweep/README.md) include
720 per-workload rows, per-operation tables, and the dedicated size figure.
The remaining substantive export is the concrete tensor/schedule case analysis.
Team-report integration and validation of actual tariff/output charges and
billing also remain. Sections 3–4 retain historical reproduction commands;
the size runs are not pending work.

The revised four-slide setup/workflow/results/diagnosis contribution is
available as an
[editable PowerPoint](presentations/tenset-rq1/tenset_rq1_slides.pptx),
[PDF](presentations/tenset-rq1/tenset_rq1_slides.pdf), and
[speaker notes](presentations/tenset-rq1/speaker_notes.md).

## 1. Update the Mac checkout

```bash
cd "$HOME/Projects/jev-ml-compiler-eval"
git fetch origin
git switch main
git pull --ff-only
uv sync --locked --extra tenset
```

Keep the existing prepared files, completed result bundles, and caches. Git
refuses a branch switch that would overwrite local edits; preserve those edits
before continuing. None of the commands in sections 2 or 3 makes API calls.

Historical completed runs were made at their recorded commits. The offline
analysis reads those saved artifacts without rerunning their models. Do not
resume an old execution under changed code: resume identities intentionally
require the original implementation and environment.

## 2. Complete semantic schedule-case exports offline

The completed E5 size-analysis upload already contains operation-name
breakdowns for 27 literal signatures. Its `jev_cases.csv` identifies post-hoc
metric successes and failures, but does not contain tensor shapes or selected
schedule arguments. The commands below add that missing semantic context and
can also provide Platinum operation/type breakdowns.

First export concrete cases from the completed primary ResNet-50 study:

```bash
uv run python scripts/analyze_tenset_cases.py \
  --results results/tenset/paper-resnet50/full \
  --dataset data/tenset/paper-resnet50-platinum/index.json \
  --output-dir results/tenset/paper-resnet50/semantic-case-analysis
```

Use the complete saved Mac result directory, including `experiment.yaml`,
`metrics.json`, predictions, and manifest; the four-file uploaded numerical
audit alone is not a replacement for that bundle or the prepared collection.
The case export's task-level Top-1/Top-5 are prefix diagnostics for comparing
cases. They do not replace the separately audited `paper-network-summary.json`
endpoint network scores. Inspect the selected Jev, Random, and oracle
configurations to write semantic explanations; retain the full network result.

Also analyze the historical Platinum tutorial study:

```bash
uv run python scripts/analyze_tenset_cases.py \
  --results results/tenset/tutorial-500-platinum-test \
  --dataset data/tenset/tutorial-500-platinum/index.json \
  --output-dir results/tenset/tutorial-500-platinum-case-analysis
```

Also analyze the earlier E5 study, keeping its separate target and metrics:

```bash
uv run python scripts/analyze_tenset_cases.py \
  --results results/tenset/random-vs-jev-w120-c50-s42-long-context \
  --dataset data/tenset/prepared-w120-c50-s42-metadata.json \
  --output-dir results/tenset/e5-w120-c50-case-analysis
```

The case exporter reports Top-1/Top-5 by type. For the E5 experiment's full
declared metric set, including Pairwise Accuracy and regret by operation, run
the generic offline analyzer separately:

```bash
uv run python scripts/analyze_tenset.py \
  --results results/tenset/random-vs-jev-w120-c50-s42-long-context \
  --dataset data/tenset/prepared-w120-c50-s42-metadata.json \
  --output-dir results/tenset/e5-w120-c50-full-metric-analysis
```

The script validates saved candidate pools and fingerprints against the actual
prepared data before joining tensor metadata. It exports per-workload scores,
operation/type and candidate-count tables, and post-hoc success/failure examples
with the Jev, Random, and oracle schedules. Labels, configurations, and source
record identities are read for offline analysis; none is sent to Jev.

Operation groups come from the actual program metadata. Any coarse family
classification is documented in the export; unknown types remain unknown.
Shapes and transformations in the case-study JSON can be inspected to explain
individual outcomes. These examples illustrate the observed results; selecting
large gains or losses is not a new representative evaluation.

Inspect the case export's `analysis-manifest.json`, `by_operation.csv`,
`by_operation_family.csv`, and `case_studies.md`. Preserve the original
`case_studies.json` for complete transformations. If an analysis output directory
already exists, choose a new output name rather than overwriting it.

The prepared shards are only on the Mac. The uploaded operation-name tables
can be reviewed in the cloud, but complete tensor/schedule explanations still
require these local semantic exports. They are ready-to-run analysis, not
finished case narratives.

## 3. Reproduction: freeze the candidate-size sweep offline

**Status: completed and analyzed by the user.** The new 16/32 executions and
the historical 50-schedule anchor are represented in the uploaded six-row
aggregate and 720-row per-workload table. The raw new prediction/manifest
bundles were not supplied, so the cloud review checks analysis consistency
rather than repeating the entire source-level audit.

For a reproduction, use the same 120 E5 workloads at 16, 32, and 50 schedules. Smaller pools are
prefixes of the same seeded order; they are not selected by measured speed.
The target, seed 42, metadata, no-description input, long-context mode, model
version, and metric definitions stay fixed. This is separate from the Platinum
tournament and its record-level split.

```bash
uv run python scripts/prepare_tenset_size_sweep.py \
  --prepared data/tenset/prepared-w120-c50-s42-metadata.json \
  --baseline-results results/tenset/random-vs-jev-w120-c50-s42-long-context
```

The helper checks the audited prepared-file SHA-256 and the entire cohort. It
validates the 50-schedule baseline's candidate IDs, inputs, labels, rankings,
model settings, and metrics before reusing it. It never alters that bundle.
If the validation fails, investigate the difference instead of bypassing it.
If no reusable 50-schedule bundle is available, omit `--baseline-results` to
prepare a separate new 50-schedule run.

Local profiles and the frozen manifest are written to
`experiments/tenset/local/size-sweep/`. Canonical templates are committed under
`experiments/tenset/size-sweep/`; local machine paths are ignored by Git.
Preparation prints the applicable preflight, execution, and analysis commands.
Keep the data and generated profiles unchanged after freezing them.

For a reproduction, validate the 16- and 32-schedule members without
credentials or model calls:

```bash
uv run python scripts/run_tenset_size_sweep.py \
  --manifest experiments/tenset/local/size-sweep/manifest.json \
  --candidate-count 16 --preflight-only

uv run python scripts/run_tenset_size_sweep.py \
  --manifest experiments/tenset/local/size-sweep/manifest.json \
  --candidate-count 32 --preflight-only
```

The wrapper rechecks the frozen data and configuration before execution. A
different file hash is not automatically interchangeable with the audited file.
Custom data can use an explicit expected checksum when preparing a new study;
that does not establish identity with the historical benchmark.

## 4. Reproduction: paid size-experiment commands

**The user has already completed these 16/32 runs.** These commands are
preserved for reproducibility, not as instructions to repeat paid calls to
finish the current analysis. A fresh execution against a new destination makes
**paid Jev API calls**. The earlier approximately 7.62 USD cost belongs to the
completed Platinum study; it is not the cost of this size sweep. New run dates
and billing details cannot be established from the analysis tables alone.

The key is `TYPESAFE_API_KEY` in the repository-root `.env`. An exported shell
value takes precedence. To use the value in `.env` without displaying it:

```bash
unset TYPESAFE_API_KEY
```

```bash
uv run python scripts/run_tenset_size_sweep.py \
  --manifest experiments/tenset/local/size-sweep/manifest.json \
  --candidate-count 16

uv run python scripts/run_tenset_size_sweep.py \
  --manifest experiments/tenset/local/size-sweep/manifest.json \
  --candidate-count 32
```

The existing 50-schedule anchor requires no new API calls when reuse was
verified. If preparation instead scheduled a new 50-schedule member, execute
the printed command for that member. Nonempty result directories are protected;
keep successful cached responses and the exact frozen settings after an
interruption. The single-question runner does not checkpoint per-workload
completion like the tournament collection runner, so do not assume `--resume`
is available for this sweep. Consult the helper's printed status and preserve
any partial output before choosing a new destination.

The uploaded completed analysis was generated after the new members finished.
For a reproduction, run the analysis command printed during preparation after
all requested members complete. Use `by_candidate_count.csv` and paired rows to discuss
size effects. An overall average pooled across sizes is not the size comparison.
The plot and table should show the number of workloads and the 16/32/50 budgets.
Compare Jev and Random within each budget. Increasing the pool can change its
fastest schedule, so a cross-size score change alone is not a physical speedup.

The dedicated candidate-size figure has also been uploaded. To regenerate it
offline from a completed local analysis:

```bash
uv run python scripts/plot_tenset_size_sweep.py \
  --analysis-dir results/tenset/size-sweep/analysis
```

The generic analyzer's overall comparison plot pools sizes. Use the dedicated
size figure for this ablation; it checks matched workload IDs and displays each
budget separately.

The current Platinum natural-size breakdown (468 workloads with 400 candidates,
16 with 6–88) is already published as descriptive analysis. It does not isolate
candidate-count effects, because workload identity changes between those groups.

## 5. Complete the report checklist

- [x] State paired Jev-versus-Random scope, the external MLP reference, and three distinct completed protocols.
- [x] Complete and audit the primary ResNet-50 network run and publish its endpoint scores.
- [x] Publish Top-1/Top-5, the E5 Pairwise Accuracy/regret, and reproducible metrics.
- [x] Include paired uncertainty and a separately labelled exact Random expectation.
- [x] Record failure-sentinel choices, tournament attrition, tokens, and cached usage.
- [x] Prepare plots, updated slides, and the context-aware export workflow.
- [x] Complete the user-run 16/32 conditions and analyze them with the reused 50 anchor.
- [x] Review per-budget metrics, operation-name breakdowns, and the dedicated size plot.
- [ ] Export full tensor/schedule context and write the semantic case narratives.
- [ ] Integrate the current results into the team report and finish metric-label agreement/proofreading.
- [ ] Verify actual provider pricing, output charges, and billing before claiming invoice totals.

Fresh API repeatability checks and controlled field-description, context,
presentation-order, tie-breaking, and grouping ablations are optional extensions.
The primary ResNet-50 run already uses all declared task records; additional
networks or expanded historical cohorts would be new studies. Finish semantic
case exports, the final narrative, and pricing validation before expanding scope.

For final review, the compact Mac exports to share are `by_operation.csv`,
`by_operation_family.csv`, `per_workload.csv`, and `case_studies.json` from each
case-analysis directory. The completed size analysis and dedicated plot have
already been shared. The frozen prepared data and API key do not need to be
shared. A full independent audit of the new size runs would additionally need
their original prediction and manifest bundles; their absence is explicitly
recorded in the current analysis limits.
